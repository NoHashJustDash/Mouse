// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/uci/session.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "engine/chess/attacks.h"
#include "engine/chess/movegen.h"
#include "engine/uci/emergency.h"
namespace {
struct Provider final : mouse::nn::Backend {
    mouse::nn::BackendInfo identity;
    mouse::nn::ModelMetadata model;
    std::atomic<unsigned>* calls;
    std::atomic<unsigned>* destroyed;
    unsigned fail_at;
    std::function<void()>* hook;
    std::atomic<std::size_t>* largest_batch;
    Provider(std::atomic<unsigned>& count, unsigned failure, std::function<void()>& before,
             std::atomic<std::size_t>& batch, std::atomic<unsigned>& retired)
        : calls(&count),
          destroyed(&retired),
          fail_at(failure),
          hook(&before),
          largest_batch(&batch) {
        model.input_format = 1;
        identity.onnx_sha256 = "session-test";
        identity.cache_namespace = "session-test";
        identity.provider = "CPUExecutionProvider";
        identity.precision = "fp32";
    }
    ~Provider() override { ++*destroyed; }
    Provider(const Provider&) = delete;
    Provider& operator=(const Provider&) = delete;
    Provider(Provider&&) = delete;
    Provider& operator=(Provider&&) = delete;
    mouse::Result<void> evaluate(std::span<const mouse::nn::InputPlanes> inputs,
                                 std::span<mouse::nn::NNOutput> outputs) override {
        auto old = largest_batch->load();
        while (old < inputs.size() && !largest_batch->compare_exchange_weak(old, inputs.size())) {
        }
        const auto number = ++*calls;
        if (*hook) (*hook)();
        if (number == fail_at)
            return tl::unexpected(std::string{"session injected inference failure"});
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            outputs[i] = {};
            outputs[i].wdl = {0.5F, 0.25F, 0.25F};
        }
        return {};
    }
    [[nodiscard]] const mouse::nn::BackendInfo& info() const noexcept override { return identity; }
    [[nodiscard]] const mouse::nn::ModelMetadata& metadata() const noexcept override {
        return model;
    }
};
struct Harness {
    std::mutex mutex;
    std::condition_variable changed;
    std::vector<std::string> lines;
    std::function<void(std::string_view)> hook;
    std::function<void()> evaluate_hook;
    std::string construction_error;
    bool null_backend = false;
    std::atomic<unsigned> calls{0}, created{0}, destroyed{0};
    std::function<void(const mouse::nn::BackendConfig&)> load_hook;
    std::atomic<std::size_t> largest_batch{0};
    std::unique_ptr<mouse::uci::Session> session;
    explicit Harness(unsigned fail_at = 0, bool weights = true,
                     std::function<void(const mouse::nn::BackendConfig&)> loading = {})
        : load_hook(std::move(loading)) {
        mouse::uci::Options options;
        options.backend.kind = mouse::nn::BackendKind::Cpu;
        options.backend.precision = mouse::nn::ComputePrecision::Float32;
        options.auto_minibatch = false;
        options.backend.batch_buckets = {1, 4};
        options.search.maximum_batch = 4;
        options.search.ram_limit_bytes = 8U << 20U;
        options.search.deterministic = true;
        options.nncache_entries = 0;
        if (weights) options.backend.model_path = "session-fixture.onnx";
        session = std::make_unique<mouse::uci::Session>(
            options,
            [this](std::string_view line) {
                {
                    const std::lock_guard lock(mutex);
                    lines.emplace_back(line);
                    changed.notify_all();
                }
                if (hook) hook(line);
            },
            [this, fail_at](const mouse::nn::BackendConfig& config)
                -> mouse::Result<std::unique_ptr<mouse::nn::Backend>> {
                ++created;
                if (load_hook) load_hook(config);
                if (!construction_error.empty()) return tl::unexpected(construction_error);
                if (null_backend) return std::unique_ptr<mouse::nn::Backend>{};
                return std::make_unique<Provider>(calls, fail_at, evaluate_hook, largest_batch,
                                                  destroyed);
            });
    }
    Harness(const Harness&) = delete;
    Harness& operator=(const Harness&) = delete;
    Harness(Harness&&) = delete;
    Harness& operator=(Harness&&) = delete;
    ~Harness() { session->shutdown(); }
    bool send(std::string_view text) { return session->handle(text); }
    std::size_t count(std::string_view prefix) {
        const std::lock_guard lock(mutex);
        return static_cast<std::size_t>(
            std::ranges::count_if(lines, [&](const auto& s) { return s.starts_with(prefix); }));
    }
    bool wait(std::string_view prefix) {
        std::unique_lock lock(mutex);
        return changed.wait_for(lock, std::chrono::seconds{5}, [&] {
            return std::ranges::any_of(lines, [&](const auto& s) { return s.starts_with(prefix); });
        });
    }
};
}  // namespace
TEST_CASE("UCI handshakes and missing model remain explicit and recoverable", "[uci][session]") {
    Harness h{0, false};
    CHECK(h.send("uci"));
    CHECK(h.count("id name Mouse ") == 1);
    CHECK(h.count("option name WeightsFile ") == 1);
    CHECK(h.count("uciok") == 1);
    CHECK(h.send("isready"));
    CHECK(h.count("info string error ") == 1);
    CHECK(h.count("readyok") == 1);
    CHECK(h.created == 0);
    CHECK(h.send("setoption name WeightsFile value session-fixture.onnx"));
    CHECK(h.send("isready"));
    CHECK(h.created == 1);
    CHECK(h.count("readyok") == 2);
    CHECK(h.send("nonsense"));
    CHECK(h.count("info string error ") == 2);
    CHECK_FALSE(h.send("quit"));
    CHECK(h.session->exit_code() == 0);
}
TEST_CASE("Automatic MinibatchSize is resolved for the backend before it loads",
          "[uci][session][minibatch-auto]") {
    std::vector<std::size_t> loaded;
    Harness h{0, true,
              [&loaded](const mouse::nn::BackendConfig& config) { loaded = config.batch_buckets; }};
    REQUIRE(h.send("setoption name MinibatchSize value 0"));
    REQUIRE(h.send("isready"));
    REQUIRE(h.wait("readyok"));
    // The CPU backend recommends 64 (nn::recommended_minibatch).
    CHECK(loaded == std::vector<std::size_t>{1, 8, 32, 64});
    CHECK(h.count("info string MinibatchSize auto: 64") == 1);
    REQUIRE(h.send("go nodes 64"));
    REQUIRE(h.wait("bestmove "));
    CHECK(h.largest_batch.load() <= 64);
    CHECK(h.count("info string error ") == 0);
}

TEST_CASE("Finite UCI search reports one legal filtered bestmove and validated history",
          "[uci][session]") {
    Harness h;
    REQUIRE(h.send("position startpos moves e2e4 e7e5"));
    REQUIRE(h.send("go nodes 65 searchmoves g1f3"));
    REQUIRE(h.wait("bestmove "));
    CHECK(h.count("bestmove g1f3") == 1);
    CHECK(h.count("info depth ") >= 1);
    CHECK(h.send("position startpos moves e2e4 e7e5 g1f3 b8c6"));
    CHECK(h.send("position startpos moves e2e5"));
    CHECK(h.count("info string error ") == 1);
    CHECK(h.send("go nodes 33 searchmoves f1b5"));
    // Joining through stop is deterministic and also covers a stop arriving before run accepts.
    CHECK(h.send("stop"));
    CHECK(h.count("bestmove f1b5") == 1);
    CHECK(h.created == 1);
    CHECK(h.session->exit_code() == 0);
}
TEST_CASE("UCI infinite and ponder hold results and readiness stays responsive", "[uci][session]") {
    Harness h;
    REQUIRE(h.send("go infinite nodes 17 searchmoves e2e4"));
    REQUIRE(h.wait("info depth "));
    CHECK(h.count("bestmove ") == 0);
    CHECK(h.send("isready"));
    CHECK(h.count("readyok") == 1);
    CHECK(h.send("position startpos"));
    CHECK(h.send("setoption name Threads value 2"));
    CHECK(h.count("info string error ") == 2);
    CHECK(h.send("stop"));
    CHECK(h.count("bestmove e2e4") == 1);
    CHECK(h.send("setoption name Ponder value true"));
    CHECK(h.send("go ponder nodes 17 searchmoves d2d4 wtime 10000 btime 10000"));
    CHECK(h.send("ponderhit"));
    CHECK(h.send("stop"));
    CHECK(h.count("bestmove d2d4") == 1);
    CHECK(h.count("bestmove ") == 2);
}
TEST_CASE("Session inference failure cancels idle input and never returns a substitute move",
          "[uci][session]") {
    Harness h{2};
    REQUIRE(h.send("go nodes 65"));
    REQUIRE(h.wait("info string error "));
    h.session->shutdown();
    CHECK(h.session->exit_code() == 1);
    CHECK(h.session->cancellation_token().stop_requested());
    CHECK(h.count("bestmove ") == 0);
}
TEST_CASE("Session rejects go errors before search and accepts terminal roots", "[uci][session]") {
    Harness h;
    CHECK(h.send("go nodes 3 searchmoves e2e5"));
    CHECK(h.count("info string error ") == 1);
    CHECK(h.calls == 0);
    CHECK(h.send("go wtime 1000 btime 1000 ponder"));
    CHECK(h.count("info string error ") == 2);
    CHECK(h.send("position fen 7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"));
    CHECK(h.send("go nodes 3"));
    REQUIRE(h.wait("bestmove "));
    CHECK(h.count("bestmove 0000") == 1);
    CHECK(h.calls == 0);
}

TEST_CASE("A GUI may send the next position as soon as bestmove is published",
          "[uci][session][publication]") {
    Harness h;
    std::mutex gate;
    std::condition_variable changed;
    bool entered = false, release = false, continued = false;
    h.hook = [&](std::string_view line) {
        if (!line.starts_with("bestmove ")) return;
        std::unique_lock lock(gate);
        entered = true;
        changed.notify_all();
        changed.wait(lock, [&] { return release; });
    };
    REQUIRE(h.send("go nodes 1"));
    {
        std::unique_lock lock(gate);
        REQUIRE(changed.wait_for(lock, std::chrono::seconds{3}, [&] { return entered; }));
    }
    std::jthread input([&] {
        {
            const std::lock_guard lock(gate);
            continued = true;
            changed.notify_all();
        }
        static_cast<void>(h.send("position startpos moves e2e4"));
    });
    {
        std::unique_lock lock(gate);
        REQUIRE(changed.wait_for(lock, std::chrono::seconds{3}, [&] { return continued; }));
        // Keep publication blocked while the next input call reaches its
        // lifecycle guard. Both publication/drain paths remain bounded.
        changed.wait_for(lock, std::chrono::milliseconds{100}, [] { return false; });
        release = true;
        changed.notify_all();
    }
    input.join();
    CHECK(h.count("info string error ") == 0);
    CHECK(h.send("go nodes 1 searchmoves e7e5"));
    CHECK(h.send("stop"));
    CHECK(h.count("bestmove e7e5") == 1);
}

TEST_CASE("Output failure suppresses every later bestmove publication",
          "[uci][session][output-failure]") {
    Harness h;
    bool failed = false;
    h.hook = [&](std::string_view line) {
        if (!failed && line.starts_with("info depth ")) {
            failed = true;
            throw std::runtime_error("one-shot output failure");
        }
    };
    REQUIRE(h.send("go nodes 1"));
    REQUIRE(h.wait("info depth "));
    {
        std::unique_lock lock(h.mutex);
        h.changed.wait_for(lock, std::chrono::milliseconds{500}, [&] {
            return std::ranges::any_of(
                h.lines, [](const auto& line) { return line.starts_with("bestmove "); });
        });
    }
    CHECK(h.session->exit_code() == 1);
    CHECK(h.session->cancellation_token().stop_requested());
    CHECK(h.count("bestmove ") == 0);
}
TEST_CASE("Ponderhit never restarts inference after a fatal error while draining",
          "[uci][session][ponder-failure]") {
    Harness h{1};
    std::mutex gate;
    std::condition_variable changed;
    bool entered = false, released = false;
    h.evaluate_hook = [&] {
        std::unique_lock lock(gate);
        entered = true;
        changed.notify_all();
        changed.wait(lock, [&] { return released; });
    };
    CHECK(h.send("setoption name Ponder value true"));
    REQUIRE(h.send("go ponder nodes 17"));
    {
        std::unique_lock lock(gate);
        REQUIRE(changed.wait_for(lock, std::chrono::seconds{3}, [&] { return entered; }));
    }
    std::jthread release([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds{100});
        const std::lock_guard lock(gate);
        released = true;
        changed.notify_all();
    });
    // A known fatal error must reject a new hit. In-flight errors cannot be
    // predicted synchronously; the separate deadline regression covers them.
    REQUIRE(h.wait("info string error "));
    CHECK_FALSE(h.send("ponderhit"));
    h.session->shutdown();
    CHECK(h.calls == 1);
    CHECK(h.count("bestmove ") == 0);
    CHECK(h.session->exit_code() == 1);
}

TEST_CASE("Backend construction failure is fatal at readiness and direct search",
          "[uci][session][construction-failure]") {
    Harness h{0, false};
    h.construction_error = "injected TensorRT construction failure";
    REQUIRE(h.send("setoption name WeightsFile value session-fixture.onnx"));
    SECTION("readiness never claims a failed backend is ready") { CHECK_FALSE(h.send("isready")); }
    SECTION("direct search fails instead of leaving a live silent process") {
        static_cast<void>(h.send("go nodes 30"));
        REQUIRE(h.wait("info string error "));
        h.session->shutdown();
    }
    CHECK(h.created == 1);
    CHECK(h.calls == 0);
    CHECK(h.count("info string error injected TensorRT construction failure") == 1);
    CHECK(h.count("readyok") == 0);
    CHECK(h.count("bestmove ") == 0);
    CHECK(h.session->exit_code() == 1);
    CHECK(h.session->cancellation_token().stop_requested());
    CHECK_FALSE(h.send("isready"));
    CHECK(h.created == 1);
}

TEST_CASE("Negative remaining clock still produces a legal bestmove",
          "[uci][session][negative-clock]") {
    Harness h;
    REQUIRE(h.send("position startpos"));
    REQUIRE(h.send("go wtime -20 btime 1000 winc 100 binc 100 searchmoves e2e4"));
    REQUIRE(h.wait("bestmove "));
    CHECK(h.count("bestmove e2e4") == 1);
    CHECK(h.count("info string error ") == 0);
    CHECK(h.session->exit_code() == 0);
}

TEST_CASE("Ponder limits outstanding batch size before an emergency clock hit",
          "[uci][session][ponder-batch]") {
    Harness h;
    std::mutex gate;
    std::condition_variable changed;
    bool entered = false, released = false;
    h.evaluate_hook = [&] {
        if (h.calls.load() != 2) return;
        std::unique_lock lock(gate);
        entered = true;
        changed.notify_all();
        changed.wait(lock, [&] { return released; });
    };
    REQUIRE(h.send("setoption name Ponder value true"));
    REQUIRE(h.send("go ponder wtime 50 btime 50"));
    {
        std::unique_lock lock(gate);
        REQUIRE(changed.wait_for(lock, std::chrono::seconds{3}, [&] { return entered; }));
        CHECK(h.largest_batch == 1);
        released = true;
        changed.notify_all();
    }
    REQUIRE(h.send("ponderhit"));
    REQUIRE(h.send("stop"));
    CHECK(h.count("bestmove ") == 1);
    CHECK(h.count("info string error ") == 0);
}

TEST_CASE("Claimable and material draw roots answer UCI with an independently validated legal move",
          "[uci][session][draw-root]") {
    constexpr std::array<std::string_view, 3> positions{
        "position startpos moves g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8",
        "position fen 4k3/8/8/8/8/8/8/R3K3 w - - 100 1",
        "position fen 4k3/8/8/8/8/8/8/4K3 w - - 0 1"};
    for (const auto text : positions) {
        Harness h;
        REQUIRE(h.send(text));
        REQUIRE(h.send("go nodes 30"));
        REQUIRE(h.wait("bestmove "));
        CHECK(h.count("bestmove ") == 1);
        CHECK(h.count("bestmove 0000") == 0);
        CHECK(h.count("info string error ") == 0);
        CHECK(h.calls >= 1);
        const auto command = mouse::uci::parse_command(text);
        REQUIRE(command);
        const auto& parsed = std::get<mouse::uci::PositionCommand>(*command);
        const mouse::chess::AttackTables attacks;
        auto board = mouse::chess::Position::from_fen(parsed.fen, attacks);
        REQUIRE(board);
        for (const auto& move_text : parsed.moves) {
            const auto move = mouse::chess::parse_uci_move(*board, move_text, attacks, false);
            REQUIRE(move);
            static_cast<void>(board->make_move(*move, attacks));
        }
        std::string best;
        {
            const std::lock_guard lock(h.mutex);
            for (const auto& line : h.lines)
                if (line.starts_with("bestmove ")) best = line.substr(9);
        }
        CHECK(mouse::chess::parse_uci_move(*board, best, attacks, false));
        CHECK(h.session->exit_code() == 0);
    }
}

TEST_CASE("A highly visited repeated child refreshes once before an expired-clock reply",
          "[uci][session][retained-draw-clock]") {
    Harness h;
    REQUIRE(h.send("position startpos moves g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1"));
    REQUIRE(h.send("go nodes 500 searchmoves f6g8"));
    REQUIRE(h.wait("bestmove f6g8"));
    REQUIRE(h.send("position startpos moves g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8"));
    const auto before = h.calls.load();
    // An expired clock cannot require an evaluation before publication. Prime
    // this retained draw root explicitly, then verify the exact one-call total
    // across preparation and the expired-clock reply. No post-reply work is owed.
    REQUIRE(h.send("go nodes 1 searchmoves d2d4"));
    REQUIRE(h.wait("bestmove d2d4"));
    REQUIRE(h.send("isready"));
    REQUIRE(h.send("go wtime 0 btime 1000 searchmoves e2e4"));
    REQUIRE(h.wait("bestmove e2e4"));
    CHECK(h.calls == before + 1);
    CHECK(h.count("bestmove e2e4") == 1);
    CHECK(h.count("bestmove 0000") == 0);
    CHECK(h.count("info string error ") == 0);
}

TEST_CASE("A populated unrelated position jump fits the 50 ms clock minus move overhead",
          "[uci][session][root-reset-clock]") {
    Harness h;
    REQUIRE(h.send("setoption name RamLimitMb value 512"));
    REQUIRE(h.send("isready"));
    REQUIRE(h.send("go nodes 257 searchmoves e2e4"));
    REQUIRE(h.wait("bestmove e2e4"));
    const std::array<std::pair<std::string_view, std::string_view>, 2> cases{
        {{"position fen 4k3/8/8/8/8/8/P7/R3K3 b - - 0 1", "e8e7"}, {"position startpos", "d2d4"}}};
    for (const auto& [root, expected] : cases) {
        const auto started = std::chrono::steady_clock::now();
        REQUIRE(h.send(root));
        REQUIRE(h.send("go wtime 50 btime 50 searchmoves " + std::string{expected}));
        REQUIRE(h.wait("bestmove " + std::string{expected}));
        const auto elapsed = std::chrono::steady_clock::now() - started;
        const auto elapsed_ms = std::chrono::duration<double, std::milli>(elapsed).count();
        INFO("position plus go to bestmove took " << elapsed_ms << " ms");
        CHECK(elapsed < std::chrono::milliseconds{20});
        CHECK(h.count("bestmove " + std::string{expected}) == 1);
        CHECK(h.count("info string error ") == 0);
    }
    CHECK(h.created == 1);
    CHECK(h.session->exit_code() == 0);
}

TEST_CASE("Timed search recovers after fresh physical root evaluations following one stall",
          "[uci][session][latency-recovery]") {
    Harness h;
    h.evaluate_hook = [&] {
        if (h.calls.load() == 1) std::this_thread::sleep_for(std::chrono::milliseconds{100});
    };
    const auto finish_search = [&](std::string_view command) {
        {
            const std::lock_guard lock(h.mutex);
            h.lines.clear();
        }
        REQUIRE(h.send(command));
        REQUIRE(h.wait("bestmove "));
        REQUIRE(h.send("isready"));  // Joins the completed search before the next observation.
        REQUIRE(h.count("info string error ") == 0);
    };
    finish_search("go nodes 1");
    REQUIRE(h.calls == 1);
    for (unsigned sample = 0; sample < 8; ++sample) {
        REQUIRE(h.send(sample % 2 == 0 ? "position startpos moves e2e4"
                                       : "position startpos moves d2d4"));
        const auto before = h.calls.load();
        finish_search("go nodes 1");
        REQUIRE(h.calls == before + 1);
    }
    REQUIRE(h.send("ucinewgame"));
    REQUIRE(h.send("position startpos moves c2c4"));
    const auto before = h.calls.load();
    finish_search("go nodes 16 movetime 50");
    {
        const std::lock_guard lock(h.mutex);
        CHECK(std::ranges::any_of(h.lines, [](const auto& line) {
            return line.starts_with("info depth ") && line.find(" nodes 16 ") != std::string::npos;
        }));
    }
    CHECK(h.calls > before + 1);
    CHECK(h.session->exit_code() == 0);
}

TEST_CASE("Cached reused and terminal primes cannot erase a slow physical latency reserve",
          "[uci][session][latency-recovery]") {
    Harness h;
    bool cached_newgames = false, terminal_roots = false;
    SECTION("reused root with the cache disabled") {}
    SECTION("cache hits across new games") { cached_newgames = true; }
    SECTION("checkmated roots perform no physical inference") { terminal_roots = true; }
    REQUIRE(h.send("setoption name NNCacheSize value 1"));
    h.evaluate_hook = [] { std::this_thread::sleep_for(std::chrono::milliseconds{100}); };
    const auto finish_search = [&](std::string_view command) {
        {
            const std::lock_guard lock(h.mutex);
            h.lines.clear();
        }
        REQUIRE(h.send(command));
        REQUIRE(h.wait("bestmove "));
        REQUIRE(h.send("isready"));
        REQUIRE(h.count("info string error ") == 0);
    };
    if (!cached_newgames && !terminal_roots) REQUIRE(h.send("setoption name NNCacheSize value 0"));
    finish_search("go nodes 1");
    REQUIRE(h.calls == 1);
    for (unsigned repeat = 0; repeat < 12; ++repeat) {
        if (cached_newgames) REQUIRE(h.send("ucinewgame"));
        if (terminal_roots) REQUIRE(h.send("position fen 7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"));
        finish_search("go nodes 1");
        REQUIRE(h.calls == 1);
    }
    if (terminal_roots) REQUIRE(h.send("position startpos"));
    finish_search("go nodes 16 movetime 50");
    CHECK(h.calls == 1);
    {
        const std::lock_guard lock(h.mutex);
        CHECK(std::ranges::any_of(h.lines, [](const auto& line) {
            return line.starts_with("info depth ") && line.find(" nodes 1 ") != std::string::npos;
        }));
    }
    CHECK(h.session->exit_code() == 0);
}

TEST_CASE("Replacing the physical backend clears its former timing reserve",
          "[uci][session][latency-recovery]") {
    Harness h;
    h.evaluate_hook = [&] {
        if (h.calls.load() == 1) std::this_thread::sleep_for(std::chrono::milliseconds{100});
    };
    REQUIRE(h.send("go nodes 1"));
    REQUIRE(h.wait("bestmove "));
    REQUIRE(h.send("isready"));
    REQUIRE(h.calls == 1);
    REQUIRE(h.send("setoption name WeightsFile value replacement-fixture.onnx"));
    // Isolate latency-reserve reset from asynchronous model/tree startup.
    REQUIRE(h.send("isready"));
    {
        const std::lock_guard lock(h.mutex);
        h.lines.clear();
    }
    // 80ms minus the source-compatible30ms overhead preserves the original50ms budget.
    REQUIRE(h.send("go nodes 16 movetime 80"));
    REQUIRE(h.wait("bestmove "));
    REQUIRE(h.send("isready"));
    CHECK(h.created == 2);
    CHECK(h.calls > 2);
    {
        const std::lock_guard lock(h.mutex);
        CHECK(std::ranges::any_of(h.lines, [](const auto& line) {
            return line.starts_with("info depth ") && line.find(" nodes 16 ") != std::string::npos;
        }));
    }
    CHECK(h.count("info string error ") == 0);
    CHECK(h.session->exit_code() == 0);
}

TEST_CASE("A success-null backend factory fails readiness before wrapping or dereferencing it",
          "[uci][session][latency-recovery]") {
    Harness h{0, false};
    h.null_backend = true;
    REQUIRE(h.send("setoption name WeightsFile value session-fixture.onnx"));
    CHECK_FALSE(h.send("isready"));
    CHECK(h.created == 1);
    CHECK(h.calls == 0);
    CHECK(h.count("info string error ") == 1);
    CHECK(h.count("readyok") == 0);
    CHECK(h.count("bestmove ") == 0);
    CHECK(h.session->exit_code() == 1);
    CHECK(h.session->cancellation_token().stop_requested());
}

TEST_CASE("A two-second provider stall cannot delay a timed UCI bestmove",
          "[uci][session][blocked-deadline]") {
    Harness h;
    // This regression isolates an accepted inference call, not model startup.
    REQUIRE(h.send("isready"));
    bool warm_root = false;
    SECTION("The mandatory fresh root evaluation stalls") {}
    SECTION("A main-search evaluation stalls after a completed root") { warm_root = true; }
    if (warm_root) {
        REQUIRE(h.send("go nodes 1"));
        REQUIRE(h.wait("bestmove "));
        REQUIRE(h.send("isready"));
        const std::lock_guard lock(h.mutex);
        h.lines.clear();
    }
    const auto calls_before = h.calls.load();
    std::atomic<bool> evaluating{false};
    h.evaluate_hook = [&] {
        evaluating.store(true, std::memory_order_release);
        std::this_thread::sleep_for(std::chrono::seconds{2});
        evaluating.store(false, std::memory_order_release);
    };
    const auto started = std::chrono::steady_clock::now();
    REQUIRE(h.send("go wtime 300 btime 300"));
    bool answered = false;
    {
        std::unique_lock lock(h.mutex);
        answered = h.changed.wait_until(lock, started + std::chrono::milliseconds{270}, [&] {
            return std::ranges::any_of(
                h.lines, [](const auto& line) { return line.starts_with("bestmove "); });
        });
    }
    const auto reply_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
            .count();
    INFO("blocked-provider reply observed after " << reply_ms << " ms");
    CHECK(answered);
    if (answered) {
        // Design decision 2026-10-17: accepted work may
        // drain after bestmove, but no new provider call may begin.
        CHECK(evaluating.load(std::memory_order_acquire));
        const auto accepted = h.calls.load();
        std::this_thread::sleep_for(std::chrono::milliseconds{30});
        CHECK(h.calls == accepted);
        const mouse::chess::AttackTables attacks;
        auto root = mouse::chess::Position::from_fen(mouse::chess::kStartFen, attacks);
        REQUIRE(root);
        const std::lock_guard lock(h.mutex);
        for (const auto& line : h.lines)
            if (line.starts_with("bestmove "))
                CHECK(mouse::chess::parse_uci_move(*root, line.substr(9), attacks, false));
    }
    h.session->shutdown();
    CHECK_FALSE(evaluating.load(std::memory_order_acquire));  // Shutdown joined the call.
    CHECK(h.calls == calls_before + 1);  // The intended physical stall actually ran.
    CHECK(h.session->exit_code() == 0);
}

TEST_CASE("Deadline publication accepts a new timed root while its predecessor drains",
          "[uci][session][deadline-ownership]") {
    Harness h;
    REQUIRE(h.send("isready"));
    std::atomic<bool> entered{false}, release{false};
    h.evaluate_hook = [&] {
        entered.store(true, std::memory_order_release);
        while (!release.load(std::memory_order_acquire))
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
    };
    // Always release the real provider, including on an assertion failure.
    std::jthread unblock([&] {
        std::this_thread::sleep_for(std::chrono::seconds{1});
        release.store(true, std::memory_order_release);
    });
    const auto wait_best = [&](std::string_view move, std::chrono::steady_clock::time_point end) {
        std::unique_lock lock(h.mutex);
        return h.changed.wait_until(lock, end, [&] {
            return std::ranges::any_of(
                h.lines, [&](const auto& line) { return line == "bestmove " + std::string{move}; });
        });
    };
    auto started = std::chrono::steady_clock::now();
    REQUIRE(h.send("go wtime 300 btime 300 searchmoves e2e4"));
    CHECK(wait_best("e2e4", started + std::chrono::milliseconds{270}));
    CHECK(entered.load(std::memory_order_acquire));
    started = std::chrono::steady_clock::now();
    CHECK(h.send("position startpos moves e2e4 e7e5"));
    CHECK(h.send("go wtime 300 btime 300 searchmoves g1f3"));
    CHECK(wait_best("g1f3", started + std::chrono::milliseconds{270}));
    CHECK(std::chrono::steady_clock::now() - started < std::chrono::milliseconds{270});
    CHECK_FALSE(release.load(std::memory_order_acquire));
    release.store(true, std::memory_order_release);
    h.session->shutdown();
    CHECK(h.calls == 1);
    CHECK(h.count("bestmove ") == 2);
    CHECK(h.count("info string error ") == 0);
    CHECK(h.session->exit_code() == 0);
}

TEST_CASE("An inference failure discovered after deadline publication is still fatal",
          "[uci][session][deadline-ownership]") {
    Harness h{1};
    std::mutex gate;
    std::condition_variable changed;
    bool entered = false, released = false;
    h.evaluate_hook = [&] {
        std::unique_lock lock(gate);
        entered = true;
        changed.notify_all();
        changed.wait_for(lock, std::chrono::seconds{3}, [&] { return released; });
    };
    // This case tests a late fatal result, not the separate 270 ms clock gate.
    // Confirm actual provider acceptance before waiting for its early reply.
    REQUIRE(h.send("go movetime 1000 searchmoves e2e4"));
    {
        std::unique_lock lock(gate);
        REQUIRE(changed.wait_for(lock, std::chrono::seconds{2}, [&] { return entered; }));
    }
    CHECK(h.wait("bestmove e2e4"));
    {
        const std::lock_guard lock(gate);
        released = true;
        changed.notify_all();
    }
    REQUIRE(h.wait("info string error "));
    h.session->shutdown();
    CHECK(h.count("bestmove e2e4") == 1);
    CHECK(h.count("bestmove ") == 1);
    CHECK(h.calls == 1);
    CHECK(h.session->exit_code() == 1);
    CHECK(h.session->cancellation_token().stop_requested());
}

TEST_CASE("UCI debug timestamps bracket every real inference and bestmove",
          "[uci][session][deadline-trace]") {
    Harness h;
    REQUIRE(h.send("debug on"));
    REQUIRE(h.send("go nodes 1"));
    REQUIRE(h.wait("bestmove "));
    REQUIRE(h.send("isready"));
    std::vector<std::string> events;
    {
        const std::lock_guard lock(h.mutex);
        for (const auto& line : h.lines)
            if (line.starts_with("info string trace ")) events.push_back(line);
    }
    REQUIRE(events.size() == 6);
    CHECK(events[0].find("go steady_ns ") != std::string::npos);
    CHECK(events[1].find("batch_begin steady_ns ") != std::string::npos);
    CHECK(events[1].find(" batch 1 physical_batch 1") != std::string::npos);
    CHECK(events[2].find("batch_end steady_ns ") != std::string::npos);
    CHECK(events[2].find(" batch 1 physical_batch 1") != std::string::npos);
    CHECK(events[3].find("publication_begin steady_ns ") != std::string::npos);
    CHECK(events[4].find("publication_snapshot steady_ns ") != std::string::npos);
    CHECK(events[5].find("bestmove steady_ns ") != std::string::npos);
    REQUIRE(h.send("debug off"));
    REQUIRE(h.send("position startpos moves e2e4"));
    REQUIRE(h.send("go nodes 1"));
    REQUIRE(h.send("stop"));
    CHECK(h.count("info string trace ") == events.size());
    CHECK(h.session->exit_code() == 0);
}

TEST_CASE("GUI first go charges backend loading against its original clock",
          "[uci][session][startup-v1]") {
    Harness h{0, false,
              [](const auto&) { std::this_thread::sleep_for(std::chrono::milliseconds{2000}); }};
    REQUIRE(h.send("uci"));
    REQUIRE(h.send("setoption name WeightsFile value session-fixture.onnx"));
    REQUIRE(h.send("position startpos"));
    const auto began = std::chrono::steady_clock::now();
    REQUIRE(h.send("go wtime 10000 btime 10000 winc 100 binc 100"));
    REQUIRE(h.wait("bestmove "));
    const auto elapsed = std::chrono::steady_clock::now() - began;
    INFO("first go including model load ms="
         << (std::chrono::duration<double, std::milli>(elapsed).count()));
    // Real lc0 legacy first allocation at10000+100,ply1,overhead30 is895ms.
    CHECK(elapsed <= std::chrono::milliseconds{945});
    CHECK(h.count("bestmove a2a3") == 0);
    CHECK(h.count("info string warning: emergency move without evaluation") == 1);
    h.session->shutdown();
    CHECK(h.created == 1);
    CHECK(h.destroyed == 1);
}

TEST_CASE("A blocked model load cannot hold a timed publisher or the input owner",
          "[uci][session][startup-v1]") {
    std::atomic<bool> loading{false}, release{false};
    Harness h{0, false, [&](const auto&) {
                  loading.store(true, std::memory_order_release);
                  while (!release.load(std::memory_order_acquire))
                      std::this_thread::sleep_for(std::chrono::milliseconds{1});
              }};
    // Bound the red implementation too: it blocks synchronously in go.
    std::jthread unblock([&] {
        std::this_thread::sleep_for(std::chrono::seconds{1});
        release.store(true, std::memory_order_release);
    });
    REQUIRE(h.send("setoption name WeightsFile value session-fixture.onnx"));
    const auto began = std::chrono::steady_clock::now();
    REQUIRE(h.send("go movetime 100"));
    std::atomic<bool> ready_returned{false}, ready_success{false};
    std::jthread readiness([&] {
        ready_success.store(h.send("isready"), std::memory_order_release);
        ready_returned.store(true, std::memory_order_release);
    });
    REQUIRE(h.wait("bestmove "));
    const auto elapsed = std::chrono::steady_clock::now() - began;
    INFO("blocked load reply ms=" << (std::chrono::duration<double, std::milli>(elapsed).count()));
    CHECK(elapsed <= std::chrono::milliseconds{150});
    CHECK(loading.load(std::memory_order_acquire));
    CHECK_FALSE(release.load(std::memory_order_acquire));
    CHECK_FALSE(ready_returned.load(std::memory_order_acquire));
    release.store(true, std::memory_order_release);
    readiness.join();
    CHECK(ready_success.load(std::memory_order_acquire));
    CHECK(h.count("readyok") == 1);
    CHECK(h.created == 1);
    CHECK(h.session->exit_code() == 0);
}

TEST_CASE("Backend revisions discard an obsolete background load without leaking",
          "[uci][session][startup-v1]") {
    bool fail_old = false;
    SECTION("obsolete success is destroyed") {}
    SECTION("obsolete failure cannot poison the replacement") { fail_old = true; }
    std::atomic<bool> entered{false}, release{false};
    Harness h{0, false, [&](const auto& config) {
                  if (config.model_path.filename() != "old-fixture.onnx") return;
                  entered.store(true, std::memory_order_release);
                  while (!release.load(std::memory_order_acquire))
                      std::this_thread::sleep_for(std::chrono::milliseconds{1});
                  if (fail_old) throw std::runtime_error("obsolete backend error");
              }};
    std::jthread unblock([&] {
        std::this_thread::sleep_for(std::chrono::seconds{1});
        release.store(true, std::memory_order_release);
    });
    REQUIRE(h.send("setoption name WeightsFile value old-fixture.onnx"));
    const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds{150};
    while (!entered.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < end)
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    CHECK(entered.load(std::memory_order_acquire));
    REQUIRE(h.send("setoption name WeightsFile value replacement-fixture.onnx"));
    REQUIRE(h.send("setoption name Threads value 2"));
    release.store(true, std::memory_order_release);
    REQUIRE(h.send("isready"));
    CHECK(h.created == 2);
    CHECK(h.destroyed == (fail_old ? 0U : 1U));
    CHECK(h.calls == 0);
    h.session->shutdown();
    CHECK(h.destroyed == (fail_old ? 1U : 2U));
    CHECK(h.session->exit_code() == 0);
}

TEST_CASE("New game retires large trees off the input thread over twenty turns",
          "[uci][session][newgame-v1]") {
    Harness h;
    REQUIRE(h.send("setoption name RamLimitMb value 512"));
    REQUIRE(h.send("setoption name MoveOverheadMs value 5"));
    REQUIRE(h.count("info string error ") == 0);
    REQUIRE(h.send("isready"));
    double slowest = 0;
    for (unsigned repetition = 0; repetition < 20; ++repetition) {
        {
            const std::lock_guard lock(h.mutex);
            h.lines.clear();
        }
        const auto began = std::chrono::steady_clock::now();
        REQUIRE(h.send("ucinewgame"));
        const auto reset_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - began)
                .count();
        slowest = std::max(slowest, reset_ms);
        CHECK(reset_ms <= 5.0);
        REQUIRE(h.send("go wtime 30 btime 30"));
        REQUIRE(h.wait("bestmove "));
        CHECK(std::chrono::steady_clock::now() - began < std::chrono::milliseconds{25});
        REQUIRE(h.send("isready"));
        {
            const std::lock_guard lock(h.mutex);
            h.lines.clear();
        }
        REQUIRE(h.send("go nodes 1"));
        REQUIRE(h.wait("bestmove "));
        REQUIRE(h.send("isready"));
    }
    INFO("ucinewgame twenty repetitions max ms=" << slowest);
    CHECK(slowest <= 5.0);
}

TEST_CASE("Emergency choices rank visits Q and priors before reused evidence",
          "[uci][session][emergency-v1]") {
    using namespace mouse;
    const chess::Move a{chess::square(4, 1), chess::square(4, 3)};
    const chess::Move b{chess::square(3, 1), chess::square(3, 3)};
    const chess::Move c{chess::square(1, 0), chess::square(2, 2)};
    search::RootStatistics roots;
    roots.size = 2;
    roots.moves[0] = {.move = a, .visits = 4, .q = -0.5, .prior = 0.1F, .child_proof = {}};
    roots.moves[1] = {.move = b, .visits = 3, .q = 0.5, .prior = 0.9F, .child_proof = {}};
    CHECK(uci::emergency_choice(roots, {.reused = c, .deterministic = c}).move == a);
    roots.moves[1].visits = 4;
    CHECK(uci::emergency_choice(roots, {.reused = c, .deterministic = c}).move == b);
    roots.moves[0].q = 0.5;
    CHECK(uci::emergency_choice(roots, {.reused = c, .deterministic = c}).move == b);
    roots.moves[0].visits = roots.moves[1].visits = 0;
    roots.moves[0].q = 1.0;
    CHECK(uci::emergency_choice(roots, {.reused = c, .deterministic = c}).move == b);
    roots.size = 0;
    const auto reused = uci::emergency_choice(roots, {.reused = b, .deterministic = c});
    CHECK(reused.move == b);
    CHECK_FALSE(reused.without_evaluation);
    const auto fallback =
        uci::emergency_choice(roots, {.reused = std::nullopt, .deterministic = c});
    CHECK(fallback.move == c);
    CHECK(fallback.without_evaluation);
    const chess::AttackTables attacks;
    auto start = chess::Position::from_fen(chess::kStartFen, attacks);
    REQUIRE(start);
    const auto legal = chess::legal_moves(*start, attacks);
    chess::MoveList reversed;
    for (auto it = legal.end(); it != legal.begin();) reversed.push_back(*--it);
    CHECK(uci::deterministic_emergency(*start, legal, attacks) == b);
    CHECK(uci::deterministic_emergency(*start, reversed, attacks) == b);
}

TEST_CASE("An expired restricted turn cannot inherit another root move",
          "[uci][session][emergency-v1]") {
    Harness h;
    REQUIRE(h.send("go nodes 65 searchmoves d2d4"));
    REQUIRE(h.wait("bestmove d2d4"));
    REQUIRE(h.send("isready"));
    for (unsigned i = 0; i < 20; ++i) {
        {
            const std::lock_guard lock(h.mutex);
            h.lines.clear();
        }
        REQUIRE(h.send("go wtime 1 btime 1 searchmoves e2e4"));
        REQUIRE(h.wait("bestmove e2e4"));
        REQUIRE(h.send("isready"));
        CHECK(h.count("bestmove d2d4") == 0);
    }
}

TEST_CASE("A background load failure remains fatal after its timed reply",
          "[uci][session][startup-v1][late-load-v1]") {
    std::atomic<bool> release{false};
    Harness h{0, false, [&](const auto&) {
                  while (!release.load(std::memory_order_acquire))
                      std::this_thread::sleep_for(std::chrono::milliseconds{1});
                  throw std::runtime_error("late model load failure");
              }};
    std::jthread unblock([&] {
        std::this_thread::sleep_for(std::chrono::seconds{1});
        release.store(true, std::memory_order_release);
    });
    REQUIRE(h.send("setoption name WeightsFile value session-fixture.onnx"));
    REQUIRE(h.send("go movetime 100"));
    REQUIRE(h.wait("bestmove "));
    release.store(true, std::memory_order_release);
    CHECK(h.wait("info string error "));
    h.session->shutdown();
    CHECK(h.session->exit_code() == 1);
    CHECK(h.session->cancellation_token().stop_requested());
    CHECK(h.count("bestmove ") == 1);
    CHECK(h.created == 1);
    CHECK(h.destroyed == 0);
}

TEST_CASE("Ponderhit starts its deadline without waiting for accepted inference to drain",
          "[uci][session][legacy-v1][ponder-deadline-v1]") {
    unsigned failure = 0;
    SECTION("accepted inference drains successfully") {}
    SECTION("a late accepted inference error remains fatal") { failure = 1; }
    std::atomic<bool> entered{false}, released{false};
    Harness h{failure};
    REQUIRE(h.send("setoption name Ponder value true"));
    REQUIRE(h.send("isready"));
    h.evaluate_hook = [&] {
        entered.store(true, std::memory_order_release);
        // Model an in-flight provider waiting for device completion without
        // consuming a hosted CI CPU in a busy-yield loop.
        released.wait(false, std::memory_order_acquire);
    };
    std::jthread unblock([&] {
        std::this_thread::sleep_for(std::chrono::seconds{1});
        released.store(true, std::memory_order_release);
        released.notify_all();
    });
    REQUIRE(h.send("go ponder wtime 1000 btime 1000 winc 10 binc 10"));
    const auto wait_until = std::chrono::steady_clock::now() + std::chrono::milliseconds{500};
    while (!entered.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < wait_until)
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    REQUIRE(entered.load(std::memory_order_acquire));
    const auto hit = std::chrono::steady_clock::now();
    REQUIRE(h.send("ponderhit"));
    CHECK(std::chrono::steady_clock::now() - hit <= std::chrono::milliseconds{20});
    REQUIRE(h.wait("bestmove "));
    // Native legacy first allocation87ms, plus the required50ms publication allowance.
    CHECK(std::chrono::steady_clock::now() - hit <= std::chrono::milliseconds{137});
    CHECK_FALSE(released.load(std::memory_order_acquire));
    CHECK(h.calls == 1);
    CHECK(h.count("bestmove ") == 1);
    h.session->shutdown();
    CHECK(h.calls == 1);
    CHECK(h.destroyed == 1);
    CHECK(h.session->exit_code() == static_cast<int>(failure));
    CHECK(h.count("info string error ") == failure);
}

TEST_CASE("A known slow provider preserves the full clock without accepting an over-budget batch",
          "[uci][session][legacy-v1]") {
    Harness h;
    REQUIRE(h.send("isready"));
    h.evaluate_hook = [] { std::this_thread::sleep_for(std::chrono::milliseconds{100}); };
    REQUIRE(h.send("go nodes 1"));
    REQUIRE(h.wait("bestmove "));
    REQUIRE(h.send("isready"));
    REQUIRE(h.calls == 1);
    {
        const std::lock_guard lock(h.mutex);
        h.lines.clear();
    }
    const auto began = std::chrono::steady_clock::now();
    REQUIRE(h.send("go movetime 80"));  // Source common stopper subtracts30ms overhead.
    REQUIRE(h.wait("bestmove "));
    CHECK(std::chrono::steady_clock::now() - began >= std::chrono::milliseconds{50});
    CHECK(std::chrono::steady_clock::now() - began <= std::chrono::milliseconds{100});
    REQUIRE(h.send("isready"));
    CHECK(h.calls == 1);
}

#ifdef _WIN32
TEST_CASE("Windows session keeps millisecond deadline waits precise",
          "[uci][session][timer-precision-v1]") {
    Harness h{0, false};
    std::array<double, 200> samples{};
    std::mutex mutex;
    std::condition_variable_any changed;
    for (auto& sample : samples) {
        std::unique_lock lock(mutex);
        const auto started = std::chrono::steady_clock::now();
        changed.wait_until(lock, std::stop_token{}, started + std::chrono::milliseconds{1},
                           [] { return false; });
        sample =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
                .count();
    }
    std::ranges::sort(samples);
    INFO("median1mswait=" << samples.at(samples.size() / 2));
    CHECK(samples.at(samples.size() / 2) <= 3.0);
}
#endif

TEST_CASE(
    "Legal cache resize preserves the loaded backend and search-only updates retain cached values",
    "[uci][session][legal-cache]") {
    Harness h;
    REQUIRE(h.send("setoption name NNCacheSize value 2"));
    REQUIRE(h.send("go nodes 1"));
    REQUIRE(h.wait("bestmove "));
    REQUIRE(h.send("isready"));
    CHECK(h.created == 1);
    CHECK(h.calls == 1);
    REQUIRE(h.send("setoption name NNCacheSize value 3"));
    REQUIRE(h.send("isready"));
    CHECK(h.created == 1);
    {
        const std::lock_guard lock(h.mutex);
        h.lines.clear();
    }
    REQUIRE(h.send("go nodes 1"));
    REQUIRE(h.wait("bestmove "));
    REQUIRE(h.send("isready"));
    CHECK(h.calls == 2);
    REQUIRE(h.send("setoption name PolicyTemperature value 2"));
    REQUIRE(h.send("isready"));
    {
        const std::lock_guard lock(h.mutex);
        h.lines.clear();
    }
    REQUIRE(h.send("go nodes 1"));
    REQUIRE(h.wait("bestmove "));
    REQUIRE(h.send("isready"));
    CHECK(h.created == 1);
    CHECK(h.calls == 2);
    CHECK(h.session->exit_code() == 0);
}

TEST_CASE("An obsolete search preparation cannot poison a valid replacement during model load",
          "[uci][session][preparation-search-v1]") {
    std::atomic<bool> entered{false}, release{false};
    Harness h{0, false, [&](const mouse::nn::BackendConfig&) {
                  entered.store(true, std::memory_order_release);
                  while (!release.load(std::memory_order_acquire))
                      std::this_thread::sleep_for(std::chrono::milliseconds{1});
              }};
    REQUIRE(h.send("setoption name MinibatchSize value 64"));
    REQUIRE(h.send("setoption name RamLimitMb value 1"));
    REQUIRE(h.send("setoption name WeightsFile value session-fixture.onnx"));
    const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds{150};
    while (!entered.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < end)
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    CHECK(entered.load(std::memory_order_acquire));
    CHECK(h.send("setoption name RamLimitMb value 8"));
    release.store(true, std::memory_order_release);
    REQUIRE(h.send("isready"));
    CHECK(h.count("info string error ") == 0);
    CHECK(h.created == 1);
    CHECK(h.session->exit_code() == 0);
}

TEST_CASE("Final reported PV agrees with bestmove while completed batches race publication",
          "[uci][session][publication-pv-v1]") {
    Harness h;
    REQUIRE(h.send("setoption name MultiPV value 16"));
    REQUIRE(h.send("setoption name SmartPruningFactor value 0"));
    REQUIRE(h.send("setoption name MoveOverheadMs value 0"));
    h.evaluate_hook = [] { std::this_thread::sleep_for(std::chrono::milliseconds{1}); };
    REQUIRE(h.send("isready"));
    for (unsigned repetition = 0; repetition < 32; ++repetition) {
        REQUIRE(h.send("ucinewgame"));
        REQUIRE(h.send("position startpos"));
        REQUIRE(h.send("go movetime 60"));
        std::unique_lock lock(h.mutex);
        REQUIRE(h.changed.wait_for(lock, std::chrono::seconds{5}, [&] {
            return std::ranges::count_if(
                       h.lines, [](const auto& line) { return line.starts_with("bestmove "); }) > 0;
        }));
        std::string last_pv, last_info, best;
        for (const auto& line : h.lines) {
            if (line.starts_with("info depth ") && line.find(" multipv 1 ") != std::string::npos) {
                last_info = line;
                const auto index = line.find(" pv ");
                last_pv = index == std::string::npos ? std::string{} : line.substr(index + 4, 4);
            }
            if (line.starts_with("bestmove ")) {
                best = line.substr(9, 4);
                if (last_pv.empty()) {
                    // A completed root without a visited child has no PV.
                    // Enforce the existing golden contract; never fabricate one.
                    CHECK(last_info.starts_with("info depth 0 seldepth 0 "));
                    CHECK(last_info.find(" nodes 1 ") != std::string::npos);
                } else {
                    CHECK(last_pv == best);
                }
            }
        }
        REQUIRE_FALSE(best.empty());
        h.lines.clear();
    }
}

TEST_CASE("Search statistics are opt-in and emitted once after each drained move",
          "[uci][search-stats]") {
    Harness h;
    REQUIRE(h.send("uci"));
    CHECK(h.count("option name LogSearchStats type check default false") == 1);
    REQUIRE(h.send("go nodes 9"));
    REQUIRE(h.wait("bestmove "));
    REQUIRE(h.send("isready"));
    CHECK(h.count("info string searchstats ") == 0);
    REQUIRE(h.send("setoption name LogSearchStats value true"));
    REQUIRE(h.send("ucinewgame"));
    REQUIRE(h.send("position startpos"));
    REQUIRE(h.send("go nodes 65"));
    REQUIRE(h.wait("info string searchstats "));
    REQUIRE(h.send("isready"));
    CHECK(h.count("info string searchstats ") == 1);
    const std::lock_guard lock(h.mutex);
    const auto found = std::ranges::find_if(
        h.lines, [](const auto& line) { return line.starts_with("info string searchstats "); });
    REQUIRE(found != h.lines.end());
    for (const auto field :
         {"batches ", "mean ", "p50 ", "max ", "collisions ", "ooo ", "gather_ms ", "nn_wait_ms ",
          "backup_ms ", "gpu_idle_ms ", "idle_basis host_provider_gaps"})
        CHECK(found->find(field) != std::string::npos);
}

TEST_CASE("A periodic report cannot consume the final low-clock publication window",
          "[uci][session][periodic-deadline]") {
    Harness h;
    REQUIRE(h.send("setoption name MoveOverheadMs value 0"));
    REQUIRE(h.send("setoption name SmartPruningFactor value 0"));
    REQUIRE(h.send("isready"));
    REQUIRE(h.send("debug on"));
    h.evaluate_hook = [] { std::this_thread::sleep_for(std::chrono::milliseconds{2}); };
    std::atomic<bool> began_periodic{false}, finalizing{false};
    h.hook = [&](std::string_view line) {
        if (line.starts_with("info string trace publication_begin ")) finalizing = true;
        if (line.starts_with("info depth ") && !finalizing && !began_periodic.exchange(true))
            std::this_thread::sleep_for(std::chrono::milliseconds{20});
    };
    const auto started = std::chrono::steady_clock::now();
    REQUIRE(h.send("go wtime 105 btime 105 winc 100 binc 100"));
    bool answered = false;
    {
        std::unique_lock lock(h.mutex);
        answered = h.changed.wait_until(lock, started + std::chrono::milliseconds{105}, [&] {
            return std::ranges::any_of(
                h.lines, [](const auto& line) { return line.starts_with("bestmove "); });
        });
    }
    CHECK(answered);
    REQUIRE(h.wait("bestmove "));
    h.session->shutdown();
    CHECK(h.count("bestmove ") == 1);
    CHECK(h.session->exit_code() == 0);
}

TEST_CASE("Deadline delivery never acquires the live search tree mutex",
          "[uci][session][deadline-tree-lock]") {
    Harness h;
    REQUIRE(h.send("isready"));
    REQUIRE(h.send("debug on"));
    REQUIRE(h.send("setoption name SmartPruningFactor value 0"));
    h.evaluate_hook = [&] {
        if (h.calls.load() >= 2) std::this_thread::sleep_for(std::chrono::milliseconds{200});
    };
    const auto started = std::chrono::steady_clock::now();
    REQUIRE(h.send("go movetime 60"));
    bool answered = false;
    {
        std::unique_lock lock(h.mutex);
        answered = h.changed.wait_until(lock, started + std::chrono::milliseconds{60}, [&] {
            return std::ranges::any_of(
                h.lines, [](const auto& line) { return line.starts_with("bestmove "); });
        });
    }
    CHECK(answered);
    REQUIRE(h.wait("bestmove "));
    REQUIRE(h.send("isready"));
    CHECK(h.calls >= 2);
    std::string snapshot;
    {
        const std::lock_guard lock(h.mutex);
        for (const auto& line : h.lines)
            if (line.starts_with("info string trace publication_snapshot ")) snapshot = line;
    }
    INFO(snapshot);
    CHECK(snapshot.find(" tree_lock_count 0 tree_lock_wait_ns 0") != std::string::npos);
    CHECK(h.count("bestmove ") == 1);
    CHECK(h.session->exit_code() == 0);
}