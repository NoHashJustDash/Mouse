// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/search/search.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

#include "engine/chess/attacks.h"
#include "engine/chess/movegen.h"
#include "engine/nn/policy_map.h"
#include "tests/required_optional.h"

namespace {
using mouse::tests::required_optional;
struct TestBackend final : public mouse::nn::Backend {
    TestBackend() {
        metadata_.input_format = 1;
        info_.onnx_sha256 = "search-test-original";
    }
    mouse::Result<void> evaluate(std::span<const mouse::nn::InputPlanes> inputs,
                                 std::span<mouse::nn::NNOutput> outputs) override {
        ++calls;
        if (calls == nonstd_call) throw 42;
        const auto ready = std::chrono::steady_clock::now() + delay;
        while (std::chrono::steady_clock::now() < ready) std::this_thread::yield();
        if (long_error) return tl::unexpected(std::string(1U << 20U, 'X'));
        if (fail) return tl::unexpected(std::string{"injected provider failure"});
        for (std::size_t row = 0; row < inputs.size(); ++row) {
            mouse::nn::NNOutput value;
            value.wdl = {0.6F, 0.2F, 0.2F};
            value.moves_left = 10.0F;
            const bool black = inputs[row][108].mask != 0;
            if (white_advantage)
                value.wdl =
                    black ? std::array{0.005F, 0.0F, 0.995F} : std::array{0.995F, 0.0F, 0.005F};
            if (vary_mlh) {
                const auto white_pawns = inputs[row].at(black ? 6U : 0U).mask;
                const auto marker = mouse::chess::bit(mouse::chess::square(0, black ? 4 : 3));
                value.moves_left = (white_pawns & marker) != 0 ? 2.0F : 40.0F;
            }
            if (!supplied_mlh) value.moves_left.reset();
            if (concentrated) {
                // Authored nonzero priors model a narrow neural policy without
                // importing T1 weights into CI or changing the production code.
                std::uint64_t shape = 1469598103934665603ULL;
                for (std::size_t piece = 0; piece < 12; ++piece)
                    shape = (shape ^ inputs[row].at(piece).mask) * 1099511628211ULL;
                const auto win = static_cast<float>((shape >> 32) & 1023U) / 1023.0F;
                value.wdl = {win, 0.0F, 1.0F - win};
                for (std::size_t i = 0; i < value.policy_logits.size(); ++i)
                    value.policy_logits.at(i) =
                        static_cast<float>(static_cast<int>((i * 37 + shape % 101) % 101) - 50) /
                        10.0F;
            }
            if (cycle || dominant) {
                value.policy_logits.fill(-100.0F);
                value.policy_logits.at(*mouse::nn::PolicyMap::index(
                    mouse::chess::Move{mouse::chess::square(1, 0), mouse::chess::square(2, 2)},
                    mouse::chess::Color::White)) = 100.0F;
                if (cycle)
                    value.policy_logits.at(*mouse::nn::PolicyMap::index(
                        mouse::chess::Move{mouse::chess::square(2, 2), mouse::chess::square(1, 0)},
                        mouse::chess::Color::White)) = 100.0F;
            }
            if (invalid && row + 1 == inputs.size())
                value.wdl[0] = std::numeric_limits<float>::quiet_NaN();
            outputs[row] = value;
            std::uint64_t hash = 1469598103934665603ULL;
            for (const auto& plane : inputs[row]) {
                hash = (hash ^ plane.mask) * 1099511628211ULL;
                hash = (hash ^ std::bit_cast<std::uint32_t>(plane.value)) * 1099511628211ULL;
            }
            trace.at(rows++) = hash;
        }
        batches.at(batch_count++) = inputs.size();
        return {};
    }
    [[nodiscard]] const mouse::nn::ModelMetadata& metadata() const noexcept override {
        return metadata_;
    }
    [[nodiscard]] const mouse::nn::BackendInfo& info() const noexcept override { return info_; }
    mouse::nn::BackendInfo& mutable_info() noexcept { return info_; }
    std::size_t calls = 0, rows = 0, batch_count = 0;
    std::size_t nonstd_call = 0;
    std::array<std::uint64_t, 8192> trace{};
    std::array<std::size_t, 8192> batches{};
    bool fail = false, invalid = false, cycle = false, dominant = false, long_error = false;
    bool white_advantage = false, vary_mlh = false, supplied_mlh = true, concentrated = false;
    std::chrono::milliseconds delay{0};

    mouse::nn::ModelMetadata metadata_;
    mouse::nn::BackendInfo info_;
};
mouse::search::SearchConfig config() {
    mouse::search::SearchConfig result;
    result.threads = 1;
    result.maximum_batch = 4;
    result.ram_limit_bytes = 4U << 20U;
    result.max_depth = 32;
    result.deterministic = true;
    return result;
}
mouse::chess::Position position(const mouse::chess::AttackTables& attacks,
                                std::string_view fen = mouse::chess::kStartFen) {
    auto parsed = mouse::chess::Position::from_fen(fen, attacks);
    if (!parsed) throw std::runtime_error(parsed.error());
    return std::move(*parsed);
}
mouse::nn::EncodingHistory history(const mouse::chess::Position& root) {
    mouse::nn::EncodingHistory result;
    result.reset(root);
    return result;
}
auto create(const mouse::search::SearchConfig& selected, const mouse::chess::AttackTables& attacks,
            TestBackend& backend, const mouse::chess::Position& root,
            const mouse::nn::EncodingHistory& frames) {
    auto engine = mouse::search::SearchEngine::create(selected, attacks, backend);
    if (!engine) throw std::runtime_error(engine.error());
    auto set = (*engine)->set_position(root, frames);
    if (!set) throw std::runtime_error(set.error());
    return std::move(*engine);
}
mouse::search::SearchResult run(mouse::search::SearchEngine& engine, std::uint64_t nodes) {
    mouse::search::SearchLimits limits;
    limits.nodes = nodes;
    auto searched = engine.run(limits);
    if (!searched) throw std::runtime_error(searched.error());
    return *searched;
}
}  // namespace

TEST_CASE("Search validates configuration and root history without fallback", "[search]") {
    const mouse::chess::AttackTables attacks;
    TestBackend backend;
    auto selected = config();
    SECTION("No worker") { selected.threads = 0; }
    SECTION("No batch") { selected.maximum_batch = 0; }
    SECTION("Oversized batch") { selected.maximum_batch = 1025; }
    SECTION("No managed memory") { selected.ram_limit_bytes = 0; }
    SECTION("Invalid depth") { selected.max_depth = 513; }
    SECTION("Invalid selection") {
        selected.selection.cpuct = std::numeric_limits<double>::quiet_NaN();
    }
    CHECK_FALSE(mouse::search::SearchEngine::create(selected, attacks, backend));
}

TEST_CASE("Search counts completed root playouts exactly and backs up alternating values",
          "[search]") {
    const mouse::chess::AttackTables attacks;
    auto root = position(attacks);
    TestBackend backend;
    auto engine = create(config(), attacks, backend, root, history(root));
    const auto first = run(*engine, 1);
    CHECK(first.info.nodes == 1);
    CHECK(first.info.new_playouts == 1);
    CHECK(first.info.inherited_visits == 0);
    CHECK(std::abs(first.info.q - 0.4) < 1e-7);
    REQUIRE(first.best_move.has_value());
    const auto second = run(*engine, 2);
    CHECK(second.info.nodes == 2);
    CHECK(second.info.new_playouts == 1);
    CHECK(second.info.inherited_visits == 1);
    CHECK(std::abs(second.info.q) < 1e-7);
    const auto result = run(*engine, 257);
    CHECK(result.info.nodes == 257);
    CHECK(result.info.new_playouts == 255);
    CHECK(result.reason == mouse::search::StopReason::NodeLimit);
    std::uint64_t child_visits = 0;
    const auto moves = engine->root_moves();
    for (std::size_t index = 0; index < moves.size; ++index)
        child_visits += moves.moves.at(index).visits;
    CHECK(child_visits == 256);
    CHECK(result.info.charged_bytes <= result.info.budget_bytes);
    CHECK(result.info.tree_nodes <= result.info.tree_capacity);
    auto legal = mouse::chess::legal_moves(root, attacks);
    CHECK(std::ranges::find(legal, required_optional(result.best_move)) != legal.end());
}

TEST_CASE("Deterministic search reproduces all encoded rows and batch boundaries", "[search]") {
    const mouse::chess::AttackTables attacks;
    const auto root = position(attacks);
    TestBackend first_backend, second_backend;
    auto first = create(config(), attacks, first_backend, root, history(root));
    auto selected = config();
    selected.threads = 8;
    auto second = create(selected, attacks, second_backend, root, history(root));
    const auto a = run(*first, 400), b = run(*second, 400);
    CHECK(a.best_move == b.best_move);
    CHECK(a.info.nodes == b.info.nodes);
    CHECK(a.info.seldepth == b.info.seldepth);
    CHECK(std::bit_cast<std::uint64_t>(a.info.q) == std::bit_cast<std::uint64_t>(b.info.q));
    CHECK(std::bit_cast<std::uint64_t>(a.info.draw) == std::bit_cast<std::uint64_t>(b.info.draw));
    CHECK(first_backend.rows == second_backend.rows);
    CHECK(first_backend.batch_count == second_backend.batch_count);
    CHECK(std::equal(first_backend.trace.begin(),
                     first_backend.trace.begin() + static_cast<std::ptrdiff_t>(first_backend.rows),
                     second_backend.trace.begin()));
    CHECK(std::equal(
        first_backend.batches.begin(),
        first_backend.batches.begin() + static_cast<std::ptrdiff_t>(first_backend.batch_count),
        second_backend.batches.begin()));
    const auto x = first->root_moves(), y = second->root_moves();
    REQUIRE(x.size == y.size);
    for (std::size_t i = 0; i < x.size; ++i) {
        CHECK(x.moves.at(i).move == y.moves.at(i).move);
        CHECK(x.moves.at(i).visits == y.moves.at(i).visits);
        CHECK(x.moves.at(i).q == y.moves.at(i).q);
        CHECK(x.moves.at(i).draw == y.moves.at(i).draw);
    }
}

TEST_CASE("Terminal roots use exact game values without neural calls", "[search]") {
    const mouse::chess::AttackTables attacks;
    TestBackend backend;
    auto root = position(attacks, "7k/6Q1/6K1/8/8/8/8/8 b - - 100 1");
    double expected_q = -1, expected_draw = 0;
    SECTION("Checkmate before rule50") {}
    SECTION("Stalemate") {
        root = position(attacks, "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");
        expected_q = 0;
        expected_draw = 1;
    }
    auto engine = create(config(), attacks, backend, root, history(root));
    const auto result = run(*engine, 50);
    CHECK(result.reason == mouse::search::StopReason::Terminal);
    CHECK_FALSE(result.best_move);
    CHECK(result.info.q == expected_q);
    CHECK(result.info.draw == expected_draw);
    CHECK(backend.calls == 0);
}

TEST_CASE("Search failure publishes no failed batch and drains reservations for recovery",
          "[search]") {
    const mouse::chess::AttackTables attacks;
    const auto root = position(attacks);
    TestBackend backend;
    auto engine = create(config(), attacks, backend, root, history(root));
    SECTION("Provider error") { backend.fail = true; }
    SECTION("Invalid full output") { backend.invalid = true; }
    const auto failed = engine->run({});
    REQUIRE_FALSE(failed);
    CHECK(engine->snapshot().nodes == 0);
    CHECK(engine->root_moves().size == 0);
    backend.fail = false;
    backend.invalid = false;
    const auto result = run(*engine, 80);
    CHECK(result.info.nodes == 80);
    REQUIRE(result.best_move);
    const auto advanced = engine->advance(required_optional(result.best_move));
    REQUIRE(advanced.has_value());
}

TEST_CASE("Search reuses only full root identities and verified continuations", "[search]") {
    const mouse::chess::AttackTables attacks;
    auto root = position(attacks);
    auto frames = history(root);
    TestBackend backend;
    auto engine = create(config(), attacks, backend, root, frames);
    const auto initial = run(*engine, 128);
    REQUIRE(engine->set_position(root, frames));
    CHECK(engine->snapshot().nodes == 128);
    const auto statistics = engine->root_moves();
    REQUIRE(initial.best_move);
    std::uint64_t retained = 0;
    for (std::size_t i = 0; i < statistics.size; ++i)
        if (statistics.moves.at(i).move == required_optional(initial.best_move))
            retained = statistics.moves.at(i).visits;
    const auto advanced = engine->advance(required_optional(initial.best_move));
    REQUIRE(advanced);
    CHECK_FALSE(advanced->discarded_context);
    const auto next = run(*engine, retained + 20);
    CHECK(next.info.inherited_visits == retained);
    CHECK(next.info.new_playouts == 20);
    CHECK_FALSE(engine->advance(mouse::chess::Move{0, 0}));
    CHECK(engine->snapshot().nodes == next.info.nodes);
    REQUIRE(engine->set_position(root, frames));
    CHECK(engine->snapshot().nodes == 0);
    auto inconsistent = history(position(attacks, "4k3/8/8/8/8/8/8/R3K3 w - - 0 1"));
    CHECK_FALSE(engine->set_position(root, inconsistent));
    CHECK(engine->snapshot().nodes == 0);
}

TEST_CASE("Contextual tree twofold draws are discarded on advance", "[search]") {
    const mouse::chess::AttackTables attacks;
    const auto root = position(attacks);
    TestBackend backend;
    backend.cycle = true;
    auto selected = config();
    selected.selection.cpuct = 100;
    auto engine = create(selected, attacks, backend, root, history(root));
    const auto result = run(*engine, 64);
    REQUIRE(result.best_move);
    CHECK(required_optional(result.best_move) ==
          mouse::chess::Move{mouse::chess::square(1, 0), mouse::chess::square(2, 2)});
    const auto moved = engine->advance(required_optional(result.best_move));
    REQUIRE(moved);
    CHECK(moved->discarded_context);
    CHECK(engine->snapshot().nodes == 0);
}

TEST_CASE("Depth frontiers are heuristic and thaw when a retained child becomes root", "[search]") {
    const mouse::chess::AttackTables attacks;
    const auto root = position(attacks);
    TestBackend backend;
    auto selected = config();
    selected.max_depth = 1;
    auto engine = create(selected, attacks, backend, root, history(root));
    const auto result = run(*engine, 100);
    CHECK(result.reason == mouse::search::StopReason::NodeLimit);
    CHECK(result.info.seldepth == 1);
    CHECK(backend.rows <= 21);
    REQUIRE(result.best_move);
    REQUIRE(engine->advance(required_optional(result.best_move)));
    const auto inherited = engine->snapshot().nodes;
    const auto before = backend.calls;
    CHECK(run(*engine, inherited + 1).info.nodes == inherited + 1);
    CHECK(backend.calls > before);
    CHECK(engine->root_moves().size > 0);
}

TEST_CASE("Search validates limits and immutable backend identity", "[search]") {
    const mouse::chess::AttackTables attacks;
    const auto root = position(attacks);
    TestBackend backend;
    auto engine = create(config(), attacks, backend, root, history(root));
    mouse::search::SearchLimits limits;
    SECTION("zero nodes") { limits.nodes = 0; }
    SECTION("zero time") { limits.movetime = std::chrono::milliseconds{0}; }
    SECTION("zero depth") { limits.depth = 0; }
    SECTION("oversized depth") { limits.depth = 33; }
    SECTION("different model") { backend.mutable_info().onnx_sha256 = "changed"; }
    CHECK_FALSE(engine->run(limits));
    CHECK(engine->snapshot().nodes == 0);
    CHECK(backend.calls == 0);
}
TEST_CASE("Depth and RAM stops retain completed legal results", "[search]") {
    const mouse::chess::AttackTables attacks;
    const auto root = position(attacks);
    TestBackend backend;
    auto selected = config();
    bool memory = false;
    SECTION("Requested depth") {}
    SECTION("Arena edge exhaustion") {
        selected.ram_limit_bytes = 512U << 10U;
        memory = true;
    }
    auto engine = create(selected, attacks, backend, root, history(root));
    mouse::search::SearchLimits limits;
    limits.nodes = 100000;
    if (!memory) limits.depth = 2;
    const auto result = engine->run(limits);
    REQUIRE(result);
    CHECK(result->reason == (memory ? mouse::search::StopReason::MemoryLimit
                                    : mouse::search::StopReason::DepthLimit));
    CHECK(result->info.nodes > 0);
    CHECK(result->info.nodes < 100000);
    CHECK(result->info.charged_bytes <= result->info.budget_bytes);
    REQUIRE(result->best_move);
    CHECK(engine->advance(required_optional(result->best_move)));
}
TEST_CASE("Raw double-push EP is compatible with normalized appended neural history", "[search]") {
    const mouse::chess::AttackTables attacks;
    auto root = position(attacks);
    auto frames = history(root);
    const auto move = mouse::chess::parse_uci_move(root, "e2e4", attacks);
    REQUIRE(move);
    static_cast<void>(root.make_move(*move, attacks));
    REQUIRE(frames.append(root));
    REQUIRE(root.en_passant_square() != mouse::chess::kNoSquare);
    REQUIRE(frames.frames().back().en_passant_square() == mouse::chess::kNoSquare);
    TestBackend backend;
    auto engine = create(config(), attacks, backend, root, frames);
    CHECK(run(*engine, 1).info.nodes == 1);
    // Same board from FEN has raw EP and different complete repetition history.
    const auto fresh = position(attacks, root.fen());
    auto raw = history(fresh);
    REQUIRE(engine->set_position(fresh, raw));
    CHECK(engine->snapshot().nodes == 0);
    CHECK(run(*engine, 1).info.nodes == 1);
}
TEST_CASE("Pre-root repetitions do not remove legal root move choice", "[search]") {
    const mouse::chess::AttackTables attacks;
    auto root = position(attacks);
    auto frames = history(root);
    const auto cycle = [&] {
        for (const auto token : {"b1c3", "b8c6", "c3b1", "c6b8"}) {
            auto move = mouse::chess::parse_uci_move(root, token, attacks);
            REQUIRE(move);
            static_cast<void>(root.make_move(*move, attacks));
            REQUIRE(frames.append(root));
        }
    };
    cycle();
    REQUIRE(root.is_repetition(2));
    REQUIRE_FALSE(root.is_repetition(3));
    TestBackend backend;
    auto engine = create(config(), attacks, backend, root, frames);
    CHECK(run(*engine, 1).reason == mouse::search::StopReason::NodeLimit);
    CHECK(backend.calls == 1);
    cycle();
    REQUIRE(root.is_repetition(3));
    REQUIRE(engine->set_position(root, frames));
    const auto result = run(*engine, 100);
    CHECK(result.reason == mouse::search::StopReason::NodeLimit);
    CHECK(result.info.nodes == 100);
    REQUIRE(result.best_move);
    CHECK(backend.calls > 1);
}
TEST_CASE("Node-only smart pruning preserves exact limits and large clocks are bounded",
          "[search]") {
    const mouse::chess::AttackTables attacks;
    const auto root = position(attacks);
    TestBackend backend;
    auto engine = create(config(), attacks, backend, root, history(root));
    mouse::search::SearchLimits limits;
    limits.nodes = 50;
    limits.movetime = std::chrono::milliseconds::max();
    const auto result = engine->run(limits);
    REQUIRE(result);
    CHECK(result->reason == mouse::search::StopReason::NodeLimit);
    CHECK(result->info.nodes == 50);
    limits.nodes = 100;
    limits.movetime.reset();
    const auto next = engine->run(limits);
    REQUIRE(next);
    CHECK(next->reason == mouse::search::StopReason::NodeLimit);
    CHECK(next->info.nodes == 100);
}

TEST_CASE("Timed smart pruning stops only after measured startup and a decisive visit lead",
          "[search]") {
    const mouse::chess::AttackTables attacks;
    const auto root = position(attacks);
    TestBackend backend;
    backend.dominant = true;
    backend.delay = std::chrono::milliseconds{1};
    auto engine = create(config(), attacks, backend, root, history(root));
    mouse::search::SearchLimits limits;
    limits.nodes.reset();
    limits.movetime = std::chrono::milliseconds{300};
    const auto result = engine->run(limits);
    REQUIRE(result);
    CHECK(result->reason == mouse::search::StopReason::SmartPruning);
    CHECK(result->info.new_playouts >= 100);
    CHECK(result->info.elapsed >= std::chrono::milliseconds{200});
    CHECK(result->info.elapsed < *limits.movetime);
    REQUIRE(result->best_move);
}

TEST_CASE("Provider diagnostics cannot retain unbounded search storage", "[search]") {
    const mouse::chess::AttackTables attacks;
    const auto root = position(attacks);
    TestBackend backend;
    auto engine = create(config(), attacks, backend, root, history(root));
    backend.long_error = true;
    const auto failed = engine->run({});
    REQUIRE_FALSE(failed);
    CHECK(failed.error().size() < 1200);
    CHECK(failed.error().find("truncated") != std::string::npos);
    CHECK(engine->snapshot().charged_bytes <= engine->snapshot().budget_bytes);
    backend.long_error = false;
    CHECK(run(*engine, 1).info.nodes == 1);
}

TEST_CASE("Exact mate leaves dominate neural heuristics for either root colour", "[search]") {
    const mouse::chess::AttackTables attacks;
    auto root = position(attacks, "7k/8/5KQ1/8/8/8/8/8 w - - 0 1");
    std::string_view mate = "g6g7";
    SECTION("White") {}
    SECTION("Black") {
        root = position(attacks, "8/8/8/8/8/5kq1/8/7K b - - 0 1");
        mate = "g3g2";
    }
    const auto expected = mouse::chess::parse_uci_move(root, mate, attacks);
    REQUIRE(expected);
    TestBackend backend;
    auto engine = create(config(), attacks, backend, root, history(root));
    const auto result = run(*engine, 500);
    REQUIRE(result.best_move);
    CHECK(required_optional(result.best_move) == *expected);
    CHECK(result.info.pv.size == 1);
    const auto statistics = engine->root_moves();
    for (std::size_t i = 0; i < statistics.size; ++i)
        if (statistics.moves.at(i).move == *expected) {
            CHECK(statistics.moves.at(i).q == 1);
            CHECK(statistics.moves.at(i).draw == 0);
        }
}
TEST_CASE("Replay classification failures restore workers and release reservations", "[search]") {
    const mouse::chess::AttackTables attacks;
    const auto root = position(
        attacks, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 18446744073709551614");
    const auto unchanged = position(attacks, root.fen());
    TestBackend backend;
    auto engine = create(config(), attacks, backend, root, history(root));
    const auto failed = engine->run({});
    REQUIRE_FALSE(failed);
    CHECK(failed.error().find("counter") != std::string::npos);
    CHECK(root == unchanged);
    const auto pv = engine->principal_variation();
    REQUIRE(pv.size > 0);
    CHECK(engine->advance(pv.moves[0]));
}

TEST_CASE("Reported PV contains completed edges and never exceeds selective depth", "[search]") {
    const mouse::chess::AttackTables attacks;
    const auto root = position(attacks);
    TestBackend backend;
    auto engine = create(config(), attacks, backend, root, history(root));
    const auto first = run(*engine, 1);
    CHECK(first.info.pv.size == 0);
    CHECK(first.info.depth == 0);
    CHECK(first.info.seldepth == 0);
    CHECK(first.info.depth <= first.info.seldepth);
    const auto result = run(*engine, 100);
    CHECK(result.info.depth <= result.info.seldepth);
    const auto again = run(*engine, 100);
    CHECK(again.info.depth <= again.info.seldepth);
    CHECK(again.info.seldepth == result.info.seldepth);
    CHECK(again.info.new_playouts == 0);
    REQUIRE(result.best_move);
    REQUIRE(engine->advance(required_optional(result.best_move)));
    CHECK(engine->snapshot().depth <= engine->snapshot().seldepth);
}

TEST_CASE("Backend nonstandard exceptions become recoverable search errors",
          "[search][pipeline][nonstd]") {
    const mouse::chess::AttackTables attacks;
    const auto root = position(attacks);
    TestBackend backend;
    std::uint64_t completed = 0;
    SECTION("First inference call") { backend.nonstd_call = 1; }
    SECTION("Later inference call") {
        backend.nonstd_call = 2;
        completed = 1;
    }
    auto engine = create(config(), attacks, backend, root, history(root));
    const auto failed = engine->run({});
    REQUIRE_FALSE(failed);
    CHECK(failed.error().find("non-standard") != std::string::npos);
    CHECK(engine->snapshot().nodes == completed);
    const auto statistics = engine->root_moves();
    for (std::size_t i = 0; i < statistics.size; ++i) CHECK(statistics.moves.at(i).visits == 0);
    backend.nonstd_call = 0;
    const auto recovered = run(*engine, 50);
    CHECK(recovered.info.nodes == 50);
    REQUIRE(recovered.best_move);
    CHECK(engine->advance(required_optional(recovered.best_move)));
}

TEST_CASE("Known MLH changes uniform-Q allocation without changing the most-visits best rule",
          "[search][moves-left]") {
    const mouse::chess::AttackTables attacks;
    auto root = position(attacks);
    TestBackend active, disabled;
    active.white_advantage = disabled.white_advantage = true;
    active.vary_mlh = disabled.vary_mlh = true;
    auto on = config(), off = config();
    off.moves_left_utility = false;
    auto a = create(on, attacks, active, root, history(root));
    auto b = create(off, attacks, disabled, root, history(root));
    const auto x = run(*a, 1024), y = run(*b, 1024);
    const auto short_move = mouse::chess::parse_uci_move(root, "a2a4", attacks);
    REQUIRE(short_move);
    const auto first = a->root_moves(), second = b->root_moves();
    std::uint64_t active_short = 0, disabled_short = 0;
    for (std::size_t i = 0; i < first.size; ++i) {
        if (first.moves.at(i).move == *short_move) active_short = first.moves.at(i).visits;
        if (second.moves.at(i).move == *short_move) disabled_short = second.moves.at(i).visits;
    }
    CHECK(active_short > disabled_short);
    CHECK(x.info.nodes == 1024);
    CHECK(y.info.nodes == 1024);
    REQUIRE(x.best_move);
    REQUIRE(y.best_move);
    for (const auto* stats : {&first, &second}) {
        const auto chosen =
            stats == &first ? required_optional(x.best_move) : required_optional(y.best_move);
        std::uint64_t chosen_visits = 0;
        for (std::size_t i = 0; i < stats->size; ++i)
            if (stats->moves.at(i).move == chosen) chosen_visits = stats->moves.at(i).visits;
        for (std::size_t i = 0; i < stats->size; ++i)
            CHECK(chosen_visits >= stats->moves.at(i).visits);
    }
    CHECK(x.info.charged_bytes <= x.info.budget_bytes);
}
TEST_CASE("An absent MLH preserves the disabled-feature Q plus U trace exactly",
          "[search][moves-left]") {
    const mouse::chess::AttackTables attacks;
    const auto root = position(attacks);
    TestBackend absent, disabled;
    absent.white_advantage = disabled.white_advantage = true;
    absent.vary_mlh = disabled.vary_mlh = true;
    absent.supplied_mlh = false;
    auto on = config(), off = config();
    off.moves_left_utility = false;
    auto a = create(on, attacks, absent, root, history(root));
    auto b = create(off, attacks, disabled, root, history(root));
    const auto x = run(*a, 400), y = run(*b, 400);
    CHECK(x.best_move == y.best_move);
    CHECK(x.info.q == y.info.q);
    CHECK(x.info.draw == y.info.draw);
    REQUIRE(absent.rows == disabled.rows);
    REQUIRE(absent.batch_count == disabled.batch_count);
    for (std::size_t i = 0; i < absent.rows; ++i) CHECK(absent.trace.at(i) == disabled.trace.at(i));
    for (std::size_t i = 0; i < absent.batch_count; ++i)
        CHECK(absent.batches.at(i) == disabled.batches.at(i));
    const auto first = a->root_moves(), second = b->root_moves();
    REQUIRE(first.size == second.size);
    for (std::size_t i = 0; i < first.size; ++i) {
        CHECK(first.moves.at(i).visits == second.moves.at(i).visits);
        CHECK(first.moves.at(i).q == second.moves.at(i).q);
    }
}
TEST_CASE("MLH search and retained continuations reproduce deterministic traces",
          "[search][moves-left]") {
    const mouse::chess::AttackTables attacks;
    const auto root = position(attacks);
    TestBackend first, second;
    first.white_advantage = second.white_advantage = true;
    first.vary_mlh = second.vary_mlh = true;
    auto one = config(), many = config();
    many.threads = 8;
    auto a = create(one, attacks, first, root, history(root));
    auto b = create(many, attacks, second, root, history(root));
    const auto initial = run(*a, 200), matching = run(*b, 200);
    REQUIRE(initial.best_move);
    CHECK(initial.best_move == matching.best_move);
    REQUIRE(a->advance(required_optional(initial.best_move)));
    REQUIRE(b->advance(required_optional(initial.best_move)));
    const auto limit = a->snapshot().nodes + 200;
    const auto x = run(*a, limit), y = run(*b, limit);
    CHECK(x.best_move == y.best_move);
    CHECK(x.info.q == y.info.q);
    REQUIRE(first.rows == second.rows);
    REQUIRE(first.batch_count == second.batch_count);
    for (std::size_t i = 0; i < first.rows; ++i) CHECK(first.trace.at(i) == second.trace.at(i));
    for (std::size_t i = 0; i < first.batch_count; ++i)
        CHECK(first.batches.at(i) == second.batches.at(i));
}

TEST_CASE("Legal rule-draw roots return a move instead of a terminal no-move result",
          "[search][root-draw]") {
    const mouse::chess::AttackTables attacks;
    auto root = position(attacks);
    auto frames = history(root);
    SECTION("Threefold") {
        for (int cycle = 0; cycle < 2; ++cycle) {
            for (const auto* text : {"g1f3", "g8f6", "f3g1", "f6g8"}) {
                const auto move = mouse::chess::parse_uci_move(root, text, attacks);
                REQUIRE(move);
                static_cast<void>(root.make_move(*move, attacks));
                REQUIRE(frames.append(root));
            }
        }
        REQUIRE(root.is_repetition(3));
    }
    SECTION("Fifty move") {
        root = position(attacks, "4k3/8/8/8/8/8/8/R3K3 w - - 100 1");
        frames = history(root);
        REQUIRE(root.is_rule50_draw());
    }
    SECTION("Insufficient material") {
        root = position(attacks, "4k3/8/8/8/8/8/8/4K3 w - - 0 1");
        frames = history(root);
        REQUIRE(root.insufficient_material());
    }
    TestBackend backend;
    auto engine = create(config(), attacks, backend, root, frames);
    const auto result = run(*engine, 30);
    CHECK(result.reason != mouse::search::StopReason::Terminal);
    REQUIRE(result.best_move);
    const auto legal = mouse::chess::legal_moves(root, attacks);
    CHECK(std::ranges::find(legal, required_optional(result.best_move)) != legal.end());
    CHECK(result.info.nodes == 30);
    CHECK(backend.calls >= 1);
    CHECK(frames.frames().back().side_to_move() == root.side_to_move());
}

TEST_CASE("A retained terminal rule-draw child is searched afresh when it becomes root",
          "[search][root-draw]") {
    const mouse::chess::AttackTables attacks;
    auto root = position(attacks, "4k3/8/8/8/8/8/8/R3K3 w - - 99 1");
    TestBackend backend;
    auto frames = history(root);
    auto engine = create(config(), attacks, backend, root, frames);
    const auto only = mouse::chess::parse_uci_move(root, "a1a2", attacks);
    REQUIRE(only);
    mouse::search::SearchLimits initial;
    initial.nodes = 50;
    initial.searchmoves[0] = *only;
    initial.searchmoves_count = 1;
    const auto first = engine->run(initial);
    REQUIRE(first);
    REQUIRE(first->best_move);
    const auto roots = engine->root_moves();
    const auto selected = std::ranges::find_if(
        roots.moves.begin(), roots.moves.begin() + static_cast<std::ptrdiff_t>(roots.size),
        [](const auto& move) { return move.visits != 0; });
    REQUIRE(selected != roots.moves.begin() + static_cast<std::ptrdiff_t>(roots.size));
    REQUIRE(selected->child_proof.outcome == mouse::search::ProofOutcome::Draw);
    REQUIRE(selected->visits == 49);
    const auto before_calls = backend.calls;
    REQUIRE(engine->advance(selected->move));
    static_cast<void>(root.make_move(selected->move, attacks));
    REQUIRE(frames.append(root));
    REQUIRE(root.is_rule50_draw());
    CHECK(engine->snapshot().nodes == 0);
    REQUIRE(engine->set_position(root, frames));
    CHECK(engine->snapshot().nodes == 0);
    const auto prime = run(*engine, std::max(std::uint64_t{1}, engine->snapshot().nodes));
    CHECK(prime.info.new_playouts == 1);
    CHECK(prime.info.nodes == 1);
    CHECK(backend.calls == before_calls + 1);
    const auto second = run(*engine, 30);
    CHECK(second.reason != mouse::search::StopReason::Terminal);
    REQUIRE(second.best_move);
    const auto legal = mouse::chess::legal_moves(root, attacks);
    CHECK(std::ranges::find(legal, required_optional(second.best_move)) != legal.end());
    CHECK(second.info.nodes == 30);
    CHECK(backend.calls > before_calls);
}

TEST_CASE("Unrelated roots clear populated statistics and evaluate their own neural input",
          "[search][root-reset]") {
    const mouse::chess::AttackTables attacks;
    const auto first = position(attacks);
    auto second = position(attacks, "4k3/8/8/8/8/8/P7/R3K3 b - - 0 1");
    TestBackend backend, reference;
    auto engine = create(config(), attacks, backend, first, history(first));
    REQUIRE(run(*engine, 257).info.tree_nodes > 100);
    const auto capacity = engine->snapshot().tree_capacity;
    for (int repetition = 0; repetition < 4; ++repetition) {
        const auto& next = repetition % 2 == 0 ? second : first;
        auto fresh = create(config(), attacks, reference, next, history(next));
        const auto expected_row = reference.rows;
        REQUIRE(run(*fresh, 1).info.nodes == 1);
        REQUIRE(engine->set_position(next, history(next)));
        CHECK(engine->snapshot().tree_capacity == capacity);
        CHECK(engine->snapshot().tree_nodes == 1);
        CHECK(engine->snapshot().nodes == 0);
        CHECK(engine->snapshot().inherited_visits == 0);
        CHECK(engine->snapshot().proof.outcome == mouse::search::ProofOutcome::Unknown);
        CHECK(engine->root_moves().size == 0);
        const auto actual_row = backend.rows;
        const auto result = run(*engine, 257);
        CHECK(backend.trace.at(actual_row) == reference.trace.at(expected_row));
        CHECK(result.info.nodes == 257);
        CHECK(result.info.inherited_visits == 0);
        CHECK(result.info.charged_bytes <= result.info.budget_bytes);
        REQUIRE(result.best_move);
        auto probe = next;
        const auto legal = mouse::chess::legal_moves(probe, attacks);
        CHECK(std::ranges::find(legal, required_optional(result.best_move)) != legal.end());
        const auto before_rejection = engine->snapshot();
        const auto wrong_history = history(repetition % 2 == 0 ? first : second);
        CHECK_FALSE(engine->set_position(next, wrong_history));
        CHECK(engine->snapshot().nodes == before_rejection.nodes);
        CHECK(engine->snapshot().q == before_rejection.q);
        CHECK(engine->snapshot().proof == before_rejection.proof);
    }
}

TEST_CASE("Root history growth shrinks the arena and excessive history clears recoverably",
          "[search][root-reset]") {
    const mouse::chess::AttackTables attacks;
    auto root = position(attacks);
    auto frames = history(root);
    auto selected = config();
    SECTION("Growth beyond the original arena headroom") {}
    SECTION("History exceeds the complete managed budget") {
        selected.ram_limit_bytes = 512U << 10U;
    }
    TestBackend backend;
    auto engine = create(selected, attacks, backend, root, frames);
    REQUIRE(run(*engine, 31).best_move);
    const auto before = engine->snapshot();
    for (int cycle = 0; cycle < 1024; ++cycle) {
        for (const auto* token : {"g1f3", "g8f6", "f3g1", "f6g8"}) {
            const auto move = mouse::chess::parse_uci_move(root, token, attacks);
            REQUIRE(move);
            static_cast<void>(root.make_move(*move, attacks));
            REQUIRE(frames.append(root));
        }
    }
    auto changed = engine->set_position(root, frames);
    if (selected.ram_limit_bytes == (512U << 10U)) {
        REQUIRE_FALSE(changed);
        CHECK(changed.error().find("cleared old root") != std::string::npos);
        CHECK(engine->snapshot().nodes == 0);
        CHECK_FALSE(engine->run({}));
        const auto fresh = position(attacks);
        REQUIRE(engine->set_position(fresh, history(fresh)));
        CHECK(run(*engine, 1).best_move);
    } else {
        REQUIRE(changed);
        CHECK(engine->snapshot().tree_capacity < before.tree_capacity);
        CHECK(engine->snapshot().charged_bytes <= selected.ram_limit_bytes);
        CHECK(engine->snapshot().nodes == 0);
        CHECK(run(*engine, 1).best_move);
    }
}

TEST_CASE("A new-game root discards evaluated visits while retaining arena capacity",
          "[search][root-reset][newgame-v1]") {
    const mouse::chess::AttackTables attacks;
    const auto root = position(attacks);
    const auto frames = history(root);
    TestBackend backend;
    auto engine = create(config(), attacks, backend, root, frames);
    const auto previous = run(*engine, 128);
    REQUIRE(previous.info.nodes == 128);
    const auto capacity = previous.info.tree_capacity;
    REQUIRE(engine->set_position(root, frames, false));
    CHECK(engine->snapshot().nodes == 0);
    CHECK(engine->root_moves().size == 0);
    CHECK(engine->snapshot().tree_capacity == capacity);
    const auto fresh = run(*engine, 1);
    CHECK(fresh.info.nodes == 1);
    CHECK(fresh.info.inherited_visits == 0);
    CHECK(fresh.info.tree_capacity == capacity);
}

TEST_CASE("Pending preferred edges participate in PUCT instead of forcing zero-prior alternatives",
          "[search][collision-parity-v1]") {
    const mouse::chess::AttackTables attacks;
    const auto root = position(attacks);
    TestBackend backend;
    backend.dominant = true;
    backend.white_advantage = true;
    auto settings = config();
    settings.maximum_batch = 64;
    settings.threads = 4;
    settings.deterministic = false;
    auto engine = create(settings, attacks, backend, root, history(root));
    const auto result = run(*engine, 128);
    REQUIRE(result.best_move);
    const mouse::chess::Move preferred{mouse::chess::square(1, 0), mouse::chess::square(2, 2)};
    REQUIRE(required_optional(result.best_move) == preferred);
    const auto roots = engine->root_moves();
    for (std::size_t i = 0; i < roots.size; ++i) {
        const auto& edge = roots.moves.at(i);
        if (edge.move != preferred) {
            REQUIRE(edge.prior == 0.0F);
            CHECK(edge.visits == 0);
        }
    }
}

TEST_CASE("Zero smart pruning factor honors the full timed search", "[search][legacy-v1]") {
    const mouse::chess::AttackTables attacks;
    const auto root = position(attacks);
    TestBackend backend;
    backend.dominant = true;
    backend.delay = std::chrono::milliseconds{1};
    auto engine = create(config(), attacks, backend, root, history(root));
    mouse::search::SearchLimits limits;
    limits.nodes.reset();
    limits.movetime = std::chrono::milliseconds{300};
    limits.smart_pruning_factor = 0;
    const auto result = engine->run(limits);
    REQUIRE(result);
    CHECK(result->reason == mouse::search::StopReason::TimeLimit);
    CHECK(result->info.elapsed >= std::chrono::milliseconds{300});
}

TEST_CASE("Collision budgets keep gathering useful leaves into full batches",
          "[search][collision-budget-v1]") {
    const mouse::chess::AttackTables attacks;
    const auto root = position(attacks);
    TestBackend backend;
    backend.concentrated = true;
    auto settings = config();
    settings.maximum_batch = 64;
    settings.ram_limit_bytes = 8U << 20U;
    auto engine = create(settings, attacks, backend, root, history(root));
    mouse::search::SearchLimits limits;
    limits.nodes = 4096;
    limits.smart_pruning = false;
    const auto result = engine->run(limits);
    REQUIRE(result);
    CHECK(result->reason == mouse::search::StopReason::NodeLimit);
    CHECK(result->info.nodes == 4096);
    const auto batches = std::span{backend.batches}.first(backend.batch_count);
    REQUIRE_FALSE(batches.empty());
    const auto largest = std::ranges::max(batches);
    const auto full = std::ranges::count(batches, std::size_t{64});
    INFO("provider calls=" << backend.calls << " rows=" << backend.rows << " largest=" << largest
                           << " full=" << full);
    // A concentrated policy makes a narrow tree whose in-flight leaves absorb
    // the collision budget (lc0 classic/search.cc GatherMinibatch), so only
    // reaching full batches is required here; the wide-tree case below checks
    // that steady-state batches fill.
    CHECK(largest == 64);
    CHECK(full > 0);
    // Virtual collisions cannot survive into reset or inflate completed visits.
    REQUIRE(engine->set_position(root, history(root), false));
    CHECK(engine->snapshot().nodes == 0);
    const auto again = engine->run(limits);
    REQUIRE(again);
    CHECK(again->info.nodes == 4096);
    CHECK(again->best_move == result->best_move);
}

TEST_CASE("Single-walk gather fills steady-state batches on a wide tree",
          "[search][collision-budget-v1]") {
    const mouse::chess::AttackTables attacks;
    const auto root = position(attacks);
    TestBackend backend;
    auto settings = config();
    settings.maximum_batch = 64;
    settings.ram_limit_bytes = 8U << 20U;
    auto engine = create(settings, attacks, backend, root, history(root));
    mouse::search::SearchLimits limits;
    limits.nodes = 4096;
    limits.smart_pruning = false;
    const auto result = engine->run(limits);
    REQUIRE(result);
    CHECK(result->info.nodes == 4096);
    // lc0 classic/search.cc CalculateCollisionsLeft allows one collision visit
    // until the tree has 28 visits, so early batches are small by design. Once
    // 1024 rows are evaluated the collision budget exceeds two batches.
    const auto batches = std::span{backend.batches}.first(backend.batch_count);
    std::size_t evaluated = 0, steady_calls = 0, steady_full = 0;
    for (const auto size : batches) {
        if (evaluated >= 1024) {
            ++steady_calls;
            if (size == 64) ++steady_full;
        }
        evaluated += size;
    }
    INFO("provider calls=" << backend.calls << " steady calls=" << steady_calls
                           << " steady full=" << steady_full);
    REQUIRE(steady_calls > 0);
    CHECK(steady_full * 10 >= steady_calls * 9);
}

TEST_CASE("Collision reservations drain after a provider exception before retained-root reuse",
          "[search][collision-budget-v1]") {
    const mouse::chess::AttackTables attacks;
    const auto root = position(attacks);
    TestBackend backend;
    backend.concentrated = true;
    backend.nonstd_call = 50;
    auto settings = config();
    settings.maximum_batch = 64;
    settings.ram_limit_bytes = 8U << 20U;
    auto engine = create(settings, attacks, backend, root, history(root));
    mouse::search::SearchLimits limits;
    limits.nodes = 4096;
    limits.smart_pruning = false;
    const auto failed = engine->run(limits);
    REQUIRE_FALSE(failed);
    CHECK(failed.error().find("non-standard exception") != std::string::npos);
    const auto completed = engine->snapshot().nodes;
    CHECK(completed > 28);
    CHECK(completed < 4096);
    backend.nonstd_call = 0;
    REQUIRE(engine->set_position(root, history(root), true));
    CHECK(engine->snapshot().nodes == completed);
    const auto resumed = engine->run(limits);
    REQUIRE(resumed);
    CHECK(resumed->reason == mouse::search::StopReason::NodeLimit);
    CHECK(resumed->info.nodes == 4096);
    // Tree::reset rejects any surviving real or virtual reservation.
    REQUIRE(engine->set_position(root, history(root), false));
    CHECK(engine->snapshot().nodes == 0);
}
