// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/uci/session.h"

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "engine/chess/attacks.h"
#include "engine/chess/movegen.h"
#include "engine/nn/command_runner.h"
#include "engine/nn/encoder.h"
#include "engine/nn/legal_cache.h"
#include "engine/nn/policy_map.h"
#include "engine/uci/emergency.h"
#include "engine/uci/report.h"
#include "engine/uci/score.h"
#include "engine/uci/time.h"
#include "engine/util/build_info.h"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#include <timeapi.h>
#endif
namespace mouse::uci {
namespace {
using Clock = std::chrono::steady_clock;
// Minimum spacing of search-progress view rebuilds (see search_turn).
constexpr auto kProgressViewInterval = std::chrono::milliseconds{5};
#ifdef _WIN32
// Windows condition-variable timeouts otherwise inherit a coarse process timer.
// Balance every successful request after the session's last owned thread joins.
// https://learn.microsoft.com/en-us/windows/win32/api/timeapi/nf-timeapi-timebeginperiod
class MillisecondWaitResolution {
   public:
    MillisecondWaitResolution() {
        // Windows 11 ignores timer requests from processes without a visible
        // window (how GUIs and match runners start engines) unless the process
        // opts out of timer-resolution power throttling. Windows 10 has no such
        // throttling and rejects the flag, so a failure here changes nothing.
        // https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setprocessinformation
        PROCESS_POWER_THROTTLING_STATE state{};
        state.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
        state.ControlMask = PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION;
        state.StateMask = 0;
        static_cast<void>(SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &state,
                                                sizeof(state)));
        if (timeBeginPeriod(1) != TIMERR_NOERROR)
            throw std::runtime_error("Windows cannot provide the required 1ms UCI wait resolution");
    }
    ~MillisecondWaitResolution() { static_cast<void>(timeEndPeriod(1)); }
    MillisecondWaitResolution(const MillisecondWaitResolution&) = delete;
    MillisecondWaitResolution& operator=(const MillisecondWaitResolution&) = delete;
    MillisecondWaitResolution(MillisecondWaitResolution&&) = delete;
    MillisecondWaitResolution& operator=(MillisecondWaitResolution&&) = delete;
};
#endif
// Install below the result cache so cache hits cannot masquerade as fresh
// provider latency evidence. The session owns the counter longer than this
// wrapper and joins all search calls before replacement/destruction. Atomic
// counts support the backend's serialized-call contract across search workers;
// identity/metadata remain immutable and errors/outputs pass through unchanged.
class ObservedBackend final : public nn::Backend {
   public:
    ObservedBackend(std::unique_ptr<nn::Backend> backend, std::atomic<std::uint64_t>& calls,
                    BucketLatencies& latencies,
                    std::function<void(std::string_view, std::size_t, std::size_t)> trace)
        : backend_(std::move(backend)),
          calls_(calls),
          latencies_(latencies),
          trace_(std::move(trace)) {}
    Result<void> evaluate(std::span<const nn::InputPlanes> inputs,
                          std::span<nn::NNOutput> outputs) override {
        const auto physical = backend_->physical_batch_size(inputs.size());
        trace_("batch_begin", inputs.size(), physical);
        const auto began = std::chrono::steady_clock::now();
        try {
            auto evaluated = backend_->evaluate(inputs, outputs);
            trace_(evaluated ? "batch_end" : "batch_error", inputs.size(), physical);
            if (evaluated) {
                latencies_.observe(physical, std::chrono::steady_clock::now() - began);
                calls_.fetch_add(1, std::memory_order_relaxed);
            }
            return evaluated;
        } catch (...) {
            trace_("batch_error", inputs.size(), physical);
            throw;
        }
    }
    [[nodiscard]] std::size_t physical_batch_size(std::size_t logical) const noexcept override {
        return backend_->physical_batch_size(logical);
    }
    [[nodiscard]] const nn::ModelMetadata& metadata() const noexcept override {
        return backend_->metadata();
    }
    [[nodiscard]] const nn::BackendInfo& info() const noexcept override { return backend_->info(); }

   private:
    std::unique_ptr<nn::Backend> backend_;
    std::atomic<std::uint64_t>& calls_;
    BucketLatencies& latencies_;
    std::function<void(std::string_view, std::size_t, std::size_t)> trace_;
};
std::string diagnostic(std::string_view value) {
    constexpr std::size_t kLimit = 4096;
    std::string result;
    result.reserve(std::min(value.size(), kLimit));
    for (const char character : value.substr(0, kLimit)) {
        const auto c = static_cast<unsigned char>(character);
        result.push_back(c < 32 || c == 127 ? ' ' : character);
    }
    if (value.size() > kLimit) result += " [truncated]";
    return result;
}
}  // namespace
struct Session::Impl {
    struct State {
        Options options;
        chess::Position position;
        nn::EncodingHistory history;
        std::string base_fen;
        std::vector<chess::Move> moves;
        std::uint64_t backend_revision, search_revision, game_generation;
    };
    struct View {
        search::SearchResult result;
        search::RootStatistics roots;
        std::vector<search::PrincipalVariation> variations;
        // Lc0 EnsureBestMoveKnown: the move drawn by temperature, when one is in force.
        std::optional<chess::Move> temperature_move;
        std::uint64_t inherited = 0;
        search::SnapshotCost snapshot_cost;
        // A policy-head or value-head move: its info line has already gone out.
        bool instant = false;
    };
    struct Turn {
        State state;
        GoCommand command;
        search::SearchLimits limits;
        TimeBudget time;
        std::optional<TimeBudget> ponder_budget;
        Clock::time_point started;
        std::optional<Clock::time_point> deadline, hard_deadline;
        std::optional<chess::Move> emergency_move, reused_move;
        std::stop_source cancel;
        // The worker alone borrows the tree and builds immutable delivery views.
        // Publisher loads retain their lifetime independently of worker drain;
        // no publisher path acquires the live tree or a worker-owned mutex.
        search::SearchEngine* live = nullptr;
        std::atomic<std::shared_ptr<const View>> published{std::make_shared<const View>()};
        std::uint64_t inherited = 0, game_ply = 0;
        std::atomic<bool> done{false}, primed{false}, stop_requested{false};
        std::atomic<bool> hold_completed{false};
        Turn(State configured, GoCommand request, search::SearchLimits bounded,
             TimeBudget allocated, Clock::time_point accepted)
            : state(std::move(configured)),
              command(std::move(request)),
              limits(std::move(bounded)),
              time(allocated),
              started(accepted) {
            if (time.budget) {
                const auto available = std::chrono::duration_cast<std::chrono::milliseconds>(
                    Clock::time_point::max() - started);
                deadline =
                    *time.budget >= available ? Clock::time_point::max() : started + *time.budget;
                auto hard_ms = state.position.side_to_move() == chess::Color::White ? command.wtime
                                                                                    : command.btime;
                if (command.movetime)
                    hard_ms = hard_ms ? std::min(*hard_ms, *command.movetime) : command.movetime;
                if (hard_ms)
                    hard_deadline = *hard_ms >= static_cast<std::uint64_t>(available.count())
                                        ? Clock::time_point::max()
                                        : started + std::chrono::milliseconds{
                                                        static_cast<std::int64_t>(*hard_ms)};
            }
        }
    };
    // Only the input owner mutates logical configuration and position. Every
    // accepted go takes a complete immutable copy, independent of prior drain.
#ifdef _WIN32
    MillisecondWaitResolution wait_resolution;
#endif
    Options options;
    Output output;
    BackendFactory factory;
    const chess::AttackTables attacks;
    std::optional<chess::Position> position;
    nn::EncodingHistory history;
    std::string base_fen;
    std::vector<chess::Move> game_moves;
    std::uint64_t lc0_game_ply = 1;
    LegacyTimeManager time_manager;
    std::uint64_t backend_revision = 0, search_revision = 0, game_generation = 0;
    // Backend/tree/latency have one persistent worker owner. Readiness waits
    // for its configuration acknowledgement; input never borrows these objects.
    // A previous call can drain while publication serves the next clock.
    std::atomic<std::uint64_t> physical_calls{0};
    std::unique_ptr<nn::Backend> backend;
    std::unique_ptr<nn::LegalCache> neural_cache;
    std::unique_ptr<search::SearchEngine> engine;
    // Minibatch resolved for the loaded backend when MinibatchSize is 0 (auto);
    // 0 while no backend is loaded or the batch is explicit.
    std::size_t resolved_minibatch = 0;
    std::vector<std::size_t> resolved_buckets;
    std::uint64_t prepared_backend_revision = 0, prepared_search_revision = 0,
                  prepared_game_generation = 0;
    std::string engine_base_fen;
    std::vector<chess::Move> engine_moves;
    // Only successful physical root paths age these worker-owned samples.
    // Cache/reused/terminal roots cannot erase an observed inference stall;
    // retain the window across new games, clear only on backend reconfiguration.
    LatencyWindow root_latency;
    BucketLatencies bucket_latency;
    std::ofstream log_stream;  // output_mutex protects it.
    // The current go's root moves came from the tables (one tbhit, as in lc0).
    bool root_tablebase_hit = false;
    // Only the joined publisher reads/writes delivery timing, never inference.
    LatencyWindow publication_latency;
    std::mutex output_mutex, wait_mutex;
    std::condition_variable_any changed;
    // wait_mutex protects the single pending slot, working/current ownership
    // and shutdown. Expired pending turns are replaced, never accumulated.
    std::shared_ptr<Turn> current, pending, working;
    std::jthread job, worker;
    std::stop_source input_cancel;
    std::atomic<bool> active{false}, suppress_best{false}, publishing{false}, output_failed{false};
    std::atomic<int> status{0};
    std::atomic<bool> debug_enabled{false};
    bool closed = false;
    // Configuration and readiness requests are coalesced; only the worker may
    // construct, retire, or reset a backend/tree. Never hold wait_mutex in a
    // provider call, allocation, or destructor.
    std::optional<State> preparation;
    bool preparing = false;
    std::uint64_t requested_backend_revision = 0;
    std::optional<std::string> preparation_error;
    std::optional<std::uint64_t> loaded_backend_revision;
    std::optional<GoCommand> current_go;
    Impl(Options configured, Output writer, BackendFactory make)
        : options(std::move(configured)), output(std::move(writer)), factory(std::move(make)) {
        if (!output || !factory)
            throw std::invalid_argument("UCI session requires output and backend factory");
        auto start = chess::Position::from_fen(chess::kStartFen, attacks);
        if (!start) throw std::logic_error("Invalid built-in start position: " + start.error());
        position = std::move(*start);
        history.reset(*position);
        base_fen = position->fen();
        if (!options.backend.model_path.empty()) preparation = capture();
        worker = std::jthread([this](const std::stop_token& stop) { worker_loop(stop); });
    }
    void cancel_requests() noexcept {
        const std::lock_guard lock(wait_mutex);
        if (current) current->cancel.request_stop();
        if (pending) pending->cancel.request_stop();
        if (working) working->cancel.request_stop();
    }
    // Lc0 LogFile: every UCI line in and out, stamped. A log that cannot be opened or written is
    // reported once and then closed; it never disturbs the game. Callers hold output_mutex.
    void log_line(std::string_view direction, std::string_view line) {
        if (log_stream.is_open()) {
            const auto now = std::chrono::system_clock::now();
            const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(
                                    now.time_since_epoch())
                                    .count();
            log_stream << micros << ' ' << direction << ' ' << line << '\n' << std::flush;
            if (!log_stream) log_stream.close();
        }
    }
    void open_log(const std::string& path) {
        const std::lock_guard lock(output_mutex);
        if (log_stream.is_open()) log_stream.close();
        log_stream.clear();
        if (path.empty()) return;
        log_stream.open(std::filesystem::path(std::u8string(path.begin(), path.end())),
                        std::ios::app);
        if (!log_stream.is_open()) output("info string error Cannot open LogFile: " + path);
    }
    bool emit(std::string_view line, bool require_healthy = false) noexcept {
        if (output_failed.load(std::memory_order_acquire)) return false;
        try {
            const std::lock_guard lock(output_mutex);
            if (output_failed.load(std::memory_order_acquire) ||
                (require_healthy && status.load(std::memory_order_acquire) != 0))
                return false;
            output(line);
            log_line("<<", line);
            return true;
        } catch (...) {
            output_failed.store(true, std::memory_order_release);
            status.store(1, std::memory_order_release);
            input_cancel.request_stop();
            cancel_requests();
            changed.notify_all();
            return false;
        }
    }
    void error(std::string_view message) {
        static_cast<void>(emit("info string error " + diagnostic(message)));
    }
    void fatal(std::string_view message) noexcept {
        status.store(1, std::memory_order_release);
        input_cancel.request_stop();
        cancel_requests();
        changed.notify_all();
        try {
            error(message);
        } catch (...) {
            std::fputs("Mouse error: UCI fatal diagnostic could not be emitted.\n", stderr);
        }
    }
    void finish(bool publish) {
        if (!job.joinable()) return;
        suppress_best.store(!publish, std::memory_order_release);
        {
            const std::lock_guard lock(wait_mutex);
            if (current) {
                current->stop_requested.store(true, std::memory_order_release);
                // Preserve the mandatory evaluated root for an explicit stop.
                // An absolute deadline remains independently enforceable even
                // while that initial physical call is blocked.
                if (!publish || current->primed.load(std::memory_order_acquire))
                    current->cancel.request_stop();
            }
        }
        changed.notify_all();
        job.join();
        active.store(false, std::memory_order_release);
    }
    void join_finished() {
        if (job.joinable() &&
            (!active.load(std::memory_order_acquire) || publishing.load(std::memory_order_acquire)))
            job.join();
    }
    void wait_idle() {
        std::unique_lock lock(wait_mutex);
        changed.wait(lock, [&] { return !working && !pending && !preparing && !preparation; });
    }
    void trace(std::string_view event, std::size_t batch = 0, std::size_t physical = 0,
               const search::SnapshotCost* cost = nullptr,
               const search::SnapshotCost* producer_cost = nullptr) noexcept {
        if (!debug_enabled.load(std::memory_order_relaxed)) return;
        try {
            // Host acceptance/completion envelopes on a single steady clock.
            // They enclose synchronous provider work, not device kernel timings.
            const auto stamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   Clock::now().time_since_epoch())
                                   .count();
            std::ostringstream line;
            line << "info string trace " << event << " steady_ns " << stamp;
            if (batch != 0) line << " batch " << batch << " physical_batch " << physical;
            if (cost)
                line << " tree_lock_count " << cost->tree_lock_count << " tree_lock_wait_ns "
                     << cost->tree_lock_wait.count();
            if (producer_cost)
                line << " producer_tree_lock_count " << producer_cost->tree_lock_count
                     << " producer_tree_lock_wait_ns " << producer_cost->tree_lock_wait.count();
            static_cast<void>(emit(line.str()));
        } catch (...) {
            fatal("Cannot render UCI debug timestamp");
        }
    }
    [[nodiscard]] State capture() const {
        if (!position.has_value()) throw std::logic_error("UCI session has no root position");
        return {.options = options,
                .position = *position,
                .history = history,
                .base_fen = base_fen,
                .moves = game_moves,
                .backend_revision = backend_revision,
                .search_revision = search_revision,
                .game_generation = game_generation};
    }
    // Search batch and buckets in force for the loaded backend.
    [[nodiscard]] std::size_t minibatch(const Options& selected) const {
        return selected.auto_minibatch && resolved_minibatch != 0 ? resolved_minibatch
                                                                  : selected.search.maximum_batch;
    }
    [[nodiscard]] const std::vector<std::size_t>& buckets(const Options& selected) const {
        return selected.auto_minibatch && resolved_minibatch != 0 ? resolved_buckets
                                                                  : selected.backend.batch_buckets;
    }
    Result<void> prepare(const State& state, const std::stop_token& cancel = {}) {
        if (cancel.stop_requested()) return {};
        {
            const std::lock_guard lock(wait_mutex);
            if (preparation_error && state.backend_revision == requested_backend_revision)
                return tl::unexpected(*preparation_error);
        }
        if (state.options.backend.model_path.empty())
            return tl::unexpected(
                std::string{"WeightsFile is required before neural readiness or search"});
        if (prepared_backend_revision != state.backend_revision) {
            engine.reset();
            neural_cache.reset();
            backend.reset();
            resolved_minibatch = 0;
            resolved_buckets.clear();
            physical_calls.store(0, std::memory_order_relaxed);
            root_latency.clear();
            bucket_latency.clear();
        } else if (prepared_search_revision != state.search_revision) {
            engine.reset();
        }
        if (!backend) {
            auto config = state.options.backend;
            config.initialization_progress = [this](std::string_view text) {
                static_cast<void>(emit("info string " + diagnostic(text)));
            };
            auto paths = nn::resolve_backend_cli_paths(config);
            if (!paths) return tl::unexpected(paths.error());
            if (state.options.auto_minibatch) {
                auto batch = nn::recommended_minibatch(config);
                if (!batch) return tl::unexpected(batch.error());
                resolved_buckets = minibatch_buckets(*batch);
                config.batch_buckets = resolved_buckets;
                static_cast<void>(
                    emit("info string MinibatchSize auto: " + std::to_string(*batch)));
            }
            auto made = factory(config);
            if (!made) return tl::unexpected(made.error());
            if (!*made) return tl::unexpected(std::string{"Backend factory returned no backend"});
            auto observed = std::make_unique<ObservedBackend>(
                std::move(*made), physical_calls, bucket_latency,
                [this](std::string_view event, std::size_t batch, std::size_t physical) {
                    trace(event, batch, physical);
                });
            backend = std::move(observed);
            if (state.options.auto_minibatch) resolved_minibatch = resolved_buckets.back();
            // Backend and search readiness are separate. A newer search-only
            // configuration must keep this model rather than start another load.
            prepared_backend_revision = state.backend_revision;
        }
        if (cancel.stop_requested()) return {};
        // Background preparation has no turn token; accepted turns always do.
        if (!cancel.stop_possible()) {
            const std::lock_guard lock(wait_mutex);
            // Every queued snapshot supersedes this root/search preparation,
            // including positions and new games with unchanged option revisions.
            if (preparation) return {};
        }
        if (neural_cache &&
            (neural_cache->stats().capacity != state.options.nncache_entries ||
             neural_cache->history_length() != state.options.cache_history_length)) {
            engine.reset();
            neural_cache.reset();
        }
        if (!neural_cache && state.options.nncache_entries != 0) {
            auto cache = nn::LegalCache::create(state.options.nncache_entries,
                                                backend->info().cache_namespace,
                                                state.options.cache_history_length);
            if (!cache) return tl::unexpected(cache.error());
            neural_cache = std::move(*cache);
        }
        if (!engine) {
            auto search_config = state.options.search;
            search_config.maximum_batch = minibatch(state.options);
            search_config.dag = state.options.algorithm == SearchAlgorithm::DagPreview;
            search_config.cache_history_length =
                static_cast<std::uint32_t>(state.options.cache_history_length);
            if (state.options.task_workers >= 0 && !search_config.deterministic)
                search_config.threads = static_cast<std::size_t>(state.options.task_workers);
            auto made =
                search::SearchEngine::create(search_config, attacks, *backend, neural_cache.get());
            if (!made) return tl::unexpected(made.error());
            engine = std::move(*made);
            engine_base_fen.clear();
            engine_moves.clear();
        }
        const bool prefix =
            prepared_game_generation == state.game_generation &&
            engine_base_fen == state.base_fen && state.moves.size() >= engine_moves.size() &&
            std::equal(engine_moves.begin(), engine_moves.end(), state.moves.begin());
        if (prefix) {
            for (std::size_t i = engine_moves.size(); i < state.moves.size(); ++i) {
                const auto advanced = engine->advance(state.moves[i]);
                if (!advanced) {
                    engine.reset();
                    return tl::unexpected("Search reuse cleared after position error: " +
                                          advanced.error());
                }
            }
        }
        // set_position also resets a terminal leaf that became the root. The equal-
        // root path retains completed visits; unrelated roots reset their arena.
        auto configured = engine->set_position(state.position, state.history,
                                               prepared_game_generation == state.game_generation);
        if (!configured) {
            engine.reset();
            return tl::unexpected(configured.error());
        }
        engine_base_fen = state.base_fen;
        engine_moves = state.moves;
        prepared_backend_revision = state.backend_revision;
        prepared_search_revision = state.search_revision;
        prepared_game_generation = state.game_generation;
        return {};
    }
    void request_preparation() {
        if (options.backend.model_path.empty()) return;
        auto next = capture();
        std::optional<State> retired;
        {
            const std::lock_guard lock(wait_mutex);
            if (requested_backend_revision != backend_revision) preparation_error.reset();
            requested_backend_revision = backend_revision;
            retired.swap(preparation);
            preparation.emplace(std::move(next));
        }
        changed.notify_all();
    }
    Result<void> await_backend() {
        std::unique_lock lock(wait_mutex);
        changed.wait(lock, [&] {
            return status.load(std::memory_order_acquire) != 0 || preparation_error ||
                   (loaded_backend_revision && *loaded_backend_revision == backend_revision);
        });
        if (preparation_error) return tl::unexpected(*preparation_error);
        if (status.load(std::memory_order_acquire) != 0)
            return tl::unexpected(std::string{"Backend readiness failed"});
        return {};
    }
    Result<void> ready() {
        if (options.backend.model_path.empty())
            return tl::unexpected(
                std::string{"WeightsFile is required before neural readiness or search"});
        request_preparation();
        wait_idle();
        if (status.load(std::memory_order_acquire) != 0)
            return tl::unexpected(std::string{"Backend readiness failed"});
        const std::lock_guard lock(wait_mutex);
        if (preparation_error) return tl::unexpected(*preparation_error);
        return {};
    }
    Result<void> configure_position(const PositionCommand& command) {
        auto next = chess::Position::from_fen(command.fen, attacks);
        if (!next) return tl::unexpected("Invalid FEN: " + next.error());
        auto variant = chess::validate_uci_castling_mode(*next, options.chess960);
        if (!variant) return tl::unexpected(variant.error());
        const auto initial = next->fen(options.chess960);
        const auto initial_ply = next->fullmove_number();
        nn::EncodingHistory next_history;
        next_history.reset(*next);
        auto reserved = next->reserve_history(command.moves.size());
        if (!reserved) return tl::unexpected(reserved.error());
        reserved = next_history.reserve_history(command.moves.size());
        if (!reserved) return tl::unexpected(reserved.error());
        std::vector<chess::Move> moves;
        moves.reserve(command.moves.size());
        for (const auto& text : command.moves) {
            auto parsed = chess::parse_uci_move(*next, text, attacks, options.chess960);
            if (!parsed)
                return tl::unexpected("Illegal position move " + text + ": " + parsed.error());
            moves.push_back(*parsed);
            static_cast<void>(next->make_move(*parsed, attacks));
            auto appended = next_history.append(*next);
            if (!appended) return tl::unexpected(appended.error());
        }
        if (initial != base_fen || moves.size() < game_moves.size() ||
            !std::equal(game_moves.begin(), game_moves.end(), moves.begin()))
            time_manager.reset();
        // Pinned lc0 v0.32.1 Position::FromFen passes the FEN fullmove counter
        // directly to ply_count_ (chess/position.cc:53-56; board.cc:SetFromFen).
        // Each supplied UCI move then increments it once. Preserve this actual
        // reference behavior instead of deriving standard 2*(fullmove-1)+side.
        lc0_game_ply = initial_ply > std::numeric_limits<std::uint64_t>::max() - moves.size()
                           ? std::numeric_limits<std::uint64_t>::max()
                           : initial_ply + moves.size();
        position = std::move(*next);
        history = std::move(next_history);
        base_fen = initial;
        game_moves = std::move(moves);
        // Capacity preparation and retirement happen on the persistent owner,
        // including ucinewgame. Input never drains a previous provider call.
        request_preparation();
        return {};
    }
    // Lc0 SyzygyPath: an empty path closes the tables. Tables that cannot be opened are reported,
    // and the engine plays on without them, as it would with none configured.
    void load_tablebase() {
        options.search.tablebase.reset();
        if (options.syzygy_path.empty()) return;
        auto opened = search::Tablebase::open(options.syzygy_path);
        if (!opened) {
            error(opened.error());
            return;
        }
        options.search.tablebase = std::move(*opened);
        static_cast<void>(emit("info string Syzygy tablebases loaded, up to " +
                               std::to_string(options.search.tablebase->max_cardinality()) +
                               " pieces"));
    }
    Result<search::SearchLimits> limits(const GoCommand& command) {
        if (!position.has_value())
            return tl::unexpected(std::string{"UCI session has no position"});
        root_tablebase_hit = false;
        search::SearchLimits result;
        result.nodes = command.nodes;
        result.depth = command.depth;
        result.smart_pruning_factor = options.smart_pruning_factor;
        result.minimum_smart_pruning_batches = options.minimum_smart_pruning_batches;
        result.log_search_stats = options.log_search_stats;
        // lc0 MakeRootMoveFilter: searchmoves override the tables; otherwise the search only
        // looks at the moves they call best.
        if (options.search.tablebase && command.searchmoves.empty()) {
            if (const auto filter = options.search.tablebase->root_filter(
                    *position, attacks, position->repeated_since_zeroing(),
                    options.syzygy_fast_play)) {
                result.root_in_dtz = filter->used_dtz;
                root_tablebase_hit = true;
                for (const auto move : filter->moves)
                    result.searchmoves.at(result.searchmoves_count++) = move;
            }
        }
        // lc0 fd71a2d classic/search.cc Search::Search: "play" becomes the side to move (the
        // other one when pondering), and no side at all for an infinite search.
        switch (options.contempt_mode) {
            case ContemptMode::Play:
                if (command.infinite) {
                    result.contempt_side = search::ContemptSide::None;
                    if (search::wdl_rescale_parameters(options.search.draw_rate).diff != 0.0F)
                        static_cast<void>(emit(
                            "info string WARNING: Contempt mode set to 'disable' as 'play' not "
                            "supported for infinite search."));
                } else
                    result.contempt_side =
                        (position->side_to_move() == chess::Color::Black) != command.ponder
                            ? search::ContemptSide::Black
                            : search::ContemptSide::White;
                break;
            case ContemptMode::WhiteSideAnalysis:
                result.contempt_side = search::ContemptSide::White;
                break;
            case ContemptMode::BlackSideAnalysis:
                result.contempt_side = search::ContemptSide::Black;
                break;
            case ContemptMode::Disable:
                result.contempt_side = search::ContemptSide::None;
                break;
        }
        if (command.mate) {
            const auto plies = *command.mate * 2 - 1;
            result.depth = std::min(result.depth.value_or(plies), plies);
        }
        for (const auto& text : command.searchmoves) {
            const auto move = chess::parse_uci_move(*position, text, attacks, options.chess960);
            if (!move)
                return tl::unexpected("Illegal searchmoves move " + text + ": " + move.error());
            result.searchmoves.at(result.searchmoves_count++) = *move;
        }
        return result;
    }
    [[nodiscard]] View collect(const Turn& turn) const {
        View view;
        auto* cost = debug_enabled.load(std::memory_order_relaxed) ? &view.snapshot_cost : nullptr;
        auto& searcher = *turn.live;
        // result_snapshot captures its best move and completed PV under one
        // tree lock. Later root statistics must not replace that decision.
        view.result = searcher.result_snapshot(cost);
        view.roots = searcher.root_moves(cost);
        if (turn.state.options.search.temperature.temperature != 0.0 ||
            turn.state.options.search.temperature.endgame != 0.0)
            view.temperature_move = searcher.final_move(turn.game_ply);
        // Preserve the immutable turn's root restriction in every delivered view.
        if (turn.limits.searchmoves_count != 0) {
            std::size_t kept = 0;
            for (std::size_t i = 0; i < view.roots.size; ++i) {
                const auto move = view.roots.moves.at(i).move;
                const auto end = turn.limits.searchmoves.begin() +
                                 static_cast<std::ptrdiff_t>(turn.limits.searchmoves_count);
                if (std::find(turn.limits.searchmoves.begin(), end, move) != end)
                    view.roots.moves.at(kept++) = view.roots.moves.at(i);
            }
            view.roots.size = kept;
        }
        rank_report_roots(view.roots, view.result.best_move);
        const auto count = std::min(turn.state.options.multipv, view.roots.size);
        view.variations.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            const auto move = view.roots.moves.at(i).move;
            view.variations.push_back(i == 0 && view.result.best_move == move
                                          ? view.result.info.pv
                                          : searcher.principal_variation(move, cost));
        }
        view.inherited = turn.inherited;
        return view;
    }
    void publish_view(Turn& turn) const {
        turn.published.store(std::make_shared<const View>(collect(turn)),
                             std::memory_order_release);
    }
    [[nodiscard]] static std::shared_ptr<const View> view_of(const Turn& turn) {
        return turn.published.load(std::memory_order_acquire);
    }
    void report(const Turn& turn, const View& view) {
        if (view.instant) return;
        const auto& info = view.result.info;
        // An unevaluated root has no NN score, WDL, priors or PV to report.
        if (info.nodes == 0 && turn.emergency_move.has_value()) return;
        const auto elapsed = std::max(
            std::chrono::milliseconds{0},
            std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - turn.started));
        const auto nps = elapsed.count() == 0
                             ? 0.0
                             : static_cast<double>(
                                   info.nodes >= view.inherited ? info.nodes - view.inherited : 0) *
                                   1000.0 / static_cast<double>(elapsed.count());
        const auto hashfull = info.tree_arena_bytes == 0
                                  ? 0
                                  : static_cast<unsigned>(std::min(
                                        1000.0, static_cast<double>(info.tree_used_bytes) * 1000.0 /
                                                    static_cast<double>(info.tree_arena_bytes)));
        const auto count = view.variations.size();
        const auto lines = std::max(std::size_t{1}, count);
        for (std::size_t i = 0; i < lines; ++i) {
            const auto* root = i < count ? &view.roots.moves.at(i) : nullptr;
            const auto pv = i < count ? view.variations.at(i) : search::PrincipalVariation{};
            const double q = root && root->visits != 0 ? root->q : info.q;
            const double draw = root && root->visits != 0 ? root->draw : info.draw;
            const auto& chosen = turn.state.options;
            const auto contempt_side = turn.limits.contempt_side;
            const auto rescale = search::wdl_rescale_parameters(chosen.search.draw_rate);
            const bool black_root = turn.state.position.side_to_move() == chess::Color::Black;
            const auto draw_score = (black_root ? -1.0 : 1.0) * chosen.search.draw_score;
            const auto score = format_score(
                {.q = static_cast<float>(q),
                 .draw = static_cast<float>(draw),
                 .type = chosen.score_type,
                 .draw_score_q = static_cast<float>(q + draw_score * draw),
                 .rescale_ratio = rescale.ratio,
                 .rescale_diff =
                     contempt_side == search::ContemptSide::None
                         ? 0.0F
                         : rescale.diff * static_cast<float>(chosen.search.draw_rate.eval_objectivity),
                 .rescale_sign =
                     (contempt_side == search::ContemptSide::Black) == black_root ? 1.0F : -1.0F,
                 .max_s = static_cast<float>(chosen.search.draw_rate.max_s)});
            std::ostringstream line;
            line << "info depth " << pv.size << " seldepth " << info.seldepth << " nodes "
                 << (chosen.per_pv_counters && root ? root->visits : info.nodes) << " nps "
                 << (nps >= static_cast<double>(std::numeric_limits<std::uint64_t>::max())
                         ? std::numeric_limits<std::uint64_t>::max()
                         : static_cast<std::uint64_t>(nps))
                 << " time " << elapsed.count() << " hashfull " << hashfull << " tbhits "
                 << info.tb_hits + (root_tablebase_hit ? 1U : 0U) << " multipv " << i + 1;
            if (score) {
                // lc0 fd71a2d classic/search.cc SendUciInfo: a proven move reports mate in moves,
                // from round(M) / 2 + 1, plus 100 for a tablebase result (which has no real
                // distance to mate, so its M is the estimate it was given).
                std::optional<int> mate;
                if (root && root->visits != 0 &&
                    (root->child_proof.outcome == search::ProofOutcome::Win ||
                     root->child_proof.outcome == search::ProofOutcome::Loss)) {
                    const bool tablebase =
                        root->child_proof.mate_upper_bound >= search::kTablebaseMateDistance;
                    const double distance = tablebase ? root->moves_left.value_or(0.0)
                                                      : root->child_proof.mate_upper_bound;
                    const auto moves =
                        static_cast<int>(std::round(distance) / 2.0 + (tablebase ? 101.0 : 1.0));
                    mate = root->child_proof.outcome == search::ProofOutcome::Loss ? moves : -moves;
                }
                if (mate)
                    line << " score mate " << *mate;
                else
                    line << " score cp " << score->centipawns;
                if (turn.state.options.show_wdl)
                    line << " wdl " << score->wdl[0] << ' ' << score->wdl[1] << ' '
                         << score->wdl[2];
            } else
                error("Cannot represent UCI score: " + score.error());
            // lc0 SendUciInfo: (1 + edge M, or 1 + the root's M with no visits) / 2, in moves.
            if (chosen.show_moves_left) {
                const auto own = root ? root->moves_left : std::nullopt;
                const auto fallback = info.moves_left ? std::optional{1.0 + *info.moves_left}
                                                      : std::nullopt;
                if (const auto moves = own ? own : fallback)
                    line << " movesleft " << static_cast<int>((1.0 + *moves) / 2.0);
            }
            if (pv.size != 0) {
                line << " pv";
                for (std::size_t j = 0; j < pv.size; ++j)
                    line << ' ' << chess::move_to_uci(pv.moves.at(j), turn.state.options.chess960);
            }
            static_cast<void>(emit(line.str()));
        }
    }
    void report_stats(const search::SearchStats& stats) {
        std::uint64_t batches = 0, positions = 0, maximum = 0, p50 = 0, cumulative = 0;
        for (std::size_t i = 1; i < stats.batch_sizes.size(); ++i) {
            batches += stats.batch_sizes.at(i);
            positions += i * stats.batch_sizes.at(i);
            if (stats.batch_sizes.at(i) != 0) maximum = i;
        }
        for (std::size_t i = 1; i < stats.batch_sizes.size() && p50 == 0 && batches != 0; ++i) {
            cumulative += stats.batch_sizes.at(i);
            if (cumulative >= (batches + 1) / 2) p50 = i;
        }
        const auto ms = [](auto duration) {
            return std::chrono::duration<double, std::milli>(duration).count();
        };
        std::ostringstream line;
        line << "info string searchstats batches " << batches << " positions " << positions
             << " mean "
             << (batches ? static_cast<double>(positions) / static_cast<double>(batches) : 0.0)
             << " p50 " << p50 << " max " << maximum << " collisions " << stats.collisions
             << " ooo " << stats.out_of_order << " proof_scans " << stats.proof_scans
             << " selection_hits " << stats.selection_hits << " twofold_reverts "
             << stats.twofold_reverts << " gather_ms " << ms(stats.gather) << " nn_wait_ms "
             << ms(stats.nn_wait) << " backup_ms " << ms(stats.backup) << " provider_ms "
             << ms(stats.provider) << " wall_ms " << ms(stats.wall) << " gpu_idle_ms "
             << ms(std::max(stats.wall - stats.provider, std::chrono::nanoseconds::zero()))
             << " idle_basis host_provider_gaps";
        static_cast<void>(emit(line.str()));
    }
    // Evaluates in batches the backend's largest bucket can hold.
    [[nodiscard]] Result<void> evaluate_rows(std::span<const nn::InputPlanes> inputs,
                                             std::span<nn::NNOutput> outputs,
                                             std::size_t bucket) {
        for (std::size_t at = 0; at < inputs.size(); at += bucket) {
            const auto rows = std::min(bucket, inputs.size() - at);
            auto done = backend->evaluate(inputs.subspan(at, rows), outputs.subspan(at, rows));
            if (!done) return done;
        }
        return {};
    }
    // lc0 fd71a2d search/instamove/instamove.cc: a move from the network alone, with no search.
    [[nodiscard]] Result<void> instamove_turn(Turn& turn) {
        auto root = turn.state.position;
        auto frames = turn.state.history;
        const auto legal = chess::legal_moves(root, attacks);
        View view;
        view.instant = true;
        if (legal.empty()) {
            turn.published.store(std::make_shared<const View>(std::move(view)),
                                 std::memory_order_release);
            return {};
        }
        const auto fill = turn.state.options.search.history_fill;
        const auto input_format = backend->metadata().input_format;
        const auto bucket = minibatch(turn.state.options);
        const auto line = [](std::uint64_t nodes, std::optional<int> mate, double q, double draw) {
            std::ostringstream text;
            text << "info depth 1 seldepth 1 nodes " << nodes;
            if (mate)
                text << " score mate " << *mate;
            else
                text << " score cp " << static_cast<int>(90.0 * std::tan(1.5637541897 * q))
                     << " wdl " << static_cast<int>(std::round(500.0 * (1.0 + q - draw))) << ' '
                     << static_cast<int>(std::round(1000.0 * draw)) << ' '
                     << static_cast<int>(std::round(500.0 * (1.0 - q - draw)));
            return text.str();
        };
        const auto wdl_value = [](const nn::NNOutput& row) {
            const double sum = static_cast<double>(row.wdl[0]) + row.wdl[1] + row.wdl[2];
            return std::pair{(static_cast<double>(row.wdl[0]) - row.wdl[2]) / sum,
                             row.wdl[1] / sum};
        };
        if (turn.state.options.algorithm == SearchAlgorithm::PolicyHead) {
            auto encoded = nn::encode_classical(frames.frames(), input_format, fill);
            if (!encoded) return tl::unexpected(encoded.error());
            std::array<nn::InputPlanes, 1> input{*encoded};
            std::array<nn::NNOutput, 1> evaluated{};
            auto done = backend->evaluate(input, evaluated);
            if (!done) return done;
            std::optional<chess::Move> best;
            float best_logit = -std::numeric_limits<float>::infinity();
            for (const auto move : legal) {
                const auto index = nn::PolicyMap::index(move, root.side_to_move());
                if (!index) return tl::unexpected(index.error());
                if (evaluated[0].policy_logits.at(*index) > best_logit) {
                    best_logit = evaluated[0].policy_logits.at(*index);
                    best = move;
                }
            }
            const auto [q, draw] = wdl_value(evaluated[0]);
            static_cast<void>(emit(line(1, std::nullopt, q, draw)));
            view.result.best_move = best;
        } else {
            struct Score {
                double negative_q = 0, draw = 0;
                std::optional<int> mate;
            };
            std::vector<Score> scores(legal.size());
            std::vector<nn::InputPlanes> inputs;
            std::vector<std::size_t> rows;
            for (std::size_t i = 0; i < legal.size(); ++i) {
                auto child = root;
                static_cast<void>(child.make_move(legal[i], attacks));
                const auto replies = chess::legal_moves(child, attacks);
                if (replies.empty() && child.in_check(attacks)) {
                    scores[i] = {.negative_q = -1, .draw = 0, .mate = 1};
                } else if (replies.empty() || child.is_rule50_draw() ||
                           child.insufficient_material() || child.is_repetition(3)) {
                    scores[i] = {.negative_q = 0, .draw = 1};
                } else {
                    auto next = frames;
                    auto appended = next.append(child);
                    if (!appended) return appended;
                    auto encoded = nn::encode_classical(next.frames(), input_format, fill);
                    if (!encoded) return tl::unexpected(encoded.error());
                    inputs.push_back(*encoded);
                    rows.push_back(i);
                }
            }
            std::vector<nn::NNOutput> outputs(inputs.size());
            auto done = evaluate_rows(inputs, outputs, bucket);
            if (!done) return done;
            for (std::size_t row = 0; row < rows.size(); ++row) {
                const auto [q, draw] = wdl_value(outputs[row]);
                scores[rows[row]] = {.negative_q = q, .draw = draw};
            }
            std::size_t best = 0;
            const auto better = [](const Score& a, const Score& b) {
                if (a.mate && !b.mate) return true;
                if (!a.mate && b.mate) return false;
                if (a.mate && b.mate) return *a.mate < *b.mate;
                return a.negative_q < b.negative_q;
            };
            for (std::size_t i = 1; i < scores.size(); ++i)
                if (better(scores[i], scores[best])) best = i;
            static_cast<void>(emit(line(legal.size(), scores[best].mate, -scores[best].negative_q,
                                        scores[best].draw)));
            view.result.best_move = legal[best];
        }
        turn.published.store(std::make_shared<const View>(std::move(view)),
                             std::memory_order_release);
        return {};
    }
    void search_turn(Turn& turn) {
        if (turn.cancel.stop_requested()) return;
        const auto prepared = prepare(turn.state, turn.cancel.get_token());
        if (!prepared) {
            // A cancelled search-only allocation is obsolete; an actual model
            // load failure is still fatal, including after an on-time reply.
            if (turn.cancel.stop_requested() && backend &&
                prepared_backend_revision == turn.state.backend_revision)
                return;
            fatal(prepared.error());
            return;
        }
        if (turn.cancel.stop_requested()) return;
        if (turn.state.options.algorithm != SearchAlgorithm::Classic &&
            turn.state.options.algorithm != SearchAlgorithm::DagPreview) {
            auto played = instamove_turn(turn);
            if (!played) fatal(played.error());
            turn.primed.store(true, std::memory_order_release);
            changed.notify_all();
            return;
        }
        turn.inherited = engine->snapshot().nodes;
        turn.live = engine.get();
        auto bounded = turn.limits;
        bounded.stop_token = turn.cancel.get_token();
        if (turn.time.smooth) {
            turn.time.smooth->begin(turn.inherited);
            bounded.time_stopper = [plan = turn.time.smooth,
                                    started = turn.started](const search::TimeSample& sample) {
                return plan->decide(sample, std::chrono::duration_cast<std::chrono::milliseconds>(
                                                Clock::now() - started));
            };
        }
        // Rebuilding the delivery view walks the root, PVs and snapshot under the
        // tree lock on the search thread, so after every batch it costs ~5% nps.
        // A deadline delivery may use a view up to this old; a search that ends
        // normally republishes its final view below before bestmove.
        bounded.on_progress = [this, &turn, last = Clock::time_point{}]() mutable {
            const auto now = Clock::now();
            if (now - last < kProgressViewInterval) return;
            last = now;
            publish_view(turn);
        };
        auto prime = bounded;
        prime.nodes = std::max(std::uint64_t{1}, engine->snapshot().nodes);
        prime.depth.reset();
        prime.movetime.reset();
        prime.smart_pruning = false;
        const auto calls_before = physical_calls.load(std::memory_order_relaxed);
        const auto prime_started = Clock::now();
        auto result = engine->run(prime);
        if (!result) {
            fatal(result.error());
            return;
        }
        publish_view(turn);
        auto totals = result->stats;
        if (physical_calls.load(std::memory_order_relaxed) != calls_before)
            root_latency.observe(Clock::now() - prime_started);
        turn.primed.store(true, std::memory_order_release);
        if (turn.stop_requested.load(std::memory_order_acquire)) turn.cancel.request_stop();
        changed.notify_all();
        std::array<BucketCost, BucketLatencies::kCapacity> bucket_storage{};
        const auto measured = bucket_latency.snapshot(bucket_storage);
        const auto schedule = schedule_search(
            turn.time, turn.started, Clock::now(), root_latency.estimate(),
            buckets(turn.state.options), minibatch(turn.state.options), measured);
        bounded.deadline = schedule.deadline;
        bounded.maximum_batch = schedule.maximum_batch;
        if (turn.command.ponder && !turn.command.infinite) {
            const auto& hit_time = turn.ponder_budget;
            const auto now = Clock::now();
            const auto hit_schedule =
                hit_time
                    ? schedule_search(*hit_time, now, now, root_latency.estimate(),
                                      buckets(turn.state.options), minibatch(turn.state.options),
                                      measured)
                    : SearchSchedule{
                          .deadline = std::nullopt, .maximum_batch = 1, .exhausted = true};
            bounded.maximum_batch = hit_schedule.exhausted ? 1 : hit_schedule.maximum_batch;
            if (!hit_time || !hit_time->budget) bounded.maximum_batch = 1;
        }
        const bool complete = bounded.nodes && result->info.nodes >= *bounded.nodes;
        // A known slow provider may not fit the remaining physical budget.
        // Avoid accepting it, but do not mistake that forecast for a TM stop:
        // the independent publisher still uses the full allocation.
        if (!complete && schedule.exhausted && turn.time.budget &&
            result->reason != search::StopReason::Terminal)
            turn.hold_completed.store(true, std::memory_order_release);
        if (!turn.cancel.stop_requested() && result->reason != search::StopReason::Terminal &&
            !schedule.exhausted && !complete) {
            result = engine->run(bounded);
            if (!result) {
                fatal(result.error());
                return;
            }
            publish_view(turn);
            totals += result->stats;
        }
        if (turn.state.options.log_search_stats) report_stats(totals);
        if (turn.state.options.verbose_move_stats)
            for (const auto& line : engine->verbose_move_stats(turn.state.options.chess960))
                static_cast<void>(emit("info string " + line));
    }
    void worker_loop(const std::stop_token& stop) noexcept {
        while (!stop.stop_requested()) {
            std::shared_ptr<Turn> turn;
            std::optional<State> prepare_state;
            {
                std::unique_lock lock(wait_mutex);
                changed.wait(lock, stop, [&] { return pending || preparation; });
                if (stop.stop_requested()) break;
                if (preparation) {
                    prepare_state = std::move(preparation);
                    preparation.reset();
                    preparing = true;
                } else {
                    turn = std::move(pending);
                    working = turn;
                }
            }
            if (prepare_state) {
                Result<void> prepared;
                try {
                    prepared = prepare(*prepare_state);
                } catch (const std::exception& e) {
                    prepared = tl::unexpected(std::string{"Backend preparation: "} + e.what());
                } catch (...) {
                    prepared = tl::unexpected(
                        std::string{"Backend preparation failed with non-standard exception"});
                }
                bool obsolete = false, failed_turn = false;
                {
                    const std::lock_guard lock(wait_mutex);
                    obsolete = prepare_state->backend_revision != requested_backend_revision;
                    // Check and acknowledge the same revision under one lock:
                    // a later option must never inherit an obsolete load error.
                    const bool replaced_search = preparation && preparation->backend_revision ==
                                                                    prepare_state->backend_revision;
                    const bool obsolete_search_error =
                        !prepared && replaced_search && backend &&
                        prepared_backend_revision == prepare_state->backend_revision;
                    if (!obsolete) {
                        if (prepared || obsolete_search_error)
                            loaded_backend_revision = prepare_state->backend_revision;
                        else {
                            preparation_error = prepared.error();
                            failed_turn = current && current->state.backend_revision ==
                                                         prepare_state->backend_revision;
                        }
                    }
                }
                // A later backend option invalidates both successes and errors
                // from the old load. Destruction remains on this joined owner.
                if (obsolete) {
                    engine.reset();
                    neural_cache.reset();
                    backend.reset();
                }
                if (failed_turn) fatal(prepared.error());
                {
                    const std::lock_guard lock(wait_mutex);
                    preparing = false;
                }
                changed.notify_all();
                continue;
            }
            try {
                search_turn(*turn);
            } catch (const std::exception& e) {
                fatal(std::string{"UCI search: "} + e.what());
            } catch (...) {
                fatal("UCI search failed with non-standard exception");
            }
            turn->live = nullptr;
            turn->done.store(true, std::memory_order_release);
            {
                const std::lock_guard lock(wait_mutex);
                working.reset();
            }
            changed.notify_all();
        }
    }
    void publish_turn(const std::shared_ptr<Turn>& turn) noexcept {
        try {
            constexpr auto report_interval = std::chrono::milliseconds{100};
            auto next_report = turn->started + report_interval;
            bool reported_completion = false;
            auto deadline = turn->deadline;
            if (deadline)
                deadline = publication_start(turn->started, *deadline, turn->hard_deadline,
                                             publication_latency.estimate());
            auto wake_lateness = Clock::duration::zero();
            for (;;) {
                if (status.load(std::memory_order_acquire) != 0 || turn->cancel.stop_requested())
                    break;
                const auto now = Clock::now();
                const bool done = turn->done.load(std::memory_order_acquire);
                if (deadline.has_value() && now >= *deadline) {
                    wake_lateness = now - *deadline;
                    break;
                }
                if ((turn->stop_requested.load(std::memory_order_acquire) &&
                     turn->primed.load(std::memory_order_acquire)) ||
                    (done && !turn->time.hold_until_signal &&
                     !turn->hold_completed.load(std::memory_order_acquire)))
                    break;
                if (now >= next_report || (done && !reported_completion)) {
                    // The final report supersedes periodic output in this interval.
                    // Starting another snapshot/write here can block the same
                    // publisher beyond its deadline, even with a warm provider.
                    const bool final_interval = deadline && *deadline - now <= report_interval;
                    if (!final_interval) report(*turn, *view_of(*turn));
                    next_report = final_interval ? *deadline : Clock::now() + report_interval;
                    reported_completion = done;
                }
                auto wake = next_report;
                if (deadline.has_value()) wake = std::min(wake, *deadline);
                std::unique_lock lock(wait_mutex);
                changed.wait_until(lock, turn->cancel.get_token(), wake, [&] {
                    return status.load(std::memory_order_acquire) != 0 ||
                           (turn->stop_requested.load(std::memory_order_acquire) &&
                            turn->primed.load(std::memory_order_acquire)) ||
                           (!turn->time.hold_until_signal &&
                            turn->done.load(std::memory_order_acquire) &&
                            !turn->hold_completed.load(std::memory_order_acquire));
                });
            }
            const auto delivery_started = Clock::now();
            trace("publication_begin");
            // Stop accepting work before freezing the decision. A provider call
            // already accepted stays owned by the worker until it returns; the
            // publisher neither joins it nor invents a neural result for it.
            turn->cancel.request_stop();
            changed.notify_all();
            if (status.load(std::memory_order_acquire) == 0 &&
                !suppress_best.load(std::memory_order_acquire)) {
                search::SnapshotCost snapshot_cost;
                std::optional<search::SnapshotCostScope> recording;
                if (debug_enabled.load(std::memory_order_relaxed)) recording.emplace(snapshot_cost);
                const auto view = view_of(*turn);
                trace("publication_snapshot", 0, 0, &snapshot_cost, &view->snapshot_cost);
                report(*turn, *view);
                // Keep the publication value initialized independently of optional resets.
                // GCC 13 loses optional engagement tracking across the ranking branches.
                auto best = view->temperature_move ? *view->temperature_move
                                                   : view->result.best_move.value_or(chess::Move{});
                bool has_best = view->temperature_move || view->result.best_move.has_value();
                if (has_best && turn->limits.searchmoves_count != 0 &&
                    std::find(turn->limits.searchmoves.begin(),
                              turn->limits.searchmoves.begin() +
                                  static_cast<std::ptrdiff_t>(turn->limits.searchmoves_count),
                              best) ==
                        turn->limits.searchmoves.begin() +
                            static_cast<std::ptrdiff_t>(turn->limits.searchmoves_count))
                    has_best = false;
                if (!has_best && turn->emergency_move) {
                    const auto choice = emergency_choice(
                        view->roots,
                        {.reused = turn->reused_move, .deterministic = turn->emergency_move});
                    best = choice.move.value_or(chess::Move{});
                    has_best = choice.move.has_value();
                    if (choice.without_evaluation)
                        static_cast<void>(
                            emit("info string warning: emergency move without evaluation"));
                }
                time_manager.complete(turn->time,
                                      std::chrono::duration_cast<std::chrono::milliseconds>(
                                          Clock::now() - turn->started),
                                      view->result.info.nodes);
                publishing.store(true, std::memory_order_release);
                trace("bestmove");
                if (status.load(std::memory_order_acquire) == 0)
                    static_cast<void>(emit(
                        has_best
                            ? "bestmove " + chess::move_to_uci(best, turn->state.options.chess960)
                            : "bestmove 0000",
                        true));
                publication_latency.observe(wake_lateness + Clock::now() - delivery_started);
            }
        } catch (const std::exception& e) {
            fatal(std::string{"UCI publication: "} + e.what());
        } catch (...) {
            fatal("UCI publication failed with non-standard exception");
        }
        active.store(false, std::memory_order_release);
    }
    Result<void> start(const GoCommand& command, Clock::time_point started = Clock::now()) {
        if (status.load(std::memory_order_acquire) != 0)
            return tl::unexpected(std::string{"UCI session has failed"});
        if (!position) return tl::unexpected(std::string{"UCI session has no position"});
        if (command.ponder && !options.ponder)
            return tl::unexpected(std::string{"go ponder requires Ponder=true"});
        const auto bounded = limits(command);
        if (!bounded) return tl::unexpected(bounded.error());
        auto allocated_manager = time_manager;
        const auto time =
            allocated_manager.allocate(command, {.side_to_move = position->side_to_move(),
                                                 .overhead_ms = options.move_overhead_ms,
                                                 .game_ply = lc0_game_ply});
        if (!time) return tl::unexpected(time.error());
        if (options.backend.model_path.empty())
            return tl::unexpected(
                std::string{"WeightsFile is required before neural readiness or search"});
        std::optional<std::string> failed_load;
        {
            const std::lock_guard lock(wait_mutex);
            failed_load = preparation_error;
        }
        if (failed_load) {
            fatal(*failed_load);
            return tl::unexpected(*failed_load);
        }
        auto turn = std::make_shared<Turn>(capture(), command, *bounded, *time, started);
        turn->game_ply = lc0_game_ply;
        if (command.ponder && !command.infinite) {
            auto hit = command;
            hit.ponder = false;
            auto preview = allocated_manager;
            const auto future = preview.allocate(hit, {.side_to_move = position->side_to_move(),
                                                       .overhead_ms = options.move_overhead_ms,
                                                       .game_ply = lc0_game_ply});
            if (future) turn->ponder_budget = *future;
        }
        auto probe = turn->state.position;
        const auto legal = chess::legal_moves(probe, attacks);
        chess::MoveList allowed;
        for (const auto move : legal) {
            if (bounded->searchmoves_count == 0 ||
                std::find(bounded->searchmoves.begin(),
                          bounded->searchmoves.begin() +
                              static_cast<std::ptrdiff_t>(bounded->searchmoves_count),
                          move) != bounded->searchmoves.begin() +
                                       static_cast<std::ptrdiff_t>(bounded->searchmoves_count))
                allowed.push_back(move);
        }
        turn->emergency_move = deterministic_emergency(probe, allowed, attacks);
        // Preserve an evaluated descendant along the previous principal
        // variation while that turn's accepted inference is still draining.
        std::shared_ptr<Turn> previous;
        {
            const std::lock_guard lock(wait_mutex);
            previous = current;
        }
        if (previous && previous->state.backend_revision == backend_revision &&
            previous->state.search_revision == search_revision &&
            previous->state.game_generation == game_generation &&
            previous->state.base_fen == turn->state.base_fen &&
            turn->state.moves.size() >= previous->state.moves.size() &&
            std::equal(previous->state.moves.begin(), previous->state.moves.end(),
                       turn->state.moves.begin())) {
            const auto prior = view_of(*previous);
            const auto offset = turn->state.moves.size() - previous->state.moves.size();
            const auto& pv = prior->result.info.pv;
            if (offset < pv.size &&
                std::equal(pv.moves.begin(), pv.moves.begin() + static_cast<std::ptrdiff_t>(offset),
                           turn->state.moves.begin() +
                               static_cast<std::ptrdiff_t>(previous->state.moves.size())) &&
                std::ranges::find(allowed, pv.moves.at(offset)) != allowed.end())
                turn->reused_move = pv.moves.at(offset);
        }
        if (legal.empty()) {
            const bool mate = chess::game_status(probe, attacks) == chess::GameStatus::Checkmate;
            auto terminal = std::make_shared<View>();
            terminal->result.reason = search::StopReason::Terminal;
            terminal->result.info.q = mate ? -1.0 : 0.0;
            terminal->result.info.draw = mate ? 0.0 : 1.0;
            turn->published.store(std::move(terminal), std::memory_order_release);
        }
        current_go = command;
        trace("go");
        suppress_best.store(false, std::memory_order_release);
        publishing.store(false, std::memory_order_release);
        active.store(true, std::memory_order_release);
        try {
            time_manager = allocated_manager;
            job = std::jthread([this, turn] { publish_turn(turn); });
            {
                const std::lock_guard lock(wait_mutex);
                current = turn;
                if (pending) pending->cancel.request_stop();
                pending = turn;
            }
            changed.notify_all();
        } catch (...) {
            active.store(false, std::memory_order_release);
            throw;
        }
        return {};
    }
    bool handle(const Command& command) {
        join_finished();
        if (std::holds_alternative<EmptyCommand>(command)) return true;
        if (std::holds_alternative<UciCommand>(command)) {
            static_cast<void>(emit(std::string{"id name Mouse "} + std::string{build::version}));
            static_cast<void>(emit("id author Zulfeqar Malik"));
            for (const auto& line : option_lines(options)) static_cast<void>(emit(line));
            static_cast<void>(emit("uciok"));
        } else if (std::holds_alternative<IsReadyCommand>(command)) {
            {
                const auto prepared =
                    active.load(std::memory_order_acquire) ? await_backend() : ready();
                if (!prepared) {
                    if (!options.backend.model_path.empty()) {
                        if (status.load(std::memory_order_acquire) == 0) fatal(prepared.error());
                        return false;
                    }
                    error(prepared.error());
                }
            }
            static_cast<void>(emit("readyok"));
        } else if (std::holds_alternative<QuitCommand>(command)) {
            finish(false);
            return false;
        } else if (std::holds_alternative<StopCommand>(command)) {
            finish(true);
        } else if (std::holds_alternative<PonderHitCommand>(command)) {
            if (!active.load(std::memory_order_acquire) || !current_go || !current_go->ponder) {
                error("ponderhit requires an active ponder search");
            } else {
                const auto hit = Clock::now();
                auto next = *current_go;
                next.ponder = false;
                finish(false);
                // The accepted ponder call drains on its worker. Publish the
                // hit's clock independently; a later error remains fatal.
                if (status.load(std::memory_order_acquire) != 0) return false;
                const auto restarted = start(next, hit);
                if (!restarted && status.load(std::memory_order_acquire) == 0)
                    error(restarted.error());
            }
        } else if (std::holds_alternative<DebugCommand>(command)) {
            debug_enabled.store(std::get<DebugCommand>(command).enabled, std::memory_order_relaxed);
            static_cast<void>(emit(std::get<DebugCommand>(command).enabled
                                       ? "info string debug on"
                                       : "info string debug off"));
        } else if (active.load(std::memory_order_acquire)) {
            error(
                "Stop the active search before changing position, options or starting another "
                "search");
        } else if (const auto* update = std::get_if<SetOptionCommand>(&command)) {
            if (!position.has_value()) throw std::logic_error("UCI session has no root position");
            const auto changed_option = apply_option(options, *update);
            if (!changed_option)
                error(changed_option.error());
            else {
                const auto variant =
                    chess::validate_uci_castling_mode(*position, changed_option->options.chess960);
                if (!variant)
                    error(variant.error());
                else {
                    if (changed_option->impact != Reconfigure::None) ++search_revision;
                    if (changed_option->impact == Reconfigure::Backend) ++backend_revision;
                    const bool new_log = changed_option->options.log_file != options.log_file;
                    const bool new_tables =
                        changed_option->options.syzygy_path != options.syzygy_path;
                    if (changed_option->options.time_manager != options.time_manager)
                        time_manager = LegacyTimeManager{changed_option->options.time_manager};
                    options = changed_option->options;
                    if (new_log) open_log(options.log_file);
                    if (new_tables) load_tablebase();
                    if (changed_option->impact != Reconfigure::None) request_preparation();
                }
            }
        } else if (const auto* proposed = std::get_if<PositionCommand>(&command)) {
            const auto configured = configure_position(*proposed);
            if (!configured) error(configured.error());
        } else if (std::holds_alternative<NewGameCommand>(command)) {
            ++game_generation;
            time_manager.reset();
            const auto configured = configure_position(PositionCommand{});
            if (!configured) error(configured.error());
        } else if (const auto* go = std::get_if<GoCommand>(&command)) {
            const auto begun = start(*go);
            if (!begun && status.load(std::memory_order_acquire) == 0) error(begun.error());
        }
        return status.load(std::memory_order_acquire) == 0;
    }
};
Session::Session(Options options, Output output, BackendFactory factory)
    : impl_(std::make_unique<Impl>(std::move(options), std::move(output), std::move(factory))) {
    impl_->time_manager = LegacyTimeManager{impl_->options.time_manager};
    impl_->open_log(impl_->options.log_file);
    impl_->load_tablebase();
}
Session::~Session() { shutdown(); }
bool Session::handle(std::string_view line) {
    if (impl_->closed || impl_->status.load(std::memory_order_acquire) != 0) return false;
    try {
        {
            const std::lock_guard lock(impl_->output_mutex);
            impl_->log_line(">>", line);
        }
        const auto command = parse_command(line);
        if (!command) {
            impl_->error(command.error());
            return true;
        }
        return impl_->handle(*command);
    } catch (const std::exception& e) {
        impl_->fatal(std::string{"UCI session: "} + e.what());
    } catch (...) {
        impl_->fatal("UCI session failed with non-standard exception");
    }
    return false;
}
bool Session::report_input_error(std::string_view message) {
    if (impl_->closed || impl_->status.load(std::memory_order_acquire) != 0) return false;
    try {
        impl_->error(message);
    } catch (...) {
        impl_->fatal("Cannot render rejected UCI input");
    }
    return impl_->status.load(std::memory_order_acquire) == 0;
}
void Session::shutdown() noexcept {
    impl_->closed = true;
    impl_->input_cancel.request_stop();
    try {
        impl_->finish(false);
        impl_->cancel_requests();
        impl_->worker.request_stop();
        impl_->changed.notify_all();
        if (impl_->worker.joinable()) impl_->worker.join();
        impl_->engine.reset();
        impl_->backend.reset();
    } catch (...) {
        impl_->status.store(1, std::memory_order_release);
    }
}
int Session::exit_code() const noexcept { return impl_->status.load(std::memory_order_acquire); }
std::stop_token Session::cancellation_token() const noexcept {
    return impl_->input_cancel.get_token();
}
}  // namespace mouse::uci
