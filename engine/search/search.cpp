// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/search/search.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <iomanip>
#include <limits>
#include <mutex>
#include <new>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "engine/chess/attacks.h"
#include "engine/chess/movegen.h"
#include "engine/nn/legal_evaluator.h"
#include "engine/nn/policy_map.h"
#include "engine/search/collisions.h"
#include "engine/search/moves_left.h"
#include "engine/search/pick_queue.h"
#include "engine/search/prior.h"
#include "engine/search/pruning.h"
#include "engine/search/rules.h"

#if defined(_MSC_VER)
#include <immintrin.h>
#endif

namespace mouse::search {
namespace {
using Clock = std::chrono::steady_clock;
constexpr std::size_t kMaximumDepth = 512;
// Largest number of legal moves in any chess position (standard or Chess960).
// Tree edges are legal moves, so pick scratch is sized to it, not to the
// pseudo-legal MoveList capacity.
constexpr std::size_t kMaximumLegalMoves = 218;
constexpr std::size_t kAllocationAllowance = 64;
// Thread control/callable bookkeeping; OS stacks are outside the managed budget.
constexpr std::size_t kThreadAllowance = 1024;
// Lc0 classic/search.cc MAX_TASKS: the most subtrees one pick hands to other threads.
constexpr std::size_t kMaximumPickTasks = 100;
inline void spin_pause() noexcept {
#if defined(_MSC_VER)
    _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#else
    std::this_thread::yield();
#endif
}
// Lc0 utils/hashcat.h HashCat.
constexpr std::uint64_t hash_cat(std::uint64_t seed, std::uint64_t value) noexcept {
    return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U));
}
auto identity(const nn::BackendInfo& v) {
    return std::tie(v.onnx_sha256, v.provider, v.precision, v.device_name, v.device_uuid,
                    v.driver_version, v.cuda_version, v.tensorrt_version, v.cache_namespace);
}
bool matches(const chess::Position& root, const nn::EncodingHistory& history) {
    if (history.frames().empty()) return false;
    const auto& last = history.frames().back();
    if (last.side_to_move() != root.side_to_move() ||
        last.halfmove_clock() != root.halfmove_clock())
        return false;
    for (auto color : {chess::Color::White, chess::Color::Black}) {
        for (auto type :
             {chess::PieceType::Pawn, chess::PieceType::Knight, chess::PieceType::Bishop,
              chess::PieceType::Rook, chess::PieceType::Queen, chess::PieceType::King})
            if (last.pieces(color, type) != root.pieces(color, type)) return false;
        for (bool kingside : {false, true})
            if (last.castling_rook(color, kingside) != root.castling_rook(color, kingside))
                return false;
    }
    auto ep = root.en_passant_square();
    // Lc0 v0.32.1 src/chess/board.cc ApplyMove: geometric EP, including pinned
    // capturers. A FEN root instead preserves its stated EP verbatim.
    if (history.frames().size() > 1 && ep != chess::kNoSquare) {
        const int rank = chess::rank_of(ep) + (root.side_to_move() == chess::Color::White ? -1 : 1);
        bool capturer = false;
        for (int delta : {-1, 1}) {
            const int file = chess::file_of(ep) + delta;
            if (file >= 0 && file < 8 &&
                (root.pieces(root.side_to_move(), chess::PieceType::Pawn) &
                 chess::bit(chess::square(file, rank))) != 0)
                capturer = true;
        }
        if (!capturer) ep = chess::kNoSquare;
    }
    return ep == last.en_passant_square();
}
}  // namespace
struct SearchEngine::Impl {
    struct Path {
        std::array<NodeId, kMaximumDepth + 1> nodes{};
        std::array<chess::Move, kMaximumDepth> moves{};
        std::size_t size = 0;
    };
    // Every selected path has exactly one owner, including exception unwinding.
    struct Reservation {
        Tree* tree = nullptr;
        std::mutex* mutex = nullptr;
        Path* path = nullptr;
        bool active = false, owns_pending = false;
        Reservation() = default;
        Reservation(const Reservation&) = delete;
        Reservation& operator=(const Reservation&) = delete;
        Reservation(Reservation&&) = delete;
        Reservation& operator=(Reservation&&) = delete;
        ~Reservation() { release(); }
        void release_locked() noexcept {
            if (!active) return;
            for (std::size_t i = 0; i < path->size; ++i) --tree->node(path->nodes.at(i)).in_flight;
            auto& leaf = tree->node(path->nodes.at(path->size - 1));
            if (owns_pending && leaf.state == NodeState::Pending)
                leaf.state = NodeState::Unexpanded;
            active = false;
        }
        void release() noexcept {
            if (active) {
                const std::lock_guard lock(*mutex);
                release_locked();
            }
        }
    };
    struct Request {
        Path path;
        chess::MoveList legal;
        chess::Color side = chess::Color::White;
        std::array<std::uint16_t, chess::MoveList::kCapacity> priors{};
        std::array<EdgeSeed, chess::MoveList::kCapacity> seeds{};
        // Known requests back up their stored leaf value (terminal, frontier or
        // depth ceiling); the others need classification and evaluation.
        bool known = false;
        bool ready = false;
        // OutOfOrderEval off: a value that needs no NN waits for the batch to back up.
        bool deferred = false;
        double deferred_q = 0, deferred_draw = 0;
        std::optional<double> deferred_moves_left;
        // DAG only: what finds this position's low node.
        std::uint64_t low_key = 0;
        Reservation reservation;
    };
    // Lc0 dag_classic BackupPath's repetitions and moves left, per node of a request's path.
    struct DagPath {
        std::array<std::uint8_t, kMaximumDepth + 1> repetitions{};
        std::array<std::uint16_t, kMaximumDepth + 1> cycles{};
    };
    enum class SlotState : std::uint8_t { Idle, Ready, Running, Done, Cancelled };
    // Provider diagnostics are untrusted in length. Retain bounded charged
    // storage, with explicit truncation in the returned failure description.
    struct ErrorText {
        std::array<char, 1024> bytes{};
        std::size_t used = 0;
        bool truncated = false;
        [[nodiscard]] bool empty() const noexcept { return used == 0; }
        void clear() noexcept {
            used = 0;
            truncated = false;
        }
        void append(std::string_view text) noexcept {
            const auto count = std::min(text.size(), bytes.size() - used);
            for (std::size_t i = 0; i < count; ++i) bytes.at(used + i) = text[i];
            used += count;
            truncated = truncated || count != text.size();
        }
        void assign(std::string_view text) noexcept {
            clear();
            append(text);
        }
        [[nodiscard]] std::string string() const {
            std::string result(bytes.data(), used);
            if (truncated) result += " [diagnostic truncated]";
            return result;
        }
    };
    struct Collision {
        std::size_t offset = 0, size = 0;
        std::uint32_t visits = 0;
    };
    struct Slot {
        std::vector<Request> requests;
        std::vector<nn::InputPlanes> inputs;
        std::vector<nn::NNOutput> outputs;
        std::vector<nn::LegalOutput> legal_outputs;
        std::vector<nn::PolicyIndices> policies;
        std::vector<std::size_t> rows;
        std::size_t count = 0;
        // Lc0 MaxPrefetch rows follow the count real rows in inputs/policies/legal_outputs.
        // They reach the NN and the cache only; no tree node waits for them.
        std::size_t prefetched = 0;
        SlotState state = SlotState::Idle;
        std::uint64_t sequence = 0;
        std::atomic<std::size_t> next{0};
        std::atomic<bool> halt{false};
        // Rows the publication phase prepared; publishing requires every one of them.
        std::atomic<std::size_t> prepared{0};
        // Tree mutex protects weighted ancestor reservations. Records and flat
        // paths are bounded/preallocated; pending leaves are never NN rows.
        std::vector<Collision> collisions;
        std::vector<NodeId> collision_nodes;
        std::size_t collision_count = 0, collision_nodes_used = 0;
        std::size_t out_of_order_count = 0;
        std::uint32_t collision_visits = 0, collision_root_visits = 0, collision_budget = 1;
        // Rows picked by the current round, processed in parallel by workers.
        std::vector<std::size_t> picked;
        std::size_t picked_count = 0;
        std::vector<DagPath> dag_paths;
        ErrorText error;
    };
    // lc0 PickNodesToExtendTask visits_to_perform, stored sparsely: one entry
    // per edge that still has visits to place below it after the leaf visit it
    // started during the round loop. A level owns entries [begin, end).
    struct PickEntry {
        std::uint16_t index = 0;
        std::uint32_t visits = 0;
    };
    struct PickLevel {
        NodeId node = kNoNode;
        std::size_t begin = 0, end = 0, next = 0;
    };
    // One thread's walk over part of the tree. The coordinator holds tree_mutex throughout, and
    // while it does a walk owns every node at or below base_depth on its path.
    struct PickWalk {
        std::vector<PickLevel> levels;
        // Entries on the walk stack. A level's entries hold at least one visit
        // each and the descended entry carries the next level's limit, so the
        // stack never holds more than walk limit + depth entries, and the walk
        // limit is at most one batch.
        std::vector<PickEntry> entries;
        // Per-edge PUCT terms of the node being distributed. The round loop
        // reads them through a span sized to the node's edges.
        struct EdgeTerms {
            double utility = 0, exploration = 0;
            std::uint64_t started = 0;
            std::uint16_t entry = 0;
            std::uint16_t cycle = 0;
            std::uint8_t repetitions = 0;
            bool eligible = false;
            bool repetitions_known = false;
        };
        std::array<EdgeTerms, kMaximumLegalMoves> terms{};
        Path path;
        DagPath dag;
        // Nodes above base_depth belong to whoever handed this walk its subtree; the
        // visits below them are added to those nodes once every walk has finished.
        std::size_t base_depth = 0;
        std::uint64_t ancestor_visits = 0;
        // lc0 collision_limit, passed_off and completed_visits.
        std::uint32_t task_limit = 0, passed_off = 0, completed = 0;
        // lc0 max_limit / NodeToProcess::maxvisit: a collision that takes the
        // whole walk may grow to the smallest visits-to-change-best seen.
        std::uint32_t walk_limit = 0;
        std::uint64_t max_limit = 0;
        std::size_t upsize_record = 0;
        std::uint64_t upsize_cap = 0;
        std::uint64_t collisions = 0, selection_hits = 0;
    };
    // Lc0 PickTask: a subtree another thread may place visits in.
    struct PickTask {
        Path path;
        std::uint32_t limit = 0;
        std::uint64_t ancestor_visits = 0;
    };
    struct Picker {
        PickWalk main;
        std::vector<std::size_t> free_rows;
        std::atomic<std::size_t> free_used{0};
        std::atomic<std::uint64_t> physical_left{0};
        // lc0 PickNodesToExtendTask root catch-up pruning: this gather's remaining
        // playout estimate (latest_time_manager_hints_) and, fetched once per walk, the
        // visits of the current best root edge (best_node_n).
        std::uint64_t remaining_playouts = std::numeric_limits<std::uint64_t>::max();
        std::uint64_t best_visits = 0;
        // Parallel picking. Empty when off. One more walk than threads: the coordinator
        // takes remaining tasks itself rather than wait for a thread to wake.
        std::vector<PickWalk> task_walks;
        std::vector<PickTask> tasks;
        PickQueue queue;
        std::atomic<std::uint64_t> wake{0};
        std::atomic<bool> closed{true}, shutdown{false};
        Slot* slot = nullptr;
        // Tree::allocate_node, tasks, and what classify changes outside a walk's own nodes.
        std::mutex allocation, tasks_mutex, shared_state;
        std::string error;
    };
    struct Worker {
        // Bounded worker-local hot-path scratch, charged through sizeof(Worker).
        std::optional<chess::Position> position;
        nn::EncodingHistory history;
        std::array<chess::Undo, kMaximumDepth> undo{};
        std::array<std::uint64_t, kMaximumDepth + 1> keys{};
    };
    SearchConfig config;
    // Lc0 applies draw-rate rescaling only when the ratio differs from 1, and the diff only
    // with a contempt side in force.
    WdlRescale wdl_rescale;
    const chess::AttackTables& attacks;
    nn::Backend& backend;
    nn::BackendInfo backend_identity;
    nn::ModelMetadata metadata;
    std::unique_ptr<nn::LegalEvaluator> evaluator;
    nn::LegalCache* legal_cache = nullptr;  // Borrowed, internally synchronized by LegalCache.
    std::mutex lifecycle;
    mutable std::mutex tree_mutex;
    std::mutex work_mutex;
    std::condition_variable work_changed, phase_done, inference_changed;
    std::array<Slot, 2> slots;
    Picker picker;
    std::vector<Worker> workers;
    std::vector<std::thread> worker_threads;
    std::vector<std::thread> pick_threads;
    std::thread inference_thread;
    std::unique_ptr<Tree> tree;
    std::optional<chess::Position> root;
    nn::EncodingHistory root_history;
    std::atomic<bool> external_stop{false};
    bool shutdown = false;
    std::size_t epoch = 0, remaining_workers = 0;
    // What the workers do with `gathering` this epoch: process its picked rows, or
    // prepare its evaluated rows for publication.
    enum class WorkPhase : std::uint8_t { Process, Publish };
    WorkPhase phase = WorkPhase::Process;
    Slot* gathering = nullptr;
    // Scratch taking part in this epoch, coordinator included (lc0 num_tasks).
    std::size_t active_tasks = 0;
    // DrawScore's share at even tree depths, from the root side's view (lc0 GetDrawScore).
    double even_draw_score = 0;
    std::atomic<std::uint64_t> tb_hits{0};
    std::uint64_t next_sequence = 0;
    SearchLimits limits;
    Clock::time_point started{}, finished{};
    std::optional<Clock::time_point> deadline;
    std::uint64_t inherited = 0, completed_batches = 0;
    // lc0 Search::current_best_edge_: the root edge the search would play now, empty at
    // the start of each search and refreshed by backup_locked.
    std::optional<std::size_t> current_best;
    SmartPruning pruning;
    SearchStats diagnostics;  // tree_mutex protects every diagnostic update.
    std::size_t seldepth = 0, overhead = 0;
    bool running = false;
    std::optional<StopReason> stopped;
    // tree_mutex protects it; noise and temperature are drawn from it.
    std::mt19937_64 random;
    // tree_mutex protects it. Edges the rows still to publish in this batch were promised.
    std::size_t batch_edges_reserved = 0;
    Impl(const SearchConfig& selected, const chess::AttackTables& table, nn::Backend& network,
         nn::LegalCache* cache)
        : config(selected),
          wdl_rescale(wdl_rescale_parameters(selected.draw_rate)),
          attacks(table),
          backend(network),
          backend_identity(network.info()),
          metadata(network.metadata()),
          random(selected.random_seed != 0 ? selected.random_seed : std::random_device{}()) {
        if (cache && cache->stats().capacity != 0) legal_cache = cache;
        // Normal mode: the provider thread only evaluates. Lookups happen while
        // processing picked rows and inserts while publishing, both off the
        // provider thread; like lc0 classic search, a batch is not deduplicated.
        // Deterministic mode has no out-of-order cache backups (identical
        // cache-on/off trees), so the evaluator itself reuses cached rows.
        if (cache)
            evaluator = std::make_unique<nn::LegalEvaluator>(
                network, selected.maximum_batch, selected.deterministic ? cache : nullptr);
    }
    ~Impl() = default;
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&) = delete;
    Impl& operator=(Impl&&) = delete;
    [[nodiscard]] bool same_backend() const {
        return identity(backend_identity) == identity(backend.info()) &&
               metadata == backend.metadata();
    }
    [[nodiscard]] std::size_t history_charge() const {
        std::size_t result = root ? root->history_storage_bytes() : 0;
        result += root_history.history_storage_bytes();
        for (const auto& worker : workers) {
            if (worker.position) result += worker.position.value().history_storage_bytes();
            result += worker.history.history_storage_bytes();
        }
        return result;
    }
    [[nodiscard]] std::size_t charge_without_tree() const {
        std::size_t result = sizeof(SearchEngine) + sizeof(Impl) + 2 * kAllocationAllowance;
        for (const auto& slot : slots) {
            result += slot.requests.capacity() * sizeof(Request) +
                      slot.inputs.capacity() * sizeof(nn::InputPlanes) +
                      slot.outputs.capacity() * sizeof(nn::NNOutput) +
                      slot.rows.capacity() * sizeof(std::size_t) +
                      slot.legal_outputs.capacity() * sizeof(nn::LegalOutput) +
                      slot.policies.capacity() * sizeof(nn::PolicyIndices) +
                      slot.collisions.capacity() * sizeof(Collision) +
                      slot.collision_nodes.capacity() * sizeof(NodeId) +
                      slot.picked.capacity() * sizeof(std::size_t) +
                      slot.dag_paths.capacity() * sizeof(DagPath) + 10 * kAllocationAllowance;
        }
        const auto walk_charge = [](const PickWalk& walk) {
            return walk.levels.capacity() * sizeof(PickLevel) +
                   walk.entries.capacity() * sizeof(PickEntry) + 2 * kAllocationAllowance;
        };
        result += walk_charge(picker.main) + picker.free_rows.capacity() * sizeof(std::size_t) +
                  kAllocationAllowance;
        for (const auto& walk : picker.task_walks) result += sizeof(PickWalk) + walk_charge(walk);
        result += picker.tasks.capacity() * sizeof(PickTask) + 2 * kAllocationAllowance;
        result += pick_threads.capacity() * sizeof(std::thread) +
                  picker.task_walks.size() * kThreadAllowance;
        if (evaluator)
            result +=
                sizeof(nn::LegalEvaluator) + evaluator->scratch_bytes() + kAllocationAllowance;
        result += workers.capacity() * sizeof(Worker) + kAllocationAllowance;
        result += worker_threads.capacity() * sizeof(std::thread) + kAllocationAllowance;
        result += (helper_threads() + 1) * kThreadAllowance;
        result += history_charge();
        for (const auto* text : {&backend_identity.onnx_sha256, &backend_identity.provider,
                                 &backend_identity.precision, &backend_identity.device_name,
                                 &backend_identity.device_uuid, &backend_identity.driver_version,
                                 &backend_identity.cuda_version, &backend_identity.tensorrt_version,
                                 &backend_identity.cache_namespace, &metadata.converter_version,
                                 &metadata.source_sha256, &metadata.precision})
            result += text->capacity() + 1 + kAllocationAllowance;
        return result;
    }
    void initialize_scratch() {
        for (auto& slot : slots) {
            slot.requests = std::vector<Request>(config.maximum_batch);
            slot.collisions.resize(config.gather.max_collision_events);
            slot.collision_nodes.resize(config.gather.max_collision_events * config.max_depth);
            slot.inputs = std::vector<nn::InputPlanes>(config.maximum_batch);
            if (!evaluator) slot.outputs = std::vector<nn::NNOutput>(config.maximum_batch);
            slot.rows = std::vector<std::size_t>(config.maximum_batch);
            slot.picked = std::vector<std::size_t>(config.maximum_batch);
            if (evaluator) {
                slot.legal_outputs.resize(config.maximum_batch);
                slot.policies.resize(config.maximum_batch);
            }
            if (config.dag) slot.dag_paths.resize(config.maximum_batch);
        }
        // A pick walk descends at most max_depth edges below the root.
        const auto size_walk = [&](PickWalk& walk) {
            walk.levels = std::vector<PickLevel>(config.max_depth + 1);
            walk.entries = std::vector<PickEntry>(config.maximum_batch + config.max_depth + 1);
        };
        size_walk(picker.main);
        picker.free_rows = std::vector<std::size_t>(config.maximum_batch);
        // Lc0 PickNodesToExtendTask hands subtrees to TaskWorkers threads. Its dag_classic holds
        // one lock across every task, so the DAG search picks alone here as well.
        if (!config.deterministic && !config.dag && config.threads != 0) {
            picker.task_walks = std::vector<PickWalk>(config.threads + 1);
            for (auto& walk : picker.task_walks) size_walk(walk);
            picker.tasks = std::vector<PickTask>(kMaximumPickTasks);
            pick_threads.reserve(config.threads);
        }
        // The last scratch serves the coordinator, which works through rows alongside the
        // worker threads instead of waiting for them (not in deterministic mode).
        workers.resize(config.deterministic ? 1 : config.threads + 1);
        worker_threads.reserve(helper_threads());
        overhead = charge_without_tree();
    }
    [[nodiscard]] Result<void> prepare_histories() {
        if (!root) return tl::unexpected(std::string{"Search history setup requires root"});
        auto& current_root = *root;
        auto reserved = current_root.reserve_history(config.max_depth + 1);
        if (!reserved) return reserved;
        reserved = root_history.reserve_history(config.max_depth + 1);
        if (!reserved) return reserved;
        for (auto& worker : workers) {
            worker.position = current_root;
            worker.history = root_history;
            reserved = worker.position.value().reserve_history(config.max_depth + 1);
            if (!reserved) return reserved;
            reserved = worker.history.reserve_history(config.max_depth + 1);
            if (!reserved) return reserved;
        }
        overhead = charge_without_tree();
        return {};
    }
    void clear_root() {
        tree.reset();
        root.reset();
        root_history = {};
        for (auto& worker : workers) {
            worker.position.reset();
            worker.history = {};
        }
        inherited = 0;
        seldepth = 0;
        overhead = charge_without_tree();
    }
    [[nodiscard]] Result<void> resize_tree() {
        overhead = charge_without_tree();
        if (overhead >= config.ram_limit_bytes)
            return tl::unexpected(
                std::string{"Search RAM limit cannot fit worker histories and pipeline"});
        const auto available = config.ram_limit_bytes - overhead;
        // Keep history growth headroom instead of consuming the entire budget.
        auto fresh = Tree::create(available - available / 8, config.dag);
        if (!fresh) return tl::unexpected(fresh.error());
        tree = std::move(*fresh);
        return {};
    }
    [[nodiscard]] Result<bool> invalidate_root_terminal() {
        if (tree->node(0).state != NodeState::Terminal) return false;
        if (!root.has_value()) return tl::unexpected(std::string{"Search has no root"});
        const auto status = classify_leaf(*root, attacks, {}, LeafContext::Root);
        if (!status) return tl::unexpected(status.error());
        if (status->reason != TerminalReason::None) return false;
        // A rule-draw leaf loses its terminal meaning when it becomes a legal
        // root. Clear before exposing snapshot visits: callers use those visits
        // to decide whether mandatory root NN priming is already complete.
        const auto reset = tree->reset();
        if (!reset) return tl::unexpected(reset.error());
        inherited = 0;
        seldepth = 0;
        return true;
    }
    void cancel_collisions_locked(Slot& slot) noexcept {
        // lc0 classic/search.cc CancelSharedCollisions: cancel ancestors only;
        // no collided visit becomes an evaluation, value backup or real visit.
        for (std::size_t i = 0; i < slot.collision_count; ++i) {
            const auto& collision = slot.collisions.at(i);
            for (std::size_t j = 0; j < collision.size; ++j) {
                auto& node = tree->node(slot.collision_nodes.at(collision.offset + j));
                assert(node.in_flight >= collision.visits);
                node.in_flight -= collision.visits;
            }
        }
        slot.collision_count = 0;
        slot.collision_nodes_used = 0;
        slot.collision_root_visits = 0;
    }
    // lc0 GatherMinibatch reverts the collisions of a pick round that saw an
    // out-of-order result. They no longer spend this batch's collision budget.
    // A completed batch may already have cancelled them (publish_batch).
    void cancel_round_collisions_locked(Slot& slot, std::size_t first,
                                        std::size_t first_node) noexcept {
        if (first >= slot.collision_count) return;
        for (std::size_t i = first; i < slot.collision_count; ++i) {
            const auto& collision = slot.collisions.at(i);
            for (std::size_t j = 0; j < collision.size; ++j) {
                auto& node = tree->node(slot.collision_nodes.at(collision.offset + j));
                assert(node.in_flight >= collision.visits);
                node.in_flight -= collision.visits;
            }
            slot.collision_visits -= collision.visits;
            if (collision.size != 0) slot.collision_root_visits -= collision.visits;
        }
        slot.collision_nodes_used = first_node;
        slot.collision_count = first;
    }
    void release_slot_locked(Slot& slot) {
        cancel_collisions_locked(slot);
        for (std::size_t i = 0; i < config.maximum_batch; ++i) {
            auto& request = slot.requests[i];
            request.reservation.release_locked();
            request.ready = false;
            request.deferred = false;
        }
        slot.count = 0;
        slot.prefetched = 0;
    }
    [[nodiscard]] bool allowed(chess::Move move) const {
        const auto count = limits.searchmoves_count;
        return count == 0 ||
               std::find(limits.searchmoves.begin(),
                         limits.searchmoves.begin() + static_cast<std::ptrdiff_t>(count),
                         move) != limits.searchmoves.begin() + static_cast<std::ptrdiff_t>(count);
    }
    void release_slot(Slot& slot) {
        const std::lock_guard lock(tree_mutex);
        release_slot_locked(slot);
    }
    [[nodiscard]] std::size_t ceiling() const { return limits.depth.value_or(config.max_depth); }
    bool should_stop_locked() {
        if (stopped) return true;
        if (external_stop.load(std::memory_order_relaxed))
            stopped = StopReason::ExternalStop;
        else if (deadline && Clock::now() >= *deadline)
            stopped = StopReason::TimeLimit;
        return stopped.has_value();
    }
    [[nodiscard]] std::size_t out_of_order_limit() const noexcept {
        // lc0 classic/params.cc MaxOutOfOrderEvalsFactor; search.h converts the
        // scaled minibatch size to an integer by truncation, and keeps at least one.
        return std::max(
            std::size_t{1},
            static_cast<std::size_t>(
                static_cast<float>(config.gather.max_out_of_order_factor) *
                static_cast<float>(limits.maximum_batch.value_or(config.maximum_batch))));
    }
    // Lc0 Search::GetDrawScore: root depth is even; the value favours white's own view.
    [[nodiscard]] double draw_score_at(std::size_t depth) const noexcept {
        return depth % 2 == 0 ? even_draw_score : -even_draw_score;
    }
    // How a picked node is handled. Mouse states map to lc0's checks in
    // PickNodesToExtendTask: Pending is an in-flight leaf (N == 0 with visits
    // started), Unexpanded a new leaf, Known a terminal or depth-limited leaf
    // whose stored value is backed up again, Interior an expanded node.
    enum class PickKind : std::uint8_t { Collision, Fresh, Known, Interior };
    struct PickLocation {
        NodeId id;
        std::size_t depth;
    };
    // Lc0 dag_classic GetTotalVisits: exploration counts every visit to the position, whichever
    // path brought it.
    [[nodiscard]] std::uint64_t selection_visits(NodeId id) const {
        const auto low_id = tree->low_of(id);
        return low_id == kNoLow ? tree->node(id).visits : tree->low(low_id).visits;
    }
    // Lc0 dag_classic SearchWorker::GetRepetitions, for a position `depth` plies below the root.
    [[nodiscard]] std::pair<std::uint8_t, std::uint16_t> dag_repetitions(
        const chess::Position& position, std::size_t depth) const {
        const auto found = position.repetitions();
        if (found.count == 0) return {0, 0};
        if (found.count >= 2) return {static_cast<std::uint8_t>(std::min(found.count, 255U)), 0};
        if (config.two_fold_draws && depth >= 4 && depth >= found.plies_since_previous)
            return {1, static_cast<std::uint16_t>(found.plies_since_previous)};
        return {0, 0};
    }
    // Lc0 PositionHistory::HashLast over Position::Hash, which also covers repetitions. Earlier
    // positions contribute their key alone: Mouse keeps no repetition count for them.
    [[nodiscard]] std::uint64_t low_key(const chess::Position& position) const {
        const auto wanted = std::size_t{config.cache_history_length} + 1;
        std::uint64_t hash = wanted;
        const auto keys = position.key_history();
        hash = hash_cat(hash, hash_cat(position.key(), position.repetitions().count));
        for (std::size_t back = 1; back < wanted && back < keys.size(); ++back)
            hash = hash_cat(hash, keys[keys.size() - 1 - back]);
        return hash_cat(hash, position.halfmove_clock());
    }
    // Lc0 dag_classic SearchWorker::ShouldStopPickingHere, for an expanded node below the root.
    [[nodiscard]] bool dag_stop_here(NodeId id, std::uint8_t repetitions) const {
        constexpr double kWlDifferenceLimit = 0.01, kDrawDifferenceLimit = 0.01;
        constexpr double kMovesDifferenceLimit = 2.0;
        if (repetitions >= 2) return true;
        const auto low_id = tree->low_of(id);
        if (low_id == kNoLow) return false;
        const auto& shared = tree->low(low_id);
        if (!shared.transposition) return false;
        const auto& node = tree->node(id);
        if (shared.proof.outcome != ProofOutcome::Unknown && shared.proof != node.proof) return true;
        if (shared.visits == 0 || node.visits == 0) return false;
        const auto low_visits = static_cast<double>(shared.visits);
        const auto node_visits = static_cast<double>(node.visits);
        if (std::abs(shared.value_sum / low_visits - node.value_sum / node_visits) >=
            kWlDifferenceLimit)
            return true;
        if (std::abs(shared.draw_sum / low_visits - node.draw_sum / node_visits) >=
            kDrawDifferenceLimit)
            return true;
        if (shared.moves_left_samples == shared.visits && node.moves_left_samples == node.visits &&
            std::abs(shared.moves_left_sum / low_visits + 1.0 - node.moves_left_sum / node_visits) >=
                kMovesDifferenceLimit)
            return true;
        return false;
    }
    PickKind classify_locked(PickWalk& walk, PickLocation location,
                             std::uint8_t repetitions = 0) {
        const auto [id, depth] = location;
        auto& node = tree->node(id);
        // lc0 fd71a2d classic/search.cc EnsureNodeTwoFoldCorrectForDepth: a
        // reused twofold whose earlier occurrence is now above the root is no
        // longer a draw. walk.path starts with root..parent. One with visits
        // in flight stays terminal until a later pick reaches it quiescent.
        if (node.state == NodeState::Terminal && node.twofold && node.in_flight == 0 &&
            static_cast<double>(depth) < *node.leaf_moves_left) {
            const std::lock_guard lock(picker.shared_state);
            const auto reverted =
                tree->revert_twofold(std::span{walk.path.nodes}.first(depth), id);
            if (!reverted) throw std::logic_error(reverted.error());
            inherited -= std::min(inherited, *reverted);
            if (limits.log_search_stats) ++diagnostics.twofold_reverts;
        }
        if (node.state == NodeState::Pending) return PickKind::Collision;
        if (node.state == NodeState::Frontier && depth < ceiling())
            node.state = NodeState::Unexpanded;
        if (node.state == NodeState::Unexpanded) return PickKind::Fresh;
        if (node.state == NodeState::Terminal || node.state == NodeState::Frontier ||
            depth == ceiling())
            return PickKind::Known;
        if (config.dag) {
            // A table hit whose first backup waits for its batch has no value to pick from yet.
            if (node.visits == 0) return PickKind::Collision;
            if (depth != 0 && dag_stop_here(id, repetitions)) return PickKind::Known;
        }
        return PickKind::Interior;
    }
    // Reserves one real visit along walk.path (root..leaf) in a free row. The caller has taken
    // it from physical_left. Rows and picked entries share one index, so `picked` needs no lock.
    void start_request_locked(Slot& slot, PickWalk& walk, bool known) {
        const auto& path = walk.path;
        for (std::size_t i = walk.base_depth; i < path.size; ++i)
            if (tree->node(path.nodes.at(i)).in_flight == std::numeric_limits<std::uint32_t>::max())
                throw std::overflow_error("Search in-flight counter exhausted");
        const auto index = picker.free_used.fetch_add(1, std::memory_order_relaxed);
        const auto row = picker.free_rows.at(index);
        auto& request = slot.requests[row];
        request.path.size = path.size;
        std::copy_n(path.nodes.begin(), path.size, request.path.nodes.begin());
        std::copy_n(path.moves.begin(), path.size - 1, request.path.moves.begin());
        request.known = known;
        request.ready = false;
        auto& reservation = request.reservation;
        reservation.tree = tree.get();
        reservation.mutex = &tree_mutex;
        reservation.path = &request.path;
        reservation.active = true;
        reservation.owns_pending = !known;
        for (std::size_t i = walk.base_depth; i < path.size; ++i)
            ++tree->node(path.nodes.at(i)).in_flight;
        if (walk.base_depth != 0) ++walk.ancestor_visits;
        if (!known) tree->node(path.nodes.at(path.size - 1)).state = NodeState::Pending;
        if (config.dag) {
            auto& stored = slot.dag_paths.at(row);
            std::copy_n(walk.dag.repetitions.begin(), path.size, stored.repetitions.begin());
            std::copy_n(walk.dag.cycles.begin(), path.size, stored.cycles.begin());
        }
        slot.picked.at(index) = row;
        ++walk.completed;
    }
    [[nodiscard]] bool take_physical() noexcept {
        auto left = picker.physical_left.load(std::memory_order_relaxed);
        while (left != 0)
            if (picker.physical_left.compare_exchange_weak(left, left - 1,
                                                           std::memory_order_relaxed))
                return true;
        return false;
    }
    // lc0 NodeToProcess::Collision: `visits` are reserved on the leaf's
    // ancestors only (the first `ancestors` nodes of walk.path), never
    // evaluated, and cancelled after the batch backs up.
    void add_collision_locked(Slot& slot, PickWalk& walk, std::size_t ancestors,
                              std::uint32_t visits) {
        if (visits == 0) return;
        const auto& path = walk.path;
        walk.completed += visits;
        // Before anything is claimed: a counted record has to be a filled one.
        for (std::size_t i = walk.base_depth; i < ancestors; ++i)
            if (visits >
                std::numeric_limits<std::uint32_t>::max() - tree->node(path.nodes.at(i)).in_flight)
                throw std::overflow_error("Collision virtual-visit counter exhausted");
        // Tasks share these counters. Each walk fills the record and the nodes it reserved.
        std::atomic_ref<std::size_t> records(slot.collision_count);
        const auto reserved = reserve_range(std::atomic_ref<std::size_t>(slot.collision_nodes_used),
                                            ancestors, slot.collision_nodes.size());
        if (!reserved) {
            // Bounded record storage (lc0 MaxCollisionEvents) ends this gather.
            slot.halt.store(true, std::memory_order_relaxed);
            return;
        }
        const auto offset = *reserved;
        const auto record = records.fetch_add(1, std::memory_order_relaxed);
        if (record >= slot.collisions.size()) {
            records.fetch_sub(1, std::memory_order_relaxed);
            slot.halt.store(true, std::memory_order_relaxed);
            return;
        }
        for (std::size_t i = 0; i < ancestors; ++i) {
            if (i >= walk.base_depth) tree->node(path.nodes.at(i)).in_flight += visits;
            slot.collision_nodes.at(offset + i) = path.nodes.at(i);
        }
        if (walk.base_depth != 0) walk.ancestor_visits += visits;
        if (visits == walk.walk_limit && walk.max_limit > visits) {
            walk.upsize_record = record;
            walk.upsize_cap = walk.max_limit;
        }
        slot.collisions.at(record) = {.offset = offset, .size = ancestors, .visits = visits};
        std::atomic_ref<std::uint32_t>(slot.collision_visits)
            .fetch_add(visits, std::memory_order_relaxed);
        if (ancestors != 0)
            std::atomic_ref<std::uint32_t>(slot.collision_root_visits)
                .fetch_add(visits, std::memory_order_relaxed);
        if (limits.log_search_stats) ++walk.collisions;
    }
    // Starts the one leaf visit lc0's TryStartScoreUpdate allows for `child`
    // (walk.path ends at its parent). Returns whether a request started.
    bool start_leaf_locked(Slot& slot, PickWalk& walk, std::size_t depth, Edge& edge,
                           PickKind kind) {
        if (kind != PickKind::Fresh && kind != PickKind::Known) return false;
        if (!take_physical()) return false;
        if (edge.child == kNoNode) {
            const std::lock_guard lock(picker.allocation);
            auto allocated = tree->allocate_node();
            if (!allocated) {
                picker.physical_left.fetch_add(1, std::memory_order_relaxed);
                stopped = StopReason::MemoryLimit;
                slot.halt.store(true, std::memory_order_relaxed);
                return false;
            }
            edge.child = *allocated;
        }
        auto& path = walk.path;
        path.nodes.at(depth + 1) = edge.child;
        path.moves.at(depth) = edge.move;
        path.size = depth + 2;
        start_request_locked(slot, walk, kind == PickKind::Known);
        path.size = depth + 1;
        return true;
    }
    // lc0 fd71a2d classic/search.cc PickNodesToExtendTask: give `limit` visits
    // to the edges of an expanded node (walk.path ends at it) in rounds. Each
    // round the best PUCT edge receives the visits it needs before the
    // runner-up overtakes it. An expanded child's started count grows by all of
    // them; a fresh or known leaf starts one real visit now, and what remains is
    // placed below (an expanded child) or becomes a collision (a leaf).
    struct PickDistribution {
        std::size_t depth;
        std::uint32_t limit;
        std::size_t first_entry;
    };
    // lc0 fd71a2d classic/search.cc PickNodesToExtendTask (1738-1747): a root edge that
    // cannot reach the current best edge's visits with the remaining playouts is not
    // considered, except the current best edge itself. FPU's visited prior mass still
    // counts it, as lc0 computes that before this check.
    [[nodiscard]] bool can_catch_up(std::size_t index, const Node* child) const noexcept {
        if (current_best == index) return true;
        const auto visits = child ? child->visits : 0;
        return picker.best_visits <= visits ||
               picker.best_visits - visits <= picker.remaining_playouts;
    }
    // DAG picking keeps the position of the node it stands on in the coordinator's scratch.
    [[nodiscard]] chess::Position& dag_position() {
        return *workers.back().position;
    }
    void distribute_locked(Slot& slot, PickWalk& walk, PickLevel& level,
                           PickDistribution distribution) {
        auto [depth, limit, first_entry] = distribution;
        const auto id = level.node;
        const auto& node = tree->node(id);
        const auto edges = tree->edges(id);
        if (edges.size() > kMaximumLegalMoves)
            throw std::logic_error("Search node has more edges than legal chess moves");
        const bool at_root = depth == 0;
        const double draw_score = draw_score_at(depth);
        const double parent_q =
            node.visits == 0
                ? 0
                : (node.value_sum + draw_score * node.draw_sum) / static_cast<double>(node.visits);
        double mass = 0;
        for (const auto& edge : edges)
            if ((!at_root || allowed(edge.move)) && edge.child != kNoNode &&
                tree->node(edge.child).visits != 0)
                mass += decode_prior(edge.prior_bits);
        const double fpu = first_play_urgency(config.selection, at_root, parent_q, mass);
        const double multiplier = exploration_multiplier(config.selection, selection_visits(id));
        const auto terms = std::span{walk.terms}.first(edges.size());
        for (std::size_t index = 0; index < edges.size(); ++index) {
            const auto& edge = edges[index];
            const Node* child = edge.child == kNoNode ? nullptr : &tree->node(edge.child);
            double utility = child && child->visits != 0
                                 ? (draw_score * child->draw_sum - child->value_sum) /
                                       static_cast<double>(child->visits)
                                 : fpu;
            std::optional<double> proof_utility;
            if (config.terminal_proof && child &&
                (node.proof.outcome == ProofOutcome::Win ||
                 node.proof.outcome == ProofOutcome::Loss)) {
                const auto value = proof_selection_utility(node.proof, child->proof);
                if (!value) throw std::logic_error(value.error());
                proof_utility = *value;
            }
            if (proof_utility)
                utility += *proof_utility;
            else if (config.moves_left_utility && child)
                utility += moves_left_utility(node, *child, config.moves_left);
            // Pending children remain PUCT candidates (lc0 classic/node.h
            // EdgeAndNode::GetU); reservations affect only NStarted.
            terms[index] = {
                .utility = utility,
                .exploration = multiplier * decode_prior(edge.prior_bits),
                .started = child ? child->visits + child->in_flight : 0,
                .entry = std::numeric_limits<std::uint16_t>::max(),
                .eligible = !at_root || (allowed(edge.move) && can_catch_up(index, child))};
        }
        level.begin = level.end = level.next = first_entry;
        std::uint64_t rounds = 0;
        while (limit != 0 && !slot.halt.load(std::memory_order_relaxed)) {
            std::size_t chosen = edges.size();
            double best = -std::numeric_limits<double>::infinity(), second_best = best;
            for (std::size_t index = 0; index < terms.size(); ++index) {
                const auto& term = terms[index];
                if (!term.eligible) continue;
                const auto score =
                    term.utility + term.exploration / (1.0 + static_cast<double>(term.started));
                if (chosen == edges.size() || score > best) {
                    second_best = best;
                    chosen = index;
                    best = score;
                } else
                    second_best = std::max(second_best, score);
            }
            if (chosen == edges.size()) break;
            std::uint32_t granted = limit;
            if (std::isfinite(second_best)) {
                const CollisionSelection selection{.utility = terms[chosen].utility,
                                                   .exploration = terms[chosen].exploration,
                                                   .started = terms[chosen].started,
                                                   .second_best = second_best};
                walk.max_limit = std::min<std::uint64_t>(
                    walk.max_limit,
                    collision_multiplicity(selection, std::numeric_limits<std::uint32_t>::max()));
                granted = collision_multiplicity(selection, limit);
            }
            limit -= granted;
            ++rounds;
            auto& edge = edges[chosen];
            if (config.dag) resolve_repetitions(terms[chosen], edge, depth);
            const auto kind =
                edge.child == kNoNode
                    ? PickKind::Fresh
                    : classify_locked(walk, {.id = edge.child, .depth = depth + 1},
                                      terms[chosen].repetitions);
            if (kind == PickKind::Interior) {
                terms[chosen].started += granted;
            } else {
                if (config.dag) {
                    walk.dag.repetitions.at(depth + 1) = terms[chosen].repetitions;
                    walk.dag.cycles.at(depth + 1) = terms[chosen].cycle;
                }
                if (start_leaf_locked(slot, walk, depth, edge, kind)) {
                    ++terms[chosen].started;
                    --granted;
                }
            }
            if (granted == 0) continue;
            auto& slot_index = terms[chosen].entry;
            if (slot_index == std::numeric_limits<std::uint16_t>::max()) {
                if (level.end == walk.entries.size())
                    throw std::logic_error("Search pick walk exceeded its entry bound");
                slot_index = static_cast<std::uint16_t>(level.end - level.begin);
                walk.entries[level.end++] = {.index = static_cast<std::uint16_t>(chosen),
                                             .visits = 0};
            }
            walk.entries[level.begin + slot_index].visits += granted;
        }
        // Place visits in edge order, as lc0 walks visits_to_perform.
        std::sort(walk.entries.begin() + static_cast<std::ptrdiff_t>(level.begin),
                  walk.entries.begin() + static_cast<std::ptrdiff_t>(level.end),
                  [](const PickEntry& a, const PickEntry& b) { return a.index < b.index; });
        // Rounds after the first reuse this node's PUCT terms without recomputing them.
        if (limits.log_search_stats && rounds > 1) walk.selection_hits += rounds - 1;
    }
    void resolve_repetitions(PickWalk::EdgeTerms& term, const Edge& edge, std::size_t depth) {
        if (term.repetitions_known) return;
        auto& position = dag_position();
        const auto undo = position.make_move(edge.move, attacks);
        const auto [repetitions, cycle] = dag_repetitions(position, depth + 1);
        position.unmake_move(edge.move, undo);
        term.repetitions = repetitions;
        term.cycle = cycle;
        term.repetitions_known = true;
    }
    // Lc0 PickNodesToExtendTask "actively do any splits now": a child taking enough of what is
    // left goes to another thread, unless too little would remain here.
    void split_locked(PickWalk& walk, PickLevel& level, std::size_t depth) {
        if (picker.tasks.empty()) return;
        const auto edges = tree->edges(level.node);
        const auto minimum = static_cast<std::int64_t>(config.gather.minimum_picking_work);
        const auto remainder =
            static_cast<std::int64_t>(config.gather.minimum_remaining_picking_work);
        for (std::size_t at = level.begin; at != level.end; ++at) {
            auto& entry = walk.entries[at];
            const auto visits = static_cast<std::int64_t>(entry.visits);
            const auto limit = static_cast<std::int64_t>(walk.task_limit);
            const auto spent = static_cast<std::int64_t>(walk.passed_off) +
                               static_cast<std::int64_t>(walk.completed);
            if (visits <= minimum || visits >= (limit - spent) * 2 / 3 ||
                visits + spent >= limit - remainder)
                continue;
            const auto& edge = edges[entry.index];
            // Don't split if not expanded or terminal.
            if (edge.child == kNoNode) continue;
            walk.path.nodes.at(depth + 1) = edge.child;
            walk.path.moves.at(depth) = edge.move;
            walk.path.size = depth + 2;
            if (classify_locked(walk, {.id = edge.child, .depth = depth + 1}) != PickKind::Interior)
                continue;
            {
                const std::lock_guard lock(picker.tasks_mutex);
                const auto index = picker.queue.size();
                if (index == picker.tasks.size()) return;
                auto& task = picker.tasks[index];
                task.path.size = walk.path.size;
                std::copy_n(walk.path.nodes.begin(), walk.path.size, task.path.nodes.begin());
                std::copy_n(walk.path.moves.begin(), walk.path.size - 1, task.path.moves.begin());
                task.limit = entry.visits;
                task.ancestor_visits = 0;
                picker.queue.publish(index + 1);
                if (index == 0) {
                    picker.wake.fetch_add(1, std::memory_order_release);
                    picker.wake.notify_all();
                }
            }
            walk.passed_off += entry.visits;
            entry.visits = 0;
        }
    }
    // Places the visits of an expanded node on walk.path (lc0 PickNodesToExtendTask's main loop).
    void walk_below_locked(Slot& slot, PickWalk& walk, std::size_t start, std::uint32_t limit) {
        auto& path = walk.path;
        walk.levels.at(start).node = path.nodes.at(start);
        distribute_locked(slot, walk, walk.levels.at(start),
                          {.depth = start, .limit = limit, .first_entry = 0});
        split_locked(walk, walk.levels.at(start), start);
        std::size_t depth = start;
        auto& undo = workers.back().undo;
        for (;;) {
            auto& level = walk.levels.at(depth);
            const auto edges = tree->edges(level.node);
            bool descended = false;
            while (level.next < level.end && !slot.halt.load(std::memory_order_relaxed)) {
                const auto entry = walk.entries[level.next++];
                const auto visits = entry.visits;
                if (visits == 0) continue;
                const auto& edge = edges[entry.index];
                path.size = depth + 1;
                if (edge.child == kNoNode) {
                    // A fresh leaf that could not start (node allowance spent)
                    // has no node yet; its remainder is held on the path to it.
                    add_collision_locked(slot, walk, depth + 1, visits);
                    continue;
                }
                path.nodes.at(depth + 1) = edge.child;
                path.moves.at(depth) = edge.move;
                path.size = depth + 2;
                std::uint8_t repetitions = 0;
                if (config.dag) {
                    PickWalk::EdgeTerms term;
                    resolve_repetitions(term, edge, depth);
                    repetitions = term.repetitions;
                    walk.dag.repetitions.at(depth + 1) = term.repetitions;
                    walk.dag.cycles.at(depth + 1) = term.cycle;
                }
                if (classify_locked(walk, {.id = edge.child, .depth = depth + 1}, repetitions) !=
                    PickKind::Interior) {
                    add_collision_locked(slot, walk, depth + 1, visits);
                    continue;
                }
                if (config.dag) undo.at(depth) = dag_position().make_move(edge.move, attacks);
                auto& next = walk.levels.at(depth + 1);
                next.node = edge.child;
                distribute_locked(slot, walk, next,
                                  {.depth = depth + 1, .limit = visits, .first_entry = level.end});
                split_locked(walk, next, depth + 1);
                ++depth;
                descended = true;
                break;
            }
            if (descended) continue;
            if (depth == start) return;
            --depth;
            if (config.dag) dag_position().unmake_move(path.moves.at(depth), undo.at(depth));
        }
    }
    void run_pick_task(Slot& slot, PickWalk& walk, PickTask& task) {
        const auto size = task.path.size;
        walk.path.size = size;
        std::copy_n(task.path.nodes.begin(), size, walk.path.nodes.begin());
        std::copy_n(task.path.moves.begin(), size - 1, walk.path.moves.begin());
        walk.base_depth = size - 1;
        walk.ancestor_visits = 0;
        walk.task_limit = task.limit;
        walk.passed_off = 0;
        walk.completed = 0;
        // Only the coordinator's own walk upsizes a collision (lc0 tests base_depth == 0).
        walk.walk_limit = 0;
        walk.max_limit = std::numeric_limits<std::uint64_t>::max();
        walk.collisions = 0;
        walk.selection_hits = 0;
        try {
            walk_below_locked(slot, walk, size - 1, task.limit);
        } catch (const std::exception& error) {
            slot.halt.store(true, std::memory_order_relaxed);
            const std::lock_guard lock(picker.tasks_mutex);
            if (picker.error.empty()) picker.error = error.what();
        }
        task.ancestor_visits = walk.ancestor_visits;
        if (limits.log_search_stats) {
            const std::lock_guard lock(picker.shared_state);
            diagnostics.collisions += walk.collisions;
            diagnostics.selection_hits += walk.selection_hits;
        }
    }
    [[nodiscard]] bool run_one_pick_task(PickWalk& walk) {
        const auto ticket = picker.queue.peek();
        if (!ticket) return false;
        const auto index = picker.queue.claim(*ticket);
        if (!index) return true;
        run_pick_task(*picker.slot, walk, picker.tasks[*index]);
        picker.queue.complete();
        return true;
    }
    // Lc0 SearchWorker::RunTasks, for picking only: asleep until a pick hands out its first task.
    // A task never waits on a lock the coordinator holds, as it holds tree_mutex for them all.
    void pick_loop(std::size_t index) {
        std::uint64_t seen = 0;
        auto& walk = picker.task_walks.at(index);
        for (;;) {
            picker.wake.wait(seen, std::memory_order_acquire);
            seen = picker.wake.load(std::memory_order_acquire);
            if (picker.shutdown.load(std::memory_order_acquire)) return;
            while (!picker.closed.load(std::memory_order_acquire))
                if (!run_one_pick_task(walk)) spin_pause();
        }
    }
    // One walk places up to `limit` visits (lc0 PickNodesToExtend). Picking runs under tree_mutex;
    // request processing happens afterwards, unlocked.
    void pick_locked(Slot& slot, std::uint32_t limit) {
        auto& walk = picker.main;
        auto& path = walk.path;
        path.size = 1;
        path.nodes[0] = 0;
        walk.base_depth = 0;
        walk.ancestor_visits = 0;
        walk.task_limit = limit;
        walk.passed_off = 0;
        walk.completed = 0;
        walk.walk_limit = limit;
        walk.max_limit = std::numeric_limits<std::uint64_t>::max();
        walk.upsize_record = slot.collisions.size();
        walk.upsize_cap = 0;
        walk.collisions = 0;
        walk.selection_hits = 0;
        picker.best_visits = 0;
        if (current_best) {
            const auto child = tree->edges(0)[*current_best].child;
            if (child != kNoNode) picker.best_visits = tree->node(child).visits;
        }
        if (config.dag) {
            walk.dag.repetitions[0] = 0;
            walk.dag.cycles[0] = 0;
        }
        const auto root_kind = classify_locked(walk, {.id = 0, .depth = 0});
        if (root_kind != PickKind::Interior) {
            // The root has no parent round, so it starts its own leaf visit.
            if ((root_kind == PickKind::Fresh || root_kind == PickKind::Known) && take_physical()) {
                start_request_locked(slot, walk, root_kind == PickKind::Known);
                --limit;
            }
            add_collision_locked(slot, walk, 0, limit);
            finish_walk_stats(walk);
            return;
        }
        const bool parallel = !picker.tasks.empty();
        if (parallel) {
            picker.slot = &slot;
            picker.queue.reset();
            picker.closed.store(false, std::memory_order_release);
        }
        std::exception_ptr failed;
        try {
            walk_below_locked(slot, walk, 0, limit);
        } catch (...) {
            failed = std::current_exception();
        }
        if (config.dag) unwind_dag_position();
        finish_walk_stats(walk);
        if (parallel) {
            auto& spare = picker.task_walks.back();
            for (;;) {
                if (run_one_pick_task(spare)) continue;
                if (picker.queue.drained()) break;
                spin_pause();
            }
            picker.closed.store(true, std::memory_order_release);
            // Nothing reads a node's count while another walk works below it, so what each task
            // owes the nodes above its subtree can wait until they have all finished.
            const auto added = picker.queue.size();
            for (std::size_t t = 0; t < added; ++t) {
                const auto& task = picker.tasks[t];
                for (std::size_t i = 0; i + 1 < task.path.size; ++i) {
                    auto& node = tree->node(task.path.nodes.at(i));
                    if (task.ancestor_visits >
                        std::numeric_limits<std::uint32_t>::max() - node.in_flight)
                        throw std::overflow_error("Search in-flight counter exhausted");
                    node.in_flight += static_cast<std::uint32_t>(task.ancestor_visits);
                }
            }
            if (!picker.error.empty()) {
                const auto text = std::move(picker.error);
                picker.error.clear();
                throw std::logic_error(text);
            }
        }
        if (failed) std::rethrow_exception(failed);
    }
    // A walk stopped by a halt or an error can leave the position below the root.
    void unwind_dag_position() {
        auto& position = dag_position();
        if (position.history_size() != root->history_size()) position = *root;
    }
    void finish_walk_stats(const PickWalk& walk) {
        if (!limits.log_search_stats) return;
        diagnostics.collisions += walk.collisions;
        diagnostics.selection_hits += walk.selection_hits;
    }
    struct Value {
        double q = 0, draw = 0;
        std::optional<double> moves_left;
    };
    // A network evaluation as the tree stores it: side-to-move Q and D from the
    // normalized WDL, draw-rate rescaled as Lc0 does on fetching an evaluation
    // for computed and cached rows alike (fd71a2d classic/search.cc FetchSingleNodeResult).
    // `depth` is the leaf's distance from the root: contempt favours one colour, so its sign
    // follows the side to move at the leaf.
    [[nodiscard]] Value network_value(const std::array<float, 3>& wdl,
                                      std::optional<float> moves_left,
                                      std::size_t depth) const noexcept {
        const double sum = static_cast<double>(wdl[0]) + wdl[1] + wdl[2];
        Value value{.q = (static_cast<double>(wdl[0]) - wdl[2]) / sum,
                    .draw = wdl[1] / sum,
                    .moves_left = moves_left};
        const bool contempt = wdl_rescale.diff != 0.0F && limits.contempt_side != ContemptSide::None;
        if (wdl_rescale.ratio != 1.0F || contempt) {
            auto q = static_cast<float>(value.q);
            auto draw = static_cast<float>(value.draw);
            const bool root_holds_contempt =
                root && (limits.contempt_side == ContemptSide::White) ==
                            (root->side_to_move() == chess::Color::White);
            const float sign = root_holds_contempt == (depth % 2 == 0) ? 1.0F : -1.0F;
            if (rescale_wdl(q, draw,
                            {.ratio = wdl_rescale.ratio,
                             .diff = contempt ? wdl_rescale.diff : 0.0F,
                             .sign = sign,
                             .invert = false,
                             .max_s = static_cast<float>(config.draw_rate.max_s)})) {
                value.q = q;
                value.draw = draw;
            }
        }
        return value;
    }
    void refresh_proof_locked(NodeId id) {
        auto& node = tree->node(id);
        if (!config.terminal_proof || node.state != NodeState::Expanded) return;
        if (limits.log_search_stats) ++diagnostics.proof_scans;
        // Selection eligibility never restricts this complete legal edge set.
        // In particular unallocated children contribute Unknown, not an omitted
        // defense. The accumulator itself is bounded automatic storage.
        const auto edges = tree->edges(id);
        auto accumulator = ProofAccumulator::create(edges.size());
        if (!accumulator) throw std::logic_error(accumulator.error());
        for (const auto& edge : edges) {
            const auto added = accumulator->add(
                edge.child == kNoNode ? NodeProof{} : tree->node(edge.child).proof);
            if (!added) throw std::logic_error(added.error());
        }
        const auto result = accumulator->finish();
        if (!result) throw std::logic_error(result.error());
        node.proof = *result;
    }
    // A moved child with visits in flight is still named by the paths reserving them. They hold
    // it at its depth, so each is checked once. Paths are only read under tree_mutex.
    void relocate_paths_locked(NodeRelocation moved, std::size_t depth) {
        for (auto& slot : slots) {
            for (std::size_t row = 0; row < config.maximum_batch; ++row) {
                auto& request = slot.requests[row];
                if (request.reservation.active && request.path.size > depth &&
                    request.path.nodes.at(depth) == moved.from)
                    request.path.nodes.at(depth) = moved.to;
            }
            for (std::size_t i = 0; i < slot.collision_count; ++i) {
                const auto& collision = slot.collisions.at(i);
                if (collision.size > depth &&
                    slot.collision_nodes.at(collision.offset + depth) == moved.from)
                    slot.collision_nodes.at(collision.offset + depth) = moved.to;
            }
        }
    }
    // Lc0 DoBackupUpdateSingleNode: a node reaching SolidTreeThreshold becomes solid.
    void maybe_solidify_locked(NodeId id, std::size_t depth) {
        std::array<NodeRelocation, kMaximumLegalMoves> moved{};
        const auto count = tree->solidify(id, moved, batch_edges_reserved);
        if (!count) return;
        for (std::size_t i = 0; i < *count; ++i)
            if (tree->node(moved.at(i).to).in_flight != 0)
                relocate_paths_locked(moved.at(i), depth + 1);
    }
    void backup_locked(Request& request, Value value) {
        const auto began = limits.log_search_stats ? Clock::now() : Clock::time_point{};
        if (config.dag) {
            backup_dag_locked(request);
            if (limits.log_search_stats) diagnostics.backup += Clock::now() - began;
            return;
        }
        // Preflight before changing any ancestor, including the uint64 boundary.
        for (std::size_t i = 0; i < request.path.size; ++i)
            if (tree->node(request.path.nodes.at(i)).visits ==
                std::numeric_limits<std::uint64_t>::max())
                throw std::overflow_error("Search visit counter exhausted");
        // lc0 fd71a2d classic/search.cc DoBackupUpdateSingleNode propagates
        // bounds only when terminal information changes. NN samples cannot
        // change a proof. Mouse also propagates mate-bound refinements, stopping
        // once the complete parent proof is unchanged (not just its outcome).
        bool proof_changed =
            tree->node(request.path.nodes.at(request.path.size - 1)).proof.outcome !=
            ProofOutcome::Unknown;
        bool root_child_proof_changed = false;
        for (std::size_t i = request.path.size; i > 0; --i) {
            auto& node = tree->node(request.path.nodes.at(i - 1));
            ++node.visits;
            if (i != request.path.size && proof_changed) {
                const auto previous = node.proof;
                refresh_proof_locked(request.path.nodes.at(i - 1));
                proof_changed = node.proof != previous;
            }
            if (i == 2) root_child_proof_changed = proof_changed;
            if (node.proof.outcome != ProofOutcome::Unknown) {
                value.q = node.proof.outcome == ProofOutcome::Win
                              ? 1.0
                              : (node.proof.outcome == ProofOutcome::Loss ? -1.0 : 0.0);
                value.draw = node.proof.outcome == ProofOutcome::Draw ? 1.0 : 0.0;
                // Exact outcomes replace historical NN dilution at the existing
                // real visit count. Never synthesize visits or collapse a proven
                // expanded node: descendants must keep receiving real playouts.
                node.value_sum = value.q * static_cast<double>(node.visits);
                node.draw_sum = value.draw * static_cast<double>(node.visits);
            } else {
                node.value_sum += value.q;
                node.draw_sum += value.draw;
            }
            backup_moves_left_sample(node, value.moves_left);
            if (node.visits >= config.solid_tree_threshold && !node.solid &&
                node.state == NodeState::Expanded)
                maybe_solidify_locked(request.path.nodes.at(i - 1), i - 1);
            value.q = -value.q;
        }
        finish_backup_locked(request, root_child_proof_changed);
        if (limits.log_search_stats) diagnostics.backup += Clock::now() - began;
    }
    // lc0 DoBackupUpdateSingleNode (2266-2278): a root child that just became
    // terminal, or that is not the best edge but now has at least its visits, makes
    // the search re-pick its best edge.
    void finish_backup_locked(Request& request, bool root_child_proof_changed) {
        if (request.path.size >= 2) {
            const auto child = request.path.nodes.at(1);
            const auto root_edges = tree->edges(0);
            const auto best_child = current_best ? root_edges[*current_best].child : kNoNode;
            if (root_child_proof_changed ||
                (child != best_child && (best_child == kNoNode || tree->node(best_child).visits <=
                                                                      tree->node(child).visits))) {
                const auto* best = best_edge(0);
                if (best)
                    current_best = static_cast<std::size_t>(best - root_edges.data());
                else
                    current_best.reset();
            }
        }
        seldepth = std::max(seldepth, request.path.size - 1);
        if (limits.depth && request.path.size - 1 >= *limits.depth)
            stopped = StopReason::DepthLimit;
        request.reservation.release_locked();
    }
    // Lc0 fd71a2d dag_classic/search.cc DoBackupUpdateSingleNode. Node and low node values both
    // use the side to move at that position here, where lc0 keeps a node's from its parent's side,
    // so a value flips between a node and its parent's low node rather than within a level.
    struct DagValue {
        double q = 0, draw = 0;
        std::optional<double> moves_left;
        std::uint64_t to_fix = 0;
        double q_delta = 0, draw_delta = 0, moves_delta = 0;
    };
    static void set_proof_values(LowNode& shared) noexcept {
        const auto count = static_cast<double>(shared.visits);
        shared.value_sum = shared.proof.outcome == ProofOutcome::Win    ? count
                           : shared.proof.outcome == ProofOutcome::Loss ? -count
                                                                        : 0.0;
        shared.draw_sum = shared.proof.outcome == ProofOutcome::Draw ? count : 0.0;
    }
    // Lc0 MaybeAdjustForTerminalOrTransposition.
    bool dag_adjust_locked(NodeId id, DagValue& value) {
        auto& node = tree->node(id);
        if (node.state == NodeState::Terminal || node.state == NodeState::Frontier) {
            value.q = node.leaf_q;
            value.draw = node.leaf_draw;
            // Lc0 Node::MakeTerminal: a game's end is one ply from this edge's point of view.
            if (node.leaf_moves_left)
                value.moves_left = *node.leaf_moves_left == 0.0F ? 1.0 : *node.leaf_moves_left;
            else
                value.moves_left.reset();
            return true;
        }
        const auto low_id = tree->low_of(id);
        if (low_id == kNoLow) return false;
        const auto& shared = tree->low(low_id);
        const bool proven = shared.proof.outcome != ProofOutcome::Unknown;
        if (!shared.transposition && !proven) return false;
        // Whatever the position has learned by any path replaces what this edge had.
        const auto visits = static_cast<double>(shared.visits);
        value.q = shared.value_sum / visits;
        value.draw = shared.draw_sum / visits;
        value.moves_left.reset();
        if (shared.moves_left_samples == shared.visits)
            value.moves_left = shared.moves_left_sum / visits + 1.0;
        value.to_fix = node.visits;
        value.q_delta = node.visits == 0
                            ? 0.0
                            : value.q - node.value_sum / static_cast<double>(node.visits);
        value.draw_delta = node.visits == 0
                               ? 0.0
                               : value.draw - node.draw_sum / static_cast<double>(node.visits);
        value.moves_delta =
            value.moves_left && node.visits != 0 && node.moves_left_samples == node.visits
                ? *value.moves_left -
                      node.moves_left_sum / static_cast<double>(node.moves_left_samples)
                : 0.0;
        if (config.terminal_proof && proven) node.proof = shared.proof;
        return true;
    }
    static void dag_add(Node& node, const DagValue& value) noexcept {
        ++node.visits;
        node.value_sum += value.q;
        node.draw_sum += value.draw;
        if (value.moves_left) {
            node.moves_left_sum += *value.moves_left;
            ++node.moves_left_samples;
        }
    }
    static void dag_fix(Node& node, const DagValue& value) noexcept {
        const auto count = static_cast<double>(value.to_fix);
        node.value_sum += count * value.q_delta;
        node.draw_sum += count * value.draw_delta;
        if (node.moves_left_samples == node.visits)
            node.moves_left_sum += count * value.moves_delta;
    }
    void backup_dag_locked(Request& request) {
        const auto row = static_cast<std::size_t>(&request - slot_of(request).requests.data());
        const auto& dag = slot_of(request).dag_paths.at(row);
        const auto& path = request.path;
        for (std::size_t i = 0; i < path.size; ++i)
            if (tree->node(path.nodes.at(i)).visits == std::numeric_limits<std::uint64_t>::max())
                throw std::overflow_error("Search visit counter exhausted");
        auto last = path.size - 1;
        auto leaf_id = path.nodes.at(last);
        auto& leaf = tree->node(leaf_id);
        bool proof_changed = leaf.proof.outcome != ProofOutcome::Unknown && leaf.visits == 0;
        bool root_child_proof_changed = false;
        const auto leaf_low = tree->low_of(leaf_id);
        // The low node at the start of the backup path counts only its first visit.
        if (leaf_low != kNoLow && tree->low(leaf_low).visits == 0) {
            auto& shared = tree->low(leaf_low);
            shared.visits = 1;
            shared.value_sum = shared.leaf_q;
            shared.draw_sum = shared.leaf_draw;
            if (shared.leaf_moves_left) {
                shared.moves_left_sum = *shared.leaf_moves_left;
                shared.moves_left_samples = 1;
            }
            leaf.leaf_q = shared.leaf_q;
            leaf.leaf_draw = shared.leaf_draw;
            leaf.leaf_moves_left = shared.leaf_moves_left;
        }
        DagValue value;
        if (dag.repetitions.at(last) >= 2) {
            // A threefold has to count as a terminal to produce relevant results, unlike a
            // twofold, whose edge keeps updating its real value.
            leaf.repetition = true;
            value.q = 0.0;
            value.draw = 1.0;
            value.moves_left = 1.0;
        } else if (!dag_adjust_locked(leaf_id, value)) {
            // Nothing better: the original network values, for this edge.
            const auto& shared = tree->low(leaf_low);
            const auto visits = static_cast<double>(shared.visits);
            value.q = shared.value_sum / visits;
            value.draw = shared.draw_sum / visits;
            if (shared.moves_left_samples == shared.visits)
                value.moves_left = shared.moves_left_sum / visits + 1.0;
        }
        for (auto i = last;; --i) {
            const auto id = path.nodes.at(i);
            auto& node = tree->node(id);
            dag_add(node, value);
            if (value.to_fix != 0 && node.state != NodeState::Terminal) dag_fix(node, value);
            if (i == 1) root_child_proof_changed = proof_changed;
            // Stop adjusting at a repetition, and report a draw above any that holds on this path.
            // Only after updating the edge, so it has good values if play goes here.
            if (dag.repetitions.at(i) == 1 && node.state != NodeState::Terminal) {
                node.repetition = true;
                value.q = 0.0;
                value.draw = 1.0;
                value.moves_left = static_cast<double>(dag.cycles.at(i)) + 1.0;
            }
            if (node.repetition) value.to_fix = 0;
            if (i == 0) break;
            // Into the parent position's viewpoint.
            value.q = -value.q;
            value.q_delta = -value.q_delta;
            const auto parent_id = path.nodes.at(i - 1);
            const auto parent_low = tree->low_of(parent_id);
            auto& shared = tree->low(parent_low);
            if (shared.proof.outcome != ProofOutcome::Unknown) {
                const auto visits = static_cast<double>(std::max<std::uint64_t>(shared.visits, 1));
                value.q = shared.value_sum / visits;
                value.draw = shared.draw_sum / visits;
                value.to_fix = 0;
            }
            ++shared.visits;
            shared.value_sum += value.q;
            shared.draw_sum += value.draw;
            if (value.moves_left) {
                shared.moves_left_sum += *value.moves_left;
                ++shared.moves_left_samples;
            }
            if (value.to_fix != 0) {
                const auto count = static_cast<double>(value.to_fix);
                shared.value_sum += count * value.q_delta;
                shared.draw_sum += count * value.draw_delta;
                if (shared.moves_left_samples == shared.visits)
                    shared.moves_left_sum += count * value.moves_delta;
            }
            // Lc0 MaybeSetBounds: a child's new result may settle the position above it.
            if (proof_changed) {
                auto& parent = tree->node(parent_id);
                const auto previous = parent.proof;
                refresh_proof_locked(parent_id);
                proof_changed = parent.proof != previous;
                if (proof_changed && parent.proof.outcome != ProofOutcome::Unknown) {
                    shared.proof = parent.proof;
                    set_proof_values(shared);
                }
            }
            if (value.moves_left) *value.moves_left += 1.0;
            // A parent's own edge may also differ from what its position now knows.
            auto adjusted = value;
            adjusted.to_fix = 0;
            if (dag_adjust_locked(parent_id, adjusted)) value = adjusted;
        }
        finish_backup_locked(request, root_child_proof_changed);
    }
    [[nodiscard]] Slot& slot_of(const Request& request) {
        for (auto& slot : slots)
            if (&request >= slot.requests.data() &&
                &request < slot.requests.data() + slot.requests.size())
                return slot;
        throw std::logic_error("Search request belongs to no slot");
    }
    // Complete undo/history restoration on every ordinary or exceptional exit.
    struct Replay {
        Worker& worker;
        chess::Position& position;
        const Path& path;
        const chess::AttackTables& attacks;
        std::size_t applied = 0;
        bool unmatched_move = false;
        static chess::Position& checked_position(Worker& worker) {
            if (!worker.position) throw std::logic_error("Search worker has no root");
            return *worker.position;
        }
        Replay(Worker& selected, const Path& selected_path, const chess::AttackTables& table)
            : worker(selected),
              position(checked_position(selected)),
              path(selected_path),
              attacks(table) {}
        Replay(const Replay&) = delete;
        Replay& operator=(const Replay&) = delete;
        Replay(Replay&&) = delete;
        Replay& operator=(Replay&&) = delete;
        ~Replay() {
            if (unmatched_move)
                position.unmake_move(path.moves.at(applied), worker.undo.at(applied));
            while (applied != 0) {
                --applied;
                static_cast<void>(worker.history.pop());
                position.unmake_move(path.moves.at(applied), worker.undo.at(applied));
            }
        }
        [[nodiscard]] Result<void> apply() {
            worker.keys[0] = position.key();
            for (std::size_t i = 0; i + 1 < path.size; ++i) {
                worker.undo.at(i) = position.make_move(path.moves.at(i), attacks);
                unmatched_move = true;
                auto appended = worker.history.append(position);
                if (!appended) return appended;
                unmatched_move = false;
                ++applied;
                worker.keys.at(i + 1) = position.key();
            }
            return {};
        }
    };
    [[nodiscard]] static Result<nn::PolicyIndices> policy_indices(const Request& request) {
        nn::PolicyIndices policy;
        policy.size = request.legal.size();
        for (std::size_t i = 0; i < policy.size; ++i) {
            const auto index = nn::PolicyMap::index(request.legal[i], request.side);
            if (!index) return tl::unexpected(index.error());
            policy.values.at(i) = *index;
        }
        return policy;
    }
    // Lc0 classic/search.cc ApplyDirichletNoise. Runs on the priors the root is about to take.
    [[nodiscard]] Result<void> apply_root_noise_locked(Request& request) {
        if (config.noise_epsilon == 0.0 || request.path.size != 1) return {};
        const auto count = request.legal.size();
        std::gamma_distribution<float> gamma(static_cast<float>(config.noise_alpha), 1.0F);
        std::array<float, chess::MoveList::kCapacity> noise{};
        float total = 0.0F;
        for (std::size_t i = 0; i < count; ++i) {
            noise.at(i) = gamma(random);
            total += noise.at(i);
        }
        if (total < std::numeric_limits<float>::min()) return {};
        const auto epsilon = static_cast<float>(config.noise_epsilon);
        for (std::size_t i = 0; i < count; ++i) {
            const auto mixed = decode_prior(request.seeds.at(i).prior_bits) * (1.0F - epsilon) +
                               epsilon * noise.at(i) / total;
            auto bits = encode_prior(std::min(mixed, 1.0F));
            if (!bits) return tl::unexpected(bits.error());
            request.seeds.at(i).prior_bits = *bits;
        }
        return {};
    }
    [[nodiscard]] Result<void> publish_value_locked(Request& request, Value value) {
        const auto leaf_id = request.path.nodes.at(request.path.size - 1);
        auto& leaf = tree->node(leaf_id);
        auto noised = apply_root_noise_locked(request);
        if (!noised) return noised;
        if (request.path.size - 1 == ceiling())
            leaf.state = NodeState::Frontier;
        else if (config.dag) {
            // Lc0 dag_classic FetchSingleNodeResult: the same position may have been evaluated
            // meanwhile, and a noised root keeps a copy of its own.
            const bool noisy_root = config.noise_epsilon != 0.0 && request.path.size == 1;
            auto low_id = noisy_root ? kNoLow : tree->find_low(request.low_key);
            if (low_id == kNoLow) {
                auto created = tree->create_low(request.low_key,
                                                {request.seeds.data(), request.legal.size()},
                                                !noisy_root);
                if (!created) return tl::unexpected(created.error());
                low_id = *created;
                auto& shared = tree->low(low_id);
                shared.leaf_q = static_cast<float>(value.q);
                shared.leaf_draw = static_cast<float>(value.draw);
                shared.leaf_moves_left =
                    value.moves_left.has_value()
                        ? std::optional<float>{static_cast<float>(*value.moves_left)}
                        : std::nullopt;
            }
            auto attached = tree->attach_low(leaf_id, low_id);
            if (!attached) return attached;
        } else {
            auto expanded =
                tree->expand(leaf_id, {request.seeds.data(), request.legal.size()});
            if (!expanded) return tl::unexpected(expanded.error());
        }
        leaf.leaf_q = static_cast<float>(value.q);
        leaf.leaf_draw = static_cast<float>(value.draw);
        leaf.leaf_moves_left = value.moves_left.has_value()
                                   ? std::optional<float>{static_cast<float>(*value.moves_left)}
                                   : std::nullopt;
        backup_locked(request, value);
        return {};
    }
    [[nodiscard]] Result<bool> apply_cached(Slot& slot, Request& request,
                                            const nn::LegalOutput& output) {
        // lc0 fd71a2d classic/search.cc ProcessPickedTask/GatherMinibatch:
        // ready cache values expand and back up before waiting for the NN.
        const auto count = request.legal.size();
        auto valid = nn::validate_legal_output(output, count);
        if (!valid) return tl::unexpected(valid.error());
        auto priors = encode_legal_logits(std::span{output.logits}.first(count),
                                          config.selection.policy_temperature,
                                          std::span{request.priors}.first(count));
        if (!priors) return tl::unexpected(priors.error());
        for (std::size_t i = 0; i < count; ++i)
            request.seeds.at(i) = {.move = request.legal[i], .prior_bits = request.priors.at(i)};
        const auto value = network_value(output.wdl, output.moves_left, request.path.size - 1);
        const std::lock_guard lock(tree_mutex);
        if (slot.out_of_order_count >= out_of_order_limit()) return false;
        const auto statistics = tree->stats();
        if (request.path.size - 1 < ceiling() &&
            (count > statistics.edge_capacity - statistics.edge_count ||
             (config.dag && statistics.low_count == statistics.low_capacity))) {
            stopped = StopReason::MemoryLimit;
            request.reservation.release_locked();
            return true;
        }
        auto published = publish_value_locked(request, value);
        if (!published) return tl::unexpected(published.error());
        ++slot.out_of_order_count;
        if (limits.log_search_stats) ++diagnostics.out_of_order;
        return true;
    }
    // lc0 fd71a2d classic/search.cc ProcessPickedTask: classify and encode one
    // picked visit outside the tree lock. Known leaves, new terminals and cache
    // hits back up immediately (out of order); the rest wait for the NN.
    [[nodiscard]] Result<void> process_request(Slot& slot, Worker& worker, Request& request,
                                               nn::InputPlanes& input) {
        // lc0 ProcessPickedTask: with OutOfOrderEval off a value needing no NN stays in the
        // minibatch and backs up with it.
        const auto defer = [&](double q, double draw, std::optional<double> moves_left) {
            request.deferred_q = q;
            request.deferred_draw = draw;
            request.deferred_moves_left = moves_left;
            request.deferred = true;
        };
        if (request.known) {
            const std::lock_guard lock(tree_mutex);
            const auto& leaf = tree->node(request.path.nodes.at(request.path.size - 1));
            if (!config.gather.out_of_order_eval) {
                defer(leaf.leaf_q, leaf.leaf_draw, leaf.leaf_moves_left);
                return {};
            }
            backup_locked(
                request,
                {.q = leaf.leaf_q, .draw = leaf.leaf_draw, .moves_left = leaf.leaf_moves_left});
            ++slot.out_of_order_count;
            if (limits.log_search_stats) ++diagnostics.out_of_order;
            return {};
        }
        Replay replay(worker, request.path, attacks);
        auto replayed = replay.apply();
        if (!replayed) return tl::unexpected(replayed.error());
        auto status =
            classify_leaf(replay.position, attacks, {worker.keys.data(), request.path.size - 1},
                          request.path.size == 1 ? LeafContext::Root : LeafContext::NonRoot,
                          config.two_fold_draws && !config.dag, !config.dag);
        if (!status) return tl::unexpected(status.error());
        if (status->reason != TerminalReason::None) {
            const std::lock_guard lock(tree_mutex);
            auto& leaf = tree->node(request.path.nodes.at(request.path.size - 1));
            leaf.state = NodeState::Terminal;
            leaf.proof = config.terminal_proof ? proof_from_terminal(status->reason) : NodeProof{};
            leaf.leaf_q = status->q;
            leaf.leaf_draw = status->draw;
            // lc0 fd71a2d classic/search.cc ExtendNode: rule terminals have
            // M = 0; a tree twofold uses its cycle length.
            leaf.twofold = status->reason == TerminalReason::TreeRepetition;
            leaf.leaf_moves_left = static_cast<float>(status->repetition_cycle);
            if (!config.gather.out_of_order_eval) {
                defer(status->q, status->draw, static_cast<double>(status->repetition_cycle));
                return {};
            }
            backup_locked(request, {.q = status->q,
                                    .draw = status->draw,
                                    .moves_left = static_cast<double>(status->repetition_cycle)});
            ++slot.out_of_order_count;
            if (limits.log_search_stats) ++diagnostics.out_of_order;
            return {};
        }
        // lc0 fd71a2d classic/search.cc ExtendNode: below the root, a position in the tables is
        // terminal. Cursed wins and blessed losses count as draws, and having no NN evaluation it
        // takes its moves left from its parent.
        if (config.tablebase && !limits.root_in_dtz && request.path.size > 1 &&
            config.tablebase->can_probe(replay.position)) {
            if (const auto wdl = config.tablebase->probe_wdl(replay.position, attacks)) {
                tb_hits.fetch_add(1, std::memory_order_relaxed);
                const double q = *wdl == TablebaseWdl::Win ? 1.0 : *wdl == TablebaseWdl::Loss ? -1.0 : 0.0;
                const double draw = q == 0.0 ? 1.0 : 0.0;
                const std::lock_guard lock(tree_mutex);
                const auto& parent = tree->node(request.path.nodes.at(request.path.size - 2));
                const double parent_moves =
                    parent.moves_left_samples != 0
                        ? parent.moves_left_sum / static_cast<double>(parent.moves_left_samples)
                        : 0.0;
                const double moves_left = std::max(0.0, parent_moves - 1.0);
                auto& leaf = tree->node(request.path.nodes.at(request.path.size - 1));
                leaf.state = NodeState::Terminal;
                leaf.proof = !config.terminal_proof ? NodeProof{}
                             : q > 0.0 ? NodeProof{.outcome = ProofOutcome::Win,
                                                   .mate_upper_bound = kTablebaseMateDistance}
                             : q < 0.0 ? NodeProof{.outcome = ProofOutcome::Loss,
                                                   .mate_upper_bound = kTablebaseMateDistance}
                                       : NodeProof{.outcome = ProofOutcome::Draw};
                leaf.leaf_q = static_cast<float>(q);
                leaf.leaf_draw = static_cast<float>(draw);
                leaf.twofold = false;
                leaf.leaf_moves_left = static_cast<float>(moves_left);
                if (!config.gather.out_of_order_eval) {
                    defer(q, draw, moves_left);
                    return {};
                }
                backup_locked(request, {.q = q, .draw = draw, .moves_left = moves_left});
                ++slot.out_of_order_count;
                if (limits.log_search_stats) ++diagnostics.out_of_order;
                return {};
            }
        }
        // Lc0 dag_classic ExtendNode: the transposition table comes before the cache and the NN.
        // A hit waits for no evaluation, so it backs up now like any other out-of-order result.
        if (config.dag) {
            request.low_key = low_key(replay.position);
            const std::lock_guard lock(tree_mutex);
            const auto found = tree->find_low(request.low_key);
            if (found != kNoLow && request.path.size - 1 < ceiling() &&
                !(config.noise_epsilon != 0.0 && request.path.size == 1)) {
                auto attached =
                    tree->attach_low(request.path.nodes.at(request.path.size - 1), found);
                if (!attached) return attached;
                if (!config.gather.out_of_order_eval) {
                    defer(0.0, 0.0, std::nullopt);
                    return {};
                }
                backup_locked(request, {});
                ++slot.out_of_order_count;
                if (limits.log_search_stats) ++diagnostics.out_of_order;
                return {};
            }
        }
        auto encoded = nn::encode_classical(worker.history.frames(), metadata.input_format,
                                            config.history_fill);
        if (!encoded) return tl::unexpected(encoded.error());
        request.legal = status->legal_moves;
        request.side = replay.position.side_to_move();
        input = *encoded;
        // Deterministic mode preserves identical cache-on/off batch barriers. With OutOfOrderEval
        // off a cached position takes its row in the batch: it reaches the same value in batch
        // order, at the cost of one more evaluation.
        if (legal_cache && !config.deterministic && config.gather.out_of_order_eval) {
            const auto policy = policy_indices(request);
            if (!policy) return tl::unexpected(policy.error());
            const auto cached = legal_cache->lookup(input, *policy);
            if (!cached) return tl::unexpected(cached.error());
            const auto& value = *cached;
            if (value.has_value()) {
                auto applied = apply_cached(slot, request, *value);
                if (!applied) return tl::unexpected(applied.error());
                if (*applied) return {};
            }
        }
        request.ready = true;
        return {};
    }
    void worker_loop(std::size_t index) {
        std::size_t seen = 0;
        for (;;) {
            std::unique_lock lock(work_mutex);
            work_changed.wait(lock, [&] { return shutdown || epoch != seen; });
            if (shutdown) return;
            seen = epoch;
            auto& slot = *gathering;
            const auto selected = phase;
            const bool takes_part = index + 1 < active_tasks;
            lock.unlock();
            if (takes_part) work_rows(slot, workers.at(index), selected);
            lock.lock();
            if (--remaining_workers == 0) phase_done.notify_one();
        }
    }
    // Takes rows of the current phase until none are left or one failed.
    void work_rows(Slot& slot, Worker& worker, WorkPhase selected) {
        const auto rows =
            selected == WorkPhase::Process ? slot.picked_count : slot.count + slot.prefetched;
        while (!slot.halt.load(std::memory_order_relaxed)) {
            const auto next = slot.next.fetch_add(1, std::memory_order_relaxed);
            if (next >= rows) break;
            try {
                auto result =
                    selected == WorkPhase::Publish
                        ? prepare_row(slot, next)
                        : process_request(slot, worker, slot.requests[slot.picked.at(next)],
                                          slot.inputs[slot.picked.at(next)]);
                if (!result) {
                    slot.halt.store(true, std::memory_order_relaxed);
                    const std::lock_guard error_lock(work_mutex);
                    if (slot.error.empty()) slot.error.assign(result.error());
                }
            } catch (const std::exception& error) {
                slot.halt.store(true, std::memory_order_relaxed);
                const std::lock_guard error_lock(work_mutex);
                if (slot.error.empty()) {
                    slot.error.assign("Search worker: ");
                    slot.error.append(error.what());
                }
            } catch (...) {
                slot.halt.store(true, std::memory_order_relaxed);
                const std::lock_guard error_lock(work_mutex);
                if (slot.error.empty())
                    slot.error.assign("Search worker failed with non-standard exception");
            }
        }
    }
    // Worker threads; the coordinator's own scratch, when it has one, is the last.
    [[nodiscard]] std::size_t helper_threads() const noexcept {
        return config.deterministic || workers.empty() ? workers.size() : workers.size() - 1;
    }
    void inference_loop() {
        for (;;) {
            std::unique_lock lock(work_mutex);
            inference_changed.wait(lock, [&] {
                return shutdown || slots[0].state == SlotState::Ready ||
                       slots[1].state == SlotState::Ready;
            });
            if (shutdown) return;
            Slot* selected = nullptr;
            for (auto& slot : slots)
                if (slot.state == SlotState::Ready &&
                    (!selected || slot.sequence < selected->sequence))
                    selected = &slot;
            {
                // Work-before-tree is the only nested lock order. Recheck at
                // inference acceptance as stop may race queue publication.
                const std::lock_guard tree_lock(tree_mutex);
                if (should_stop_locked()) {
                    selected->state = SlotState::Cancelled;
                    inference_changed.notify_all();
                    continue;
                }
                selected->state = SlotState::Running;
            }
            lock.unlock();
            try {
                nn::EvaluationTiming timing;
                const auto evaluate = [&]() -> Result<void> {
                    if (evaluator) {
                        const auto rows = selected->count + selected->prefetched;
                        return evaluator->evaluate({selected->inputs.data(), rows},
                                                   {selected->policies.data(), rows},
                                                   {selected->legal_outputs.data(), rows},
                                                   limits.log_search_stats ? &timing : nullptr);
                    }
                    // A successful partial provider write cannot reuse valid scratch.
                    for (std::size_t i = 0; i < selected->count; ++i) {
                        selected->outputs[i].policy_logits.fill(
                            std::numeric_limits<float>::quiet_NaN());
                        selected->outputs[i].wdl.fill(std::numeric_limits<float>::quiet_NaN());
                        selected->outputs[i].moves_left = std::numeric_limits<float>::quiet_NaN();
                    }
                    const auto began = limits.log_search_stats ? Clock::now() : Clock::time_point{};
                    auto evaluated = backend.evaluate({selected->inputs.data(), selected->count},
                                                      {selected->outputs.data(), selected->count});
                    if (limits.log_search_stats) {
                        timing.positions = backend.physical_batch_size(selected->count);
                        timing.provider = Clock::now() - began;
                    }
                    return evaluated;
                };
                auto result = evaluate();
                if (limits.log_search_stats && timing.positions != 0) {
                    const std::lock_guard stats_lock(tree_mutex);
                    ++diagnostics.batch_sizes.at(timing.positions);
                    diagnostics.provider += timing.provider;
                }
                // The provider thread only evaluates. The coordinator backs the
                // batch up while the next queued batch already runs here, the
                // overlap lc0 gets from search workers computing in turn.
                if (!result) {
                    selected->error.assign("Search inference: ");
                    selected->error.append(result.error());
                }
            } catch (const std::exception& error) {
                selected->error.assign("Search inference exception: ");
                selected->error.append(error.what());
            } catch (...) {
                selected->error.assign("Search inference failed with non-standard exception");
            }
            lock.lock();
            selected->state = SlotState::Done;
            inference_changed.notify_all();
        }
    }
    void stop_threads() noexcept {
        {
            const std::lock_guard lock(work_mutex);
            shutdown = true;
        }
        work_changed.notify_all();
        inference_changed.notify_all();
        for (auto& thread : worker_threads)
            if (thread.joinable()) thread.join();
        worker_threads.clear();
        picker.shutdown.store(true, std::memory_order_release);
        picker.closed.store(true, std::memory_order_release);
        picker.wake.fetch_add(1, std::memory_order_release);
        picker.wake.notify_all();
        for (auto& thread : pick_threads)
            if (thread.joinable()) thread.join();
        pick_threads.clear();
        if (inference_thread.joinable()) inference_thread.join();
        for (auto& slot : slots) {
            release_slot(slot);
            slot.state = SlotState::Idle;
        }
    }
    struct ThreadScope {
        Impl& owner;
        explicit ThreadScope(Impl& value) : owner(value) {}
        ~ThreadScope() { owner.stop_threads(); }
        ThreadScope(const ThreadScope&) = delete;
        ThreadScope& operator=(const ThreadScope&) = delete;
        ThreadScope(ThreadScope&&) = delete;
        ThreadScope& operator=(ThreadScope&&) = delete;
    };
    // Real visits the node limit still allows. Virtual collisions are cancelled,
    // so they never spend it; Mouse keeps an exact cap on real requests.
    [[nodiscard]] std::uint64_t physical_allowance_locked() const {
        const auto& root_node = tree->node(0);
        if (!limits.nodes && root_node.visits == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("Search visit counter exhausted");
        const auto allowance = limits.nodes.value_or(std::numeric_limits<std::uint64_t>::max());
        const auto virtual_visits = slots[0].collision_root_visits + slots[1].collision_root_visits;
        assert(root_node.in_flight >= virtual_visits);
        const auto physical_in_flight = root_node.in_flight - virtual_visits;
        if (root_node.visits >= allowance) return 0;
        const auto open = allowance - root_node.visits;
        return open > physical_in_flight ? open - physical_in_flight : 0;
    }
    // lc0 GatherMinibatch "upsize the collision to exit sooner": a collision
    // that took a whole walk grows to its maxvisit, bounded by the budget left.
    void upsize_collision_locked(Slot& slot, std::size_t collisions_left) {
        if (picker.main.upsize_record >= slot.collision_count) return;
        auto& collision = slot.collisions.at(picker.main.upsize_record);
        const auto target = std::min<std::uint64_t>(picker.main.upsize_cap, collisions_left);
        if (target <= collision.visits) return;
        const auto extra = static_cast<std::uint32_t>(target - collision.visits);
        for (std::size_t j = 0; j < collision.size; ++j)
            if (extra > std::numeric_limits<std::uint32_t>::max() -
                            tree->node(slot.collision_nodes.at(collision.offset + j)).in_flight)
                throw std::overflow_error("Collision virtual-visit counter exhausted");
        for (std::size_t j = 0; j < collision.size; ++j)
            tree->node(slot.collision_nodes.at(collision.offset + j)).in_flight += extra;
        collision.visits += extra;
        slot.collision_visits += extra;
        if (collision.size != 0) slot.collision_root_visits += extra;
    }
    // Runs one phase over the slot's rows on the worker threads and waits for all of them.
    [[nodiscard]] Result<void> run_workers(Slot& slot, WorkPhase selected) {
        const auto helpers = helper_threads();
        // lc0 GatherMinibatch: tasks only accelerate a round of at least MinimumProcessingWork
        // rows. TaskWorkers 0 leaves the coordinator to do every phase alone.
        if (!config.deterministic &&
            (helpers == 0 || (selected == WorkPhase::Process &&
                              slot.picked_count < config.gather.minimum_processing_work))) {
            slot.next = 0;
            work_rows(slot, workers.back(), selected);
            if (!slot.error.empty()) return tl::unexpected(slot.error.string());
            return {};
        }
        std::unique_lock lock(work_mutex);
        slot.next = 0;
        gathering = &slot;
        phase = selected;
        remaining_workers = helpers;
        // lc0 num_tasks: clamp(rows / MinimumPerTaskProcessing, 2, task_workers + 1).
        active_tasks =
            selected == WorkPhase::Process && !config.deterministic
                ? std::clamp<std::size_t>(
                      slot.picked_count / config.gather.minimum_per_task_processing, 2, helpers + 1)
                : helpers + 1;
        ++epoch;
        work_changed.notify_all();
        // Picked rows are heavy enough (replay, legal moves, encoding, cache lookup) that
        // the coordinator, which would only wait, takes some too. Measured (M1 on the CPU
        // provider, 2 threads, 2026-10-28): per round 82-85 us -> 58-60 us. Publication
        // rows are a few microseconds each; there it only got in the workers' way (per
        // batch 216-223 us with the workers alone, 307 us with the coordinator too).
        if (!config.deterministic && selected == WorkPhase::Process) {
            lock.unlock();
            work_rows(slot, workers.back(), selected);
            lock.lock();
        }
        phase_done.wait(lock, [&] { return remaining_workers == 0; });
        if (!slot.error.empty()) return tl::unexpected(slot.error.string());
        return {};
    }
    // lc0 fd71a2d classic/search.cc GatherMinibatch: repeat single-walk picks
    // until the batch is full, the collision budget is spent, or too many
    // results arrived out of order. A round with any out-of-order result
    // reverts that round's collisions, because the new values change selection.
    [[nodiscard]] Result<void> gather_once(Slot& slot) {
        release_slot(slot);
        slot.error.clear();
        slot.halt = false;
        {
            const std::lock_guard tree_lock(tree_mutex);
            const auto visits = tree->node(0).visits;
            auto remaining = std::numeric_limits<std::uint64_t>::max();
            // lc0 common.cc enables VisitsStopper's remaining-node hint only
            // when smart pruning is enabled. A fixed-node parity run disables it.
            if (limits.smart_pruning && limits.smart_pruning_factor > 0.0) {
                const auto ceiling_nodes = limits.nodes.value_or(4'000'000'000ULL);
                remaining = ceiling_nodes > visits ? ceiling_nodes - visits : 1;
                remaining = std::min(remaining, pruning.estimated_remaining_playouts());
            }
            slot.collision_budget = collision_visit_budget(visits, remaining, config.gather);
            picker.remaining_playouts = remaining;
            slot.collision_visits = 0;
            slot.out_of_order_count = 0;
        }
        const auto capacity = limits.maximum_batch.value_or(config.maximum_batch);
        std::size_t rows_for_network = 0, rows_deferred = 0;
        for (;;) {
            // lc0 GatherMinibatch: something that needs no NN goes first, and a backend with
            // nothing to do gets what is ready rather than idling while the batch fills.
            if (rows_deferred != 0 && rows_for_network == 0) break;
            if (rows_for_network > config.gather.idling_minimum_work && !config.deterministic &&
                backend_idle(slot))
                break;
            std::size_t first_collision = 0, first_collision_node = 0, out_of_order_before = 0;
            std::uint32_t collision_visits_before = 0;
            {
                const std::lock_guard tree_lock(tree_mutex);
                if (should_stop_locked() || slot.halt.load(std::memory_order_relaxed)) break;
                std::size_t free = 0;
                for (std::size_t row = 0; row < capacity; ++row) {
                    const auto& request = slot.requests[row];
                    if (!request.ready && !request.reservation.active)
                        picker.free_rows.at(free++) = row;
                }
                const auto out_of_order_left = out_of_order_limit() > slot.out_of_order_count
                                                   ? out_of_order_limit() - slot.out_of_order_count
                                                   : 0;
                const auto collisions_left = slot.collision_budget > slot.collision_visits
                                                 ? slot.collision_budget - slot.collision_visits
                                                 : 0;
                const auto limit =
                    std::min({static_cast<std::size_t>(collisions_left), free, out_of_order_left});
                if (limit == 0) break;
                picker.free_used.store(0, std::memory_order_relaxed);
                picker.physical_left.store(physical_allowance_locked(), std::memory_order_relaxed);
                first_collision = slot.collision_count;
                first_collision_node = slot.collision_nodes_used;
                out_of_order_before = slot.out_of_order_count;
                collision_visits_before = slot.collision_visits;
                pick_locked(slot, static_cast<std::uint32_t>(limit));
                slot.picked_count = picker.free_used.load(std::memory_order_relaxed);
                upsize_collision_locked(slot, collisions_left);
            }
            // A round of collisions only still raised its ancestors' in-flight
            // counts, which steers the next walk elsewhere (lc0 keeps picking
            // until collisions_left is spent).
            if (slot.picked_count == 0) {
                // Nothing placed at all (no selectable edge): stop gathering.
                if (slot.collision_visits == collision_visits_before) break;
                continue;
            }
            auto processed = run_workers(slot, WorkPhase::Process);
            if (!processed) return processed;
            for (std::size_t i = 0; i < slot.picked_count; ++i) {
                const auto& request = slot.requests[slot.picked[i]];
                rows_for_network += request.ready ? 1U : 0U;
                rows_deferred += request.deferred ? 1U : 0U;
            }
            const std::lock_guard tree_lock(tree_mutex);
            if (slot.out_of_order_count != out_of_order_before)
                cancel_round_collisions_locked(slot, first_collision, first_collision_node);
        }
        // Compact only inputs, retaining stable Request/RAII addresses.
        for (std::size_t i = 0; i < config.maximum_batch; ++i) {
            if (!slot.requests[i].ready) continue;
            slot.rows[slot.count] = i;
            if (slot.count != i) slot.inputs[slot.count] = slot.inputs[i];
            if (evaluator) {
                const auto policy = policy_indices(slot.requests.at(i));
                if (!policy) return tl::unexpected(policy.error());
                slot.policies.at(slot.count) = *policy;
            }
            ++slot.count;
        }
        return {};
    }
    [[nodiscard]] Result<void> gather(Slot& slot) {
        const auto began = limits.log_search_stats ? Clock::now() : Clock::time_point{};
        const auto result = gather_impl(slot);
        if (limits.log_search_stats) {
            const std::lock_guard lock(tree_mutex);
            diagnostics.gather += Clock::now() - began;
        }
        return result;
    }
    // Backs up the rows OutOfOrderEval held back, in row order. Returns how many there were.
    std::size_t back_up_deferred_locked(Slot& slot) {
        std::size_t count = 0;
        for (std::size_t i = 0; i < config.maximum_batch; ++i) {
            auto& request = slot.requests[i];
            if (!request.deferred) continue;
            request.deferred = false;
            backup_locked(request, {.q = request.deferred_q,
                                    .draw = request.deferred_draw,
                                    .moves_left = request.deferred_moves_left});
            ++count;
        }
        return count;
    }
    [[nodiscard]] Result<void> gather_impl(Slot& slot) {
        for (;;) {
            auto gathered = gather_once(slot);
            if (!gathered) return gathered;
            const std::lock_guard lock(tree_mutex);
            const auto deferred = slot.count == 0 ? back_up_deferred_locked(slot) : 0;
            if (slot.count != 0 || (slot.out_of_order_count == 0 && deferred == 0)) {
                prefetch_locked(slot);
                return {};
            }
            ++completed_batches;
            smart_prune_locked();
            if (should_stop_locked()) return {};
        }
    }
    // lc0 GatherMinibatch: thread_count - backend_waiting_counter > ThreadIdlingThreshold, with
    // the two pipeline slots as Mouse's search threads and the other slot's queue as the wait.
    [[nodiscard]] bool backend_idle(const Slot& current) {
        std::uint32_t waiting = 0;
        const std::lock_guard lock(work_mutex);
        for (const auto& slot : slots)
            if (&slot != &current &&
                (slot.state == SlotState::Ready || slot.state == SlotState::Running))
                ++waiting;
        return slots.size() - waiting > config.gather.thread_idling_threshold;
    }
    // lc0 fd71a2d classic/search.cc MaybePrefetchIntoCache / PrefetchIntoCache: a batch smaller
    // than MaxPrefetch also evaluates positions PUCT is likely to reach soon, caching the results
    // so those visits later back up out of order without waiting for the NN. Runs on the
    // coordinator, whose worker scratch is free between rounds.
    void prefetch_locked(Slot& slot) {
        slot.prefetched = 0;
        // Lc0's dag_classic has no prefetch.
        if (!legal_cache || config.deterministic || config.dag || workers.empty() ||
            slot.count == 0)
            return;
        const auto limit = std::min<std::size_t>(config.gather.max_prefetch,
                                                 limits.maximum_batch.value_or(config.maximum_batch));
        if (slot.count >= limit || should_stop_locked()) return;
        auto& worker = workers.back();
        if (!worker.position) return;
        static_cast<void>(
            prefetch_walk(slot, worker, 0, static_cast<std::int64_t>(limit - slot.count), 0));
    }
    std::int64_t prefetch_walk(Slot& slot, Worker& worker, NodeId id, std::int64_t budget,
                               std::size_t depth) {
        if (budget <= 0) return 0;
        const Node* node = id == kNoNode ? nullptr : &tree->node(id);
        // A leaf nothing has started on yet.
        if (!node || node->visits + node->in_flight == 0) return prefetch_leaf(slot, worker);
        // Being extended by a visit in flight, or terminal: nothing to prefetch here.
        if (node->visits == 0 || node->state == NodeState::Terminal ||
            node->state == NodeState::Frontier || depth >= ceiling() ||
            depth + 1 >= config.max_depth)
            return 0;
        struct Scored {
            double score = 0;
            std::uint32_t index = 0;
        };
        std::array<Scored, kMaximumLegalMoves> scored{};
        std::size_t count = 0;
        const auto edges = tree->edges(id);
        const bool at_root = depth == 0;
        const double draw_score = draw_score_at(depth);
        const double parent_q =
            (node->value_sum + draw_score * node->draw_sum) / static_cast<double>(node->visits);
        double mass = 0;
        for (const auto& edge : edges)
            if (edge.child != kNoNode && tree->node(edge.child).visits != 0)
                mass += decode_prior(edge.prior_bits);
        const double fpu = first_play_urgency(config.selection, at_root, parent_q, mass);
        const double multiplier = exploration_multiplier(config.selection, selection_visits(id));
        const auto q_of = [&](const Node* child, double unvisited) {
            return child && child->visits != 0
                       ? (draw_score * child->draw_sum - child->value_sum) /
                             static_cast<double>(child->visits)
                       : unvisited;
        };
        for (std::size_t i = 0; i < edges.size(); ++i) {
            const auto& edge = edges[i];
            const auto prior = decode_prior(edge.prior_bits);
            if (prior == 0.0F || (at_root && !allowed(edge.move))) continue;
            const Node* child = edge.child == kNoNode ? nullptr : &tree->node(edge.child);
            const auto visits_started = child ? child->visits + child->in_flight : 0;
            scored[count++] = {.score = -(multiplier * prior /
                                          (1.0 + static_cast<double>(visits_started))) -
                                        q_of(child, fpu),
                               .index = static_cast<std::uint32_t>(i)};
        }
        std::size_t first_unsorted = 0;
        std::int64_t total = 0;
        auto to_spend = budget;
        for (std::size_t i = 0; i < count && budget > 0; ++i) {
            // Lc0 sorts the next chunk of three at a time; most walks stop early.
            if (first_unsorted != count && i + 2 >= first_unsorted) {
                const auto next = std::min(count, first_unsorted + (budget < 2 ? 2U : 3U));
                std::partial_sort(scored.begin() + static_cast<std::ptrdiff_t>(first_unsorted),
                                  scored.begin() + static_cast<std::ptrdiff_t>(next),
                                  scored.begin() + static_cast<std::ptrdiff_t>(count),
                                  [](const Scored& a, const Scored& b) { return a.score < b.score; });
                first_unsorted = next;
            }
            const auto& edge = edges[scored[i].index];
            const Node* child = edge.child == kNoNode ? nullptr : &tree->node(edge.child);
            // The last child keeps the budget of the one before it.
            if (i + 1 != count) {
                const double next_score = -scored[i + 1].score;
                const double q = q_of(child, -fpu);
                if (next_score > q) {
                    const auto visits_started = child ? child->visits + child->in_flight : 0;
                    const auto wanted = static_cast<std::int64_t>(
                        multiplier * decode_prior(edge.prior_bits) / (next_score - q) -
                        static_cast<double>(visits_started));
                    to_spend = std::min(budget, wanted + 1);
                } else
                    to_spend = budget;
            }
            auto& position = *worker.position;
            worker.undo.at(depth) = position.make_move(edge.move, attacks);
            auto appended = worker.history.append(position);
            std::int64_t spent = 0;
            if (appended) {
                spent = prefetch_walk(slot, worker, edge.child, to_spend, depth + 1);
                static_cast<void>(worker.history.pop());
            }
            position.unmake_move(edge.move, worker.undo.at(depth));
            if (!appended) return total;
            budget -= spent;
            total += spent;
        }
        return total;
    }
    // Queues the position at the end of the walk unless the cache already holds it. A cached or
    // unusable position still counts, as in lc0: finding one that needs work is not worth the time.
    std::int64_t prefetch_leaf(Slot& slot, Worker& worker) {
        const auto row = slot.count + slot.prefetched;
        if (row >= config.maximum_batch || row >= slot.inputs.size()) return 1;
        auto& position = *worker.position;
        const auto legal = chess::legal_moves(position, attacks);
        if (legal.empty()) return 1;
        auto encoded = nn::encode_classical(worker.history.frames(), metadata.input_format,
                                            config.history_fill);
        if (!encoded) return 1;
        nn::PolicyIndices policy;
        policy.size = legal.size();
        for (std::size_t i = 0; i < policy.size; ++i) {
            const auto index = nn::PolicyMap::index(legal[i], position.side_to_move());
            if (!index) return 1;
            policy.values.at(i) = *index;
        }
        const auto cached = legal_cache->lookup(*encoded, policy);
        if (!cached || cached->has_value()) return 1;
        slot.inputs[row] = *encoded;
        slot.policies.at(row) = policy;
        ++slot.prefetched;
        return 1;
    }
    void submit(Slot& slot) {
        if (slot.count == 0) {
            slot.prefetched = 0;
            return;
        }
        const std::lock_guard lock(work_mutex);
        const std::lock_guard tree_lock(tree_mutex);
        if (should_stop_locked()) {
            release_slot_locked(slot);
            return;
        }
        slot.sequence = next_sequence++;
        slot.state = SlotState::Ready;
        inference_changed.notify_one();
    }

    [[nodiscard]] Result<void> apply(Slot& slot) {
        {
            std::unique_lock lock(work_mutex);
            inference_changed.wait(lock, [&] {
                return slot.state == SlotState::Done || slot.state == SlotState::Cancelled;
            });
            if (slot.state == SlotState::Cancelled) {
                // No provider ran, so there are no outputs/visits to publish.
                // The coordinator still owns these shapes and reservations.
                release_slot(slot);
                slot.state = SlotState::Idle;
                return {};
            }
            if (!slot.error.empty()) return tl::unexpected(slot.error.string());
        }
        auto published = publish_batch(slot);
        if (!published) return published;
        const std::lock_guard lock(work_mutex);
        slot.state = SlotState::Idle;
        return {};
    }
    // One evaluated row, independent of every other: validate its output, encode its
    // priors and edge seeds, then (lc0 caches a computed batch before backing it up)
    // cache it. A row is cached only after its own output validated; any invalid row
    // fails the batch before the tree changes.
    [[nodiscard]] Result<void> prepare_row(Slot& slot, std::size_t row) {
        if (row >= slot.count) {
            // A prefetched row only fills the cache.
            const auto valid =
                nn::validate_legal_output(slot.legal_outputs.at(row), slot.policies.at(row).size);
            if (!valid)
                return tl::unexpected(std::string{"Invalid prefetch NN output: "} + valid.error());
            auto inserted = legal_cache->insert(slot.inputs[row], slot.policies.at(row),
                                                slot.legal_outputs.at(row));
            if (!inserted) return inserted;
            slot.prepared.fetch_add(1, std::memory_order_relaxed);
            return {};
        }
        auto& request = slot.requests[slot.rows[row]];
        const auto count = request.legal.size();
        const auto valid = evaluator ? nn::validate_legal_output(slot.legal_outputs.at(row), count)
                                     : nn::validate_nn_output(slot.outputs[row]);
        if (!valid)
            return tl::unexpected(std::string{"Invalid search NN output: "} + valid.error());
        auto priors = evaluator
                          ? encode_legal_logits(
                                std::span{slot.legal_outputs.at(row).logits}.first(count),
                                config.selection.policy_temperature, {request.priors.data(), count})
                          : encode_legal_priors({request.legal.begin(), request.legal.end()},
                                                request.side, slot.outputs[row].policy_logits,
                                                config.selection.policy_temperature,
                                                {request.priors.data(), count});
        if (!priors) return tl::unexpected(priors.error());
        for (std::size_t i = 0; i < count; ++i)
            request.seeds.at(i) = {.move = request.legal[i], .prior_bits = request.priors.at(i)};
        if (legal_cache && !config.deterministic) {
            auto inserted = legal_cache->insert(slot.inputs[row], slot.policies.at(row),
                                                slot.legal_outputs.at(row));
            if (!inserted) return inserted;
        }
        slot.prepared.fetch_add(1, std::memory_order_relaxed);
        return {};
    }
    [[nodiscard]] Result<void> publish_batch(Slot& slot) {
        // The workers prepare every row in parallel; only the backup below is serial.
        // Gathering may have ended through halt (collision records or tree nodes full);
        // here it means only that a publication row failed, so it starts cleared.
        slot.halt.store(false, std::memory_order_relaxed);
        slot.prepared.store(0, std::memory_order_relaxed);
        auto prepared = run_workers(slot, WorkPhase::Publish);
        if (!prepared) return prepared;
        if (slot.prepared.load(std::memory_order_relaxed) != slot.count + slot.prefetched)
            return tl::unexpected(std::string{"Search batch has unprepared rows"});
        std::size_t required = 0;
        for (std::size_t row = 0; row < slot.count; ++row) {
            const auto& request = slot.requests[slot.rows[row]];
            if (request.path.size - 1 < ceiling()) required += request.legal.size();
        }
        {
            const std::lock_guard lock(tree_mutex);
            const auto statistics = tree->stats();
            if (required > statistics.edge_capacity - statistics.edge_count ||
                (config.dag && slot.count > statistics.low_capacity - statistics.low_count)) {
                stopped = StopReason::MemoryLimit;
                for (std::size_t i = 0; i < config.maximum_batch; ++i)
                    slot.requests[i].reservation.release_locked();
            } else {
                // Backing up one row may solidify a node, which must leave the later rows' edges.
                const struct ReserveScope {
                    std::size_t& reserved;
                    ~ReserveScope() { reserved = 0; }
                } reserve_scope{batch_edges_reserved};
                batch_edges_reserved = required;
                for (std::size_t row = 0; row < slot.count; ++row) {
                    auto& request = slot.requests[slot.rows[row]];
                    if (request.path.size - 1 < ceiling())
                        batch_edges_reserved -= request.legal.size();
                    const auto& wdl =
                        evaluator ? slot.legal_outputs.at(row).wdl : slot.outputs[row].wdl;
                    const auto moves_left = evaluator ? slot.legal_outputs.at(row).moves_left
                                                      : slot.outputs[row].moves_left;
                    auto published = publish_value_locked(
                        request, network_value(wdl, moves_left, request.path.size - 1));
                    if (!published) return tl::unexpected(published.error());
                }
                back_up_deferred_locked(slot);
            }
            // lc0 DoBackupUpdate cancels shared collisions after real backup.
            // Both slots may have gathered virtual reservations in the meantime.
            for (auto& pending : slots) cancel_collisions_locked(pending);
        }
        return {};
    }
    [[nodiscard]] const Edge* best_edge(NodeId id, std::size_t depth = 0) const {
        const Edge* best = nullptr;
        FinalMoveStatistics best_stats;
        const double draw_score = draw_score_at(depth);
        for (const auto& edge : tree->edges(id)) {
            if (id == 0 && !allowed(edge.move)) continue;
            const auto* child = edge.child == kNoNode ? nullptr : &tree->node(edge.child);
            const auto n = child ? child->visits : 0;
            const FinalMoveStatistics stats{
                .visits = n,
                .q = n == 0 ? 0
                            : (draw_score * child->draw_sum - child->value_sum) /
                                  static_cast<double>(n),
                .prior = decode_prior(edge.prior_bits),
                .child_proof = child ? child->proof : NodeProof{}};
            if (!best || prefer_final_move(stats, best_stats)) {
                best = &edge;
                best_stats = stats;
            }
        }
        return best;
    }
    // Lc0 fd71a2d classic/search.cc EnsureBestMoveKnown and GetBestRootChildWithTemperature.
    [[nodiscard]] std::optional<chess::Move> final_move_locked(std::uint64_t game_ply) {
        if (!tree) return std::nullopt;
        const auto& settings = config.temperature;
        auto temperature = static_cast<float>(settings.temperature);
        const auto moves = game_ply / 2;
        const auto endgame = static_cast<float>(settings.endgame);
        if (settings.cutoff_move != 0 && moves + 1 >= settings.cutoff_move) {
            temperature = endgame;
        } else if (temperature != 0.0F && settings.decay_moves != 0) {
            const std::uint64_t delay = settings.decay_delay_moves;
            if (moves >= delay + settings.decay_moves) {
                temperature = 0.0F;
            } else if (moves >= delay) {
                temperature *= static_cast<float>(delay + settings.decay_moves - moves) /
                               static_cast<float>(settings.decay_moves);
            }
            temperature = std::max(temperature, endgame);
        }
        const auto no_temperature = [&]() -> std::optional<chess::Move> {
            const auto* best = best_edge(0);
            return best ? std::optional<chess::Move>{best->move} : std::nullopt;
        };
        if (temperature == 0.0F) return no_temperature();
        const auto& node = tree->node(0);
        const auto edges = tree->edges(0);
        if (node.visits == 0 || edges.empty()) return no_temperature();
        const double draw_score = draw_score_at(0);
        double mass = 0;
        for (const auto& edge : edges)
            if (allowed(edge.move) && edge.child != kNoNode && tree->node(edge.child).visits != 0)
                mass += decode_prior(edge.prior_bits);
        const double parent_q =
            (node.value_sum + draw_score * node.draw_sum) / static_cast<double>(node.visits);
        const double fpu = first_play_urgency(config.selection, true, parent_q, mass);
        const auto offset = static_cast<float>(settings.visit_offset);
        const auto n_of = [&](const Edge& edge) -> std::uint64_t {
            return edge.child == kNoNode ? 0 : tree->node(edge.child).visits;
        };
        const auto q_of = [&](const Edge& edge) {
            const auto n = n_of(edge);
            if (n == 0) return static_cast<float>(fpu);
            const auto& child = tree->node(edge.child);
            return static_cast<float>((draw_score * child.draw_sum - child.value_sum) /
                                      static_cast<double>(n));
        };
        float max_n = 0.0F, max_eval = -1.0F;
        for (const auto& edge : edges) {
            if (!allowed(edge.move)) continue;
            const auto n = static_cast<float>(n_of(edge));
            if (n + offset > max_n) {
                max_n = n + offset;
                max_eval = q_of(edge);
            }
        }
        const float min_eval = max_eval - static_cast<float>(settings.value_cutoff) / 50.0F;
        std::array<float, chess::MoveList::kCapacity> sums{};
        std::array<const Edge*, chess::MoveList::kCapacity> candidates{};
        std::size_t count = 0;
        float sum = 0.0F;
        for (const auto& edge : edges) {
            if (!allowed(edge.move) || q_of(edge) < min_eval) continue;
            const float weight = max_n <= 0.0F
                                     ? decode_prior(edge.prior_bits)
                                     : (static_cast<float>(n_of(edge)) + offset) / max_n;
            sum += std::pow(std::max(0.0F, weight), 1.0F / temperature);
            sums.at(count) = sum;
            candidates.at(count++) = &edge;
        }
        if (count == 0 || sum <= 0.0F) return no_temperature();
        const float toss = std::uniform_real_distribution<float>(0.0F, sum)(random);
        const auto at = static_cast<std::size_t>(
            std::lower_bound(sums.begin(), sums.begin() + static_cast<std::ptrdiff_t>(count), toss) -
            sums.begin());
        return candidates.at(std::min(at, count - 1))->move;
    }
    [[nodiscard]] PrincipalVariation pv_locked(
        std::optional<chess::Move> first = std::nullopt) const {
        PrincipalVariation result;
        if (!tree || tree->node(0).state == NodeState::Terminal || (first && !allowed(*first)))
            return result;
        NodeId current = 0;
        // Lc0 dag_classic SendUciInfo: a line ends at a draw by repetition.
        std::optional<chess::Position> replay;
        if (config.dag && root) replay = *root;
        while (result.size < result.moves.size()) {
            const Edge* edge = nullptr;
            if (result.size == 0 && first) {
                for (const auto& candidate : tree->edges(0))
                    if (candidate.move == *first) edge = &candidate;
            } else
                edge = best_edge(current, result.size);
            // A legal prior choice is a best_move, but an unvisited edge is
            // not part of a completed principal variation.
            if (!edge || edge->child == kNoNode || tree->node(edge->child).visits == 0) break;
            result.moves.at(result.size++) = edge->move;
            current = edge->child;
            if (replay) {
                static_cast<void>(replay->make_move(edge->move, attacks));
                if (replay->repetitions().count >= 2) break;
            }
        }
        return result;
    }
    [[nodiscard]] std::size_t deepest_completed_locked() const {
        struct Cursor {
            NodeId node = 0;
            std::size_t next = 0;
        };
        std::array<Cursor, kMaximumDepth + 1> stack{};
        std::size_t depth = 0, deepest = 0;
        for (;;) {
            auto& cursor = stack.at(depth);
            const auto children = tree->edges(cursor.node);
            if (cursor.next == children.size()) {
                if (depth == 0) break;
                --depth;
                continue;
            }
            const auto child = children[cursor.next++].child;
            if (child == kNoNode || tree->node(child).visits == 0) continue;
            if (depth == kMaximumDepth)
                throw std::logic_error("Search retained path exceeds depth capacity");
            ++depth;
            stack.at(depth) = {.node = child, .next = 0};
            deepest = std::max(deepest, depth);
        }
        return deepest;
    }
    [[nodiscard]] std::unique_lock<std::mutex> snapshot_lock(SnapshotCost* cost) const {
        auto* scoped = SnapshotCostScope::current();
        const auto began = cost || scoped ? Clock::now() : Clock::time_point{};
        std::unique_lock lock(tree_mutex);
        if (cost || scoped) {
            const auto elapsed = Clock::now() - began;
            if (cost) {
                ++cost->tree_lock_count;
                cost->tree_lock_wait += elapsed;
            }
            if (scoped && scoped != cost) {
                ++scoped->tree_lock_count;
                scoped->tree_lock_wait += elapsed;
            }
        }
        return lock;
    }
    [[nodiscard]] SearchInfo snapshot_locked() const {
        SearchInfo result;
        result.budget_bytes = config.ram_limit_bytes;
        result.charged_bytes = overhead;
        if (!tree) return result;
        const auto stats = tree->stats();
        result.charged_bytes += stats.charged_bytes;
        result.tree_nodes = stats.node_count;
        result.tree_capacity = stats.node_limit;
        result.tree_used_bytes = stats.used_bytes;
        result.tree_arena_bytes = stats.arena_bytes;
        const auto& node = tree->node(0);
        result.nodes = node.visits;
        result.proof = node.proof;
        result.inherited_visits = inherited;
        result.new_playouts = node.visits >= inherited ? node.visits - inherited : 0;
        result.q =
            node.visits == 0 ? node.leaf_q : node.value_sum / static_cast<double>(node.visits);
        result.draw =
            node.visits == 0 ? node.leaf_draw : node.draw_sum / static_cast<double>(node.visits);
        if (node.visits != 0 && node.moves_left_samples == node.visits)
            result.moves_left = node.moves_left_sum / static_cast<double>(node.moves_left_samples);
        result.tb_hits = tb_hits.load(std::memory_order_relaxed);
        result.seldepth = seldepth;
        if (started != Clock::time_point{})
            result.elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                (running ? Clock::now() : finished) - started);
        result.pv = pv_locked();
        result.depth = result.pv.size;
        return result;
    }
    void smart_prune_locked() {
        if ((!limits.smart_pruning && !limits.time_stopper) || stopped) return;
        std::array<std::uint64_t, chess::MoveList::kCapacity> visits{};
        std::size_t count = 0, losing = 0;
        bool win_found = false, may_resign = true;
        // Lc0 PopulateCommonIterationStats: whether the most visited move also has the best Q + M.
        double max_q_plus_m = -1000.0;
        std::uint64_t max_n = 0;
        bool max_n_has_max_q_plus_m = true;
        const auto& root_node = tree->node(0);
        const double draw_score = draw_score_at(0);
        double fpu = 0.0;
        if (limits.time_stopper && root_node.visits != 0) {
            double mass = 0;
            for (const auto& edge : tree->edges(0))
                if (allowed(edge.move) && edge.child != kNoNode &&
                    tree->node(edge.child).visits != 0)
                    mass += decode_prior(edge.prior_bits);
            const double parent_q = (root_node.value_sum + draw_score * root_node.draw_sum) /
                                    static_cast<double>(root_node.visits);
            fpu = first_play_urgency(config.selection, true, parent_q, mass);
        }
        for (const auto& edge : tree->edges(0)) {
            // lc0 search.cc PopulateCommonIterationStats enumerates every root
            // edge, including those excluded by a searchmoves restriction.
            const Node* child = nullptr;
            if (edge.child != kNoNode) {
                child = &tree->node(edge.child);
                visits.at(count) = child->visits;
                if (child->visits != 0) {
                    if (child->proof.outcome == ProofOutcome::Win) ++losing;
                    if (child->proof.outcome == ProofOutcome::Loss) win_found = true;
                    // Pinned source's resignability threshold is-0.98f.
                    const auto q = -child->value_sum / static_cast<double>(child->visits);
                    if (q > -0.98F) may_resign = false;
                }
            }
            if (limits.time_stopper) {
                const auto n = child ? child->visits : 0;
                const double q = n != 0 ? (draw_score * child->draw_sum - child->value_sum) /
                                              static_cast<double>(n)
                                        : fpu;
                const double m = config.moves_left_utility && child
                                     ? moves_left_utility(root_node, *child, config.moves_left)
                                     : 0.0;
                if (max_n < n) {
                    max_n = n;
                    max_n_has_max_q_plus_m = false;
                }
                if (max_q_plus_m <= q + m) {
                    max_n_has_max_q_plus_m = max_n == n;
                    max_q_plus_m = q + m;
                }
            }
            ++count;
        }
        const auto now = Clock::now();
        const auto root_visits = root_node.visits;
        const auto new_nodes = root_visits - std::min(inherited, root_visits);
        TimeDecision decision;
        if (limits.time_stopper) {
            decision = limits.time_stopper({.new_nodes = new_nodes,
                                            .total_nodes = root_visits,
                                            .root_visits = std::span{visits}.first(count),
                                            .need_more_time = !max_n_has_max_q_plus_m});
            if (decision.stop) {
                stopped = StopReason::TimeLimit;
                return;
            }
        }
        if (!limits.smart_pruning) return;
        // lc0 common.cc always installs a4,000,000,000-visit overflow stopper;
        // timemgr.cc hints use100,000,000,000ms when no clock bound exists.
        const auto node_limit = limits.nodes.value_or(4'000'000'000ULL);
        const auto remaining_nodes = node_limit > root_visits ? node_limit - root_visits : 1;
        const auto remaining_time =
            decision.remaining
                ? *decision.remaining
                : (deadline ? std::chrono::duration_cast<std::chrono::milliseconds>(*deadline - now)
                            : std::chrono::milliseconds{100'000'000'000LL});
        if (pruning.should_stop(
                {.new_nodes = root_visits - inherited,
                 .completed_batches = completed_batches,
                 .remaining_nodes = remaining_nodes,
                 .elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - started),
                 .remaining_time = remaining_time,
                 .root_visits = std::span{visits}.first(count),
                 .losing_moves = losing,
                 .win_found = win_found,
                 .may_resign = may_resign,
                 .nps = decision.nps}))
            stopped = StopReason::SmartPruning;
    }
    // lc0 SearchWorker::ExecuteOneIteration: waste time to hold NodesPerSecondLimit.
    void wait_for_nps_limit() {
        if (config.nps_limit <= 0.0) return;
        for (;;) {
            {
                const std::lock_guard lock(tree_mutex);
                if (should_stop_locked()) return;
                const auto playouts = tree->node(0).visits - std::min(inherited, tree->node(0).visits);
                const auto elapsed =
                    std::chrono::duration<double, std::milli>(Clock::now() - started).count();
                if (elapsed <= 0.0 ||
                    static_cast<double>(playouts) * 1000.0 / elapsed <= config.nps_limit)
                    return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
        }
    }
    [[nodiscard]] Result<SearchResult> execute(const SearchLimits& selected) {
        {
            const std::lock_guard lock(tree_mutex);
            limits = selected;
            inherited = tree->node(0).visits;
            completed_batches = 0;
            current_best.reset();
            diagnostics = {};
            pruning = SmartPruning{static_cast<float>(limits.smart_pruning_factor),
                                   limits.minimum_smart_pruning_batches};
            even_draw_score = root && root->side_to_move() == chess::Color::Black
                                  ? -config.draw_score
                                  : config.draw_score;
            tb_hits.store(0, std::memory_order_relaxed);
            started = Clock::now();
            finished = started;
            running = true;
            stopped.reset();
            deadline = limits.deadline;
            if (limits.movetime) {
                const auto available = std::chrono::duration_cast<std::chrono::milliseconds>(
                    Clock::time_point::max() - started);
                const auto relative = *limits.movetime >= available ? Clock::time_point::max()
                                                                    : started + *limits.movetime;
                deadline = deadline ? std::min(*deadline, relative) : relative;
            }
        }
        auto& root_position = workers[0].position;
        if (!root_position.has_value())
            return tl::unexpected(std::string{"Search worker has no root"});
        auto root_status = classify_leaf(*root_position, attacks, {}, LeafContext::Root);
        if (!root_status) return tl::unexpected(root_status.error());
        if (root_status->reason != TerminalReason::None) {
            const std::lock_guard lock(tree_mutex);
            auto& node = tree->node(0);
            node.state = NodeState::Terminal;
            node.proof =
                config.terminal_proof ? proof_from_terminal(root_status->reason) : NodeProof{};
            node.leaf_q = root_status->q;
            node.leaf_draw = root_status->draw;
            node.value_sum = root_status->q * static_cast<double>(node.visits);
            node.draw_sum = root_status->draw * static_cast<double>(node.visits);
            node.leaf_moves_left = 0.0F;
            node.moves_left_sum = 0;
            node.moves_left_samples = node.visits;
            stopped = StopReason::Terminal;
        } else {
            ThreadScope cleanup(*this);
            shutdown = false;
            epoch = 0;
            next_sequence = 0;
            for (std::size_t i = 0; i < helper_threads(); ++i)
                worker_threads.emplace_back([this, i] { worker_loop(i); });
            picker.shutdown.store(false, std::memory_order_release);
            for (std::size_t i = 0; i + 1 < picker.task_walks.size(); ++i)
                pick_threads.emplace_back([this, i] { pick_loop(i); });
            inference_thread = std::thread([this] { inference_loop(); });
            auto* current = &slots[0];
            auto* next = &slots[1];
            auto gathered = gather(*current);
            if (!gathered) return tl::unexpected(gathered.error());
            submit(*current);
            while (current->count != 0) {
                gathered = gather(*next);
                if (!gathered) return tl::unexpected(gathered.error());
                // Queue it before backing up the current batch, so the provider
                // never waits for host-side publication.
                submit(*next);
                {
                    const auto began = limits.log_search_stats ? Clock::now() : Clock::time_point{};
                    std::unique_lock lock(work_mutex);
                    inference_changed.wait(lock, [&] {
                        return current->state == SlotState::Done ||
                               current->state == SlotState::Cancelled;
                    });
                    if (limits.log_search_stats) {
                        const std::lock_guard stats_lock(tree_mutex);
                        diagnostics.nn_wait += Clock::now() - began;
                    }
                    if (!current->error.empty()) return tl::unexpected(current->error.string());
                }
                // One inference thread guarantees provider serialization.
                auto applied = apply(*current);
                if (!applied) return tl::unexpected(applied.error());
                if (limits.on_progress) limits.on_progress();
                current->count = 0;
                {
                    const std::lock_guard lock(tree_mutex);
                    ++completed_batches;
                    smart_prune_locked();
                }
                wait_for_nps_limit();
                if (next->count == 0) {
                    gathered = gather(*next);
                    if (!gathered) return tl::unexpected(gathered.error());
                    submit(*next);
                }
                std::swap(current, next);
            }
        }
        const std::lock_guard lock(tree_mutex);
        if (!stopped)
            stopped = limits.nodes && tree->node(0).visits >= *limits.nodes
                          ? StopReason::NodeLimit
                          : StopReason::ExternalStop;
        running = false;
        finished = Clock::now();
        SearchResult result;
        if (limits.log_search_stats) {
            diagnostics.wall = finished - started;
            result.stats = diagnostics;
        }
        result.info = snapshot_locked();
        result.reason = *stopped;
        if (result.reason != StopReason::Terminal) {
            const auto* best = best_edge(0);
            if (best) result.best_move = best->move;
        }
        return result;
    }
};

Result<void> validate_search_tuning(const SearchConfig& config) {
    // Ranges from lc0 fd71a2d classic/params.cc Populate.
    const auto& gather = config.gather;
    if (gather.max_collision_events == 0 || gather.max_collision_events > 65536)
        return tl::unexpected(std::string{"MaxCollisionEvents requires 1..65536"});
    if (gather.max_collision_visits == 0 || gather.max_collision_visits > 100000000)
        return tl::unexpected(std::string{"MaxCollisionVisits requires 1..100000000"});
    if (gather.collision_scaling_start == 0 || gather.collision_scaling_start > 100000)
        return tl::unexpected(std::string{"MaxCollisionVisitsScalingStart requires 1..100000"});
    if (gather.collision_scaling_end > 100000000)
        return tl::unexpected(std::string{"MaxCollisionVisitsScalingEnd requires 0..100000000"});
    if (!std::isfinite(gather.collision_scaling_power) || gather.collision_scaling_power < 0.01 ||
        gather.collision_scaling_power > 100)
        return tl::unexpected(std::string{"MaxCollisionVisitsScalingPower requires 0.01..100"});
    if (!std::isfinite(gather.max_out_of_order_factor) || gather.max_out_of_order_factor < 0 ||
        gather.max_out_of_order_factor > 100)
        return tl::unexpected(std::string{"MaxOutOfOrderEvalsFactor requires 0..100"});
    if (gather.minimum_processing_work < 2 || gather.minimum_processing_work > 100000)
        return tl::unexpected(std::string{"MinimumProcessingWork requires 2..100000"});
    if (gather.minimum_per_task_processing == 0 || gather.minimum_per_task_processing > 100000)
        return tl::unexpected(std::string{"MinimumPerTaskProcessing requires 1..100000"});
    if (gather.minimum_picking_work == 0 || gather.minimum_picking_work > 100000)
        return tl::unexpected(std::string{"MinimumPickingWork requires 1..100000"});
    if (gather.minimum_remaining_picking_work > 100000)
        return tl::unexpected(std::string{"MinimumRemainingPickingWork requires 0..100000"});
    if (config.solid_tree_threshold == 0 || config.solid_tree_threshold > 2000000000U)
        return tl::unexpected(std::string{"SolidTreeThreshold requires 1..2000000000"});
    if (!std::isfinite(config.noise_epsilon) || config.noise_epsilon < 0 ||
        config.noise_epsilon > 1)
        return tl::unexpected(std::string{"DirichletNoiseEpsilon requires 0..1"});
    if (!std::isfinite(config.noise_alpha) || config.noise_alpha < 0 ||
        config.noise_alpha > 10000000 || (config.noise_epsilon != 0 && config.noise_alpha == 0))
        return tl::unexpected(
            std::string{"DirichletNoiseAlpha requires 0..10000000, and above 0 with noise on"});
    const auto& temperature = config.temperature;
    if (!std::isfinite(temperature.temperature) || temperature.temperature < 0 ||
        temperature.temperature > 100 || temperature.decay_moves > 640 ||
        temperature.decay_delay_moves > 100 || temperature.cutoff_move > 1000 ||
        !std::isfinite(temperature.endgame) || temperature.endgame < 0 ||
        temperature.endgame > 100 || !std::isfinite(temperature.value_cutoff) ||
        temperature.value_cutoff < 0 || temperature.value_cutoff > 100 ||
        !std::isfinite(temperature.visit_offset) || temperature.visit_offset < -1000 ||
        temperature.visit_offset > 1000)
        return tl::unexpected(std::string{"A temperature option is outside its range"});
    if (config.cache_history_length > 7)
        return tl::unexpected(std::string{"CacheHistoryLength requires 0..7"});
    if (gather.max_prefetch > 1024 || gather.idling_minimum_work > 10000 ||
        gather.thread_idling_threshold > 128)
        return tl::unexpected(std::string{"Search gather option outside its range"});
    if (!std::isfinite(config.draw_score) || config.draw_score < -1 || config.draw_score > 1)
        return tl::unexpected(std::string{"DrawScore requires -1..1"});
    if (!std::isfinite(config.nps_limit) || config.nps_limit < 0 || config.nps_limit > 1e6)
        return tl::unexpected(std::string{"NodesPerSecondLimit requires 0..1000000"});
    return {};
}

SearchEngine::SearchEngine(FactoryToken, std::unique_ptr<Impl> implementation)
    : impl_(std::move(implementation)) {}
SearchEngine::~SearchEngine() = default;
Result<std::unique_ptr<SearchEngine>> SearchEngine::create(const SearchConfig& config,
                                                           const chess::AttackTables& attacks,
                                                           nn::Backend& backend,
                                                           nn::LegalCache* cache) {
    auto parameters = validate_selection_parameters(config.selection);
    if (!parameters) return tl::unexpected(parameters.error());
    if (auto valid = validate_moves_left_parameters(config.moves_left); !valid)
        return tl::unexpected(valid.error());
    if (auto valid = validate_draw_rate_parameters(config.draw_rate); !valid)
        return tl::unexpected(valid.error());
    // Zero helper threads is lc0's TaskWorkers 0: the coordinator does all the host work.
    if ((config.threads == 0 && config.deterministic) || config.threads > 128 ||
        config.maximum_batch == 0 || config.maximum_batch > 1024 || config.ram_limit_bytes == 0 ||
        config.max_depth == 0 || config.max_depth > kMaximumDepth)
        return tl::unexpected(
            std::string{"Invalid search worker, batch, RAM or depth configuration"});
    if (auto valid = validate_search_tuning(config); !valid) return tl::unexpected(valid.error());
    if (backend.metadata().input_format != 1)
        return tl::unexpected(std::string{"Search requires classical input format 1"});
    try {
        const auto count = config.deterministic ? 1 : config.threads;
        const auto slot_output_bytes =
            cache ? sizeof(nn::LegalOutput) + sizeof(nn::PolicyIndices) : sizeof(nn::NNOutput);
        const auto evaluator_bytes =
            cache ? sizeof(nn::LegalEvaluator) +
                        config.maximum_batch * (sizeof(nn::InputPlanes) + sizeof(nn::NNOutput) +
                                                sizeof(nn::LegalOutput) + sizeof(std::size_t)) +
                        6 * kAllocationAllowance
                  : 0;
        const auto minimum =
            sizeof(SearchEngine) + sizeof(Impl) +
            2 * config.maximum_batch *
                (sizeof(Impl::Request) + sizeof(nn::InputPlanes) + slot_output_bytes +
                 sizeof(std::size_t)) +
            evaluator_bytes +
            count * (sizeof(Impl::Worker) + sizeof(std::thread) + kThreadAllowance) +
            16 * kAllocationAllowance;
        if (minimum >= config.ram_limit_bytes)
            return tl::unexpected(
                std::string{"Search RAM limit cannot fit fixed pipeline scratch"});
        auto implementation = std::make_unique<Impl>(config, attacks, backend, cache);
        implementation->initialize_scratch();
        if (implementation->overhead >= config.ram_limit_bytes)
            return tl::unexpected(
                std::string{"Search RAM limit cannot fit fixed pipeline scratch"});
        return std::make_unique<SearchEngine>(FactoryToken{}, std::move(implementation));
    } catch (const std::exception& error) {
        return tl::unexpected(std::string{"Cannot create search: "} + error.what());
    }
}
Result<void> SearchEngine::set_position(const chess::Position& root,
                                        const nn::EncodingHistory& history, bool retain_visits) {
    std::unique_lock lifecycle(impl_->lifecycle, std::try_to_lock);
    if (!lifecycle.owns_lock())
        return tl::unexpected(std::string{"Search lifecycle operation already running"});
    if (!impl_->same_backend())
        return tl::unexpected(std::string{"Search backend identity changed"});
    if (!matches(root, history))
        return tl::unexpected(std::string{"Search root and NN history do not match"});
    const std::lock_guard lock(impl_->tree_mutex);
    impl_->limits.searchmoves_count = 0;
    const auto& existing_root = impl_->root;
    if (retain_visits && existing_root.has_value() && *existing_root == root &&
        impl_->root_history == history) {
        const auto invalidated = impl_->invalidate_root_terminal();
        if (!invalidated) return tl::unexpected(invalidated.error());
        return {};
    }
    impl_->overhead = impl_->charge_without_tree();
    const auto fixed_charge = impl_->overhead - impl_->history_charge();
    const auto additional = impl_->config.max_depth + 1;
    const auto per_context =
        (impl_->config.ram_limit_bytes - std::min(fixed_charge, impl_->config.ram_limit_bytes)) /
        (impl_->workers.size() + 1);
    const auto source_position_bytes = root.history_size() * sizeof(std::uint64_t);
    const auto reserve_bytes =
        additional * (sizeof(nn::HistoryFrame) + sizeof(std::uint64_t)) + 2 * kAllocationAllowance;
    if (source_position_bytes > per_context ||
        reserve_bytes > per_context - source_position_bytes ||
        history.frames().size() >
            (per_context - source_position_bytes - reserve_bytes) / sizeof(nn::HistoryFrame)) {
        impl_->clear_root();
        return tl::unexpected(
            std::string{"Search setup cleared old root: history exceeds RAM limit"});
    }
    try {
        // Position changes are accepted before a queued go. Keep the drained
        // arena and all context capacities instead of touching hundreds of MiB
        // outside that go's absolute deadline. No old statistics survive reset.
        const auto position_entries = root.history_size() + additional;
        const auto frame_entries = history.frames().size() + additional;
        const auto position_bytes = position_entries * sizeof(std::uint64_t) + kAllocationAllowance;
        const auto frame_bytes = frame_entries * sizeof(nn::HistoryFrame) + kAllocationAllowance;
        const auto peak_fits = [&](std::size_t tree_charge) {
            const auto budget = impl_->config.ram_limit_bytes;
            if (impl_->overhead > budget || tree_charge > budget - impl_->overhead) return false;
            auto projected = impl_->overhead + tree_charge;
            struct CapacityChange {
                std::size_t required = 0, previous = 0, temporary = 0;
            };
            const auto reserve_peak = [&](const CapacityChange& change) {
                if (change.required <= change.previous) return true;
                // vector::reserve allocates its complete new buffer before
                // releasing the old one. Check that peak, then its final delta.
                if (change.required + change.temporary > budget - projected) return false;
                projected += change.required - change.previous;
                return true;
            };
            const auto position_peak = [&](const auto& context) {
                return reserve_peak(
                    {.required = position_bytes,
                     .previous = context ? context->history_storage_bytes() : 0,
                     .temporary = context ? 0 : source_position_bytes + kAllocationAllowance});
            };
            if (!position_peak(impl_->root) ||
                !reserve_peak({.required = frame_bytes,
                               .previous = impl_->root_history.history_storage_bytes()}))
                return false;
            for (const auto& worker : impl_->workers)
                if (!position_peak(worker.position) ||
                    !reserve_peak({.required = frame_bytes,
                                   .previous = worker.history.history_storage_bytes()}))
                    return false;
            return true;
        };
        if (impl_->tree && !peak_fits(impl_->tree->stats().charged_bytes)) impl_->tree.reset();
        if (!peak_fits(0)) {
            impl_->clear_root();
            return tl::unexpected(std::string{
                "Search setup cleared old root: history replacement peak exceeds RAM limit"});
        }
        const auto reserve_position = [&](auto& context) -> Result<void> {
            if (!context) context = root;
            if (position_entries <= context->history_size()) return {};
            return context->reserve_history(position_entries - context->history_size());
        };
        const auto reserve_frames = [&](auto& context) -> Result<void> {
            if (frame_entries <= context.frames().size()) return {};
            return context.reserve_history(frame_entries - context.frames().size());
        };
        auto prepared = reserve_position(impl_->root);
        if (prepared) prepared = reserve_frames(impl_->root_history);
        for (auto& worker : impl_->workers) {
            if (prepared) prepared = reserve_position(worker.position);
            if (prepared) prepared = reserve_frames(worker.history);
        }
        if (prepared) {
            impl_->root = root;
            impl_->root_history = history;
            prepared = impl_->prepare_histories();
        }
        if (prepared && impl_->tree && impl_->overhead <= impl_->config.ram_limit_bytes &&
            impl_->tree->stats().charged_bytes <= impl_->config.ram_limit_bytes - impl_->overhead)
            prepared = impl_->tree->reset();
        else if (prepared) {
            impl_->tree.reset();
            prepared = impl_->resize_tree();
        }
        if (!prepared) {
            impl_->clear_root();
            return tl::unexpected(std::string{"Search setup cleared old root: "} +
                                  prepared.error());
        }
        impl_->inherited = 0;
        impl_->seldepth = 0;
        impl_->started = {};
        impl_->finished = {};
        return {};
    } catch (const std::exception& error) {
        impl_->clear_root();
        return tl::unexpected(std::string{"Search setup cleared old root: "} + error.what());
    }
}
Result<ReuseSummary> SearchEngine::advance(chess::Move move) {
    std::unique_lock lifecycle(impl_->lifecycle, std::try_to_lock);
    if (!lifecycle.owns_lock())
        return tl::unexpected(std::string{"Search lifecycle operation already running"});
    if (!impl_->same_backend())
        return tl::unexpected(std::string{"Search backend identity changed"});
    const std::lock_guard lock(impl_->tree_mutex);
    auto& root_position = impl_->root;
    if (!root_position.has_value() || !impl_->tree)
        return tl::unexpected(std::string{"Search has no root"});
    try {
        auto& current_root = *root_position;
        auto legal = chess::legal_moves(current_root, impl_->attacks);
        if (std::ranges::find(legal, move) == legal.end())
            return tl::unexpected(std::string{"Illegal search continuation"});
        NodeId child = kNoNode;
        for (const auto& edge : impl_->tree->edges(0))
            if (edge.move == move) child = edge.child;
        auto undo = current_root.make_move(move, impl_->attacks);
        auto appended = impl_->root_history.append(current_root);
        if (!appended) {
            current_root.unmake_move(move, undo);
            return tl::unexpected(appended.error());
        }
        auto prepared = impl_->prepare_histories();
        if (!prepared) {
            impl_->clear_root();
            return tl::unexpected(std::string{"Search continuation setup cleared root: "} +
                                  prepared.error());
        }
        ReuseSummary summary;
        if (impl_->overhead + impl_->tree->stats().charged_bytes > impl_->config.ram_limit_bytes) {
            impl_->tree.reset();
            prepared = impl_->resize_tree();
            if (!prepared) {
                impl_->clear_root();
                return tl::unexpected(prepared.error());
            }
            summary.kept_nodes = 1;
        } else if (child == kNoNode) {
            auto reset = impl_->tree->reset();
            if (!reset) return tl::unexpected(reset.error());
            summary.kept_nodes = 1;
        } else {
            auto retained = impl_->tree->re_root(child);
            if (!retained) return tl::unexpected(retained.error());
            summary = *retained;
        }
        const auto invalidated = impl_->invalidate_root_terminal();
        if (!invalidated) {
            impl_->clear_root();
            return tl::unexpected("Search continuation cleared root: " + invalidated.error());
        }
        if (*invalidated) summary = {.kept_nodes = 1, .kept_edges = 0, .reset_terminal_root = true};
        impl_->limits.searchmoves_count = 0;
        impl_->inherited = impl_->tree->node(0).visits;
        // A DAG has too many paths to walk; this search's own depth still counts from zero.
        impl_->seldepth = impl_->config.dag ? 0 : impl_->deepest_completed_locked();
        impl_->started = {};
        impl_->finished = {};
        return summary;
    } catch (const std::exception& error) {
        impl_->clear_root();
        return tl::unexpected(std::string{"Search continuation cleared root: "} + error.what());
    }
}
Result<SearchResult> SearchEngine::run(const SearchLimits& limits) {
    std::unique_lock lifecycle(impl_->lifecycle, std::try_to_lock);
    if (!lifecycle.owns_lock())
        return tl::unexpected(std::string{"Search lifecycle operation already running"});
    // This reset is the acceptance/linearization point for a new run. A stop
    // observed after acceptance belongs to this operation, including while the
    // borrowed backend identity is being validated. Never clear it in execute.
    impl_->external_stop.store(false, std::memory_order_relaxed);
    const std::stop_callback cancelled(limits.stop_token, [this]() noexcept { request_stop(); });
    if (!impl_->root || !impl_->tree) return tl::unexpected(std::string{"Search has no root"});
    if (!impl_->same_backend())
        return tl::unexpected(std::string{"Search backend identity changed"});
    if (!std::isfinite(limits.smart_pruning_factor) || limits.smart_pruning_factor < 0.0 ||
        limits.smart_pruning_factor > 10.0 || limits.minimum_smart_pruning_batches > 10000)
        return tl::unexpected(std::string{"Invalid smart-pruning configuration"});
    if ((limits.nodes && *limits.nodes == 0) ||
        (limits.movetime && limits.movetime->count() <= 0) ||
        (limits.maximum_batch &&
         (*limits.maximum_batch == 0 || *limits.maximum_batch > impl_->config.maximum_batch)) ||
        (limits.depth && (*limits.depth == 0 || *limits.depth > impl_->config.max_depth)))
        return tl::unexpected(std::string{"Invalid search node, time, batch or depth limit"});
    if (limits.searchmoves_count > limits.searchmoves.size())
        return tl::unexpected(std::string{"Root searchmoves exceeds capacity"});
    if (limits.searchmoves_count != 0) {
        const auto& root_position = impl_->root;
        if (!root_position.has_value()) return tl::unexpected(std::string{"Search has no root"});
        auto probe = *root_position;
        const auto legal = chess::legal_moves(probe, impl_->attacks);
        for (std::size_t i = 0; i < limits.searchmoves_count; ++i) {
            const auto move = limits.searchmoves.at(i);
            if (std::ranges::find(legal, move) == legal.end())
                return tl::unexpected(std::string{"Root searchmoves contains an illegal move"});
            for (std::size_t prior = 0; prior < i; ++prior)
                if (limits.searchmoves.at(prior) == move)
                    return tl::unexpected(
                        std::string{"Root searchmoves contains a duplicate move"});
        }
    }
    try {
        auto result = impl_->execute(limits);
        if (result) return result;
        const std::lock_guard lock(impl_->tree_mutex);
        impl_->running = false;
        impl_->finished = Clock::now();
        return result;
    } catch (const std::exception& error) {
        const std::lock_guard lock(impl_->tree_mutex);
        impl_->running = false;
        impl_->finished = Clock::now();
        return tl::unexpected(std::string{"Search failed: "} + error.what());
    }
}
void SearchEngine::request_stop() noexcept {
    impl_->external_stop.store(true, std::memory_order_relaxed);
}
thread_local SnapshotCost* SnapshotCostScope::active_ = nullptr;
SnapshotCostScope::SnapshotCostScope(SnapshotCost& cost) noexcept : previous_(active_) {
    active_ = &cost;
}
SnapshotCostScope::~SnapshotCostScope() { active_ = previous_; }
SnapshotCost* SnapshotCostScope::current() noexcept { return active_; }

SearchInfo SearchEngine::snapshot() const {
    const auto lock = impl_->snapshot_lock(nullptr);
    return impl_->snapshot_locked();
}
SearchResult SearchEngine::result_snapshot(SnapshotCost* cost) const {
    const auto lock = impl_->snapshot_lock(cost);
    SearchResult result;
    result.info = impl_->snapshot_locked();
    result.reason = impl_->stopped.value_or(StopReason::ExternalStop);
    if (impl_->tree) {
        const auto* best = impl_->best_edge(0);
        if (best) result.best_move = best->move;
    }
    return result;
}
RootStatistics SearchEngine::root_moves(SnapshotCost* cost) const {
    const auto lock = impl_->snapshot_lock(cost);
    RootStatistics result;
    if (!impl_->tree) return result;
    for (const auto& edge : impl_->tree->edges(0)) {
        if (!impl_->allowed(edge.move)) continue;
        auto& item = result.moves.at(result.size++);
        item.move = edge.move;
        item.prior = decode_prior(edge.prior_bits);
        if (edge.child != kNoNode) {
            const auto& child = impl_->tree->node(edge.child);
            item.visits = child.visits;
            item.child_proof = child.proof;
            item.in_flight = child.in_flight;
            item.terminal = child.state == NodeState::Terminal;
            if (child.visits != 0) {
                item.q = -child.value_sum / static_cast<double>(child.visits);
                item.draw = child.draw_sum / static_cast<double>(child.visits);
                item.draw_score_share = impl_->draw_score_at(0) * item.draw;
                if (child.moves_left_samples == child.visits)
                    item.moves_left =
                        child.moves_left_sum / static_cast<double>(child.moves_left_samples);
            }
        }
    }
    return result;
}
std::vector<std::string> SearchEngine::verbose_move_stats(bool chess960) const {
    // Output format of lc0 fd71a2d classic/search.cc GetVerboseStats, for the root. Mouse keeps
    // each node's rescaled network value, so V is that rather than a fresh cache lookup, and it
    // stores exact proofs, so a node prints (T) or nothing rather than one-sided (W) or (L).
    const auto lock = impl_->snapshot_lock(nullptr);
    std::vector<std::string> lines;
    if (!impl_->tree || !impl_->root) return lines;
    const auto& tree = *impl_->tree;
    const auto& node = tree.node(0);
    const auto& config = impl_->config;
    const auto side = impl_->root->side_to_move();
    const double draw_score = impl_->draw_score_at(0);
    const auto edges = tree.edges(0);
    double mass = 0;
    for (const auto& edge : edges)
        if (edge.child != kNoNode && tree.node(edge.child).visits != 0)
            mass += decode_prior(edge.prior_bits);
    const double parent_q =
        node.visits == 0
            ? 0
            : (node.value_sum + draw_score * node.draw_sum) / static_cast<double>(node.visits);
    const double fpu = first_play_urgency(config.selection, true, parent_q, mass);
    const double multiplier = exploration_multiplier(config.selection, impl_->selection_visits(0));
    const auto rescale = impl_->wdl_rescale;
    const auto contempt_side = impl_->limits.contempt_side;
    const bool holds = contempt_side != ContemptSide::None &&
                       (contempt_side == ContemptSide::White) == (side == chess::Color::White);
    const auto reported = [&](float& wl, float& draw) {
        if (contempt_side == ContemptSide::None) return;
        rescale_wdl(wl, draw,
                    {.ratio = rescale.ratio,
                     .diff = rescale.diff * static_cast<float>(config.draw_rate.eval_objectivity),
                     .sign = holds ? 1.0F : -1.0F,
                     .invert = true,
                     .max_s = static_cast<float>(config.draw_rate.max_s)});
    };
    const auto head = [](std::ostringstream& out, std::string_view label, std::uint64_t index,
                         std::uint64_t visits, std::uint64_t in_flight, double policy) {
        out << std::fixed << std::left << std::setw(5) << std::setprecision(0) << label << ' ';
        out << '(' << std::setw(4) << index << ") " << std::right;
        out << "N: " << std::setw(7) << visits << ' ';
        out << "(+" << std::setw(2) << in_flight << ") ";
        out << "(P: " << std::setw(5) << std::setprecision(policy >= 0.99995 ? 1 : 2)
            << policy * 100 << "%) ";
    };
    const auto stats = [&](std::ostringstream& out, const Node* shown, double sign) {
        if (!shown || shown->visits == 0) {
            out << "(WL:  -.-----) (D: -.---) (M:  -.-) ";
            out << "(Q: " << std::setw(8) << std::setprecision(5) << fpu << ") ";
            return;
        }
        const auto visits = static_cast<double>(shown->visits);
        auto wl = static_cast<float>(sign * shown->value_sum / visits);
        auto draw = static_cast<float>(shown->draw_sum / visits);
        reported(wl, draw);
        const auto moves = shown->moves_left_samples == shown->visits
                               ? shown->moves_left_sum / visits
                               : 0.0;
        out << "(WL: " << std::setw(8) << std::setprecision(5) << wl << ") ";
        out << "(D: " << std::setw(5) << std::setprecision(3) << draw << ") ";
        out << "(M: " << std::setw(4) << std::setprecision(1) << moves << ") ";
        out << "(Q: " << std::setw(8) << std::setprecision(5) << wl + draw_score * draw << ") ";
    };
    const auto tail = [](std::ostringstream& out, const Node* shown, double sign) {
        if (shown && shown->visits != 0)
            out << "(V: " << std::setw(7) << std::setprecision(4) << sign * shown->leaf_q << ") ";
        else
            out << "(V:  -.----) ";
        if (shown && shown->proof.outcome != ProofOutcome::Unknown) out << "(T) ";
    };
    struct Row {
        std::uint64_t visits = 0;
        double score = 0;
        std::size_t index = 0;
    };
    std::vector<Row> rows;
    rows.reserve(edges.size());
    const auto q_of = [&](const Node* child) {
        return child && child->visits != 0 ? (draw_score * child->draw_sum - child->value_sum) /
                                                 static_cast<double>(child->visits)
                                           : fpu;
    };
    const auto u_of = [&](const Edge& edge, const Node* child) {
        const auto started = child ? child->visits + child->in_flight : 0;
        return multiplier * decode_prior(edge.prior_bits) / (1.0 + static_cast<double>(started));
    };
    for (std::size_t i = 0; i < edges.size(); ++i) {
        const Node* child = edges[i].child == kNoNode ? nullptr : &tree.node(edges[i].child);
        rows.push_back({.visits = child ? child->visits : 0,
                        .score = q_of(child) + u_of(edges[i], child),
                        .index = i});
    }
    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) {
        return a.visits != b.visits ? a.visits < b.visits : a.score < b.score;
    });
    lines.reserve(rows.size() + 1);
    for (const auto& row : rows) {
        const auto& edge = edges[row.index];
        const Node* child = edge.child == kNoNode ? nullptr : &tree.node(edge.child);
        const double q = q_of(child);
        const double u = u_of(edge, child);
        const double m = config.moves_left_utility && child
                             ? moves_left_utility(node, *child, config.moves_left)
                             : 0.0;
        const auto index = nn::PolicyMap::index(edge.move, side);
        std::ostringstream out;
        head(out, chess::move_to_uci(edge.move, chess960), index ? *index : 0,
             child ? child->visits : 0, child ? child->in_flight : 0, decode_prior(edge.prior_bits));
        stats(out, child, -1.0);
        out << "(U: " << std::setw(6) << std::setprecision(5) << u << ") ";
        out << "(S: " << std::setw(8) << std::setprecision(5) << q + u + m << ") ";
        tail(out, child, -1.0);
        lines.push_back(out.str());
    }
    std::ostringstream out;
    head(out, "node ", edges.size(), node.visits, node.in_flight, mass);
    stats(out, &node, 1.0);
    tail(out, &node, 1.0);
    lines.push_back(out.str());
    return lines;
}
std::optional<chess::Move> SearchEngine::final_move(std::uint64_t game_ply) {
    const auto lock = impl_->snapshot_lock(nullptr);
    return impl_->final_move_locked(game_ply);
}
RootStatistics SearchEngine::all_root_moves() const {
    const auto lock = impl_->snapshot_lock(nullptr);
    RootStatistics result;
    if (!impl_->tree) return result;
    for (const auto& edge : impl_->tree->edges(0)) {
        auto& item = result.moves.at(result.size++);
        item.move = edge.move;
        item.prior = decode_prior(edge.prior_bits);
        if (edge.child == kNoNode) continue;
        const auto& child = impl_->tree->node(edge.child);
        item.visits = child.visits;
        item.child_proof = child.proof;
        if (child.visits != 0) {
            item.q = -child.value_sum / static_cast<double>(child.visits);
            item.draw = child.draw_sum / static_cast<double>(child.visits);
            if (child.moves_left_samples == child.visits)
                item.moves_left =
                    child.moves_left_sum / static_cast<double>(child.moves_left_samples);
        }
    }
    return result;
}
PrincipalVariation SearchEngine::principal_variation(std::optional<chess::Move> first,
                                                     SnapshotCost* cost) const {
    const auto lock = impl_->snapshot_lock(cost);
    return impl_->pv_locked(first);
}
}  // namespace mouse::search
