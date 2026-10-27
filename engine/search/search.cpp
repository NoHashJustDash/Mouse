// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/search/search.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <limits>
#include <mutex>
#include <new>
#include <span>
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
#include "engine/search/prior.h"
#include "engine/search/pruning.h"
#include "engine/search/rules.h"

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
        Reservation reservation;
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
        SlotState state = SlotState::Idle;
        std::uint64_t sequence = 0;
        std::atomic<std::size_t> next{0};
        std::atomic<bool> halt{false};
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
    // Picking runs under tree_mutex on the gathering thread only.
    struct Picker {
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
            bool eligible = false;
        };
        std::array<EdgeTerms, kMaximumLegalMoves> terms{};
        Path path;
        std::vector<std::size_t> free_rows;
        std::size_t free_used = 0;
        std::uint64_t physical_left = 0;
        // lc0 max_limit / NodeToProcess::maxvisit: a collision that takes the
        // whole walk may grow to the smallest visits-to-change-best seen.
        std::uint32_t walk_limit = 0;
        std::uint64_t max_limit = 0;
        std::size_t upsize_record = 0;
        std::uint64_t upsize_cap = 0;
    };
    struct Worker {
        // Bounded worker-local hot-path scratch, charged through sizeof(Worker).
        std::optional<chess::Position> position;
        nn::EncodingHistory history;
        std::array<chess::Undo, kMaximumDepth> undo{};
        std::array<std::uint64_t, kMaximumDepth + 1> keys{};
    };
    SearchConfig config;
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
    std::thread inference_thread;
    std::unique_ptr<Tree> tree;
    std::optional<chess::Position> root;
    nn::EncodingHistory root_history;
    std::atomic<bool> external_stop{false};
    bool shutdown = false;
    std::size_t epoch = 0, remaining_workers = 0;
    Slot* gathering = nullptr;
    std::uint64_t next_sequence = 0;
    SearchLimits limits;
    Clock::time_point started{}, finished{};
    std::optional<Clock::time_point> deadline;
    std::uint64_t inherited = 0, completed_batches = 0;
    SmartPruning pruning;
    SearchStats diagnostics;  // tree_mutex protects every diagnostic update.
    std::size_t seldepth = 0, overhead = 0;
    bool running = false;
    std::optional<StopReason> stopped;
    Impl(const SearchConfig& selected, const chess::AttackTables& table, nn::Backend& network,
         nn::LegalCache* cache)
        : config(selected),
          attacks(table),
          backend(network),
          backend_identity(network.info()),
          metadata(network.metadata()) {
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
                      slot.picked.capacity() * sizeof(std::size_t) + 9 * kAllocationAllowance;
        }
        result += picker.levels.capacity() * sizeof(PickLevel) +
                  picker.entries.capacity() * sizeof(PickEntry) +
                  picker.free_rows.capacity() * sizeof(std::size_t) + 3 * kAllocationAllowance;
        if (evaluator)
            result +=
                sizeof(nn::LegalEvaluator) + evaluator->scratch_bytes() + kAllocationAllowance;
        result += workers.capacity() * sizeof(Worker) + kAllocationAllowance;
        result += worker_threads.capacity() * sizeof(std::thread) + kAllocationAllowance;
        result += (workers.size() + 1) * kThreadAllowance;
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
            slot.collisions.resize(kMaximumCollisionEvents);
            slot.collision_nodes.resize(kMaximumCollisionEvents * config.max_depth);
            slot.inputs = std::vector<nn::InputPlanes>(config.maximum_batch);
            if (!evaluator) slot.outputs = std::vector<nn::NNOutput>(config.maximum_batch);
            slot.rows = std::vector<std::size_t>(config.maximum_batch);
            slot.picked = std::vector<std::size_t>(config.maximum_batch);
            if (evaluator) {
                slot.legal_outputs.resize(config.maximum_batch);
                slot.policies.resize(config.maximum_batch);
            }
        }
        // A pick walk descends at most max_depth edges below the root.
        picker.levels = std::vector<PickLevel>(config.max_depth + 1);
        picker.entries = std::vector<PickEntry>(config.maximum_batch + config.max_depth + 1);
        picker.free_rows = std::vector<std::size_t>(config.maximum_batch);
        workers.resize(config.deterministic ? 1 : config.threads);
        worker_threads.reserve(workers.size());
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
        auto fresh = Tree::create(available - available / 8);
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
    void cancel_round_collisions_locked(Slot& slot, std::size_t first) noexcept {
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
        slot.collision_nodes_used = slot.collisions.at(first).offset;
        slot.collision_count = first;
    }
    void release_slot_locked(Slot& slot) {
        cancel_collisions_locked(slot);
        for (std::size_t i = 0; i < config.maximum_batch; ++i) {
            auto& request = slot.requests[i];
            request.reservation.release_locked();
            request.ready = false;
        }
        slot.count = 0;
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
        // lc0 classic/params.cc MaxOutOfOrderEvalsFactor = 2.4f; search.h
        // converts the scaled minibatch size to an integer by truncation.
        return static_cast<std::size_t>(
            2.4F * static_cast<float>(limits.maximum_batch.value_or(config.maximum_batch)));
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
    PickKind classify_locked(PickLocation location) {
        const auto [id, depth] = location;
        auto& node = tree->node(id);
        if (node.state == NodeState::Pending) return PickKind::Collision;
        if (node.state == NodeState::Frontier && depth < ceiling())
            node.state = NodeState::Unexpanded;
        if (node.state == NodeState::Unexpanded) return PickKind::Fresh;
        if (node.state == NodeState::Terminal || node.state == NodeState::Frontier ||
            depth == ceiling())
            return PickKind::Known;
        return PickKind::Interior;
    }
    // Reserves one real visit along picker.path (root..leaf) in a free row.
    void start_request_locked(Slot& slot, bool known) {
        const auto& path = picker.path;
        for (std::size_t i = 0; i < path.size; ++i)
            if (tree->node(path.nodes.at(i)).in_flight == std::numeric_limits<std::uint32_t>::max())
                throw std::overflow_error("Search in-flight counter exhausted");
        const auto row = picker.free_rows.at(picker.free_used++);
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
        for (std::size_t i = 0; i < path.size; ++i) ++tree->node(path.nodes.at(i)).in_flight;
        if (!known) tree->node(path.nodes.at(path.size - 1)).state = NodeState::Pending;
        slot.picked.at(slot.picked_count++) = row;
        --picker.physical_left;
    }
    // lc0 NodeToProcess::Collision: `visits` are reserved on the leaf's
    // ancestors only (the first `ancestors` nodes of picker.path), never
    // evaluated, and cancelled after the batch backs up.
    void add_collision_locked(Slot& slot, std::size_t ancestors, std::uint32_t visits) {
        if (visits == 0) return;
        const auto& path = picker.path;
        if (slot.collision_count == slot.collisions.size() ||
            ancestors > slot.collision_nodes.size() - slot.collision_nodes_used) {
            // Bounded record storage (lc0 MaxCollisionEvents) ends this gather.
            slot.halt.store(true, std::memory_order_relaxed);
            return;
        }
        for (std::size_t i = 0; i < ancestors; ++i)
            if (visits >
                std::numeric_limits<std::uint32_t>::max() - tree->node(path.nodes.at(i)).in_flight)
                throw std::overflow_error("Collision virtual-visit counter exhausted");
        for (std::size_t i = 0; i < ancestors; ++i) {
            tree->node(path.nodes.at(i)).in_flight += visits;
            slot.collision_nodes.at(slot.collision_nodes_used + i) = path.nodes.at(i);
        }
        if (visits == picker.walk_limit && picker.max_limit > visits) {
            picker.upsize_record = slot.collision_count;
            picker.upsize_cap = picker.max_limit;
        }
        slot.collisions.at(slot.collision_count++) = {
            .offset = slot.collision_nodes_used, .size = ancestors, .visits = visits};
        slot.collision_nodes_used += ancestors;
        slot.collision_visits += visits;
        if (ancestors != 0) slot.collision_root_visits += visits;
        if (limits.log_search_stats) ++diagnostics.collisions;
    }
    // Starts the one leaf visit lc0's TryStartScoreUpdate allows for `child`
    // (picker.path ends at its parent). Returns whether a request started.
    bool start_leaf_locked(Slot& slot, std::size_t depth, Edge& edge, PickKind kind) {
        if (kind != PickKind::Fresh && kind != PickKind::Known) return false;
        if (picker.physical_left == 0) return false;
        if (edge.child == kNoNode) {
            auto allocated = tree->allocate_node();
            if (!allocated) {
                stopped = StopReason::MemoryLimit;
                slot.halt.store(true, std::memory_order_relaxed);
                return false;
            }
            edge.child = *allocated;
        }
        auto& path = picker.path;
        path.nodes.at(depth + 1) = edge.child;
        path.moves.at(depth) = edge.move;
        path.size = depth + 2;
        start_request_locked(slot, kind == PickKind::Known);
        path.size = depth + 1;
        return true;
    }
    // lc0 fd71a2d classic/search.cc PickNodesToExtendTask: give `limit` visits
    // to the edges of an expanded node (picker.path ends at it) in rounds. Each
    // round the best PUCT edge receives the visits it needs before the
    // runner-up overtakes it. An expanded child's started count grows by all of
    // them; a fresh or known leaf starts one real visit now, and what remains is
    // placed below (an expanded child) or becomes a collision (a leaf).
    struct PickDistribution {
        std::size_t depth;
        std::uint32_t limit;
        std::size_t first_entry;
    };
    void distribute_locked(Slot& slot, PickLevel& level, PickDistribution distribution) {
        auto [depth, limit, first_entry] = distribution;
        const auto id = level.node;
        const auto& node = tree->node(id);
        const auto edges = tree->edges(id);
        if (edges.size() > kMaximumLegalMoves)
            throw std::logic_error("Search node has more edges than legal chess moves");
        const bool at_root = depth == 0;
        const double parent_q =
            node.visits == 0 ? 0 : node.value_sum / static_cast<double>(node.visits);
        double mass = 0;
        for (const auto& edge : edges)
            if ((!at_root || allowed(edge.move)) && edge.child != kNoNode &&
                tree->node(edge.child).visits != 0)
                mass += decode_prior(edge.prior_bits);
        const double fpu = first_play_urgency(config.selection, at_root, parent_q, mass);
        const double multiplier = exploration_multiplier(config.selection, node.visits);
        const auto terms = std::span{picker.terms}.first(edges.size());
        for (std::size_t index = 0; index < edges.size(); ++index) {
            const auto& edge = edges[index];
            const Node* child = edge.child == kNoNode ? nullptr : &tree->node(edge.child);
            double utility = child && child->visits != 0
                                 ? -child->value_sum / static_cast<double>(child->visits)
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
                utility += moves_left_utility(node, *child);
            // Pending children remain PUCT candidates (lc0 classic/node.h
            // EdgeAndNode::GetU); reservations affect only NStarted.
            terms[index] = {.utility = utility,
                            .exploration = multiplier * decode_prior(edge.prior_bits),
                            .started = child ? child->visits + child->in_flight : 0,
                            .entry = std::numeric_limits<std::uint16_t>::max(),
                            .eligible = !at_root || allowed(edge.move)};
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
                picker.max_limit = std::min<std::uint64_t>(
                    picker.max_limit,
                    collision_multiplicity(selection, std::numeric_limits<std::uint32_t>::max()));
                granted = collision_multiplicity(selection, limit);
            }
            limit -= granted;
            ++rounds;
            auto& edge = edges[chosen];
            const auto kind = edge.child == kNoNode
                                  ? PickKind::Fresh
                                  : classify_locked({.id = edge.child, .depth = depth + 1});
            if (kind == PickKind::Interior) {
                terms[chosen].started += granted;
            } else if (start_leaf_locked(slot, depth, edge, kind)) {
                ++terms[chosen].started;
                --granted;
            }
            if (granted == 0) continue;
            auto& slot_index = terms[chosen].entry;
            if (slot_index == std::numeric_limits<std::uint16_t>::max()) {
                if (level.end == picker.entries.size())
                    throw std::logic_error("Search pick walk exceeded its entry bound");
                slot_index = static_cast<std::uint16_t>(level.end - level.begin);
                picker.entries[level.end++] = {.index = static_cast<std::uint16_t>(chosen),
                                               .visits = 0};
            }
            picker.entries[level.begin + slot_index].visits += granted;
        }
        // Place visits in edge order, as lc0 walks visits_to_perform.
        std::sort(picker.entries.begin() + static_cast<std::ptrdiff_t>(level.begin),
                  picker.entries.begin() + static_cast<std::ptrdiff_t>(level.end),
                  [](const PickEntry& a, const PickEntry& b) { return a.index < b.index; });
        // Rounds after the first reuse this node's PUCT terms without recomputing them.
        if (limits.log_search_stats && rounds > 1) diagnostics.selection_hits += rounds - 1;
    }
    // One walk places up to `limit` visits (lc0 PickNodesToExtend). Picking
    // runs under tree_mutex; request processing happens afterwards, unlocked.
    void pick_locked(Slot& slot, std::uint32_t limit) {
        auto& path = picker.path;
        path.size = 1;
        path.nodes[0] = 0;
        picker.walk_limit = limit;
        picker.max_limit = std::numeric_limits<std::uint64_t>::max();
        picker.upsize_record = slot.collisions.size();
        picker.upsize_cap = 0;
        const auto root_kind = classify_locked({.id = 0, .depth = 0});
        if (root_kind != PickKind::Interior) {
            // The root has no parent round, so it starts its own leaf visit.
            if ((root_kind == PickKind::Fresh || root_kind == PickKind::Known) &&
                picker.physical_left != 0) {
                start_request_locked(slot, root_kind == PickKind::Known);
                --limit;
            }
            add_collision_locked(slot, 0, limit);
            return;
        }
        picker.levels.at(0).node = 0;
        distribute_locked(slot, picker.levels.at(0),
                          {.depth = 0, .limit = limit, .first_entry = 0});
        std::size_t depth = 0;
        for (;;) {
            auto& level = picker.levels.at(depth);
            const auto edges = tree->edges(level.node);
            bool descended = false;
            while (level.next < level.end && !slot.halt.load(std::memory_order_relaxed)) {
                const auto entry = picker.entries[level.next++];
                const auto visits = entry.visits;
                const auto& edge = edges[entry.index];
                path.size = depth + 1;
                if (edge.child == kNoNode) {
                    // A fresh leaf that could not start (node allowance spent)
                    // has no node yet; its remainder is held on the path to it.
                    add_collision_locked(slot, depth + 1, visits);
                    continue;
                }
                path.nodes.at(depth + 1) = edge.child;
                path.moves.at(depth) = edge.move;
                path.size = depth + 2;
                if (classify_locked({.id = edge.child, .depth = depth + 1}) != PickKind::Interior) {
                    add_collision_locked(slot, depth + 1, visits);
                    continue;
                }
                auto& next = picker.levels.at(depth + 1);
                next.node = edge.child;
                distribute_locked(slot, next,
                                  {.depth = depth + 1, .limit = visits, .first_entry = level.end});
                ++depth;
                descended = true;
                break;
            }
            if (descended) continue;
            if (depth == 0) return;
            --depth;
        }
    }
    struct Value {
        double q = 0, draw = 0;
        std::optional<double> moves_left;
    };
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
    void backup_locked(Request& request, Value value, bool contextual) {
        const auto began = limits.log_search_stats ? Clock::now() : Clock::time_point{};
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
        for (std::size_t i = request.path.size; i > 0; --i) {
            auto& node = tree->node(request.path.nodes.at(i - 1));
            ++node.visits;
            if (i != request.path.size && proof_changed) {
                const auto previous = node.proof;
                refresh_proof_locked(request.path.nodes.at(i - 1));
                proof_changed = node.proof != previous;
            }
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
            node.contextual_draw = node.contextual_draw || contextual;
            backup_moves_left_sample(node, value.moves_left);
            value.q = -value.q;
        }
        seldepth = std::max(seldepth, request.path.size - 1);
        if (limits.depth && request.path.size - 1 >= *limits.depth)
            stopped = StopReason::DepthLimit;
        request.reservation.release_locked();
        if (limits.log_search_stats) diagnostics.backup += Clock::now() - began;
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
    [[nodiscard]] Result<void> publish_value_locked(Request& request, Value value) {
        auto& leaf = tree->node(request.path.nodes.at(request.path.size - 1));
        if (request.path.size - 1 == ceiling())
            leaf.state = NodeState::Frontier;
        else {
            auto expanded = tree->expand(request.path.nodes.at(request.path.size - 1),
                                         {request.seeds.data(), request.legal.size()});
            if (!expanded) return tl::unexpected(expanded.error());
        }
        leaf.leaf_q = static_cast<float>(value.q);
        leaf.leaf_draw = static_cast<float>(value.draw);
        leaf.leaf_moves_left = value.moves_left.has_value()
                                   ? std::optional<float>{static_cast<float>(*value.moves_left)}
                                   : std::nullopt;
        backup_locked(request, value, false);
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
        const double sum = static_cast<double>(output.wdl[0]) + output.wdl[1] + output.wdl[2];
        const Value value{.q = (static_cast<double>(output.wdl[0]) - output.wdl[2]) / sum,
                          .draw = output.wdl[1] / sum,
                          .moves_left = output.moves_left};
        const std::lock_guard lock(tree_mutex);
        if (slot.out_of_order_count >= out_of_order_limit()) return false;
        const auto statistics = tree->stats();
        if (request.path.size - 1 < ceiling() &&
            count > statistics.edge_capacity - statistics.edge_count) {
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
        if (request.known) {
            const std::lock_guard lock(tree_mutex);
            const auto& leaf = tree->node(request.path.nodes.at(request.path.size - 1));
            backup_locked(
                request,
                {.q = leaf.leaf_q, .draw = leaf.leaf_draw, .moves_left = leaf.leaf_moves_left},
                leaf.contextual_draw);
            ++slot.out_of_order_count;
            if (limits.log_search_stats) ++diagnostics.out_of_order;
            return {};
        }
        Replay replay(worker, request.path, attacks);
        auto replayed = replay.apply();
        if (!replayed) return tl::unexpected(replayed.error());
        auto status =
            classify_leaf(replay.position, attacks, {worker.keys.data(), request.path.size - 1},
                          request.path.size == 1 ? LeafContext::Root : LeafContext::NonRoot);
        if (!status) return tl::unexpected(status.error());
        if (status->reason != TerminalReason::None) {
            const std::lock_guard lock(tree_mutex);
            auto& leaf = tree->node(request.path.nodes.at(request.path.size - 1));
            leaf.state = NodeState::Terminal;
            leaf.proof = config.terminal_proof ? proof_from_terminal(status->reason) : NodeProof{};
            leaf.leaf_q = status->q;
            leaf.leaf_draw = status->draw;
            // Mouse's explicit terminal-M contract also covers contextual
            // tree draws. Lc0 uses cycle length for that distinct case.
            leaf.leaf_moves_left = 0.0F;
            backup_locked(request, {.q = status->q, .draw = status->draw, .moves_left = 0.0},
                          status->contextual);
            ++slot.out_of_order_count;
            if (limits.log_search_stats) ++diagnostics.out_of_order;
            return {};
        }
        auto encoded = nn::encode_classical(worker.history.frames(), metadata.input_format);
        if (!encoded) return tl::unexpected(encoded.error());
        request.legal = status->legal_moves;
        request.side = replay.position.side_to_move();
        input = *encoded;
        // Deterministic mode preserves identical cache-on/off batch barriers.
        if (legal_cache && !config.deterministic) {
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
            lock.unlock();
            while (!slot.halt.load(std::memory_order_relaxed)) {
                const auto next = slot.next.fetch_add(1, std::memory_order_relaxed);
                if (next >= slot.picked_count) break;
                const auto row = slot.picked.at(next);
                try {
                    auto result = process_request(slot, workers.at(index), slot.requests[row],
                                                  slot.inputs[row]);
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
            lock.lock();
            if (--remaining_workers == 0) phase_done.notify_one();
        }
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
                    if (evaluator)
                        return evaluator->evaluate(
                            {selected->inputs.data(), selected->count},
                            {selected->policies.data(), selected->count},
                            {selected->legal_outputs.data(), selected->count},
                            limits.log_search_stats ? &timing : nullptr);
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
        if (picker.upsize_record >= slot.collision_count) return;
        auto& collision = slot.collisions.at(picker.upsize_record);
        const auto target = std::min<std::uint64_t>(picker.upsize_cap, collisions_left);
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
    // Runs the picked rows on the worker threads and waits for all of them.
    [[nodiscard]] Result<void> process_picked(Slot& slot) {
        std::unique_lock lock(work_mutex);
        slot.next = 0;
        gathering = &slot;
        remaining_workers = workers.size();
        ++epoch;
        work_changed.notify_all();
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
            slot.collision_budget = collision_visit_budget(visits, remaining);
            slot.collision_visits = 0;
            slot.out_of_order_count = 0;
        }
        const auto capacity = limits.maximum_batch.value_or(config.maximum_batch);
        for (;;) {
            std::size_t first_collision = 0, out_of_order_before = 0;
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
                picker.free_used = 0;
                picker.physical_left = physical_allowance_locked();
                slot.picked_count = 0;
                first_collision = slot.collision_count;
                out_of_order_before = slot.out_of_order_count;
                collision_visits_before = slot.collision_visits;
                pick_locked(slot, static_cast<std::uint32_t>(limit));
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
            auto processed = process_picked(slot);
            if (!processed) return processed;
            const std::lock_guard tree_lock(tree_mutex);
            if (slot.out_of_order_count != out_of_order_before)
                cancel_round_collisions_locked(slot, first_collision);
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
    [[nodiscard]] Result<void> gather_impl(Slot& slot) {
        for (;;) {
            auto gathered = gather_once(slot);
            if (!gathered) return gathered;
            if (slot.count != 0 || slot.out_of_order_count == 0) return {};
            const std::lock_guard lock(tree_mutex);
            ++completed_batches;
            smart_prune_locked();
            if (should_stop_locked()) return {};
        }
    }
    void submit(Slot& slot) {
        if (slot.count == 0) return;
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
    [[nodiscard]] Result<void> publish_batch(Slot& slot) {
        std::size_t required = 0;
        // Validate every output and prior before publishing any part of a batch.
        for (std::size_t row = 0; row < slot.count; ++row) {
            auto& request = slot.requests[slot.rows[row]];
            const auto count = request.legal.size();
            const auto valid = evaluator
                                   ? nn::validate_legal_output(slot.legal_outputs.at(row), count)
                                   : nn::validate_nn_output(slot.outputs[row]);
            if (!valid)
                return tl::unexpected(std::string{"Invalid search NN output: "} + valid.error());
            auto priors =
                evaluator ? encode_legal_logits(
                                std::span{slot.legal_outputs.at(row).logits}.first(count),
                                config.selection.policy_temperature, {request.priors.data(), count})
                          : encode_legal_priors({request.legal.begin(), request.legal.end()},
                                                request.side, slot.outputs[row].policy_logits,
                                                config.selection.policy_temperature,
                                                {request.priors.data(), count});
            if (!priors) return tl::unexpected(priors.error());
            for (std::size_t i = 0; i < count; ++i)
                request.seeds.at(i) = {.move = request.legal[i],
                                       .prior_bits = request.priors.at(i)};
            if (request.path.size - 1 < ceiling()) required += count;
        }
        // lc0 caches a computed batch before backing it up.
        if (legal_cache && !config.deterministic)
            for (std::size_t row = 0; row < slot.count; ++row) {
                auto inserted = legal_cache->insert(slot.inputs[row], slot.policies.at(row),
                                                    slot.legal_outputs.at(row));
                if (!inserted) return inserted;
            }
        {
            const std::lock_guard lock(tree_mutex);
            const auto statistics = tree->stats();
            if (required > statistics.edge_capacity - statistics.edge_count) {
                stopped = StopReason::MemoryLimit;
                for (std::size_t i = 0; i < config.maximum_batch; ++i)
                    slot.requests[i].reservation.release_locked();
            } else {
                for (std::size_t row = 0; row < slot.count; ++row) {
                    auto& request = slot.requests[slot.rows[row]];
                    const auto& wdl =
                        evaluator ? slot.legal_outputs.at(row).wdl : slot.outputs[row].wdl;
                    const auto moves_left = evaluator ? slot.legal_outputs.at(row).moves_left
                                                      : slot.outputs[row].moves_left;
                    const double sum = static_cast<double>(wdl[0]) + wdl[1] + wdl[2];
                    const double q = (static_cast<double>(wdl[0]) - wdl[2]) / sum;
                    const double draw = wdl[1] / sum;
                    auto published = publish_value_locked(
                        request, {.q = q, .draw = draw, .moves_left = moves_left});
                    if (!published) return tl::unexpected(published.error());
                }
            }
            // lc0 DoBackupUpdate cancels shared collisions after real backup.
            // Both slots may have gathered virtual reservations in the meantime.
            for (auto& pending : slots) cancel_collisions_locked(pending);
        }
        return {};
    }
    [[nodiscard]] const Edge* best_edge(NodeId id) const {
        const Edge* best = nullptr;
        FinalMoveStatistics best_stats;
        for (const auto& edge : tree->edges(id)) {
            if (id == 0 && !allowed(edge.move)) continue;
            const auto* child = edge.child == kNoNode ? nullptr : &tree->node(edge.child);
            const auto n = child ? child->visits : 0;
            const FinalMoveStatistics stats{
                .visits = n,
                .q = n == 0 ? 0 : -child->value_sum / static_cast<double>(n),
                .prior = decode_prior(edge.prior_bits),
                .child_proof = child ? child->proof : NodeProof{}};
            if (!best || prefer_final_move(stats, best_stats)) {
                best = &edge;
                best_stats = stats;
            }
        }
        return best;
    }
    [[nodiscard]] PrincipalVariation pv_locked(
        std::optional<chess::Move> first = std::nullopt) const {
        PrincipalVariation result;
        if (!tree || tree->node(0).state == NodeState::Terminal || (first && !allowed(*first)))
            return result;
        NodeId current = 0;
        while (result.size < result.moves.size()) {
            const Edge* edge = nullptr;
            if (result.size == 0 && first) {
                for (const auto& candidate : tree->edges(0))
                    if (candidate.move == *first) edge = &candidate;
            } else
                edge = best_edge(current);
            // A legal prior choice is a best_move, but an unvisited edge is
            // not part of a completed principal variation.
            if (!edge || edge->child == kNoNode || tree->node(edge->child).visits == 0) break;
            result.moves.at(result.size++) = edge->move;
            current = edge->child;
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
        result.tree_capacity = stats.node_capacity;
        const auto& node = tree->node(0);
        result.nodes = node.visits;
        result.proof = node.proof;
        result.inherited_visits = inherited;
        result.new_playouts = node.visits >= inherited ? node.visits - inherited : 0;
        result.q =
            node.visits == 0 ? node.leaf_q : node.value_sum / static_cast<double>(node.visits);
        result.draw =
            node.visits == 0 ? node.leaf_draw : node.draw_sum / static_cast<double>(node.visits);
        result.seldepth = seldepth;
        if (started != Clock::time_point{})
            result.elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                (running ? Clock::now() : finished) - started);
        result.pv = pv_locked();
        result.depth = result.pv.size;
        return result;
    }
    void smart_prune_locked() {
        if (!limits.smart_pruning || stopped) return;
        std::array<std::uint64_t, chess::MoveList::kCapacity> visits{};
        std::size_t count = 0, losing = 0;
        bool win_found = false, may_resign = true;
        for (const auto& edge : tree->edges(0)) {
            // lc0 search.cc PopulateCommonIterationStats enumerates every root
            // edge, including those excluded by a searchmoves restriction.
            if (edge.child != kNoNode) {
                const auto& child = tree->node(edge.child);
                visits.at(count) = child.visits;
                if (child.visits != 0) {
                    if (child.proof.outcome == ProofOutcome::Win) ++losing;
                    if (child.proof.outcome == ProofOutcome::Loss) win_found = true;
                    // Pinned source's resignability threshold is-0.98f.
                    const auto q = -child.value_sum / static_cast<double>(child.visits);
                    if (q > -0.98F) may_resign = false;
                }
            }
            ++count;
        }
        const auto now = Clock::now();
        const auto root_visits = tree->node(0).visits;
        // lc0 common.cc always installs a4,000,000,000-visit overflow stopper;
        // timemgr.cc hints use100,000,000,000ms when no clock bound exists.
        const auto node_limit = limits.nodes.value_or(4'000'000'000ULL);
        const auto remaining_nodes = node_limit > root_visits ? node_limit - root_visits : 1;
        const auto remaining_time =
            deadline ? std::chrono::duration_cast<std::chrono::milliseconds>(*deadline - now)
                     : std::chrono::milliseconds{100'000'000'000LL};
        if (pruning.should_stop(
                {.new_nodes = root_visits - inherited,
                 .completed_batches = completed_batches,
                 .remaining_nodes = remaining_nodes,
                 .elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - started),
                 .remaining_time = remaining_time,
                 .root_visits = std::span{visits}.first(count),
                 .losing_moves = losing,
                 .win_found = win_found,
                 .may_resign = may_resign}))
            stopped = StopReason::SmartPruning;
    }
    [[nodiscard]] Result<SearchResult> execute(const SearchLimits& selected) {
        {
            const std::lock_guard lock(tree_mutex);
            limits = selected;
            inherited = tree->node(0).visits;
            completed_batches = 0;
            diagnostics = {};
            pruning = SmartPruning{static_cast<float>(limits.smart_pruning_factor),
                                   limits.minimum_smart_pruning_batches};
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
            for (std::size_t i = 0; i < workers.size(); ++i)
                worker_threads.emplace_back([this, i] { worker_loop(i); });
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

SearchEngine::SearchEngine(FactoryToken, std::unique_ptr<Impl> implementation)
    : impl_(std::move(implementation)) {}
SearchEngine::~SearchEngine() = default;
Result<std::unique_ptr<SearchEngine>> SearchEngine::create(const SearchConfig& config,
                                                           const chess::AttackTables& attacks,
                                                           nn::Backend& backend,
                                                           nn::LegalCache* cache) {
    auto parameters = validate_selection_parameters(config.selection);
    if (!parameters) return tl::unexpected(parameters.error());
    if (config.threads == 0 || config.threads > 128 || config.maximum_batch == 0 ||
        config.maximum_batch > 1024 || config.ram_limit_bytes == 0 || config.max_depth == 0 ||
        config.max_depth > kMaximumDepth)
        return tl::unexpected(
            std::string{"Invalid search worker, batch, RAM or depth configuration"});
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
        if (*invalidated) summary = {.kept_nodes = 1, .kept_edges = 0, .discarded_context = true};
        impl_->limits.searchmoves_count = 0;
        impl_->inherited = impl_->tree->node(0).visits;
        impl_->seldepth = impl_->deepest_completed_locked();
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
            if (child.visits != 0) {
                item.q = -child.value_sum / static_cast<double>(child.visits);
                item.draw = child.draw_sum / static_cast<double>(child.visits);
            }
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
