// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stop_token>

#include "engine/nn/backend.h"
#include "engine/nn/history.h"
#include "engine/search/selection.h"
#include "engine/search/tree.h"

namespace mouse::nn {
class LegalCache;
}
namespace mouse::search {
struct SearchConfig {
    SelectionParameters selection;
    std::size_t threads = 4;
    std::size_t maximum_batch = 64;
    std::size_t ram_limit_bytes = 512U << 20U;
    std::size_t max_depth = 512;
    // Preserve cache-on/off batch barriers for reproducible equality gates.
    bool deterministic = false;
    bool moves_left_utility = true;
    bool terminal_proof = true;
};
struct SearchLimits {
    std::optional<std::uint64_t> nodes = 800;
    std::optional<std::chrono::milliseconds> movetime;
    // External absolute clock budget, including caller setup. The earlier of
    // this and movetime applies. A per-run batch cap never reallocates buffers.
    std::optional<std::chrono::steady_clock::time_point> deadline;
    std::optional<std::size_t> maximum_batch;
    std::optional<std::size_t> depth;
    bool log_search_stats = false;
    bool smart_pruning = true;
    double smart_pruning_factor = 1.33;
    std::uint64_t minimum_smart_pruning_batches = 0;
    std::array<chess::Move, chess::MoveList::kCapacity> searchmoves{};
    std::size_t searchmoves_count = 0;
    std::stop_token stop_token;
    // Run owner only, after a completed batch and outside tree/queue locks.
    // Observers may snapshot the tree here without making deadline delivery wait.
    std::function<void()> on_progress;
};
enum class StopReason : std::uint8_t {
    NodeLimit,
    TimeLimit,
    DepthLimit,
    ExternalStop,
    MemoryLimit,
    Terminal,
    SmartPruning
};
struct PrincipalVariation {
    std::array<chess::Move, 513> moves{};
    std::size_t size = 0;
};
struct RootMoveStat {
    chess::Move move;
    std::uint64_t visits = 0;
    double q = 0, draw = 0;
    float prior = 0;
    NodeProof child_proof;  // Child side-to-move, unlike parent-view q above.
};
struct RootStatistics {
    std::array<RootMoveStat, chess::MoveList::kCapacity> moves{};
    std::size_t size = 0;
};
struct SearchInfo {
    std::uint64_t nodes = 0, new_playouts = 0, inherited_visits = 0;
    std::size_t depth = 0, seldepth = 0, charged_bytes = 0, budget_bytes = 0, tree_nodes = 0,
                tree_capacity = 0;
    double q = 0, draw = 0;
    std::chrono::milliseconds elapsed{};
    PrincipalVariation pv;
    NodeProof proof;  // Root side-to-move; mate distances are upper bounds.
};
struct SearchStats {
    // Completed physical provider calls; cache hits and duplicate rows excluded.
    std::array<std::uint64_t, 1025> batch_sizes{};
    std::uint64_t collisions = 0, out_of_order = 0, proof_scans = 0, selection_hits = 0;
    std::chrono::nanoseconds gather{}, nn_wait{}, backup{}, provider{}, wall{};
    SearchStats& operator+=(const SearchStats& other) noexcept {
        for (std::size_t i = 0; i < batch_sizes.size(); ++i)
            batch_sizes.at(i) += other.batch_sizes.at(i);
        collisions += other.collisions;
        out_of_order += other.out_of_order;
        proof_scans += other.proof_scans;
        selection_hits += other.selection_hits;
        gather += other.gather;
        nn_wait += other.nn_wait;
        backup += other.backup;
        provider += other.provider;
        wall += other.wall;
        return *this;
    }
};
struct SnapshotCost {
    std::uint64_t tree_lock_count = 0;
    std::chrono::nanoseconds tree_lock_wait{};
};
// Scoped per-thread diagnostic routing; nested scopes restore their caller.
// No shared mutable counter or tree access is exposed to the observer.
class SnapshotCostScope {
   public:
    explicit SnapshotCostScope(SnapshotCost&) noexcept;
    ~SnapshotCostScope();
    SnapshotCostScope(const SnapshotCostScope&) = delete;
    SnapshotCostScope& operator=(const SnapshotCostScope&) = delete;
    SnapshotCostScope(SnapshotCostScope&&) = delete;
    SnapshotCostScope& operator=(SnapshotCostScope&&) = delete;
    [[nodiscard]] static SnapshotCost* current() noexcept;

   private:
    SnapshotCost* previous_;
    static thread_local SnapshotCost* active_;
};
struct SearchResult {
    SearchInfo info;
    SearchStats stats;
    std::optional<chess::Move> best_move;
    StopReason reason = StopReason::ExternalStop;
};
// run/set_position/advance reject overlapping lifecycle operations. snapshot,
// root_moves, principal_variation and request_stop may run concurrently with run.
// Borrowed Backend/AttackTables outlive this owner. The caller configures Backend
// for deterministic compute and sufficient maximum batch when requested.
// Invalid root/history leaves the old root unchanged. A setup allocation/budget
// failure explicitly clears the old search; set_position must succeed before run.
// Unrelated roots reuse drained arena/history capacities when their actual charge
// fits. Their visits/proofs are reset; full NN history identity still governs reuse.
// Managed RAM includes fixed requests, workers, histories, queue and tree; borrowed
// backend/cache and OS thread stacks are outside this budget. No per-visit growth.
class SearchEngine {
    struct Impl;
    struct FactoryToken {
       private:
        FactoryToken() = default;
        friend class SearchEngine;
    };

   public:
    [[nodiscard]] static Result<std::unique_ptr<SearchEngine>> create(const SearchConfig&,
                                                                      const chess::AttackTables&,
                                                                      nn::Backend&,
                                                                      nn::LegalCache* = nullptr);
    explicit SearchEngine(FactoryToken, std::unique_ptr<Impl>);
    ~SearchEngine();
    SearchEngine(const SearchEngine&) = delete;
    SearchEngine& operator=(const SearchEngine&) = delete;
    SearchEngine(SearchEngine&&) = delete;
    SearchEngine& operator=(SearchEngine&&) = delete;
    // A new game discards visits while retaining drained arena capacities.
    [[nodiscard]] Result<void> set_position(const chess::Position&, const nn::EncodingHistory&,
                                            bool retain_visits = true);
    [[nodiscard]] Result<ReuseSummary> advance(chess::Move);
    [[nodiscard]] Result<SearchResult> run(const SearchLimits&);
    void request_stop() noexcept;
    [[nodiscard]] SearchInfo snapshot() const;
    // Completed current-root statistics and best edge, without waiting for inference.
    [[nodiscard]] SearchResult result_snapshot(SnapshotCost* = nullptr) const;
    [[nodiscard]] RootStatistics root_moves(SnapshotCost* = nullptr) const;
    [[nodiscard]] PrincipalVariation principal_variation(
        std::optional<chess::Move> first = std::nullopt, SnapshotCost* = nullptr) const;

   private:
    std::unique_ptr<Impl> impl_;
};
}  // namespace mouse::search
