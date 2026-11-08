// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <vector>

#include "engine/nn/backend.h"
#include "engine/nn/encoder.h"
#include "engine/nn/history.h"
#include "engine/search/collisions.h"
#include "engine/search/draw_rate.h"
#include "engine/search/moves_left.h"
#include "engine/search/selection.h"
#include "engine/search/tablebase.h"
#include "engine/search/tree.h"

namespace mouse::nn {
class LegalCache;
}
namespace mouse::search {
// Lc0 fd71a2d classic/params.cc Temperature, TempDecayMoves, TempDecayDelayMoves, TempCutoffMove,
// TempEndgame, TempValueCutoff and TempVisitOffset.
struct TemperatureParameters {
    double temperature = 0.0;
    std::uint32_t decay_moves = 0;
    std::uint32_t decay_delay_moves = 0;
    std::uint32_t cutoff_move = 0;
    double endgame = 0.0;
    double value_cutoff = 100.0;
    double visit_offset = 0.0;
    bool operator==(const TemperatureParameters&) const = default;
};
struct SearchConfig {
    SelectionParameters selection;
    std::size_t threads = 4;
    std::size_t maximum_batch = 64;
    std::size_t ram_limit_bytes = 512U << 20U;
    std::size_t max_depth = 512;
    // Preserve cache-on/off batch barriers for reproducible equality gates.
    bool deterministic = false;
    bool moves_left_utility = true;
    MovesLeftParameters moves_left;
    DrawRateParameters draw_rate;
    // Lc0 StickyEndgames.
    bool terminal_proof = true;
    GatherParameters gather;
    // Lc0 TwoFoldDraws, DrawScore (-1..1, from white's view), NodesPerSecondLimit (0 = off)
    // and HistoryFill.
    bool two_fold_draws = true;
    double draw_score = 0.0;
    double nps_limit = 0.0;
    nn::HistoryFill history_fill = nn::HistoryFill::FenOnly;
    // Lc0 SyzygyPath's tables, shared with the session; empty without any.
    std::shared_ptr<Tablebase> tablebase;
    // Lc0 SolidTreeThreshold: children of a node with this many visits get contiguous slots.
    std::uint32_t solid_tree_threshold = 100;
    // Lc0 DirichletNoiseEpsilon and DirichletNoiseAlpha, applied when the root expands.
    double noise_epsilon = 0.0;
    double noise_alpha = 0.3;
    TemperatureParameters temperature;
    // Lc0 dag_classic ("dag-preview"): positions share a low node found by a hash of the last
    // cache_history_length + 1 positions.
    bool dag = false;
    std::uint32_t cache_history_length = 0;
    // Noise and temperature draw from this; zero seeds from the system.
    std::uint64_t random_seed = 0;
};
[[nodiscard]] Result<void> validate_search_tuning(const SearchConfig& config);
// Whose side the contempt favours in this search (lc0 ContemptMode after "play" is resolved).
enum class ContemptSide : std::uint8_t { None, White, Black };
// Lc0 IterationStats, for a time manager that decides while the search runs.
struct TimeSample {
    std::uint64_t new_nodes = 0, total_nodes = 0;
    std::span<const std::uint64_t> root_visits;
    // Lc0 TimeUsageHint::kNeedMoreTime: the most visited root move lacks the best Q plus M.
    bool need_more_time = false;
};
// Lc0 SearchStopper::ShouldStop and StoppersHints.
struct TimeDecision {
    bool stop = false;
    std::optional<std::chrono::milliseconds> remaining;
    std::optional<double> nps;
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
    ContemptSide contempt_side = ContemptSide::None;
    // Lc0 root_is_in_dtz_: DTZ already ranked the root moves, so nothing below probes WDL.
    bool root_in_dtz = false;
    std::stop_token stop_token;
    // Under the tree lock after each completed batch; decides before smart pruning does.
    std::function<TimeDecision(const TimeSample&)> time_stopper;
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
    // DrawScore * draw: the final move choice compares q plus this.
    double draw_score_share = 0;
    // Mean moves left below this move, when every completed visit supplied one.
    std::optional<double> moves_left;
    std::uint32_t in_flight = 0;
    bool terminal = false;
};
struct RootStatistics {
    std::array<RootMoveStat, chess::MoveList::kCapacity> moves{};
    std::size_t size = 0;
};
struct SearchInfo {
    std::uint64_t nodes = 0, new_playouts = 0, inherited_visits = 0;
    // tree_capacity: the most nodes the arena can hold (fixed for its lifetime);
    // tree_used_bytes of tree_arena_bytes are taken by nodes and edges (hashfull).
    std::size_t depth = 0, seldepth = 0, charged_bytes = 0, budget_bytes = 0, tree_nodes = 0,
                tree_capacity = 0, tree_used_bytes = 0, tree_arena_bytes = 0;
    double q = 0, draw = 0;
    // Mean moves left at the root, when every completed visit supplied one.
    std::optional<double> moves_left;
    std::uint64_t tb_hits = 0;
    std::chrono::milliseconds elapsed{};
    PrincipalVariation pv;
    NodeProof proof;  // Root side-to-move; mate distances are upper bounds.
};
struct SearchStats {
    // Completed physical provider calls; cache hits and duplicate rows excluded.
    std::array<std::uint64_t, 1025> batch_sizes{};
    std::uint64_t collisions = 0, out_of_order = 0, proof_scans = 0, selection_hits = 0,
                  twofold_reverts = 0;
    std::chrono::nanoseconds gather{}, nn_wait{}, backup{}, provider{}, wall{};
    SearchStats& operator+=(const SearchStats& other) noexcept {
        for (std::size_t i = 0; i < batch_sizes.size(); ++i)
            batch_sizes.at(i) += other.batch_sizes.at(i);
        collisions += other.collisions;
        out_of_order += other.out_of_order;
        proof_scans += other.proof_scans;
        selection_hits += other.selection_hits;
        twofold_reverts += other.twofold_reverts;
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
    // Lc0 VerboseMoveStats: one line per root move, least visited first, then the root itself.
    [[nodiscard]] std::vector<std::string> verbose_move_stats(bool chess960) const;
    [[nodiscard]] PrincipalVariation principal_variation(
        std::optional<chess::Move> first = std::nullopt, SnapshotCost* = nullptr) const;
    // Lc0 Search::EnsureBestMoveKnown: the move to play given the game ply, sampled by visits when
    // a temperature is in force. Each call with a temperature draws again.
    [[nodiscard]] std::optional<chess::Move> final_move(std::uint64_t game_ply);
    // The move's visits and prior at the root and its proof, for each root move in edge order:
    // what a training record stores.
    [[nodiscard]] RootStatistics all_root_moves() const;

   private:
    std::unique_ptr<Impl> impl_;
};
}  // namespace mouse::search
