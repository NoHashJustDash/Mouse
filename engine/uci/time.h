// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "engine/search/search.h"
#include "engine/uci/commands.h"

namespace mouse::uci {
// Single-owner maximum of eight recent positive latency samples. No allocation.
// Root inference and final publication use distinct windows/owners.
class LatencyWindow {
   public:
    using Duration = std::chrono::steady_clock::duration;
    void observe(Duration elapsed) noexcept;
    [[nodiscard]] Duration estimate() const noexcept;
    void clear() noexcept;

   private:
    std::array<Duration, 8> samples_{};
    std::size_t next_ = 0;
};
// Wake early enough for a periodic report followed by final output. The legacy
// allocation itself is unchanged. One UCI millisecond also covers quantization.
[[nodiscard]] std::chrono::steady_clock::time_point publication_start(
    std::chrono::steady_clock::time_point accepted,
    std::chrono::steady_clock::time_point allocation_deadline,
    std::optional<std::chrono::steady_clock::time_point> hard_deadline,
    std::chrono::steady_clock::duration observed_delivery) noexcept;
class SmoothPlan;
struct TimeBudget {
    std::optional<std::chrono::milliseconds> budget;
    bool hold_until_signal = false;
    // Legacy stopper's own limit, before a simultaneous movetime bound. It is
    // used to return saved time on publication, including early node stops.
    std::optional<std::chrono::milliseconds> clock_budget = std::nullopt;
    // Lc0's smooth manager decides during the search, and budget is then only its upper bound.
    std::shared_ptr<SmoothPlan> smooth = nullptr;
};
struct SearchSchedule {
    std::optional<std::chrono::steady_clock::time_point> deadline;
    std::size_t maximum_batch = 1;
    bool exhausted = false;
};
// Measured cost of one physical provider call at one batch bucket.
struct BucketCost {
    std::size_t bucket = 0;
    std::chrono::steady_clock::duration cost{};
};
// Recent provider-call cost per physical bucket. The inference thread observes;
// the session reads between searches. Fixed storage, no allocation while searching.
class BucketLatencies {
   public:
    using Duration = std::chrono::steady_clock::duration;
    static constexpr std::size_t kCapacity = 16;
    void observe(std::size_t bucket, Duration elapsed) noexcept;
    // Writes one entry per measured bucket; returns the entries written.
    [[nodiscard]] std::span<const BucketCost> snapshot(
        std::array<BucketCost, kCapacity>& storage) const noexcept;
    void clear() noexcept;

   private:
    struct Entry {
        std::size_t bucket = 0;
        LatencyWindow window;
    };
    mutable std::mutex mutex_;
    std::array<Entry, kCapacity> entries_{};
    std::size_t used_ = 0;
};
// Pure scheduling against explicit steady-clock samples. The physical root cost
// decides whether any search still fits and bounds a bucket that has no evidence.
// A bucket with a measured cost, or one scaled up from a smaller measured bucket,
// is allowed when its two pipeline slots and a root path fit the time left. The
// deadline is exactly the TM's allocation and never restarts or shortens. An
// accepted provider call can drain after independent publication.
[[nodiscard]] SearchSchedule schedule_search(const TimeBudget&,
                                             std::chrono::steady_clock::time_point accepted,
                                             std::chrono::steady_clock::time_point now,
                                             std::chrono::steady_clock::duration single_path,
                                             std::span<const std::size_t> buckets,
                                             std::size_t maximum_batch,
                                             std::span<const BucketCost> measured = {});
// lc0 v0.32.1 classic/stoppers/legacy.cc defaults, including first-book-move
// bonus and saved-time bank. Session input/publisher ownership is serialized by
// joining the prior publisher; inference never mutates this state. Reset for a
// new game or a position that does not continue the current game.
struct TimeContext {
    chess::Color side_to_move = chess::Color::White;
    std::uint64_t overhead_ms = 0, game_ply = 1;
};
// Lc0 fd71a2d classic/stoppers/smooth.cc Params defaults.
struct SmoothParameters {
    float init_tree_reuse = 0.52F, max_tree_reuse = 0.73F, tree_reuse_update_rate = 3.39F,
          nps_update_period = 20.0F, init_timeuse = 0.7F, min_timeuse = 0.34F,
          timeuse_update_rate = 5.51F, max_move_budget = 0.42F, init_piggybank = 0.09F,
          per_move_piggybank = 0.12F, max_piggybank_use = 0.94F, max_piggybank_moves = 36.5F,
          trend_nps_update_period_ms = 3000.0F, bestmove_optimism = 0.2F, overtaker_optimism = 4.0F,
          force_piggybank_ms = 1000.0F, midpoint = 45.2F, steepness = 5.93F;
    bool operator==(const SmoothParameters&) const = default;
};
// Lc0 SmoothTimeManager's state between moves.
struct SmoothState {
    float tree_reuse = 0.0F, nps = 20000.0F, timeuse = 0.0F, avg_ms_per_move = 0.0F,
          move_allocated_ms = 0.0F;
    bool nps_is_reliable = false, is_first_move = true;
    std::int64_t last_move_final_nodes = 0, last_time = 0, piggybank_ms = 0;
};
// Lc0 SmoothTimeManager::GetStopper and SmoothStopper for one move. It works on its own copy of
// the state, which the manager takes back when the move completes, so a go that never ran changes
// nothing. The tree size it needs is known only once the worker has reused the tree.
class SmoothPlan {
   public:
    struct Inputs {
        SmoothParameters parameters;
        SmoothState state;
        std::int64_t time_ms = 0, increment_ms = 0, overhead_ms = 0;
        std::optional<std::uint64_t> movestogo;
        std::uint64_t game_ply = 0;
    };
    explicit SmoothPlan(const Inputs& inputs);
    // The most this plan can give the move, whatever the tree holds.
    [[nodiscard]] std::chrono::milliseconds upper_bound() const noexcept;
    void begin(std::uint64_t current_nodes);
    // Lc0 SmoothStopper::ShouldStop.
    [[nodiscard]] search::TimeDecision decide(const search::TimeSample& sample,
                                              std::chrono::milliseconds since_move_start);
    // Lc0 UpdateEndOfMoveStats, with what the manager should carry on with; nothing if never begun.
    [[nodiscard]] std::optional<SmoothState> finish(std::chrono::milliseconds total_move_time,
                                                    std::uint64_t total_nodes);
    // For tests: the allocation and the piggybank share this move may use.
    [[nodiscard]] std::int64_t deadline_ms() const;
    [[nodiscard]] std::int64_t allowed_piggybank_ms() const;

   private:
    void compute(std::uint64_t current_nodes);
    [[nodiscard]] float update_nps(std::int64_t since_first_batch, std::uint64_t new_nodes);
    void watch_visits(std::int64_t timestamp, std::span<const std::uint64_t> visits);
    [[nodiscard]] bool bestmove_being_overtaken(std::int64_t by_which_time) const;

    mutable std::mutex mutex_;
    Inputs inputs_;
    SmoothState state_;
    bool begun_ = false, used_piggybank_ = false;
    std::int64_t deadline_ms_ = 0, allowed_piggybank_ms_ = 0;
    std::optional<std::chrono::milliseconds> first_batch_;
    std::int64_t prev_timestamp_ = 0, cur_timestamp_ = 0, last_timestamp_ = 0;
    std::vector<std::uint64_t> prev_visits_, cur_visits_, last_visits_;
};
// Lc0 TimeManager: "name" or "name(key=value,...)", with the defaults of
// classic/stoppers/{legacy,alphazero,simple,smooth}.cc.
struct TimeManagerConfig {
    enum class Kind : std::uint8_t { Legacy, Alphazero, Simple, Smooth };
    Kind kind = Kind::Legacy;
    float slowmover = 1.0F, midpoint_move = 51.5F, steepness = 7.0F, immediate_use = 1.0F,
          first_move_bonus = 1.8F, book_ply_bonus = 0.25F;
    float alphazero_time_pct = 12.0F;
    float base_pct = 1.4F, ply_pct = 0.049F, time_factor = 1.5F, opening_bonus_pct = 0.0F;
    SmoothParameters smooth;
    bool operator==(const TimeManagerConfig&) const = default;
};
[[nodiscard]] Result<TimeManagerConfig> parse_time_manager(std::string_view text);
class LegacyTimeManager {
   public:
    LegacyTimeManager() = default;
    explicit LegacyTimeManager(const TimeManagerConfig& config) : config_(config) {
        smooth_state_.tree_reuse = config.smooth.init_tree_reuse;
        smooth_state_.timeuse = config.smooth.init_timeuse;
    }
    [[nodiscard]] Result<TimeBudget> allocate(const GoCommand&, TimeContext);
    // `total_nodes` is the root's visits at the end of the move; only smooth uses it.
    void complete(const TimeBudget&, std::chrono::milliseconds elapsed,
                  std::uint64_t total_nodes = 0) noexcept;
    void reset() noexcept;

   private:
    [[nodiscard]] Result<TimeBudget> allocate_simple(const GoCommand&, TimeContext);
    TimeManagerConfig config_;
    bool first_clock_move_ = true;
    std::int64_t saved_ms_ = 0;
    // Lc0 SimpleTimeManager state.
    float previous_budgeted_ = 0.0F, previous_available_ = 0.0F;
    bool bonus_applied_ = false;
    SmoothState smooth_state_;
};
}  // namespace mouse::uci
