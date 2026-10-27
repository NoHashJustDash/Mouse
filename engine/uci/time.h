// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

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
struct TimeBudget {
    std::optional<std::chrono::milliseconds> budget;
    bool hold_until_signal = false;
    // Legacy stopper's own limit, before a simultaneous movetime bound. It is
    // used to return saved time on publication, including early node stops.
    std::optional<std::chrono::milliseconds> clock_budget = std::nullopt;
};
struct SearchSchedule {
    std::optional<std::chrono::steady_clock::time_point> deadline;
    std::size_t maximum_batch = 1;
    bool exhausted = false;
};
// Pure scheduling against explicit steady-clock samples. Use the measured
// physical root cost to select a conservative bucket for the two pipeline
// slots. The deadline is exactly the TM's allocation and never restarts or
// shortens. An accepted provider call can drain after independent publication.
[[nodiscard]] SearchSchedule schedule_search(const TimeBudget&,
                                             std::chrono::steady_clock::time_point accepted,
                                             std::chrono::steady_clock::time_point now,
                                             std::chrono::steady_clock::duration single_path,
                                             std::span<const std::size_t> buckets,
                                             std::size_t maximum_batch);
// lc0 v0.32.1 classic/stoppers/legacy.cc defaults, including first-book-move
// bonus and saved-time bank. Session input/publisher ownership is serialized by
// joining the prior publisher; inference never mutates this state. Reset for a
// new game or a position that does not continue the current game.
struct TimeContext {
    chess::Color side_to_move = chess::Color::White;
    std::uint64_t overhead_ms = 0, game_ply = 1;
};
class LegacyTimeManager {
   public:
    [[nodiscard]] Result<TimeBudget> allocate(const GoCommand&, TimeContext);
    void complete(const TimeBudget&, std::chrono::milliseconds elapsed) noexcept;
    void reset() noexcept;

   private:
    bool first_clock_move_ = true;
    std::int64_t saved_ms_ = 0;
};
}  // namespace mouse::uci
