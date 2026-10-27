// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/uci/time.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace mouse::uci {
namespace {
std::chrono::milliseconds bounded_ms(std::uint64_t duration) {
    using Milliseconds = std::chrono::milliseconds;
    const auto maximum = static_cast<std::uint64_t>(Milliseconds::max().count());
    const auto bounded = std::max(std::uint64_t{1}, std::min(duration, maximum));
    return Milliseconds{static_cast<Milliseconds::rep>(bounded)};
}
}  // namespace

void LatencyWindow::observe(Duration elapsed) noexcept {
    // Nonpositive injected/clock-resolution samples cannot prove a lower cost.
    if (elapsed <= Duration::zero()) return;
    samples_.at(next_) = elapsed;
    next_ = (next_ + 1) % samples_.size();
}
LatencyWindow::Duration LatencyWindow::estimate() const noexcept {
    return *std::ranges::max_element(samples_);
}
void LatencyWindow::clear() noexcept {
    samples_.fill(Duration::zero());
    next_ = 0;
}

std::chrono::steady_clock::time_point publication_start(
    std::chrono::steady_clock::time_point accepted,
    std::chrono::steady_clock::time_point allocation_deadline,
    std::optional<std::chrono::steady_clock::time_point> hard_deadline,
    std::chrono::steady_clock::duration observed_delivery) noexcept {
    using Clock = std::chrono::steady_clock;
    using Milliseconds = std::chrono::milliseconds;
    // A legacy allocation is a search budget. Only its collision with the actual
    // GUI clock requires earlier publication; ordinary allocations stay intact.
    if (!hard_deadline) return allocation_deadline;
    const auto deadline = *hard_deadline;
    if (deadline <= accepted) return accepted;
    const auto cost = std::chrono::ceil<Milliseconds>(
        std::max(observed_delivery, Clock::duration{Milliseconds{1}}));
    const auto maximum = std::chrono::duration_cast<Milliseconds>(Clock::duration::max());
    if (cost >= (maximum - Milliseconds{1}) / 2) return accepted;
    // A periodic info write may start immediately before final publication.
    // Budget both output paths, each at least one UCI clock tick, plus a tick
    // for integer clock quantization. No time-manager constant is changed.
    const auto reserve = std::chrono::duration_cast<Clock::duration>(cost * 2 + Milliseconds{1});
    if (deadline.time_since_epoch() < Clock::duration::min() + reserve) return accepted;
    return std::min(allocation_deadline, std::max(accepted, deadline - reserve));
}

SearchSchedule schedule_search(const TimeBudget& time,
                               std::chrono::steady_clock::time_point accepted,
                               std::chrono::steady_clock::time_point now,
                               std::chrono::steady_clock::duration single_path,
                               std::span<const std::size_t> buckets, std::size_t maximum_batch) {
    using Clock = std::chrono::steady_clock;
    SearchSchedule result{.deadline = std::nullopt, .maximum_batch = maximum_batch};
    if (!time.budget) return result;
    const auto available =
        std::chrono::duration_cast<std::chrono::milliseconds>(Clock::time_point::max() - accepted);
    const auto end = *time.budget >= available ? Clock::time_point::max() : accepted + *time.budget;
    result.deadline = end;
    result.maximum_batch = 1;
    if (now >= end || maximum_batch == 0) {
        result.exhausted = true;
        return result;
    }
    const auto unit = std::max(single_path, Clock::duration{std::chrono::milliseconds{1}});
    const auto capacity = ((end - now) / unit - 1) / 2;
    if (capacity < 1) {
        result.exhausted = true;
        return result;
    }
    const auto desired = static_cast<std::uint64_t>(capacity) >= maximum_batch
                             ? maximum_batch
                             : static_cast<std::size_t>(capacity);
    for (const auto bucket : buckets)
        if (bucket <= desired && bucket > result.maximum_batch) result.maximum_batch = bucket;
    // Forecast only the batch size. The joined owner can drain an accepted
    // call after publication; forecasting must never shorten the TM's clock.
    return result;
}
namespace {
// https://github.com/LeelaChessZero/lc0/blob/v0.32.1/src/search/classic/stoppers/legacy.cc
// Default midpoint51.5, steepness7, first-move-bonus1.8, book-ply-bonus0.25,
// immediate-use1 and slowmover1. Preserve float arithmetic on ordinary clocks.
float estimated_moves(std::uint64_t game_ply) {
    constexpr float midpoint = 51.5F;
    constexpr float steepness = 7.0F;
    const float move = static_cast<float>(game_ply) / 2.0F;
    const float estimate =
        midpoint * std::pow(1.0F + 2.0F * std::pow(move / midpoint, steepness), 1.0F / steepness) -
        move;
    if (std::isfinite(estimate) && estimate > 0.0F) return estimate;
    // Algebraically identical residual-life formula with a reciprocal power:
    // avoid overflow for Mouse's uint64 FEN extension beyond lc0's int domain.
    return move * (std::pow(std::pow(midpoint / move, steepness) + 2.0F, 1.0F / steepness) - 1.0F);
}
std::int64_t signed_ms(std::uint64_t value) {
    return static_cast<std::int64_t>(
        std::min(value, static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())));
}
std::int64_t saturated_add(std::int64_t left, std::int64_t right) noexcept {
    constexpr auto high = std::numeric_limits<std::int64_t>::max();
    constexpr auto low = std::numeric_limits<std::int64_t>::min();
    if (right > 0 && left > high - right) return high;
    if (right < 0 && left < low - right) return low;
    return left + right;
}
std::int64_t float_ms(float value) {
    const auto high = std::numeric_limits<std::int64_t>::max();
    if (static_cast<long double>(value) >= static_cast<long double>(high)) return high;
    return static_cast<std::int64_t>(value);
}
}  // namespace
Result<TimeBudget> LegacyTimeManager::allocate(const GoCommand& command, TimeContext context) {
    const auto [side_to_move, overhead_ms, lc0_game_ply] = context;
    if (side_to_move != chess::Color::White && side_to_move != chess::Color::Black)
        return tl::unexpected(std::string{"Time allocation requires a valid side to move"});
    if ((command.movetime && *command.movetime == 0) ||
        (command.movestogo && *command.movestogo == 0))
        return tl::unexpected(std::string{"movetime and movestogo must be positive when supplied"});
    if (command.infinite || command.ponder)
        return TimeBudget{
            .budget = std::nullopt, .hold_until_signal = true, .clock_budget = std::nullopt};
    const auto remaining = side_to_move == chess::Color::White ? command.wtime : command.btime;
    const bool clock_mode =
        command.wtime || command.btime || command.winc || command.binc || command.movestogo;
    if (clock_mode && !remaining && !command.movetime)
        return tl::unexpected(
            std::string{"Clock allocation requires the side-to-move remaining time"});
    TimeBudget result{.budget = std::nullopt,
                      .hold_until_signal = !command.movetime && !remaining && !command.nodes &&
                                           !command.depth && !command.mate,
                      .clock_budget = std::nullopt};
    if (remaining) {
        const auto increment =
            (side_to_move == chess::Color::White ? command.winc : command.binc).value_or(0);
        auto moves = estimated_moves(lc0_game_ply);
        if (command.movestogo && static_cast<float>(*command.movestogo) < moves)
            moves = static_cast<float>(*command.movestogo);
        auto total = std::max(0.0F, static_cast<float>(signed_ms(*remaining)) +
                                        static_cast<float>(signed_ms(increment)) * (moves - 1.0F) -
                                        static_cast<float>(signed_ms(overhead_ms)));
        std::int64_t squander = 0;
        if (saved_ms_ > 0) {
            total = std::max(0.0F, total - static_cast<float>(saved_ms_));
            squander = saved_ms_;  // lc0 default immediate-use=1.
            saved_ms_ = 0;
        }
        auto allocation = total / moves;
        if (first_clock_move_) {
            allocation *=
                1.0F + 1.8F + 0.25F * static_cast<float>(std::min(std::uint64_t{12}, lc0_game_ply));
            first_clock_move_ = false;
        }
        // Default slowmover1 leaves both allocation and saved time unchanged.
        allocation += static_cast<float>(squander);
        const auto cap = signed_ms(*remaining) - signed_ms(overhead_ms);
        const auto limit = std::min(float_ms(allocation), cap);
        result.clock_budget = std::chrono::milliseconds{limit};
        result.budget = std::chrono::milliseconds{std::max(std::int64_t{1}, limit)};
    }
    // lc0 classic/stoppers/common.cc MakeCommonStopperChain: movetime is an
    // independent bound, subtracts overhead, and coexists with the clock stopper.
    if (command.movetime) {
        const auto usable = *command.movetime > overhead_ms ? *command.movetime - overhead_ms : 0;
        const auto fixed = bounded_ms(usable);
        result.budget = result.budget ? std::min(*result.budget, fixed) : fixed;
    }
    return result;
}
void LegacyTimeManager::complete(const TimeBudget& budget,
                                 std::chrono::milliseconds elapsed) noexcept {
    if (!budget.clock_budget) return;
    const auto spent = std::max(std::int64_t{0}, elapsed.count());
    const auto limit = budget.clock_budget->count();
    const auto delta = limit < std::numeric_limits<std::int64_t>::min() + spent
                           ? std::numeric_limits<std::int64_t>::min()
                           : limit - spent;
    saved_ms_ = saturated_add(saved_ms_, delta);
}
void LegacyTimeManager::reset() noexcept {
    first_clock_move_ = true;
    saved_ms_ = 0;
}
}  // namespace mouse::uci
