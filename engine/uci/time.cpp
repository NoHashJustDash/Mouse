// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/uci/time.h"

#include <algorithm>
#include <charconv>
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

void BucketLatencies::observe(std::size_t bucket, Duration elapsed) noexcept {
    const std::lock_guard lock(mutex_);
    for (std::size_t i = 0; i < used_; ++i) {
        if (entries_[i].bucket == bucket) {
            entries_[i].window.observe(elapsed);
            return;
        }
    }
    if (used_ == entries_.size()) return;
    entries_[used_].bucket = bucket;
    entries_[used_].window.observe(elapsed);
    ++used_;
}
std::span<const BucketCost> BucketLatencies::snapshot(
    std::array<BucketCost, kCapacity>& storage) const noexcept {
    const std::lock_guard lock(mutex_);
    std::size_t written = 0;
    for (std::size_t i = 0; i < used_; ++i) {
        const auto cost = entries_[i].window.estimate();
        if (cost > Duration::zero()) storage[written++] = {entries_[i].bucket, cost};
    }
    return std::span<const BucketCost>{storage}.first(written);
}
void BucketLatencies::clear() noexcept {
    const std::lock_guard lock(mutex_);
    for (std::size_t i = 0; i < used_; ++i) {
        entries_[i].bucket = 0;
        entries_[i].window.clear();
    }
    used_ = 0;
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
                               std::span<const std::size_t> buckets, std::size_t maximum_batch,
                               std::span<const BucketCost> measured) {
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
    // A batch does not cost its size times one position: the 96-row bucket runs in
    // about twice the time of the 1-row bucket. Where a bucket's real cost is known, use it.
    // A larger unmeasured bucket is scaled linearly from the nearest smaller measured one,
    // which overstates it, so the first search at a new size is still conservative.
    const auto spare = (end - now - unit) / 2;
    Clock::duration known{};
    std::size_t known_bucket = 0;
    for (const auto bucket : buckets) {
        if (bucket > maximum_batch) break;
        const auto found = std::ranges::find(measured, bucket, &BucketCost::bucket);
        auto cost = known;
        if (found != measured.end()) {
            cost = found->cost;
            known = cost;
            known_bucket = bucket;
        } else {
            if (known_bucket == 0) continue;
            const auto scaled = static_cast<double>(known.count()) * static_cast<double>(bucket) /
                                static_cast<double>(known_bucket);
            if (scaled > static_cast<double>(spare.count())) continue;
            cost = Clock::duration{static_cast<Clock::rep>(scaled)};
        }
        if (cost <= spare && bucket > result.maximum_batch) result.maximum_batch = bucket;
    }
    // Forecast only the batch size. The joined owner can drain an accepted
    // call after publication; forecasting must never shorten the TM's clock.
    return result;
}
namespace {
// https://github.com/LeelaChessZero/lc0/blob/v0.32.1/src/search/classic/stoppers/legacy.cc
// Default midpoint51.5, steepness7, first-move-bonus1.8, book-ply-bonus0.25,
// immediate-use1 and slowmover1. Preserve float arithmetic on ordinary clocks.
float estimated_moves(std::uint64_t game_ply, float midpoint, float steepness) {
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
Result<TimeManagerConfig> parse_time_manager(std::string_view text) {
    TimeManagerConfig config;
    const auto open = text.find('(');
    const auto name = text.substr(0, open);
    if (name == "legacy")
        config.kind = TimeManagerConfig::Kind::Legacy;
    else if (name == "alphazero")
        config.kind = TimeManagerConfig::Kind::Alphazero;
    else if (name == "simple")
        config.kind = TimeManagerConfig::Kind::Simple;
    else if (name == "smooth")
        config.kind = TimeManagerConfig::Kind::Smooth;
    else
        return tl::unexpected("Unknown or unsupported time manager: [" + std::string{name} + "]");
    if (open == std::string_view::npos) return config;
    if (text.back() != ')')
        return tl::unexpected("Time manager settings must end with ): " + std::string{text});
    auto rest_text = std::string{text.substr(open + 1, text.size() - open - 2)};
    // Lc0 reads a smooth manager's midpoint and steepness flat or from an mle-legacy(...) group.
    if (const auto group = rest_text.find("mle-legacy(");
        config.kind == TimeManagerConfig::Kind::Smooth && group != std::string::npos) {
        const auto end = rest_text.find(')', group);
        if (end == std::string::npos)
            return tl::unexpected("Time manager settings must close mle-legacy: " +
                                  std::string{text});
        rest_text.erase(end, 1);
        rest_text.erase(group, std::string_view{"mle-legacy("}.size());
    }
    auto rest = std::string_view{rest_text};
    while (!rest.empty()) {
        const auto comma = rest.find(',');
        const auto entry = rest.substr(0, comma);
        rest = comma == std::string_view::npos ? std::string_view{} : rest.substr(comma + 1);
        const auto equals = entry.find('=');
        if (equals == std::string_view::npos)
            return tl::unexpected("Time manager setting needs key=value: " + std::string{entry});
        const auto key = entry.substr(0, equals);
        const auto number = entry.substr(equals + 1);
        float value = 0;
        const auto parsed = std::from_chars(number.data(), number.data() + number.size(), value);
        if (parsed.ec != std::errc{} || parsed.ptr != number.data() + number.size() ||
            !std::isfinite(value))
            return tl::unexpected("Time manager setting needs a number: " + std::string{entry});
        float* target = nullptr;
        using Kind = TimeManagerConfig::Kind;
        if (config.kind == Kind::Legacy) {
            if (key == "slowmover") target = &config.slowmover;
            if (key == "midpoint-move") target = &config.midpoint_move;
            if (key == "steepness") target = &config.steepness;
            if (key == "immediate-use") target = &config.immediate_use;
            if (key == "first-move-bonus") target = &config.first_move_bonus;
            if (key == "book-ply-bonus") target = &config.book_ply_bonus;
        } else if (config.kind == Kind::Alphazero) {
            if (key == "alphazero-time-pct") target = &config.alphazero_time_pct;
        } else if (config.kind == Kind::Smooth) {
            auto& smooth = config.smooth;
            if (key == "init-tree-reuse") target = &smooth.init_tree_reuse;
            if (key == "max-tree-reuse") target = &smooth.max_tree_reuse;
            if (key == "tree-reuse-update-rate") target = &smooth.tree_reuse_update_rate;
            if (key == "nps-update-period") target = &smooth.nps_update_period;
            if (key == "init-timeuse") target = &smooth.init_timeuse;
            if (key == "min-timeuse") target = &smooth.min_timeuse;
            if (key == "timeuse-update-rate") target = &smooth.timeuse_update_rate;
            if (key == "max-move-budget") target = &smooth.max_move_budget;
            if (key == "init-piggybank") target = &smooth.init_piggybank;
            if (key == "per-move-piggybank") target = &smooth.per_move_piggybank;
            if (key == "max-piggybank-use") target = &smooth.max_piggybank_use;
            if (key == "max-piggybank-moves") target = &smooth.max_piggybank_moves;
            if (key == "trend-nps-update-period-ms") target = &smooth.trend_nps_update_period_ms;
            if (key == "bestmove-optimism") target = &smooth.bestmove_optimism;
            if (key == "overtaker-optimism") target = &smooth.overtaker_optimism;
            if (key == "force-piggybank-ms") target = &smooth.force_piggybank_ms;
            if (key == "midpoint") target = &smooth.midpoint;
            if (key == "steepness") target = &smooth.steepness;
        } else {
            if (key == "base-pct") target = &config.base_pct;
            if (key == "ply-pct") target = &config.ply_pct;
            if (key == "time-factor") target = &config.time_factor;
            if (key == "opening-bonus-pct") target = &config.opening_bonus_pct;
        }
        if (!target) return tl::unexpected("Unknown time manager setting: " + std::string{key});
        *target = value;
    }
    if (config.kind == TimeManagerConfig::Kind::Alphazero &&
        (config.alphazero_time_pct < 0.0F || config.alphazero_time_pct > 100.0F))
        return tl::unexpected(std::string{"alphazero-time-pct requires 0..100"});
    if (config.kind == TimeManagerConfig::Kind::Simple) {
        if (config.base_pct <= 0.0F || config.base_pct > 100.0F)
            return tl::unexpected(std::string{"base-pct requires 0..100"});
        if (config.ply_pct < 0.0F || config.ply_pct > 1.0F)
            return tl::unexpected(std::string{"ply-pct requires 0..1"});
        if (config.time_factor < 0.0F || config.time_factor > 100.0F)
            return tl::unexpected(std::string{"time-factor requires 0..100"});
        if (config.opening_bonus_pct < 0.0F || config.opening_bonus_pct > 1000.0F)
            return tl::unexpected(std::string{"opening-bonus-pct requires 0..1000"});
    }
    return config;
}
namespace {
// Lc0 smooth.cc ExponentialDecay.
float exponential_decay(float from, float to, float step, float value) {
    return to - (to - from) * std::pow(0.5F, value / step);
}
float linear_interpolate(float a, float b, float theta) {
    if (theta <= 0) return a;
    if (theta >= 1) return b;
    return a + (b - a) * theta;
}
// Lc0 smooth.cc LinearDecay.
float linear_decay(float current, float target, float since_reliable_ms, float since_update_ms,
                   float window_ms) {
    if (current <= target) return target;
    if (since_update_ms <= 0) return current;
    if (since_reliable_ms >= window_ms) return target;
    if (since_update_ms > since_reliable_ms) since_update_ms = since_reliable_ms;
    const float previous_fraction = (since_reliable_ms - since_update_ms) / window_ms;
    const float reliable = (previous_fraction * target - current) / (previous_fraction - 1);
    return linear_interpolate(reliable, target, since_reliable_ms / window_ms);
}
}  // namespace
SmoothPlan::SmoothPlan(const Inputs& inputs) : inputs_(inputs), state_(inputs.state) {}
void SmoothPlan::compute(std::uint64_t current_nodes) {
    const auto& params = inputs_.parameters;
    auto& state = state_;
    const auto time = static_cast<float>(inputs_.time_ms);
    const auto overhead = static_cast<float>(inputs_.overhead_ms);
    if (state.is_first_move) {
        state.piggybank_ms = static_cast<std::int64_t>(time * params.init_piggybank);
        state.is_first_move = false;
    }
    if (state.last_move_final_nodes != 0 && state.last_time != 0 && state.avg_ms_per_move >= 0.0F) {
        // Lc0 UpdateTreeReuseFactor.
        const float fraction = state.avg_ms_per_move <= 0.0F
                                   ? 0.0F
                                   : static_cast<float>(state.last_time) / state.avg_ms_per_move;
        const float reuse =
            static_cast<float>(current_nodes) / static_cast<float>(state.last_move_final_nodes);
        state.tree_reuse = std::min(
            exponential_decay(state.tree_reuse, reuse, params.tree_reuse_update_rate, fraction),
            params.max_tree_reuse);
    }
    state.last_time = 0;
    auto remaining_moves = estimated_moves(inputs_.game_ply, params.midpoint, params.steepness);
    if (inputs_.movestogo && *inputs_.movestogo > 0 &&
        static_cast<float>(*inputs_.movestogo) < remaining_moves)
        remaining_moves = static_cast<float>(*inputs_.movestogo);
    const auto increment = static_cast<float>(inputs_.increment_ms);
    const float max_piggybank =
        params.max_piggybank_moves *
        std::max(0.0F, time + increment * (remaining_moves - 1) - overhead) / remaining_moves;
    if (static_cast<float>(state.piggybank_ms) > max_piggybank)
        state.piggybank_ms = static_cast<std::int64_t>(max_piggybank);
    const float total_remaining =
        std::max(0.0F, time - static_cast<float>(state.piggybank_ms) +
                           increment * (remaining_moves - 1) - overhead);
    const float remaining_nodes = total_remaining * state.nps / 1000.0F;
    const float average_nodes = remaining_nodes / remaining_moves;
    state.avg_ms_per_move = total_remaining / remaining_moves;
    const float with_reuse = average_nodes / (1.0F - state.tree_reuse);
    const float estimate_nodes = std::max(0.0F, with_reuse - static_cast<float>(current_nodes));
    const float expected_brutto = estimate_nodes / state.nps * 1000.0F;
    const float to_piggybank =
        std::min(max_piggybank - static_cast<float>(state.piggybank_ms),
                 expected_brutto * params.per_move_piggybank);
    const float expected = expected_brutto - to_piggybank;
    auto allowed_piggybank = static_cast<std::int64_t>(static_cast<float>(state.piggybank_ms) *
                                                       params.max_piggybank_use);
    state.move_allocated_ms = expected / state.timeuse;
    if (state.move_allocated_ms > time * params.max_move_budget)
        state.move_allocated_ms = time * params.max_move_budget;
    if (state.move_allocated_ms > time - overhead)
        state.move_allocated_ms = std::max(0.0F, time - overhead);
    if (static_cast<float>(allowed_piggybank) > time - overhead - state.move_allocated_ms)
        allowed_piggybank = static_cast<std::int64_t>(time - overhead - state.move_allocated_ms);
    state.piggybank_ms += static_cast<std::int64_t>(to_piggybank);
    deadline_ms_ = static_cast<std::int64_t>(state.move_allocated_ms);
    allowed_piggybank_ms_ = allowed_piggybank;
}
std::chrono::milliseconds SmoothPlan::upper_bound() const noexcept {
    // Compute caps the allocation at these two, and the tree can decide anything below them. The
    // piggybank share depends on the tree only through a clamp to the time left: unclamped it is
    // the same for any tree, and clamped here it already takes the bound to all that is available.
    SmoothPlan probe(inputs_);
    probe.compute(0);
    const auto time = static_cast<float>(inputs_.time_ms);
    const auto available = std::max(0.0F, time - static_cast<float>(inputs_.overhead_ms));
    const auto allocation = std::min(time * inputs_.parameters.max_move_budget, available);
    const auto bound = std::min(allocation + static_cast<float>(std::max<std::int64_t>(
                                                 0, probe.allowed_piggybank_ms_)),
                                available);
    return std::chrono::milliseconds{std::max<std::int64_t>(1, static_cast<std::int64_t>(bound))};
}
void SmoothPlan::begin(std::uint64_t current_nodes) {
    const std::lock_guard lock(mutex_);
    if (begun_) return;
    compute(current_nodes);
    begun_ = true;
}
std::int64_t SmoothPlan::deadline_ms() const {
    const std::lock_guard lock(mutex_);
    return deadline_ms_;
}
std::int64_t SmoothPlan::allowed_piggybank_ms() const {
    const std::lock_guard lock(mutex_);
    return allowed_piggybank_ms_;
}
float SmoothPlan::update_nps(std::int64_t since_first_batch, std::uint64_t new_nodes) {
    auto& state = state_;
    if (since_first_batch <= 0) return state.nps;
    const float nps = 1000.0F * static_cast<float>(new_nodes) / static_cast<float>(since_first_batch);
    if (state.nps_is_reliable && since_first_batch > state.last_time)
        state.nps = linear_decay(state.nps, nps, static_cast<float>(since_first_batch),
                                 static_cast<float>(since_first_batch - state.last_time),
                                 inputs_.parameters.nps_update_period * 1000.0F);
    else
        state.nps = nps;
    state.last_time = since_first_batch;
    return state.nps;
}
// Lc0 smooth.cc VisitsTrendWatcher::Update.
void SmoothPlan::watch_visits(std::int64_t timestamp, std::span<const std::uint64_t> visits) {
    if (timestamp <= last_timestamp_) return;
    if (prev_visits_.empty()) {
        prev_visits_.assign(visits.begin(), visits.end());
        cur_visits_ = prev_visits_;
        prev_timestamp_ = timestamp;
        cur_timestamp_ = timestamp;
    }
    last_timestamp_ = timestamp;
    last_visits_.assign(visits.begin(), visits.end());
    if (static_cast<float>(cur_timestamp_) + inputs_.parameters.trend_nps_update_period_ms >=
        static_cast<float>(timestamp)) {
        prev_timestamp_ = cur_timestamp_;
        prev_visits_ = std::move(cur_visits_);
        cur_visits_ = last_visits_;
        cur_timestamp_ = last_timestamp_;
    }
}
// Lc0 smooth.cc VisitsTrendWatcher::IsBestmoveBeingOvertaken.
bool SmoothPlan::bestmove_being_overtaken(std::int64_t by_which_time) const {
    if (prev_timestamp_ >= last_timestamp_ || by_which_time <= last_timestamp_) return false;
    if (last_visits_.empty() || prev_visits_.size() != last_visits_.size()) return false;
    const auto& params = inputs_.parameters;
    const auto span = static_cast<float>(last_timestamp_ - prev_timestamp_);
    const auto ahead = static_cast<float>(by_which_time - last_timestamp_);
    const auto best = static_cast<std::size_t>(
        std::ranges::max_element(last_visits_) - last_visits_.begin());
    const auto rate = [&](std::size_t i) {
        return static_cast<float>(last_visits_[i] - prev_visits_[i]) / span;
    };
    const auto planned_best = static_cast<float>(last_visits_[best]) +
                              params.bestmove_optimism * rate(best) * ahead;
    for (std::size_t i = 0; i < last_visits_.size(); ++i) {
        if (i == best) continue;
        if (static_cast<float>(last_visits_[i]) + params.overtaker_optimism * rate(i) * ahead >
            planned_best)
            return true;
    }
    return false;
}
search::TimeDecision SmoothPlan::decide(const search::TimeSample& sample,
                                        std::chrono::milliseconds since_move_start) {
    const std::lock_guard lock(mutex_);
    if (sample.new_nodes > 0 && !first_batch_) first_batch_ = since_move_start;
    const auto since_first_batch = first_batch_ ? (since_move_start - *first_batch_).count() : 0;
    const auto nps = update_nps(since_first_batch, sample.new_nodes);
    const auto elapsed = since_move_start.count();
    watch_visits(elapsed, sample.root_visits);
    const auto with_piggybank = deadline_ms_ + allowed_piggybank_ms_;
    const bool forced =
        static_cast<float>(since_first_batch) <= inputs_.parameters.force_piggybank_ms;
    const bool use_piggybank =
        sample.need_more_time || forced || bestmove_being_overtaken(with_piggybank);
    const auto limit = use_piggybank ? with_piggybank : deadline_ms_;
    if (use_piggybank && elapsed >= deadline_ms_) used_piggybank_ = true;
    return {.stop = elapsed >= limit,
            .remaining = std::chrono::milliseconds{limit - elapsed},
            .nps = static_cast<double>(nps)};
}
std::optional<SmoothState> SmoothPlan::finish(std::chrono::milliseconds total_move_time,
                                              std::uint64_t total_nodes) {
    const std::lock_guard lock(mutex_);
    if (!begun_) return std::nullopt;
    const auto& params = inputs_.parameters;
    auto& state = state_;
    state.nps_is_reliable = true;
    const auto total = static_cast<float>(total_move_time.count());
    const float fraction = state.avg_ms_per_move <= 0.0F ? 0.0F : total / state.avg_ms_per_move;
    const float this_use = state.move_allocated_ms <= 0.0F ? 1.0F : total / state.move_allocated_ms;
    if (used_piggybank_)
        state.piggybank_ms -= std::max<std::int64_t>(0, total_move_time.count() - deadline_ms_);
    state.timeuse = std::max(
        exponential_decay(state.timeuse, used_piggybank_ ? 1.0F : this_use,
                          params.timeuse_update_rate, fraction),
        params.min_timeuse);
    state.last_move_final_nodes = static_cast<std::int64_t>(total_nodes);
    return state;
}
// Lc0 fd71a2d classic/stoppers/simple.cc SimpleTimeManager::GetStopper.
Result<TimeBudget> LegacyTimeManager::allocate_simple(const GoCommand& command,
                                                      TimeContext context) {
    const auto [side_to_move, overhead_ms, lc0_game_ply] = context;
    const auto remaining = side_to_move == chess::Color::White ? command.wtime : command.btime;
    const auto increment =
        (side_to_move == chess::Color::White ? command.winc : command.binc).value_or(0);
    const float available = static_cast<float>(signed_ms(*remaining)) -
                            static_cast<float>(signed_ms(overhead_ms));
    const float ratio =
        static_cast<float>(signed_ms(increment)) / static_cast<float>(signed_ms(*remaining));
    float fraction =
        (config_.base_pct + static_cast<float>(lc0_game_ply) * config_.ply_pct) * 0.01F +
        ratio * config_.time_factor;
    float budgeted = available * fraction;
    if (!bonus_applied_) {
        budgeted += budgeted * config_.opening_bonus_pct * 0.01F;
        bonus_applied_ = true;
    }
    if (previous_budgeted_ > 0.0F) {
        const float saved =
            previous_budgeted_ -
            (previous_available_ - (available - static_cast<float>(signed_ms(increment))));
        budgeted += std::max(saved, 0.0F);
    }
    budgeted = std::min(budgeted, available);
    previous_budgeted_ = budgeted;
    previous_available_ = available;
    const auto limit = float_ms(budgeted);
    return TimeBudget{.budget = std::chrono::milliseconds{std::max(std::int64_t{1}, limit)},
                      .hold_until_signal = false,
                      .clock_budget = std::chrono::milliseconds{limit}};
}
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
    if (remaining && config_.kind == TimeManagerConfig::Kind::Alphazero) {
        // Lc0 fd71a2d classic/stoppers/alphazero.cc: a fixed share of the time left.
        const auto share = (static_cast<float>(signed_ms(*remaining)) -
                            static_cast<float>(signed_ms(overhead_ms))) *
                           (config_.alphazero_time_pct / 100.0F);
        const auto limit = float_ms(share);
        result.clock_budget = std::chrono::milliseconds{limit};
        result.budget = std::chrono::milliseconds{std::max(std::int64_t{1}, limit)};
    } else if (remaining && config_.kind == TimeManagerConfig::Kind::Simple) {
        auto simple = allocate_simple(command, context);
        if (!simple) return simple;
        result.clock_budget = simple->clock_budget;
        result.budget = simple->budget;
    } else if (remaining && config_.kind == TimeManagerConfig::Kind::Smooth) {
        const auto increment =
            (side_to_move == chess::Color::White ? command.winc : command.binc).value_or(0);
        result.smooth = std::make_shared<SmoothPlan>(SmoothPlan::Inputs{
            .parameters = config_.smooth,
            .state = smooth_state_,
            .time_ms = signed_ms(*remaining),
            .increment_ms = signed_ms(increment),
            .overhead_ms = signed_ms(overhead_ms),
            .movestogo = command.movestogo,
            .game_ply = lc0_game_ply});
        result.budget = result.smooth->upper_bound();
    } else if (remaining) {
        const auto increment =
            (side_to_move == chess::Color::White ? command.winc : command.binc).value_or(0);
        auto moves = estimated_moves(lc0_game_ply, config_.midpoint_move, config_.steepness);
        if (command.movestogo && static_cast<float>(*command.movestogo) < moves)
            moves = static_cast<float>(*command.movestogo);
        auto total = std::max(0.0F, static_cast<float>(signed_ms(*remaining)) +
                                        static_cast<float>(signed_ms(increment)) * (moves - 1.0F) -
                                        static_cast<float>(signed_ms(overhead_ms)));
        std::int64_t squander = 0;
        if (saved_ms_ > 0) {
            total = std::max(0.0F, total - static_cast<float>(saved_ms_));
            // lc0 immediate-use: the share of the bank spent now; the rest stays banked.
            squander = config_.immediate_use >= 1.0F
                           ? saved_ms_
                           : static_cast<std::int64_t>(static_cast<double>(saved_ms_) *
                                                       config_.immediate_use);
            saved_ms_ -= squander;
        }
        auto allocation = total / moves;
        if (first_clock_move_) {
            allocation *=
                1.0F + config_.first_move_bonus +
                config_.book_ply_bonus * static_cast<float>(std::min(std::uint64_t{12}, lc0_game_ply));
            first_clock_move_ = false;
        }
        // lc0 extends with slowmover only when smart pruning could still reduce the move.
        constexpr float kSmartPruningToleranceMs = 200.0F;
        if (config_.slowmover < 1.0F || allocation * config_.slowmover > kSmartPruningToleranceMs) {
            saved_ms_ -= static_cast<std::int64_t>(allocation * (config_.slowmover - 1.0F));
            allocation *= config_.slowmover;
        }
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
void LegacyTimeManager::complete(const TimeBudget& budget, std::chrono::milliseconds elapsed,
                                 std::uint64_t total_nodes) noexcept {
    if (budget.smooth) {
        if (const auto state = budget.smooth->finish(elapsed, total_nodes)) smooth_state_ = *state;
        return;
    }
    // Only Lc0's legacy stopper banks unused time this way; simple works it out from the clock.
    if (!budget.clock_budget || config_.kind != TimeManagerConfig::Kind::Legacy) return;
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
    previous_budgeted_ = 0.0F;
    previous_available_ = 0.0F;
    bonus_applied_ = false;
    smooth_state_ = {};
    smooth_state_.tree_reuse = config_.smooth.init_tree_reuse;
    smooth_state_.timeuse = config_.smooth.init_timeuse;
}
}  // namespace mouse::uci
