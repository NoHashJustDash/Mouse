// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/search/pruning.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mouse::search {
SmartPruning::SmartPruning(float factor, std::uint64_t minimum_batches)
    : factor_(factor), minimum_batches_(minimum_batches) {
    // Bounds: lc0 v0.32.1 classic/stoppers/common.cc PopulateCommonStopperOptions.
    if (!std::isfinite(factor) || factor < 0.0F || factor > 10.0F || minimum_batches > 10000)
        throw std::invalid_argument("Invalid smart-pruning factor or minimum batch count");
}
bool SmartPruning::should_stop(const PruningSample& sample) noexcept {
    // https://github.com/LeelaChessZero/lc0/blob/v0.32.1/src/search/classic/stoppers/stoppers.cc
    if (factor_ == 0.0F) return false;
    if (sample.root_visits.size() == 1 ||
        sample.root_visits.size() <= sample.losing_moves + (sample.may_resign ? 0U : 1U) ||
        sample.win_found)
        return true;
    if (sample.new_nodes > 0 && !first_eval_) {
        first_eval_ = sample.elapsed;
        return false;
    }
    if (!first_eval_) return false;
    constexpr auto tolerance = std::chrono::milliseconds{200};
    if (sample.elapsed - *first_eval_ < tolerance) return false;
    // Source constants:200ms startup tolerance,300-node conservative NPS bias.
    // Widen before addition/multiplication for Mouse's uint64 visit extension.
    const auto measured = sample.elapsed - *first_eval_;
    const auto nps = std::floor(1000.0L * (static_cast<long double>(sample.new_nodes) + 300.0L) /
                                static_cast<long double>(measured.count())) +
                     1.0L;
    const auto time_remaining = static_cast<long double>(sample.remaining_time.count()) / 1000.0L;
    const auto possible = std::min(time_remaining * nps / factor_,
                                   static_cast<long double>(sample.remaining_nodes) / factor_);
    // Same hint published by lc0 SmartPruningStopper before its minimum-batch
    // check. Collision gathering consumes the latest forecast, not a new clock.
    constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
    remaining_playouts_ = possible <= 0.0L ? 0
                                           : (possible >= static_cast<long double>(maximum)
                                                  ? maximum
                                                  : static_cast<std::uint64_t>(possible));
    if (sample.completed_batches < minimum_batches_) return false;
    std::uint64_t best = 0, second = 0;
    for (const auto visits : sample.root_visits) {
        if (visits > best) {
            second = best;
            best = visits;
        } else
            second = std::max(second, visits);
    }
    return possible < static_cast<long double>(best - second);
}
}  // namespace mouse::search
