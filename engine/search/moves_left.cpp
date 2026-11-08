// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/search/moves_left.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <string>

namespace mouse::search {
namespace {
bool within(double value, double minimum, double maximum) {
    return std::isfinite(value) && value >= minimum && value <= maximum;
}
}  // namespace
Result<void> validate_moves_left_parameters(const MovesLeftParameters& parameters) {
    if (!within(parameters.max_effect, 0, 1) || !within(parameters.threshold, 0, 1) ||
        !within(parameters.slope, 0, 1) || !within(parameters.constant_factor, -1, 1) ||
        !within(parameters.scaled_factor, -2, 2) || !within(parameters.quadratic_factor, -1, 1))
        return tl::unexpected(std::string{
            "MovesLeft options require MaxEffect, Threshold, Slope 0..1, ConstantFactor -1..1, "
            "ScaledFactor -2..2 and QuadraticFactor -1..1"});
    return {};
}
// Original implementation of the pinned MEvaluator behavior, not copied code:
// https://github.com/LeelaChessZero/lc0/blob/fd71a2d921b689c5f479d3227c3806c8e272d9c5/src/search/classic/search.cc
// Defaults were also observed in the official binary's actual UCI transcript.
double moves_left_utility(const Node& parent, const Node& child,
                          const MovesLeftParameters& parameters) noexcept {
    if (parent.visits == 0 || child.visits == 0 || parent.moves_left_samples != parent.visits ||
        child.moves_left_samples != child.visits)
        return 0;
    const auto threshold = parameters.threshold;
    const auto parent_q = parent.value_sum / static_cast<double>(parent.visits);
    if (std::abs(parent_q) <= threshold) return 0;
    const auto edge_q = -child.value_sum / static_cast<double>(child.visits);
    const auto parent_m = parent.moves_left_sum / static_cast<double>(parent.moves_left_samples);
    const auto child_m = child.moves_left_sum / static_cast<double>(child.moves_left_samples);
    // Lc0 smooths Q above a threshold strictly inside (0, 1) and uses it raw otherwise.
    const auto confidence = threshold > 0 && threshold < 1
                                ? std::max(0.0, (std::abs(edge_q) - threshold) / (1.0 - threshold))
                                : edge_q;
    const auto direction = edge_q > 0 ? -1.0 : (edge_q < 0 ? 1.0 : 0.0);
    return std::clamp(parameters.slope * (child_m - parent_m), -parameters.max_effect,
                      parameters.max_effect) *
           direction *
           (parameters.constant_factor + parameters.scaled_factor * std::abs(confidence) +
            parameters.quadratic_factor * confidence * confidence);
}
void backup_moves_left_sample(Node& node, std::optional<double>& sample) noexcept {
    if (!sample) return;
    assert(node.moves_left_samples < node.visits);
    assert(std::isfinite(*sample) && *sample >= 0);
    node.moves_left_sum += *sample;
    ++node.moves_left_samples;
    *sample += 1.0;
}
}  // namespace mouse::search
