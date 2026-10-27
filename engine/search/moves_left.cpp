// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/search/moves_left.h"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace mouse::search {
namespace {
constexpr double kThreshold = 0.8;
constexpr double kSlope = 0.0027;
constexpr double kCap = 0.0345;
constexpr double kLinear = 1.6521;
constexpr double kQuadratic = -0.6521;
}  // namespace
// Original implementation of the pinned MEvaluator behavior, not copied code:
// https://github.com/LeelaChessZero/lc0/blob/fd71a2d921b689c5f479d3227c3806c8e272d9c5/src/search/classic/search.cc
// Defaults were also observed in the official binary's actual UCI transcript.
double moves_left_utility(const Node& parent, const Node& child) noexcept {
    if (parent.visits == 0 || child.visits == 0 || parent.moves_left_samples != parent.visits ||
        child.moves_left_samples != child.visits)
        return 0;
    const auto parent_q = parent.value_sum / static_cast<double>(parent.visits);
    if (std::abs(parent_q) <= kThreshold) return 0;
    const auto edge_q = -child.value_sum / static_cast<double>(child.visits);
    const auto parent_m = parent.moves_left_sum / static_cast<double>(parent.moves_left_samples);
    const auto child_m = child.moves_left_sum / static_cast<double>(child.moves_left_samples);
    const auto confidence = std::max(0.0, (std::abs(edge_q) - kThreshold) / (1.0 - kThreshold));
    const auto direction = edge_q > 0 ? -1.0 : (edge_q < 0 ? 1.0 : 0.0);
    return std::clamp(kSlope * (child_m - parent_m), -kCap, kCap) * direction *
           (kLinear * confidence + kQuadratic * confidence * confidence);
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
