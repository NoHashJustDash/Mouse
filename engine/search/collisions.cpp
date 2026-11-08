// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/search/collisions.h"

#include <algorithm>
#include <cmath>
namespace mouse::search {
std::uint32_t collision_visit_budget(std::uint64_t root_visits, std::uint64_t remaining_playouts,
                                     const GatherParameters& parameters) noexcept {
    // https://github.com/LeelaChessZero/lc0/blob/fd71a2d921b689c5f479d3227c3806c8e272d9c5/src/search/classic/search.cc
    const auto visits = std::min(root_visits, remaining_playouts);
    if (visits >= parameters.collision_scaling_end) return parameters.max_collision_visits;
    if (visits <= parameters.collision_scaling_start) return 1;
    const auto start = static_cast<float>(parameters.collision_scaling_start);
    const auto end = static_cast<float>(parameters.collision_scaling_end);
    const auto ratio = (static_cast<float>(visits) - start) / (end - start);
    const auto span = static_cast<float>(parameters.max_collision_visits) - 1.0F;
    return static_cast<std::uint32_t>(std::round(
        1.0F + span * std::pow(ratio, static_cast<float>(parameters.collision_scaling_power))));
}
std::uint32_t collision_multiplicity(const CollisionSelection& selection,
                                     std::uint32_t available) noexcept {
    if (selection.second_best <= selection.utility) return available;
    const auto count =
        std::max(1.0, selection.exploration / (selection.second_best - selection.utility) -
                          static_cast<double>(selection.started));
    // Source truncates after clamping. available <= 80000 also supplies the
    // finite integer-conversion bound when the quotient overflows to infinity.
    return static_cast<std::uint32_t>(std::min(static_cast<double>(available), count));
}
}  // namespace mouse::search
