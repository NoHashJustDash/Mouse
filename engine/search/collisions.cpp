// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/search/collisions.h"

#include <algorithm>
#include <cmath>
namespace mouse::search {
std::uint32_t collision_visit_budget(std::uint64_t root_visits,
                                     std::uint64_t remaining_playouts) noexcept {
    // https://github.com/LeelaChessZero/lc0/blob/fd71a2d921b689c5f479d3227c3806c8e272d9c5/src/search/classic/search.cc
    const auto visits = std::min(root_visits, remaining_playouts);
    if (visits >= 145000) return kMaximumCollisionVisits;
    if (visits <= 28) return 1;
    const auto ratio = (static_cast<float>(visits) - 28.0F) / (145000.0F - 28.0F);
    return static_cast<std::uint32_t>(std::round(1.0F + 79999.0F * std::pow(ratio, 1.25F)));
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
