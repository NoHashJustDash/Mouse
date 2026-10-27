// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
namespace mouse::search {
// Pinned lc0 v0.32.1 classic/params.cc defaults. The event bound also caps
// Mouse's preallocated cancellation records, independently of visit weights.
inline constexpr std::size_t kMaximumCollisionEvents = 917;
inline constexpr std::uint32_t kMaximumCollisionVisits = 80000;
// Source: classic/search.cc CalculateCollisionsLeft; endpoints checked first.
[[nodiscard]] std::uint32_t collision_visit_budget(
    std::uint64_t root_visits,
    std::uint64_t remaining_playouts = std::numeric_limits<std::uint64_t>::max()) noexcept;
struct CollisionSelection {
    double utility = 0, exploration = 0;
    std::uint64_t started = 0;
    double second_best = 0;
};
// classic/search.cc PickNodesToExtendTask: visits before the second-best edge
// overtakes this edge. Fields are finite validated selection statistics.
[[nodiscard]] std::uint32_t collision_multiplicity(const CollisionSelection& selection,
                                                   std::uint32_t available) noexcept;
}  // namespace mouse::search
