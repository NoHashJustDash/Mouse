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
// Lc0 fd71a2d classic/params.cc: how a batch is gathered, with its defaults and ranges.
struct GatherParameters {
    std::uint32_t max_collision_events = kMaximumCollisionEvents;   // 1..65536
    std::uint32_t max_collision_visits = kMaximumCollisionVisits;   // 1..100000000
    std::uint32_t collision_scaling_start = 28;                     // 1..100000
    std::uint32_t collision_scaling_end = 145000;                   // 0..100000000, 0 = no scaling
    double collision_scaling_power = 1.25;                          // 0.01..100
    double max_out_of_order_factor = 2.4;                           // 0..100
    bool out_of_order_eval = true;
    std::uint32_t max_prefetch = 32;                 // 0..1024
    std::uint32_t idling_minimum_work = 0;           // 0..10000
    std::uint32_t thread_idling_threshold = 1;       // 0..128
    std::uint32_t minimum_processing_work = 20;      // 2..100000
    std::uint32_t minimum_per_task_processing = 8;   // 1..100000
    std::uint32_t minimum_picking_work = 1;              // 1..100000
    std::uint32_t minimum_remaining_picking_work = 20;   // 0..100000
    bool operator==(const GatherParameters&) const = default;
};
// Source: classic/search.cc CalculateCollisionsLeft; endpoints checked first.
[[nodiscard]] std::uint32_t collision_visit_budget(
    std::uint64_t root_visits,
    std::uint64_t remaining_playouts = std::numeric_limits<std::uint64_t>::max(),
    const GatherParameters& parameters = {}) noexcept;
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
