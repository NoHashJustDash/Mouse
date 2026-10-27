// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/search/collisions.h"

#include <array>
#include <catch2/catch_test_macros.hpp>
TEST_CASE("Collision visit scaling retains lc0 boundaries and float power curve",
          "[search][collision-budget-v1]") {
    using mouse::search::collision_visit_budget;
    CHECK(collision_visit_budget(0) == 1);
    CHECK(collision_visit_budget(28) == 1);
    CHECK(collision_visit_budget(29) == 1);
    // Exact quarter fractions of [28,145000], rounded as in the pinned source.
    CHECK(collision_visit_budget(36271) == 14143);
    CHECK(collision_visit_budget(72514) == 33636);
    CHECK(collision_visit_budget(108757) == 55837);
    CHECK(collision_visit_budget(145000) == 80000);
    CHECK(collision_visit_budget(145001) == 80000);
    CHECK(collision_visit_budget(1000000, 28) == 1);
    CHECK(collision_visit_budget(28, 1000000) == 1);
    CHECK(collision_visit_budget(1000000, 72514) == 33636);
}
TEST_CASE("Collision multiplicity preserves second-best switching and integer bounds",
          "[search][collision-budget-v1]") {
    using mouse::search::collision_multiplicity;
    CHECK(collision_multiplicity({.utility = 0, .exploration = 8, .started = 1, .second_best = 2},
                                 80000) == 3);
    CHECK(collision_multiplicity({.utility = 0, .exploration = 8, .started = 1, .second_best = 2},
                                 2) == 2);
    CHECK(collision_multiplicity({.utility = 0, .exploration = 9, .started = 1, .second_best = 2},
                                 80000) == 3);
    CHECK(collision_multiplicity({.utility = 0, .exploration = 0, .started = 100, .second_best = 1},
                                 80000) == 1);
    CHECK(collision_multiplicity(
              {.utility = 1, .exploration = 100, .started = 10, .second_best = 1}, 80000) == 80000);
    CHECK(collision_multiplicity(
              {.utility = 2, .exploration = 100, .started = 10, .second_best = 1}, 80000) == 80000);
    CHECK(collision_multiplicity(
              {.utility = 0, .exploration = 1, .started = 0, .second_best = 1e-300}, 80000) ==
          80000);
}
