// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/search/pruning.h"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <limits>

using mouse::search::PruningSample;
using mouse::search::SmartPruning;
using Milliseconds = std::chrono::milliseconds;
TEST_CASE("Legacy pruning divides visit forecasts and waits200ms after first evaluation",
          "[search][legacy-v1]") {
    SmartPruning pruning;
    const std::array<std::uint64_t, 2> visits{1000, 100};
    PruningSample sample{.new_nodes = 1000,
                         .completed_batches = 20,
                         .remaining_nodes = 100000,
                         .elapsed = Milliseconds{0},
                         .remaining_time = Milliseconds{200},
                         .root_visits = visits};
    CHECK_FALSE(pruning.should_stop(sample));
    sample.elapsed = Milliseconds{199};
    CHECK_FALSE(pruning.should_stop(sample));
    sample.elapsed = Milliseconds{400};
    sample.new_nodes = 2000;
    // (2300*1000/400+1)*0.2/1.33 <900. Multiplication would incorrectly fail.
    CHECK(pruning.should_stop(sample));
    sample.remaining_time = Milliseconds{400};
    CHECK_FALSE(pruning.should_stop(sample));
    sample.remaining_nodes = 1000;
    CHECK(pruning.should_stop(sample));
}
TEST_CASE("Pruning disable and minimum batches retain the source early-stop ordering",
          "[search][legacy-v1]") {
    const std::array<std::uint64_t, 2> visits{1000, 100};
    PruningSample sample{.new_nodes = 1000,
                         .completed_batches = 0,
                         .remaining_nodes = 1,
                         .elapsed = Milliseconds{0},
                         .remaining_time = Milliseconds{0},
                         .root_visits = visits};
    SmartPruning disabled{0};
    CHECK_FALSE(disabled.should_stop(sample));
    SmartPruning limited{1.33F, 10};
    CHECK_FALSE(limited.should_stop(sample));
    sample.elapsed = Milliseconds{400};
    CHECK_FALSE(limited.should_stop(sample));
    sample.completed_batches = 10;
    CHECK(limited.should_stop(sample));
    SmartPruning forced{1.33F, 10000};
    sample.may_resign = false;
    sample.losing_moves = 1;
    CHECK(forced.should_stop(sample));
    CHECK_FALSE(disabled.should_stop(sample));
    sample.losing_moves = 0;
    sample.win_found = true;
    CHECK(forced.should_stop(sample));
    CHECK_FALSE(disabled.should_stop(sample));
    CHECK_THROWS(SmartPruning{std::numeric_limits<float>::quiet_NaN()});
    CHECK_THROWS(SmartPruning{-1});
    CHECK_THROWS(SmartPruning{1.33F, 10001});
}

TEST_CASE("Collision forecasts precede the minimum-batch stop check and saturate safely",
          "[search][collision-budget-v1]") {
    constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
    const std::array<std::uint64_t, 2> visits{1, 1};
    SmartPruning pruning{0.5F, 10000};
    PruningSample sample{.new_nodes = 1,
                         .remaining_nodes = 100,
                         .elapsed = Milliseconds{0},
                         .remaining_time = Milliseconds{1000},
                         .root_visits = visits};
    CHECK(pruning.estimated_remaining_playouts() == maximum);
    CHECK_FALSE(pruning.should_stop(sample));
    sample.elapsed = Milliseconds{199};
    CHECK_FALSE(pruning.should_stop(sample));
    CHECK(pruning.estimated_remaining_playouts() == maximum);
    sample.elapsed = Milliseconds{200};
    CHECK_FALSE(pruning.should_stop(sample));
    CHECK(pruning.estimated_remaining_playouts() == 200);
    sample.new_nodes = maximum;
    sample.remaining_nodes = maximum;
    CHECK_FALSE(pruning.should_stop(sample));
    CHECK(pruning.estimated_remaining_playouts() == maximum);
    sample.remaining_time = Milliseconds{-1};
    CHECK_FALSE(pruning.should_stop(sample));
    CHECK(pruning.estimated_remaining_playouts() == 0);
    SmartPruning disabled{0.0F};
    CHECK_FALSE(disabled.should_stop(sample));
    CHECK(disabled.estimated_remaining_playouts() == maximum);
}
