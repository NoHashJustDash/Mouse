// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/search/selection.h"

#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdint>
#include <limits>

TEST_CASE("PUCT schedule has no extra visit and exploration uses N minus one",
          "[search][selection]") {
    using namespace mouse::search;
    const SelectionParameters p;
    REQUIRE(validate_selection_parameters(p));
    CHECK(cpuct_value(p, 0) == p.cpuct);
    CHECK(
        cpuct_value(p, 123456) ==
        Catch::Approx(p.cpuct + p.cpuct_factor * std::log((123456.0 + p.cpuct_base) / p.cpuct_base))
            .epsilon(1e-14));
    CHECK(exploration_multiplier(p, 0) == cpuct_value(p, 0));
    CHECK(exploration_multiplier(p, 1) == cpuct_value(p, 1));
    CHECK(exploration_multiplier(p, 2) == cpuct_value(p, 2));
    CHECK(exploration_multiplier(p, 10) == Catch::Approx(cpuct_value(p, 10) * 3.0));
    CHECK(first_play_urgency(p, true, -0.5, 0.75) == Catch::Approx(-0.5 - 0.33 * std::sqrt(0.75)));
    CHECK(first_play_urgency(p, false, 0.5, 0.25) == Catch::Approx(0.335));
}

TEST_CASE("Selection configuration rejects invalid ranges and derived overflow without tuning caps",
          "[search][selection]") {
    using mouse::search::SelectionParameters;
    using mouse::search::validate_selection_parameters;
    const std::array<double SelectionParameters::*, 6> fields{
        &SelectionParameters::cpuct,        &SelectionParameters::cpuct_base,
        &SelectionParameters::cpuct_factor, &SelectionParameters::fpu_reduction,
        &SelectionParameters::root_fpu,     &SelectionParameters::policy_temperature};
    for (const auto field : fields) {
        for (const double invalid :
             {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
              -std::numeric_limits<double>::infinity()}) {
            SelectionParameters p;
            p.*field = invalid;
            CHECK_FALSE(validate_selection_parameters(p));
        }
    }
    for (const auto field : {&SelectionParameters::cpuct, &SelectionParameters::cpuct_factor,
                             &SelectionParameters::fpu_reduction}) {
        SelectionParameters p;
        p.*field = -1;
        CHECK_FALSE(validate_selection_parameters(p));
        p.*field = 0;
        CHECK(validate_selection_parameters(p));
        p.*field = std::numeric_limits<double>::max();
        CHECK_FALSE(validate_selection_parameters(p));
    }
    for (const auto field :
         {&SelectionParameters::cpuct_base, &SelectionParameters::policy_temperature}) {
        for (const double invalid : {0.0, -1.0}) {
            SelectionParameters p;
            p.*field = invalid;
            CHECK_FALSE(validate_selection_parameters(p));
        }
    }
    for (const double invalid : {std::nextafter(100.0, 101.0), std::nextafter(-100.0, -101.0)}) {
        SelectionParameters p;
        p.root_fpu = invalid;
        CHECK_FALSE(validate_selection_parameters(p));
    }
    SelectionParameters tiny;
    tiny.cpuct_base = std::numeric_limits<double>::denorm_min();
    tiny.policy_temperature = std::numeric_limits<double>::denorm_min();
    REQUIRE(validate_selection_parameters(tiny));
    CHECK(
        std::isfinite(mouse::search::cpuct_value(tiny, std::numeric_limits<std::uint64_t>::max())));
    const double wanted = tiny.cpuct + tiny.cpuct_factor * -std::log(tiny.cpuct_base);
    CHECK(mouse::search::cpuct_value(tiny, 1) == Catch::Approx(wanted).epsilon(1e-14));
}

TEST_CASE("FPU is absolute at root and reduction uses unclamped rounded visited prior mass",
          "[search][selection]") {
    mouse::search::SelectionParameters p;
    p.root_fpu_strategy = mouse::search::RootFpuStrategy::Absolute;
    CHECK(mouse::search::first_play_urgency(p, true, -1.0, 0.0) == 1.0);
    CHECK(mouse::search::first_play_urgency(p, true, 1.0, 1.001) == 1.0);
    CHECK(mouse::search::first_play_urgency(p, false, -0.9, 1.001) ==
          Catch::Approx(-0.9 - p.fpu_reduction * std::sqrt(1.001)).epsilon(1e-14));
    CHECK(mouse::search::first_play_urgency(p, false, 0.25, 0.0) == 0.25);
    CHECK(
        mouse::search::exploration_multiplier(p, std::numeric_limits<std::uint64_t>::max()) ==
        Catch::Approx(mouse::search::cpuct_value(p, std::numeric_limits<std::uint64_t>::max()) *
                      std::sqrt(static_cast<double>(std::numeric_limits<std::uint64_t>::max() - 1)))
            .epsilon(1e-14));
}

TEST_CASE("Default root FPU uses the same reduction as every other node",
          "[search][selection][root-fpu]") {
    const mouse::search::SelectionParameters p;
    for (const double parent_q : {-1.0, -0.5, 0.25, 1.0}) {
        for (const double mass : {0.0, 0.25, 0.75, 1.001}) {
            const auto expected = parent_q - 0.330 * std::sqrt(mass);
            CHECK(mouse::search::first_play_urgency(p, true, parent_q, mass) ==
                  Catch::Approx(expected).epsilon(1e-14));
            CHECK(mouse::search::first_play_urgency(p, true, parent_q, mass) ==
                  mouse::search::first_play_urgency(p, false, parent_q, mass));
        }
    }
}

TEST_CASE("Root FPU strategies use their configured value only at the root",
          "[search][selection][root-fpu]") {
    using namespace mouse::search;
    SelectionParameters p;
    p.fpu_reduction = 0.2;
    p.root_fpu = -0.75;
    p.root_fpu_strategy = RootFpuStrategy::Same;
    REQUIRE(validate_selection_parameters(p));
    CHECK(first_play_urgency(p, true, 0.5, 0.25) == Catch::Approx(0.4));
    p.root_fpu_strategy = RootFpuStrategy::Reduction;
    REQUIRE(validate_selection_parameters(p));
    CHECK(first_play_urgency(p, true, 0.5, 0.25) == Catch::Approx(0.875));
    CHECK(first_play_urgency(p, false, 0.5, 0.25) == Catch::Approx(0.4));
    p.root_fpu_strategy = RootFpuStrategy::Absolute;
    REQUIRE(validate_selection_parameters(p));
    CHECK(first_play_urgency(p, true, 0.5, 0.25) == -0.75);
    CHECK(first_play_urgency(p, false, 0.5, 0.25) == Catch::Approx(0.4));
    for (const auto strategy :
         {RootFpuStrategy::Same, RootFpuStrategy::Reduction, RootFpuStrategy::Absolute}) {
        for (const double value : {-100.0, 100.0}) {
            p.root_fpu_strategy = strategy;
            p.root_fpu = value;
            CHECK(validate_selection_parameters(p));
            CHECK(std::isfinite(first_play_urgency(p, true, -1, 1.001)));
        }
    }
    p.root_fpu_strategy = static_cast<RootFpuStrategy>(255);
    CHECK_FALSE(validate_selection_parameters(p));
}

TEST_CASE("Final move selection matches native proven mate over retained draw",
          "[search][selection][terminal-choice-v1]") {
#include "tests/data/search/terminal_choice_v1.inc"
    CHECK(mouse::search::prefer_final_move(kNativeTerminalMate, kNativeTerminalDraw));
    CHECK_FALSE(mouse::search::prefer_final_move(kNativeTerminalDraw, kNativeTerminalMate));
}

TEST_CASE("Final outcome priority requires visited proof and preserves ordinary tie rules",
          "[search][selection][terminal-choice-v1]") {
    using mouse::search::FinalMoveStatistics;
    using mouse::search::prefer_final_move;
    using mouse::search::ProofOutcome;
    const FinalMoveStatistics mate{
        .visits = 1, .q = 1, .child_proof = {.outcome = ProofOutcome::Loss}};
    const FinalMoveStatistics neural_win{.visits = 1000, .q = 1, .prior = 1};
    CHECK(prefer_final_move(mate, neural_win));
    auto loss = mate;
    loss.child_proof = {.outcome = ProofOutcome::Win, .mate_upper_bound = 3};
    loss.visits = 10000;
    loss.q = -1;
    const FinalMoveStatistics unknown{.visits = 1, .q = -1};
    CHECK(prefer_final_move(unknown, loss));
    auto unvisited = mate;
    unvisited.visits = 0;
    CHECK(prefer_final_move(unknown, unvisited));
    auto farther = mate;
    farther.child_proof.mate_upper_bound = 4;
    farther.visits = 10000;
    CHECK(prefer_final_move(mate, farther));
    auto delayed_loss = loss;
    delayed_loss.child_proof.mate_upper_bound = 5;
    delayed_loss.visits = 1;
    CHECK(prefer_final_move(delayed_loss, loss));
    auto more_visits = unknown;
    ++more_visits.visits;
    CHECK(prefer_final_move(more_visits, unknown));
    auto better_q = unknown;
    better_q.q = 0;
    CHECK(prefer_final_move(better_q, unknown));
    auto higher_prior = unknown;
    higher_prior.prior = 0.5F;
    CHECK(prefer_final_move(higher_prior, unknown));
    auto zero_a = neural_win, zero_b = unknown;
    zero_a.visits = zero_b.visits = 0;
    zero_a.prior = zero_b.prior = 0;
    CHECK_FALSE(prefer_final_move(zero_a, zero_b));
    CHECK_FALSE(prefer_final_move(zero_b, zero_a));
    const std::array values{mate,         neural_win,  loss,     unknown,      unvisited, farther,
                            delayed_loss, more_visits, better_q, higher_prior, zero_a,    zero_b};
    for (const auto& a : values) {
        CHECK_FALSE(prefer_final_move(a, a));
        for (const auto& b : values) {
            if (prefer_final_move(a, b)) CHECK_FALSE(prefer_final_move(b, a));
            for (const auto& c : values)
                if (prefer_final_move(a, b) && prefer_final_move(b, c))
                    CHECK(prefer_final_move(a, c));
        }
    }
}
