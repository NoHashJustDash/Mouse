// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/uci/report.h"

#include <catch2/catch_test_macros.hpp>
TEST_CASE("Reporting preserves the authoritative move when root visit rankings advance",
          "[uci][publication-pv-v1]") {
    mouse::search::RootStatistics roots;
    roots.size = 3;
    roots.moves[0] = {.move = mouse::chess::Move{1, 18},
                      .visits = 10,
                      .q = 0.2,
                      .prior = 0.3F,
                      .child_proof = {}};
    roots.moves[1] = {.move = mouse::chess::Move{6, 21},
                      .visits = 20,
                      .q = 0.1,
                      .prior = 0.4F,
                      .child_proof = {}};
    roots.moves[2] = {
        .move = mouse::chess::Move{8, 16}, .visits = 5, .q = 0.4, .prior = 0.3F, .child_proof = {}};
    mouse::uci::rank_report_roots(roots, roots.moves[0].move);
    CHECK(roots.moves[0].move == mouse::chess::Move{1, 18});
    CHECK(roots.moves[1].move == mouse::chess::Move{6, 21});
    CHECK(roots.moves[2].move == mouse::chess::Move{8, 16});
    CHECK(roots.moves[0].visits == 10);
    mouse::uci::rank_report_roots(roots, std::nullopt);
    CHECK(roots.moves[0].move == mouse::chess::Move{6, 21});
    mouse::uci::rank_report_roots(roots, mouse::chess::Move{9, 17});
    CHECK(roots.size == 3);
    CHECK(roots.moves[0].move == mouse::chess::Move{6, 21});
}

TEST_CASE("MultiPV diagnostics keep proven losses after ordinary alternatives",
          "[uci][publication-pv-v1][terminal-choice-v1]") {
    mouse::search::RootStatistics roots;
    roots.size = 3;
    const mouse::chess::Move win{1, 18}, ordinary{6, 21}, loss{8, 16};
    roots.moves[0] = {
        .move = loss,
        .visits = 100,
        .q = -1,
        .child_proof = {.outcome = mouse::search::ProofOutcome::Win, .mate_upper_bound = 3}};
    roots.moves[1] = {.move = ordinary, .visits = 2, .child_proof = {}};
    roots.moves[2] = {.move = win,
                      .visits = 1,
                      .q = 1,
                      .child_proof = {.outcome = mouse::search::ProofOutcome::Loss}};
    mouse::uci::rank_report_roots(roots, win);
    CHECK(roots.moves[0].move == win);
    CHECK(roots.moves[1].move == ordinary);
    CHECK(roots.moves[2].move == loss);
}
