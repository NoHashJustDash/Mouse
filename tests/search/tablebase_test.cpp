// SPDX-License-Identifier: GPL-3.0-or-later
// Known answers from lc0 fd71a2d src/syzygy/syzygy_test.cc. These read real table files, so they
// run on request: MOUSE_SYZYGY_DIR names the directory, and the tag is hidden from a plain run.
#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <string>
#include <vector>

#include "engine/chess/attacks.h"
#include "engine/chess/movegen.h"
#include "engine/search/search.h"
#include "engine/search/tablebase.h"

namespace {
using mouse::search::Tablebase;
using Wdl = mouse::search::TablebaseWdl;
std::shared_ptr<Tablebase> tables() {
    const char* directory = std::getenv("MOUSE_SYZYGY_DIR");
    REQUIRE(directory != nullptr);
    auto opened = Tablebase::open(directory);
    REQUIRE(opened);
    REQUIRE((*opened)->max_cardinality() >= 5);
    return *opened;
}
void expect(Tablebase& tablebase, const char* fen, Wdl wdl, int dtz) {
    INFO(fen);
    mouse::chess::AttackTables attacks;
    const auto position = mouse::chess::Position::from_fen(fen, attacks);
    REQUIRE(position);
    const auto probed = tablebase.probe_wdl(*position, attacks);
    REQUIRE(probed);
    CHECK(*probed == wdl);
    const auto distance = tablebase.probe_dtz(*position, attacks);
    REQUIRE(distance);
    CHECK(*distance == dtz);
}
void expect_root(Tablebase& tablebase, const char* fen, const std::vector<const char*>& valid,
                 const std::vector<const char*>& invalid,
                 const std::vector<const char*>& invalid_dtz_only = {}, bool has_repeated = false) {
    INFO(fen);
    mouse::chess::AttackTables attacks;
    auto position = mouse::chess::Position::from_fen(fen, attacks);
    REQUIRE(position);
    const auto dtz = tablebase.root_moves_dtz(*position, attacks, has_repeated);
    const auto wdl = tablebase.root_moves_wdl(*position, attacks);
    const auto holds = [&](const std::vector<mouse::chess::Move>& moves, const char* text) {
        const auto move = mouse::chess::parse_uci_move(*position, text, attacks, false);
        REQUIRE(move);
        return std::ranges::find(moves, *move) != moves.end();
    };
    for (const auto* move : valid) {
        INFO(move);
        CHECK(holds(dtz, move));
        CHECK(holds(wdl, move));
    }
    for (const auto* move : invalid) {
        INFO(move);
        CHECK_FALSE(holds(dtz, move));
        CHECK_FALSE(holds(wdl, move));
    }
    for (const auto* move : invalid_dtz_only) {
        INFO(move);
        CHECK_FALSE(holds(dtz, move));
        CHECK(holds(wdl, move));
    }
}
}  // namespace

TEST_CASE("Syzygy 3 piece probes", "[.syzygy]") {
    const auto tablebase = tables();
    expect(*tablebase, "8/8/8/8/8/8/2Rk4/1K6 b - - 0 1", Wdl::Loss, -32);
    expect(*tablebase, "8/8/8/8/8/8/2rK4/1k6 w - - 0 1", Wdl::Loss, -32);
    expect(*tablebase, "8/8/8/8/8/8/4kR2/6K1 b - - 0 1", Wdl::Loss, -32);
    expect(*tablebase, "6K1/4kR2/8/8/8/8/8/8 b - - 0 1", Wdl::Loss, -32);
    expect(*tablebase, "1K6/2Rk4/8/8/8/8/8/8 b - - 0 1", Wdl::Loss, -32);
    expect(*tablebase, "5Qk1/8/8/8/8/8/8/4K3 b - - 0 1", Wdl::Draw, 0);
    expect(*tablebase, "6k1/8/8/8/8/5p2/8/2K5 b - - 0 1", Wdl::Win, 1);
    expect(*tablebase, "8/8/8/8/8/k1p5/8/3K4 b - - 0 1", Wdl::Win, 3);
    expect(*tablebase, "8/2p5/8/8/8/5k2/8/2K5 w - - 0 1", Wdl::Loss, -8);
}
TEST_CASE("Syzygy 3 piece root moves", "[.syzygy]") {
    const auto tablebase = tables();
    expect_root(*tablebase, "5Qk1/8/8/8/8/8/8/4K3 b - - 0 1", {"g8f8"}, {"g8h7"});
    expect_root(*tablebase, "6k1/8/8/8/8/5p2/8/2K5 b - - 0 1", {"f3f2"}, {"g8h7"});
    expect_root(*tablebase, "8/8/8/8/8/k1p5/8/3K4 b - - 0 1", {"a3b3"}, {"c3c2"});
    // WDL does not know that so high a 50-move count has made this a blessed loss for black.
    expect_root(*tablebase, "8/8/8/8/8/8/2Rk4/1K6 b - - 69 71", {"d2d3"}, {}, {"d2e3"});
}
TEST_CASE("Syzygy 4 piece probes", "[.syzygy]") {
    const auto tablebase = tables();
    expect(*tablebase, "8/8/8/6B1/8/8/4k3/1K5N b - - 0 1", Wdl::Loss, -65);
    expect(*tablebase, "8/8/8/8/8/2p5/3q2k1/4K3 w - - 0 1", Wdl::Loss, -1);
    expect(*tablebase, "7k/8/8/8/Pp2K3/8/8/8 b - a3 0 1", Wdl::Win, 1);
    expect(*tablebase, "7k/8/8/8/Pp2K3/8/8/8 b - - 0 1", Wdl::Loss, -1);
}
TEST_CASE("Syzygy 5 piece probes", "[.syzygy]") {
    const auto tablebase = tables();
    expect(*tablebase, "8/8/8/8/1p2P3/4P3/1k6/3K4 w - - 0 1", Wdl::CursedWin, 101);
    expect(*tablebase, "8/6B1/8/8/B7/8/K1pk4/8 b - - 0 1", Wdl::BlessedLoss, -101);
    expect(*tablebase, "k7/p7/8/8/3Q4/8/5B2/7K w - - 0 1", Wdl::Win, 1);
    expect(*tablebase, "8/8/8/8/4pk2/R7/7r/4K3 b - - 0 1", Wdl::Draw, 0);
    expect(*tablebase, "3k4/R7/7r/2KP4/8/8/8/8 w - - 0 1", Wdl::Draw, 0);
    expect(*tablebase, "8/8/8/8/6Pp/7K/5Q2/7k b - g3 0 1", Wdl::Loss, -1);
    expect(*tablebase, "8/8/8/8/6Pp/7K/5Q2/7k b - - 0 1", Wdl::Draw, 0);
    expect(*tablebase, "kqqQK3/8/8/8/8/8/8/8 b - - 0 1", Wdl::Win, 1);
    expect(*tablebase, "kqqQK3/8/8/8/8/8/8/8 w - - 0 1", Wdl::Loss, -2);
    expect(*tablebase, "KNNNk3/8/8/8/8/8/8/8 w - - 0 1", Wdl::Win, 30);
    expect(*tablebase, "8/1k6/1p1r4/5K2/8/8/8/2R5 w - - 0 1", Wdl::Draw, 0);
    expect(*tablebase, "8/7p/5k2/8/5PK1/7P/8/8 b - - 0 1", Wdl::Draw, 0);
    expect(*tablebase, "1k1n4/8/p7/5KP1/8/8/8/8 b - - 0 1", Wdl::Win, 5);
    expect(*tablebase, "8/k7/8/2R5/8/4q3/8/4B2K w - - 0 1", Wdl::Draw, 0);
}
TEST_CASE("Syzygy 5 piece root moves", "[.syzygy]") {
    const auto tablebase = tables();
    expect_root(*tablebase, "8/8/8/Q7/8/1k1K4/1r6/8 w - - 79 44", {"a5a1"}, {}, {"a5d5"});
    expect_root(*tablebase, "8/8/8/3Q4/k7/3K4/1r6/8 w - - 81 45", {"d5a8"}, {}, {"d3c3"});
    expect_root(*tablebase, "8/8/8/Q7/8/1k1K4/1r6/8 w - - 60 44", {"a5a1", "a5d5"}, {});
    // A repetition in the history makes DTZ insist on the equal lowest distance.
    expect_root(*tablebase, "8/8/8/Q7/8/1k1K4/1r6/8 w - - 60 44", {"a5a1"}, {}, {"a5d5"}, true);
}
