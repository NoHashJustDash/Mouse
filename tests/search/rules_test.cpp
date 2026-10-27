// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/search/rules.h"

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <limits>
#include <string_view>

#include "engine/chess/attacks.h"
#include "engine/nn/history.h"

TEST_CASE("Search mate precedes rule50 and Position history reserve is explicit",
          "[search][rules]") {
    const mouse::chess::AttackTables attacks;
    auto position = mouse::chess::Position::from_fen("7k/6Q1/5K2/8/8/8/8/8 b - - 100 50", attacks);
    REQUIRE(position);
    const auto status = mouse::search::classify_leaf(*position, attacks, {});
    REQUIRE(status);
    CHECK(status->reason == mouse::search::TerminalReason::Checkmate);
    CHECK(status->q == -1.0F);
    CHECK(status->draw == 0.0F);
    REQUIRE(position->reserve_history(513));
    CHECK(position->history_storage_bytes() >= 514 * sizeof(std::uint64_t));
    mouse::nn::EncodingHistory history;
    history.reset(*position);
    const auto before = history;
    REQUIRE(history.reserve_history(513));
    CHECK(history == before);
    CHECK(history.frames().front() == before.frames().front());
}

namespace {
mouse::chess::Position parsed(std::string_view fen, const mouse::chess::AttackTables& attacks) {
    auto result = mouse::chess::Position::from_fen(fen, attacks);
    if (!result) FAIL(result.error());
    return *result;
}
void cycle(mouse::chess::Position& position, const mouse::chess::AttackTables& attacks) {
    for (const auto text : {"g1f3", "g8f6", "f3g1", "f6g8"}) {
        const auto move = mouse::chess::parse_uci_move(position, text, attacks);
        REQUIRE(move);
        (void)position.make_move(*move, attacks);
    }
}
}  // namespace

TEST_CASE("Leaf classification distinguishes exact terminal reasons and both perspectives",
          "[search][rules]") {
    using mouse::search::TerminalReason;
    const mouse::chess::AttackTables attacks;
    struct Row {
        std::string_view fen;
        TerminalReason reason;
        float q;
        float draw;
    };
    const std::array rows{Row{.fen = "8/8/8/8/8/5k2/6q1/7K w - - 100 50",
                              .reason = TerminalReason::Checkmate,
                              .q = -1,
                              .draw = 0},
                          Row{.fen = "7k/5K2/6Q1/8/8/8/8/8 b - - 100 50",
                              .reason = TerminalReason::Stalemate,
                              .q = 0,
                              .draw = 1},
                          Row{.fen = "8/8/8/8/8/1q6/2k5/K7 w - - 100 50",
                              .reason = TerminalReason::Stalemate,
                              .q = 0,
                              .draw = 1},
                          Row{.fen = "7k/8/8/8/8/8/8/R3K3 w - - 100 50",
                              .reason = TerminalReason::FiftyMove,
                              .q = 0,
                              .draw = 1},
                          Row{.fen = "7k/8/8/8/8/8/8/R3K3 w - - 99 50",
                              .reason = TerminalReason::None,
                              .q = 0,
                              .draw = 0},
                          Row{.fen = "7k/8/8/8/8/8/8/4K3 w - - 0 1",
                              .reason = TerminalReason::InsufficientMaterial,
                              .q = 0,
                              .draw = 1},
                          Row{.fen = "7k/8/8/8/8/8/8/4K3 b - - 0 1",
                              .reason = TerminalReason::InsufficientMaterial,
                              .q = 0,
                              .draw = 1}};
    for (const auto& row : rows) {
        auto position = parsed(row.fen, attacks);
        const auto before = position;
        const auto status = mouse::search::classify_leaf(position, attacks, {});
        REQUIRE(status);
        CHECK(status->reason == row.reason);
        CHECK(status->q == row.q);
        CHECK(status->draw == row.draw);
        CHECK_FALSE(status->contextual);
        CHECK(position == before);
    }
}

TEST_CASE("Game threefold and current-tree twofold have separate contexts", "[search][rules]") {
    using mouse::search::TerminalReason;
    const mouse::chess::AttackTables attacks;
    auto position = parsed(mouse::chess::kStartFen, attacks);
    cycle(position, attacks);
    REQUIRE(position.is_repetition(2));
    REQUIRE_FALSE(position.is_repetition(3));
    const auto ongoing = mouse::search::classify_leaf(position, attacks, {});
    REQUIRE(ongoing);
    CHECK(ongoing->reason == TerminalReason::None);
    const std::array ancestors{position.key()};
    const auto repeated = mouse::search::classify_leaf(position, attacks, ancestors);
    REQUIRE(repeated);
    CHECK(repeated->reason == TerminalReason::TreeRepetition);
    CHECK(repeated->q == 0);
    CHECK(repeated->draw == 1);
    CHECK(repeated->contextual);
    cycle(position, attacks);
    const auto game = mouse::search::classify_leaf(position, attacks, ancestors);
    REQUIRE(game);
    CHECK(game->reason == TerminalReason::Threefold);
    CHECK_FALSE(game->contextual);
}

TEST_CASE("Exact encoding history equality includes clocks repetitions EP and earlier frames",
          "[search][history]") {
    const mouse::chess::AttackTables attacks;
    auto root = parsed(mouse::chess::kStartFen, attacks);
    const auto frame = mouse::nn::HistoryFrame::capture(root);
    CHECK(frame != mouse::nn::HistoryFrame::capture(root, 1));
    auto clock = parsed("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 1 1", attacks);
    CHECK(frame != mouse::nn::HistoryFrame::capture(clock));
    auto ep = parsed("1n2r1k1/8/8/3pP3/8/8/8/1N2K3 w - d6 0 2", attacks);
    auto no_ep = parsed("1n2r1k1/8/8/3pP3/8/8/8/1N2K3 w - - 0 2", attacks);
    REQUIRE(ep.key() == no_ep.key());
    CHECK(mouse::nn::HistoryFrame::capture(ep) != mouse::nn::HistoryFrame::capture(no_ep));
    mouse::nn::EncodingHistory history;
    history.reset(root);
    for (const auto text : {"e2e4", "e7e5", "g1f3"}) {
        auto move = mouse::chess::parse_uci_move(root, text, attacks);
        REQUIRE(move);
        (void)root.make_move(*move, attacks);
        REQUIRE(history.append(root));
    }
    const auto copy = history;
    CHECK(copy == history);
    mouse::nn::EncodingHistory fresh;
    fresh.reset(root);
    CHECK(fresh.frames().back() == history.frames().back());
    CHECK(fresh != history);
}

TEST_CASE("Reserved histories retain actual capacities through512 plies and one legal scratch ply",
          "[search][history]") {
    const mouse::chess::AttackTables attacks;
    auto position = parsed(mouse::chess::kStartFen, attacks);
    const auto before = position;
    mouse::nn::EncodingHistory history;
    CHECK(history.history_storage_bytes() == 0);
    history.reset(position);
    const auto original = history;
    REQUIRE(position.reserve_history(513));
    REQUIRE(history.reserve_history(513));
    const auto position_bytes = position.history_storage_bytes();
    const auto history_bytes = history.history_storage_bytes();
    CHECK(position_bytes >= 514 * sizeof(std::uint64_t) + 64);
    CHECK(history_bytes >= 514 * sizeof(mouse::nn::HistoryFrame) + 64);
    CHECK_FALSE(position.reserve_history(std::numeric_limits<std::size_t>::max()));
    CHECK_FALSE(history.reserve_history(std::numeric_limits<std::size_t>::max()));
    CHECK(position == before);
    CHECK(history == original);
    std::array<mouse::chess::Move, 513> moves{};
    std::array<mouse::chess::Undo, 513> undos{};
    constexpr std::array texts{"g1f3", "g8f6", "f3g1", "f6g8"};
    for (std::size_t ply = 0; ply < 513; ++ply) {
        const auto move = mouse::chess::parse_uci_move(position, texts.at(ply % 4), attacks);
        REQUIRE(move);
        moves.at(ply) = *move;
        undos.at(ply) = position.make_move(*move, attacks);
        REQUIRE(history.append(position));
        CHECK(position.history_storage_bytes() == position_bytes);
        CHECK(history.history_storage_bytes() == history_bytes);
    }
    for (std::size_t ply = 513; ply-- > 0;) {
        REQUIRE(history.pop());
        position.unmake_move(moves.at(ply), undos.at(ply));
    }
    CHECK(position == before);
    CHECK(history == original);
}

TEST_CASE("Rule classification reports counter failure without changing the board",
          "[search][rules]") {
    const mouse::chess::AttackTables attacks;
    auto position = parsed("7k/8/8/8/8/8/8/K7 b - - 0 18446744073709551615", attacks);
    const auto before = position;
    const auto result = mouse::search::classify_leaf(position, attacks, {});
    CHECK_FALSE(result);
    CHECK(position == before);
}

TEST_CASE("Copy assignment retains separately reserved worker capacities", "[search][history]") {
    const mouse::chess::AttackTables attacks;
    const auto root = parsed(mouse::chess::kStartFen, attacks);
    mouse::nn::EncodingHistory root_history;
    root_history.reset(root);
    auto worker = root;
    auto history = root_history;
    REQUIRE(worker.reserve_history(513));
    REQUIRE(history.reserve_history(513));
    const auto position_bytes = worker.history_storage_bytes();
    const auto history_bytes = history.history_storage_bytes();
    for (int iteration = 0; iteration < 32; ++iteration) {
        worker = root;
        history = root_history;
        CHECK(worker == root);
        CHECK(history == root_history);
        CHECK(worker.history_storage_bytes() == position_bytes);
        CHECK(history.history_storage_bytes() == history_bytes);
    }
}

TEST_CASE("Explicit root context skips only rule draws and retains complete legal moves",
          "[search][rules][root-draw]") {
    using namespace mouse::search;
    const mouse::chess::AttackTables attacks;
    for (const auto fen : {"7k/8/8/8/8/8/8/R3K3 w - - 100 50", "7k/8/8/8/8/8/8/4K3 w - - 0 1"}) {
        auto p = parsed(fen, attacks);
        const auto before = p;
        const auto child = classify_leaf(p, attacks, {}, LeafContext::NonRoot);
        const auto root = classify_leaf(p, attacks, {}, LeafContext::Root);
        REQUIRE(child);
        REQUIRE(root);
        CHECK(child->reason != TerminalReason::None);
        CHECK(root->reason == TerminalReason::None);
        CHECK(root->draw == 0);
        REQUIRE_FALSE(root->legal_moves.empty());
        CHECK(std::ranges::equal(root->legal_moves, child->legal_moves));
        CHECK(p == before);
    }
    auto repeated = parsed(mouse::chess::kStartFen, attacks);
    cycle(repeated, attacks);
    cycle(repeated, attacks);
    const std::array ancestors{repeated.key()};
    const auto root = classify_leaf(repeated, attacks, ancestors, LeafContext::Root);
    const auto child = classify_leaf(repeated, attacks, ancestors, LeafContext::NonRoot);
    REQUIRE(root);
    REQUIRE(child);
    CHECK(root->reason == TerminalReason::None);
    CHECK_FALSE(root->contextual);
    CHECK(child->reason == TerminalReason::Threefold);
    for (const auto fen :
         {"7k/6Q1/5K2/8/8/8/8/8 b - - 100 50", "7k/5K2/6Q1/8/8/8/8/8 b - - 100 50"}) {
        auto p = parsed(fen, attacks);
        const auto terminal = classify_leaf(p, attacks, {}, LeafContext::Root);
        REQUIRE(terminal);
        CHECK(terminal->legal_moves.empty());
        CHECK(terminal->reason != TerminalReason::None);
    }
}
