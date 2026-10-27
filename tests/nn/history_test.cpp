// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/nn/history.h"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <string_view>
#include <vector>

#include "engine/chess/attacks.h"
#include "engine/chess/movegen.h"

namespace {
using mouse::chess::AttackTables;
using mouse::chess::Position;
using mouse::nn::EncodingHistory;

Position position(std::string_view fen, const AttackTables& attacks) {
    auto parsed = Position::from_fen(fen, attacks);
    if (!parsed) FAIL(parsed.error());
    return *parsed;
}

mouse::chess::Undo append_move(Position& current, EncodingHistory& history, std::string_view text,
                               const AttackTables& attacks) {
    const auto move = mouse::chess::parse_uci_move(current, text, attacks, true);
    if (!move) FAIL(move.error());
    const auto undo = current.make_move(*move, attacks);
    const auto appended = history.append(current);
    if (!appended) FAIL(appended.error());
    return undo;
}

constexpr std::array<std::string_view, 4> knight_cycle{"b1c3", "b8c6", "c3b1", "c6b8"};
}  // namespace

TEST_CASE("Encoding history refuses invalid ownership operations without mutation",
          "[nn][history]") {
    const AttackTables attacks;
    auto current = position(mouse::chess::kStartFen, attacks);
    EncodingHistory history;
    CHECK(history.frames().empty());
    CHECK_FALSE(history.pop());
    CHECK_FALSE(history.append(current));
    history.reset(current);
    REQUIRE(history.frames().size() == 1);
    const auto before = mouse::nn::encode_classical(history.frames());
    REQUIRE(before.has_value());
    const auto repeated_side = history.append(current);
    REQUIRE_FALSE(repeated_side);
    CHECK(repeated_side.error().find("side") != std::string::npos);
    CHECK_FALSE(history.pop());
    REQUIRE(history.frames().size() == 1);
    CHECK(mouse::nn::encode_classical(history.frames()) == before);
}

TEST_CASE("Encoding history counts reversible repetitions independently of counters",
          "[nn][history]") {
    const AttackTables attacks;
    auto current = position(mouse::chess::kStartFen, attacks);
    EncodingHistory history;
    history.reset(current);
    for (std::uint32_t cycle = 1; cycle <= 20; ++cycle) {
        for (const auto move : knight_cycle) append_move(current, history, move, attacks);
        REQUIRE(history.frames().size() == 1 + 4 * cycle);
        CHECK(history.frames().back().repetitions() == cycle);
        CHECK(history.frames().back().halfmove_clock() == std::uint64_t{4} * cycle);
        CHECK(current.fullmove_number() == 1 + 2 * cycle);
    }
    CHECK(history.frames().front().repetitions() == 0);
    history.reset(current);
    REQUIRE(history.frames().size() == 1);
    CHECK(history.frames().front().repetitions() == 0);
    CHECK(history.frames().front().halfmove_clock() == 80);
}

TEST_CASE("Pinned raw root EP differs from FIDE repetition until the second cycle",
          "[nn][history]") {
    const AttackTables attacks;
    auto current = position("1n2r1k1/8/8/3pP3/8/8/8/1N2K3 w - d6 0 2", attacks);
    EncodingHistory history;
    history.reset(current);
    CHECK(history.frames().front().en_passant_square() == mouse::chess::square(3, 5));
    for (const auto move : knight_cycle) append_move(current, history, move, attacks);
    CHECK(current.is_repetition(2));
    CHECK(history.frames().back().repetitions() == 0);
    CHECK(history.frames().back().en_passant_square() == mouse::chess::kNoSquare);
    for (const auto move : knight_cycle) append_move(current, history, move, attacks);
    CHECK(current.is_repetition(3));
    CHECK(history.frames().back().repetitions() == 1);
    const auto encoded = mouse::nn::encode_classical(history.frames());
    REQUIRE(encoded.has_value());
    CHECK((*encoded)[12].mask == ~std::uint64_t{0});
    CHECK((*encoded)[4 * 13 + 12].mask == 0);
}

TEST_CASE("Appended double pushes retain EP only for a geometric capturer", "[nn][history]") {
    const AttackTables attacks;
    SECTION("No adjacent pawn") {
        auto current = position(mouse::chess::kStartFen, attacks);
        EncodingHistory history;
        history.reset(current);
        append_move(current, history, "e2e4", attacks);
        CHECK(current.en_passant_square() == mouse::chess::square(4, 2));
        CHECK(history.frames().back().en_passant_square() == mouse::chess::kNoSquare);
        history.reset(current);
        CHECK(history.frames().front().en_passant_square() == mouse::chess::square(4, 2));
    }
    SECTION("Adjacent black pawn") {
        auto current = position("4k3/8/8/8/3p4/8/4P3/4K3 w - - 0 1", attacks);
        EncodingHistory history;
        history.reset(current);
        append_move(current, history, "e2e4", attacks);
        CHECK(history.frames().back().en_passant_square() == mouse::chess::square(4, 2));
    }
    SECTION("Adjacent pinned white pawn still qualifies") {
        auto current = position("1n2r1k1/3p4/8/4P3/8/8/8/1N2K3 b - - 0 1", attacks);
        EncodingHistory history;
        history.reset(current);
        append_move(current, history, "d7d5", attacks);
        CHECK(history.frames().back().en_passant_square() == mouse::chess::square(3, 5));
        CHECK(current.key() == position("1n2r1k1/8/8/3pP3/8/8/8/1N2K3 w - - 0 2", attacks).key());
    }
}

TEST_CASE("Repetition identities include active castling rights", "[nn][history]") {
    const AttackTables attacks;
    auto current = position("rn2k2r/8/8/8/8/8/8/RN2K2R w KQkq - 0 1", attacks);
    EncodingHistory history;
    history.reset(current);
    constexpr std::array<std::string_view, 4> cycle{"e1e2", "e8e7", "e2e1", "e7e8"};
    for (const auto move : cycle) append_move(current, history, move, attacks);
    CHECK(history.frames().back().repetitions() == 0);
    for (const auto move : cycle) append_move(current, history, move, attacks);
    CHECK(history.frames().back().repetitions() == 1);
}

TEST_CASE("History pop follows Position unmake and reset retains no stale repetitions",
          "[nn][history]") {
    const AttackTables attacks;
    auto current = position(mouse::chess::kStartFen, attacks);
    const auto root = current;
    EncodingHistory history;
    history.reset(current);
    std::vector<mouse::chess::Move> moves;
    std::vector<mouse::chess::Undo> undos;
    std::vector<mouse::nn::InputPlanes> encoded;
    for (int cycle = 0; cycle < 3; ++cycle) {
        for (const auto text : knight_cycle) {
            const auto planes = mouse::nn::encode_classical(history.frames());
            REQUIRE(planes.has_value());
            encoded.push_back(*planes);
            const auto move = mouse::chess::parse_uci_move(current, text, attacks, true);
            REQUIRE(move.has_value());
            moves.push_back(*move);
            undos.push_back(append_move(current, history, text, attacks));
        }
    }
    while (!moves.empty()) {
        REQUIRE(history.pop().has_value());
        current.unmake_move(moves.back(), undos.back());
        CHECK(history.frames().back().side_to_move() == current.side_to_move());
        CHECK(history.frames().back().halfmove_clock() == current.halfmove_clock());
        CHECK(mouse::nn::encode_classical(history.frames()) == encoded.back());
        moves.pop_back();
        undos.pop_back();
        encoded.pop_back();
    }
    CHECK(current == root);
    CHECK_FALSE(history.pop());
    append_move(current, history, "e2e4", attacks);
    CHECK(history.frames().back().halfmove_clock() == 0);
    CHECK(history.frames().back().repetitions() == 0);
    history.reset(root);
    REQUIRE(history.frames().size() == 1);
    CHECK(history.frames().front().is_initial_board());
    CHECK(history.frames().front().repetitions() == 0);
}
