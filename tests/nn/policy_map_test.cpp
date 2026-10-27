// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/nn/policy_map.h"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <string>

#include "engine/chess/attacks.h"
#include "engine/chess/movegen.h"

namespace {
using mouse::chess::Color;
using mouse::chess::Move;
using mouse::chess::MoveKind;
using mouse::chess::PieceType;
using mouse::chess::Square;
using mouse::chess::square;
using mouse::nn::PolicyMap;

Square mirrored(Square at) {
    return square(mouse::chess::file_of(at), 7 - mouse::chess::rank_of(at));
}

void require_index(Move move, Color color, std::uint16_t expected) {
    const auto actual = PolicyMap::index(move, color);
    if (!actual) FAIL(actual.error());
    CHECK(*actual == expected);
}
}  // namespace

TEST_CASE("Policy geometry enumerates every fixed ordinary and Q R B promotion slot",
          "[nn][policy]") {
    std::array<bool, 1858> seen{};
    std::uint16_t expected = 0;
    // Build the expected ordinary destinations by walking independent rays and
    // knight displacements, then visit the resulting squares in board order.
    constexpr std::array ray_directions{std::pair{-1, -1}, std::pair{-1, 0}, std::pair{-1, 1},
                                        std::pair{0, -1},  std::pair{0, 1},  std::pair{1, -1},
                                        std::pair{1, 0},   std::pair{1, 1}};
    constexpr std::array knight_steps{std::pair{-2, -1}, std::pair{-2, 1}, std::pair{-1, -2},
                                      std::pair{-1, 2},  std::pair{1, -2}, std::pair{1, 2},
                                      std::pair{2, -1},  std::pair{2, 1}};
    for (Square from = 0; from < 64; ++from) {
        std::array<bool, 64> destinations{};
        for (const auto& [dx, dy] : ray_directions) {
            int file = mouse::chess::file_of(from) + dx;
            int rank = mouse::chess::rank_of(from) + dy;
            while (file >= 0 && file < 8 && rank >= 0 && rank < 8) {
                destinations.at(square(file, rank)) = true;
                file += dx;
                rank += dy;
            }
        }
        for (const auto& [dx, dy] : knight_steps) {
            const int file = mouse::chess::file_of(from) + dx;
            const int rank = mouse::chess::rank_of(from) + dy;
            if (file >= 0 && file < 8 && rank >= 0 && rank < 8)
                destinations.at(square(file, rank)) = true;
        }
        for (Square to = 0; to < 64; ++to) {
            if (!destinations.at(to)) continue;
            const auto slot = PolicyMap::index(Move{from, to}, Color::White);
            REQUIRE(slot.has_value());
            REQUIRE(*slot < seen.size());
            CHECK_FALSE(seen.at(*slot));
            seen.at(*slot) = true;
            CHECK(*slot == expected);
            require_index(Move{mirrored(from), mirrored(to)}, Color::Black, expected);
            ++expected;
        }
    }
    REQUIRE(expected == 1792);
    for (int from_file = 0; from_file < 8; ++from_file) {
        for (int to_file = 0; to_file < 8; ++to_file) {
            if (to_file < from_file - 1 || to_file > from_file + 1) continue;
            for (const auto promotion : {PieceType::Queen, PieceType::Rook, PieceType::Bishop}) {
                const Move move{square(from_file, 6), square(to_file, 7), promotion};
                const auto slot = PolicyMap::index(move, Color::White);
                REQUIRE(slot.has_value());
                REQUIRE(*slot < seen.size());
                CHECK_FALSE(seen.at(*slot));
                seen.at(*slot) = true;
                CHECK(*slot == expected);
                require_index(Move{square(from_file, 1), square(to_file, 0), promotion},
                              Color::Black, expected);
                ++expected;
            }
        }
    }
    CHECK(expected == 1858);
    for (const bool present : seen) CHECK(present);
}

TEST_CASE("Pinned Lc0 policy anchors distinguish knight promotions and king to rook castles",
          "[nn][policy]") {
    for (const Color color : {Color::White, Color::Black}) {
        const int home = color == Color::White ? 0 : 7;
        const int from_rank = color == Color::White ? 6 : 1;
        const int to_rank = color == Color::White ? 7 : 0;
        require_index(Move{square(0, from_rank), square(0, to_rank), PieceType::Knight}, color,
                      1401);
        require_index(Move{square(0, from_rank), square(0, to_rank), PieceType::Queen}, color,
                      1792);
        require_index(Move{square(0, from_rank), square(0, to_rank), PieceType::Rook}, color, 1793);
        require_index(Move{square(0, from_rank), square(0, to_rank), PieceType::Bishop}, color,
                      1794);
        require_index(Move{square(4, home), square(7, home), PieceType::None, MoveKind::Castle},
                      color, 103);
        require_index(Move{square(4, home), square(0, home), PieceType::None, MoveKind::Castle},
                      color, 97);
        require_index(Move{square(4, home), square(6, home)}, color, 102);
        require_index(Move{square(4, home), square(2, home)}, color, 99);
    }
}

TEST_CASE("Legal policy inverse restores promotion en-passant and Chess960 castling flags",
          "[nn][policy]") {
    const mouse::chess::AttackTables attacks;
    const std::array cases{
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
        "4k3/8/8/8/8/8/8/5K1R w H - 0 1",       "5k1r/8/8/8/8/8/8/4K3 b h - 0 1",
        "4k3/8/8/8/8/8/8/6KR w H - 0 1",        "6kr/8/8/8/8/8/8/4K3 b h - 0 1",
        "1r5k/P7/8/8/8/8/8/7K w - - 0 1",       "7k/8/8/8/8/8/p7/1R5K b - - 0 1",
        "4k3/8/8/4pP2/8/8/8/4K3 w - e6 0 1",    "4k3/8/8/8/4Pp2/8/8/4K3 b - e3 0 1"};
    std::size_t castles = 0;
    std::size_t promotions = 0;
    std::size_t en_passant = 0;
    for (const auto fen : cases) {
        INFO(fen);
        auto position = mouse::chess::Position::from_fen(fen, attacks);
        REQUIRE(position.has_value());
        const auto before = *position;
        std::array<bool, 1858> seen{};
        for (const Move move : mouse::chess::legal_moves(*position, attacks)) {
            const auto slot = PolicyMap::index(move, position->side_to_move());
            REQUIRE(slot.has_value());
            CHECK_FALSE(seen.at(*slot));
            seen.at(*slot) = true;
            const auto decoded = PolicyMap::move(*slot, *position, attacks);
            REQUIRE(decoded.has_value());
            CHECK(*decoded == move);
            CHECK(*position == before);
            castles += move.kind() == MoveKind::Castle ? 1 : 0;
            promotions += move.is_promotion() ? 1 : 0;
            en_passant += move.kind() == MoveKind::EnPassant ? 1 : 0;
        }
    }
    CHECK(castles >= 8);
    CHECK(promotions == 16);
    CHECK(en_passant == 2);
}

TEST_CASE("Policy mapping rejects invalid geometry enum and flag combinations", "[nn][policy]") {
    const std::array invalid{
        Move{square(0, 0), square(0, 0)},
        Move{square(0, 0), square(1, 3)},
        Move{square(0, 6), square(0, 7), PieceType::Pawn},
        Move{square(0, 6), square(0, 7), PieceType::King},
        Move{square(0, 6), square(0, 7), static_cast<PieceType>(7)},
        Move{square(0, 1), square(0, 2), PieceType::Queen},
        Move{square(0, 6), square(2, 7), PieceType::Knight},
        Move{square(4, 1), square(7, 1), PieceType::None, MoveKind::Castle},
        Move{square(0, 6), square(1, 7), PieceType::None, MoveKind::EnPassant},
        Move{square(4, 4), square(5, 5), PieceType::Queen, MoveKind::EnPassant},
        Move{square(4, 0), square(7, 0), PieceType::Knight, MoveKind::Castle},
        Move{square(0, 0), square(1, 0), PieceType::None, static_cast<MoveKind>(3)}};
    for (const Move move : invalid) {
        const auto slot = PolicyMap::index(move, Color::White);
        REQUIRE_FALSE(slot.has_value());
        CHECK_FALSE(slot.error().empty());
    }
    CHECK_FALSE(
        PolicyMap::index(Move{square(0, 0), square(1, 0)}, static_cast<Color>(2)).has_value());
    const mouse::chess::AttackTables attacks;
    auto start = mouse::chess::Position::from_fen(mouse::chess::kStartFen, attacks);
    REQUIRE(start.has_value());
    CHECK_FALSE(PolicyMap::move(1858, *start, attacks).has_value());
    CHECK_FALSE(
        PolicyMap::move(std::numeric_limits<std::uint16_t>::max(), *start, attacks).has_value());
    CHECK_FALSE(PolicyMap::move(1401, *start, attacks).has_value());
    CHECK_FALSE(PolicyMap::move(103, *start, attacks).has_value());
    auto overflow =
        mouse::chess::Position::from_fen("7k/8/8/8/8/8/8/K7 b - - 0 18446744073709551615", attacks);
    REQUIRE(overflow.has_value());
    CHECK_FALSE(PolicyMap::move(0, *overflow, attacks).has_value());
}

TEST_CASE("Every legal move in ten thousand random positions has a unique reversible policy slot",
          "[nn][policy]") {
    const mouse::chess::AttackTables attacks;
    auto parsed = mouse::chess::Position::from_fen(mouse::chess::kStartFen, attacks);
    REQUIRE(parsed.has_value());
    auto position = std::move(*parsed);
    std::mt19937_64 random{0x706f6c696379ULL};
    std::array<std::size_t, 2> colors{};
    std::uint64_t checked_moves = 0;
    std::size_t checked_positions = 0;
    for (; checked_positions < 10000; ++checked_positions) {
        const auto color = position.side_to_move();
        ++colors.at(mouse::chess::index(color));
        const auto before = position;
        const auto moves = mouse::chess::legal_moves(position, attacks);
        std::array<bool, 1858> seen{};
        for (const Move move : moves) {
            const auto slot = PolicyMap::index(move, color);
            if (!slot) FAIL(slot.error());
            REQUIRE(*slot < seen.size());
            CHECK_FALSE(seen.at(*slot));
            seen.at(*slot) = true;
            const auto decoded = PolicyMap::move(*slot, position, attacks);
            if (!decoded) FAIL(decoded.error());
            CHECK(*decoded == move);
            CHECK(position == before);
            ++checked_moves;
        }
        if (mouse::chess::game_status(position, attacks) != mouse::chess::GameStatus::Ongoing) {
            auto fresh = mouse::chess::Position::from_fen(mouse::chess::kStartFen, attacks);
            REQUIRE(fresh.has_value());
            position = std::move(*fresh);
        } else {
            REQUIRE_FALSE(moves.empty());
            const Move chosen =
                moves[std::uniform_int_distribution<std::size_t>{0, moves.size() - 1}(random)];
            (void)position.make_move(chosen, attacks);
        }
    }
    CHECK(checked_positions == 10000);
    CHECK(colors.at(0) > 4000);
    CHECK(colors.at(1) > 4000);
    std::cout << "MOUSE_POLICY_RESULT positions=" << checked_positions << " moves=" << checked_moves
              << " white=" << colors.at(0) << " black=" << colors.at(1) << '\n';
}
