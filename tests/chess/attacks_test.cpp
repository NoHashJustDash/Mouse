// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/chess/attacks.h"

#include <array>
#include <bit>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <random>

namespace mouse::chess {
namespace {
struct Direction {
    int file;
    int rank;
};
constexpr std::array<Direction, 4> kOrthogonal{{{.file = 1, .rank = 0},
                                                {.file = -1, .rank = 0},
                                                {.file = 0, .rank = 1},
                                                {.file = 0, .rank = -1}}};
constexpr std::array<Direction, 4> kDiagonal{{{.file = 1, .rank = 1},
                                              {.file = 1, .rank = -1},
                                              {.file = -1, .rank = 1},
                                              {.file = -1, .rank = -1}}};

bool on_board(int file, int rank) { return file >= 0 && file < 8 && rank >= 0 && rank < 8; }

// Oracle deliberately walks coordinates instead of using the indexed production tables.
Bitboard reference_ray(Square from, const std::array<Direction, 4>& directions, Bitboard occupied) {
    Bitboard result = 0;
    for (const auto& direction : directions) {
        auto file = static_cast<int>(file_of(from)) + direction.file;
        auto rank = static_cast<int>(rank_of(from)) + direction.rank;
        while (on_board(file, rank)) {
            const Bitboard destination = Bitboard{1} << (8 * rank + file);
            result |= destination;
            if ((occupied & destination) != 0) {
                break;
            }
            file += direction.file;
            rank += direction.rank;
        }
    }
    return result;
}

Bitboard relevant_mask(Square from, const std::array<Direction, 4>& directions) {
    Bitboard result = 0;
    for (const auto& direction : directions) {
        auto file = static_cast<int>(file_of(from)) + direction.file;
        auto rank = static_cast<int>(rank_of(from)) + direction.rank;
        while (on_board(file + direction.file, rank + direction.rank)) {
            result |= Bitboard{1} << (8 * rank + file);
            file += direction.file;
            rank += direction.rank;
        }
    }
    return result;
}

int difference(int left, int right) { return left > right ? left - right : right - left; }

TEST_CASE("Leaper attacks stay on board and have the required coordinate displacement",
          "[chess][attacks]") {
    const AttackTables attacks;
    for (int origin = 0; origin < 64; ++origin) {
        const auto from = static_cast<Square>(origin);
        Bitboard expected_knight = 0;
        Bitboard expected_king = 0;
        std::array<Bitboard, 2> expected_pawn{};
        for (int destination = 0; destination < 64; ++destination) {
            const auto to = static_cast<Square>(destination);
            const auto files = difference(origin % 8, destination % 8);
            const auto ranks = difference(origin / 8, destination / 8);
            if ((files == 1 && ranks == 2) || (files == 2 && ranks == 1)) {
                expected_knight |= bit(to);
            }
            if (files <= 1 && ranks <= 1 && origin != destination) {
                expected_king |= bit(to);
            }
            if (files == 1 && destination / 8 == origin / 8 + 1) {
                expected_pawn[0] |= bit(to);
            }
            if (files == 1 && destination / 8 == origin / 8 - 1) {
                expected_pawn[1] |= bit(to);
            }
        }
        CAPTURE(origin);
        CHECK(attacks.knight(from) == expected_knight);
        CHECK(attacks.king(from) == expected_king);
        CHECK(attacks.pawn(Color::White, from) == expected_pawn[0]);
        CHECK(attacks.pawn(Color::Black, from) == expected_pawn[1]);
    }
}

TEST_CASE("Sliding attacks match the ray oracle for every relevant occupancy",
          "[chess][attacks][exhaustive]") {
    const AttackTables attacks;
    std::size_t configurations = 0;
    for (int origin = 0; origin < 64; ++origin) {
        const auto from = static_cast<Square>(origin);
        for (const bool diagonal : {false, true}) {
            const auto& directions = diagonal ? kDiagonal : kOrthogonal;
            const auto mask = relevant_mask(from, directions);
            Bitboard occupied = 0;
            const auto count = std::size_t{1} << std::popcount(mask);
            for (std::size_t entry = 0; entry < count; ++entry) {
                CAPTURE(origin, diagonal, occupied);
                const auto actual =
                    diagonal ? attacks.bishop(from, occupied) : attacks.rook(from, occupied);
                const auto expected = reference_ray(from, directions, occupied);
                CHECK(actual == expected);
                // Every irrelevant bit, including the origin and edge endpoints, may be occupied.
                const auto with_irrelevant = occupied | ~mask;
                CHECK((diagonal ? attacks.bishop(from, with_irrelevant)
                                : attacks.rook(from, with_irrelevant)) ==
                      reference_ray(from, directions, with_irrelevant));
                ++configurations;
                occupied = (occupied - mask) & mask;
            }
        }
    }
    CHECK(configurations == 107648);
}

TEST_CASE("Every first blocker is included and blocks every square beyond it", "[chess][attacks]") {
    const AttackTables attacks;
    for (int origin = 0; origin < 64; ++origin) {
        const auto from = static_cast<Square>(origin);
        for (int blocker = 0; blocker < 64; ++blocker) {
            const Bitboard occupied = Bitboard{1} << blocker;
            CAPTURE(origin, blocker);
            CHECK(attacks.rook(from, occupied) == reference_ray(from, kOrthogonal, occupied));
            CHECK(attacks.bishop(from, occupied) == reference_ray(from, kDiagonal, occupied));
        }
        CHECK(attacks.rook(from, ~Bitboard{0}) == reference_ray(from, kOrthogonal, ~Bitboard{0}));
        CHECK(attacks.bishop(from, ~Bitboard{0}) == reference_ray(from, kDiagonal, ~Bitboard{0}));
    }
}

TEST_CASE("Queen attacks and random full-board occupancies match independent rays",
          "[chess][attacks]") {
    const AttackTables attacks;
    std::mt19937_64 random{0x4d6f757365415454ULL};
    for (int sample = 0; sample < 100000; ++sample) {
        const auto from = static_cast<Square>(random() & 63U);
        const Bitboard occupied = random();
        const auto expected_rook = reference_ray(from, kOrthogonal, occupied);
        const auto expected_bishop = reference_ray(from, kDiagonal, occupied);
        CAPTURE(sample, from, occupied);
        CHECK(attacks.rook(from, occupied) == expected_rook);
        CHECK(attacks.bishop(from, occupied) == expected_bishop);
        CHECK(attacks.queen(from, occupied) == (expected_rook | expected_bishop));
    }
}
}  // namespace
}  // namespace mouse::chess
