// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "engine/chess/types.h"

namespace mouse::chess {
// Built once by the owning engine context. Queries are read-only and may run
// concurrently without synchronization; no references to mutable storage escape.
// Squares follow Lc0 src/chess/bitboard.h: a1=0, h8=63.
class AttackTables {
   public:
    // PEXT builds reject unsupported CPUs before executing a BMI2 instruction.
    AttackTables();

    [[nodiscard]] Bitboard pawn(Color color, Square from) const noexcept;
    [[nodiscard]] Bitboard knight(Square from) const noexcept;
    [[nodiscard]] Bitboard king(Square from) const noexcept;
    [[nodiscard]] Bitboard bishop(Square from, Bitboard occupied) const noexcept;
    [[nodiscard]] Bitboard rook(Square from, Bitboard occupied) const noexcept;
    [[nodiscard]] Bitboard queen(Square from, Bitboard occupied) const noexcept;

   private:
    struct Slider {
        Bitboard mask = 0;
        Bitboard multiplier = 0;
        std::size_t offset = 0;
        unsigned shift = 0;
    };
    [[nodiscard]] static std::size_t index(const Slider& slider, Bitboard occupied) noexcept;
    void initialize_slider(Square from, bool diagonal, Slider& slider);

    std::array<std::array<Bitboard, 64>, 2> pawns_{};
    std::array<Bitboard, 64> knights_{};
    std::array<Bitboard, 64> kings_{};
    std::array<Slider, 64> bishops_{};
    std::array<Slider, 64> rooks_{};
    // One allocation, partitioned by immutable offsets; no hot-path allocation.
    std::vector<Bitboard> sliding_;
};
}  // namespace mouse::chess
