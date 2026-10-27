// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "engine/chess/types.h"
#include "engine/util/result.h"
namespace mouse::chess {
class AttackTables;
struct PositionState {
    Color side = Color::White;
    std::array<std::array<Square, 2>, 2> castling{{{kNoSquare, kNoSquare}, {kNoSquare, kNoSquare}}};
    Square en_passant = kNoSquare;
    Square hashed_en_passant = kNoSquare;
    std::uint64_t halfmove = 0;
    std::uint64_t fullmove = 1;
    std::uint64_t key = 0;
    bool operator==(const PositionState&) const = default;
};
struct Undo {
    PositionState state;
    Piece moving;
    Piece captured;
    Square captured_square = kNoSquare;
    std::size_t history_size = 0;
};
// A Position has one mutating owner. Independent copies and immutable AttackTables
// can be used concurrently. make_move accepts only moves produced by legal_moves
// (or the internally filtered pseudo-legal generator); parse_uci_move validates input.
class Position {
   public:
    [[nodiscard]] static Result<Position> from_fen(std::string_view fen,
                                                   const AttackTables& attacks);
    [[nodiscard]] std::string fen(bool shredder_castling = false) const;
    [[nodiscard]] Color side_to_move() const noexcept { return state_.side; }
    [[nodiscard]] Piece piece_at(Square at) const { return board_.at(at); }
    [[nodiscard]] Bitboard pieces(Color color, PieceType type) const {
        return pieces_.at(index(color)).at(index(type));
    }
    [[nodiscard]] Bitboard occupancy(Color color) const { return occupied_.at(index(color)); }
    [[nodiscard]] Bitboard occupancy() const noexcept { return occupied_[0] | occupied_[1]; }
    [[nodiscard]] Square king_square(Color color) const;
    [[nodiscard]] Square castling_rook(Color color, bool kingside) const {
        return state_.castling.at(index(color)).at(kingside ? 1 : 0);
    }
    [[nodiscard]] Square en_passant_square() const noexcept { return state_.en_passant; }
    [[nodiscard]] std::uint64_t halfmove_clock() const noexcept { return state_.halfmove; }
    [[nodiscard]] std::uint64_t fullmove_number() const noexcept { return state_.fullmove; }
    [[nodiscard]] std::uint64_t key() const noexcept { return state_.key; }
    [[nodiscard]] std::uint64_t recompute_key(const AttackTables& attacks) const;
    [[nodiscard]] bool in_check(const AttackTables& attacks) const;
    [[nodiscard]] bool is_repetition(int occurrences = 3) const;
    [[nodiscard]] bool is_rule50_draw() const noexcept { return state_.halfmove >= 100; }
    [[nodiscard]] bool insufficient_material() const;
    // Preallocate additional plies before worker replay. State is unchanged on failure.
    [[nodiscard]] Result<void> reserve_history(std::size_t additional_plies);
    // Actual vector capacity payload plus one 64-byte allocation allowance; object
    // control is charged separately by Search. Zero for an unallocated vector.
    [[nodiscard]] std::size_t history_storage_bytes() const noexcept;
    [[nodiscard]] std::size_t history_size() const noexcept { return history_.size(); }
    [[nodiscard]] Undo make_move(Move move, const AttackTables& attacks);
    void unmake_move(Move move, const Undo& undo);
    bool operator==(const Position&) const = default;

   private:
    Position() = default;
    void set_piece(Square at, Piece piece);
    [[nodiscard]] Square effective_en_passant(const AttackTables& attacks) const;
    [[nodiscard]] std::uint64_t metadata_key() const;
    std::array<Piece, 64> board_{};
    std::array<std::array<Bitboard, 6>, 2> pieces_{};
    std::array<Bitboard, 2> occupied_{};
    PositionState state_;
    std::vector<std::uint64_t> history_;
};
}  // namespace mouse::chess
