// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "engine/chess/position.h"
namespace mouse::chess {
struct AttackView {
    Bitboard occupied = 0;
    Bitboard removed = 0;
};
[[nodiscard]] Bitboard attackers_to(const Position& position, Square target, Color by,
                                    const AttackTables& attacks, AttackView view);
[[nodiscard]] Bitboard attackers_to(const Position& position, Square target, Color by,
                                    const AttackTables& attacks);
[[nodiscard]] Result<void> validate_uci_castling_mode(const Position& position, bool chess960);
[[nodiscard]] bool is_attacked(const Position& position, Square target, Color by,
                               const AttackTables& attacks, AttackView view);
[[nodiscard]] bool is_attacked(const Position& position, Square target, Color by,
                               const AttackTables& attacks);
[[nodiscard]] MoveList legal_moves(Position& position, const AttackTables& attacks);
// The move must be generated for the selected mode. Nonstandard castling with
// chess960=false throws invalid_argument; validate the Position at UCI boundaries.
[[nodiscard]] std::string move_to_uci(Move move, bool chess960 = false);
[[nodiscard]] Result<Move> parse_uci_move(Position& position, std::string_view text,
                                          const AttackTables& attacks, bool chess960 = false);
enum class GameStatus : std::uint8_t {
    Ongoing,
    Checkmate,
    Stalemate,
    DrawFiftyMove,
    DrawRepetition,
    DrawInsufficientMaterial
};
[[nodiscard]] GameStatus game_status(Position& position, const AttackTables& attacks);
}  // namespace mouse::chess
