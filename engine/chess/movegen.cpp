// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/chess/movegen.h"

#include <bit>
#include <optional>
#include <stdexcept>

#include "engine/chess/attacks.h"
namespace mouse::chess {
Bitboard attackers_to(const Position& position, Square target, Color by,
                      const AttackTables& attacks, AttackView view) {
    const auto pieces = [&](PieceType type) { return position.pieces(by, type) & ~view.removed; };
    return (attacks.pawn(opposite(by), target) & pieces(PieceType::Pawn)) |
           (attacks.knight(target) & pieces(PieceType::Knight)) |
           (attacks.king(target) & pieces(PieceType::King)) |
           (attacks.bishop(target, view.occupied) &
            (pieces(PieceType::Bishop) | pieces(PieceType::Queen))) |
           (attacks.rook(target, view.occupied) &
            (pieces(PieceType::Rook) | pieces(PieceType::Queen)));
}
Bitboard attackers_to(const Position& position, Square target, Color by,
                      const AttackTables& attacks) {
    return attackers_to(position, target, by, attacks,
                        AttackView{.occupied = position.occupancy()});
}
bool is_attacked(const Position& position, Square target, Color by, const AttackTables& attacks,
                 AttackView view) {
    return attackers_to(position, target, by, attacks, view) != 0;
}
bool is_attacked(const Position& position, Square target, Color by, const AttackTables& attacks) {
    return attackers_to(position, target, by, attacks) != 0;
}
Result<void> validate_uci_castling_mode(const Position& position, bool chess960) {
    if (chess960) return {};
    for (const Color color : {Color::White, Color::Black}) {
        const int home_rank = color == Color::White ? 0 : 7;
        for (const bool kingside : {false, true}) {
            const Square rook = position.castling_rook(color, kingside);
            if (rook != kNoSquare && (position.king_square(color) != square(4, home_rank) ||
                                      rook != square(kingside ? 7 : 0, home_rank)))
                return tl::unexpected(
                    std::string{"Nonstandard castling configuration requires UCI_Chess960=true"});
        }
    }
    return {};
}
namespace {
void emit_pawn(MoveList& result, Square from, Square to, MoveKind kind = MoveKind::Normal) {
    if (rank_of(to) == 0 || rank_of(to) == 7) {
        for (const PieceType promotion :
             {PieceType::Queen, PieceType::Rook, PieceType::Bishop, PieceType::Knight})
            result.push_back({from, to, promotion});
    } else
        result.push_back({from, to, PieceType::None, kind});
}
Bitboard horizontal_path(Square from, Square to) {
    assert(rank_of(from) == rank_of(to));
    Bitboard path = bit(from);
    const int direction = file_of(to) > file_of(from) ? 1 : -1;
    for (int file = file_of(from); file != file_of(to);) {
        file += direction;
        path |= bit(square(file, rank_of(from)));
    }
    return path;
}
bool can_castle(const Position& position, bool kingside, const AttackTables& attacks) {
    const Color us = position.side_to_move();
    const Square king = position.king_square(us);
    const Square rook = position.castling_rook(us, kingside);
    if (rook == kNoSquare || position.in_check(attacks)) return false;
    const int rank = us == Color::White ? 0 : 7;
    const Square king_to = square(kingside ? 6 : 2, rank);
    const Square rook_to = square(kingside ? 5 : 3, rank);
    const Bitboard origins = bit(king) | bit(rook);
    if (((horizontal_path(king, king_to) | horizontal_path(rook, rook_to)) & position.occupancy() &
         ~origins) != 0)
        return false;
    // FIDE Chess960 castling: test the king's transit before moving the rook;
    // the final position is checked after make_move, exposing a pinned rook.
    // Reference behavior: lc0 src/chess/board.cc GeneratePseudolegalMoves/IsLegalMove.
    if (king != king_to) {
        const int step = king_to > king ? 1 : -1;
        for (int at = static_cast<int>(king) + step; at != king_to; at += step) {
            if (is_attacked(position, static_cast<Square>(at), opposite(us), attacks,
                            AttackView{.occupied = position.occupancy() & ~bit(king)}))
                return false;
        }
    }
    return true;
}
MoveList pseudo_legal_moves(const Position& position, const AttackTables& attacks) {
    MoveList result;
    const Color us = position.side_to_move();
    const Color them = opposite(us);
    const Bitboard occupied = position.occupancy();
    const Bitboard capturable = position.occupancy(them) & ~position.pieces(them, PieceType::King);
    const Bitboard allowed = ~position.occupancy(us) & ~position.pieces(them, PieceType::King);
    Bitboard pawns = position.pieces(us, PieceType::Pawn);
    while (pawns != 0) {
        const auto from = static_cast<Square>(std::countr_zero(pawns));
        pawns &= pawns - 1;
        const int forward = us == Color::White ? 8 : -8;
        const auto to = static_cast<Square>(static_cast<int>(from) + forward);
        if ((occupied & bit(to)) == 0) {
            emit_pawn(result, from, to);
            if (rank_of(from) == (us == Color::White ? 1 : 6)) {
                const auto twice = static_cast<Square>(static_cast<int>(to) + forward);
                if ((occupied & bit(twice)) == 0) result.push_back({from, twice});
            }
        }
        Bitboard captures = attacks.pawn(us, from) & capturable;
        while (captures != 0) {
            const auto target = static_cast<Square>(std::countr_zero(captures));
            captures &= captures - 1;
            emit_pawn(result, from, target);
        }
        if (position.en_passant_square() != kNoSquare &&
            (attacks.pawn(us, from) & bit(position.en_passant_square())) != 0)
            result.push_back(
                {from, position.en_passant_square(), PieceType::None, MoveKind::EnPassant});
    }
    for (const PieceType type : {PieceType::Knight, PieceType::Bishop, PieceType::Rook,
                                 PieceType::Queen, PieceType::King}) {
        Bitboard pieces = position.pieces(us, type);
        while (pieces != 0) {
            const auto from = static_cast<Square>(std::countr_zero(pieces));
            pieces &= pieces - 1;
            Bitboard targets = 0;
            switch (type) {
                case PieceType::Knight:
                    targets = attacks.knight(from);
                    break;
                case PieceType::Bishop:
                    targets = attacks.bishop(from, occupied);
                    break;
                case PieceType::Rook:
                    targets = attacks.rook(from, occupied);
                    break;
                case PieceType::Queen:
                    targets = attacks.queen(from, occupied);
                    break;
                case PieceType::King:
                    targets = attacks.king(from);
                    break;
                case PieceType::Pawn:
                case PieceType::None:
                    assert(false);
                    break;
            }
            targets &= allowed;
            while (targets != 0) {
                const auto to = static_cast<Square>(std::countr_zero(targets));
                targets &= targets - 1;
                result.push_back({from, to});
            }
        }
    }
    for (const bool kingside : {false, true})
        if (can_castle(position, kingside, attacks))
            result.push_back({position.king_square(us), position.castling_rook(us, kingside),
                              PieceType::None, MoveKind::Castle});
    return result;
}
}  // namespace
MoveList legal_moves(Position& position, const AttackTables& attacks) {
    MoveList result;
    const Color us = position.side_to_move();
    for (const Move move : pseudo_legal_moves(position, attacks)) {
        const Undo undo = position.make_move(move, attacks);
        const bool legal = !is_attacked(position, position.king_square(us), opposite(us), attacks);
        position.unmake_move(move, undo);
        if (legal) result.push_back(move);
    }
    return result;
}
std::string move_to_uci(Move move, bool chess960) {
    if (move.kind() == MoveKind::Castle && !chess960 &&
        (file_of(move.from()) != 4 || (rank_of(move.from()) != 0 && rank_of(move.from()) != 7) ||
         rank_of(move.to()) != rank_of(move.from()) ||
         (file_of(move.to()) != 0 && file_of(move.to()) != 7)))
        throw std::invalid_argument(
            "Nonstandard castling serialization requires UCI_Chess960=true");
    const Square to =
        move.kind() == MoveKind::Castle && !chess960
            ? square(file_of(move.to()) > file_of(move.from()) ? 6 : 2, rank_of(move.from()))
            : move.to();
    std::string result = square_name(move.from()) + square_name(to);
    if (move.is_promotion()) result += std::string_view{"pnbrqk"}.at(index(move.promotion()));
    return result;
}
Result<Move> parse_uci_move(Position& position, std::string_view text, const AttackTables& attacks,
                            bool chess960) {
    if (text.size() != 4 && text.size() != 5)
        return tl::unexpected(
            std::string{"UCI move requires four square characters and an optional promotion"});
    const auto mode = validate_uci_castling_mode(position, chess960);
    if (!mode) return tl::unexpected(mode.error());
    try {
        std::optional<Move> match;
        for (const Move move : legal_moves(position, attacks)) {
            if (move_to_uci(move, chess960) == text) {
                if (match) return tl::unexpected("Ambiguous UCI move: " + std::string{text});
                match = move;
            }
        }
        if (match) return *match;
        return tl::unexpected("Illegal or malformed UCI move: " + std::string{text});
    } catch (const std::overflow_error& error) {
        return tl::unexpected(std::string{error.what()});
    }
}
GameStatus game_status(Position& position, const AttackTables& attacks) {
    if (legal_moves(position, attacks).empty())
        return position.in_check(attacks) ? GameStatus::Checkmate : GameStatus::Stalemate;
    if (position.is_rule50_draw()) return GameStatus::DrawFiftyMove;
    if (position.is_repetition()) return GameStatus::DrawRepetition;
    if (position.insufficient_material()) return GameStatus::DrawInsufficientMaterial;
    return GameStatus::Ongoing;
}
}  // namespace mouse::chess
