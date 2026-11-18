// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/chess/movegen.h"

#include <bit>
#include <cstdint>
#include <limits>
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
// `accept(from, to, kind)` decides whether a generated move is kept. Promotion choices
// share one origin and destination, so one decision covers all four.
template <typename Accept>
void emit_pawn(MoveList& result, Square from, Square to, const Accept& accept) {
    if (!accept(from, to, MoveKind::Normal)) return;
    if (rank_of(to) == 0 || rank_of(to) == 7) {
        for (const PieceType promotion :
             {PieceType::Queen, PieceType::Rook, PieceType::Bishop, PieceType::Knight})
            result.push_back({from, to, promotion});
    } else
        result.push_back({from, to});
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
// Pseudo-legal non-castling moves in the fixed generation order: pawns (push, double push,
// captures, en passant), then knights, bishops, rooks, queens and the king. Search and policy
// mapping consume moves in this order, so `accept` may only drop moves, never reorder them.
template <typename Accept>
void generate_moves(const Position& position, const AttackTables& attacks, MoveList& result,
                    const Accept& accept) {
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
            emit_pawn(result, from, to, accept);
            if (rank_of(from) == (us == Color::White ? 1 : 6)) {
                const auto twice = static_cast<Square>(static_cast<int>(to) + forward);
                if ((occupied & bit(twice)) == 0 && accept(from, twice, MoveKind::Normal))
                    result.push_back({from, twice});
            }
        }
        Bitboard captures = attacks.pawn(us, from) & capturable;
        while (captures != 0) {
            const auto target = static_cast<Square>(std::countr_zero(captures));
            captures &= captures - 1;
            emit_pawn(result, from, target, accept);
        }
        if (position.en_passant_square() != kNoSquare &&
            (attacks.pawn(us, from) & bit(position.en_passant_square())) != 0 &&
            accept(from, position.en_passant_square(), MoveKind::EnPassant))
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
                if (accept(from, to, MoveKind::Normal)) result.push_back({from, to});
            }
        }
    }
}
Move castle_move(const Position& position, bool kingside) {
    const Color us = position.side_to_move();
    return {position.king_square(us), position.castling_rook(us, kingside), PieceType::None,
            MoveKind::Castle};
}
// Castling moves follow every other move, queenside first.
MoveList pseudo_legal_moves(const Position& position, const AttackTables& attacks) {
    MoveList result;
    generate_moves(position, attacks, result, [](Square, Square, MoveKind) { return true; });
    for (const bool kingside : {false, true})
        if (can_castle(position, kingside, attacks))
            result.push_back(castle_move(position, kingside));
    return result;
}
// The complete test: play the move and look at the mover's king.
bool keeps_king_safe(Position& position, Move move, const AttackTables& attacks) {
    const Color us = position.side_to_move();
    const Undo undo = position.make_move(move, attacks);
    const bool legal = !is_attacked(position, position.king_square(us), opposite(us), attacks);
    position.unmake_move(move, undo);
    return legal;
}
// make_move throws std::overflow_error for a move that would overflow a move counter
// (position.cpp). Callers (perft, parse_uci_move, search classify_leaf) report that error,
// so these positions keep the complete per-move test, which raises it exactly as before.
bool at_counter_limit(const Position& position) noexcept {
    constexpr auto kMaximum = std::numeric_limits<std::uint64_t>::max();
    return position.halfmove_clock() == kMaximum ||
           (position.side_to_move() == Color::Black && position.fullmove_number() == kMaximum);
}
// Squares strictly between two squares that share a rank, file or diagonal.
Bitboard squares_between(Square a, Square b, const AttackTables& attacks) {
    if (file_of(a) == file_of(b) || rank_of(a) == rank_of(b))
        return attacks.rook(a, bit(b)) & attacks.rook(b, bit(a));
    return attacks.bishop(a, bit(b)) & attacks.bishop(b, bit(a));
}
// Every square of the whole line through two squares that share a rank, file or
// diagonal, except those two squares.
Bitboard line_through(Square a, Square b, const AttackTables& attacks) {
    if (file_of(a) == file_of(b) || rank_of(a) == rank_of(b))
        return attacks.rook(a, Bitboard{0}) & attacks.rook(b, Bitboard{0});
    return attacks.bishop(a, Bitboard{0}) & attacks.bishop(b, Bitboard{0});
}
// What makes a move legal for the side to move, computed once per position.
struct Legality {
    Square king = kNoSquare;
    Bitboard occupied_without_king = 0;
    // Destinations of non-king moves that leave no check: everything when not in check,
    // otherwise the checker and the squares between it and the king.
    Bitboard check_mask = ~Bitboard{0};
    bool double_check = false;
    // Our pieces that are the only piece between our king and an enemy slider.
    Bitboard pinned = 0;
};
Legality legality_of(const Position& position, const AttackTables& attacks) {
    const Color us = position.side_to_move();
    const Color them = opposite(us);
    Legality result;
    result.king = position.king_square(us);
    const Bitboard occupied = position.occupancy();
    result.occupied_without_king = occupied & ~bit(result.king);
    const Bitboard queens = position.pieces(them, PieceType::Queen);
    const Bitboard orthogonal = position.pieces(them, PieceType::Rook) | queens;
    const Bitboard diagonal = position.pieces(them, PieceType::Bishop) | queens;
    const Bitboard checkers = attackers_to(position, result.king, them, attacks);
    if (checkers != 0) {
        result.double_check = (checkers & (checkers - 1)) != 0;
        result.check_mask = checkers;
        // A sliding checker can also be blocked; a pawn or knight only captured.
        if (!result.double_check && (checkers & (orthogonal | diagonal)) != 0)
            result.check_mask |= squares_between(
                result.king, static_cast<Square>(std::countr_zero(checkers)), attacks);
    }
    Bitboard snipers = (attacks.rook(result.king, Bitboard{0}) & orthogonal) |
                       (attacks.bishop(result.king, Bitboard{0}) & diagonal);
    const Bitboard ours = position.occupancy(us);
    while (snipers != 0) {
        const auto sniper = static_cast<Square>(std::countr_zero(snipers));
        snipers &= snipers - 1;
        const Bitboard blockers = squares_between(result.king, sniper, attacks) & occupied;
        if (blockers != 0 && (blockers & (blockers - 1)) == 0 && (blockers & ours) != 0)
            result.pinned |= blockers;
    }
    return result;
}
// The en passant test of Position::effective_en_passant: the king stays put, the capturing
// pawn and the captured pawn leave their squares, and the capturing pawn lands on the target.
// This covers a pawn pinned on a diagonal, both pawns leaving one rank, and capturing a
// checking pawn.
bool en_passant_keeps_king_safe(const Position& position, Square from, Square to,
                                const AttackTables& attacks) {
    const Color us = position.side_to_move();
    const auto captured = static_cast<Square>(static_cast<int>(to) + (us == Color::White ? -8 : 8));
    const Bitboard occupied = (position.occupancy() & ~bit(from) & ~bit(captured)) | bit(to);
    return !is_attacked(position, position.king_square(us), opposite(us), attacks,
                        AttackView{.occupied = occupied, .removed = bit(captured)});
}
}  // namespace
// Pin- and check-aware legality: checkers and pinned pieces are found once, so ordinary
// moves need no make/unmake. Castling keeps the complete test. The result is the
// pseudo-legal list in its original order with exactly the illegal moves removed.
MoveList legal_moves(Position& position, const AttackTables& attacks) {
    MoveList result;
    if (at_counter_limit(position)) {
        for (const Move move : pseudo_legal_moves(position, attacks))
            if (keeps_king_safe(position, move, attacks)) result.push_back(move);
        return result;
    }
    const Legality legality = legality_of(position, attacks);
    const Color them = opposite(position.side_to_move());
    const auto accept = [&](Square from, Square to, MoveKind kind) {
        if (kind == MoveKind::EnPassant)
            return en_passant_keeps_king_safe(position, from, to, attacks);
        // The king must not be attacked on its destination. It is removed from the
        // occupancy so that it cannot shield that square from a slider checking it.
        if (from == legality.king)
            return !is_attacked(position, to, them, attacks,
                                AttackView{.occupied = legality.occupied_without_king});
        if (legality.double_check || (legality.check_mask & bit(to)) == 0) return false;
        // A pinned piece stays on the line through its king and the pinning slider.
        return (legality.pinned & bit(from)) == 0 ||
               (line_through(legality.king, from, attacks) & bit(to)) != 0;
    };
    generate_moves(position, attacks, result, accept);
    for (const bool kingside : {false, true}) {
        if (!can_castle(position, kingside, attacks)) continue;
        const Move castle = castle_move(position, kingside);
        if (keeps_king_safe(position, castle, attacks)) result.push_back(castle);
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
