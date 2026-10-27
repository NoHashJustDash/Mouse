// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/chess/position.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstdlib>
#include <limits>
#include <new>
#include <stdexcept>

#include "engine/chess/attacks.h"
#include "engine/chess/movegen.h"
#include "engine/chess/zobrist.h"
namespace mouse::chess {
Result<void> Position::reserve_history(std::size_t additional_plies) {
    if (additional_plies > history_.max_size() - history_.size())
        return tl::unexpected(std::string{"History too big"});
    try {
        history_.reserve(history_.size() + additional_plies);
        return {};
    } catch (const std::bad_alloc&) {
        return tl::unexpected(std::string{"History OOM"});
    } catch (const std::length_error&) {
        return tl::unexpected(std::string{"History too big"});
    }
}
std::size_t Position::history_storage_bytes() const noexcept {
    // Match the allocator allowance in NNCache: MSVC large blocks request up to
    // 39 extra alignment/bookkeeping bytes; 64 covers that allocation overhead.
    constexpr std::size_t allocation_allowance = 64;
    return history_.capacity() == 0
               ? 0
               : history_.capacity() * sizeof(std::uint64_t) + allocation_allowance;
}
Square Position::king_square(Color color) const {
    const Bitboard kings = pieces(color, PieceType::King);
    assert(std::popcount(kings) == 1);
    return static_cast<Square>(std::countr_zero(kings));
}
void Position::set_piece(Square at, Piece piece) {
    const Piece old = board_.at(at);
    if (!old.empty()) {
        pieces_.at(index(old.color)).at(index(old.type)) &= ~bit(at);
        occupied_.at(index(old.color)) &= ~bit(at);
        state_.key ^= zobrist::piece(old.color, old.type, at);
    }
    board_.at(at) = piece;
    if (!piece.empty()) {
        pieces_.at(index(piece.color)).at(index(piece.type)) |= bit(at);
        occupied_.at(index(piece.color)) |= bit(at);
        state_.key ^= zobrist::piece(piece.color, piece.type, at);
    }
}
std::uint64_t Position::metadata_key() const {
    auto key = state_.side == Color::Black ? zobrist::side() : 0;
    for (const Color color : {Color::White, Color::Black}) {
        for (const bool kingside : {false, true}) {
            const Square rook = castling_rook(color, kingside);
            if (rook != kNoSquare) key ^= zobrist::castling(color, kingside, rook);
        }
    }
    if (state_.hashed_en_passant != kNoSquare) key ^= zobrist::en_passant(state_.hashed_en_passant);
    return key;
}
Square Position::effective_en_passant(const AttackTables& attacks) const {
    // FIDE repetition identity depends on a legal EP capture. Lc0 v0.32.1
    // src/chess/board.cc ApplyMove/SetFromFen instead retains pseudo-legal EP.
    // Preserve raw EP separately for future Lc0 input-encoder parity.
    if (state_.en_passant == kNoSquare) return kNoSquare;
    const Color us = side_to_move();
    const Square target = state_.en_passant;
    const auto captured =
        static_cast<Square>(static_cast<int>(target) + (us == Color::White ? -8 : 8));
    Bitboard candidates = attacks.pawn(opposite(us), target) & pieces(us, PieceType::Pawn);
    while (candidates != 0) {
        const auto from = static_cast<Square>(std::countr_zero(candidates));
        candidates &= candidates - 1;
        const Bitboard occupied = (occupancy() & ~bit(from) & ~bit(captured)) | bit(target);
        if (!is_attacked(*this, king_square(us), opposite(us), attacks,
                         AttackView{.occupied = occupied, .removed = bit(captured)}))
            return target;
    }
    return kNoSquare;
}
std::uint64_t Position::recompute_key(const AttackTables& attacks) const {
    std::uint64_t key = 0;
    for (Square at = 0; at < 64; ++at) {
        const Piece piece = piece_at(at);
        if (!piece.empty()) key ^= zobrist::piece(piece.color, piece.type, at);
    }
    key ^= metadata_key();
    if (state_.hashed_en_passant != kNoSquare) key ^= zobrist::en_passant(state_.hashed_en_passant);
    const Square effective = effective_en_passant(attacks);
    if (effective != kNoSquare) key ^= zobrist::en_passant(effective);
    return key;
}
bool Position::in_check(const AttackTables& attacks) const {
    return is_attacked(*this, king_square(side_to_move()), opposite(side_to_move()), attacks);
}
Undo Position::make_move(Move move, const AttackTables& attacks) {
    const Color us = side_to_move();
    const Piece moving = piece_at(move.from());
    assert(!moving.empty() && moving.color == us);
    const bool capture = move.kind() != MoveKind::Castle &&
                         (move.kind() == MoveKind::EnPassant || !piece_at(move.to()).empty());
    if ((us == Color::Black && state_.fullmove == std::numeric_limits<std::uint64_t>::max()) ||
        (moving.type != PieceType::Pawn && !capture &&
         state_.halfmove == std::numeric_limits<std::uint64_t>::max())) {
        throw std::overflow_error("Chess move counter exceeds uint64 capacity");
    }
    // Grow before touching state: an allocation failure must not leave half a move.
    if (history_.size() == history_.capacity()) history_.reserve(history_.size() + 1024);
    Undo undo{.state = state_,
              .moving = moving,
              .captured = {},
              .captured_square = kNoSquare,
              .history_size = history_.size()};
    state_.key ^= metadata_key();
    state_.en_passant = kNoSquare;
    state_.hashed_en_passant = kNoSquare;
    if (move.kind() == MoveKind::Castle) {
        // Lc0 src/chess/board.cc ApplyMove: castling is king -> original rook.
        // Remove both origins before placing either piece, including Chess960 swaps.
        const bool kingside = file_of(move.to()) > file_of(move.from());
        const int rank = us == Color::White ? 0 : 7;
        set_piece(move.from(), {});
        set_piece(move.to(), {});
        set_piece(square(kingside ? 6 : 2, rank), moving);
        set_piece(square(kingside ? 5 : 3, rank), {.type = PieceType::Rook, .color = us});
    } else {
        undo.captured_square =
            move.kind() == MoveKind::EnPassant
                ? static_cast<Square>(static_cast<int>(move.to()) + (us == Color::White ? -8 : 8))
                : move.to();
        undo.captured = piece_at(undo.captured_square);
        assert(undo.captured.empty() ||
               (undo.captured.color != us && undo.captured.type != PieceType::King));
        set_piece(undo.captured_square, {});
        set_piece(move.from(), {});
        set_piece(move.to(),
                  {.type = move.is_promotion() ? move.promotion() : moving.type, .color = us});
        if (moving.type == PieceType::Pawn &&
            std::abs(static_cast<int>(move.to()) - static_cast<int>(move.from())) == 16)
            state_.en_passant =
                static_cast<Square>((static_cast<int>(move.from()) + move.to()) / 2);
    }
    for (const Color color : {Color::White, Color::Black}) {
        for (const bool kingside : {false, true}) {
            auto& rook = state_.castling.at(index(color)).at(kingside ? 1 : 0);
            if ((moving.type == PieceType::King && color == us) || rook == move.from() ||
                (!undo.captured.empty() && rook == undo.captured_square))
                rook = kNoSquare;
        }
    }
    state_.halfmove =
        (moving.type == PieceType::Pawn || !undo.captured.empty()) ? 0 : state_.halfmove + 1;
    if (us == Color::Black) ++state_.fullmove;
    state_.side = opposite(us);
    state_.hashed_en_passant = effective_en_passant(attacks);
    state_.key ^= metadata_key();
    history_.push_back(state_.key);
    return undo;
}
void Position::unmake_move(Move move, const Undo& undo) {
    assert(history_.size() == undo.history_size + 1);
    if (move.kind() == MoveKind::Castle) {
        const bool kingside = file_of(move.to()) > file_of(move.from());
        const int rank = undo.moving.color == Color::White ? 0 : 7;
        set_piece(square(kingside ? 6 : 2, rank), {});
        set_piece(square(kingside ? 5 : 3, rank), {});
        set_piece(move.from(), undo.moving);
        set_piece(move.to(), {.type = PieceType::Rook, .color = undo.moving.color});
    } else {
        set_piece(move.to(), {});
        set_piece(move.from(), undo.moving);
        if (!undo.captured.empty()) set_piece(undo.captured_square, undo.captured);
    }
    state_ = undo.state;
    history_.resize(undo.history_size);
}
bool Position::is_repetition(int occurrences) const {
    assert(occurrences >= 2);
    int matches = 0;
    const auto available = std::min<std::uint64_t>(state_.halfmove, history_.size() - 1);
    const auto start = history_.size() - 1 - static_cast<std::size_t>(available);
    for (std::size_t at = history_.size(); at-- > start;) {
        if (history_.at(at) == key() && ++matches >= occurrences) return true;
    }
    return false;
}
bool Position::insufficient_material() const {
    const auto both = [this](PieceType type) {
        return pieces(Color::White, type) | pieces(Color::Black, type);
    };
    if ((both(PieceType::Pawn) | both(PieceType::Rook) | both(PieceType::Queen)) != 0) return false;
    const Bitboard minors = both(PieceType::Knight) | both(PieceType::Bishop);
    if (std::popcount(minors) <= 1) return true;
    if (both(PieceType::Knight) != 0) return false;
    // Lc0 src/chess/board.cc HasMatingMaterial: bishops on a single square
    // complex cannot help either king deliver mate, including promoted bishops.
    constexpr Bitboard light = 0x55aa55aa55aa55aaULL;
    return (minors & light) == 0 || (minors & ~light) == 0;
}
}  // namespace mouse::chess
