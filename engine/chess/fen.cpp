// SPDX-License-Identifier: GPL-3.0-or-later
#include <algorithm>
#include <bit>
#include <charconv>
#include <limits>
#include <memory>
#include <sstream>
#include <system_error>

#include "engine/chess/attacks.h"
#include "engine/chess/movegen.h"
#include "engine/chess/position.h"
namespace mouse::chess {
namespace {
Result<std::uint64_t> parse_counter(const std::string& text, bool positive) {
    std::uint64_t value = 0;
    const auto parsed = std::from_chars(text.data(), std::to_address(text.end()), value);
    if (parsed.ec != std::errc{} || parsed.ptr != std::to_address(text.end()) ||
        (positive && value == 0))
        return tl::unexpected(std::string{
            "FEN move counters must be unsigned 64-bit integers; fullmove must be positive"});
    return value;
}
Piece decode_piece(char value) {
    const bool black = value >= 'a' && value <= 'z';
    const char upper = black ? static_cast<char>(value - 'a' + 'A') : value;
    const auto at = std::string_view{"PNBRQK"}.find(upper);
    return at == std::string_view::npos ? Piece{}
                                        : Piece{.type = static_cast<PieceType>(at),
                                                .color = black ? Color::Black : Color::White};
}
char encode_piece(Piece piece) {
    const char upper = std::string_view{"PNBRQK"}.at(index(piece.type));
    return piece.color == Color::White ? upper : static_cast<char>(upper - 'A' + 'a');
}
}  // namespace
Result<Position> Position::from_fen(std::string_view fen, const AttackTables& attacks) {
    if (fen.size() > 256)
        return tl::unexpected(std::string{"FEN exceeds the supported 256-byte input limit"});
    std::istringstream stream{std::string{fen}};
    std::array<std::string, 6> fields;
    for (auto& field : fields)
        if (!(stream >> field))
            return tl::unexpected(std::string{"FEN requires exactly six fields"});
    std::string extra;
    if (stream >> extra) return tl::unexpected(std::string{"FEN has extra fields"});
    Position position;
    int rank = 7;
    int file = 0;
    bool previous_digit = false;
    for (const char value : fields.at(0)) {
        if (value == '/') {
            if (file != 8 || rank == 0)
                return tl::unexpected(std::string{"FEN ranks must each contain eight squares"});
            --rank;
            file = 0;
            previous_digit = false;
        } else if (value >= '1' && value <= '8') {
            if (previous_digit)
                return tl::unexpected(
                    std::string{"FEN adjacent empty-square digits are not canonical"});
            file += value - '0';
            if (file > 8)
                return tl::unexpected(std::string{"FEN rank contains more than eight squares"});
            previous_digit = true;
        } else {
            const Piece piece = decode_piece(value);
            if (piece.empty() || file >= 8)
                return tl::unexpected(std::string{"FEN contains an invalid piece or rank"});
            if (piece.type == PieceType::Pawn && (rank == 0 || rank == 7))
                return tl::unexpected(
                    std::string{"FEN pawns cannot occupy the first or eighth rank"});
            position.set_piece(square(file++, rank), piece);
            previous_digit = false;
        }
    }
    if (rank != 0 || file != 8)
        return tl::unexpected(std::string{"FEN piece placement must contain eight complete ranks"});
    for (const Color color : {Color::White, Color::Black}) {
        if (std::popcount(position.pieces(color, PieceType::King)) != 1)
            return tl::unexpected(std::string{"FEN requires exactly one king of each color"});
        const int pawns = std::popcount(position.pieces(color, PieceType::Pawn));
        if (pawns > 8 || std::popcount(position.occupancy(color)) > 16)
            return tl::unexpected(std::string{"FEN exceeds the legal pawn or piece count"});
        int promotions = 0;
        for (const PieceType type : {PieceType::Knight, PieceType::Rook, PieceType::Queen}) {
            const int original = type == PieceType::Queen ? 1 : 2;
            promotions += std::max(0, std::popcount(position.pieces(color, type)) - original);
        }
        constexpr Bitboard light_squares = 0x55aa55aa55aa55aaULL;
        const Bitboard bishops = position.pieces(color, PieceType::Bishop);
        promotions += std::max(0, std::popcount(bishops & light_squares) - 1);
        promotions += std::max(0, std::popcount(bishops & ~light_squares) - 1);
        if (promotions > 8 - pawns)
            return tl::unexpected(std::string{"FEN has more promoted pieces than missing pawns"});
    }
    if (fields.at(1) != "w" && fields.at(1) != "b")
        return tl::unexpected(std::string{"FEN active color must be w or b"});
    position.state_.side = fields.at(1) == "w" ? Color::White : Color::Black;
    if (fields.at(2) != "-") {
        for (const char token : fields.at(2)) {
            const bool black = token >= 'a' && token <= 'z';
            const Color color = black ? Color::Black : Color::White;
            const char upper = black ? static_cast<char>(token - 'a' + 'A') : token;
            const Square king = position.king_square(color);
            const int home_rank = black ? 7 : 0;
            if (rank_of(king) != home_rank)
                return tl::unexpected(std::string{"FEN castling king must be on its home rank"});
            Square rook = kNoSquare;
            if (upper == 'K' || upper == 'Q') {
                const bool kingside = upper == 'K';
                for (int candidate = kingside ? 7 : 0; candidate >= 0 && candidate < 8;
                     candidate += kingside ? -1 : 1) {
                    if (candidate == file_of(king)) break;
                    if (position.piece_at(square(candidate, home_rank)) ==
                        Piece{.type = PieceType::Rook, .color = color}) {
                        rook = square(candidate, home_rank);
                        break;
                    }
                }
            } else if (upper >= 'A' && upper <= 'H')
                rook = square(upper - 'A', home_rank);
            else
                return tl::unexpected(std::string{"FEN contains an invalid castling symbol"});
            if (rook == kNoSquare || rook == king ||
                position.piece_at(rook) != Piece{.type = PieceType::Rook, .color = color})
                return tl::unexpected(
                    std::string{"FEN castling right requires the corresponding rook"});
            const bool kingside = file_of(rook) > file_of(king);
            auto& right = position.state_.castling.at(index(color)).at(kingside ? 1 : 0);
            if (right != kNoSquare)
                return tl::unexpected(std::string{"FEN has duplicate castling rights on one side"});
            right = rook;
        }
    }
    const auto halfmove = parse_counter(fields.at(4), false);
    const auto fullmove = parse_counter(fields.at(5), true);
    if (!halfmove) return tl::unexpected(halfmove.error());
    if (!fullmove) return tl::unexpected(fullmove.error());
    position.state_.halfmove = *halfmove;
    position.state_.fullmove = *fullmove;
    if (fields.at(3) != "-") {
        const auto& target = fields.at(3);
        if (target.size() != 2 || target.at(0) < 'a' || target.at(0) > 'h' ||
            target.at(1) != (position.side_to_move() == Color::White ? '6' : '3'))
            return tl::unexpected(std::string{"FEN has an invalid en-passant target"});
        const Square ep = square(target.at(0) - 'a', target.at(1) - '1');
        const int backward = position.side_to_move() == Color::White ? -8 : 8;
        const auto captured = static_cast<Square>(static_cast<int>(ep) + backward);
        const auto origin = static_cast<Square>(static_cast<int>(ep) - backward);
        if (!position.piece_at(ep).empty() || !position.piece_at(origin).empty() ||
            position.piece_at(captured) !=
                Piece{.type = PieceType::Pawn, .color = opposite(position.side_to_move())} ||
            *halfmove != 0)
            return tl::unexpected(std::string{
                "FEN en-passant target is inconsistent with the last double pawn push"});
        // A double push can create a check, but the opposing king cannot already
        // have been in check before that push: its preceding move had to be legal.
        // Reconstruct both occupancy and the pawn's original attacks, without
        // imposing any assumptions about analysis FEN fullmove counters.
        const Color moving = position.side_to_move();
        const Color pushed = opposite(moving);
        const Square king = position.king_square(moving);
        const Bitboard previous_occupancy = (position.occupancy() & ~bit(captured)) | bit(origin);
        const bool previous_check =
            is_attacked(position, king, pushed, attacks,
                        AttackView{.occupied = previous_occupancy, .removed = bit(captured)}) ||
            (attacks.pawn(pushed, origin) & bit(king)) != 0;
        if (previous_check)
            return tl::unexpected(
                std::string{"FEN en-passant push cannot follow a position with the moving king "
                            "already in check"});
        position.state_.en_passant = ep;
    }
    if (is_attacked(position, position.king_square(opposite(position.side_to_move())),
                    position.side_to_move(), attacks))
        return tl::unexpected(std::string{"FEN leaves the non-moving king in check"});
    // A legal move can give at most double check; count individual attackers,
    // not attacking piece types (two bishops plus a rook must count as three).
    if (std::popcount(attackers_to(position, position.king_square(position.side_to_move()),
                                   opposite(position.side_to_move()), attacks)) > 2)
        return tl::unexpected(std::string{"FEN has more than two pieces checking the moving king"});
    position.state_.hashed_en_passant = position.effective_en_passant(attacks);
    position.state_.key = position.recompute_key(attacks);
    position.history_.reserve(1024);
    position.history_.push_back(position.key());
    return position;
}
std::string Position::fen(bool shredder_castling) const {
    std::string result;
    for (int rank = 7; rank >= 0; --rank) {
        int empty = 0;
        for (int file = 0; file < 8; ++file) {
            const Piece piece = piece_at(square(file, rank));
            if (piece.empty()) {
                ++empty;
                continue;
            }
            if (empty != 0) {
                result += static_cast<char>('0' + empty);
                empty = 0;
            }
            result += encode_piece(piece);
        }
        if (empty != 0) result += static_cast<char>('0' + empty);
        if (rank != 0) result += '/';
    }
    result += side_to_move() == Color::White ? " w " : " b ";
    std::string rights;
    for (const Color color : {Color::White, Color::Black}) {
        for (const bool kingside : {true, false}) {
            const Square rook = castling_rook(color, kingside);
            if (rook == kNoSquare) continue;
            const bool standard =
                file_of(king_square(color)) == 4 && file_of(rook) == (kingside ? 7 : 0);
            char token = !shredder_castling && standard ? (kingside ? 'K' : 'Q')
                                                        : static_cast<char>('A' + file_of(rook));
            if (color == Color::Black) token = static_cast<char>(token - 'A' + 'a');
            rights += token;
        }
    }
    result += rights.empty() ? "-" : rights;
    result += ' ';
    result += en_passant_square() == kNoSquare ? "-" : square_name(en_passant_square());
    result += ' ' + std::to_string(halfmove_clock()) + ' ' + std::to_string(fullmove_number());
    return result;
}
}  // namespace mouse::chess
