// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace mouse::chess {
using Bitboard = std::uint64_t;
using Square = std::uint8_t;
inline constexpr Square kNoSquare = 64;
inline constexpr std::string_view kStartFen =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
enum class Color : std::uint8_t { White, Black };
enum class PieceType : std::uint8_t { Pawn, Knight, Bishop, Rook, Queen, King, None };
[[nodiscard]] constexpr Color opposite(Color color) noexcept {
    return color == Color::White ? Color::Black : Color::White;
}
[[nodiscard]] constexpr Square square(int file, int rank) noexcept {
    return static_cast<Square>(rank * 8 + file);
}
[[nodiscard]] constexpr int file_of(Square value) noexcept { return value % 8; }
[[nodiscard]] constexpr int rank_of(Square value) noexcept { return value / 8; }
[[nodiscard]] constexpr Bitboard bit(Square value) noexcept {
    assert(value < 64);
    return Bitboard{1} << value;
}
[[nodiscard]] constexpr std::size_t index(Color value) noexcept {
    return static_cast<std::size_t>(value);
}
[[nodiscard]] constexpr std::size_t index(PieceType value) noexcept {
    return static_cast<std::size_t>(value);
}
struct Piece {
    PieceType type = PieceType::None;
    Color color = Color::White;
    [[nodiscard]] constexpr bool empty() const noexcept { return type == PieceType::None; }
    bool operator==(const Piece&) const = default;
};
enum class MoveKind : std::uint8_t { Normal, EnPassant, Castle };
class Move {
   public:
    constexpr Move() = default;
    constexpr Move(Square from, Square to, PieceType promotion = PieceType::None,
                   MoveKind kind = MoveKind::Normal)
        : value_(static_cast<std::uint32_t>(from) | (static_cast<std::uint32_t>(to) << 6U) |
                 (static_cast<std::uint32_t>(promotion) << 12U) |
                 (static_cast<std::uint32_t>(kind) << 15U)) {}
    [[nodiscard]] constexpr Square from() const noexcept {
        return static_cast<Square>(value_ & 63U);
    }
    [[nodiscard]] constexpr Square to() const noexcept {
        return static_cast<Square>((value_ >> 6U) & 63U);
    }
    [[nodiscard]] constexpr PieceType promotion() const noexcept {
        return static_cast<PieceType>((value_ >> 12U) & 7U);
    }
    [[nodiscard]] constexpr MoveKind kind() const noexcept {
        return static_cast<MoveKind>((value_ >> 15U) & 3U);
    }
    [[nodiscard]] constexpr bool is_promotion() const noexcept {
        return promotion() != PieceType::None;
    }
    bool operator==(const Move&) const = default;

   private:
    std::uint32_t value_ = static_cast<std::uint32_t>(PieceType::None) << 12U;
};
// A side has at most 16 pieces. Each sliding piece has at most 27 targets,
// each knight 8, each pawn 12 (four promotion choices), and each king 10.
// 16*27 is a conservative upper bound even for pseudo-legal positions; this
// capacity deliberately covers that bound rather than relying on the 218 legal maximum.
class MoveList {
   public:
    static constexpr std::size_t kCapacity = 432;
    void push_back(Move move) {
        assert(size_ < kCapacity);
        moves_.at(size_++) = move;
    }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
    [[nodiscard]] Move operator[](std::size_t at) const {
        assert(at < size_);
        return moves_.at(at);
    }
    [[nodiscard]] auto begin() const noexcept {
        return std::span<const Move>{moves_.data(), size_}.begin();
    }
    [[nodiscard]] auto end() const noexcept {
        return std::span<const Move>{moves_.data(), size_}.end();
    }

   private:
    std::array<Move, kCapacity> moves_{};
    std::size_t size_ = 0;
};
[[nodiscard]] std::string square_name(Square value);
}  // namespace mouse::chess
