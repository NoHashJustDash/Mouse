// SPDX-License-Identifier: GPL-3.0-or-later
// Mouse's chess types under the names Lc0's syzygy.cc uses, so that file stays close to its
// upstream. Lc0 boards are always seen from the side to move, mirrored vertically when that is
// black; this keeps the same convention.
#pragma once
#include <bit>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/chess/attacks.h"
#include "engine/chess/movegen.h"
#include "engine/chess/position.h"

namespace lczero {
class Exception : public std::runtime_error {
   public:
    using std::runtime_error::runtime_error;
};
class Mutex {
   public:
    class Lock {
       public:
        explicit Lock(Mutex& mutex) : lock_(mutex.mutex_) {}

       private:
        std::lock_guard<std::mutex> lock_;
    };

   private:
    std::mutex mutex_;
};
// Lc0's stderr log line: whatever is streamed into it, then a newline.
class ErrorLine {
   public:
    ~ErrorLine() { std::cerr << text_.str() << '\n'; }
    template <typename T>
    ErrorLine& operator<<(const T& value) {
        text_ << value;
        return *this;
    }

   private:
    std::ostringstream text_;
};
class Square {
   public:
    explicit constexpr Square(int index) : index_(index) {}
    [[nodiscard]] constexpr int as_idx() const noexcept { return index_; }
    [[nodiscard]] constexpr int file() const noexcept { return index_ & 7; }

   private:
    int index_;
};
class BitBoard {
   public:
    constexpr BitBoard() = default;
    explicit constexpr BitBoard(std::uint64_t bits) : bits_(bits) {}
    [[nodiscard]] constexpr BitBoard operator&(BitBoard other) const noexcept {
        return BitBoard{bits_ & other.bits_};
    }
    [[nodiscard]] constexpr BitBoard operator|(BitBoard other) const noexcept {
        return BitBoard{bits_ | other.bits_};
    }
    [[nodiscard]] constexpr bool operator==(const BitBoard&) const noexcept = default;
    [[nodiscard]] int count() const noexcept { return std::popcount(bits_); }
    [[nodiscard]] int count_few() const noexcept { return std::popcount(bits_); }
    [[nodiscard]] constexpr bool get(Square square) const noexcept {
        return ((bits_ >> square.as_idx()) & 1U) != 0;
    }
    class Iterator {
       public:
        explicit constexpr Iterator(std::uint64_t bits) : bits_(bits) {}
        [[nodiscard]] Square operator*() const noexcept { return Square{std::countr_zero(bits_)}; }
        Iterator& operator++() noexcept {
            bits_ &= bits_ - 1;
            return *this;
        }
        [[nodiscard]] constexpr bool operator!=(const Iterator& other) const noexcept {
            return bits_ != other.bits_;
        }

       private:
        std::uint64_t bits_;
    };
    [[nodiscard]] constexpr Iterator begin() const noexcept { return Iterator{bits_}; }
    [[nodiscard]] constexpr Iterator end() const noexcept { return Iterator{0}; }

   private:
    std::uint64_t bits_ = 0;
};
class Move {
   public:
    Move(mouse::chess::Move raw, bool flipped) : raw_(raw), flipped_(flipped) {}
    [[nodiscard]] Square from() const noexcept { return Square{raw_.from() ^ (flipped_ ? 56 : 0)}; }
    [[nodiscard]] Square to() const noexcept { return Square{raw_.to() ^ (flipped_ ? 56 : 0)}; }
    [[nodiscard]] mouse::chess::Move raw() const noexcept { return raw_; }

   private:
    mouse::chess::Move raw_;
    bool flipped_;
};
using MoveList = std::vector<Move>;
class ChessBoard {
   public:
    ChessBoard(const mouse::chess::Position& position, const mouse::chess::AttackTables& attacks)
        : position_(&position), attacks_(&attacks) {}
    [[nodiscard]] bool flipped() const noexcept {
        return position_->side_to_move() == mouse::chess::Color::Black;
    }
    [[nodiscard]] BitBoard ours() const noexcept {
        return view(position_->occupancy(position_->side_to_move()));
    }
    [[nodiscard]] BitBoard theirs() const noexcept {
        return view(position_->occupancy(mouse::chess::opposite(position_->side_to_move())));
    }
    [[nodiscard]] BitBoard pawns() const noexcept { return both(mouse::chess::PieceType::Pawn); }
    [[nodiscard]] BitBoard knights() const noexcept { return both(mouse::chess::PieceType::Knight); }
    [[nodiscard]] BitBoard bishops() const noexcept { return both(mouse::chess::PieceType::Bishop); }
    [[nodiscard]] BitBoard rooks() const noexcept { return both(mouse::chess::PieceType::Rook); }
    [[nodiscard]] BitBoard queens() const noexcept { return both(mouse::chess::PieceType::Queen); }
    [[nodiscard]] BitBoard kings() const noexcept { return both(mouse::chess::PieceType::King); }
    [[nodiscard]] bool IsUnderCheck() const { return position_->in_check(*attacks_); }
    [[nodiscard]] MoveList GenerateLegalMoves() const {
        auto copy = *position_;
        const auto legal = mouse::chess::legal_moves(copy, *attacks_);
        MoveList moves;
        moves.reserve(legal.size());
        for (const auto move : legal) moves.emplace_back(move, flipped());
        return moves;
    }

   private:
    [[nodiscard]] BitBoard view(std::uint64_t bits) const noexcept {
        if (!flipped()) return BitBoard{bits};
        // Vertical mirror: swap the eight rank bytes.
        bits = ((bits >> 8U) & 0x00FF00FF00FF00FFULL) | ((bits & 0x00FF00FF00FF00FFULL) << 8U);
        bits = ((bits >> 16U) & 0x0000FFFF0000FFFFULL) | ((bits & 0x0000FFFF0000FFFFULL) << 16U);
        return BitBoard{(bits >> 32U) | (bits << 32U)};
    }
    [[nodiscard]] BitBoard both(mouse::chess::PieceType type) const {
        return view(position_->pieces(mouse::chess::Color::White, type) |
                    position_->pieces(mouse::chess::Color::Black, type));
    }
    const mouse::chess::Position* position_;
    const mouse::chess::AttackTables* attacks_;
};
class Position {
   public:
    Position(const mouse::chess::Position& position, const mouse::chess::AttackTables& attacks)
        : position_(position), attacks_(&attacks) {}
    Position(const Position& parent, const Move& move)
        : position_(parent.position_), attacks_(parent.attacks_) {
        static_cast<void>(position_.make_move(move.raw(), *attacks_));
    }
    [[nodiscard]] ChessBoard GetBoard() const { return ChessBoard{position_, *attacks_}; }
    [[nodiscard]] int GetRule50Ply() const noexcept {
        return static_cast<int>(position_.halfmove_clock());
    }

   private:
    mouse::chess::Position position_;
    const mouse::chess::AttackTables* attacks_;
};
}  // namespace lczero
#define CERR ::lczero::ErrorLine()
