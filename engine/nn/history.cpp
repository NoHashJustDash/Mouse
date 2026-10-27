// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/nn/history.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>

namespace mouse::nn {
Result<void> EncodingHistory::reserve_history(std::size_t additional_plies) {
    if (additional_plies > frames_.max_size() - frames_.size())
        return tl::unexpected(std::string{"History too big"});
    try {
        frames_.reserve(frames_.size() + additional_plies);
        return {};
    } catch (const std::bad_alloc&) {
        return tl::unexpected(std::string{"History OOM"});
    } catch (const std::length_error&) {
        return tl::unexpected(std::string{"History too big"});
    }
}
std::size_t EncodingHistory::history_storage_bytes() const noexcept {
    // Capacity payload only; Search separately charges the owning object.
    // 64 bytes cover MSVC's alignment/allocation bookkeeping, as in NNCache.
    constexpr std::size_t allocation_allowance = 64;
    return frames_.capacity() == 0
               ? 0
               : frames_.capacity() * sizeof(HistoryFrame) + allocation_allowance;
}
void EncodingHistory::reset(const chess::Position& root) {
    const auto frame = HistoryFrame::capture(root);
    // Reserve before changing the existing sequence. HistoryFrame copies cannot
    // throw; reset retains capacity, and append uses vector's amortized growth.
    if (frames_.capacity() == 0) frames_.reserve(1);
    frames_.clear();
    frames_.push_back(frame);
}

bool EncodingHistory::same_identity(const HistoryFrame& left, const HistoryFrame& right) noexcept {
    // Active castling rook squares encode both presence and Chess960 rook file.
    // Inactive rights are kNoSquare. Neither clock nor repetition count belongs
    // to the board identity. Compare exact fields rather than a hash collision.
    return left.pieces_ == right.pieces_ && left.side_ == right.side_ &&
           left.castling_ == right.castling_ && left.en_passant_ == right.en_passant_;
}

chess::Square EncodingHistory::normalized_ep(const HistoryFrame& frame) noexcept {
    if (frame.en_passant_ == chess::kNoSquare) return chess::kNoSquare;
    const int target_file = chess::file_of(frame.en_passant_);
    const int source_rank =
        chess::rank_of(frame.en_passant_) - (frame.side_ == chess::Color::White ? 1 : -1);
    const auto pawns =
        frame.pieces_.at(chess::index(frame.side_)).at(chess::index(chess::PieceType::Pawn));
    for (const int offset : {-1, 1}) {
        const int source_file = target_file + offset;
        if (source_file >= 0 && source_file < 8 && source_rank >= 0 && source_rank < 8 &&
            (pawns & chess::bit(chess::square(source_file, source_rank))) != 0)
            return frame.en_passant_;
    }
    return chess::kNoSquare;
}

Result<void> EncodingHistory::append(const chess::Position& position) {
    if (frames_.empty()) return tl::unexpected("EncodingHistory append requires a root");
    if (position.side_to_move() == frames_.back().side_)
        return tl::unexpected("EncodingHistory append requires alternating side to move");
    auto frame = HistoryFrame::capture(position);
    // Independent implementation of pinned Lc0 v0.32.1 behavior:
    // src/chess/board.cc ApplyMove keeps double-push EP only when an adjacent
    // enemy pawn can geometrically capture; SetFromFen preserves stated EP.
    // src/chess/position.cc ComputeLastMoveRepetitions compares board identity
    // on same-side plies within the reversible window, excluding clocks.
    // https://github.com/LeelaChessZero/lc0/blob/fd71a2d921b689c5f479d3227c3806c8e272d9c5/src/chess/board.cc
    // https://github.com/LeelaChessZero/lc0/blob/fd71a2d921b689c5f479d3227c3806c8e272d9c5/src/chess/position.cc
    frame.en_passant_ = normalized_ep(frame);
    const auto limit = std::min<std::uint64_t>(frames_.size(), frame.halfmove_);
    for (std::uint64_t distance = 4; distance <= limit;) {
        const auto index = frames_.size() - static_cast<std::size_t>(distance);
        if (same_identity(frames_[index], frame)) {
            if (frame.repetitions_ == std::numeric_limits<std::uint32_t>::max())
                return tl::unexpected("EncodingHistory repetition count exceeds uint32 range");
            ++frame.repetitions_;
        }
        if (limit - distance < 2) break;
        distance += 2;
    }
    // vector offers the strong exception guarantee for this nonthrowing frame
    // type, so allocation failure cannot leave a partially appended history.
    frames_.push_back(frame);
    return {};
}

Result<void> EncodingHistory::pop() {
    if (frames_.size() <= 1)
        return tl::unexpected("EncodingHistory pop cannot remove the root or an empty history");
    frames_.pop_back();
    return {};
}
}  // namespace mouse::nn
