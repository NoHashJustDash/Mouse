// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <array>
#include <cstdint>
#include <span>

#include "engine/chess/position.h"
#include "engine/util/result.h"

namespace mouse::nn {
inline constexpr std::size_t kInputPlaneCount = 112;
inline constexpr std::size_t kInputElementCount = kInputPlaneCount * 64;
enum class HistoryFill : std::uint8_t { No, FenOnly, Always };
struct InputPlane {
    std::uint64_t mask = 0;
    float value = 1.0F;
    bool operator==(const InputPlane&) const = default;
};
using InputPlanes = std::array<InputPlane, kInputPlaneCount>;

// A compact immutable snapshot. The caller supplies the Lc0 input repetition
// count, which may differ from FIDE repetition for a pinned en-passant right.
// Chronological snapshots must represent consecutive legal plies. A history
// view and its frames can be shared by readers; mutation belongs to its owner.
class HistoryFrame {
   public:
    [[nodiscard]] static HistoryFrame capture(const chess::Position& position,
                                              std::uint32_t repetitions = 0);
    [[nodiscard]] chess::Color side_to_move() const noexcept { return side_; }
    [[nodiscard]] std::uint64_t pieces(chess::Color color, chess::PieceType type) const {
        return pieces_.at(chess::index(color)).at(chess::index(type));
    }
    [[nodiscard]] chess::Square castling_rook(chess::Color color, bool kingside) const {
        return castling_.at(chess::index(color)).at(kingside ? 1 : 0);
    }
    [[nodiscard]] chess::Square en_passant_square() const noexcept { return en_passant_; }
    [[nodiscard]] std::uint64_t halfmove_clock() const noexcept { return halfmove_; }
    [[nodiscard]] std::uint32_t repetitions() const noexcept { return repetitions_; }
    [[nodiscard]] bool is_initial_board() const noexcept;
    bool operator==(const HistoryFrame&) const = default;

   private:
    friend class EncodingHistory;
    HistoryFrame() = default;
    std::array<std::array<std::uint64_t, 6>, 2> pieces_{};
    std::array<std::array<chess::Square, 2>, 2> castling_{};
    chess::Color side_ = chess::Color::White;
    chess::Square en_passant_ = chess::kNoSquare;
    std::uint64_t halfmove_ = 0;
    std::uint32_t repetitions_ = 0;
};

// Only the actual T1 classical format (1) is supported. History is oldest first.
// Fill policy is explicit rather than inferred from input_format. Failures are
// recoverable; the caller must not fall back to another encoder or history mode.
[[nodiscard]] Result<InputPlanes> encode_classical(std::span<const HistoryFrame> history,
                                                   std::uint32_t input_format = 1,
                                                   HistoryFill fill = HistoryFill::FenOnly);
[[nodiscard]] Result<void> expand_planes(const InputPlanes& planes, std::span<float> output);
}  // namespace mouse::nn
