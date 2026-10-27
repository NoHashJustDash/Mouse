// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/nn/encoder.h"

#include <cmath>
#include <limits>
#include <string>

namespace mouse::nn {
namespace {
constexpr auto kFull = std::numeric_limits<std::uint64_t>::max();
constexpr std::array<std::array<std::uint64_t, 6>, 2> kInitialPieces{
    {{0xff00ULL, 0x42ULL, 0x24ULL, 0x81ULL, 0x08ULL, 0x10ULL},
     {0x00ff000000000000ULL, 0x4200000000000000ULL, 0x2400000000000000ULL, 0x8100000000000000ULL,
      0x0800000000000000ULL, 0x1000000000000000ULL}}};
constexpr std::uint64_t orient(std::uint64_t bits, chess::Color perspective) {
    if (perspective == chess::Color::White) return bits;
    std::uint64_t result = 0;
    for (unsigned rank = 0; rank < 8; ++rank) {
        result |= ((bits >> (rank * 8U)) & 0xffULL) << ((7U - rank) * 8U);
    }
    return result;
}
std::uint64_t history_pieces(const HistoryFrame& frame, chess::Color color, chess::PieceType type,
                             bool reconstruct_push) {
    auto board = frame.pieces(color, type);
    if (reconstruct_push && type == chess::PieceType::Pawn &&
        color == chess::opposite(frame.side_to_move())) {
        const auto ep = static_cast<int>(frame.en_passant_square());
        const int direction = frame.side_to_move() == chess::Color::White ? 8 : -8;
        const auto advanced = static_cast<chess::Square>(ep - direction);
        const auto origin = static_cast<chess::Square>(ep + direction);
        board = (board & ~chess::bit(advanced)) | chess::bit(origin);
    }
    return board;
}
}  // namespace

HistoryFrame HistoryFrame::capture(const chess::Position& position, std::uint32_t repetitions) {
    HistoryFrame frame;
    for (const auto color : {chess::Color::White, chess::Color::Black}) {
        for (std::size_t type = 0; type < 6; ++type) {
            frame.pieces_.at(chess::index(color)).at(type) =
                position.pieces(color, static_cast<chess::PieceType>(type));
        }
        frame.castling_.at(chess::index(color)) = {position.castling_rook(color, false),
                                                   position.castling_rook(color, true)};
    }
    frame.side_ = position.side_to_move();
    frame.en_passant_ = position.en_passant_square();
    frame.halfmove_ = position.halfmove_clock();
    frame.repetitions_ = repetitions;
    return frame;
}
bool HistoryFrame::is_initial_board() const noexcept {
    return side_ == chess::Color::White && pieces_ == kInitialPieces &&
           castling_[0] == std::array<chess::Square, 2>{0, 7} &&
           castling_[1] == std::array<chess::Square, 2>{56, 63} && en_passant_ == chess::kNoSquare;
}

Result<InputPlanes> encode_classical(std::span<const HistoryFrame> history,
                                     std::uint32_t input_format, HistoryFill fill) {
    if (input_format != 1) {
        return tl::unexpected<std::string>("Unsupported neural input format: " +
                                           std::to_string(input_format));
    }
    if (fill != HistoryFill::No && fill != HistoryFill::FenOnly && fill != HistoryFill::Always) {
        return tl::unexpected<std::string>("Unsupported neural history fill mode");
    }
    if (history.empty())
        return tl::unexpected<std::string>("Neural encoding requires at least one history frame");
    for (std::size_t frame = 1; frame < history.size(); ++frame) {
        if (history[frame].side_to_move() == history[frame - 1].side_to_move()) {
            return tl::unexpected<std::string>(
                "Neural history must contain consecutive alternating plies");
        }
    }

    // Behavior reference: pinned Lc0 v0.32.1 neural/encoder.cc, classical format.
    // Original implementation uses absolute-color snapshots and a fixed current
    // perspective, avoiding dependence on its alternating relative-board storage.
    InputPlanes result{};
    const auto& current = history.back();
    const auto perspective = current.side_to_move();
    for (std::size_t step = 0; step < 8; ++step) {
        const bool missing = step >= history.size();
        const auto& frame = missing ? history.front() : history[history.size() - step - 1];
        const bool reconstruct_push = missing && frame.en_passant_square() != chess::kNoSquare;
        if (missing) {
            if (fill == HistoryFill::No && (step - history.size() >= 1 || !reconstruct_push)) {
                break;
            }
            if (fill == HistoryFill::FenOnly && frame.is_initial_board()) break;
        }
        for (std::size_t owner = 0; owner < 2; ++owner) {
            const auto color = owner == 0 ? perspective : chess::opposite(perspective);
            for (std::size_t type = 0; type < 6; ++type) {
                const auto board = history_pieces(frame, color, static_cast<chess::PieceType>(type),
                                                  reconstruct_push);
                result.at(step * 13 + owner * 6 + type).mask = orient(board, perspective);
            }
        }
        if (frame.repetitions() >= 1) result.at(step * 13 + 12).mask = kFull;
    }
    // Raw rule50 is intentionally not divided by 100 for classical input1.
    for (std::size_t owner = 0; owner < 2; ++owner) {
        const auto color = owner == 0 ? perspective : chess::opposite(perspective);
        if (current.castling_rook(color, false) != chess::kNoSquare) {
            result.at(104 + owner * 2).mask = kFull;
        }
        if (current.castling_rook(color, true) != chess::kNoSquare) {
            result.at(105 + owner * 2).mask = kFull;
        }
    }
    if (perspective == chess::Color::Black) result[108].mask = kFull;
    result[109] = {.mask = kFull, .value = static_cast<float>(current.halfmove_clock())};
    result[111].mask = kFull;
    return result;
}
Result<void> expand_planes(const InputPlanes& planes, std::span<float> output) {
    if (output.size() != kInputElementCount) {
        return tl::unexpected<std::string>(
            "Neural dense output must contain exactly 7168 float values");
    }
    for (const auto& plane : planes) {
        if (!std::isfinite(plane.value))
            return tl::unexpected<std::string>("Neural plane contains a non-finite value");
    }
    for (std::size_t plane = 0; plane < planes.size(); ++plane) {
        for (unsigned square = 0; square < 64; ++square) {
            output[plane * 64 + square] =
                (planes.at(plane).mask & (std::uint64_t{1} << square)) != 0 ? planes.at(plane).value
                                                                            : 0.0F;
        }
    }
    return {};
}
}  // namespace mouse::nn
