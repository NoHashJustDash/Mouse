// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "engine/chess/types.h"
namespace mouse::chess::zobrist {
[[nodiscard]] std::uint64_t piece(Color color, PieceType type, Square at);
[[nodiscard]] std::uint64_t side();
[[nodiscard]] std::uint64_t castling(Color color, bool kingside, Square rook);
[[nodiscard]] std::uint64_t en_passant(Square target);
}  // namespace mouse::chess::zobrist
