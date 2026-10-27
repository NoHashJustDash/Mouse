// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <vector>

#include "engine/chess/position.h"
namespace mouse::chess {
inline constexpr int kMaxPerftDepth = 128;
struct DivideEntry {
    Move move;
    std::uint64_t nodes = 0;
};
[[nodiscard]] Result<std::uint64_t> perft(Position& position, const AttackTables& attacks,
                                          int depth);
[[nodiscard]] Result<std::vector<DivideEntry>> perft_divide(Position& position,
                                                            const AttackTables& attacks, int depth);
namespace detail {
[[nodiscard]] Result<std::uint64_t> checked_node_sum(std::uint64_t left, std::uint64_t right);
}
}  // namespace mouse::chess
