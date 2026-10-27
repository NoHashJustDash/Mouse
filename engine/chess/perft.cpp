// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/chess/perft.h"

#include <limits>
#include <stdexcept>

#include "engine/chess/movegen.h"
namespace mouse::chess {
Result<std::uint64_t> detail::checked_node_sum(std::uint64_t left, std::uint64_t right) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left)
        return tl::unexpected(std::string{"Perft node count exceeds uint64 capacity"});
    return left + right;
}
namespace {
class ScopedMove {
   public:
    ScopedMove(Position& position, Move move, const AttackTables& attacks)
        : position_(position), move_(move), undo_(position.make_move(move, attacks)) {}
    ~ScopedMove() { position_.unmake_move(move_, undo_); }
    ScopedMove(const ScopedMove&) = delete;
    ScopedMove& operator=(const ScopedMove&) = delete;
    ScopedMove(ScopedMove&&) = delete;
    ScopedMove& operator=(ScopedMove&&) = delete;

   private:
    Position& position_;
    Move move_;
    Undo undo_;
};
Result<std::uint64_t> count_nodes(Position& position, const AttackTables& attacks, int depth) {
    if (depth == 0) return std::uint64_t{1};
    const auto moves = legal_moves(position, attacks);
    if (depth == 1) return static_cast<std::uint64_t>(moves.size());
    std::uint64_t total = 0;
    for (const Move move : moves) {
        const ScopedMove applied{position, move, attacks};
        const auto child = count_nodes(position, attacks, depth - 1);
        if (!child) return tl::unexpected(child.error());
        const auto sum = detail::checked_node_sum(total, *child);
        if (!sum) return tl::unexpected(sum.error());
        total = *sum;
    }
    return total;
}
}  // namespace
Result<std::uint64_t> perft(Position& position, const AttackTables& attacks, int depth) {
    if (depth < 0 || depth > kMaxPerftDepth)
        return tl::unexpected("Perft depth must be between 0 and " +
                              std::to_string(kMaxPerftDepth));
    // Perft deliberately ignores repetition and move-clock draws: its definition
    // counts legal move trees, including continuations after a claimable draw.
    try {
        return count_nodes(position, attacks, depth);
    } catch (const std::overflow_error& error) {
        return tl::unexpected(std::string{error.what()});
    }
}
Result<std::vector<DivideEntry>> perft_divide(Position& position, const AttackTables& attacks,
                                              int depth) {
    if (depth < 1 || depth > kMaxPerftDepth)
        return tl::unexpected("Perft divide depth must be between 1 and " +
                              std::to_string(kMaxPerftDepth));
    try {
        std::vector<DivideEntry> result;
        const auto moves = legal_moves(position, attacks);
        result.reserve(moves.size());
        std::uint64_t total = 0;
        for (const Move move : moves) {
            const ScopedMove applied{position, move, attacks};
            const auto child = count_nodes(position, attacks, depth - 1);
            if (!child) return tl::unexpected(child.error());
            const auto sum = detail::checked_node_sum(total, *child);
            if (!sum) return tl::unexpected(sum.error());
            total = *sum;
            result.push_back({.move = move, .nodes = *child});
        }
        return result;
    } catch (const std::overflow_error& error) {
        return tl::unexpected(std::string{error.what()});
    }
}
}  // namespace mouse::chess
