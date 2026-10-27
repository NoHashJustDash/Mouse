// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cstdint>
#include <span>

#include "engine/chess/movegen.h"
#include "engine/util/result.h"

namespace mouse::search {
enum class TerminalReason : std::uint8_t {
    None,
    Checkmate,
    Stalemate,
    FiftyMove,
    Threefold,
    InsufficientMaterial,
    TreeRepetition
};
enum class LeafContext : std::uint8_t { NonRoot, Root };
struct LeafStatus {
    chess::MoveList legal_moves;
    TerminalReason reason = TerminalReason::None;
    float q = 0.0F;
    float draw = 0.0F;
    bool contextual = false;
};
// The caller exclusively owns position during legal generation. Ancestor keys
// contain only ancestors from the current search root, excluding this leaf;
// pre-root game history must not be supplied as tree-twofold context.
// Checkmate/stalemate precede rule draws. A root with legal moves must return a
// move even when a GUI has not adjudicated a draw: rule draws apply only below
// the root. Context is explicit; an empty ancestor span alone does not imply root.
// Q uses the leaf side-to-move viewpoint.
[[nodiscard]] Result<LeafStatus> classify_leaf(chess::Position& position,
                                               const chess::AttackTables& attacks,
                                               std::span<const std::uint64_t> search_ancestor_keys,
                                               LeafContext context = LeafContext::NonRoot);
}  // namespace mouse::search
