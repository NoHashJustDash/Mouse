// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/search/rules.h"

#include <algorithm>
#include <exception>
#include <new>
#include <string>

namespace mouse::search {
Result<LeafStatus> classify_leaf(chess::Position& position, const chess::AttackTables& attacks,
                                 std::span<const std::uint64_t> search_ancestor_keys,
                                 LeafContext context) {
    try {
        LeafStatus result;
        result.legal_moves = chess::legal_moves(position, attacks);
        if (result.legal_moves.empty()) {
            if (position.in_check(attacks)) {
                result.reason = TerminalReason::Checkmate;
                result.q = -1.0F;
            } else {
                result.reason = TerminalReason::Stalemate;
                result.draw = 1.0F;
            }
            return result;
        }
        if (context == LeafContext::Root) return result;
        if (context != LeafContext::NonRoot)
            return tl::unexpected(std::string{"Invalid search leaf context"});
        // Mate/stalemate precede all rule draws. FIDE game history uses Position's
        // legal-EP key, independently of NN pseudo-legal-EP repetition semantics.
        if (position.is_rule50_draw())
            result.reason = TerminalReason::FiftyMove;
        else if (position.insufficient_material())
            result.reason = TerminalReason::InsufficientMaterial;
        else if (position.is_repetition(3))
            result.reason = TerminalReason::Threefold;
        else if (std::ranges::find(search_ancestor_keys, position.key()) !=
                 search_ancestor_keys.end()) {
            result.reason = TerminalReason::TreeRepetition;
            result.contextual = true;
        }
        if (result.reason != TerminalReason::None) result.draw = 1.0F;
        return result;
    } catch (const std::bad_alloc&) {
        return tl::unexpected(std::string{"Leaf OOM"});
    } catch (const std::exception& error) {
        return tl::unexpected(std::string{"Cannot classify search leaf: "} + error.what());
    }
}
}  // namespace mouse::search
