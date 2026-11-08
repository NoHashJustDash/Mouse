// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/search/rules.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <new>
#include <string>

namespace mouse::search {
Result<LeafStatus> classify_leaf(chess::Position& position, const chess::AttackTables& attacks,
                                 std::span<const std::uint64_t> search_ancestor_keys,
                                 LeafContext context, bool two_fold_draws, bool threefold_draws) {
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
        else if (threefold_draws && position.is_repetition(3))
            result.reason = TerminalReason::Threefold;
        else if (two_fold_draws)
            for (std::size_t i = search_ancestor_keys.size(); i > 0; --i) {
                if (search_ancestor_keys[i - 1] != position.key()) continue;
                const auto cycle = search_ancestor_keys.size() - (i - 1);
                if (cycle > std::numeric_limits<std::uint16_t>::max())
                    return tl::unexpected(std::string{"Search repetition cycle out of range"});
                result.reason = TerminalReason::TreeRepetition;
                result.repetition_cycle = static_cast<std::uint16_t>(cycle);
                break;
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
