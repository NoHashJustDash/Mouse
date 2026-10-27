// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/uci/emergency.h"

#include <array>
#include <cstdlib>
#include <tuple>

#include "engine/chess/movegen.h"
namespace mouse::uci {
EmergencyChoice emergency_choice(const search::RootStatistics& roots,
                                 const EmergencyEvidence& evidence) {
    const search::RootMoveStat* best = nullptr;
    for (std::size_t i = 0; i < roots.size; ++i) {
        const auto& candidate = roots.moves.at(i);
        const auto rank = std::tuple{candidate.visits, candidate.q, candidate.prior};
        if (!best || rank > std::tuple{best->visits, best->q, best->prior} ||
            (rank == std::tuple{best->visits, best->q, best->prior} &&
             chess::move_to_uci(candidate.move, true) < chess::move_to_uci(best->move, true)))
            best = &candidate;
    }
    // Unvisited children have no Q evidence: compare only their evaluated priors.
    if (best && best->visits == 0) {
        for (std::size_t i = 0; i < roots.size; ++i) {
            const auto& candidate = roots.moves.at(i);
            if (candidate.prior > best->prior ||
                (candidate.prior == best->prior &&
                 chess::move_to_uci(candidate.move, true) < chess::move_to_uci(best->move, true)))
                best = &candidate;
        }
    }
    if (best) return {.move = best->move};
    if (evidence.reused) return {.move = evidence.reused};
    return {.move = evidence.deterministic,
            .without_evaluation = evidence.deterministic.has_value()};
}
std::optional<chess::Move> deterministic_emergency(const chess::Position& position,
                                                   const chess::MoveList& legal,
                                                   const chess::AttackTables& attacks) {
    constexpr std::array<int, 7> value{1, 3, 3, 5, 9, 0, 0};
    const auto rank = [&](chess::Move move) {
        const int captured = move.kind() == chess::MoveKind::EnPassant
                                 ? 1
                                 : value.at(chess::index(position.piece_at(move.to()).type));
        const int attacker = value.at(chess::index(position.piece_at(move.from()).type));
        auto next = position;
        static_cast<void>(next.make_move(move, attacks));
        const int center = 14 - std::abs(2 * chess::file_of(move.to()) - 7) -
                           std::abs(2 * chess::rank_of(move.to()) - 7);
        return std::tuple{captured != 0, captured * 16 - attacker * (captured != 0),
                          next.in_check(attacks), value.at(chess::index(move.promotion())), center};
    };
    std::optional<chess::Move> best;
    for (const auto move : legal)
        if (!best || rank(move) > rank(*best) ||
            (rank(move) == rank(*best) &&
             chess::move_to_uci(move, true) < chess::move_to_uci(*best, true)))
            best = move;
    return best;
}
}  // namespace mouse::uci
