// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/uci/report.h"

#include <algorithm>
#include <iterator>

#include "engine/uci/emergency.h"
namespace mouse::uci {
void rank_report_roots(search::RootStatistics& roots, std::optional<chess::Move> preferred) {
    std::stable_sort(
        roots.moves.begin(), roots.moves.begin() + static_cast<std::ptrdiff_t>(roots.size),
        [](const auto& a, const auto& b) {
            return search::prefer_final_move(
                {.visits = a.visits, .q = a.q, .prior = a.prior, .child_proof = a.child_proof},
                {.visits = b.visits, .q = b.q, .prior = b.prior, .child_proof = b.child_proof});
        });
    const auto end = roots.moves.begin() + static_cast<std::ptrdiff_t>(roots.size);
    auto selected = std::find_if(roots.moves.begin(), end, [&](const auto& root) {
        return preferred && root.move == *preferred;
    });
    if (selected == end) {
        const auto fallback = emergency_choice(roots, {}).move;
        selected = std::find_if(roots.moves.begin(), end, [&](const auto& root) {
            return fallback && root.move == *fallback;
        });
    }
    if (selected != end) std::rotate(roots.moves.begin(), selected, std::next(selected));
}
}  // namespace mouse::uci
