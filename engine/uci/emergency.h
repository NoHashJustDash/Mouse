// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "engine/search/search.h"
namespace mouse::uci {
struct EmergencyEvidence {
    std::optional<chess::Move> reused, deterministic;
};
struct EmergencyChoice {
    std::optional<chess::Move> move;
    bool without_evaluation = false;
};
// Ranked completed evidence first; final ties use UCI text, never generation order.
[[nodiscard]] EmergencyChoice emergency_choice(const search::RootStatistics&,
                                               const EmergencyEvidence&);
// No neural evidence: MVV-LVA captures, checks, promotions, central destinations,
// and finally lexical UCI order. This is explicitly reported as unevaluated.
[[nodiscard]] std::optional<chess::Move> deterministic_emergency(const chess::Position&,
                                                                 const chess::MoveList&,
                                                                 const chess::AttackTables&);
}  // namespace mouse::uci
