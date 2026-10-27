// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <array>

#include "engine/util/result.h"

namespace mouse::uci {
struct ScoreInput {
    float q;
    float draw;
};
struct UciScore {
    int centipawns;
    std::array<int, 3> wdl;
};
// Default pinned Lc0 WDL_mu reporting for a WDL value in the root player's
// perspective. Inputs remain unchanged; the inverse reporting transform is local.
// Finite Q in [-1,1], D in [0,1], and float32 abs(Q)+D<=1 are required.
// Terminal mate reporting belongs to the caller; Q endpoints here remain cp.
// Stateless and thread-safe, with checked integer conversions on every output.
[[nodiscard]] Result<UciScore> format_score(ScoreInput input);
}  // namespace mouse::uci
