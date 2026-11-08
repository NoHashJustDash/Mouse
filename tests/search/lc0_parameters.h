// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "engine/search/draw_rate.h"
#include "engine/search/selection.h"

namespace mouse::test {
// Lc0's search defaults (lc0 src/search/classic/params.cc), for tests whose expected
// values were derived from them. Mouse's own defaults are tuned and differ.
inline constexpr search::SelectionParameters kLc0Selection{
    .cpuct = 1.745,
    .cpuct_base = 38739.0,
    .cpuct_factor = 3.894,
    .fpu_reduction = 0.330,
    .root_fpu = 1.0,
    .root_fpu_strategy = search::RootFpuStrategy::Same,
    .policy_temperature = 1.359,
};
inline constexpr search::DrawRateParameters kLc0DrawRate{
    .target = 0.0, .reference = 0.5, .max_s = 1.4};
}  // namespace mouse::test
