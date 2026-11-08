// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cstdint>

#include "engine/search/proof.h"
#include "engine/util/result.h"

namespace mouse::search {
enum class RootFpuStrategy : std::uint8_t { Same, Reduction, Absolute };
// Child proof uses the opponent's viewpoint; q already uses the parent.
// Inputs come from validated tree statistics. Neural +/-1 is not a proof.
struct FinalMoveStatistics {
    std::uint64_t visits = 0;
    double q = 0;
    float prior = 0;
    NodeProof child_proof{};
};
[[nodiscard]] bool prefer_final_move(const FinalMoveStatistics& first,
                                     const FinalMoveStatistics& second) noexcept;
// cpuct, fpu_reduction and policy_temperature are Mouse's tuned defaults (SPSA on M1.5,
// SPRT +45.7 +/- 15.5 Elo against Lc0's values at 10+0.1, docs/LC0_DEFAULTS.md). Lc0's
// defaults are 1.745, 0.330 and 1.359; lc0 parity tests set them explicitly.
struct SelectionParameters {
    double cpuct = 1.26;
    double cpuct_base = 38739.0;
    double cpuct_factor = 3.894;
    double fpu_reduction = 0.323;
    // Lc0 FpuStrategy: "absolute" uses fpu_reduction as the unvisited value itself (-100..100).
    bool fpu_absolute = false;
    double root_fpu = 1.0;
    RootFpuStrategy root_fpu_strategy = RootFpuStrategy::Same;
    double policy_temperature = 1.352;
};
[[nodiscard]] Result<void> validate_selection_parameters(const SelectionParameters& parameters);
// Internal numerical helpers require previously validated immutable parameters.
// FPU additionally requires finite parent_q in [-1,1] and finite visited mass >=0.
// Mass is not clamped to 1: rounded half priors can sum slightly above 1.
// All helpers are stateless and safe to call concurrently.
[[nodiscard]] double cpuct_value(const SelectionParameters& parameters,
                                 std::uint64_t parent_visits);
[[nodiscard]] double exploration_multiplier(const SelectionParameters& parameters,
                                            std::uint64_t parent_visits);
[[nodiscard]] double first_play_urgency(const SelectionParameters& parameters, bool root,
                                        double parent_q, double visited_prior_mass);
}  // namespace mouse::search
