// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/nn/output.h"

#include <algorithm>
#include <cmath>
#include <string>

namespace mouse::nn {
Result<void> validate_nn_output(const NNOutput& output) {
    return validate_nn_output(output.policy_logits, output.wdl, output.moves_left);
}
Result<void> validate_nn_output(std::span<const float, 1858> policy_logits,
                                std::span<const float, 3> wdl_values,
                                std::optional<float> moves_left) {
    return validate_nn_heads(policy_logits, wdl_values, moves_left);
}
Result<void> validate_nn_heads(std::span<const float> policy_logits,
                               std::span<const float, 3> wdl_values,
                               std::optional<float> moves_left) {
    const bool policy =
        std::ranges::all_of(policy_logits, [](float value) { return std::isfinite(value); });
    const bool wdl = std::ranges::all_of(wdl_values, [](float value) {
        return std::isfinite(value) && value >= 0.0F && value <= 1.0F;
    });
    const bool positive = std::ranges::any_of(wdl_values, [](float value) { return value > 0.0F; });
    if (!policy || !wdl || !positive ||
        (moves_left && (!std::isfinite(*moves_left) || *moves_left < 0.0F))) {
        return tl::unexpected(
            std::string{"NN output requires finite policy, WDL and MLH values; "
                        "WDL must be in [0,1] with positive sum and MLH nonnegative"});
    }
    return {};
}
}  // namespace mouse::nn
