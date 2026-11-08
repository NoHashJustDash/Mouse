// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/nn/output.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
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
    // NaN and infinities are exactly the floats whose exponent bits are all ones; adding
    // one exponent step then carries into bit 31. A sum of those carries is a reduction
    // MSVC vectorises with baseline SSE2 (it does not vectorise OR reductions); this runs
    // over 1858 logits per position after every provider call.
    // (MSVC reports reason 1105 for a range-for with bit_cast; this index form vectorises.)
    std::uint32_t non_finite = 0;
    const float* const logits = policy_logits.data();
    for (std::size_t i = 0; i < policy_logits.size(); ++i) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, logits + i, sizeof bits);
        non_finite += ((bits & 0x7f800000U) + 0x00800000U) >> 31U;
    }
    const bool policy = non_finite == 0;
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
