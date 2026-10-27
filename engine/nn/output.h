// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <array>
#include <optional>
#include <span>

#include "engine/util/result.h"

namespace mouse::nn {
// Raw network output from the current side's perspective. Policy masking and
// temperature belong to search; caches retain these unmodified values.
struct NNOutput {
    std::array<float, 1858> policy_logits{};
    std::array<float, 3> wdl{};
    std::optional<float> moves_left;
    bool operator==(const NNOutput&) const = default;
};
// Shared public validation for raw backend outputs, cache insertion and the
// cached decorator. WDL need not be approximately normalized here: each value
// must be in [0,1] and the sum must be positive. This never modifies the output.
// Borrowed heads must remain alive and unchanged during this call. Fixed extents
// keep the same head-shape contract without copying values into an NNOutput.
[[nodiscard]] Result<void> validate_nn_output(std::span<const float, 1858> policy_logits,
                                              std::span<const float, 3> wdl,
                                              std::optional<float> moves_left);
[[nodiscard]] Result<void> validate_nn_output(const NNOutput& output);
// Common validation for complete raw or explicitly legal-only head spans.
[[nodiscard]] Result<void> validate_nn_heads(std::span<const float> policy_logits,
                                             std::span<const float, 3> wdl,
                                             std::optional<float> moves_left);
}  // namespace mouse::nn
