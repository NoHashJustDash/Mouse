// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

#include "engine/chess/types.h"
#include "engine/nn/output.h"
namespace mouse::nn {
struct PolicyIndices {
    std::array<std::uint16_t, chess::MoveList::kCapacity> values{};
    std::size_t size = 0;
};
// Raw float32 logits in the caller's legal-policy order; temperature stays in
// search. Unrequested policy slots are absent, never synthesized as NNOutput.
struct LegalOutput {
    std::array<float, chess::MoveList::kCapacity> logits{};
    std::array<float, 3> wdl{};
    std::optional<float> moves_left;
    bool operator==(const LegalOutput&) const = default;
};
[[nodiscard]] Result<void> validate_policy_indices(const PolicyIndices&);
[[nodiscard]] Result<void> validate_legal_output(const LegalOutput&, std::size_t count);
}  // namespace mouse::nn
