// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/nn/legal_output.h"

#include <bitset>
namespace mouse::nn {
Result<void> validate_policy_indices(const PolicyIndices& policy) {
    if (policy.size == 0 || policy.size > policy.values.size())
        return tl::unexpected("Legal policy requires a nonempty bounded index list");
    std::bitset<1858> seen;
    for (std::size_t i = 0; i < policy.size; ++i) {
        const auto index = policy.values.at(i);
        if (index >= 1858 || seen.test(index))
            return tl::unexpected("Legal policy indices must be unique and in [0,1857]");
        seen.set(index);
    }
    return {};
}
Result<void> validate_legal_output(const LegalOutput& output, std::size_t count) {
    if (count == 0 || count > output.logits.size())
        return tl::unexpected("Invalid legal policy output shape");
    return validate_nn_heads(std::span{output.logits}.first(count), output.wdl, output.moves_left);
}
}  // namespace mouse::nn
