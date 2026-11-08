// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <optional>

#include "engine/search/tree.h"
#include "engine/util/result.h"

namespace mouse::search {
// Lc0's MovesLeft* search options, with its defaults and ranges
// (fd71a2d src/search/classic/params.cc).
struct MovesLeftParameters {
    double max_effect = 0.0345;         // MovesLeftMaxEffect 0..1
    double threshold = 0.8;             // MovesLeftThreshold 0..1
    double slope = 0.0027;              // MovesLeftSlope 0..1
    double constant_factor = 0.0;       // MovesLeftConstantFactor -1..1
    double scaled_factor = 1.6521;      // MovesLeftScaledFactor -2..2
    double quadratic_factor = -0.6521;  // MovesLeftQuadraticFactor -1..1
    bool operator==(const MovesLeftParameters&) const = default;
};
[[nodiscard]] Result<void> validate_moves_left_parameters(const MovesLeftParameters& parameters);
// Internal numerical helpers consume completed, validated search statistics.
// Callers synchronize Node access. Edge Q is from the PARENT viewpoint, unlike
// the child Node's stored value. Unknown or partial MLH samples disable utility.
[[nodiscard]] double moves_left_utility(const Node& parent, const Node& child,
                                        const MovesLeftParameters& parameters = {}) noexcept;
// The completed visit has already been added to node. Record a known M sample,
// then advance it by one ply for the caller's parent. Missing stays missing.
// Samples originate from a validated optional float NN head or exact terminal0.
void backup_moves_left_sample(Node& node, std::optional<double>& sample) noexcept;
}  // namespace mouse::search
