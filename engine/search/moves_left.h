// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <optional>

#include "engine/search/tree.h"

namespace mouse::search {
// Internal numerical helpers consume completed, validated search statistics.
// Callers synchronize Node access. Edge Q is from the PARENT viewpoint, unlike
// the child Node's stored value. Unknown or partial MLH samples disable utility.
[[nodiscard]] double moves_left_utility(const Node& parent, const Node& child) noexcept;
// The completed visit has already been added to node. Record a known M sample,
// then advance it by one ply for the caller's parent. Missing stays missing.
// Samples originate from a validated optional float NN head or exact terminal0.
void backup_moves_left_sample(Node& node, std::optional<double>& sample) noexcept;
}  // namespace mouse::search
