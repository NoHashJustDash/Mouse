// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cstddef>
#include <cstdint>
#include <optional>

#include "engine/util/result.h"

namespace mouse::search {
enum class TerminalReason : std::uint8_t;
enum class ProofOutcome : std::uint8_t { Unknown, Win, Draw, Loss };
inline constexpr std::uint16_t kMaximumProofPlies = 512;
// Outcome uses this node's side-to-move viewpoint. A mate upper bound is an
// established strategy bound, not shortest DTM while unknown children remain.
// Unknown/Draw have bound0; Loss0 is a genuinely checkmated position.
struct NodeProof {
    ProofOutcome outcome = ProofOutcome::Unknown;
    std::uint16_t mate_upper_bound = 0;
    bool operator==(const NodeProof&) const = default;
};
// Only genuine rule terminals seed known proof. Tree-twofold/NN/frontier values
// are deliberately ineligible; unrecognized reasons also cannot establish proof.
[[nodiscard]] NodeProof proof_from_terminal(TerminalReason reason) noexcept;
// Original bounded selection heuristic, not exact DTM or an Lc0 parity claim.
// A compatible proof bonus replaces NN MLH; nullopt leaves ordinary selection.
// Every edge remains eligible, including unknown and currently longer bounds.
[[nodiscard]] Result<std::optional<double>> proof_selection_utility(NodeProof parent,
                                                                    NodeProof child);
// Pure, allocation-free successful accumulation. Caller enumerates the complete
// legal edge set, including Unknown for missing/unallocated children. Counts,
// child encoding and bound+1 validate before mutation/publication. An empty set
// is Unknown: only chess classification can establish a terminal position.
class ProofAccumulator {
   public:
    [[nodiscard]] static Result<ProofAccumulator> create(std::size_t legal_children);
    [[nodiscard]] Result<void> add(NodeProof child);
    [[nodiscard]] Result<NodeProof> finish() const;

   private:
    std::uint16_t expected_ = 0, seen_ = 0, wins_ = 0;
    std::uint16_t minimum_loss_ = kMaximumProofPlies, maximum_win_ = 0;
    bool has_loss_ = false, has_draw_ = false, has_unknown_ = false;
};
}  // namespace mouse::search