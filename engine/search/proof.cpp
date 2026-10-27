// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/search/proof.h"

#include <algorithm>
#include <string>

#include "engine/search/rules.h"

namespace mouse::search {
namespace {
Result<void> validate_proof(NodeProof value) {
    if (value.mate_upper_bound > kMaximumProofPlies)
        return tl::unexpected(std::string{"Proof child mate upper bound exceeds capacity"});
    switch (value.outcome) {
        case ProofOutcome::Unknown:
        case ProofOutcome::Draw:
            if (value.mate_upper_bound != 0)
                return tl::unexpected(std::string{"Unknown/drawn proof cannot carry a mate bound"});
            break;
        case ProofOutcome::Win:
            if (value.mate_upper_bound == 0)
                return tl::unexpected(std::string{"Winning proof requires a positive mate bound"});
            break;
        case ProofOutcome::Loss:
            break;
        default:
            return tl::unexpected(std::string{"Proof child has an invalid outcome"});
    }
    return {};
}
}  // namespace

NodeProof proof_from_terminal(TerminalReason reason) noexcept {
    switch (reason) {
        case TerminalReason::Checkmate:
            return {.outcome = ProofOutcome::Loss};
        case TerminalReason::Stalemate:
        case TerminalReason::FiftyMove:
        case TerminalReason::Threefold:
        case TerminalReason::InsufficientMaterial:
            return {.outcome = ProofOutcome::Draw};
        case TerminalReason::None:
        case TerminalReason::TreeRepetition:
            return {};
    }
    return {};
}
Result<std::optional<double>> proof_selection_utility(NodeProof parent, NodeProof child) {
    const auto valid_parent = validate_proof(parent);
    if (!valid_parent) return tl::unexpected(valid_parent.error());
    const auto valid_child = validate_proof(child);
    if (!valid_child) return tl::unexpected(valid_child.error());
    int gap = 0;
    if (parent.outcome == ProofOutcome::Win && child.outcome == ProofOutcome::Loss) {
        gap = 1 + static_cast<int>(child.mate_upper_bound) - parent.mate_upper_bound;
    } else if (parent.outcome == ProofOutcome::Loss && parent.mate_upper_bound != 0 &&
               child.outcome == ProofOutcome::Win) {
        gap = static_cast<int>(parent.mate_upper_bound) - 1 - child.mate_upper_bound;
    } else {
        return std::optional<double>{};
    }
    // The scale is the independently verified Lc0 MLH cap (search.cc at
    // fd71a2d); the decreasing gap formula is Mouse's own heuristic. A finite
    // bonus never prunes a line whose current upper bound can still improve.
    constexpr double kMaximumProofUtility = 0.0345;
    return std::optional<double>{kMaximumProofUtility / (1.0 + std::max(0, gap))};
}
Result<ProofAccumulator> ProofAccumulator::create(std::size_t legal_children) {
    if (legal_children > chess::MoveList::kCapacity)
        return tl::unexpected(std::string{"Proof legal-child count exceeds capacity"});
    ProofAccumulator result;
    result.expected_ = static_cast<std::uint16_t>(legal_children);
    return result;
}
Result<void> ProofAccumulator::add(NodeProof child) {
    if (seen_ == expected_)
        return tl::unexpected(std::string{"Proof received more than the complete legal child set"});
    const auto valid = validate_proof(child);
    if (!valid) return tl::unexpected(valid.error());
    // No mutation until all validation succeeds. A caller can repair/retry an
    // invalid child without losing the accumulator's previously observed set.
    ++seen_;
    switch (child.outcome) {
        case ProofOutcome::Unknown:
            has_unknown_ = true;
            break;
        case ProofOutcome::Draw:
            has_draw_ = true;
            break;
        case ProofOutcome::Win:
            ++wins_;
            maximum_win_ = std::max(maximum_win_, child.mate_upper_bound);
            break;
        case ProofOutcome::Loss:
            has_loss_ = true;
            minimum_loss_ = std::min(minimum_loss_, child.mate_upper_bound);
            break;
    }
    return {};
}
Result<NodeProof> ProofAccumulator::finish() const {
    if (seen_ != expected_)
        return tl::unexpected(std::string{"Proof requires the complete legal child set"});
    if (expected_ == 0) return NodeProof{};
    if (has_loss_ || wins_ == expected_) {
        const auto child_bound = has_loss_ ? minimum_loss_ : maximum_win_;
        if (child_bound >= kMaximumProofPlies)
            return tl::unexpected(std::string{"Proof mate upper-bound addition exceeds capacity"});
        return NodeProof{.outcome = has_loss_ ? ProofOutcome::Win : ProofOutcome::Loss,
                         .mate_upper_bound = static_cast<std::uint16_t>(child_bound + 1U)};
    }
    if (!has_unknown_ && has_draw_) return NodeProof{.outcome = ProofOutcome::Draw};
    return NodeProof{};
}
}  // namespace mouse::search