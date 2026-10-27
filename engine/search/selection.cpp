// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/search/selection.h"

#include <array>
#include <cassert>
#include <cmath>
#include <limits>

#include "engine/chess/types.h"

namespace mouse::search {
bool prefer_final_move(const FinalMoveStatistics& first,
                       const FinalMoveStatistics& second) noexcept {
    // lc0 v0.32.1 src/search/classic/search.cc GetBestChildrenNoTemperature:
    // proven wins precede ordinary/drawn moves, which precede proven losses.
    // Mouse stores established mate strategy bounds, not exact shortest DTM.
    enum class Outcome : std::uint8_t { Loss, Ordinary, Win };
    const auto rank = [](const FinalMoveStatistics& move) {
        if (move.visits != 0 && move.child_proof.outcome == ProofOutcome::Loss) return Outcome::Win;
        if (move.visits != 0 && move.child_proof.outcome == ProofOutcome::Win) return Outcome::Loss;
        return Outcome::Ordinary;
    };
    const auto left = rank(first), right = rank(second);
    if (left != right) return left > right;
    if (left != Outcome::Ordinary &&
        first.child_proof.mate_upper_bound != second.child_proof.mate_upper_bound) {
        return left == Outcome::Win
                   ? first.child_proof.mate_upper_bound < second.child_proof.mate_upper_bound
                   : first.child_proof.mate_upper_bound > second.child_proof.mate_upper_bound;
    }
    // Unlike lc0, proven draws have no stored distance here. Preserve the
    // ordinary visit/Q/prior order for those ties; document that difference.
    if (first.visits != second.visits) return first.visits > second.visits;
    const auto first_q = first.visits == 0 ? 0.0 : first.q;
    const auto second_q = second.visits == 0 ? 0.0 : second.q;
    if (first_q != second_q) return first_q > second_q;
    return first.prior > second.prior;
}

namespace {
double logarithmic_visits(const SelectionParameters& parameters, std::uint64_t visits) {
    const auto count = static_cast<double>(visits);
    const double ratio = count / parameters.cpuct_base;
    if (std::isfinite(ratio)) return std::log1p(ratio);
    // Algebraically log((N+base)/base), without overflowing N/base for a
    // representable tiny base. N is positive whenever that ratio overflows.
    return std::log(count + parameters.cpuct_base) - std::log(parameters.cpuct_base);
}
}  // namespace

Result<void> validate_selection_parameters(const SelectionParameters& parameters) {
    const std::array values{parameters.cpuct,        parameters.cpuct_base,
                            parameters.cpuct_factor, parameters.fpu_reduction,
                            parameters.root_fpu,     parameters.policy_temperature};
    for (const double value : values)
        if (!std::isfinite(value)) return tl::unexpected("Selection parameters must all be finite");
    if (parameters.cpuct < 0 || parameters.cpuct_factor < 0 || parameters.fpu_reduction < 0 ||
        parameters.cpuct_base <= 0 || parameters.policy_temperature <= 0 ||
        parameters.root_fpu < -100 || parameters.root_fpu > 100)
        return tl::unexpected("Selection parameters are outside their mathematical ranges");
    if (parameters.root_fpu_strategy != RootFpuStrategy::Same &&
        parameters.root_fpu_strategy != RootFpuStrategy::Reduction &&
        parameters.root_fpu_strategy != RootFpuStrategy::Absolute)
        return tl::unexpected("Invalid root FPU strategy");
    constexpr auto maximum_visits = std::numeric_limits<std::uint64_t>::max();
    if (!std::isfinite(cpuct_value(parameters, maximum_visits)) ||
        !std::isfinite(exploration_multiplier(parameters, maximum_visits)) ||
        !std::isfinite(parameters.fpu_reduction *
                       std::sqrt(static_cast<double>(chess::MoveList::kCapacity))) ||
        !std::isfinite(parameters.root_fpu *
                       std::sqrt(static_cast<double>(chess::MoveList::kCapacity))))
        return tl::unexpected("Selection parameters overflow representable search arithmetic");
    return {};
}

double cpuct_value(const SelectionParameters& parameters, std::uint64_t parent_visits) {
    // Independent implementation of pinned Lc0 classic/search.cc428-451:
    // initial + factor*log((N+base)/base). There is no additional +1 visit.
    return parameters.cpuct +
           parameters.cpuct_factor * logarithmic_visits(parameters, parent_visits);
}

double exploration_multiplier(const SelectionParameters& parameters, std::uint64_t parent_visits) {
    // Pinned Lc0 classic/search.cc1708-1734: sqrt(max(N-1,1)). Guard zero
    // before unsigned subtraction; in-flight visits belong only to child denominators.
    const auto effective = parent_visits > 1 ? parent_visits - 1 : std::uint64_t{1};
    return cpuct_value(parameters, parent_visits) * std::sqrt(static_cast<double>(effective));
}

double first_play_urgency(const SelectionParameters& parameters, bool root, double parent_q,
                          double visited_prior_mass) {
    assert(std::isfinite(parent_q) && parent_q >= -1 && parent_q <= 1);
    assert(std::isfinite(visited_prior_mass) && visited_prior_mass >= 0);
    // Lc0 fd71a2d classic/params.cc567-568,657-662 and search.cc428-445:
    // "same" inherits the ordinary reduction; root value applies only to an
    // explicit root override. Half-rounded visited prior mass is not clamped.
    if (root && parameters.root_fpu_strategy == RootFpuStrategy::Absolute)
        return parameters.root_fpu;
    const double reduction = root && parameters.root_fpu_strategy == RootFpuStrategy::Reduction
                                 ? parameters.root_fpu
                                 : parameters.fpu_reduction;
    return parent_q - reduction * std::sqrt(visited_prior_mass);
}
}  // namespace mouse::search
