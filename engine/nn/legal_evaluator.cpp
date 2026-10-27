// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/nn/legal_evaluator.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace mouse::nn {
namespace {
constexpr auto kHit = std::numeric_limits<std::size_t>::max();
bool same(const InputPlanes& left, const InputPlanes& right) {
    return std::ranges::equal(left, right, [](const InputPlane& a, const InputPlane& b) {
        return a.mask == b.mask &&
               std::bit_cast<std::uint32_t>(a.value) == std::bit_cast<std::uint32_t>(b.value);
    });
}
}  // namespace
LegalEvaluator::LegalEvaluator(Backend& backend, std::size_t maximum_batch, LegalCache* cache)
    : backend_(backend),
      cache_(cache && cache->stats().capacity != 0 ? cache : nullptr),
      maximum_batch_(maximum_batch) {
    if (maximum_batch == 0 || maximum_batch > 1024)
        throw std::invalid_argument("Invalid legal evaluator batch capacity");
    if (cache_ && !cache_->matches_identity(backend.info().cache_namespace))
        throw std::invalid_argument("Legal cache model/provider namespace mismatch");
    inputs_.resize(maximum_batch);
    outputs_.resize(maximum_batch);
    staged_.resize(maximum_batch);
    misses_.resize(maximum_batch);
}
std::size_t LegalEvaluator::scratch_bytes() const noexcept {
    return inputs_.capacity() * sizeof(InputPlanes) + outputs_.capacity() * sizeof(NNOutput) +
           staged_.capacity() * sizeof(LegalOutput) + misses_.capacity() * sizeof(std::size_t) +
           std::size_t{4} * 64;
}
Result<void> LegalEvaluator::evaluate(std::span<const InputPlanes> inputs,
                                      std::span<const PolicyIndices> policy,
                                      std::span<LegalOutput> outputs, EvaluationTiming* timing) {
    if (timing) *timing = {};
    if (inputs.empty() || inputs.size() > maximum_batch_ || policy.size() != inputs.size() ||
        outputs.size() != inputs.size())
        return tl::unexpected("Legal evaluation requires matching bounded nonempty batch shapes");
    for (std::size_t row = 0; row < inputs.size(); ++row) {
        if (!std::ranges::all_of(
                inputs[row], [](const InputPlane& plane) { return std::isfinite(plane.value); }))
            return tl::unexpected("Legal evaluation input values must be finite");
        auto valid = validate_policy_indices(policy[row]);
        if (!valid) return valid;
    }
    const std::lock_guard lock(mutex_);
    std::size_t count = 0;
    for (std::size_t row = 0; row < inputs.size(); ++row) {
        misses_.at(row) = kHit;
        if (cache_) {
            auto found = cache_->lookup(inputs[row], policy[row]);
            if (!found) return tl::unexpected(found.error());
            const auto& cached = *found;
            if (cached.has_value()) {
                staged_.at(row) = cached.value();
                continue;
            }
        }
        std::size_t index = 0;
        if (cache_)
            while (index < count && !same(inputs_.at(index), inputs[row])) ++index;
        else
            index = count;  // Cache-off preserves original physical batch cardinality.
        if (index == count) {
            inputs_.at(count) = inputs[row];
            ++count;
        }
        misses_.at(row) = index;
    }
    if (count != 0) {
        for (std::size_t i = 0; i < count; ++i) {
            outputs_.at(i).policy_logits.fill(std::numeric_limits<float>::quiet_NaN());
            outputs_.at(i).wdl.fill(std::numeric_limits<float>::quiet_NaN());
            outputs_.at(i).moves_left = std::numeric_limits<float>::quiet_NaN();
        }
        const auto began =
            timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        auto evaluated =
            backend_.evaluate(std::span{inputs_}.first(count), std::span{outputs_}.first(count));
        if (timing) {
            timing->positions = backend_.physical_batch_size(count);
            timing->provider = std::chrono::steady_clock::now() - began;
        }
        if (!evaluated) return evaluated;
        for (std::size_t i = 0; i < count; ++i) {
            auto valid = validate_nn_output(outputs_.at(i));
            if (!valid) return valid;
        }
        for (std::size_t row = 0; row < inputs.size(); ++row) {
            if (misses_.at(row) == kHit) continue;
            const auto& physical = outputs_.at(misses_.at(row));
            auto& result = staged_.at(row);
            result = {};
            result.wdl = physical.wdl;
            result.moves_left = physical.moves_left;
            for (std::size_t j = 0; j < policy[row].size; ++j)
                result.logits.at(j) = physical.policy_logits.at(policy[row].values.at(j));
        }
        for (std::size_t row = 0; cache_ && row < inputs.size(); ++row)
            if (misses_.at(row) != kHit) {
                auto inserted = cache_->insert(inputs[row], policy[row], staged_.at(row));
                if (!inserted) return inserted;
            }
    }
    std::ranges::copy(std::span{staged_}.first(inputs.size()), outputs.begin());
    return {};
}
}  // namespace mouse::nn
