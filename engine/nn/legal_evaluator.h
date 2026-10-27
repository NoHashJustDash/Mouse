// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <chrono>
#include <mutex>
#include <vector>

#include "engine/nn/backend.h"
#include "engine/nn/legal_cache.h"
namespace mouse::nn {
// Borrowed backend/cache outlive this owner. Serialized evaluation stages the
// entire batch before output publication. All physical NNOutput slots (including
// illegal logits) are validated before extracting legal values or caching them.
// Hits submitted here complete with their batch. Search may consume cached
// values directly before submitting its remaining misses to this evaluator.
struct EvaluationTiming {
    std::size_t positions = 0;
    std::chrono::nanoseconds provider{};
};
class LegalEvaluator {
   public:
    LegalEvaluator(Backend&, std::size_t maximum_batch, LegalCache* = nullptr);
    ~LegalEvaluator() = default;
    LegalEvaluator(const LegalEvaluator&) = delete;
    LegalEvaluator& operator=(const LegalEvaluator&) = delete;
    LegalEvaluator(LegalEvaluator&&) = delete;
    LegalEvaluator& operator=(LegalEvaluator&&) = delete;
    [[nodiscard]] Result<void> evaluate(std::span<const InputPlanes>,
                                        std::span<const PolicyIndices>, std::span<LegalOutput>,
                                        EvaluationTiming* timing = nullptr);
    [[nodiscard]] std::size_t scratch_bytes() const noexcept;

   private:
    Backend& backend_;
    LegalCache* cache_;
    const std::size_t maximum_batch_;
    std::mutex mutex_;
    std::vector<InputPlanes> inputs_;
    std::vector<NNOutput> outputs_;
    std::vector<LegalOutput> staged_;
    std::vector<std::size_t> misses_;
};
}  // namespace mouse::nn
