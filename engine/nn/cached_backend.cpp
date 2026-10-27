// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/nn/cached_backend.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "engine/nn/cache.h"

namespace mouse::nn {
namespace {
constexpr std::size_t kMaximumBatch = 1024;
constexpr std::size_t kNoMiss = std::numeric_limits<std::size_t>::max();
static_assert(std::is_nothrow_copy_assignable_v<NNOutput>);

bool exact_input(const InputPlanes& left, const InputPlanes& right) {
    return std::ranges::equal(left, right, [](const InputPlane& first, const InputPlane& second) {
        return first.mask == second.mask && std::bit_cast<std::uint32_t>(first.value) ==
                                                std::bit_cast<std::uint32_t>(second.value);
    });
}
void poison_output(NNOutput& output) {
    constexpr auto invalid = std::numeric_limits<float>::quiet_NaN();
    output.policy_logits.fill(invalid);
    output.wdl.fill(invalid);
    output.moves_left = invalid;
}

class CachedBackend final : public Backend {
   public:
    CachedBackend(std::unique_ptr<Backend> backend, std::unique_ptr<NNCache> cache,
                  std::size_t max_batch)
        : backend_(std::move(backend)),
          cache_(std::move(cache)),
          max_batch_(max_batch),
          miss_inputs_(max_batch),
          row_misses_(max_batch, kNoMiss),
          miss_outputs_(max_batch),
          staged_outputs_(max_batch) {}

    Result<void> evaluate(std::span<const InputPlanes> inputs,
                          std::span<NNOutput> outputs) override {
        try {
            if (inputs.empty() || inputs.size() > max_batch_ || inputs.size() != outputs.size()) {
                return tl::unexpected(
                    std::string{"Cached backend requires matching nonempty input/output "
                                "batches within max_batch"});
            }
            const bool finite = std::ranges::all_of(inputs, [](const InputPlanes& input) {
                return std::ranges::all_of(
                    input, [](const InputPlane& plane) { return std::isfinite(plane.value); });
            });
            if (!finite) {
                return tl::unexpected(
                    std::string{"Cached backend input plane values must be finite"});
            }
            const std::lock_guard lock{mutex_};
            std::size_t miss_count = 0;
            for (std::size_t row = 0; row < inputs.size(); ++row) {
                row_misses_.at(row) = kNoMiss;
                auto found = cache_->lookup(inputs[row]);
                if (!found) return tl::unexpected(std::move(found.error()));
                const auto& hit = *found;
                if (hit.has_value()) {
                    staged_outputs_.at(row) = *hit;
                    continue;
                }
                // Exact within-batch deduplication is bounded by max_batch <=1024.
                // It needs no second hash implementation or dynamically grown map.
                std::size_t unique = 0;
                while (unique < miss_count && !exact_input(miss_inputs_.at(unique), inputs[row])) {
                    ++unique;
                }
                if (unique == miss_count) {
                    miss_inputs_.at(miss_count) = inputs[row];
                    ++miss_count;
                }
                row_misses_.at(row) = unique;
            }
            if (miss_count != 0) {
                auto pending = std::span{miss_outputs_}.first(miss_count);
                for (auto& output : pending) poison_output(output);
                auto evaluated =
                    backend_->evaluate(std::span{miss_inputs_}.first(miss_count), pending);
                if (!evaluated) return evaluated;
                // Validate the complete batch before any new cache entry. Poison
                // prevents a successful partial provider write reusing old data.
                for (const auto& output : pending) {
                    auto valid = validate_nn_output(output);
                    if (!valid) return valid;
                }
                for (std::size_t index = 0; index < miss_count; ++index) {
                    auto inserted = cache_->insert(miss_inputs_.at(index), miss_outputs_.at(index));
                    if (!inserted) return inserted;
                }
                for (std::size_t row = 0; row < inputs.size(); ++row) {
                    const auto index = row_misses_.at(row);
                    if (index != kNoMiss) staged_outputs_.at(row) = miss_outputs_.at(index);
                }
            }
            // NNOutput assignment cannot throw; no caller row changes before
            // every input, provider result, validation and cache insert succeeds.
            std::ranges::copy(std::span{staged_outputs_}.first(inputs.size()), outputs.begin());
            return {};
        } catch (const std::exception& error) {
            return tl::unexpected(std::string{"Cached backend evaluation failed: "} + error.what());
        } catch (...) {
            return tl::unexpected(
                std::string{"Cached backend evaluation failed with an unknown exception"});
        }
    }
    [[nodiscard]] const ModelMetadata& metadata() const noexcept override {
        return backend_->metadata();
    }
    [[nodiscard]] const BackendInfo& info() const noexcept override { return backend_->info(); }

   private:
    std::unique_ptr<Backend> backend_;
    std::unique_ptr<NNCache> cache_;
    const std::size_t max_batch_;
    std::mutex mutex_;
    std::vector<InputPlanes> miss_inputs_;
    std::vector<std::size_t> row_misses_;
    std::vector<NNOutput> miss_outputs_;
    std::vector<NNOutput> staged_outputs_;
};
}  // namespace

Result<std::unique_ptr<Backend>> create_cached_backend(std::unique_ptr<Backend> backend,
                                                       std::size_t budget_bytes,
                                                       std::size_t max_batch) {
    if (!backend) return tl::unexpected(std::string{"Cached backend requires an owned backend"});
    if (max_batch == 0 || max_batch > kMaximumBatch) {
        return tl::unexpected(std::string{"Cached backend max_batch must be in [1,1024]"});
    }
    if (budget_bytes == 0) return backend;
    // Validate cache and staging as one addressable allocation configuration.
    // The LRU byte budget excludes staging, which is bounded by max_batch.
    constexpr auto row_bytes = sizeof(InputPlanes) + 2 * sizeof(NNOutput) + sizeof(std::size_t);
    constexpr auto control_bytes = sizeof(CachedBackend) + std::size_t{4} * 64;
    constexpr auto addressable =
        static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
    if (budget_bytes > addressable - control_bytes - max_batch * row_bytes) {
        return tl::unexpected(
            std::string{"Cached backend cache and scratch exceed addressable storage"});
    }
    auto cache = NNCache::create(budget_bytes, backend->info().cache_namespace);
    if (!cache) return tl::unexpected(std::move(cache.error()));
    try {
        std::unique_ptr<Backend> result =
            std::make_unique<CachedBackend>(std::move(backend), std::move(*cache), max_batch);
        return result;
    } catch (const std::bad_alloc&) {
        return tl::unexpected(std::string{"Cache OOM"});
    } catch (const std::length_error&) {
        return tl::unexpected(std::string{"Cached backend scratch exceeds container limits"});
    }
}
}  // namespace mouse::nn
