// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string_view>
#include <vector>

#include "engine/nn/encoder.h"
#include "engine/nn/output.h"
#include "engine/util/result.h"

namespace mouse::nn {
struct CacheStats {
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;
    std::uint64_t evictions = 0;
    std::uint64_t bucket_collisions = 0;
    std::size_t entries = 0;
    std::size_t capacity = 0;
    std::size_t charged_bytes = 0;
    std::size_t budget_bytes = 0;
    bool operator==(const CacheStats&) const = default;
};

// One mutex linearizes lookup, insert, clear and stats. Callers keep argument
// storage alive and unchanged for the call. Lookup returns an independent copy;
// nothing borrows a node or output that another thread could replace or evict.
//
// Identity must describe the actual graph bytes, provider, precision, device and
// runtime options. It is copied once at creation and cannot be changed. An
// instance never shares entries with another namespace. Every key includes all
// 112 masks and float32 value bits, including signed zero and empty-plane values.
class NNCache {
    struct FactoryToken {
       private:
        FactoryToken() = default;
        friend class NNCache;
    };

   public:
    // Budget charges sizeof(NNCache), copied identity capacity, Node capacity
    // (including full inputs/outputs and all links), bucket index capacity, and
    // a conservative 64 bytes per buffer for allocation/alignment bookkeeping.
    // Allocator/OS metadata outside requested C++ storage cannot be measured by
    // the standard allocator and is not part of this payload/bookkeeping budget.
    // A nonzero budget unable to fit one entry fails. Zero allocates only the
    // fixed control object, retaining no identity, node or bucket payload.
    [[nodiscard]] static Result<std::unique_ptr<NNCache>> create(std::size_t budget_bytes,
                                                                 std::string_view identity);
    explicit NNCache(FactoryToken);
    ~NNCache();
    NNCache(const NNCache&) = delete;
    NNCache& operator=(const NNCache&) = delete;
    NNCache(NNCache&&) = delete;
    NNCache& operator=(NNCache&&) = delete;

    // Invalid values leave contents, LRU order and every counter unchanged.
    // Valid lookup/insert/clear/stats perform no dynamic allocation.
    [[nodiscard]] Result<std::optional<NNOutput>> lookup(const InputPlanes& input);
    [[nodiscard]] Result<void> insert(const InputPlanes& input, const NNOutput& output);
    // Clear removes all entries, preserving capacity and cumulative counters.
    void clear();
    [[nodiscard]] CacheStats stats() const;

   private:
    static constexpr std::size_t kNone = std::numeric_limits<std::size_t>::max();
    struct Node {
        InputPlanes input{};
        NNOutput output{};
        std::uint64_t hash = 0;
        std::size_t bucket_next = kNone;
        std::size_t previous = kNone;
        std::size_t next = kNone;
    };
    [[nodiscard]] std::uint64_t hash_input(const InputPlanes& input) const noexcept;
    [[nodiscard]] std::size_t find(std::uint64_t hash, const InputPlanes& input);
    void unlink(std::size_t index);
    void link_front(std::size_t index);
    void touch(std::size_t index);
    void remove_bucket(std::size_t index);

    mutable std::mutex mutex_;
    std::vector<char> identity_;
    std::vector<Node> nodes_;
    std::vector<std::size_t> buckets_;
    std::uint64_t namespace_seed_ = 0;
    std::size_t head_ = kNone;
    std::size_t tail_ = kNone;
    CacheStats stats_{};
};
}  // namespace mouse::nn
