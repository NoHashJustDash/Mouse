// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <list>
#include <memory>
#include <memory_resource>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "engine/nn/cache.h"
#include "engine/nn/legal_output.h"
#include "engine/util/hash128.h"
namespace mouse::nn {
struct CacheByteBudget {
    std::size_t bytes = 0;
    std::size_t identity_bytes = 0;
};
// Serialized LRU with on-demand entries and variable-length float32 legal logits.
// A 128-bit fingerprint covers the provider/model namespace, the current input
// frame plus `history_length` earlier frames, the auxiliary planes and the
// ordered policy indices. lc0 CacheHistoryLength (default 0) likewise keys on the
// last position, its repetitions and rule50 only (chess/position.cc HashLast),
// so one evaluation serves the same position reached through another history.
// kFullCacheHistory keys on every input bit. 128-bit fingerprints replace full
// key comparison; collisions are probabilistic, not mathematically impossible.
// All mutable storage and accounting are protected by mutex_; identity is immutable.
class LegalCache {
    struct FactoryToken {
       private:
        FactoryToken() = default;
        friend class LegalCache;
    };

   public:
    static constexpr std::size_t kFullCacheHistory = 7;
    [[nodiscard]] static Result<std::unique_ptr<LegalCache>> create(
        std::size_t entries, std::string_view identity,
        std::size_t history_length = kFullCacheHistory);
    [[nodiscard]] static Result<std::size_t> budget_for_entries(std::size_t entries,
                                                                std::size_t identity_bytes);
    [[nodiscard]] static Result<std::size_t> capacity_for_budget(CacheByteBudget);
    explicit LegalCache(FactoryToken, std::size_t allocation_budget);
    ~LegalCache();
    LegalCache(const LegalCache&) = delete;
    LegalCache& operator=(const LegalCache&) = delete;
    LegalCache(LegalCache&&) = delete;
    LegalCache& operator=(LegalCache&&) = delete;
    [[nodiscard]] Result<std::optional<LegalOutput>> lookup(const InputPlanes&,
                                                            const PolicyIndices&);
    [[nodiscard]] Result<void> insert(const InputPlanes&, const PolicyIndices&, const LegalOutput&);
    // Hits and same-key updates allocate nothing. New entries may allocate;
    // allocation failures preserve existing entries and are returned explicitly.
    void clear();
    [[nodiscard]] CacheStats stats() const;
    [[nodiscard]] bool matches_identity(std::string_view) const noexcept;
    [[nodiscard]] std::size_t history_length() const noexcept { return history_length_; }

   private:
    class Memory final : public std::pmr::memory_resource {
       public:
        explicit Memory(std::size_t limit) noexcept : limit_(limit) {}
        [[nodiscard]] std::size_t charged() const noexcept { return charged_; }

       private:
        void* do_allocate(std::size_t, std::size_t) override;
        void do_deallocate(void*, std::size_t, std::size_t) override;
        [[nodiscard]] bool do_is_equal(const std::pmr::memory_resource&) const noexcept override;
        std::size_t limit_ = 0, charged_ = 0;
    };
    struct Entry {
        Hash128 key;
        std::pmr::vector<float> logits;
        std::array<float, 3> wdl;
        std::optional<float> moves_left;
        Entry(Hash128, std::span<const float>, const LegalOutput&, std::pmr::memory_resource*);
    };
    using Entries = std::pmr::list<Entry>;
    struct KeyHash {
        [[nodiscard]] std::size_t operator()(const Hash128& key) const noexcept {
            return static_cast<std::size_t>(key[0] ^ key[1]);
        }
    };
    using Index = std::pmr::unordered_map<Hash128, Entries::iterator, KeyHash>;
    struct Storage {
        std::pmr::string identity;
        Entries entries;
        Index index;
        explicit Storage(std::pmr::memory_resource*);
    };
    [[nodiscard]] Hash128 hash(const InputPlanes&, const PolicyIndices&) const noexcept;
    [[nodiscard]] Entries::iterator find(Storage&, const Hash128&);
    mutable std::mutex mutex_;
    Memory memory_;
    std::optional<Storage> storage_;
    Hash128 namespace_{};
    std::size_t history_length_ = kFullCacheHistory;
    CacheStats stats_;
};
}  // namespace mouse::nn