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
// Creation-time key and index policy, fixed for the lifetime of a LegalCache.
struct LegalCacheOptions {
    // Mix the rule-50 plane (classical plane 109) into the key. Lc0 v0.32.1
    // leaves it out (see LegalCache); true keeps the stricter key of
    // chess/position.cc PositionHistory::HashLast, which mixes in rule 50 but
    // which the v0.32.1 cache no longer calls. kFullCacheHistory requires true.
    bool key_rule50 = true;
    // Reserve index buckets for capacity + 1 entries at creation, so filling
    // the cache never rehashes under the lock. The bucket array is charged to
    // the budget's per-entry index allowance; false allocates it on demand.
    bool reserve_index = false;
};
// Serialized LRU with on-demand entries and variable-length float32 legal logits.
// A 128-bit fingerprint covers the provider/model namespace, the current input
// frame plus `history_length` earlier frames, the auxiliary planes (castling,
// side to move, constant planes and, when `key_rule50` is set, rule 50) and the
// ordered policy indices. Lc0 v0.32.1 keys its evaluation cache on the last
// position only: neural/memcache.cc ComputeEvalPositionHash returns
// pos.pos.back().Hash(), and chess/position.cc Position::Hash() combines the
// board hash (pieces, castling, en passant, side to move) with the repetition
// count, leaving out rule 50 and earlier history. history_length 0 with
// key_rule50 false approximates that key: frame 0 carries the repetition plane
// (repeated or not, where Lc0 hashes the count), and the ordered policy indices,
// stricter than Lc0's stored legal-move-count check, separate positions whose
// legal moves differ, which covers an en passant right that can be used. So one
// evaluation serves the same position reached through another history or at
// another 50-move count; such a hit is the network's output for the count it was
// stored with, not for the query's.
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
        std::size_t history_length = kFullCacheHistory, LegalCacheOptions options = {});
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
    [[nodiscard]] bool keys_rule50() const noexcept { return options_.key_rule50; }
    [[nodiscard]] bool reserves_index() const noexcept { return options_.reserve_index; }
    // Current index bucket count (0 without storage), so tests and diagnostics
    // can observe that a reserved index never rehashes.
    [[nodiscard]] std::size_t index_bucket_count() const;

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
    LegalCacheOptions options_{};
    CacheStats stats_;
};
}  // namespace mouse::nn