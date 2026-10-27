// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/nn/legal_cache.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <iterator>
#include <limits>
#include <new>
#include <stdexcept>
namespace mouse::nn {
namespace {
constexpr std::size_t kAllocationReserve = 64, kStartupReserve = 1024, kIndexReserve = 512;
std::size_t allocation_count(std::size_t bytes) noexcept {
    return bytes == 0 ? 1 : 1 + (bytes - 1) / sizeof(std::max_align_t);
}
std::uint64_t increment(std::uint64_t value) noexcept {
    return value + static_cast<std::uint64_t>(value != std::numeric_limits<std::uint64_t>::max());
}
bool finite(const InputPlanes& input) {
    return std::ranges::all_of(input,
                               [](const InputPlane& plane) { return std::isfinite(plane.value); });
}
}  // namespace
// All dynamic cache allocations use this per-instance resource. Charge requested
// bytes plus an allocator allowance; neither the capacity nor its upper byte
// budget commits storage in advance. The OS private-commit gate is measured
// separately, rather than equating this accounting with resident/committed pages.
void* LegalCache::Memory::do_allocate(std::size_t bytes, std::size_t alignment) {
    const auto available = limit_ - charged_;
    if (bytes > available || alignment > alignof(std::max_align_t)) throw std::bad_alloc{};
    const auto count = allocation_count(bytes);
    const auto allocated = count * sizeof(std::max_align_t);
    if (allocated > available || kAllocationReserve > available - allocated) throw std::bad_alloc{};
    // These containers need at most max_align_t alignment. A local allocator
    // keeps allocation/failure instrumentation valid across Windows DLL CRTs.
    auto* result = std::allocator<std::max_align_t>{}.allocate(count);
    charged_ += allocated + kAllocationReserve;
    return result;
}
void LegalCache::Memory::do_deallocate(void* pointer, std::size_t bytes, std::size_t) {
    const auto count = allocation_count(bytes);
    const auto allocated = count * sizeof(std::max_align_t);
    assert(allocated <= charged_ && kAllocationReserve <= charged_ - allocated);
    std::allocator<std::max_align_t>{}.deallocate(static_cast<std::max_align_t*>(pointer), count);
    charged_ -= allocated + kAllocationReserve;
}
bool LegalCache::Memory::do_is_equal(const std::pmr::memory_resource& other) const noexcept {
    return this == &other;
}
LegalCache::Entry::Entry(Hash128 fingerprint, std::span<const float> values,
                         const LegalOutput& output, std::pmr::memory_resource* memory)
    : key(fingerprint),
      logits(values.begin(), values.end(), memory),
      wdl(output.wdl),
      moves_left(output.moves_left) {}
LegalCache::Storage::Storage(std::pmr::memory_resource* memory)
    : identity(memory), entries(memory), index(memory) {}
LegalCache::LegalCache(FactoryToken, std::size_t allocation_budget) : memory_(allocation_budget) {}
LegalCache::~LegalCache() = default;
Result<std::size_t> LegalCache::budget_for_entries(std::size_t entries,
                                                   std::size_t identity_bytes) {
    if (entries == 0) return std::size_t{0};
    constexpr auto limit = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
    // Include maximum legal payloads, container nodes/bucket growth, allocator
    // allowances and one extra entry so replacement can fail without eviction.
    constexpr auto entry = sizeof(Entry) + chess::MoveList::kCapacity * sizeof(float) + 512;
    constexpr auto fixed = sizeof(LegalCache) + kStartupReserve + entry;
    constexpr auto width = entry + kIndexReserve;
    if (identity_bytes == 0 || identity_bytes > limit - fixed ||
        entries > (limit - fixed - identity_bytes) / width)
        return tl::unexpected(std::string{"Legal cache capacity exceeds addressable storage"});
    return fixed + identity_bytes + entries * width;
}
Result<std::size_t> LegalCache::capacity_for_budget(CacheByteBudget request) {
    const auto [bytes, identity_bytes] = request;
    if (bytes == 0) return std::size_t{0};
    const auto minimum = budget_for_entries(1, identity_bytes);
    if (!minimum) return tl::unexpected(minimum.error());
    constexpr auto width =
        sizeof(Entry) + chess::MoveList::kCapacity * sizeof(float) + 512 + kIndexReserve;
    if (bytes < *minimum ||
        bytes > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
        return tl::unexpected(
            std::string{"Legal cache budget cannot fit addressable complete entries"});
    return std::size_t{1} + (bytes - *minimum) / width;
}
Result<std::unique_ptr<LegalCache>> LegalCache::create(std::size_t entries,
                                                       std::string_view identity,
                                                       std::size_t history_length) {
    if (identity.empty())
        return tl::unexpected(std::string{"Legal cache identity must be nonempty"});
    if (history_length > kFullCacheHistory)
        return tl::unexpected(std::string{"Legal cache history length must be 0..7"});
    const auto budget = budget_for_entries(entries, identity.size());
    if (!budget) return tl::unexpected(budget.error());
    try {
        auto cache = std::make_unique<LegalCache>(FactoryToken{},
                                                  *budget == 0 ? 0 : *budget - sizeof(LegalCache));
        cache->stats_.budget_bytes = *budget;
        cache->history_length_ = history_length;
        if (entries == 0) return cache;
        auto& data = cache->storage_.emplace(&cache->memory_);
        data.identity.assign(identity);
        cache->namespace_ = hash128(std::as_bytes(std::span{identity}));
        cache->stats_.capacity = entries;
        return cache;
    } catch (const std::bad_alloc&) {
        return tl::unexpected(std::string{"NN cache OOM"});
    } catch (const std::length_error&) {
        return tl::unexpected(std::string{"Cache too large"});
    }
}
bool LegalCache::matches_identity(std::string_view identity) const noexcept {
    return storage_ && std::ranges::equal(storage_->identity, identity);
}
Hash128 LegalCache::hash(const InputPlanes& input, const PolicyIndices& policy) const noexcept {
    // Canonical little-endian serialization: no struct padding, lost empty-plane
    // values or signed-zero canonicalization. Namespace and index order matter.
    constexpr auto bytes = 16 + 112 * 12 + 8 + 2 * chess::MoveList::kCapacity;
    std::array<std::byte, bytes> key{};
    std::size_t used = 0;
    const auto append = [&](std::uint64_t value, std::size_t count) {
        for (std::size_t i = 0; i < count; ++i) {
            key.at(used++) = static_cast<std::byte>(value & 0xffU);
            value >>= 8;
        }
    };
    for (const auto word : namespace_) append(word, 8);
    // Classical 112-plane layout: 8 frames of 13 planes (current first), then
    // 8 auxiliary planes (castling, side to move, rule50, constant planes).
    constexpr std::size_t kFramePlanes = 13, kAuxiliaryBegin = 104;
    const auto keyed_frames = kFramePlanes * (history_length_ + 1);
    for (std::size_t index = 0; index < input.size(); ++index) {
        if (index >= keyed_frames && index < kAuxiliaryBegin) continue;
        append(input.at(index).mask, 8);
        append(std::bit_cast<std::uint32_t>(input.at(index).value), 4);
    }
    append(policy.size, 8);
    for (std::size_t i = 0; i < policy.size; ++i) append(policy.values.at(i), 2);
    return hash128(std::span{key}.first(used));
}
LegalCache::Entries::iterator LegalCache::find(Storage& data, const Hash128& key) {
    if (!data.index.empty()) {
        const auto bucket = data.index.bucket(key);
        for (auto it = data.index.begin(bucket); it != data.index.end(bucket); ++it) {
            if (it->first == key) return it->second;
            stats_.bucket_collisions = increment(stats_.bucket_collisions);
        }
    }
    return data.entries.end();
}
Result<std::optional<LegalOutput>> LegalCache::lookup(const InputPlanes& input,
                                                      const PolicyIndices& policy) {
    if (!finite(input))
        return tl::unexpected(std::string{"Legal cache input values must be finite"});
    const auto valid = validate_policy_indices(policy);
    if (!valid) return tl::unexpected(valid.error());
    const std::lock_guard lock(mutex_);
    if (storage_) {
        auto& data = *storage_;
        const auto entry = find(data, hash(input, policy));
        if (entry != data.entries.end()) {
            if (entry->logits.size() != policy.size)
                return tl::unexpected(std::string{"Legal cache fingerprint shape collision"});
            LegalOutput output;
            std::ranges::copy(entry->logits, output.logits.begin());
            output.wdl = entry->wdl;
            output.moves_left = entry->moves_left;
            data.entries.splice(data.entries.begin(), data.entries, entry);
            stats_.hits = increment(stats_.hits);
            return std::optional<LegalOutput>{output};
        }
    }
    stats_.misses = increment(stats_.misses);
    return std::optional<LegalOutput>{};
}
Result<void> LegalCache::insert(const InputPlanes& input, const PolicyIndices& policy,
                                const LegalOutput& output) {
    if (!finite(input))
        return tl::unexpected(std::string{"Legal cache input values must be finite"});
    auto valid = validate_policy_indices(policy);
    if (!valid) return valid;
    valid = validate_legal_output(output, policy.size);
    if (!valid) return valid;
    const std::lock_guard lock(mutex_);
    if (!storage_) return {};
    auto& data = *storage_;
    const auto before = stats_;
    const auto key = hash(input, policy);
    const auto found = find(data, key);
    const auto logits = std::span{output.logits}.first(policy.size);
    if (found != data.entries.end()) {
        if (found->logits.size() != policy.size)
            return tl::unexpected(std::string{"Legal cache fingerprint shape collision"});
        std::ranges::copy(logits, found->logits.begin());
        found->wdl = output.wdl;
        found->moves_left = output.moves_left;
        data.entries.splice(data.entries.begin(), data.entries, found);
        return {};
    }
    try {
        data.entries.emplace_front(key, logits, output, &memory_);
        try {
            const auto inserted = data.index.emplace(key, data.entries.begin()).second;
            if (!inserted) {
                data.entries.pop_front();
                stats_ = before;
                return tl::unexpected(std::string{"Legal cache duplicate fingerprint"});
            }
        } catch (...) {
            data.entries.pop_front();
            throw;
        }
        if (stats_.entries == stats_.capacity) {
            const auto victim = std::prev(data.entries.end());
            data.index.erase(victim->key);
            data.entries.erase(victim);
            stats_.evictions = increment(stats_.evictions);
        } else {
            ++stats_.entries;
        }
        return {};
    } catch (const std::bad_alloc&) {
        stats_ = before;
        return tl::unexpected(std::string{"NN cache OOM"});
    } catch (const std::length_error&) {
        stats_ = before;
        return tl::unexpected(std::string{"Cache too large"});
    }
}
void LegalCache::clear() {
    const std::lock_guard lock(mutex_);
    if (storage_) {
        storage_->index.clear();
        storage_->entries.clear();
    }
    stats_.entries = 0;
}
CacheStats LegalCache::stats() const {
    const std::lock_guard lock(mutex_);
    auto result = stats_;
    result.charged_bytes = sizeof(LegalCache) + memory_.charged();
    return result;
}
}  // namespace mouse::nn