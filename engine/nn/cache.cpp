// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/nn/cache.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>

namespace mouse::nn {
namespace {
constexpr std::uint64_t kHashOffset = 0xcbf29ce484222325ULL;
constexpr std::uint64_t kHashPrime = 0x100000001b3ULL;
// MSVC's large vector blocks request up to 39 extra bytes for 32-byte
// alignment bookkeeping. Reserve 64 for each of our three allocated buffers.
constexpr std::size_t kBufferAllowance = 64;
constexpr std::size_t kAllocatedBuffers = 3;
constexpr auto kAllocationReserve = kBufferAllowance * kAllocatedBuffers;

constexpr std::uint64_t next_counter(std::uint64_t current) noexcept {
    return current +
           static_cast<std::uint64_t>(current != std::numeric_limits<std::uint64_t>::max());
}
static_assert(next_counter(0) == 1);
static_assert(next_counter(std::numeric_limits<std::uint64_t>::max() - 1) ==
              std::numeric_limits<std::uint64_t>::max());
static_assert(next_counter(std::numeric_limits<std::uint64_t>::max()) ==
              std::numeric_limits<std::uint64_t>::max());

bool finite_input(const InputPlanes& input) {
    return std::ranges::all_of(input,
                               [](const InputPlane& plane) { return std::isfinite(plane.value); });
}
bool same_input(const InputPlanes& first, const InputPlanes& second) {
    return std::ranges::equal(first, second, [](const InputPlane& left, const InputPlane& right) {
        return left.mask == right.mask && std::bit_cast<std::uint32_t>(left.value) ==
                                              std::bit_cast<std::uint32_t>(right.value);
    });
}
}  // namespace

NNCache::NNCache(FactoryToken) {}
NNCache::~NNCache() = default;

Result<std::unique_ptr<NNCache>> NNCache::create(std::size_t budget_bytes,
                                                 std::string_view identity) {
    if (identity.empty()) return tl::unexpected(std::string{"NN cache identity must be nonempty"});
    if (budget_bytes != 0 &&
        (budget_bytes < sizeof(NNCache) || identity.size() > budget_bytes - sizeof(NNCache) ||
         budget_bytes - sizeof(NNCache) - identity.size() < kAllocationReserve)) {
        return tl::unexpected(
            std::string{"NN cache budget cannot fit control, identity and allocation bookkeeping"});
    }
    if (budget_bytes > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
        return tl::unexpected(std::string{"NN cache budget exceeds addressable storage"});
    }
    constexpr auto bytes_per_entry = sizeof(Node) + 2 * sizeof(std::size_t);
    const auto capacity =
        budget_bytes == 0
            ? 0
            : (budget_bytes - sizeof(NNCache) - identity.size() - kAllocationReserve) /
                  bytes_per_entry;
    if (budget_bytes != 0 && capacity == 0) {
        return tl::unexpected(std::string{"NN cache budget cannot fit one complete entry"});
    }
    try {
        auto result = std::make_unique<NNCache>(FactoryToken{});
        result->stats_.budget_bytes = budget_bytes;
        result->stats_.charged_bytes = sizeof(NNCache);
        if (budget_bytes == 0) return result;
        result->stats_.charged_bytes += kAllocationReserve;
        if (capacity > result->nodes_.max_size() || capacity > result->buckets_.max_size() / 2) {
            return tl::unexpected(std::string{"NN cache capacity exceeds container limits"});
        }
        result->identity_.assign(identity.begin(), identity.end());
        result->namespace_seed_ = kHashOffset;
        for (const char symbol : result->identity_) {
            result->namespace_seed_ =
                (result->namespace_seed_ ^ static_cast<unsigned char>(symbol)) * kHashPrime;
        }
        result->nodes_.resize(capacity);
        result->buckets_.assign(2 * capacity, kNone);
        // Charge actual capacities, not requested sizes. Reject implementations
        // that reserve more storage than the configured budget can represent.
        auto charge = [&](std::size_t count, std::size_t width) {
            const auto remaining = budget_bytes - result->stats_.charged_bytes;
            if (count > remaining / width) return false;
            result->stats_.charged_bytes += count * width;
            return true;
        };
        if (!charge(result->identity_.capacity(), sizeof(char)) ||
            !charge(result->nodes_.capacity(), sizeof(Node)) ||
            !charge(result->buckets_.capacity(), sizeof(std::size_t))) {
            return tl::unexpected(
                std::string{"NN cache allocated capacity exceeds its byte budget"});
        }
        result->stats_.capacity = capacity;
        return result;
    } catch (const std::bad_alloc&) {
        return tl::unexpected(std::string{"Cache OOM"});
    } catch (const std::length_error&) {
        return tl::unexpected(std::string{"Cache too large"});
    }
}

std::uint64_t NNCache::hash_input(const InputPlanes& input) const noexcept {
    auto hash = namespace_seed_;
    for (const auto& plane : input) {
        hash = (hash ^ plane.mask) * kHashPrime;
        hash = (hash ^ std::bit_cast<std::uint32_t>(plane.value)) * kHashPrime;
    }
    // Avalanche every bit before bucket selection, including high-rank masks.
    hash ^= hash >> 33;
    hash *= 0xff51afd7ed558ccdULL;
    hash ^= hash >> 33;
    hash *= 0xc4ceb9fe1a85ec53ULL;
    return hash ^ (hash >> 33);
}

std::size_t NNCache::find(std::uint64_t hash, const InputPlanes& input) {
    auto index = buckets_.at(hash % buckets_.size());
    while (index != kNone) {
        const auto& node = nodes_.at(index);
        // The hash only selects candidates. Every hit compares all input bits;
        // even a full 64-bit hash collision cannot return another input's output.
        if (node.hash == hash && same_input(node.input, input)) return index;
        stats_.bucket_collisions = next_counter(stats_.bucket_collisions);
        index = node.bucket_next;
    }
    return kNone;
}

void NNCache::unlink(std::size_t index) {
    auto& node = nodes_.at(index);
    if (node.previous != kNone)
        nodes_.at(node.previous).next = node.next;
    else
        head_ = node.next;
    if (node.next != kNone)
        nodes_.at(node.next).previous = node.previous;
    else
        tail_ = node.previous;
    node.previous = kNone;
    node.next = kNone;
}
void NNCache::link_front(std::size_t index) {
    auto& node = nodes_.at(index);
    node.previous = kNone;
    node.next = head_;
    if (head_ != kNone)
        nodes_.at(head_).previous = index;
    else
        tail_ = index;
    head_ = index;
}
void NNCache::touch(std::size_t index) {
    if (index == head_) return;
    unlink(index);
    link_front(index);
}
void NNCache::remove_bucket(std::size_t index) {
    auto& bucket = buckets_.at(nodes_.at(index).hash % buckets_.size());
    auto current = bucket;
    auto previous = kNone;
    while (current != index) {
        previous = current;
        current = nodes_.at(current).bucket_next;
    }
    if (previous == kNone)
        bucket = nodes_.at(index).bucket_next;
    else
        nodes_.at(previous).bucket_next = nodes_.at(index).bucket_next;
}

Result<std::optional<NNOutput>> NNCache::lookup(const InputPlanes& input) {
    if (!finite_input(input)) {
        return tl::unexpected(std::string{"NN cache input plane values must be finite"});
    }
    const std::lock_guard lock{mutex_};
    if (!nodes_.empty()) {
        const auto index = find(hash_input(input), input);
        if (index != kNone) {
            touch(index);
            stats_.hits = next_counter(stats_.hits);
            return std::optional<NNOutput>{nodes_.at(index).output};
        }
    }
    stats_.misses = next_counter(stats_.misses);
    return std::optional<NNOutput>{};
}

Result<void> NNCache::insert(const InputPlanes& input, const NNOutput& output) {
    if (!finite_input(input)) {
        return tl::unexpected(std::string{"NN cache input plane values must be finite"});
    }
    auto validation = validate_nn_output(output);
    if (!validation) return validation;
    const std::lock_guard lock{mutex_};
    if (nodes_.empty()) return {};
    const auto hash = hash_input(input);
    auto index = find(hash, input);
    if (index != kNone) {
        nodes_.at(index).output = output;
        touch(index);
        return {};
    }
    if (stats_.entries < stats_.capacity) {
        index = stats_.entries;
        ++stats_.entries;
    } else {
        index = tail_;
        remove_bucket(index);
        unlink(index);
        stats_.evictions = next_counter(stats_.evictions);
    }
    auto& node = nodes_.at(index);
    node.input = input;
    node.output = output;
    node.hash = hash;
    auto& bucket = buckets_.at(hash % buckets_.size());
    node.bucket_next = bucket;
    bucket = index;
    link_front(index);
    return {};
}

void NNCache::clear() {
    const std::lock_guard lock{mutex_};
    std::ranges::fill(buckets_, kNone);
    stats_.entries = 0;
    head_ = kNone;
    tail_ = kNone;
}
CacheStats NNCache::stats() const {
    const std::lock_guard lock{mutex_};
    return stats_;
}
}  // namespace mouse::nn
