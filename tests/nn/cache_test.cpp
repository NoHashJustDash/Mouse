// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/nn/cache.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "tests/required_optional.h"

namespace {
using namespace mouse::nn;
using mouse::tests::required_optional;
constexpr std::string_view kIdentity = "onnx:fixture-a|provider:cpu|precision:fp32";

InputPlanes key(std::size_t identifier) {
    InputPlanes result{};
    result[0].mask = static_cast<std::uint64_t>(identifier) + 1;
    result[13].mask = static_cast<std::uint64_t>(identifier) * 0x9e3779b97f4a7c15ULL;
    result[109].mask = std::numeric_limits<std::uint64_t>::max();
    result[109].value = static_cast<float>(identifier % 100);
    return result;
}
NNOutput evaluation(float tag) {
    NNOutput output;
    output.policy_logits.fill(tag);
    output.wdl = {0.25F, 0.5F, 0.25F};
    output.moves_left = tag;
    return output;
}
std::unique_ptr<NNCache> cache_with_capacity(std::size_t required) {
    std::size_t upper = 4096;
    for (;;) {
        auto candidate = NNCache::create(upper, kIdentity);
        if (candidate && (*candidate)->stats().capacity >= required) break;
        REQUIRE(upper < 64ULL * 1024ULL * 1024ULL);
        upper *= 2;
    }
    std::size_t lower = 1;
    while (lower < upper) {
        const auto middle = lower + (upper - lower) / 2;
        auto candidate = NNCache::create(middle, kIdentity);
        if (candidate && (*candidate)->stats().capacity >= required)
            upper = middle;
        else
            lower = middle + 1;
    }
    auto result = NNCache::create(lower, kIdentity);
    REQUIRE(result.has_value());
    REQUIRE((*result)->stats().capacity == required);
    return std::move(*result);
}
std::optional<NNOutput> lookup(NNCache& cache, const InputPlanes& input) {
    auto result = cache.lookup(input);
    if (!result) FAIL(result.error());
    return *result;
}
}  // namespace

TEST_CASE("NN cache zero budget disables every payload and index allocation", "[nn][cache]") {
    auto result = NNCache::create(0, kIdentity);
    REQUIRE(result.has_value());
    auto& cache = **result;
    const auto before = cache.stats();
    CHECK(before.budget_bytes == 0);
    CHECK(before.charged_bytes == sizeof(NNCache));
    CHECK(before.capacity == 0);
    CHECK(before.entries == 0);
    REQUIRE(cache.insert(key(0), evaluation(1)).has_value());
    CHECK_FALSE(lookup(cache, key(0)).has_value());
    cache.clear();
    const auto after = cache.stats();
    CHECK(after.capacity == 0);
    CHECK(after.entries == 0);
    CHECK(after.charged_bytes == sizeof(NNCache));
    CHECK(after.misses == 1);
}

TEST_CASE("NN cache construction checks identity and exact memory boundaries", "[nn][cache]") {
    CHECK_FALSE(NNCache::create(0, "").has_value());
    CHECK_FALSE(NNCache::create(sizeof(NNCache), kIdentity).has_value());
    CHECK_FALSE(NNCache::create(1024, std::string(2048, 'x')).has_value());
    CHECK_FALSE(NNCache::create(std::numeric_limits<std::size_t>::max(), kIdentity).has_value());
    auto cache = cache_with_capacity(1);
    const auto stats = cache->stats();
    CHECK(stats.charged_bytes <= stats.budget_bytes);
    CHECK(stats.charged_bytes >= sizeof(NNCache) + kIdentity.size() + sizeof(InputPlanes) +
                                     sizeof(NNOutput) + 3ULL * 64ULL);
    CHECK_FALSE(NNCache::create(stats.budget_bytes - 1, kIdentity).has_value());
    const auto budget = stats.budget_bytes;
    for (std::size_t index = 0; index < 1000; ++index) {
        REQUIRE(cache->insert(key(index), evaluation(static_cast<float>(index))).has_value());
        REQUIRE(lookup(*cache, key(index)).has_value());
    }
    CHECK(cache->stats().charged_bytes == stats.charged_bytes);
    CHECK(cache->stats().budget_bytes == budget);
    CHECK(cache->stats().entries == 1);
    CHECK(cache->stats().evictions == 999);
}

TEST_CASE("NN cache returns independent copies of complete raw outputs", "[nn][cache]") {
    auto cache = cache_with_capacity(2);
    auto output = evaluation(17);
    for (std::size_t index = 0; index < output.policy_logits.size(); ++index)
        output.policy_logits.at(index) = static_cast<float>(index) / 2000.0F;
    const auto expected = output;
    REQUIRE(cache->insert(key(0), output).has_value());
    output.policy_logits[0] = -100;
    auto copied = lookup(*cache, key(0));
    REQUIRE(copied.has_value());
    CHECK(required_optional(copied) == expected);
    if (copied.has_value()) copied->policy_logits[1] = -200;
    CHECK(required_optional(lookup(*cache, key(0))) == expected);
    cache->clear();
    CHECK(required_optional(copied) != evaluation(5));
    CHECK_FALSE(lookup(*cache, key(0)).has_value());
}

TEST_CASE("NN cache evicts exact LRU order and refreshes updates", "[nn][cache]") {
    auto cache = cache_with_capacity(2);
    REQUIRE(cache->insert(key(0), evaluation(1)).has_value());
    REQUIRE(cache->insert(key(1), evaluation(2)).has_value());
    REQUIRE(lookup(*cache, key(0)).has_value());
    REQUIRE(cache->insert(key(2), evaluation(3)).has_value());
    CHECK_FALSE(lookup(*cache, key(1)).has_value());
    CHECK(required_optional(lookup(*cache, key(0))) == evaluation(1));
    REQUIRE(cache->insert(key(0), evaluation(4)).has_value());
    REQUIRE(cache->insert(key(3), evaluation(5)).has_value());
    CHECK_FALSE(lookup(*cache, key(2)).has_value());
    CHECK(required_optional(lookup(*cache, key(0))) == evaluation(4));
    CHECK(required_optional(lookup(*cache, key(3))) == evaluation(5));
    CHECK(cache->stats().entries == 2);
    CHECK(cache->stats().evictions == 2);
    const auto before = cache->stats();
    cache->clear();
    CHECK(cache->stats().entries == 0);
    CHECK(cache->stats().hits == before.hits);
    CHECK(cache->stats().misses == before.misses);
    CHECK(cache->stats().charged_bytes == before.charged_bytes);
    REQUIRE(cache->insert(key(1), evaluation(6)).has_value());
    CHECK(required_optional(lookup(*cache, key(1))) == evaluation(6));
}

TEST_CASE("NN cache compares all masks values history and raw rule50", "[nn][cache]") {
    auto cache = cache_with_capacity(2);
    const auto original = key(0);
    REQUIRE(cache->insert(original, evaluation(1)).has_value());
    for (std::size_t index = 0; index < original.size(); ++index) {
        auto changed = original;
        changed.at(index).mask ^= 1;
        CHECK_FALSE(lookup(*cache, changed).has_value());
        changed = original;
        changed.at(index).value += 1.0F;
        CHECK_FALSE(lookup(*cache, changed).has_value());
    }
    auto positive_zero = original;
    auto negative_zero = original;
    positive_zero[13].value = 0.0F;
    negative_zero[13].value = -0.0F;
    REQUIRE(cache->insert(positive_zero, evaluation(2)).has_value());
    CHECK_FALSE(lookup(*cache, negative_zero).has_value());
    REQUIRE(cache->insert(negative_zero, evaluation(3)).has_value());
    CHECK(required_optional(lookup(*cache, positive_zero)) == evaluation(2));
    CHECK(required_optional(lookup(*cache, negative_zero)) == evaluation(3));
}

TEST_CASE("NN cache resolves real bucket collisions and removes chain entries on eviction",
          "[nn][cache]") {
    auto cache = cache_with_capacity(64);
    for (std::size_t index = 0; index < 1000; ++index)
        REQUIRE(cache->insert(key(index), evaluation(static_cast<float>(index))).has_value());
    CHECK(cache->stats().bucket_collisions > 0);
    for (std::size_t index = 936; index < 1000; ++index) {
        auto found = lookup(*cache, key(index));
        REQUIRE(found.has_value());
        CHECK(required_optional(found) == evaluation(static_cast<float>(index)));
    }
    for (std::size_t index = 0; index < 936; ++index)
        CHECK_FALSE(lookup(*cache, key(index)).has_value());
    CHECK(cache->stats().entries == 64);
    CHECK(cache->stats().evictions == 936);
}

TEST_CASE("NN cache rejects nonfinite inputs before changing data or counters", "[nn][cache]") {
    auto cache = cache_with_capacity(1);
    const auto original = key(0);
    REQUIRE(cache->insert(original, evaluation(1)).has_value());
    const auto before = cache->stats();
    constexpr std::array invalid{std::numeric_limits<float>::quiet_NaN(),
                                 std::numeric_limits<float>::infinity(),
                                 -std::numeric_limits<float>::infinity()};
    for (std::size_t index = 0; index < original.size(); ++index) {
        for (float value : invalid) {
            auto changed = original;
            changed.at(index).value = value;
            CHECK_FALSE(cache->lookup(changed).has_value());
            CHECK_FALSE(cache->insert(changed, evaluation(2)).has_value());
            CHECK(cache->stats() == before);
        }
    }
    CHECK(required_optional(lookup(*cache, original)) == evaluation(1));
}

TEST_CASE("NN cache invalid outputs cannot replace an existing value", "[nn][cache]") {
    auto cache = cache_with_capacity(1);
    REQUIRE(cache->insert(key(0), evaluation(1)).has_value());
    const auto before = cache->stats();
    for (std::size_t index = 0; index < NNOutput{}.policy_logits.size(); ++index) {
        auto invalid = evaluation(2);
        invalid.policy_logits.at(index) = std::numeric_limits<float>::quiet_NaN();
        CHECK_FALSE(cache->insert(key(0), invalid).has_value());
    }
    for (float value : {-0.1F, 1.1F, std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::quiet_NaN()}) {
        for (std::size_t index = 0; index < 3; ++index) {
            auto invalid = evaluation(2);
            invalid.wdl.at(index) = value;
            CHECK_FALSE(cache->insert(key(0), invalid).has_value());
        }
    }
    auto invalid = evaluation(2);
    invalid.wdl = {};
    CHECK_FALSE(cache->insert(key(0), invalid).has_value());
    for (float value :
         {-1.0F, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        invalid = evaluation(2);
        invalid.moves_left = value;
        CHECK_FALSE(cache->insert(key(0), invalid).has_value());
    }
    CHECK(cache->stats() == before);
    CHECK(required_optional(lookup(*cache, key(0))) == evaluation(1));
    auto valid = evaluation(2);
    valid.wdl = {1.0F, 1.0F, 1.0F};
    valid.moves_left.reset();
    REQUIRE(cache->insert(key(0), valid).has_value());
    CHECK(required_optional(lookup(*cache, key(0))) == valid);
}

TEST_CASE("NN cache namespace is copied once and caches remain isolated", "[nn][cache]") {
    std::string identity = "onnx:fixture-a|provider:cuda|precision:fp16";
    auto first = NNCache::create(32768, identity);
    REQUIRE(first.has_value());
    identity.assign("onnx:fixture-b|provider:trt|precision:fp16");
    auto second = NNCache::create(32768, identity);
    REQUIRE(second.has_value());
    REQUIRE((*first)->insert(key(0), evaluation(1)).has_value());
    CHECK_FALSE(lookup(**second, key(0)).has_value());
    REQUIRE((*second)->insert(key(0), evaluation(2)).has_value());
    CHECK(required_optional(lookup(**first, key(0))) == evaluation(1));
    CHECK(required_optional(lookup(**second, key(0))) == evaluation(2));
}

TEST_CASE("NN cache serializes simultaneous writers and returns untorn copies", "[nn][cache]") {
    auto cache = cache_with_capacity(128);
    std::atomic<bool> failed{false};
    constexpr std::size_t workers = 8;
    constexpr std::size_t iterations = 500;
    std::vector<std::jthread> threads;
    threads.reserve(workers);
    for (std::size_t worker = 0; worker < workers; ++worker) {
        threads.emplace_back([&, worker] {
            const auto output = evaluation(static_cast<float>(worker + 1));
            for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
                const auto input = key(worker * 16 + iteration % 16);
                if (!cache->insert(input, output)) {
                    failed.store(true);
                    return;
                }
                const auto found = cache->lookup(input);
                if (!found || !*found || **found != output) {
                    failed.store(true);
                    return;
                }
            }
        });
    }
    threads.clear();
    CHECK_FALSE(failed.load());
    const auto stats = cache->stats();
    CHECK(stats.hits == workers * iterations);
    CHECK(stats.misses == 0);
    CHECK(stats.entries == 128);
    CHECK(stats.evictions == 0);
}

TEST_CASE("NN cache clear races safely with shared-key readers and writers", "[nn][cache]") {
    auto cache = cache_with_capacity(2);
    std::atomic<bool> failed{false};
    constexpr std::size_t iterations = 500;
    std::vector<std::jthread> threads;
    threads.emplace_back([&] {
        for (std::size_t index = 0; index < iterations; ++index) cache->clear();
    });
    for (std::size_t worker = 0; worker < 4; ++worker) {
        threads.emplace_back([&, worker] {
            for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
                if (!cache->insert(key(0), evaluation(static_cast<float>(worker + 1)))) {
                    failed.store(true);
                    return;
                }
                const auto found = cache->lookup(key(0));
                if (!found) {
                    failed.store(true);
                    return;
                }
                if (*found) {
                    const auto& output = **found;
                    const float tag = output.policy_logits[0];
                    if (tag < 1 || tag > 4 || output != evaluation(tag)) {
                        failed.store(true);
                        return;
                    }
                }
            }
        });
    }
    threads.clear();
    CHECK_FALSE(failed.load());
    const auto stats = cache->stats();
    CHECK(stats.hits + stats.misses == 4 * iterations);
    CHECK(stats.entries <= stats.capacity);
    CHECK(stats.charged_bytes <= stats.budget_bytes);
}

TEST_CASE("NN cache invalid replacement does not refresh LRU order", "[nn][cache]") {
    auto cache = cache_with_capacity(2);
    REQUIRE(cache->insert(key(0), evaluation(1)).has_value());
    REQUIRE(cache->insert(key(1), evaluation(2)).has_value());
    const auto before = cache->stats();
    auto invalid = evaluation(3);
    invalid.policy_logits[0] = std::numeric_limits<float>::infinity();
    CHECK_FALSE(cache->insert(key(0), invalid).has_value());
    CHECK(cache->stats() == before);
    REQUIRE(cache->insert(key(2), evaluation(4)).has_value());
    CHECK_FALSE(lookup(*cache, key(0)).has_value());
    CHECK(required_optional(lookup(*cache, key(1))) == evaluation(2));
    CHECK(required_optional(lookup(*cache, key(2))) == evaluation(4));
}

TEST_CASE("required optional assertions reject absent values", "[nn][cache]") {
    const std::optional<int> absent;
    REQUIRE_THROWS_AS(required_optional(absent), std::logic_error);
    const std::optional<int> present{17};
    CHECK(required_optional(present) == 17);
}
