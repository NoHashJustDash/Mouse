// SPDX-License-Identifier: GPL-3.0-or-later
#include <atomic>
#include <bit>
#include <catch2/catch_test_macros.hpp>
#include <limits>
#include <thread>

#include "engine/chess/attacks.h"
#include "engine/nn/history.h"
#include "engine/nn/legal_evaluator.h"
#include "engine/search/search.h"
#include "tests/required_optional.h"
namespace {
using namespace mouse::nn;
InputPlanes input(std::uint64_t tag) {
    InputPlanes p{};
    p[0].mask = tag;
    p[13].value = static_cast<float>(tag);
    return p;
}
PolicyIndices policy() {
    PolicyIndices p;
    p.size = 3;
    p.values[0] = 3;
    p.values[1] = 1857;
    p.values[2] = 7;
    return p;
}
LegalOutput result() {
    LegalOutput r;
    r.logits[0] = 0.123456791F;
    r.logits[1] = -1024.00012F;
    r.logits[2] = 0.000000001F;
    r.wdl = {0.2F, 0.3F, 0.5F};
    r.moves_left = 10.125F;
    return r;
}
std::unique_ptr<LegalCache> cache(std::size_t capacity) {
    auto c = LegalCache::create(capacity, "legal-test");
    REQUIRE(c);
    return std::move(*c);
}
struct Provider final : Backend {
    ModelMetadata model;
    BackendInfo identity;
    std::size_t calls = 0, rows = 0;
    bool bad = false, partial = false, fail = false;
    bool out_of_range = false;
    const InputPlanes* submitted = nullptr;
    Provider() {
        model.input_format = 1;
        identity.cache_namespace = "legal-test";
    }
    [[nodiscard]] const ModelMetadata& metadata() const noexcept override { return model; }
    [[nodiscard]] const BackendInfo& info() const noexcept override { return identity; }
    mouse::Result<void> evaluate(std::span<const InputPlanes> in,
                                 std::span<NNOutput> out) override {
        ++calls;
        rows += in.size();
        submitted = in.data();
        for (std::size_t i = 0; i < in.size(); ++i) {
            if (partial && i + 1 == in.size()) continue;
            out[i] = {};
            out[i].wdl = {0.2F, 0.3F, 0.5F};
            out[i].moves_left = 10.125F;
            for (std::size_t j = 0; j < 1858; ++j)
                out[i].policy_logits.at(j) =
                    static_cast<float>(in[i][0].mask) + static_cast<float>(j) * 0.123456791F;
            if (bad) out[i].policy_logits[1856] = std::numeric_limits<float>::quiet_NaN();
            if (out_of_range && i + 1 == in.size()) out[i].wdl[1] = 1.5F;
        }
        if (fail) return tl::unexpected(std::string{"provider failure"});
        return {};
    }
};
}  // namespace
TEST_CASE("Legal cache retains exact input history and legal float bits", "[nn][legal-cache]") {
    auto c = cache(4);
    auto p = policy();
    auto key = input(1);
    const auto r = result();
    REQUIRE(c->insert(key, p, r));
    auto found = c->lookup(key, p);
    REQUIRE(found);
    REQUIRE(*found);
    CHECK(mouse::tests::required_optional(*found) == r);
    for (std::size_t i = 0; i < key.size(); ++i) {
        auto changed = key;
        changed.at(i).mask ^= 1;
        auto hit = c->lookup(changed, p);
        REQUIRE(hit);
        CHECK_FALSE(*hit);
        changed = key;
        changed.at(i).value += 1.0F;
        hit = c->lookup(changed, p);
        REQUIRE(hit);
        CHECK_FALSE(*hit);
    }
    key[0].value = -0.0F;
    auto miss = c->lookup(key, p);
    REQUIRE(miss);
    CHECK_FALSE(*miss);
    key = input(1);
    std::swap(p.values[0], p.values[1]);
    miss = c->lookup(key, p);
    REQUIRE(miss);
    CHECK_FALSE(*miss);
    CHECK(c->stats().entries == 1);
}
TEST_CASE("Legal cache capacity budgets LRU eviction and reused storage agree",
          "[nn][legal-cache][cache-capacity-v1]") {
    auto c = cache(2);
    auto p = policy();
    REQUIRE(c->insert(input(1), p, result()));
    REQUIRE(c->insert(input(2), p, result()));
    auto first = c->lookup(input(1), p);
    REQUIRE(first);
    REQUIRE(*first);
    REQUIRE(c->insert(input(3), p, result()));
    auto evicted = c->lookup(input(2), p);
    REQUIRE(evicted);
    CHECK_FALSE(*evicted);
    CHECK(c->stats().evictions == 1);
    CHECK(c->stats().capacity == 2);
    const auto budget = LegalCache::budget_for_entries(2, 10);
    REQUIRE(budget);
    CHECK(c->stats().budget_bytes == *budget);
    CHECK(c->stats().charged_bytes <= *budget);
    const auto capacity = LegalCache::capacity_for_budget({.bytes = *budget, .identity_bytes = 10});
    REQUIRE(capacity);
    CHECK(*capacity == 2);
    for (unsigned repeat = 0; repeat < 20; ++repeat) {
        c->clear();
        REQUIRE(c->insert(input(repeat), p, result()));
        CHECK(c->stats().entries == 1);
    }
    CHECK_FALSE(LegalCache::budget_for_entries(std::numeric_limits<std::size_t>::max(), 1));
    CHECK_FALSE(LegalCache::budget_for_entries(1, std::numeric_limits<std::size_t>::max()));
    CHECK_FALSE(LegalCache::create(1, ""));
    CHECK_FALSE(LegalCache::capacity_for_budget({.bytes = 1, .identity_bytes = 10}));
    auto disabled = cache(0);
    REQUIRE(disabled->insert(input(1), p, result()));
    CHECK(disabled->stats().charged_bytes == sizeof(LegalCache));
    CHECK(disabled->stats().entries == 0);
}
TEST_CASE("Legal cache invalid arguments preserve all state", "[nn][legal-cache]") {
    auto c = cache(2);
    const auto p = policy();
    auto key = input(1);
    auto r = result();
    REQUIRE(c->insert(key, p, r));
    const auto before = c->stats();
    key[111].value = std::numeric_limits<float>::infinity();
    CHECK_FALSE(c->insert(key, p, r));
    CHECK_FALSE(c->lookup(key, p));
    auto duplicate = p;
    duplicate.values[1] = duplicate.values[0];
    CHECK_FALSE(c->lookup(input(1), duplicate));
    duplicate = p;
    duplicate.values[0] = 1858;
    CHECK_FALSE(c->lookup(input(1), duplicate));
    duplicate.size = 433;
    CHECK_FALSE(c->lookup(input(1), duplicate));
    r.logits[1] = std::numeric_limits<float>::quiet_NaN();
    CHECK_FALSE(c->insert(input(1), p, r));
    r = result();
    r.wdl = {};
    CHECK_FALSE(c->insert(input(1), p, r));
    r = result();
    r.moves_left = -1.0F;
    CHECK_FALSE(c->insert(input(1), p, r));
    CHECK(c->stats() == before);
}
TEST_CASE("Legal evaluator validates entire physical rows and publishes complete batches",
          "[nn][legal-cache]") {
    Provider provider;
    auto c = cache(4);
    LegalEvaluator evaluator(provider, 4, c.get());
    const std::array inputs{input(1), input(1), input(2)};
    std::array policies{policy(), policy(), policy()};
    policies[1].values[0] = 11;
    std::array<LegalOutput, 3> outputs{};
    REQUIRE(evaluator.evaluate(inputs, policies, outputs));
    CHECK(provider.calls == 1);
    CHECK(provider.rows == 2);
    for (std::size_t row = 0; row < inputs.size(); ++row)
        for (std::size_t j = 0; j < policies.at(row).size; ++j)
            CHECK(outputs.at(row).logits.at(j) ==
                  static_cast<float>(inputs.at(row)[0].mask) +
                      static_cast<float>(policies.at(row).values.at(j)) * 0.123456791F);
    const auto expected = outputs;
    REQUIRE(evaluator.evaluate(inputs, policies, outputs));
    CHECK(outputs == expected);
    CHECK(provider.calls == 1);
    for (const unsigned mode : {0U, 1U, 2U}) {
        provider.bad = mode == 0;
        provider.partial = mode == 1;
        provider.fail = mode == 2;
        c->clear();
        outputs = expected;
        CHECK_FALSE(evaluator.evaluate(inputs, policies, outputs));
        CHECK(outputs == expected);
        CHECK(c->stats().entries == 0);
    }
    CHECK_THROWS_AS(LegalEvaluator(provider, 0), std::invalid_argument);
    auto wrong = LegalCache::create(1, "other-model");
    REQUIRE(wrong);
    CHECK_THROWS_AS(LegalEvaluator(provider, 4, wrong->get()), std::invalid_argument);
}

TEST_CASE("Legal cache and raw backend preserve deterministic node visitation",
          "[search][legal-cache]") {
    const mouse::chess::AttackTables attacks;
    auto position = mouse::chess::Position::from_fen(mouse::chess::kStartFen, attacks);
    REQUIRE(position);
    mouse::nn::EncodingHistory history;
    history.reset(*position);
    Provider raw_provider, cached_provider;
    auto c = cache(1024);
    const mouse::search::SearchConfig config{.selection = {},
                                             .threads = 1,
                                             .maximum_batch = 4,
                                             .ram_limit_bytes = 8U << 20U,
                                             .deterministic = true,
                                             .moves_left = {},
                                             .draw_rate = {},
                                             .gather = {},
                                             .tablebase = {},
                                             .temperature = {}};
    auto raw = mouse::search::SearchEngine::create(config, attacks, raw_provider);
    auto cached = mouse::search::SearchEngine::create(config, attacks, cached_provider, c.get());
    REQUIRE(raw);
    REQUIRE(cached);
    for (unsigned repeat = 0; repeat < 2; ++repeat) {
        REQUIRE((*raw)->set_position(*position, history, false));
        REQUIRE((*cached)->set_position(*position, history, false));
        mouse::search::SearchLimits limits;
        limits.nodes = 256;
        limits.smart_pruning = false;
        const auto before = (*raw)->run(limits), after = (*cached)->run(limits);
        REQUIRE(before);
        REQUIRE(after);
        CHECK(before->best_move == after->best_move);
        CHECK(before->reason == after->reason);
        CHECK(before->info.nodes == after->info.nodes);
        CHECK(before->info.q == after->info.q);
        CHECK(before->info.draw == after->info.draw);
        const auto a = (*raw)->root_moves(), b = (*cached)->root_moves();
        REQUIRE(a.size == b.size);
        for (std::size_t i = 0; i < a.size; ++i) {
            CHECK(a.moves.at(i).move == b.moves.at(i).move);
            CHECK(a.moves.at(i).visits == b.moves.at(i).visits);
            CHECK(a.moves.at(i).q == b.moves.at(i).q);
            CHECK(a.moves.at(i).prior == b.moves.at(i).prior);
        }
    }
    CHECK(cached_provider.rows < raw_provider.rows);
}

TEST_CASE("Legal cache rejects the historical full 64-bit collision", "[nn][legal-cache]") {
    auto c = cache(4);
    auto a = input(1), b = input(2);
    const auto p = policy();
    // Adversarial FNV fixture: compensate the differing prefix at the last mask.
    // This forces equality before the shared policy suffix and final avalanche.
    auto prefix = [](const InputPlanes& in) {
        constexpr std::uint64_t prime = 0x100000001b3ULL;
        std::uint64_t h = 0xcbf29ce484222325ULL;
        for (const char ch : std::string_view{"legal-test"})
            h = (h ^ static_cast<unsigned char>(ch)) * prime;
        for (std::size_t i = 0; i + 1 < in.size(); ++i) {
            h = (h ^ in.at(i).mask) * prime;
            h = (h ^ std::bit_cast<std::uint32_t>(in.at(i).value)) * prime;
        }
        return h;
    };
    b.back().mask = prefix(a) ^ a.back().mask ^ prefix(b);
    REQUIRE(c->insert(a, p, result()));
    auto miss = c->lookup(b, p);
    REQUIRE(miss);
    CHECK_FALSE(*miss);
    auto alternate = result();
    alternate.logits[0] = -0.0F;
    alternate.moves_left.reset();
    REQUIRE(c->insert(b, p, alternate));
    auto hit = c->lookup(b, p);
    REQUIRE(hit);
    REQUIRE(*hit);
    const auto& value = mouse::tests::required_optional(*hit);
    CHECK(std::bit_cast<std::uint32_t>(value.logits[0]) == std::bit_cast<std::uint32_t>(-0.0F));
    CHECK_FALSE(value.moves_left);
    hit = c->lookup(a, p);
    REQUIRE(hit);
    REQUIRE(*hit);
    CHECK(mouse::tests::required_optional(*hit) == result());
    // The approved 128-bit key separates the historical 64-bit fixture.
    // Also exercise an actual bucket collision without assuming hash internals.
    for (std::uint64_t tag = 3; tag < 4099 && c->stats().bucket_collisions == 0; ++tag) {
        const auto other = c->lookup(input(tag), p);
        REQUIRE(other);
        CHECK_FALSE(*other);
    }
    CHECK(c->stats().bucket_collisions > 0);
}
TEST_CASE("Legal cache serializes concurrent insert lookup and clear", "[nn][legal-cache]") {
    auto c = cache(16);
    std::atomic<bool> valid{true};
    std::array<std::thread, 4> workers;
    for (std::size_t worker = 0; worker < workers.size(); ++worker)
        workers.at(worker) = std::thread([&, worker] {
            for (std::uint64_t i = 0; i < 1000; ++i) {
                const auto tag = i * 4 + worker;
                const auto key = input(tag);
                auto value = result();
                value.logits[0] = static_cast<float>(tag);
                auto inserted = c->insert(key, policy(), value);
                auto found = c->lookup(key, policy());
                if (!inserted || !found ||
                    (*found && mouse::tests::required_optional(*found) != value))
                    valid = false;
                if (i % 100 == 0) c->clear();
            }
        });
    for (auto& worker : workers) worker.join();
    CHECK(valid.load());
    CHECK(c->stats().entries <= 16);
    CHECK(c->stats().capacity == 16);
}

TEST_CASE("A zero-capacity legal cache preserves the uncached physical batch",
          "[nn][legal-cache]") {
    Provider provider;
    auto disabled = cache(0);
    LegalEvaluator evaluator(provider, 4, disabled.get());
    const std::array inputs{input(1), input(1), input(2)};
    const std::array policies{policy(), policy(), policy()};
    std::array<LegalOutput, 3> outputs{};
    REQUIRE(evaluator.evaluate(inputs, policies, outputs));
    CHECK(provider.rows == 3);
    REQUIRE(evaluator.evaluate(inputs, policies, outputs));
    CHECK(provider.rows == 6);
    CHECK(provider.calls == 2);
}

TEST_CASE("A cache-off legal evaluator submits caller rows in order without copying them",
          "[nn][legal-cache]") {
    Provider provider;
    LegalEvaluator evaluator(provider, 4);
    const std::array inputs{input(3), input(1), input(3), input(2)};
    std::array policies{policy(), policy(), policy(), policy()};
    policies[2].values[0] = 11;
    const auto expect_rows = [&](std::span<const LegalOutput> outputs, std::size_t first) {
        for (std::size_t row = 0; row < outputs.size(); ++row) {
            const auto& legal = policies.at(first + row);
            for (std::size_t j = 0; j < legal.size; ++j)
                CHECK(outputs[row].logits.at(j) ==
                      static_cast<float>(inputs.at(first + row)[0].mask) +
                          static_cast<float>(legal.values.at(j)) * 0.123456791F);
        }
    };
    std::array<LegalOutput, 4> outputs{};
    REQUIRE(evaluator.evaluate(inputs, policies, outputs));
    // Duplicates stay separate rows and the provider reads the caller's own storage.
    CHECK(provider.calls == 1);
    CHECK(provider.rows == 4);
    CHECK(provider.submitted == inputs.data());
    expect_rows(outputs, 0);
    // A smaller later batch returns only its own rows and leaves the rest unchanged.
    auto smaller = outputs;
    REQUIRE(evaluator.evaluate(std::span{inputs}.subspan(1, 2), std::span{policies}.subspan(1, 2),
                               std::span{smaller}.first(2)));
    CHECK(provider.submitted == inputs.data() + 1);
    CHECK(provider.rows == 6);
    expect_rows(std::span{smaller}.first(2), 1);
    CHECK(smaller[2] == outputs[2]);
    CHECK(smaller[3] == outputs[3]);
}

TEST_CASE("A cache-off legal evaluator rejects invalid or incomplete provider rows",
          "[nn][legal-cache]") {
    Provider provider;
    LegalEvaluator evaluator(provider, 4);
    const std::array inputs{input(1), input(2), input(3)};
    const std::array policies{policy(), policy(), policy()};
    std::array<LegalOutput, 3> outputs{};
    REQUIRE(evaluator.evaluate(inputs, policies, outputs));
    const auto expected = outputs;
    const std::array<LegalOutput, 3> untouched{};
    // NaN in an illegal slot, an unwritten last row, a provider error and an
    // out-of-range WDL value in the last row each fail the whole batch unpublished.
    for (const unsigned mode : {0U, 1U, 2U, 3U}) {
        INFO("mode " << mode);
        provider.bad = mode == 0;
        provider.partial = mode == 1;
        provider.fail = mode == 2;
        provider.out_of_range = mode == 3;
        std::array<LegalOutput, 3> changed{};
        CHECK_FALSE(evaluator.evaluate(inputs, policies, changed));
        CHECK(changed == untouched);
    }
    provider.out_of_range = false;
    std::array<LegalOutput, 3> recovered{};
    REQUIRE(evaluator.evaluate(inputs, policies, recovered));
    CHECK(recovered == expected);
}

TEST_CASE("Legal cache capacity allocates entries and legal payloads on demand",
          "[nn][legal-cache][cache-memory-v2]") {
    constexpr std::size_t capacity = 10000;
    auto narrow = cache(capacity);
    auto wide = cache(capacity);
    const auto empty = narrow->stats();
    CHECK(empty.capacity == capacity);
    CHECK(empty.entries == 0);
    CHECK(empty.charged_bytes <= std::size_t{64} * 1024);
    CHECK(wide->stats().charged_bytes == empty.charged_bytes);
    auto default_capacity = cache(2'000'000);
    CHECK(default_capacity->stats().capacity == 2'000'000);
    CHECK(default_capacity->stats().charged_bytes == empty.charged_bytes);
    const auto small_policy = policy();
    PolicyIndices large_policy;
    large_policy.size = 128;
    for (std::size_t i = 0; i < large_policy.size; ++i)
        large_policy.values.at(i) = static_cast<std::uint16_t>(i);
    const auto value = result();
    for (std::uint64_t i = 0; i < 1000; ++i) {
        REQUIRE(narrow->insert(input(i), small_policy, value));
        REQUIRE(wide->insert(input(i), large_policy, value));
    }
    const auto small = narrow->stats(), large = wide->stats();
    CHECK(small.entries == 1000);
    CHECK(small.capacity == capacity);
    CHECK(small.charged_bytes > empty.charged_bytes);
    CHECK(small.charged_bytes - empty.charged_bytes <= std::size_t{1024} * 1024);
    CHECK(large.charged_bytes > small.charged_bytes);
    CHECK(small.charged_bytes <= small.budget_bytes);
    CHECK(large.charged_bytes <= large.budget_bytes);
    const auto hit = narrow->lookup(input(500), small_policy);
    REQUIRE(hit);
    REQUIRE(*hit);
    CHECK(mouse::tests::required_optional(*hit) == value);
    narrow->clear();
    CHECK(narrow->stats().entries == 0);
    CHECK(narrow->stats().capacity == capacity);
    CHECK(narrow->stats().charged_bytes <= empty.charged_bytes + std::size_t{16} * 1024);
}

TEST_CASE("Legal cache history length keys only the configured earlier frames",
          "[nn][legal-cache][cache-history]") {
    // Classical layout: frame 0 is planes 0..12, frame 1 planes 13..25, and the
    // auxiliary planes (castling, side to move, rule50) start at 104.
    auto current = LegalCache::create(4, "legal-test", 0);
    auto two = LegalCache::create(4, "legal-test", 2);
    auto full = LegalCache::create(4, "legal-test");
    REQUIRE(current);
    REQUIRE(two);
    REQUIRE(full);
    CHECK((*full)->history_length() == LegalCache::kFullCacheHistory);
    CHECK_FALSE(LegalCache::create(4, "legal-test", 8));
    const auto p = policy();
    const auto r = result();
    InputPlanes stored{};
    stored[0].mask = 0x10;
    stored[13].mask = 0x20;
    stored[50].mask = 0x40;
    stored[109].value = 7.0F;
    for (auto* c : {current->get(), two->get(), full->get()}) REQUIRE(c->insert(stored, p, r));
    const auto hit = [&](LegalCache& c, const InputPlanes& key) {
        auto found = c.lookup(key, p);
        REQUIRE(found);
        return found->has_value();
    };
    auto other_history = stored;
    other_history[13].mask = 0x21;  // frame 1 differs
    CHECK(hit(**current, other_history));
    CHECK_FALSE(hit(**two, other_history));
    CHECK_FALSE(hit(**full, other_history));
    auto old_history = stored;
    old_history[50].mask = 0x41;  // frame 3 differs
    CHECK(hit(**current, old_history));
    CHECK(hit(**two, old_history));
    CHECK_FALSE(hit(**full, old_history));
    auto other_position = stored;
    other_position[0].mask = 0x11;
    CHECK_FALSE(hit(**current, other_position));
    auto other_rule50 = stored;
    other_rule50[109].value = 8.0F;  // lc0 HashLast also mixes in rule50
    CHECK_FALSE(hit(**current, other_rule50));
}

TEST_CASE("Legal cache can key on the board and repetitions without rule 50 as Lc0 does",
          "[nn][legal-cache][cache-history]") {
    // Lc0 v0.32.1 neural/memcache.cc ComputeEvalPositionHash keys on
    // chess/position.cc Position::Hash(): board (castling and side to move
    // included) plus repetitions, but not rule 50.
    auto lc0 =
        LegalCache::create(4, "legal-test", 0, {.key_rule50 = false, .reserve_index = false});
    auto strict =
        LegalCache::create(4, "legal-test", 0, {.key_rule50 = true, .reserve_index = false});
    auto full = LegalCache::create(4, "legal-test");
    REQUIRE(lc0);
    REQUIRE(strict);
    REQUIRE(full);
    CHECK_FALSE((*lc0)->keys_rule50());
    CHECK((*lc0)->history_length() == 0);
    CHECK((*strict)->keys_rule50());
    CHECK((*full)->keys_rule50());
    CHECK((*full)->history_length() == LegalCache::kFullCacheHistory);
    CHECK_FALSE(LegalCache::create(4, "legal-test", LegalCache::kFullCacheHistory,
                                   {.key_rule50 = false, .reserve_index = false}));

    // Encode one position at two 50-move counts so the test pins the plane the
    // real encoder uses for rule 50, not an assumed index.
    const mouse::chess::AttackTables attacks;
    const auto encode = [&](std::string_view fen) {
        const auto position = mouse::chess::Position::from_fen(fen, attacks);
        REQUIRE(position);
        const std::array frames{HistoryFrame::capture(*position)};
        auto planes = encode_classical(frames, 1, HistoryFill::No);
        REQUIRE(planes);
        return *planes;
    };
    const auto early = encode("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 40");
    const auto late = encode("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 37 40");
    for (std::size_t plane = 0; plane < early.size(); ++plane)
        CHECK((early.at(plane) == late.at(plane)) == (plane != 109));

    const auto p = policy();
    const auto r = result();
    for (auto* c : {lc0->get(), strict->get(), full->get()}) REQUIRE(c->insert(early, p, r));
    const auto hit = [&](LegalCache& c, const InputPlanes& key) {
        auto found = c.lookup(key, p);
        REQUIRE(found);
        return found->has_value();
    };
    CHECK(hit(**lc0, late));
    CHECK_FALSE(hit(**strict, late));
    CHECK_FALSE(hit(**full, late));
    const auto served = (*lc0)->lookup(late, p);
    REQUIRE(served);
    REQUIRE(*served);
    CHECK(mouse::tests::required_optional(*served) == r);

    // Frame-0 repetition (12), castling (104..107) and side to move (108)
    // are part of Lc0's key, so a change to any of them must miss.
    const auto flip = [](InputPlanes key, std::size_t plane) {
        key.at(plane).mask ^= std::numeric_limits<std::uint64_t>::max();
        return key;
    };
    for (const std::size_t plane : {std::size_t{12}, std::size_t{104}, std::size_t{105},
                                    std::size_t{106}, std::size_t{107}, std::size_t{108}}) {
        CHECK_FALSE(hit(**lc0, flip(late, plane)));
        CHECK_FALSE(hit(**strict, flip(early, plane)));
        CHECK_FALSE(hit(**full, flip(early, plane)));
    }
    CHECK((*lc0)->stats().entries == 1);
}

TEST_CASE("A reserved legal cache index never rehashes while filling",
          "[nn][legal-cache][cache-memory-v2]") {
    constexpr std::size_t capacity = 1000;
    auto reserved = LegalCache::create(capacity, "legal-test", LegalCache::kFullCacheHistory,
                                       {.key_rule50 = true, .reserve_index = true});
    auto lazy = LegalCache::create(capacity, "legal-test");
    REQUIRE(reserved);
    REQUIRE(lazy);
    CHECK((*reserved)->reserves_index());
    CHECK_FALSE((*lazy)->reserves_index());
    const auto buckets = (*reserved)->index_bucket_count();
    const auto lazy_buckets = (*lazy)->index_bucket_count();
    // insert() holds capacity + 1 keys before evicting, so that many must fit.
    CHECK(buckets >= capacity + 1);
    CHECK((*reserved)->stats().charged_bytes <= (*reserved)->stats().budget_bytes);
    const auto p = policy();
    const auto r = result();
    bool stable = true;
    for (std::uint64_t i = 0; i < capacity + 100; ++i) {
        REQUIRE((*reserved)->insert(input(i), p, r));
        REQUIRE((*lazy)->insert(input(i), p, r));
        stable = stable && (*reserved)->index_bucket_count() == buckets;
    }
    CHECK(stable);
    // The unreserved index grew over the same inserts, so the check above
    // would fail without the reserve.
    CHECK((*lazy)->index_bucket_count() > lazy_buckets);
    const auto filled = (*reserved)->stats();
    CHECK(filled.entries == capacity);
    CHECK(filled.evictions == 100);
    CHECK(filled.charged_bytes <= filled.budget_bytes);
    (*reserved)->clear();
    CHECK((*reserved)->index_bucket_count() == buckets);
    for (std::uint64_t i = 0; i < capacity; ++i) REQUIRE((*reserved)->insert(input(i), p, r));
    CHECK((*reserved)->index_bucket_count() == buckets);

    auto disabled = LegalCache::create(0, "legal-test", LegalCache::kFullCacheHistory,
                                       {.key_rule50 = true, .reserve_index = true});
    REQUIRE(disabled);
    CHECK((*disabled)->index_bucket_count() == 0);
    CHECK((*disabled)->stats().charged_bytes == sizeof(LegalCache));
}
