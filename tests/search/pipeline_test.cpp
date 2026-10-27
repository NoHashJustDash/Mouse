// SPDX-License-Identifier: GPL-3.0-or-later
#include <algorithm>
#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>

#include "engine/chess/attacks.h"
#include "engine/chess/movegen.h"
#include "engine/nn/encoder.h"
#include "engine/nn/legal_cache.h"
#include "engine/nn/policy_map.h"
#include "engine/search/search.h"
#include "tests/required_optional.h"

namespace {
using mouse::tests::required_optional;
struct GateBackend final : mouse::nn::Backend {
    mouse::nn::ModelMetadata model;
    mouse::nn::BackendInfo identity;
    std::mutex mutex;
    std::condition_variable changed;
    mutable std::mutex info_mutex;
    mutable std::condition_variable info_changed;
    mutable bool block_info = false, entered_info = false, released_info = false;
    bool entered = false, released = false, invalid = false, invalid_mlh = false;
    std::atomic<std::size_t> calls{0}, active{0}, maximum_active{0}, largest_batch{0};
    GateBackend() {
        model.input_format = 1;
        identity.onnx_sha256 = "pipeline-test";
    }
    [[nodiscard]] mouse::Result<void> evaluate(std::span<const mouse::nn::InputPlanes> inputs,
                                               std::span<mouse::nn::NNOutput> outputs) override {
        const auto concurrent = ++active;
        auto old = maximum_active.load();
        while (old < concurrent && !maximum_active.compare_exchange_weak(old, concurrent)) {
        }
        auto batch = largest_batch.load();
        while (batch < inputs.size() &&
               !largest_batch.compare_exchange_weak(batch, inputs.size())) {
        }
        const auto call = ++calls;
        if (call == 2) {
            std::unique_lock lock(mutex);
            entered = true;
            changed.notify_all();
            changed.wait(lock, [&] { return released; });
        }
        for (std::size_t row = 0; row < inputs.size(); ++row) {
            mouse::nn::NNOutput value;
            value.wdl = {0.5F, 0.25F, 0.25F};
            value.moves_left = 20.0F;
            outputs[row] = value;
        }
        if (call == 2 && invalid)
            outputs.back().policy_logits[1857] = std::numeric_limits<float>::infinity();
        if (call == 2 && invalid_mlh)
            outputs.back().moves_left = std::numeric_limits<float>::quiet_NaN();
        --active;
        return {};
    }
    [[nodiscard]] const mouse::nn::ModelMetadata& metadata() const noexcept override {
        return model;
    }
    [[nodiscard]] const mouse::nn::BackendInfo& info() const noexcept override {
        std::unique_lock lock(info_mutex);
        if (block_info) {
            entered_info = true;
            info_changed.notify_all();
            info_changed.wait(lock, [&] { return released_info; });
        }
        return identity;
    }
    void arm_info() {
        const std::lock_guard lock(info_mutex);
        block_info = true;
        entered_info = false;
        released_info = false;
    }
    bool wait_info() {
        std::unique_lock lock(info_mutex);
        return info_changed.wait_for(lock, std::chrono::seconds{3}, [&] { return entered_info; });
    }
    void release_info() {
        const std::lock_guard lock(info_mutex);
        released_info = true;
        info_changed.notify_all();
    }
    bool wait() {
        std::unique_lock lock(mutex);
        return changed.wait_for(lock, std::chrono::seconds{3}, [&] { return entered; });
    }
    void release() {
        const std::lock_guard lock(mutex);
        released = true;
        changed.notify_all();
    }
};
struct Running {
    GateBackend& backend;
    std::optional<mouse::Result<mouse::search::SearchResult>> result;
    std::thread thread;
    Running(GateBackend& provider, mouse::search::SearchEngine& engine,
            const mouse::search::SearchLimits& limits)
        : backend(provider), thread([this, &engine, limits] { result = engine.run(limits); }) {}
    ~Running() {
        backend.release();
        backend.release_info();
        if (thread.joinable()) thread.join();
    }
    Running(const Running&) = delete;
    Running& operator=(const Running&) = delete;
    Running(Running&&) = delete;
    Running& operator=(Running&&) = delete;
    void finish() {
        backend.release();
        backend.release_info();
        thread.join();
    }
};
struct Fixture {
    mouse::chess::AttackTables attacks;
    mouse::chess::Position root;
    mouse::nn::EncodingHistory history;
    GateBackend backend;
    std::unique_ptr<mouse::search::SearchEngine> engine;
    explicit Fixture(std::size_t threads = 1)
        : root(*mouse::chess::Position::from_fen(mouse::chess::kStartFen, attacks)) {
        history.reset(root);
        mouse::search::SearchConfig config;
        config.threads = threads;
        config.maximum_batch = 4;
        config.max_depth = 32;
        config.ram_limit_bytes = 8U << 20U;
        auto created = mouse::search::SearchEngine::create(config, attacks, backend);
        if (!created) throw std::runtime_error(created.error());
        engine = std::move(*created);
        auto set = engine->set_position(root, history);
        if (!set) throw std::runtime_error(set.error());
    }
};
}  // namespace
TEST_CASE("Stop drains running inference without starting the gathered next batch",
          "[search][pipeline]") {
    std::size_t threads = 1;
    SECTION("One gather worker") {}
    SECTION("Eight gather workers") { threads = 8; }
    Fixture fixture(threads);
    mouse::search::SearchLimits limits;
    limits.nodes.reset();
    Running running(fixture.backend, *fixture.engine, limits);
    REQUIRE(fixture.backend.wait());
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{3};
    while (fixture.engine->snapshot().tree_nodes < 9 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    const auto blocked = fixture.engine->snapshot();
    CHECK(blocked.nodes == 1);
    CHECK(blocked.tree_nodes >= 9);
    CHECK(blocked.charged_bytes <= blocked.budget_bytes);
    CHECK_FALSE(fixture.engine->run({}));
    CHECK_FALSE(fixture.engine->set_position(fixture.root, fixture.history));
    CHECK_FALSE(fixture.engine->advance({0, 0}));
    CHECK(fixture.engine->root_moves().size == 20);
    CHECK(fixture.engine->principal_variation().size == 0);
    fixture.engine->request_stop();
    running.finish();
    REQUIRE(running.result);
    REQUIRE(required_optional(running.result));
    CHECK((required_optional(running.result))->reason == mouse::search::StopReason::ExternalStop);
    CHECK((required_optional(running.result))->info.nodes == 5);
    CHECK(fixture.backend.calls == 2);
    CHECK(fixture.backend.maximum_active == 1);
    CHECK(fixture.backend.active == 0);
    mouse::search::SearchLimits resumed;
    resumed.nodes = 100;
    const auto result = fixture.engine->run(resumed);
    REQUIRE(result);
    CHECK(result->info.nodes == 100);
    REQUIRE(result->best_move);
    CHECK(fixture.engine->advance(required_optional(result->best_move)));
}
TEST_CASE("An expired deadline drains running inference without starting gathered work",
          "[search][pipeline][deadline-drain]") {
    Fixture fixture;
    // Complete the root before timing the controlled blocked provider call.
    mouse::search::SearchLimits prime;
    prime.nodes = 1;
    const auto root = fixture.engine->run(prime);
    REQUIRE(root);
    REQUIRE(root->info.nodes == 1);
    REQUIRE(fixture.backend.calls == 1);
    mouse::search::SearchLimits limits;
    limits.nodes.reset();
    limits.movetime = std::chrono::seconds{1};
    Running running(fixture.backend, *fixture.engine, limits);
    REQUIRE(fixture.backend.wait());
    const auto ready_deadline = std::chrono::steady_clock::now() + std::chrono::seconds{3};
    while (fixture.engine->snapshot().tree_nodes < 9 &&
           std::chrono::steady_clock::now() < ready_deadline)
        std::this_thread::yield();
    REQUIRE(fixture.engine->snapshot().tree_nodes >= 9);
    // The provider entered after the run's deadline origin. Holding it for a
    // full budget guarantees expiry without a tight scheduler-sensitive bound.
    std::this_thread::sleep_for(*limits.movetime);
    running.finish();
    REQUIRE(running.result);
    REQUIRE(required_optional(running.result));
    const auto& result = *required_optional(running.result);
    CHECK(result.reason == mouse::search::StopReason::TimeLimit);
    CHECK(result.info.nodes == 5);
    CHECK(result.info.inherited_visits == 1);
    CHECK(result.info.new_playouts == 4);
    CHECK(fixture.backend.calls == 2);
    CHECK(fixture.backend.maximum_active == 1);
    CHECK(fixture.backend.active == 0);
    std::uint64_t child_visits = 0;
    const auto moves = fixture.engine->root_moves();
    for (std::size_t i = 0; i < moves.size; ++i) child_visits += moves.moves.at(i).visits;
    CHECK(child_visits == 4);
    REQUIRE(result.best_move);
    // Re-root checks every allocated node, including canceled disconnected
    // slots: any leaked Pending/in_flight reservation fails immediately.
    REQUIRE(fixture.engine->advance(required_optional(result.best_move)));
    const auto retained = fixture.engine->snapshot().nodes;
    mouse::search::SearchLimits resumed;
    resumed.nodes = retained + 32;
    const auto recovered = fixture.engine->run(resumed);
    REQUIRE(recovered);
    CHECK(recovered->reason == mouse::search::StopReason::NodeLimit);
    CHECK(recovered->info.inherited_visits == retained);
    CHECK(recovered->info.new_playouts == 32);
    CHECK(recovered->info.nodes == retained + 32);
}
TEST_CASE("A later invalid batch publishes no valid prefix and releases both slots",
          "[search][pipeline]") {
    Fixture fixture;
    fixture.backend.invalid = true;
    SECTION("Invalid policy") {}
    SECTION("Invalid MLH") {
        fixture.backend.invalid = false;
        fixture.backend.invalid_mlh = true;
    }
    mouse::search::SearchLimits limits;
    limits.nodes = 200;
    Running running(fixture.backend, *fixture.engine, limits);
    REQUIRE(fixture.backend.wait());
    running.finish();
    REQUIRE(running.result);
    REQUIRE_FALSE(required_optional(running.result));
    CHECK(fixture.engine->snapshot().nodes == 1);
    const auto moves = fixture.engine->root_moves();
    for (std::size_t i = 0; i < moves.size; ++i) CHECK(moves.moves.at(i).visits == 0);
    fixture.backend.invalid = false;
    fixture.backend.invalid_mlh = false;
    const auto recovered = fixture.engine->run(limits);
    REQUIRE(recovered);
    CHECK(recovered->info.nodes == 200);
    CHECK(recovered->info.inherited_visits == 1);
    CHECK(fixture.backend.maximum_active == 1);
}
TEST_CASE("Parallel gather workers respect exact root node issuance bounds", "[search][pipeline]") {
    Fixture fixture(8);
    fixture.backend.release();
    mouse::search::SearchLimits limits;
    limits.nodes = 513;
    const auto result = fixture.engine->run(limits);
    REQUIRE(result);
    CHECK(result->info.nodes == 513);
    CHECK(result->reason == mouse::search::StopReason::NodeLimit);
    std::uint64_t visits = 0;
    const auto moves = fixture.engine->root_moves();
    for (std::size_t i = 0; i < moves.size; ++i) visits += moves.moves.at(i).visits;
    CHECK(visits == 512);
    CHECK(fixture.backend.maximum_active == 1);
    REQUIRE(result->best_move);
    CHECK(fixture.engine->advance(required_optional(result->best_move)));
}

TEST_CASE("Stop accepted during run identity validation is never cleared", "[search][pipeline]") {
    Fixture fixture;
    fixture.backend.arm_info();
    mouse::search::SearchLimits limits;
    limits.nodes = 257;
    Running running(fixture.backend, *fixture.engine, limits);
    REQUIRE(fixture.backend.wait_info());
    fixture.engine->request_stop();
    running.finish();
    REQUIRE(running.result);
    REQUIRE(required_optional(running.result));
    CHECK((required_optional(running.result))->reason == mouse::search::StopReason::ExternalStop);
    CHECK((required_optional(running.result))->info.nodes == 0);
    CHECK((required_optional(running.result))->info.new_playouts == 0);
    CHECK_FALSE((required_optional(running.result))->best_move);
    CHECK(fixture.backend.calls == 0);
    CHECK(fixture.engine->snapshot().nodes == 0);
    // The next accepted run resets the previous operation's cancellation.
    mouse::search::SearchLimits resumed;
    resumed.nodes = 5;
    const auto result = fixture.engine->run(resumed);
    REQUIRE(result);
    CHECK(result->reason == mouse::search::StopReason::NodeLimit);
    CHECK(result->info.nodes == 5);
}

TEST_CASE("Searchmoves constrain selection and every root view including retained trees",
          "[search][uci][root-filter]") {
    Fixture fixture;
    fixture.backend.released = true;
    auto first = mouse::chess::parse_uci_move(fixture.root, "h2h4", fixture.attacks);
    auto second = mouse::chess::parse_uci_move(fixture.root, "a2a3", fixture.attacks);
    REQUIRE(first);
    REQUIRE(second);
    mouse::search::SearchLimits limits;
    limits.nodes = 129;
    limits.searchmoves[0] = *first;
    limits.searchmoves_count = 1;
    const auto result = fixture.engine->run(limits);
    REQUIRE(result);
    REQUIRE(result->best_move);
    CHECK(required_optional(result->best_move) == *first);
    CHECK(result->info.nodes == 129);
    const auto roots = fixture.engine->root_moves();
    REQUIRE(roots.size == 1);
    CHECK(roots.moves[0].move == *first);
    CHECK(roots.moves[0].visits == 128);
    REQUIRE(result->info.pv.size != 0);
    CHECK(result->info.pv.moves[0] == *first);
    CHECK(fixture.engine->principal_variation(*second).size == 0);

    limits.nodes = 257;
    limits.searchmoves[0] = *second;
    const auto reused = fixture.engine->run(limits);
    REQUIRE(reused);
    REQUIRE(reused->best_move);
    CHECK(required_optional(reused->best_move) == *second);
    REQUIRE(fixture.engine->root_moves().size == 1);
    CHECK(fixture.engine->root_moves().moves[0].move == *second);
    CHECK(fixture.engine->principal_variation(*first).size == 0);
    CHECK(fixture.engine->principal_variation().moves[0] == *second);
    limits.searchmoves_count = 0;
    limits.nodes = 258;
    REQUIRE(fixture.engine->run(limits));
    CHECK(fixture.engine->root_moves().size == 20);
}
TEST_CASE("Invalid root filters reject atomically before any provider call",
          "[search][uci][root-filter]") {
    Fixture fixture;
    mouse::search::SearchLimits limits;
    fixture.backend.released = true;
    auto move = mouse::chess::parse_uci_move(fixture.root, "e2e4", fixture.attacks);
    REQUIRE(move);
    limits.searchmoves_count = 2;
    limits.searchmoves[0] = *move;
    limits.searchmoves[1] = *move;
    CHECK_FALSE(fixture.engine->run(limits));
    limits.searchmoves_count = 1;
    limits.searchmoves[0] = {0, 0};
    CHECK_FALSE(fixture.engine->run(limits));
    limits.searchmoves_count = limits.searchmoves.size() + 1;
    CHECK_FALSE(fixture.engine->run(limits));
    CHECK(fixture.backend.calls == 0);
    CHECK(fixture.engine->snapshot().nodes == 0);
}
TEST_CASE("A pre-requested run stop token cannot be cleared by acceptance",
          "[search][uci][cancellation]") {
    Fixture fixture;
    fixture.backend.released = true;
    std::stop_source cancelled;
    cancelled.request_stop();
    mouse::search::SearchLimits limits;
    limits.nodes = 257;
    limits.stop_token = cancelled.get_token();
    const auto result = fixture.engine->run(limits);
    REQUIRE(result);
    CHECK(result->reason == mouse::search::StopReason::ExternalStop);
    CHECK(result->info.nodes == 0);
    CHECK(fixture.backend.calls == 0);
}

TEST_CASE("Per-run timed batch caps bound real provider rows and do not change allocated capacity",
          "[search][pipeline][timed-batch]") {
    Fixture f;
    f.backend.release();
    mouse::search::SearchLimits limits;
    limits.nodes = 17;
    limits.smart_pruning = false;
    limits.maximum_batch = 1;
    const auto result = f.engine->run(limits);
    REQUIRE(result);
    CHECK(result->info.nodes == 17);
    CHECK(f.backend.largest_batch == 1);
    CHECK(result->best_move);
    limits.maximum_batch = 0;
    CHECK_FALSE(f.engine->run(limits));
    limits.maximum_batch = 5;
    CHECK_FALSE(f.engine->run(limits));
    REQUIRE(f.engine->set_position(f.root, f.history));
    limits.maximum_batch.reset();
    limits.nodes = 65;
    const auto full = f.engine->run(limits);
    REQUIRE(full);
    CHECK(full->info.nodes == 65);
    CHECK(f.backend.largest_batch > 1);
}

TEST_CASE("An absolute deadline that elapsed before acceptance is never restarted",
          "[search][pipeline][absolute-deadline]") {
    Fixture f;
    f.backend.release();
    mouse::search::SearchLimits limits;
    limits.nodes = 17;
    limits.movetime = std::chrono::seconds{1};
    limits.deadline = std::chrono::steady_clock::now() - std::chrono::milliseconds{1};
    const auto result = f.engine->run(limits);
    REQUIRE(result);
    CHECK(result->reason == mouse::search::StopReason::TimeLimit);
    CHECK(result->info.nodes == 0);
    CHECK(f.backend.calls == 0);
    limits.deadline.reset();
    const auto recovered = f.engine->run(limits);
    REQUIRE(recovered);
    CHECK(recovered->info.nodes == 17);
    CHECK(recovered->best_move);
}

TEST_CASE("A cached leaf backs up while another leaf waits for inference",
          "[search][pipeline][out-of-order-v1]") {
    const mouse::chess::AttackTables attacks;
    auto root = mouse::chess::Position::from_fen(mouse::chess::kStartFen, attacks);
    REQUIRE(root);
    mouse::nn::EncodingHistory history;
    history.reset(*root);
    GateBackend backend;
    backend.identity.cache_namespace = "pipeline-cache";
    auto cache = mouse::nn::LegalCache::create(1000, backend.identity.cache_namespace);
    REQUIRE(cache);
    auto child = *root;
    const auto root_moves = mouse::chess::legal_moves(child, attacks);
    REQUIRE(root_moves.size() > 1);
    const auto cached_move = root_moves[1];
    static_cast<void>(child.make_move(cached_move, attacks));
    auto child_history = history;
    REQUIRE(child_history.append(child));
    auto encoded = mouse::nn::encode_classical(child_history.frames(), 1);
    REQUIRE(encoded);
    const auto legal = mouse::chess::legal_moves(child, attacks);
    mouse::nn::PolicyIndices policy;
    policy.size = legal.size();
    for (std::size_t i = 0; i < legal.size(); ++i) {
        const auto index = mouse::nn::PolicyMap::index(legal[i], child.side_to_move());
        REQUIRE(index);
        policy.values.at(i) = *index;
    }
    mouse::nn::LegalOutput output;
    output.wdl = {0.5F, 0.25F, 0.25F};
    output.moves_left = 20.0F;
    REQUIRE((*cache)->insert(*encoded, policy, output));
    mouse::search::SearchConfig config;
    config.threads = 1;
    config.maximum_batch = 4;
    config.max_depth = 32;
    config.ram_limit_bytes = 8U << 20U;
    auto engine = mouse::search::SearchEngine::create(config, attacks, backend, cache->get());
    REQUIRE(engine);
    REQUIRE((*engine)->set_position(*root, history));
    mouse::search::SearchLimits limits;
    limits.nodes = 64;
    limits.smart_pruning = false;
    Running running(backend, **engine, limits);
    REQUIRE(backend.wait());
    CHECK((*engine)->snapshot().nodes >= 2);
    const auto statistics = (*engine)->root_moves();
    const auto moves = std::span{statistics.moves}.first(statistics.size);
    const auto cached = std::ranges::find(moves, cached_move, &mouse::search::RootMoveStat::move);
    REQUIRE(cached != moves.end());
    CHECK(cached->visits == 1);
    CHECK(backend.calls == 2);
    running.finish();
    const auto& result = required_optional(running.result);
    REQUIRE(result);
    CHECK(result->info.nodes == 64);
    REQUIRE((*engine)->set_position(*root, history, false));
    CHECK((*engine)->snapshot().nodes == 0);
}

TEST_CASE("Cache-only gathers cross the out-of-order quota without an empty provider call",
          "[search][pipeline][out-of-order-v1]") {
    const mouse::chess::AttackTables attacks;
    auto root = mouse::chess::Position::from_fen(mouse::chess::kStartFen, attacks);
    REQUIRE(root);
    mouse::nn::EncodingHistory history;
    history.reset(*root);
    GateBackend backend;
    backend.identity.cache_namespace = "pipeline-cache-only";
    backend.released = true;
    auto cache = mouse::nn::LegalCache::create(1000, backend.identity.cache_namespace);
    REQUIRE(cache);
    const auto seed = [&](mouse::chess::Position child, const mouse::nn::EncodingHistory& frames) {
        auto encoded = mouse::nn::encode_classical(frames.frames(), 1);
        REQUIRE(encoded);
        const auto legal = mouse::chess::legal_moves(child, attacks);
        mouse::nn::PolicyIndices policy;
        policy.size = legal.size();
        for (std::size_t i = 0; i < legal.size(); ++i) {
            const auto index = mouse::nn::PolicyMap::index(legal[i], child.side_to_move());
            REQUIRE(index);
            policy.values.at(i) = *index;
        }
        mouse::nn::LegalOutput output;
        output.wdl = {0.5F, 0.25F, 0.25F};
        output.moves_left = 20.0F;
        REQUIRE((*cache)->insert(*encoded, policy, output));
    };
    seed(*root, history);
    for (const auto move : mouse::chess::legal_moves(*root, attacks)) {
        auto child = *root;
        static_cast<void>(child.make_move(move, attacks));
        auto frames = history;
        REQUIRE(frames.append(child));
        seed(child, frames);
    }
    mouse::search::SearchConfig config;
    config.threads = 1;
    config.maximum_batch = 1;  // Source OOO limit truncates 2.4 * 1 to 2.
    config.max_depth = 32;
    config.ram_limit_bytes = 8U << 20U;
    auto engine = mouse::search::SearchEngine::create(config, attacks, backend, cache->get());
    REQUIRE(engine);
    REQUIRE((*engine)->set_position(*root, history));
    mouse::search::SearchLimits limits;
    limits.nodes = 4;
    limits.smart_pruning = false;
    const auto result = (*engine)->run(limits);
    REQUIRE(result);
    CHECK(result->reason == mouse::search::StopReason::NodeLimit);
    CHECK(result->info.nodes == 4);
    CHECK(backend.calls == 0);
    REQUIRE((*engine)->set_position(*root, history, false));
    CHECK((*engine)->snapshot().nodes == 0);
}

TEST_CASE("Search diagnostics count completed provider rows and reset per run",
          "[search][pipeline][search-stats]") {
    Fixture fixture(4);
    fixture.backend.release();
    mouse::search::SearchLimits limits;
    limits.nodes = 257;
    limits.log_search_stats = true;
    const auto result = fixture.engine->run(limits);
    REQUIRE(result);
    std::uint64_t batches = 0, positions = 0;
    for (std::size_t size = 0; size < result->stats.batch_sizes.size(); ++size) {
        batches += result->stats.batch_sizes.at(size);
        positions += size * result->stats.batch_sizes.at(size);
    }
    CHECK(batches == fixture.backend.calls.load());
    CHECK(positions == 257);
    CHECK(result->info.nodes == 257);
    CHECK(result->stats.batch_sizes.at(0) == 0);
    CHECK(result->stats.provider.count() > 0);
    CHECK(result->stats.gather.count() > 0);
    CHECK(result->stats.backup.count() > 0);
    CHECK(result->stats.wall >= result->stats.provider);
    REQUIRE(fixture.engine->set_position(fixture.root, fixture.history, false));
    limits.nodes = 1;
    const auto reset = fixture.engine->run(limits);
    REQUIRE(reset);
    CHECK(reset->stats.batch_sizes[1] == 1);
    CHECK(reset->stats.out_of_order == 0);
    REQUIRE(fixture.engine->set_position(fixture.root, fixture.history, false));
    limits.log_search_stats = false;
    const auto disabled = fixture.engine->run(limits);
    REQUIRE(disabled);
    CHECK(disabled->stats.batch_sizes[1] == 0);
    CHECK(disabled->stats.wall.count() == 0);
}

TEST_CASE("Snapshot cost scopes measure every reader and restore thread-local nesting",
          "[search][pipeline][snapshot-cost]") {
    Fixture fixture;
    mouse::search::SnapshotCost outer, inner, explicit_cost;
    {
        mouse::search::SnapshotCostScope scope(outer);
        static_cast<void>(fixture.engine->result_snapshot(&explicit_cost));
        {
            mouse::search::SnapshotCostScope nested(inner);
            static_cast<void>(fixture.engine->root_moves());
            static_cast<void>(fixture.engine->principal_variation());
        }
        std::thread independent([&] { static_cast<void>(fixture.engine->snapshot()); });
        independent.join();
        static_cast<void>(fixture.engine->snapshot());
    }
    static_cast<void>(fixture.engine->result_snapshot());
    CHECK(outer.tree_lock_count == 2);
    CHECK(inner.tree_lock_count == 2);
    CHECK(explicit_cost.tree_lock_count == 1);
    CHECK(mouse::search::SnapshotCostScope::current() == nullptr);
}