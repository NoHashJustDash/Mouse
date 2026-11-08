// SPDX-License-Identifier: GPL-3.0-or-later
// SolidTreeThreshold, parallel picking, Dirichlet noise, temperature, dag-preview and TimeManager
// smooth: lc0 features added after the first feature-parity pass.
#include <atomic>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>

#include "engine/chess/attacks.h"
#include "engine/chess/movegen.h"
#include "engine/nn/encoder.h"
#include "engine/nn/history.h"
#include "engine/nn/legal_cache.h"
#include "engine/search/search.h"
#include "engine/search/tree.h"
#include "engine/uci/time.h"
#include "tests/required_optional.h"

namespace {
using namespace mouse::search;
using mouse::chess::Move;
using mouse::tests::required_optional;

// Every position gets the same value and a policy that depends on the input alone, so a search's
// shape depends on its options and not on when a batch happened to run.
struct HashBackend final : mouse::nn::Backend {
    mouse::nn::ModelMetadata model;
    mouse::nn::BackendInfo identity;
    std::atomic<std::size_t> rows{0}, calls{0};
    HashBackend() {
        model.input_format = 1;
        identity.onnx_sha256 = "parity-features";
        identity.cache_namespace = "parity-features";
    }
    [[nodiscard]] mouse::Result<void> evaluate(std::span<const mouse::nn::InputPlanes> inputs,
                                               std::span<mouse::nn::NNOutput> outputs) override {
        rows += inputs.size();
        ++calls;
        for (std::size_t row = 0; row < inputs.size(); ++row) {
            std::uint64_t mix = 0xcbf29ce484222325ULL;
            for (const auto& plane : inputs[row]) mix = (mix ^ plane.mask) * 0x100000001b3ULL;
            mouse::nn::NNOutput value;
            auto seed = mix;
            for (auto& logit : value.policy_logits) {
                seed = seed * 0x9E3779B97F4A7C15ULL + 1;
                logit = static_cast<float>(seed >> 54U) / 256.0F;
            }
            const auto bias = static_cast<float>(mix % 41U) / 100.0F;
            value.wdl = {0.3F + bias / 2.0F, 0.4F, 0.3F - bias / 2.0F};
            value.moves_left = 30.0F;
            outputs[row] = value;
        }
        return {};
    }
    [[nodiscard]] const mouse::nn::ModelMetadata& metadata() const noexcept override {
        return model;
    }
    [[nodiscard]] const mouse::nn::BackendInfo& info() const noexcept override { return identity; }
};
struct Fixture {
    mouse::chess::AttackTables attacks;
    mouse::chess::Position root;
    mouse::nn::EncodingHistory history;
    HashBackend backend;
    std::unique_ptr<mouse::nn::LegalCache> cache;
    std::unique_ptr<SearchEngine> engine;
    explicit Fixture(SearchConfig config, std::string_view fen = mouse::chess::kStartFen,
                     bool with_cache = true)
        : root(*mouse::chess::Position::from_fen(fen, attacks)) {
        history.reset(root);
        if (with_cache) {
            auto made = mouse::nn::LegalCache::create(200000, "parity-features");
            REQUIRE(made.has_value());
            cache = std::move(*made);
        }
        config.ram_limit_bytes = 128ULL << 20U;
        auto created = SearchEngine::create(config, attacks, backend, cache.get());
        if (!created) FAIL(created.error());
        engine = std::move(*created);
        REQUIRE(engine->set_position(root, history));
    }
    SearchResult run(std::uint64_t nodes, bool pruning = false) {
        SearchLimits limits;
        limits.nodes = nodes;
        limits.smart_pruning = pruning;
        auto result = engine->run(limits);
        REQUIRE(result.has_value());
        return *result;
    }
};
std::map<std::string, std::uint64_t> visits_of(const SearchEngine& engine) {
    std::map<std::string, std::uint64_t> result;
    const auto roots = engine.all_root_moves();
    for (std::size_t i = 0; i < roots.size; ++i)
        result[mouse::chess::move_to_uci(roots.moves.at(i).move)] = roots.moves.at(i).visits;
    return result;
}
EdgeSeed seed(unsigned int tag) {
    return {.move = Move{static_cast<mouse::chess::Square>(tag % 64),
                         static_cast<mouse::chess::Square>((tag + 1) % 64)},
            .prior_bits = 0x3800};
}
}  // namespace

TEST_CASE("Solidify gives children contiguous slots and reports the moved", "[solid-tree]") {
    auto made = Tree::create(1U << 20U);
    REQUIRE(made.has_value());
    auto& tree = **made;
    const std::array seeds{seed(1), seed(2), seed(3)};
    REQUIRE(tree.expand(0, seeds));
    const auto existing = tree.allocate_node();
    REQUIRE(existing.has_value());
    tree.edges(0)[1].child = *existing;
    tree.node(*existing).visits = 7;
    tree.node(*existing).value_sum = 2.5;
    std::array<NodeRelocation, 8> moved{};
    const auto count = tree.solidify(0, moved);
    REQUIRE(count.has_value());
    REQUIRE(*count == 1);
    CHECK(moved[0].from == *existing);
    const auto first = tree.edges(0)[0].child;
    CHECK(tree.edges(0)[1].child == first + 1);
    CHECK(tree.edges(0)[2].child == first + 2);
    CHECK(moved[0].to == first + 1);
    CHECK(tree.node(first + 1).visits == 7);
    CHECK(tree.node(first + 1).value_sum == 2.5);
    CHECK(tree.node(*existing).visits == 0);
    CHECK(tree.node(0).solid);
    CHECK_FALSE(tree.solidify(0, moved).has_value());
}

TEST_CASE("Reuse survives children with lower IDs than their parent", "[solid-tree]") {
    auto made = Tree::create(1U << 20U);
    REQUIRE(made.has_value());
    auto& tree = **made;
    const std::array seeds{seed(1), seed(2)};
    REQUIRE(tree.expand(0, seeds));
    const auto child = tree.allocate_node();
    REQUIRE(child.has_value());
    tree.edges(0)[0].child = *child;
    REQUIRE(tree.expand(*child, seeds));
    const auto grandchild = tree.allocate_node();
    REQUIRE(grandchild.has_value());
    tree.edges(*child)[1].child = *grandchild;
    tree.node(*grandchild).visits = 3;
    // The child moves above its own child.
    std::array<NodeRelocation, 8> moved{};
    REQUIRE(tree.solidify(0, moved).has_value());
    const auto new_child = tree.edges(0)[0].child;
    REQUIRE(new_child > *grandchild);
    tree.node(new_child).visits = 9;
    const auto kept = tree.re_root(new_child);
    REQUIRE(kept.has_value());
    CHECK(kept->kept_nodes == 2);
    CHECK(tree.node(0).visits == 9);
    const auto below = tree.edges(0)[1].child;
    REQUIRE(below != kNoNode);
    CHECK(tree.node(below).visits == 3);
    CHECK(tree.edges(0)[0].child == kNoNode);
}

TEST_CASE("SolidTreeThreshold changes no search result", "[solid-tree]") {
    SearchConfig base;
    base.deterministic = true;
    base.threads = 1;
    base.maximum_batch = 16;
    auto run_with = [&](std::uint32_t threshold) {
        auto config = base;
        config.solid_tree_threshold = threshold;
        Fixture fixture(config);
        const auto result = fixture.run(3000);
        return std::pair{visits_of(*fixture.engine), result.info.q};
    };
    const auto [solid, solid_q] = run_with(1);
    const auto [plain, plain_q] = run_with(2000000000);
    CHECK(solid == plain);
    CHECK(solid_q == plain_q);
}

TEST_CASE("A solid tree keeps its visits across moves with paths in flight", "[solid-tree]") {
    SearchConfig config;
    config.threads = 2;
    config.maximum_batch = 32;
    config.solid_tree_threshold = 1;
    Fixture fixture(config);
    const auto first = fixture.run(6000);
    REQUIRE(first.best_move.has_value());
    const auto roots = fixture.engine->all_root_moves();
    std::uint64_t expected = 0;
    for (std::size_t i = 0; i < roots.size; ++i)
        if (roots.moves.at(i).move == *first.best_move) expected = roots.moves.at(i).visits;
    const auto advanced = fixture.engine->advance(*first.best_move);
    REQUIRE(advanced.has_value());
    CHECK(fixture.engine->snapshot().nodes == expected);
    const auto second = fixture.run(expected + 2000);
    CHECK(second.info.nodes >= expected + 2000);
}

// A batch checks its edges fit before publishing, and solidify must not spend them afterwards.
TEST_CASE("A solid tree that fills its arena stops at the memory limit", "[solid-tree]") {
    mouse::chess::AttackTables attacks;
    auto root = *mouse::chess::Position::from_fen(mouse::chess::kStartFen, attacks);
    mouse::nn::EncodingHistory history;
    history.reset(root);
    HashBackend backend;
    for (std::size_t megabytes = 24; megabytes <= 40; ++megabytes) {
        SearchConfig config;
        config.threads = 2;
        config.maximum_batch = 64;
        config.solid_tree_threshold = 1;
        config.ram_limit_bytes = megabytes << 20U;
        auto created = SearchEngine::create(config, attacks, backend, nullptr);
        if (!created) FAIL(created.error());
        REQUIRE((*created)->set_position(root, history));
        SearchLimits limits;
        limits.nodes = 100000000;
        limits.smart_pruning = false;
        const auto result = (*created)->run(limits);
        if (!result) FAIL(result.error());
        CHECK(result->reason == StopReason::MemoryLimit);
    }
}

TEST_CASE("Parallel picking places every visit and leaves nothing in flight", "[parallel-pick]") {
    SearchConfig config;
    config.threads = 4;
    config.maximum_batch = 128;
    config.gather.minimum_picking_work = 1;
    config.gather.minimum_remaining_picking_work = 0;
    Fixture fixture(config);
    const auto first = fixture.run(20000);
    CHECK(first.info.nodes >= 20000);
    REQUIRE(first.best_move.has_value());
    const auto roots = fixture.engine->all_root_moves();
    std::uint64_t total = 0;
    for (std::size_t i = 0; i < roots.size; ++i) {
        total += roots.moves.at(i).visits;
        CHECK(roots.moves.at(i).in_flight == 0);
    }
    CHECK(total + 1 == first.info.nodes);
    // Reuse checks the whole tree is quiescent and every child has one parent.
    REQUIRE(fixture.engine->advance(*first.best_move).has_value());
    const auto second = fixture.run(25000);
    CHECK(second.info.nodes >= 25000);
}

TEST_CASE("Parallel picking finds what picking alone finds", "[parallel-pick]") {
    auto searched = [](std::uint32_t minimum_picking) {
        SearchConfig config;
        config.threads = 4;
        config.maximum_batch = 64;
        config.gather.minimum_picking_work = minimum_picking;
        config.gather.minimum_remaining_picking_work = 0;
        // No result arrives while a batch is gathered and no batch leaves early, so each pick round
        // sees one fixed tree, and the same tree each time. Only the order of backups may differ.
        config.gather.out_of_order_eval = false;
        config.gather.thread_idling_threshold = 2;
        config.gather.max_prefetch = 0;
        Fixture fixture(config);
        fixture.run(8000);
        return visits_of(*fixture.engine);
    };
    const auto alone = searched(100000);
    const auto split = searched(1);
    double shared = 0, total = 0;
    for (const auto& [move, visits] : alone) {
        shared += static_cast<double>(std::min(visits, split.at(move)));
        total += static_cast<double>(visits);
    }
    CHECK(shared / total > 0.99);
}

TEST_CASE("Picking options outside lc0's ranges are refused", "[parallel-pick]") {
    SearchConfig config;
    config.gather.minimum_picking_work = 0;
    CHECK_FALSE(validate_search_tuning(config).has_value());
    config = {};
    config.gather.minimum_remaining_picking_work = 100001;
    CHECK_FALSE(validate_search_tuning(config).has_value());
    config = {};
    config.solid_tree_threshold = 0;
    CHECK_FALSE(validate_search_tuning(config).has_value());
    config = {};
    config.noise_epsilon = 1.5;
    CHECK_FALSE(validate_search_tuning(config).has_value());
    config = {};
    config.temperature.temperature = 101;
    CHECK_FALSE(validate_search_tuning(config).has_value());
}

TEST_CASE("Dirichlet noise changes root priors and nothing below", "[noise]") {
    SearchConfig config;
    config.deterministic = true;
    config.threads = 1;
    config.maximum_batch = 8;
    auto priors_of = [&](double epsilon, std::uint64_t random_seed) {
        auto selected = config;
        selected.noise_epsilon = epsilon;
        selected.noise_alpha = 0.3;
        selected.random_seed = random_seed;
        Fixture fixture(selected);
        fixture.run(50);
        std::vector<float> priors;
        const auto roots = fixture.engine->all_root_moves();
        for (std::size_t i = 0; i < roots.size; ++i) priors.push_back(roots.moves.at(i).prior);
        return priors;
    };
    const auto clean = priors_of(0.0, 5);
    const auto noised = priors_of(0.25, 5);
    const auto repeat = priors_of(0.25, 5);
    const auto other = priors_of(0.25, 6);
    CHECK(noised != clean);
    CHECK(noised == repeat);
    CHECK(noised != other);
    float sum = 0;
    for (const auto prior : noised) sum += prior;
    CHECK(sum == Catch::Approx(1.0F).margin(0.01F));
    // Mixing keeps at least 1 - epsilon of each prior.
    for (std::size_t i = 0; i < clean.size(); ++i)
        CHECK(noised[i] >= clean[i] * 0.75F - 0.001F);
}

TEST_CASE("Temperature samples root moves by visits", "[temperature]") {
    SearchConfig config;
    config.threads = 1;
    config.maximum_batch = 8;
    config.random_seed = 11;
    SECTION("none always plays the best") {
        Fixture fixture(config);
        const auto result = fixture.run(800);
        for (int i = 0; i < 50; ++i) CHECK(fixture.engine->final_move(1) == result.best_move);
    }
    SECTION("high temperature spreads over moves") {
        config.temperature.temperature = 10.0;
        Fixture fixture(config);
        fixture.run(800);
        std::set<std::string> seen;
        for (int i = 0; i < 300; ++i) seen.insert(mouse::chess::move_to_uci(required_optional(fixture.engine->final_move(1))));
        CHECK(seen.size() > 3);
    }
    SECTION("a cutoff move hands over to the endgame temperature") {
        config.temperature.temperature = 10.0;
        config.temperature.cutoff_move = 5;
        Fixture fixture(config);
        const auto result = fixture.run(800);
        // Ply 8 is move 4 + 1 = 5.
        for (int i = 0; i < 50; ++i) CHECK(fixture.engine->final_move(8) == result.best_move);
        std::set<std::string> seen;
        for (int i = 0; i < 300; ++i) seen.insert(mouse::chess::move_to_uci(required_optional(fixture.engine->final_move(2))));
        CHECK(seen.size() > 1);
    }
    SECTION("decay reaches zero after its moves") {
        config.temperature.temperature = 10.0;
        config.temperature.decay_moves = 4;
        config.temperature.decay_delay_moves = 2;
        Fixture fixture(config);
        const auto result = fixture.run(800);
        for (int i = 0; i < 50; ++i) CHECK(fixture.engine->final_move(12) == result.best_move);
    }
    SECTION("a value cutoff leaves out moves far behind the best") {
        config.temperature.temperature = 10.0;
        config.temperature.value_cutoff = 0.0;
        Fixture fixture(config);
        fixture.run(800);
        const auto roots = fixture.engine->all_root_moves();
        std::uint64_t most = 0;
        float best_q = -2.0F;
        for (std::size_t i = 0; i < roots.size; ++i)
            if (roots.moves.at(i).visits > most) {
                most = roots.moves.at(i).visits;
                best_q = static_cast<float>(roots.moves.at(i).q);
            }
        for (int i = 0; i < 100; ++i) {
            const auto choice = required_optional(fixture.engine->final_move(1));
            for (std::size_t m = 0; m < roots.size; ++m)
                if (roots.moves.at(m).move == choice)
                    CHECK(static_cast<float>(roots.moves.at(m).q) >= best_q - 1e-5F);
        }
    }
}

TEST_CASE("Dag search shares evaluations between transpositions", "[dag]") {
    auto searched = [](bool dag) {
        SearchConfig config;
        config.threads = 1;
        config.maximum_batch = 8;
        config.dag = dag;
        Fixture fixture(config);
        const auto result = fixture.run(5000);
        return std::tuple{fixture.backend.rows.load(), result.info.nodes, result.best_move};
    };
    const auto [dag_rows, dag_nodes, dag_best] = searched(true);
    const auto [tree_rows, tree_nodes, tree_best] = searched(false);
    CHECK(dag_nodes >= 5000);
    CHECK(tree_nodes >= 5000);
    REQUIRE(dag_best.has_value());
    REQUIRE(tree_best.has_value());
    // The start position transposes quickly: 1.Nf3 Nf6 2.Nc3 and 1.Nc3 Nf6 2.Nf3 are one position.
    CHECK(dag_rows < tree_rows);
}

TEST_CASE("Dag search holds together over a game", "[dag]") {
    SearchConfig config;
    config.threads = 1;
    config.maximum_batch = 16;
    config.dag = true;
    Fixture fixture(config);
    auto position = fixture.root;
    std::uint64_t target = 2000;
    for (int ply = 0; ply < 12; ++ply) {
        const auto result = fixture.run(target);
        REQUIRE(result.best_move.has_value());
        const auto legal = mouse::chess::legal_moves(position, fixture.attacks);
        CHECK(std::ranges::find(legal, *result.best_move) != legal.end());
        const auto roots = fixture.engine->all_root_moves();
        for (std::size_t i = 0; i < roots.size; ++i) CHECK(roots.moves.at(i).in_flight == 0);
        const auto pv = fixture.engine->principal_variation();
        CHECK(pv.size >= 1);
        const auto advanced = fixture.engine->advance(*result.best_move);
        REQUIRE(advanced.has_value());
        static_cast<void>(position.make_move(*result.best_move, fixture.attacks));
        target = fixture.engine->snapshot().nodes + 2000;
    }
}

TEST_CASE("Dag search keeps Q in range and finds a mate", "[dag]") {
    SearchConfig config;
    config.threads = 1;
    config.maximum_batch = 8;
    config.dag = true;
    Fixture fixture(config, "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1");
    const auto result = fixture.run(3000);
    CHECK(result.info.q >= -1.0);
    CHECK(result.info.q <= 1.0);
    REQUIRE(result.best_move.has_value());
    CHECK(mouse::chess::move_to_uci(*result.best_move) == "a1a8");
    CHECK(result.info.proof.outcome == ProofOutcome::Win);
}

TEST_CASE("Dag search treats threefold along a path as a draw", "[dag]") {
    SearchConfig config;
    config.threads = 1;
    config.maximum_batch = 8;
    config.dag = true;
    config.two_fold_draws = true;
    // Kings and rooks shuffling: the search is full of repetitions and must still finish.
    Fixture fixture(config, "8/8/8/4k3/8/8/4K3/R7 w - - 0 1");
    const auto result = fixture.run(4000);
    CHECK(result.info.nodes >= 4000);
    REQUIRE(result.best_move.has_value());
}

TEST_CASE("Dag search respects its memory limit", "[dag]") {
    SearchConfig config;
    config.threads = 1;
    config.maximum_batch = 8;
    config.dag = true;
    mouse::chess::AttackTables attacks;
    auto root = *mouse::chess::Position::from_fen(mouse::chess::kStartFen, attacks);
    mouse::nn::EncodingHistory history;
    history.reset(root);
    HashBackend backend;
    config.ram_limit_bytes = 8ULL << 20U;
    auto created = SearchEngine::create(config, attacks, backend, nullptr);
    REQUIRE(created.has_value());
    REQUIRE((*created)->set_position(root, history));
    SearchLimits limits;
    limits.nodes = 10000000;
    limits.smart_pruning = false;
    const auto result = (*created)->run(limits);
    REQUIRE(result.has_value());
    CHECK(result->reason == StopReason::MemoryLimit);
    CHECK(result->best_move.has_value());
}

TEST_CASE("Smooth reads its settings the way lc0 does", "[smooth]") {
    using mouse::uci::parse_time_manager;
    const auto plain = parse_time_manager("smooth");
    REQUIRE(plain.has_value());
    CHECK(plain->kind == mouse::uci::TimeManagerConfig::Kind::Smooth);
    CHECK(plain->smooth.init_timeuse == 0.7F);
    const auto tuned = parse_time_manager("smooth(max-move-budget=0.3,midpoint=50)");
    REQUIRE(tuned.has_value());
    CHECK(tuned->smooth.max_move_budget == 0.3F);
    CHECK(tuned->smooth.midpoint == 50.0F);
    const auto group = parse_time_manager("smooth(mle-legacy(steepness=4),init-piggybank=0.2)");
    REQUIRE(group.has_value());
    CHECK(group->smooth.steepness == 4.0F);
    CHECK(group->smooth.init_piggybank == 0.2F);
    CHECK_FALSE(parse_time_manager("smooth(nonsense=1)").has_value());
}

TEST_CASE("Smooth allocates as lc0's formulas say", "[smooth]") {
    using namespace mouse::uci;
    SmoothPlan::Inputs inputs;
    inputs.state.tree_reuse = inputs.parameters.init_tree_reuse;
    inputs.state.timeuse = inputs.parameters.init_timeuse;
    inputs.time_ms = 60000;
    inputs.increment_ms = 1000;
    inputs.overhead_ms = 100;
    inputs.game_ply = 1;
    SmoothPlan plan(inputs);
    plan.begin(0);
    // By hand from smooth.cc: ply 1, midpoint 45.2, steepness 5.93.
    const float move = 0.5F;
    const float moves =
        45.2F * std::pow(1.0F + 2.0F * std::pow(move / 45.2F, 5.93F), 1.0F / 5.93F) - move;
    const float piggy = 60000.0F * 0.09F;
    const float total = 60000.0F - std::floor(piggy) + 1000.0F * (moves - 1.0F) - 100.0F;
    const float per_move = total / moves;
    const float brutto = per_move / (1.0F - 0.52F);
    const float netto = brutto * (1.0F - 0.12F);
    const float allocated = std::min(netto / 0.7F, 60000.0F * 0.42F);
    CHECK(plan.deadline_ms() == static_cast<std::int64_t>(allocated));
    CHECK(plan.allowed_piggybank_ms() == static_cast<std::int64_t>(std::floor(piggy) * 0.94F));
    CHECK(plan.upper_bound().count() >= plan.deadline_ms());
}

// After a move that stopped early, a large share of its nodes reused raises the allocation.
TEST_CASE("Smooth's upper bound holds whatever the tree holds", "[smooth]") {
    using namespace mouse::uci;
    SmoothPlan::Inputs inputs;
    inputs.state.tree_reuse = 0.5F;
    inputs.state.timeuse = inputs.parameters.init_timeuse;
    inputs.state.is_first_move = false;
    inputs.state.avg_ms_per_move = 1000.0F;
    inputs.state.last_time = 3390;
    inputs.state.last_move_final_nodes = 100;
    inputs.time_ms = 60000;
    inputs.overhead_ms = 100;
    inputs.game_ply = 41;
    SmoothPlan plan(inputs);
    const auto bound = plan.upper_bound().count();
    plan.begin(90);
    CHECK(bound >= plan.deadline_ms() + plan.allowed_piggybank_ms());
    CHECK(bound <= inputs.time_ms - inputs.overhead_ms);
}

TEST_CASE("Smooth's first allocation matches the lc0 binary's log", "[smooth]") {
    using namespace mouse::uci;
    // lc0 v0.32.1 --logfile, TimeManager=smooth, go wtime 30000 btime 30000 winc 300 binc 300 from
    // the start position, MoveOverheadMs 200:
    //   TMGR: MOVE TIME: allocated()=2356.0ms, expected(brutto)=1874.1ms, expected(netto)=1649.2ms
    //   TMGR: PIGGYBANK: total=2924ms, increment=224.9ms, longthink=2538ms, max=35038.4ms
    const auto config = parse_time_manager("smooth");
    REQUIRE(config.has_value());
    LegacyTimeManager manager{*config};
    GoCommand go;
    go.wtime = 30000;
    go.btime = 30000;
    go.winc = 300;
    go.binc = 300;
    auto budget = manager.allocate(
        go, {.side_to_move = mouse::chess::Color::White, .overhead_ms = 200, .game_ply = 1});
    REQUIRE(budget.has_value());
    REQUIRE(budget->smooth);
    budget->smooth->begin(0);
    // The log rounds to a tenth; the deadline truncates a float to whole milliseconds.
    CHECK(budget->smooth->deadline_ms() >= 2355);
    CHECK(budget->smooth->deadline_ms() <= 2356);
    CHECK(budget->smooth->allowed_piggybank_ms() == 2538);
}

TEST_CASE("Smooth stops at its deadline and may use the piggybank", "[smooth]") {
    using namespace mouse::uci;
    SmoothPlan::Inputs inputs;
    inputs.state.tree_reuse = inputs.parameters.init_tree_reuse;
    inputs.state.timeuse = inputs.parameters.init_timeuse;
    inputs.parameters.force_piggybank_ms = 0.0F;
    inputs.time_ms = 60000;
    inputs.overhead_ms = 100;
    inputs.game_ply = 40;
    SmoothPlan plan(inputs);
    plan.begin(0);
    const std::array<std::uint64_t, 2> visits{1000, 10};
    const auto deadline = plan.deadline_ms();
    const auto with_bank = deadline + plan.allowed_piggybank_ms();
    REQUIRE(plan.allowed_piggybank_ms() > 0);
    TimeSample sample{.new_nodes = 1000, .total_nodes = 1000, .root_visits = visits};
    using Milliseconds = std::chrono::milliseconds;
    // Lc0 forces the piggybank until force-piggybank-ms after the first batch, inclusive.
    auto first = plan.decide(sample, Milliseconds{1});
    CHECK_FALSE(first.stop);
    REQUIRE(first.remaining.has_value());
    CHECK(first.remaining->count() == with_bank - 1);
    sample.new_nodes = 1500;
    auto early = plan.decide(sample, Milliseconds{deadline / 2});
    CHECK_FALSE(early.stop);
    REQUIRE(early.remaining.has_value());
    CHECK(early.remaining->count() == deadline - deadline / 2);
    CHECK(early.nps.has_value());
    sample.new_nodes = 2000;
    auto done = plan.decide(sample, Milliseconds{deadline});
    CHECK(done.stop);
    // A need for more time extends the limit to the piggybank's edge.
    sample.need_more_time = true;
    sample.new_nodes = 3000;
    auto extended = plan.decide(sample, Milliseconds{deadline + 1});
    CHECK_FALSE(extended.stop);
    sample.new_nodes = 4000;
    auto spent = plan.decide(sample, Milliseconds{with_bank});
    CHECK(spent.stop);
    const auto state = plan.finish(Milliseconds{with_bank}, 5000);
    REQUIRE(state.has_value());
    // Time beyond the deadline came out of the bank, and the piggybank use counts as full use.
    CHECK(state->nps_is_reliable);
    CHECK(state->last_move_final_nodes == 5000);
    CHECK(state->timeuse >= inputs.state.timeuse);
}

TEST_CASE("Smooth carries its state between moves only when a move ran", "[smooth]") {
    using namespace mouse::uci;
    const auto config = parse_time_manager("smooth");
    REQUIRE(config.has_value());
    LegacyTimeManager manager{*config};
    GoCommand go;
    go.wtime = 60000;
    go.btime = 60000;
    TimeContext context{.side_to_move = mouse::chess::Color::White, .overhead_ms = 100, .game_ply = 1};
    auto first = manager.allocate(go, context);
    REQUIRE(first.has_value());
    REQUIRE(first->smooth);
    REQUIRE(first->budget.has_value());
    // Never begun: nothing to learn.
    manager.complete(*first, std::chrono::milliseconds{500}, 1000);
    auto second = manager.allocate(go, context);
    REQUIRE(second.has_value());
    CHECK(second->budget == first->budget);
    second->smooth->begin(0);
    const std::array<std::uint64_t, 1> visits{5000};
    static_cast<void>(second->smooth->decide(
        {.new_nodes = 100, .total_nodes = 100, .root_visits = visits}, std::chrono::milliseconds{1}));
    static_cast<void>(second->smooth->decide(
        {.new_nodes = 90000, .total_nodes = 90000, .root_visits = visits},
        std::chrono::milliseconds{1000}));
    manager.complete(*second, std::chrono::milliseconds{1000}, 90000);
    auto third = manager.allocate(go, context);
    REQUIRE(third.has_value());
    CHECK(third->budget != first->budget);
    manager.reset();
    auto fresh = manager.allocate(go, context);
    REQUIRE(fresh.has_value());
    CHECK(fresh->budget == first->budget);
}

TEST_CASE("A running search takes the time stopper's word", "[smooth]") {
    SearchConfig config;
    config.threads = 1;
    config.maximum_batch = 8;
    Fixture fixture(config);
    int calls = 0;
    SearchLimits limits;
    limits.nodes = 10000000;
    limits.smart_pruning = false;
    limits.time_stopper = [&](const TimeSample& sample) {
        ++calls;
        return TimeDecision{.stop = sample.new_nodes >= 500};
    };
    const auto result = fixture.engine->run(limits);
    REQUIRE(result.has_value());
    CHECK(result->reason == StopReason::TimeLimit);
    CHECK(calls > 0);
    CHECK(result->info.nodes < 2000);
}
