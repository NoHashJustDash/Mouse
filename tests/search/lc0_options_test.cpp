// SPDX-License-Identifier: GPL-3.0-or-later
// Lc0 search options added on senior/lc0-feature-parity: each one changes what its name says.
#include <atomic>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <stdexcept>

#include "engine/chess/attacks.h"
#include "engine/chess/movegen.h"
#include "engine/nn/legal_cache.h"
#include "engine/search/collisions.h"
#include "engine/search/draw_rate.h"
#include "engine/search/rules.h"
#include "engine/search/search.h"
#include "engine/search/selection.h"

namespace {
using mouse::search::GatherParameters;
struct CountingBackend final : mouse::nn::Backend {
    mouse::nn::ModelMetadata model;
    mouse::nn::BackendInfo identity;
    std::atomic<std::size_t> rows{0}, calls{0};
    std::array<float, 3> wdl{0.4F, 0.4F, 0.2F};
    CountingBackend() {
        model.input_format = 1;
        identity.onnx_sha256 = "lc0-options-test";
    }
    [[nodiscard]] mouse::Result<void> evaluate(std::span<const mouse::nn::InputPlanes> inputs,
                                               std::span<mouse::nn::NNOutput> outputs) override {
        rows += inputs.size();
        ++calls;
        for (std::size_t row = 0; row < inputs.size(); ++row) {
            mouse::nn::NNOutput value;
            value.wdl = wdl;
            value.moves_left = 20.0F;
            outputs[row] = value;
        }
        return {};
    }
    [[nodiscard]] const mouse::nn::ModelMetadata& metadata() const noexcept override {
        return model;
    }
    [[nodiscard]] const mouse::nn::BackendInfo& info() const noexcept override { return identity; }
};
struct Searcher {
    mouse::chess::AttackTables attacks;
    mouse::chess::Position root;
    mouse::nn::EncodingHistory history;
    CountingBackend backend;
    std::unique_ptr<mouse::search::SearchEngine> engine;
    explicit Searcher(mouse::search::SearchConfig config,
                      mouse::nn::LegalCache* cache = nullptr,
                      std::string_view fen = mouse::chess::kStartFen)
        : root(*mouse::chess::Position::from_fen(fen, attacks)) {
        history.reset(root);
        config.ram_limit_bytes = 64U << 20U;
        config.max_depth = 64;
        auto created = mouse::search::SearchEngine::create(config, attacks, backend, cache);
        if (!created) throw std::runtime_error(created.error());
        engine = std::move(*created);
        auto set = engine->set_position(root, history);
        if (!set) throw std::runtime_error(set.error());
    }
    mouse::search::SearchResult run(std::uint64_t nodes) {
        mouse::search::SearchLimits limits;
        limits.nodes = nodes;
        limits.smart_pruning = false;
        auto result = engine->run(limits);
        if (!result) throw std::runtime_error(result.error());
        return *result;
    }
};
}  // namespace

TEST_CASE("Collision options drive the visit budget", "[lc0-options]") {
    using mouse::search::collision_visit_budget;
    constexpr auto kNoLimit = std::numeric_limits<std::uint64_t>::max();
    // Defaults reproduce the fixed budget this replaced.
    CHECK(collision_visit_budget(28) == 1);
    CHECK(collision_visit_budget(145000) == 80000);
    CHECK(collision_visit_budget(10000) > 1);
    GatherParameters small;
    small.max_collision_visits = 500;
    CHECK(collision_visit_budget(145000, kNoLimit, small) == 500);
    CHECK(collision_visit_budget(10000, kNoLimit, small) < collision_visit_budget(10000));
    // Lc0: an end of 0 turns scaling off, because the end is checked first.
    GatherParameters unscaled;
    unscaled.collision_scaling_end = 0;
    CHECK(collision_visit_budget(1, kNoLimit, unscaled) == 80000);
    GatherParameters late;
    late.collision_scaling_start = 5000;
    CHECK(collision_visit_budget(4000, kNoLimit, late) == 1);
    GatherParameters steep;
    steep.collision_scaling_power = 3.0;
    CHECK(collision_visit_budget(10000, kNoLimit, steep) < collision_visit_budget(10000));
}

TEST_CASE("Contempt text resolves as lc0 GetContempt does", "[lc0-options]") {
    using mouse::search::resolve_contempt;
    CHECK(*resolve_contempt("", "", 0.0) == 0.0);
    CHECK(*resolve_contempt("", "", 75.0) == 75.0);
    CHECK(*resolve_contempt("", "50", 75.0) == 50.0);
    CHECK(*resolve_contempt("Stockfish 17", "stock=100,20", 0.0) == 100.0);
    CHECK(*resolve_contempt("Komodo", "stock=100,20", 0.0) == 20.0);
    CHECK(*resolve_contempt("Komodo", "stock=100", 30.0) == 30.0);
    CHECK_FALSE(resolve_contempt("", "abc", 0.0));
    CHECK_FALSE(resolve_contempt("x", "x=abc", 0.0));
    CHECK_FALSE(resolve_contempt("x", "a=b=c", 0.0));
    // A malformed value for an opponent we are not playing is not read, as in lc0.
    CHECK(resolve_contempt("Komodo", "stock=abc", 0.0));
}

TEST_CASE("Contempt shifts WDL towards the side that holds it", "[lc0-options]") {
    using mouse::search::rescale_wdl;
    using mouse::search::wdl_rescale_parameters;
    mouse::search::DrawRateParameters parameters;
    parameters.target = 0.0;
    parameters.reference = 0.5;
    CHECK(wdl_rescale_parameters(parameters).ratio == 1.0F);
    CHECK(wdl_rescale_parameters(parameters).diff == 0.0F);
    parameters.contempt = 100.0;
    const auto rescale = wdl_rescale_parameters(parameters);
    CHECK(rescale.diff > 0.0F);
    float q_for = 0.1F, draw_for = 0.5F;
    REQUIRE(rescale_wdl(q_for, draw_for, {.ratio = 1.0F, .diff = rescale.diff, .sign = 1.0F}));
    CHECK(q_for > 0.1F);
    float q_against = 0.1F, draw_against = 0.5F;
    REQUIRE(rescale_wdl(q_against, draw_against, {.ratio = 1.0F, .diff = rescale.diff, .sign = -1.0F}));
    CHECK(q_against < 0.1F);
    // No diff: the same as the ratio-only transform.
    float q_plain = 0.1F, draw_plain = 0.5F, q_new = 0.1F, draw_new = 0.5F;
    REQUIRE(rescale_wdl(q_plain, draw_plain, 1.2F, 1.4F));
    REQUIRE(rescale_wdl(q_new, draw_new, {.ratio = 1.2F, .diff = 0.0F, .sign = 1.0F, .max_s = 1.4F}));
    CHECK(q_plain == q_new);
    CHECK(draw_plain == draw_new);
    // Attenuation scales it and the maximum clamps it.
    auto halved = parameters;
    halved.attenuation = 0.5;
    CHECK(wdl_rescale_parameters(halved).diff == Catch::Approx(rescale.diff * 0.5F));
    auto clamped = parameters;
    clamped.contempt_max = 10.0;
    CHECK(wdl_rescale_parameters(clamped).diff < rescale.diff);
    // Calibration Elo selects Lc0's simplified model, which sets its own width.
    auto calibrated = parameters;
    calibrated.calibration_elo = 3000.0;
    CHECK(wdl_rescale_parameters(calibrated).ratio != 1.0F);
    calibrated.contempt = 0.0;
    CHECK(wdl_rescale_parameters(calibrated).diff == 0.0F);
}

TEST_CASE("FPU strategy absolute uses the value itself", "[lc0-options]") {
    mouse::search::SelectionParameters parameters;
    parameters.fpu_reduction = 0.5;
    CHECK(first_play_urgency(parameters, false, 0.2, 0.25) == Catch::Approx(0.2 - 0.5 * 0.5));
    parameters.fpu_absolute = true;
    CHECK(first_play_urgency(parameters, false, 0.2, 0.25) == 0.5);
    // Root "same" follows it; an explicit root strategy still wins.
    CHECK(first_play_urgency(parameters, true, 0.2, 0.25) == 0.5);
    parameters.root_fpu_strategy = mouse::search::RootFpuStrategy::Reduction;
    parameters.root_fpu = 0.1;
    CHECK(first_play_urgency(parameters, true, 0.2, 0.25) == Catch::Approx(0.2 - 0.1 * 0.5));
    // Negative only makes sense as an absolute value.
    parameters.fpu_reduction = -0.5;
    CHECK(validate_selection_parameters(parameters));
    parameters.fpu_absolute = false;
    CHECK_FALSE(validate_selection_parameters(parameters));
}

TEST_CASE("TwoFoldDraws controls tree repetition draws", "[lc0-options]") {
    mouse::chess::AttackTables attacks;
    auto position = *mouse::chess::Position::from_fen(mouse::chess::kStartFen, attacks);
    const std::array keys{position.key()};
    const auto on = mouse::search::classify_leaf(position, attacks, keys,
                                                 mouse::search::LeafContext::NonRoot, true);
    const auto off = mouse::search::classify_leaf(position, attacks, keys,
                                                  mouse::search::LeafContext::NonRoot, false);
    REQUIRE(on);
    REQUIRE(off);
    CHECK(on->reason == mouse::search::TerminalReason::TreeRepetition);
    CHECK(off->reason == mouse::search::TerminalReason::None);
}

TEST_CASE("Search completes under every gather configuration", "[lc0-options]") {
    mouse::search::SearchConfig config;
    config.threads = 2;
    config.maximum_batch = 16;
    SECTION("Defaults") {}
    SECTION("OutOfOrderEval off") { config.gather.out_of_order_eval = false; }
    SECTION("TaskWorkers 0, the coordinator alone") { config.threads = 0; }
    SECTION("Rounds never large enough for helpers") { config.gather.minimum_processing_work = 100; }
    SECTION("Helpers from the second row") {
        config.gather.minimum_processing_work = 2;
        config.gather.minimum_per_task_processing = 1;
    }
    SECTION("Few collision events") { config.gather.max_collision_events = 1; }
    SECTION("No out-of-order evals") { config.gather.max_out_of_order_factor = 0.0; }
    SECTION("Never idle") { config.gather.thread_idling_threshold = 2; }
    SECTION("Always idle") { config.gather.thread_idling_threshold = 0; }
    SECTION("History fill always") { config.history_fill = mouse::nn::HistoryFill::Always; }
    SECTION("Sticky endgames off") { config.terminal_proof = false; }
    SECTION("Two-fold draws off") { config.two_fold_draws = false; }
    Searcher searcher(config);
    const auto result = searcher.run(500);
    CHECK(result.info.nodes == 500);
    REQUIRE(result.best_move);
    CHECK(searcher.engine->advance(*result.best_move));
}

TEST_CASE("OutOfOrderEval off still reaches terminal positions", "[lc0-options]") {
    mouse::search::SearchConfig config;
    config.threads = 1;
    config.maximum_batch = 8;
    config.gather.out_of_order_eval = false;
    // White mates in one with several legal moves: terminal rows wait for their batch.
    Searcher searcher(config, nullptr, "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1");
    const auto result = searcher.run(300);
    CHECK(result.info.nodes >= 300);
    REQUIRE(result.best_move);
}

TEST_CASE("DrawScore counts from white's side at the root", "[lc0-options]") {
    mouse::search::SearchConfig config;
    config.threads = 1;
    config.maximum_batch = 8;
    config.draw_score = -1.0;
    constexpr auto kBlackToMove =
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1";
    Searcher white(config);
    white.run(100);
    const auto white_moves = white.engine->root_moves();
    Searcher black(config, nullptr, kBlackToMove);
    black.run(100);
    const auto black_moves = black.engine->root_moves();
    bool white_seen = false, black_seen = false;
    for (std::size_t i = 0; i < white_moves.size; ++i)
        if (white_moves.moves[i].visits != 0) {
            CHECK(white_moves.moves[i].draw_score_share < 0.0);
            white_seen = true;
        }
    for (std::size_t i = 0; i < black_moves.size; ++i)
        if (black_moves.moves[i].visits != 0) {
            CHECK(black_moves.moves[i].draw_score_share > 0.0);
            black_seen = true;
        }
    CHECK(white_seen);
    CHECK(black_seen);
}

TEST_CASE("NodesPerSecondLimit holds the search back", "[lc0-options]") {
    mouse::search::SearchConfig config;
    config.threads = 1;
    config.maximum_batch = 8;
    config.nps_limit = 1000.0;
    Searcher searcher(config);
    mouse::search::SearchLimits limits;
    limits.nodes.reset();
    limits.movetime = std::chrono::milliseconds{300};
    limits.smart_pruning = false;
    const auto result = searcher.engine->run(limits);
    REQUIRE(result);
    // 300 ms at 1000 nps, plus what one pipeline of two batches can overshoot by.
    CHECK(result->info.nodes <= 300 + 3 * 8);
    CHECK(result->info.nodes >= 100);
}

TEST_CASE("MaxPrefetch evaluates extra positions for the cache", "[lc0-options]") {
    const auto rows_with = [](std::uint32_t prefetch) {
        auto cache = mouse::nn::LegalCache::create(100000, "lc0-options-test", 0);
        if (!cache) throw std::runtime_error(cache.error());
        mouse::search::SearchConfig config;
        config.threads = 1;
        config.maximum_batch = 32;
        config.gather.max_prefetch = prefetch;
        Searcher searcher(config, cache->get());
        const auto result = searcher.run(40);
        return std::pair{searcher.backend.rows.load(), result.info.nodes};
    };
    const auto [without, without_nodes] = rows_with(0);
    const auto [with, with_nodes] = rows_with(32);
    CHECK(without_nodes >= 40);
    CHECK(with_nodes >= 40);
    CHECK(with > without);
}

TEST_CASE("Gather options outside lc0's ranges are refused", "[lc0-options]") {
    mouse::search::SearchConfig config;
    CHECK(mouse::search::validate_search_tuning(config));
    auto bad = config;
    bad.gather.max_collision_events = 0;
    CHECK_FALSE(mouse::search::validate_search_tuning(bad));
    bad = config;
    bad.gather.collision_scaling_power = 0.0;
    CHECK_FALSE(mouse::search::validate_search_tuning(bad));
    bad = config;
    bad.gather.minimum_processing_work = 1;
    CHECK_FALSE(mouse::search::validate_search_tuning(bad));
    bad = config;
    bad.draw_score = 1.5;
    CHECK_FALSE(mouse::search::validate_search_tuning(bad));
    bad = config;
    bad.nps_limit = -1.0;
    CHECK_FALSE(mouse::search::validate_search_tuning(bad));
}
