// SPDX-License-Identifier: GPL-3.0-or-later
// Lc0 UCI options added on senior/lc0-feature-parity.
#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cmath>

#include "engine/uci/options.h"
#include "engine/uci/score.h"
#include "engine/uci/time.h"

namespace {
using mouse::uci::apply_option;
using mouse::uci::Options;
using mouse::uci::ScoreType;
using Milliseconds = std::chrono::milliseconds;
auto set(const Options& options, const char* name, const char* value) {
    return apply_option(options, {.name = name, .value = std::string{value}});
}
}  // namespace

TEST_CASE("Every Lc0 option added is offered with Lc0's default", "[lc0-options]") {
    const Options options{std::filesystem::path{"cache"}};
    const auto lines = mouse::uci::option_lines(options);
    const auto offered = [&](std::string_view text) {
        return std::ranges::any_of(
            lines, [&](const std::string& line) { return line.find(text) != std::string::npos; });
    };
    for (const auto* expected :
         {"option name MaxCollisionEvents type spin default 917 min 1 max 65536",
          "option name MaxCollisionVisits type spin default 80000 min 1 max 100000000",
          "option name MaxCollisionVisitsScalingStart type spin default 28 min 1 max 100000",
          "option name MaxCollisionVisitsScalingEnd type spin default 145000 min 0 max 100000000",
          "option name MaxCollisionVisitsScalingPower type string default 1.25",
          "option name OutOfOrderEval type check default true",
          "option name MaxOutOfOrderEvalsFactor type string default 2.4",
          "option name MaxPrefetch type spin default 32 min 0 max 1024",
          "option name IdlingMinimumWork type spin default 0 min 0 max 10000",
          "option name ThreadIdlingThreshold type spin default 1 min 0 max 128",
          "option name MinimumProcessingWork type spin default 20 min 2 max 100000",
          "option name MinimumPerTaskProcessing type spin default 8 min 1 max 100000",
          "option name TaskWorkers type spin default -1 min -1 max 128",
          "option name MaxConcurrentSearchers type spin default 1 min 0 max 128",
          "option name SearchSpinBackoff type check default false",
          "option name TwoFoldDraws type check default true",
          "option name StickyEndgames type check default true",
          "option name FpuStrategy type combo default reduction var reduction var absolute",
          "option name HistoryFill type combo default fen_only var no var fen_only var always",
          "option name DrawScore type string default 0",
          "option name NodesPerSecondLimit type string default 0",
          "option name Contempt type string default <empty>",
          "option name ContemptMode type combo default play var play var white_side_analysis",
          "option name WDLCalibrationElo type string default 0",
          "option name WDLEvalObjectivity type string default 1",
          "option name UCI_Opponent type string default <empty>",
          "option name UCI_RatingAdv type string default 0",
          "option name ScoreType type combo default WDL_mu var centipawn",
          "option name PerPVCounters type check default false",
          "option name UCI_ShowMovesLeft type check default false",
          "option name VerboseMoveStats type check default false",
          "option name LogFile type string default <empty>",
          "option name ConfigFile type string default mouse.config",
          "option name TimeManager type string default legacy",
          "option name BackendOptions type string default <empty>"})
        CHECK(offered(expected));
}

TEST_CASE("Lc0 options reach their settings and reject bad values", "[lc0-options]") {
    const Options base{std::filesystem::path{"cache"}};
    CHECK(set(base, "MaxPrefetch", "96")->options.search.gather.max_prefetch == 96);
    CHECK_FALSE(set(base, "MaxPrefetch", "2000"));
    CHECK(set(base, "TaskWorkers", "-1")->options.task_workers == -1);
    CHECK(set(base, "TaskWorkers", "4")->options.task_workers == 4);
    CHECK_FALSE(set(base, "TaskWorkers", "-2"));
    CHECK(set(base, "MaxCollisionVisitsScalingPower", "2")
              ->options.search.gather.collision_scaling_power == 2.0);
    CHECK_FALSE(set(base, "MaxCollisionVisitsScalingPower", "0"));
    CHECK_FALSE(set(base, "MaxOutOfOrderEvalsFactor", "101"));
    CHECK_FALSE(set(base, "OutOfOrderEval", "maybe"));
    CHECK(set(base, "FpuStrategy", "absolute")->options.search.selection.fpu_absolute);
    CHECK_FALSE(set(base, "FpuStrategy", "other"));
    CHECK(set(base, "HistoryFill", "always")->options.search.history_fill ==
          mouse::nn::HistoryFill::Always);
    CHECK_FALSE(set(base, "DrawScore", "2"));
    CHECK_FALSE(set(base, "NodesPerSecondLimit", "-1"));
    CHECK(set(base, "ScoreType", "W-L")->options.score_type == ScoreType::WL);
    CHECK_FALSE(set(base, "ScoreType", "cp"));
    // Search options discard the tree; display options do not.
    CHECK(set(base, "MaxPrefetch", "96")->impact == mouse::uci::Reconfigure::Search);
    CHECK(set(base, "PerPVCounters", "true")->impact == mouse::uci::Reconfigure::None);
}

TEST_CASE("Contempt options resolve together", "[lc0-options]") {
    Options options{std::filesystem::path{"cache"}};
    options = set(options, "UCI_RatingAdv", "40")->options;
    CHECK(options.search.draw_rate.contempt == 40.0);
    options = set(options, "UCI_Opponent", "none 3500 computer Stockfish")->options;
    options = set(options, "Contempt", "stockfish=120,10")->options;
    CHECK(options.search.draw_rate.contempt == 120.0);
    options = set(options, "UCI_Opponent", "<empty>")->options;
    CHECK(options.search.draw_rate.contempt == 10.0);
    options = set(options, "Contempt", "<empty>")->options;
    CHECK(options.search.draw_rate.contempt == 40.0);
    CHECK_FALSE(set(options, "Contempt", "not-a-number"));
    CHECK_FALSE(set(options, "ContemptMaxValue", "20000"));
    CHECK_FALSE(set(options, "WDLEvalObjectivity", "2"));
}

TEST_CASE("BackendOptions sets known keys and refuses others", "[lc0-options]") {
    const Options base{std::filesystem::path{"cache"}};
    const auto updated = set(base, "BackendOptions", "gpu=1,threads=2,precision=fp32");
    REQUIRE(updated);
    CHECK(updated->options.backend.device_id == 1);
    CHECK(updated->options.backend.cpu_threads == 2);
    CHECK(updated->options.backend.precision == mouse::nn::ComputePrecision::Float32);
    CHECK(updated->impact == mouse::uci::Reconfigure::Backend);
    CHECK_FALSE(set(base, "BackendOptions", "made-up=1"));
    CHECK_FALSE(set(base, "BackendOptions", "gpu"));
    CHECK_FALSE(set(base, "BackendOptions", "gpu=-1"));
    CHECK_FALSE(set(base, "BackendOptions", "precision=fp8"));
}

TEST_CASE("A config file applies lc0-style lines in order", "[lc0-options]") {
    const Options base{std::filesystem::path{"cache"}};
    const auto loaded = mouse::uci::apply_config_text(base,
                                                      "# comment\r\n"
                                                      "\n"
                                                      "--weights=C:/nets/m20.onnx\n"
                                                      "  max-prefetch=64  \n"
                                                      "--minibatch-size=128\n"
                                                      "--show-wdl=true\n"
                                                      "--task-workers=-1\n"
                                                      "--max-prefetch=16\n");
    REQUIRE(loaded);
    CHECK(loaded->backend.model_path == std::filesystem::path{"C:/nets/m20.onnx"});
    CHECK(loaded->search.gather.max_prefetch == 16);
    CHECK(loaded->search.maximum_batch == 128);
    CHECK(loaded->show_wdl);
    const auto unknown = mouse::uci::apply_config_text(base, "--ok=1\n");
    REQUIRE_FALSE(unknown);
    CHECK(unknown.error().find("line 1") != std::string::npos);
    const auto invalid = mouse::uci::apply_config_text(base, "# one\n--max-prefetch=9999\n");
    REQUIRE_FALSE(invalid);
    CHECK(invalid.error().find("line 2") != std::string::npos);
    CHECK_FALSE(mouse::uci::apply_config_text(base, "-v\n"));
    CHECK_FALSE(mouse::uci::apply_config_text(base, "--threads\n"));
}

TEST_CASE("Score types follow lc0 SendUciInfo", "[lc0-options]") {
    const auto score = [](ScoreType type, float q, float draw, float with_draw_score) {
        return mouse::uci::format_score(
                   {.q = q, .draw = draw, .type = type, .draw_score_q = with_draw_score})
            ->centipawns;
    };
    constexpr float q = 0.25F, draw = 0.5F, shifted = 0.1F;
    CHECK(score(ScoreType::Centipawn, q, draw, shifted) ==
          static_cast<int>(static_cast<float>(90.0 * std::tan(1.5637541897 * q))));
    CHECK(score(ScoreType::CentipawnWithDrawScore, q, draw, shifted) ==
          static_cast<int>(static_cast<float>(90.0 * std::tan(1.5637541897 * shifted))));
    CHECK(score(ScoreType::Centipawn2018, q, draw, shifted) ==
          static_cast<int>(static_cast<float>(290.680623072 * std::tan(1.548090806 * q))));
    CHECK(score(ScoreType::Centipawn2019, q, draw, shifted) ==
          static_cast<int>(static_cast<float>(295.0 * q / (1.0 - 0.976953126 * std::pow(q, 14)))));
    CHECK(score(ScoreType::WinPercentage, q, draw, shifted) == 6250);
    CHECK(score(ScoreType::Q, q, draw, shifted) == 1000);
    CHECK(score(ScoreType::WL, q, draw, shifted) == 2500);
    for (const auto* name : {"centipawn", "centipawn_with_drawscore", "centipawn_2019",
                             "centipawn_2018", "win_percentage", "Q", "W-L", "WDL_mu"}) {
        const auto parsed = mouse::uci::parse_score_type(name);
        REQUIRE(parsed);
        CHECK(mouse::uci::score_type_name(*parsed) == name);
    }
    CHECK_FALSE(mouse::uci::parse_score_type("cp"));
}

TEST_CASE("Contempt comes back out of the reported score by objectivity", "[lc0-options]") {
    const auto plain = mouse::uci::format_score({.q = 0.2F, .draw = 0.5F});
    const auto full = mouse::uci::format_score(
        {.q = 0.2F, .draw = 0.5F, .rescale_diff = 0.2F, .rescale_sign = 1.0F});
    const auto other = mouse::uci::format_score(
        {.q = 0.2F, .draw = 0.5F, .rescale_diff = 0.2F, .rescale_sign = -1.0F});
    REQUIRE(plain);
    REQUIRE(full);
    REQUIRE(other);
    // Inverting takes the favoured side's bonus away, and adds it for the other side.
    CHECK(full->centipawns < plain->centipawns);
    CHECK(other->centipawns > plain->centipawns);
}

TEST_CASE("Time manager text selects and configures a manager", "[lc0-options]") {
    using Kind = mouse::uci::TimeManagerConfig::Kind;
    CHECK(mouse::uci::parse_time_manager("legacy")->kind == Kind::Legacy);
    const auto legacy = mouse::uci::parse_time_manager("legacy(slowmover=1.5,steepness=6)");
    REQUIRE(legacy);
    CHECK(legacy->slowmover == 1.5F);
    CHECK(legacy->steepness == 6.0F);
    CHECK(mouse::uci::parse_time_manager("alphazero(alphazero-time-pct=5)")->alphazero_time_pct ==
          5.0F);
    CHECK(mouse::uci::parse_time_manager("simple(base-pct=2)")->base_pct == 2.0F);
    CHECK_FALSE(mouse::uci::parse_time_manager("nonsense"));
    CHECK_FALSE(mouse::uci::parse_time_manager("legacy(alphazero-time-pct=5)"));
    CHECK_FALSE(mouse::uci::parse_time_manager("legacy(slowmover)"));
    CHECK_FALSE(mouse::uci::parse_time_manager("legacy(slowmover=x)"));
    CHECK_FALSE(mouse::uci::parse_time_manager("alphazero(alphazero-time-pct=101)"));
    CHECK_FALSE(mouse::uci::parse_time_manager("simple(base-pct=0)"));
    const Options base{std::filesystem::path{"cache"}};
    CHECK(set(base, "TimeManager", "simple")->options.time_manager.kind == Kind::Simple);
    CHECK_FALSE(set(base, "TimeManager", "nonsense"));
}

TEST_CASE("Alphazero and simple managers budget as lc0 does", "[lc0-options]") {
    mouse::uci::GoCommand command;
    command.wtime = 10000;
    command.winc = 100;
    mouse::uci::LegacyTimeManager alphazero{*mouse::uci::parse_time_manager("alphazero")};
    // (10000 - 200) * 12%.
    CHECK(alphazero.allocate(command, {.overhead_ms = 200})->budget == Milliseconds{1176});
    mouse::uci::LegacyTimeManager simple{*mouse::uci::parse_time_manager("simple")};
    const auto first = simple.allocate(command, {.overhead_ms = 200, .game_ply = 40});
    REQUIRE(first);
    // (10000 - 200) * ((1.4 + 40 * 0.049) * 0.01 + 100 / 10000 * 1.5) = 9800 * 0.0486.
    CHECK(first->budget == Milliseconds{476});
    // Nothing to spare: the clock fell by exactly what was budgeted, less the increment.
    command.wtime = 10000 - 476 + 100;
    const auto second = simple.allocate(command, {.overhead_ms = 200, .game_ply = 42});
    REQUIRE(second);
    CHECK(second->budget < Milliseconds{500});
    // An opening bonus only applies once.
    mouse::uci::LegacyTimeManager bonus{*mouse::uci::parse_time_manager("simple(opening-bonus-pct=100)")};
    command.wtime = 10000;
    CHECK(bonus.allocate(command, {.overhead_ms = 200, .game_ply = 40})->budget == Milliseconds{952});
}

TEST_CASE("Legacy slowmover and immediate-use follow lc0", "[lc0-options]") {
    mouse::uci::GoCommand command;
    command.wtime = 60000;
    const mouse::uci::TimeContext context{.overhead_ms = 0, .game_ply = 40};
    const auto with = [&](const char* text) {
        mouse::uci::LegacyTimeManager manager{*mouse::uci::parse_time_manager(text)};
        return manager.allocate(command, {.overhead_ms = 0, .game_ply = 12})->budget;
    };
    CHECK(with("legacy(first-move-bonus=0,book-ply-bonus=0)") < with("legacy"));
    const auto base = with("legacy(first-move-bonus=0,book-ply-bonus=0)");
    const auto slow = with("legacy(first-move-bonus=0,book-ply-bonus=0,slowmover=2)");
    // Doubled before truncating to milliseconds, so it can land one above twice the base.
    CHECK(slow->count() >= base->count() * 2);
    CHECK(slow->count() <= base->count() * 2 + 1);
    // Half of the banked time is spent now; the other half stays banked.
    mouse::uci::LegacyTimeManager half{*mouse::uci::parse_time_manager(
        "legacy(first-move-bonus=0,book-ply-bonus=0,immediate-use=0.5)")};
    const auto first = half.allocate(command, context);
    REQUIRE(first);
    half.complete(*first, Milliseconds{0});
    const auto second = half.allocate(command, context);
    const auto third_bank = half.allocate(command, context);
    REQUIRE(second);
    REQUIRE(third_bank);
    CHECK(second->budget > first->budget);
    CHECK(third_bank->budget > first->budget);
    CHECK(third_bank->budget < second->budget);
}
