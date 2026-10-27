// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/search/commands.h"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cstdint>
#include <initializer_list>
#include <string_view>
#include <utility>
#include <vector>

namespace {
mouse::Result<mouse::search::SearchCommand> parse(
    std::initializer_list<std::string_view> arguments) {
    return mouse::search::parse_search_command({arguments.begin(), arguments.size()});
}
}  // namespace
TEST_CASE("Search CLI requires explicit weights provider and precision", "[search][commands]") {
    const auto result = parse({"--weights", "model", "--backend", "cpu", "--precision", "fp32"});
    REQUIRE(result.has_value());
    CHECK(result->fen == mouse::chess::kStartFen);
    CHECK(result->moves.empty());
    CHECK(result->limits.nodes == 800);
    CHECK(result->config.maximum_batch == 64);
    CHECK(result->config.moves_left_utility);
    CHECK(result->neural.backend.batch_buckets == std::vector<std::size_t>{1, 8, 32, 64});
    CHECK_FALSE(parse({"--weights", "model", "--precision", "fp32"}));
    CHECK_FALSE(parse({"--weights", "model", "--backend", "cpu"}));
    CHECK_FALSE(parse({"--backend", "cpu", "--precision", "fp32"}));
}
TEST_CASE("Search CLI preserves explicit history and mode", "[search][commands]") {
    const auto result =
        parse({"--weights", "model", "--backend", "cpu", "--precision", "fp32", "--moves",
               "e2e4  e7e5 g1f3", "--threads", "8", "--batch", "16", "--deterministic",
               "--chess960", "--nncache-mb", "2", "--nodes", "900"});
    REQUIRE(result.has_value());
    CHECK(result->moves == std::vector<std::string>{"e2e4", "e7e5", "g1f3"});
    CHECK(result->config.deterministic);
    CHECK(result->chess960);
    CHECK(result->config.threads == 8);
    CHECK(result->config.maximum_batch == 16);
    CHECK(result->neural.backend.batch_buckets == std::vector<std::size_t>{1, 8, 16});
    CHECK(result->neural.backend.deterministic_compute);
    CHECK(result->neural.cache_bytes == 2097152);
    CHECK(result->limits.nodes == 900);
}
TEST_CASE("Search time and depth limits do not retain an implicit node cutoff",
          "[search][commands]") {
    const auto timed = parse(
        {"--weights", "model", "--backend", "cpu", "--precision", "fp32", "--movetime", "250"});
    REQUIRE(timed.has_value());
    CHECK_FALSE(timed->limits.nodes);
    CHECK(timed->limits.movetime == std::chrono::milliseconds{250});
    const auto both = parse({"--weights", "model", "--backend", "cpu", "--precision", "fp32",
                             "--nodes", "3", "--depth", "2", "--no-smart-pruning"});
    REQUIRE(both.has_value());
    CHECK(both->limits.nodes == 3);
    CHECK(both->limits.depth == 2);
    CHECK_FALSE(both->limits.smart_pruning);
}
TEST_CASE("Search CLI rejects invalid ranges duplicates and incompatible buckets",
          "[search][commands]") {
    const std::array<std::array<std::string_view, 2>, 21> failures{
        {{"--threads", "0"},
         {"--threads", "129"},
         {"--batch", "0"},
         {"--batch", "1025"},
         {"--ram-limit-mb", "0"},
         {"--ram-limit-mb", "16385"},
         {"--nodes", "0"},
         {"--nodes", "18446744073709551616"},
         {"--movetime", "0"},
         {"--movetime", "-1"},
         {"--depth", "0"},
         {"--depth", "513"},
         {"--cpuct", "nan"},
         {"--cpuct", "-1"},
         {"--cpuct-base", "0"},
         {"--cpuct-factor", "inf"},
         {"--fpu-value", "-1"},
         {"--policy-temperature", "0"},
         {"--unknown", "value"},
         {"--moves", "broken"},
         {"--fen", ""}}};
    for (const auto& pair : failures) {
        CHECK_FALSE(parse(
            {"--weights", "model", "--backend", "cpu", "--precision", "fp32", pair[0], pair[1]}));
    }
    CHECK_FALSE(parse({"--weights", "model", "--backend", "cpu", "--precision", "fp32", "--threads",
                       "2", "--threads", "3"}));
    CHECK_FALSE(parse({"--weights", "model", "--backend", "cpu", "--precision", "fp32",
                       "--deterministic", "--deterministic"}));
    CHECK_FALSE(parse({"--weights", "model", "--backend", "cpu", "--precision", "fp32", "--buckets",
                       "1,8", "--batch", "16"}));
    CHECK_FALSE(parse({"--weights", "model", "--backend", "cpu", "--precision", "fp32",
                       "--cpu-threads", "2", "--deterministic"}));
    CHECK_FALSE(
        parse({"--weights", "model", "--backend", "cpu", "--precision", "fp32", "--threads"}));
}

TEST_CASE("Search CLI offers an explicit moves-left opt-out without changing node budgets",
          "[search][commands]") {
    constexpr std::array<std::string_view, 3> budgets{"128", "1024", "8192"};
    constexpr std::array<std::uint64_t, 3> expected{128, 1024, 8192};
    for (std::size_t index = 0; index < budgets.size(); ++index) {
        const auto enabled = parse({"--weights", "model", "--backend", "cpu", "--precision", "fp32",
                                    "--nodes", budgets.at(index)});
        REQUIRE(enabled.has_value());
        CHECK(enabled->config.moves_left_utility);
        CHECK(enabled->limits.nodes == expected.at(index));
        const auto disabled =
            parse({"--weights", "model", "--backend", "cpu", "--precision", "fp32",
                   "--no-moves-left-utility", "--nodes", budgets.at(index)});
        REQUIRE(disabled.has_value());
        CHECK_FALSE(disabled->config.moves_left_utility);
        CHECK(disabled->limits.nodes == expected.at(index));
    }
    const auto last = parse({"--weights", "model", "--backend", "cpu", "--precision", "fp32",
                             "--no-moves-left-utility"});
    REQUIRE(last.has_value());
    CHECK_FALSE(last->config.moves_left_utility);
    CHECK(last->limits.nodes == 800);
    const auto duplicate = parse({"--weights", "model", "--backend", "cpu", "--precision", "fp32",
                                  "--no-moves-left-utility", "--no-moves-left-utility"});
    REQUIRE_FALSE(duplicate.has_value());
    CHECK(duplicate.error().find("Duplicate") != std::string::npos);
    CHECK_FALSE(parse({"--weights", "model", "--backend", "cpu", "--precision", "fp32",
                       "--no-moves-left-utility", "false"}));
}
TEST_CASE("Standalone search root FPU settings share the UCI defaults and explicit modes",
          "[search][commands][root-fpu]") {
    using mouse::search::RootFpuStrategy;
    const auto defaults = parse({"--weights", "model", "--backend", "cpu", "--precision", "fp32"});
    REQUIRE(defaults);
    CHECK(defaults->config.selection.root_fpu_strategy == RootFpuStrategy::Same);
    CHECK(defaults->config.selection.fpu_reduction == 0.33);
    CHECK(defaults->config.selection.root_fpu == 1.0);
    for (const auto& pair : {std::pair{"same", RootFpuStrategy::Same},
                             std::pair{"reduction", RootFpuStrategy::Reduction},
                             std::pair{"absolute", RootFpuStrategy::Absolute}}) {
        const auto configured =
            parse({"--weights", "model", "--backend", "cpu", "--precision", "fp32",
                   "--fpu-strategy-at-root", pair.first, "--fpu-value-at-root", "-0.5"});
        REQUIRE(configured);
        CHECK(configured->config.selection.root_fpu_strategy == pair.second);
        CHECK(configured->config.selection.root_fpu == -0.5);
        CHECK(configured->config.selection.fpu_reduction == 0.33);
    }
    for (const auto& pair :
         {std::pair{"--fpu-strategy-at-root", "automatic"}, std::pair{"--fpu-value-at-root", "nan"},
          std::pair{"--fpu-value-at-root", "100.001"}})
        CHECK_FALSE(parse({"--weights", "model", "--backend", "cpu", "--precision", "fp32",
                           pair.first, pair.second}));
}
