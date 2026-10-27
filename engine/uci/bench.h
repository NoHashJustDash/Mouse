// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <chrono>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "engine/search/commands.h"

namespace mouse::uci {
inline constexpr std::string_view kBenchCorpusVersion = "MouseBenchPositions1";
struct BenchCase {
    std::string_view id;
    std::string_view fen;
    std::string_view moves;
};
struct BenchCaseReport {
    std::string id, initial_fen, moves, fen, input_sha256;
    std::size_t history_frames = 0;
    search::SearchResult result;
    std::chrono::nanoseconds setup{}, search{};
};
struct BenchReport {
    nn::BackendInfo backend;
    nn::ModelMetadata model;
    search::SearchCommand configuration;
    std::string corpus_version, corpus_sha256;
    std::vector<BenchCaseReport> cases;
    std::uint64_t nodes_per_case = 0, nodes = 0;
    std::chrono::nanoseconds setup{}, search{}, wall{};
    // Set by the standalone owner; borrowed-backend run_bench leaves these zero.
    std::chrono::nanoseconds backend_setup{}, command_wall{};
    double nps = 0;
};
[[nodiscard]] std::span<const BenchCase> fixed_bench_cases() noexcept;
[[nodiscard]] Result<search::SearchCommand> parse_bench_command(
    std::span<const std::string_view> arguments);
// One synchronous caller owns the borrowed backend for this entire operation.
// Every case owns a fresh logical root/tree/cache; there is no retained reuse.
// setup includes replay/cache/tree construction; search times only run(), before
// validation/destruction. wall includes audits/setup/search/validation/cleanup,
// excluding caller-owned backend construction/warmup. All errors return no report.
// A custom injected corpus is labeled MouseBenchCustom1, never the fixed version.
[[nodiscard]] Result<BenchReport> run_bench(nn::Backend&, const search::SearchCommand&,
                                            std::span<const BenchCase>);
// Process resources (stdout/stderr) require an externally serialized caller.
// Publishes a complete report only after every case and identity audit succeeds.
int run_bench_command(std::span<char*> arguments);
}  // namespace mouse::uci