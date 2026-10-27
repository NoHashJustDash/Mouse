// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/uci/bench.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "engine/chess/attacks.h"
#include "engine/chess/movegen.h"
#include "engine/util/sha256.h"
#include "tests/required_optional.h"

namespace {
using mouse::tests::required_optional;
using mouse::uci::BenchCase;
using mouse::uci::run_bench;
const std::filesystem::path kModel =
    std::filesystem::path{MOUSE_TEST_BACKEND_DIRECTORY} / "valid.onnx";
mouse::search::SearchCommand command(std::uint64_t nodes = 17) {
    const auto filename = kModel.string();
    const std::array<std::string_view, 6> arguments{"--weights", filename,      "--backend",
                                                    "cpu",       "--precision", "fp32"};
    auto parsed = mouse::uci::parse_bench_command(arguments);
    if (!parsed) throw std::runtime_error(parsed.error());
    parsed->config.maximum_batch = 4;
    parsed->config.ram_limit_bytes = 8U << 20U;
    parsed->config.deterministic = true;
    parsed->neural.backend.batch_buckets = {1, 4};
    parsed->limits.nodes = nodes;
    return std::move(*parsed);
}
struct Provider final : mouse::nn::Backend {
    enum class Failure : std::uint8_t { None, Error, Nonfinite, Nonstandard, Identity, File };
    mouse::nn::BackendInfo identity;
    mouse::nn::ModelMetadata model;
    std::atomic<unsigned> calls{0};
    unsigned fail_at = 0;
    Failure failure = Failure::None;
    std::filesystem::path mutation_path;
    std::vector<mouse::nn::InputPlanes> first_rows;
    Provider() {
        auto digest = mouse::sha256_file(kModel);
        if (!digest) throw std::runtime_error(digest.error());
        identity.onnx_sha256 = *digest;
        identity.provider = "CPUExecutionProvider";
        identity.precision = "fp32";
        identity.device_name = "injected CPU";
        identity.cache_namespace = "bench-test";
        model.input_format = 1;
        model.source_sha256 = std::string(64, '1');
        model.converter_version = "bench-test";
    }
    mouse::Result<void> evaluate(std::span<const mouse::nn::InputPlanes> inputs,
                                 std::span<mouse::nn::NNOutput> outputs) override {
        const auto count = ++calls;
        first_rows.push_back(inputs.front());
        if (count == fail_at && failure == Failure::Error)
            return tl::unexpected(std::string{"injected bench failure"});
        if (count == fail_at && failure == Failure::Nonstandard) throw 42;
        for (auto& output : outputs) {
            output = {};
            output.wdl = {0.6F, 0.3F, 0.1F};
        }
        if (count == fail_at) {
            if (failure == Failure::Nonfinite)
                outputs.front().policy_logits[0] = std::numeric_limits<float>::infinity();
            if (failure == Failure::Identity) identity.onnx_sha256.assign(64, '0');
            if (failure == Failure::File) {
                std::ofstream changed{mutation_path, std::ios::binary | std::ios::app};
                changed << 'x';
                if (!changed) throw std::runtime_error("Cannot mutate owned test model");
            }
        }
        return {};
    }
    [[nodiscard]] const mouse::nn::BackendInfo& info() const noexcept override { return identity; }
    [[nodiscard]] const mouse::nn::ModelMetadata& metadata() const noexcept override {
        return model;
    }
};
mouse::chess::Position replay(const BenchCase& item, const mouse::chess::AttackTables& attacks) {
    auto root = mouse::chess::Position::from_fen(item.fen, attacks);
    if (!root) throw std::runtime_error(root.error());
    std::istringstream moves{std::string{item.moves}};
    for (std::string text; moves >> text;) {
        const auto move = mouse::chess::parse_uci_move(*root, text, attacks);
        if (!move) throw std::runtime_error(move.error());
        static_cast<void>(root->make_move(*move, attacks));
    }
    return std::move(*root);
}
}  // namespace

TEST_CASE("Bench CLI fixes its corpus and requires explicit neural configuration", "[uci][bench]") {
    const auto parsed = command();
    CHECK(parsed.fen == mouse::chess::kStartFen);
    CHECK(parsed.moves.empty());
    CHECK(parsed.neural.backend.kind == mouse::nn::BackendKind::Cpu);
    CHECK_FALSE(parsed.chess960);
    const std::array<std::string_view, 6> base{"--weights", "model.onnx",  "--backend",
                                               "cpu",       "--precision", "fp32"};
    const auto defaults = mouse::uci::parse_bench_command(base);
    REQUIRE(defaults);
    CHECK(defaults->limits.nodes == 800);
    CHECK_FALSE(defaults->limits.movetime);
    CHECK_FALSE(defaults->limits.depth);
    for (const std::string_view forbidden :
         {"--fen", "--moves", "--movetime", "--depth", "--chess960", "--positions", "--corpus"}) {
        auto args = std::vector<std::string_view>{base.begin(), base.end()};
        args.push_back(forbidden);
        args.emplace_back("1");
        const auto rejected = mouse::uci::parse_bench_command(args);
        REQUIRE_FALSE(rejected);
        CHECK(rejected.error().find(forbidden) != std::string::npos);
    }
    const std::array<std::string_view, 6> option_named_file{"--weights", "--fen",       "--backend",
                                                            "cpu",       "--precision", "fp32"};
    REQUIRE(mouse::uci::parse_bench_command(option_named_file));
    CHECK_FALSE(mouse::uci::parse_bench_command({}));
    auto args = std::vector<std::string_view>{base.begin(), base.end()};
    args.insert(args.end(), {"--nodes", "0"});
    CHECK_FALSE(mouse::uci::parse_bench_command(args));
    args.back() = "37";
    const auto explicit_nodes = mouse::uci::parse_bench_command(args);
    REQUIRE(explicit_nodes);
    CHECK(explicit_nodes->limits.nodes == 37);
}

TEST_CASE("Fixed bench corpus matches its tracked records and every PV is legal", "[uci][bench]") {
    const auto cases = mouse::uci::fixed_bench_cases();
    REQUIRE(cases.size() == 16);
    std::ifstream input{std::filesystem::path{MOUSE_TEST_BENCH_DIRECTORY} / "positions-v1.tsv"};
    REQUIRE(input.good());
    std::size_t row = 0;
    for (std::string line; std::getline(input, line);) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line.starts_with('#')) continue;
        REQUIRE(row < cases.size());
        const auto& item = cases[row++];
        CHECK(line ==
              std::string{item.id} + '\t' + std::string{item.fen} + '\t' + std::string{item.moves});
    }
    CHECK(row == cases.size());
    Provider provider;
    const auto report = run_bench(provider, command(), cases);
    REQUIRE(report);
    CHECK(report->cases.size() == cases.size());
    CHECK(report->nodes_per_case == 17);
    CHECK(report->nodes == std::uint64_t{16} * 17);
    CHECK(report->backend.onnx_sha256 == provider.identity.onnx_sha256);
    CHECK(report->corpus_sha256.size() == 64);
    CHECK(report->setup.count() >= 0);
    CHECK(report->search.count() > 0);
    CHECK(report->wall >= report->search + report->setup);
    CHECK(report->nps > 0);
    const mouse::chess::AttackTables attacks;
    std::size_t black = 0, histories = 0;
    for (std::size_t i = 0; i < cases.size(); ++i) {
        auto root = replay(cases[i], attacks);
        black += root.side_to_move() == mouse::chess::Color::Black ? 1U : 0U;
        histories += cases[i].moves.empty() ? 0U : 1U;
        const auto& result = report->cases.at(i);
        CHECK(result.id == cases[i].id);
        CHECK(result.fen == root.fen());
        CHECK(result.result.info.nodes == 17);
        CHECK(result.result.info.new_playouts == 17);
        CHECK(result.result.info.inherited_visits == 0);
        CHECK(result.result.reason == mouse::search::StopReason::NodeLimit);
        REQUIRE(result.result.best_move);
        const auto legal = mouse::chess::legal_moves(root, attacks);
        if (cases[i].id.starts_with("ep-"))
            CHECK(std::ranges::any_of(
                legal, [](auto move) { return move.kind() == mouse::chess::MoveKind::EnPassant; }));
        if (cases[i].id.starts_with("castle-"))
            CHECK(std::ranges::any_of(
                legal, [](auto move) { return move.kind() == mouse::chess::MoveKind::Castle; }));
        if (cases[i].id.starts_with("promotion-"))
            CHECK(std::ranges::any_of(legal, [](auto move) { return move.is_promotion(); }));
        if (cases[i].id == "pawnless") {
            CHECK(root.pieces(mouse::chess::Color::White, mouse::chess::PieceType::Pawn) == 0);
            CHECK(root.pieces(mouse::chess::Color::Black, mouse::chess::PieceType::Pawn) == 0);
        }
        if (cases[i].id == "rule50-99") CHECK(root.halfmove_clock() == 99);
        if (cases[i].id == "history-repeat") CHECK(root.is_repetition(2));
        CHECK(std::ranges::find(legal, required_optional(result.result.best_move)) != legal.end());
        for (std::size_t j = 0; j < result.result.info.pv.size; ++j) {
            const auto next = result.result.info.pv.moves.at(j);
            const auto allowed = mouse::chess::legal_moves(root, attacks);
            CHECK(std::ranges::find(allowed, next) != allowed.end());
            static_cast<void>(root.make_move(next, attacks));
        }
    }
    CHECK(black >= 4);
    CHECK(histories >= 4);
}

TEST_CASE("Each bench case starts with fresh tree and NN cache", "[uci][bench]") {
    const std::array cases{BenchCase{.id = "first", .fen = mouse::chess::kStartFen, .moves = ""},
                           BenchCase{.id = "second", .fen = mouse::chess::kStartFen, .moves = ""}};
    Provider provider;
    auto configured = command(1);
    configured.neural.cache_bytes = 1U << 20U;
    const auto report = run_bench(provider, configured, cases);
    REQUIRE(report);
    CHECK(provider.calls == 2);
    REQUIRE(provider.first_rows.size() == 2);
    CHECK(provider.first_rows.at(0) == provider.first_rows.at(1));
    CHECK(report->cases.at(0).result.info.inherited_visits == 0);
    CHECK(report->cases.at(1).result.info.inherited_visits == 0);
    CHECK(report->nodes == 2);
}

TEST_CASE("Bench failure after a valid case returns no partial report", "[uci][bench]") {
    const std::array cases{
        BenchCase{.id = "valid-first", .fen = mouse::chess::kStartFen, .moves = ""},
        BenchCase{.id = "failing-second", .fen = mouse::chess::kStartFen, .moves = "e2e4"}};
    for (const auto failure : {Provider::Failure::Error, Provider::Failure::Nonfinite,
                               Provider::Failure::Nonstandard, Provider::Failure::Identity}) {
        Provider provider;
        provider.fail_at = 2;
        provider.failure = failure;
        const auto report = run_bench(provider, command(1), cases);
        REQUIRE_FALSE(report);
        CHECK(report.error().find("failing-second") != std::string::npos);
        CHECK(provider.calls == 2);
    }
}

TEST_CASE("Bench rejects changed corpus controls and invalid cases explicitly", "[uci][bench]") {
    Provider provider;
    const std::array valid{BenchCase{.id = "valid", .fen = mouse::chess::kStartFen, .moves = ""}};
    auto configured = command(1);
    CHECK_FALSE(run_bench(provider, configured, {}));
    const std::array duplicate{valid.front(), valid.front()};
    CHECK_FALSE(run_bench(provider, configured, duplicate));
    const std::array bad_fen{BenchCase{.id = "bad-fen", .fen = "broken", .moves = ""}};
    CHECK_FALSE(run_bench(provider, configured, bad_fen));
    const std::array illegal{
        BenchCase{.id = "illegal-move", .fen = mouse::chess::kStartFen, .moves = "e2e5"}};
    CHECK_FALSE(run_bench(provider, configured, illegal));
    configured.limits.movetime = std::chrono::milliseconds{1};
    CHECK_FALSE(run_bench(provider, configured, valid));
    configured = command(1);
    configured.limits.depth = 1;
    CHECK_FALSE(run_bench(provider, configured, valid));
    configured = command(1);
    configured.limits.nodes.reset();
    CHECK_FALSE(run_bench(provider, configured, valid));
    configured = command(1);
    configured.moves.emplace_back("e2e4");
    CHECK_FALSE(run_bench(provider, configured, valid));
    configured = command(1);
    configured.chess960 = true;
    CHECK_FALSE(run_bench(provider, configured, valid));
    CHECK(provider.calls == 0);
}

TEST_CASE("Bench rejects aggregate counter overflow before inference", "[uci][bench]") {
    Provider provider;
    const std::array cases{BenchCase{.id = "first", .fen = mouse::chess::kStartFen, .moves = ""},
                           BenchCase{.id = "second", .fen = mouse::chess::kStartFen, .moves = ""}};
    const auto report =
        run_bench(provider, command(std::numeric_limits<std::uint64_t>::max()), cases);
    REQUIRE_FALSE(report);
    CHECK(report.error().find("overflow") != std::string::npos);
    CHECK(provider.calls == 0);
}
TEST_CASE("Bench preserves full history separately from an identical final FEN", "[uci][bench]") {
    const std::array cases{
        BenchCase{.id = "history", .fen = mouse::chess::kStartFen, .moves = "g1f3 g8f6 f3g1 f6g8"},
        BenchCase{.id = "fen-only",
                  .fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 4 3",
                  .moves = ""}};
    Provider provider;
    const auto report = run_bench(provider, command(1), cases);
    REQUIRE(report);
    CHECK(report->corpus_version == "MouseBenchCustom1");
    CHECK(report->cases.at(0).fen == report->cases.at(1).fen);
    CHECK(report->cases.at(0).input_sha256 != report->cases.at(1).input_sha256);
    CHECK(report->cases.at(0).history_frames == 5);
    CHECK(report->cases.at(1).history_frames == 1);
    REQUIRE(provider.first_rows.size() == 2);
    CHECK(provider.first_rows.at(0) != provider.first_rows.at(1));
}

TEST_CASE("Bench audits actual model bytes after all case evaluations", "[uci][bench]") {
    struct OwnedFile {
        std::filesystem::path directory, model;
        OwnedFile() {
            const auto seed = std::chrono::steady_clock::now().time_since_epoch().count();
            for (unsigned attempt = 0; attempt < 32; ++attempt) {
                directory = std::filesystem::temp_directory_path() /
                            ("mouse-bench-" + std::to_string(seed) + '-' + std::to_string(attempt));
                if (std::filesystem::create_directory(directory)) {
                    model = directory / "model.onnx";
                    std::filesystem::copy_file(kModel, model);
                    return;
                }
            }
            throw std::runtime_error("Cannot create owned bench fixture directory");
        }
        ~OwnedFile() {
            std::error_code error;
            std::filesystem::remove(model, error);
            std::filesystem::remove(directory, error);
        }
        OwnedFile(const OwnedFile&) = delete;
        OwnedFile& operator=(const OwnedFile&) = delete;
        OwnedFile(OwnedFile&&) = delete;
        OwnedFile& operator=(OwnedFile&&) = delete;
    } file;
    Provider provider;
    provider.fail_at = 1;
    provider.failure = Provider::Failure::File;
    provider.mutation_path = file.model;
    auto configured = command(1);
    configured.neural.backend.model_path = file.model;
    const std::array cases{
        BenchCase{.id = "changed-model", .fen = mouse::chess::kStartFen, .moves = ""}};
    const auto report = run_bench(provider, configured, cases);
    REQUIRE_FALSE(report);
    CHECK(report.error().find("changed") != std::string::npos);
    CHECK(provider.calls == 1);
}

TEST_CASE("Bench detects explicit provider mismatch before any inference", "[uci][bench]") {
    Provider provider;
    provider.identity.provider = "CUDAExecutionProvider";
    const std::array cases{BenchCase{.id = "start", .fen = mouse::chess::kStartFen, .moves = ""}};
    const auto report = run_bench(provider, command(1), cases);
    REQUIRE_FALSE(report);
    CHECK(report.error().find("provider/precision") != std::string::npos);
    CHECK(provider.calls == 0);
}
TEST_CASE("Bench process boundary rejects corpus overrides without success output",
          "[uci][bench]") {
    struct Capture {
        std::ostringstream output, error;
        std::streambuf* previous_output = std::cout.rdbuf(output.rdbuf());
        std::streambuf* previous_error = std::cerr.rdbuf(error.rdbuf());
        Capture() = default;
        ~Capture() noexcept {
            try {
                std::cout.rdbuf(previous_output);
                std::cerr.rdbuf(previous_error);
            } catch (...) {
                std::terminate();
            }
        }
        Capture(const Capture&) = delete;
        Capture& operator=(const Capture&) = delete;
        Capture(Capture&&) = delete;
        Capture& operator=(Capture&&) = delete;
    };
    // Only rejected commands run while streams are captured; no provider/search
    // thread starts or remains alive across this process-resource test.
    for (const auto* option : {"--fen", "--moves", "--movetime", "--depth", "--chess960"}) {
        std::vector<std::string> owned{"--weights",   "missing.onnx", "--backend", "cpu",
                                       "--precision", "fp32",         option,      "1"};
        std::vector<char*> arguments;
        arguments.reserve(owned.size());
        for (auto& value : owned) arguments.push_back(value.data());
        Capture capture;
        CHECK(mouse::uci::run_bench_command(arguments) == 2);
        CHECK(capture.output.str().empty());
        CHECK(capture.error.str().find(option) != std::string::npos);
    }
}
TEST_CASE("Bench corpus scanning recognizes shared boolean flags", "[uci][bench]") {
    const auto filename = kModel.string();
    for (const std::string_view flag :
         {"--deterministic", "--no-smart-pruning", "--no-moves-left-utility"}) {
        const std::array<std::string_view, 9> args{
            "--weights", filename,      "--backend",
            "cpu",       "--precision", "fp32",
            flag,        "--fen",       mouse::chess::kStartFen};
        CHECK_FALSE(mouse::uci::parse_bench_command(args));
    }
}
