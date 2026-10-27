// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/uci/bench.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <memory>
#include <sstream>
#include <tuple>
#include <utility>

#include "engine/chess/attacks.h"
#include "engine/chess/movegen.h"
#include "engine/nn/cached_backend.h"
#include "engine/nn/command_runner.h"
#include "engine/util/build_info.h"
#include "engine/util/sha256.h"

namespace mouse::uci {
namespace {
using Clock = std::chrono::steady_clock;
constexpr std::size_t kMaximumCases = 256;
constexpr std::array kCases{
    BenchCase{.id = "start",
              .fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
              .moves = ""},
    BenchCase{.id = "queens-pawn",
              .fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
              .moves = "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6"},
    BenchCase{.id = "castle-white",
              .fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
              .moves = "e2e4 e7e5 g1f3 b8c6 f1c4 g8f6"},
    BenchCase{.id = "castle-both",
              .fen = "r3k2r/ppp2ppp/2n5/3pp3/3PP3/2N5/PPP2PPP/R3K2R w KQkq - 0 1",
              .moves = ""},
    BenchCase{.id = "castle-black",
              .fen = "r3k2r/ppp2ppp/2n5/3pp3/3PP3/2N5/PPP2PPP/R3K2R b KQkq - 0 1",
              .moves = ""},
    BenchCase{.id = "ep-white", .fen = "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2", .moves = ""},
    BenchCase{.id = "ep-black", .fen = "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 2", .moves = ""},
    BenchCase{.id = "promotion-white", .fen = "4k3/P7/8/8/8/8/7p/4K3 w - - 0 1", .moves = ""},
    BenchCase{.id = "promotion-black", .fen = "4k3/P7/8/8/8/8/7p/4K3 b - - 0 1", .moves = ""},
    BenchCase{.id = "pawnless", .fen = "8/8/3k4/8/8/4K3/6R1/7r w - - 12 40", .moves = ""},
    BenchCase{.id = "black-pawns", .fen = "8/5pk1/6p1/8/3P4/5K2/8/8 b - - 0 30", .moves = ""},
    BenchCase{.id = "rule50-99", .fen = "7k/8/8/8/8/2K5/8/R7 w - - 99 70", .moves = ""},
    BenchCase{.id = "history-repeat",
              .fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
              .moves = "g1f3 g8f6 f3g1 f6g8"},
    BenchCase{.id = "history-pawn-push",
              .fen = "4k3/8/8/8/8/8/3P4/4K3 w - - 0 1",
              .moves = "d2d4 e8f7 e1f2 f7e6"},
    BenchCase{.id = "history-capture",
              .fen = "4k3/8/8/8/8/8/4r3/R3K3 w Q - 0 1",
              .moves = "e1e2 e8d7 a1a7 d7c6"},
    BenchCase{.id = "spanish-black",
              .fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
              .moves = "e2e4 e7e5 g1f3 b8c6 f1b5"},
};
auto identity(const nn::BackendInfo& value) {
    return std::tie(value.onnx_sha256, value.provider, value.precision, value.device_name,
                    value.device_uuid, value.driver_version, value.cuda_version,
                    value.tensorrt_version, value.cache_namespace);
}
std::string_view provider_name(nn::BackendKind kind) {
    switch (kind) {
        case nn::BackendKind::Cpu:
            return "CPUExecutionProvider";
        case nn::BackendKind::Cuda:
            return "CUDAExecutionProvider";
        case nn::BackendKind::TensorRt:
            return "TensorrtExecutionProvider";
    }
    return {};
}
std::string_view precision_name(nn::ComputePrecision precision) {
    switch (precision) {
        case nn::ComputePrecision::Float32:
            return "fp32";
        case nn::ComputePrecision::Float16:
            return "fp16";
    }
    return {};
}
bool is_fixed(std::span<const BenchCase> cases) {
    return cases.size() == kCases.size() &&
           std::equal(cases.begin(), cases.end(), kCases.begin(), [](const auto& a, const auto& b) {
               return std::tie(a.id, a.fen, a.moves) == std::tie(b.id, b.fen, b.moves);
           });
}
Result<void> validate_command(const search::SearchCommand& command) {
    const auto& limit = command.limits;
    if (!limit.nodes || *limit.nodes == 0 || limit.movetime || limit.depth ||
        limit.searchmoves_count != 0 || limit.stop_token.stop_possible() || command.chess960 ||
        command.fen != chess::kStartFen || !command.moves.empty())
        return tl::unexpected(std::string{
            "Bench requires only a fixed positive node limit and its immutable corpus"});
    if (command.neural.backend.model_path.empty() ||
        provider_name(command.neural.backend.kind).empty() ||
        precision_name(command.neural.backend.precision).empty())
        return tl::unexpected(
            std::string{"Bench requires explicit valid model/provider/precision"});
    if (command.neural.backend.batch_buckets.empty() ||
        command.neural.backend.batch_buckets.back() < command.config.maximum_batch)
        return tl::unexpected(std::string{"Bench backend buckets must cover the search batch"});
    return {};
}
Result<void> validate_cases(std::span<const BenchCase> cases) {
    if (cases.empty() || cases.size() > kMaximumCases)
        return tl::unexpected(std::string{"Bench requires 1..256 cases"});
    for (std::size_t i = 0; i < cases.size(); ++i) {
        const auto& item = cases[i];
        if (item.id.empty() || item.id.size() > 64 ||
            std::ranges::any_of(item.id,
                                [](unsigned char c) {
                                    return !((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                                             (c >= '0' && c <= '9') || c == '-' || c == '_');
                                }) ||
            item.fen.empty() || item.fen.size() > 256 || item.moves.size() > (1U << 20U))
            return tl::unexpected(std::string{"Bench case has invalid ID/FEN/history length"});
        for (std::size_t j = 0; j < i; ++j)
            if (cases[j].id == item.id)
                return tl::unexpected("Bench case has duplicate ID: " + std::string{item.id});
    }
    return {};
}
// The per-case cache owns this lifetime bridge, never the borrowed backend.
// run_bench drains/destroys every engine/cache before returning to its one caller.
class BorrowedBackend final : public nn::Backend {
   public:
    explicit BorrowedBackend(nn::Backend& backend) : backend_(backend) {}
    Result<void> evaluate(std::span<const nn::InputPlanes> inputs,
                          std::span<nn::NNOutput> outputs) override {
        return backend_.evaluate(inputs, outputs);
    }
    [[nodiscard]] const nn::BackendInfo& info() const noexcept override { return backend_.info(); }
    [[nodiscard]] const nn::ModelMetadata& metadata() const noexcept override {
        return backend_.metadata();
    }

   private:
    nn::Backend& backend_;
};
struct Prepared {
    chess::Position root;
    nn::EncodingHistory history;
    std::string input_sha256;
};
Result<Prepared> prepare(const BenchCase& item, const chess::AttackTables& attacks) {
    auto root = chess::Position::from_fen(item.fen, attacks);
    if (!root) return tl::unexpected("Invalid FEN: " + root.error());
    const auto variant = chess::validate_uci_castling_mode(*root, false);
    if (!variant) return tl::unexpected(variant.error());
    nn::EncodingHistory history;
    history.reset(*root);
    std::string input_identity = "MouseSearchInput1\nchess960=0\n" + root->fen() + '\n';
    std::istringstream words{std::string{item.moves}};
    std::size_t count = 0;
    for (std::string text; words >> text;) {
        if (++count > 131072)
            return tl::unexpected(std::string{"Bench history exceeds 131072 plies"});
        const auto move = chess::parse_uci_move(*root, text, attacks);
        if (!move) return tl::unexpected("Illegal history move " + text + ": " + move.error());
        input_identity += chess::move_to_uci(*move) + '\n';
        static_cast<void>(root->make_move(*move, attacks));
        const auto appended = history.append(*root);
        if (!appended) return tl::unexpected(appended.error());
    }
    return Prepared{.root = std::move(*root),
                    .history = std::move(history),
                    .input_sha256 = sha256(std::as_bytes(std::span{input_identity}))};
}
Result<void> validate_result(const search::SearchResult& result, chess::Position root,
                             const chess::AttackTables& attacks, std::uint64_t nodes) {
    const auto& info = result.info;
    if (result.reason != search::StopReason::NodeLimit || info.nodes != nodes ||
        info.new_playouts != nodes || info.inherited_visits != 0)
        return tl::unexpected(std::string{"Search did not complete the fixed node budget"});
    if (!std::isfinite(info.q) || !std::isfinite(info.draw) || info.q < -1 || info.q > 1 ||
        info.draw < 0 || info.draw > 1 || info.pv.size > info.pv.moves.size() ||
        info.depth != info.pv.size || info.depth > info.seldepth ||
        info.charged_bytes > info.budget_bytes || info.tree_nodes > info.tree_capacity)
        return tl::unexpected(std::string{"Search returned invalid/nonfinite bench statistics"});
    const auto legal = chess::legal_moves(root, attacks);
    if (!result.best_move || std::ranges::find(legal, *result.best_move) == legal.end() ||
        (info.pv.size != 0 && info.pv.moves[0] != *result.best_move))
        return tl::unexpected(std::string{"Search returned an invalid bench best move"});
    for (std::size_t i = 0; i < info.pv.size; ++i) {
        const auto move = info.pv.moves.at(i);
        const auto allowed = chess::legal_moves(root, attacks);
        if (std::ranges::find(allowed, move) == allowed.end())
            return tl::unexpected(std::string{"Search returned an illegal bench PV"});
        static_cast<void>(root.make_move(move, attacks));
    }
    return {};
}
Result<void> add_duration(std::chrono::nanoseconds& sum, std::chrono::nanoseconds delta) {
    if (delta.count() < 0 || delta > std::chrono::nanoseconds::max() - sum)
        return tl::unexpected(std::string{"Bench timing overflow"});
    sum += delta;
    return {};
}
std::string json_string(std::string_view text) {
    constexpr std::string_view hex = "0123456789abcdef";
    std::string value{"\""};
    for (const char character : text) {
        const auto c = static_cast<unsigned char>(character);
        if (c == '"' || c == '\\') {
            value += '\\';
            value += static_cast<char>(c);
        } else if (c < 32) {
            value += "\\u00";
            value += hex[c >> 4U];
            value += hex[c & 15U];
        } else
            value += static_cast<char>(c);
    }
    value += '"';
    return value;
}
std::string path_text(const std::filesystem::path& path) {
    std::string text;
    for (const auto c : path.generic_u8string()) text += static_cast<char>(c);
    return text;
}
std::string render(const BenchReport& report) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    const auto& b = report.backend;
    const auto& m = report.model;
    const auto& c = report.configuration;
    const auto& network = c.neural.backend;
    const auto& selection = c.config.selection;
    out << std::setprecision(17) << R"({"schema":"MouseBench1","corpus_version":)"
        << json_string(report.corpus_version)
        << ",\"corpus_sha256\":" << json_string(report.corpus_sha256)
        << ",\"version\":" << json_string(build::version)
        << ",\"git\":" << json_string(build::git_hash)
        << ",\"build_type\":" << json_string(build::build_type)
        << ",\"compiler\":" << json_string(build::compiler)
        << ",\"ort_version\":" << json_string(build::required_ort_version)
        << R"(,"backend":{"model_sha256":)" << json_string(b.onnx_sha256)
        << ",\"provider\":" << json_string(b.provider)
        << ",\"precision\":" << json_string(b.precision)
        << ",\"device_name\":" << json_string(b.device_name)
        << ",\"device_uuid\":" << json_string(b.device_uuid)
        << ",\"driver_version\":" << json_string(b.driver_version)
        << ",\"cuda_version\":" << json_string(b.cuda_version)
        << ",\"tensorrt_version\":" << json_string(b.tensorrt_version)
        << ",\"cache_namespace\":" << json_string(b.cache_namespace)
        << R"(},"model":{"source_sha256":)" << json_string(m.source_sha256)
        << ",\"converter_version\":" << json_string(m.converter_version)
        << ",\"precision\":" << json_string(m.precision) << ",\"input_format\":" << m.input_format
        << ",\"network_format\":" << m.network_format
        << ",\"raw_network_format\":" << m.raw_network_format
        << ",\"parameter_count\":" << m.parameter_count << R"(},"configuration":{"weights":)"
        << json_string(path_text(network.model_path))
        << ",\"runtime\":" << json_string(path_text(network.runtime_path))
        << ",\"cache_directory\":" << json_string(path_text(network.cache_directory))
        << ",\"device\":" << network.device_id << ",\"cpu_threads\":" << network.cpu_threads
        << ",\"workspace_bytes\":" << network.workspace_bytes
        << ",\"backend_deterministic\":" << (network.deterministic_compute ? "true" : "false")
        << ",\"nncache_bytes\":" << c.neural.cache_bytes << ",\"threads\":" << c.config.threads
        << ",\"effective_threads\":" << (c.config.deterministic ? 1 : c.config.threads)
        << ",\"deterministic\":" << (c.config.deterministic ? "true" : "false")
        << ",\"moves_left_utility\":" << (c.config.moves_left_utility ? "true" : "false")
        << ",\"batch\":" << c.config.maximum_batch
        << ",\"ram_limit_bytes\":" << c.config.ram_limit_bytes
        << ",\"max_depth\":" << c.config.max_depth << ",\"cpuct\":" << selection.cpuct
        << ",\"cpuct_base\":" << selection.cpuct_base
        << ",\"cpuct_factor\":" << selection.cpuct_factor
        << ",\"fpu_value\":" << selection.fpu_reduction
        << ",\"policy_temperature\":" << selection.policy_temperature << ",\"buckets\":[";
    for (std::size_t i = 0; i < network.batch_buckets.size(); ++i) {
        if (i != 0) out << ',';
        out << network.batch_buckets.at(i);
    }
    out << "],\"gpu_dll_directories\":[";
    for (std::size_t i = 0; i < network.gpu_dll_directories.size(); ++i) {
        if (i != 0) out << ',';
        out << json_string(path_text(network.gpu_dll_directories.at(i)));
    }
    out << R"(]},"timing_boundary":"SearchEngine::run","backend_setup_ns":)"
        << report.backend_setup.count() << ",\"setup_ns\":" << report.setup.count()
        << ",\"search_ns\":" << report.search.count()
        << ",\"bench_wall_ns\":" << report.wall.count()
        << ",\"command_wall_ns\":" << report.command_wall.count()
        << ",\"nodes_per_case\":" << report.nodes_per_case << ",\"nodes\":" << report.nodes
        << ",\"nps\":" << report.nps << ",\"cases\":[";
    for (std::size_t i = 0; i < report.cases.size(); ++i) {
        const auto& row = report.cases.at(i);
        const auto& info = row.result.info;
        const auto best_move = row.result.best_move;
        if (!best_move.has_value())
            throw std::logic_error("Bench result omitted its legal root move");
        if (i != 0) out << ',';
        out << "{\"id\":" << json_string(row.id)
            << ",\"initial_fen\":" << json_string(row.initial_fen)
            << ",\"moves\":" << json_string(row.moves) << ",\"fen\":" << json_string(row.fen)
            << ",\"input_sha256\":" << json_string(row.input_sha256)
            << ",\"history_frames\":" << row.history_frames << ",\"nodes\":" << info.nodes
            << ",\"new_playouts\":" << info.new_playouts
            << ",\"inherited_visits\":" << info.inherited_visits << R"(,"stop_reason":"nodes")"
            << ",\"depth\":" << info.depth << ",\"seldepth\":" << info.seldepth
            << ",\"q\":" << info.q << ",\"draw\":" << info.draw
            << ",\"charged_bytes\":" << info.charged_bytes
            << ",\"budget_bytes\":" << info.budget_bytes << ",\"tree_nodes\":" << info.tree_nodes
            << ",\"tree_capacity\":" << info.tree_capacity << ",\"setup_ns\":" << row.setup.count()
            << ",\"search_ns\":" << row.search.count()
            << ",\"bestmove\":" << json_string(chess::move_to_uci(*best_move)) << ",\"pv\":[";
        for (std::size_t j = 0; j < info.pv.size; ++j) {
            if (j != 0) out << ',';
            out << json_string(chess::move_to_uci(info.pv.moves.at(j)));
        }
        out << "]}";
    }
    out << "]}\n";
    return out.str();
}
}  // namespace

std::span<const BenchCase> fixed_bench_cases() noexcept { return kCases; }
Result<search::SearchCommand> parse_bench_command(std::span<const std::string_view> arguments) {
    for (std::size_t index = 0; index < arguments.size(); ++index) {
        const auto option = arguments[index];
        if (option == "--fen" || option == "--moves" || option == "--movetime" ||
            option == "--depth" || option == "--chess960" || option == "--positions" ||
            option == "--corpus")
            return tl::unexpected("Bench forbids changing its fixed corpus/limit with " +
                                  std::string{option});
        // Inspect option positions only: an explicit model/path may itself be
        // named --fen. The shared parser still validates unknown/missing values.
        if (option != "--deterministic" && option != "--no-smart-pruning" &&
            option != "--no-moves-left-utility" && index + 1 < arguments.size())
            ++index;
    }
    auto command = search::parse_search_command(arguments);
    if (!command) return tl::unexpected(command.error());
    const auto valid = validate_command(*command);
    if (!valid) return tl::unexpected(valid.error());
    return command;
}
Result<BenchReport> run_bench(nn::Backend& backend, const search::SearchCommand& command,
                              std::span<const BenchCase> cases) {
    try {
        const auto wall_start = Clock::now();
        const auto valid = validate_command(command);
        if (!valid) return tl::unexpected(valid.error());
        const auto corpus_valid = validate_cases(cases);
        if (!corpus_valid) return tl::unexpected(corpus_valid.error());
        const auto nodes = command.limits.nodes;
        if (!nodes.has_value()) return tl::unexpected(std::string{"Bench requires a node limit"});
        const auto budget = *nodes;
        if (budget > std::numeric_limits<std::uint64_t>::max() / cases.size())
            return tl::unexpected(std::string{"Bench aggregate node counter would overflow"});
        BenchReport report;
        report.backend = backend.info();
        report.model = backend.metadata();
        report.configuration = command;
        report.nodes_per_case = budget;
        report.corpus_version = is_fixed(cases) ? kBenchCorpusVersion : "MouseBenchCustom1";
        if (report.backend.provider != provider_name(command.neural.backend.kind) ||
            report.backend.precision != precision_name(command.neural.backend.precision))
            return tl::unexpected(std::string{
                "Bench backend provider/precision differs from explicit configuration"});
        const auto initial_hash = sha256_file(command.neural.backend.model_path);
        if (!initial_hash) return tl::unexpected(initial_hash.error());
        if (*initial_hash != report.backend.onnx_sha256)
            return tl::unexpected(std::string{"Bench model SHA256 differs from backend snapshot"});
        std::string corpus_identity = report.corpus_version + '\n';
        for (const auto& item : cases)
            corpus_identity += std::string{item.id} + '\t' + std::string{item.fen} + '\t' +
                               std::string{item.moves} + '\n';
        report.corpus_sha256 = sha256(std::as_bytes(std::span{corpus_identity}));
        report.cases.reserve(cases.size());
        const chess::AttackTables attacks;
        for (const auto& item : cases) {
            const auto failed = [&](std::string_view message) -> Result<BenchReport> {
                return tl::unexpected("Bench case " + std::string{item.id} + ": " +
                                      std::string{message});
            };
            const auto setup_start = Clock::now();
            auto prepared = prepare(item, attacks);
            if (!prepared) return failed(prepared.error());
            auto cached =
                nn::create_cached_backend(std::make_unique<BorrowedBackend>(backend),
                                          command.neural.cache_bytes, command.config.maximum_batch);
            if (!cached) return failed(cached.error());
            auto engine = search::SearchEngine::create(command.config, attacks, **cached);
            if (!engine) return failed(engine.error());
            const auto configured = (*engine)->set_position(prepared->root, prepared->history);
            if (!configured) return failed(configured.error());
            const auto run_start = Clock::now();
            const auto result = (*engine)->run(command.limits);
            const auto run_end = Clock::now();
            if (!result) return failed(result.error());
            const auto checked = validate_result(*result, prepared->root, attacks, budget);
            if (!checked) return failed(checked.error());
            if (identity(report.backend) != identity(backend.info()) ||
                report.model != backend.metadata())
                return failed("Backend identity/metadata changed during bench");
            BenchCaseReport row{
                .id = std::string{item.id},
                .initial_fen = std::string{item.fen},
                .moves = std::string{item.moves},
                .fen = prepared->root.fen(),
                .input_sha256 = std::move(prepared->input_sha256),
                .history_frames = prepared->history.frames().size(),
                .result = *result,
                .setup =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(run_start - setup_start),
                .search =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(run_end - run_start)};
            auto summed = add_duration(report.setup, row.setup);
            if (!summed) return failed(summed.error());
            summed = add_duration(report.search, row.search);
            if (!summed) return failed(summed.error());
            report.nodes += result->info.nodes;  // Preflight budget*case_count cannot overflow.
            report.cases.push_back(std::move(row));
        }
        const auto final_hash = sha256_file(command.neural.backend.model_path);
        if (!final_hash) return tl::unexpected(final_hash.error());
        if (*final_hash != *initial_hash || identity(report.backend) != identity(backend.info()) ||
            report.model != backend.metadata())
            return tl::unexpected(std::string{"Bench model/identity changed during evaluation"});
        report.wall =
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - wall_start);
        report.nps = report.search.count() == 0 ? 0
                                                : static_cast<double>(report.nodes) * 1.0e9 /
                                                      static_cast<double>(report.search.count());
        if (!std::isfinite(report.nps))
            return tl::unexpected(std::string{"Bench produced nonfinite throughput"});
        return report;
    } catch (const std::exception& error) {
        return tl::unexpected(std::string{"Bench failed: "} + error.what());
    } catch (...) {
        return tl::unexpected(std::string{"Bench failed with a non-standard exception"});
    }
}
int run_bench_command(std::span<char*> arguments) {
    try {
        const auto command_start = Clock::now();
        std::vector<std::string_view> views;
        views.reserve(arguments.size());
        for (const auto* value : arguments) views.emplace_back(value);
        auto command = parse_bench_command(views);
        if (!command) {
            std::cerr << "Mouse error: " << command.error() << '\n';
            return 2;
        }
        const auto paths = nn::resolve_backend_cli_paths(command->neural.backend);
        if (!paths) {
            std::cerr << "Mouse error: " << paths.error() << '\n';
            return 1;
        }
        const auto backend_start = Clock::now();
        auto backend = nn::create_backend(command->neural.backend);
        const auto backend_end = Clock::now();
        if (!backend) {
            std::cerr << "Mouse error: " << backend.error() << '\n';
            return 1;
        }
        auto report = run_bench(**backend, *command, fixed_bench_cases());
        if (!report) {
            std::cerr << "Mouse error: " << report.error() << '\n';
            return 1;
        }
        report->backend_setup =
            std::chrono::duration_cast<std::chrono::nanoseconds>(backend_end - backend_start);
        report->command_wall =
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - command_start);
        const auto text = render(*report);
        std::cout << text;
        std::cout.flush();
        if (!std::cout) {
            std::cerr << "Mouse error: Cannot write complete bench report.\n";
            return 1;
        }
        std::ostringstream signature;
        signature.imbue(std::locale::classic());
        signature << "nodes " << report->nodes << " nps " << std::fixed << std::setprecision(0)
                  << report->nps << '\n';
        std::cout << signature.str();
        std::cout.flush();
        if (!std::cout) {
            std::cerr << "Mouse error: Cannot write complete bench signature.\n";
            return 1;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Mouse error: Bench failed: " << error.what() << '\n';
    } catch (...) {
        std::cerr << "Mouse error: Bench failed with a non-standard exception.\n";
    }
    return 1;
}
}  // namespace mouse::uci