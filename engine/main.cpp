// SPDX-License-Identifier: GPL-3.0-or-later
#include <charconv>
#include <chrono>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <string_view>

#include "engine/chess/attacks.h"
#include "engine/chess/movegen.h"
#include "engine/chess/perft.h"
#include "engine/nn/command_runner.h"
#include "engine/nn/model_metadata.h"
#include "engine/search/command_runner.h"
#include "engine/uci/bench.h"
#include "engine/uci/command_runner.h"
#include "engine/util/build_info.h"
#include "engine/util/runtime_library.h"

namespace {
struct PerftCommand {
    int depth = 0;
    bool divide = false;
    bool chess960 = false;
    std::string_view fen = mouse::chess::kStartFen;
};

mouse::Result<PerftCommand> parse_perft(std::span<char*> arguments) {
    PerftCommand command;
    std::size_t next = 0;
    if (!arguments.empty() && std::string_view{arguments[0]} == "divide") {
        command.divide = true;
        ++next;
    }
    if (next >= arguments.size()) {
        return tl::unexpected(std::string{"Missing perft depth"});
    }
    const std::string_view depth{arguments[next++]};
    const auto parsed = std::from_chars(std::to_address(depth.begin()),
                                        std::to_address(depth.end()), command.depth);
    if (depth.empty() || parsed.ec != std::errc{} || parsed.ptr != std::to_address(depth.end()) ||
        command.depth < (command.divide ? 1 : 0) || command.depth > mouse::chess::kMaxPerftDepth) {
        return tl::unexpected("Perft depth must be an integer between " +
                              std::to_string(command.divide ? 1 : 0) + " and " +
                              std::to_string(mouse::chess::kMaxPerftDepth));
    }
    bool seen_fen = false;
    while (next < arguments.size()) {
        const std::string_view option{arguments[next++]};
        if (option == "--chess960") {
            if (command.chess960) {
                return tl::unexpected(std::string{"Duplicate --chess960 option"});
            }
            command.chess960 = true;
        } else if (option == "--fen") {
            if (seen_fen) {
                return tl::unexpected(std::string{"Duplicate --fen option"});
            }
            seen_fen = true;
            if (next >= arguments.size()) {
                return tl::unexpected(std::string{"Missing FEN after --fen"});
            }
            command.fen = arguments[next++];
        } else {
            return tl::unexpected("Unknown perft option: " + std::string{option});
        }
    }
    return command;
}

int run_perft(std::span<char*> arguments) {
    const auto command = parse_perft(arguments);
    if (!command) {
        std::cerr << "Mouse error: " << command.error() << '\n';
        return 2;
    }
    const mouse::chess::AttackTables attacks;
    auto position = mouse::chess::Position::from_fen(command->fen, attacks);
    if (!position) {
        std::cerr << "Mouse error: Invalid FEN: " << position.error() << '\n';
        return 2;
    }
    const auto variant = mouse::chess::validate_uci_castling_mode(*position, command->chess960);
    if (!variant) {
        std::cerr << "Mouse error: " << variant.error() << '\n';
        return 2;
    }
    const auto started = std::chrono::steady_clock::now();
    std::uint64_t nodes = 0;
    if (command->divide) {
        const auto divided = mouse::chess::perft_divide(*position, attacks, command->depth);
        if (!divided) {
            std::cerr << "Mouse error: " << divided.error() << '\n';
            return 1;
        }
        for (const auto& entry : *divided) {
            const auto sum = mouse::chess::detail::checked_node_sum(nodes, entry.nodes);
            if (!sum) {
                std::cerr << "Mouse error: " << sum.error() << '\n';
                return 1;
            }
            nodes = *sum;
            std::cout << mouse::chess::move_to_uci(entry.move, command->chess960) << ": "
                      << entry.nodes << '\n';
        }
    } else {
        const auto counted = mouse::chess::perft(*position, attacks, command->depth);
        if (!counted) {
            std::cerr << "Mouse error: " << counted.error() << '\n';
            return 1;
        }
        nodes = *counted;
    }
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - started;
    const double nps = elapsed.count() > 0.0 ? static_cast<double>(nodes) / elapsed.count() : 0.0;
    std::cout << "nodes " << nodes << " time_ms " << std::fixed << std::setprecision(3)
              << elapsed.count() * 1000.0 << " nps " << std::setprecision(0) << nps << '\n';
    return 0;
}

mouse::Result<mouse::RuntimeLibrary> load_runtime() {
    const auto executable = mouse::RuntimeLibrary::executable_path();
    if (!executable) return tl::unexpected(executable.error());
    return mouse::RuntimeLibrary::load(executable->parent_path() / mouse::build::runtime_filename);
}

int describe_net(std::span<char*> arguments) {
    if (arguments.size() != 1) {
        std::cerr << "Mouse error: describe-net requires exactly one model path.\n";
        return 2;
    }
    const auto runtime = load_runtime();
    if (!runtime) {
        std::cerr << "Mouse error: " << runtime.error() << '\n';
        return 1;
    }
    const auto model = mouse::nn::inspect_onnx_model(*runtime, std::filesystem::path{arguments[0]});
    if (!model) {
        std::cerr << "Mouse error: " << model.error() << '\n';
        return 1;
    }
    std::cout << "input_format: " << model->input_format << '\n'
              << "network_format: " << model->network_format << '\n'
              << "raw_network_format: " << model->raw_network_format << '\n'
              << "parameters: " << model->parameter_count << '\n'
              << "source_sha256: " << model->source_sha256 << '\n'
              << "converter: " << model->converter_version << '\n'
              << "input: planes [batch,112,8,8] float32\n"
              << "policy: policy [batch,1858] logits\n"
              << "value: wdl [batch,3] W,D,L\n"
              << "moves_left: mlh [batch,1]\n";
    return 0;
}

int print_version() {
    const auto runtime = load_runtime();
    if (!runtime) {
        std::cerr << "Mouse error: " << runtime.error() << '\n';
        return 1;
    }
    std::cout << "Mouse " << mouse::build::version << '\n'
              << "git: " << mouse::build::git_hash << '\n'
              << "build: " << mouse::build::build_type << '\n'
              << "compiler: " << mouse::build::compiler << '\n'
              << "ONNX Runtime: " << runtime->version() << '\n';
    return 0;
}

int run(std::span<char*> arguments) {
    if (arguments.size() == 1) return mouse::uci::run_uci_command({});
    if (arguments.size() >= 2 && std::string_view{arguments[1]} == "uci")
        return mouse::uci::run_uci_command(arguments.subspan(2));
    if (arguments.size() == 2 && std::string_view{arguments[1]} == "--help") {
        std::cout
            << "Usage: mouse [uci [explicit backend/search options]]\n"
               "       mouse --version | --help\n"
               "       mouse describe-net <ONNX>\n"
               "       mouse backend-eval --weights <ONNX> --backend <cpu|cuda|tensorrt>\n"
               "         --precision <fp32|fp16> --inputs <MOUSEIN1> [--buckets 1,8,32,64]\n"
               "       mouse backendbench --weights <ONNX> --backend <cpu|cuda|tensorrt>\n"
               "         --precision <fp32|fp16> [--start-batch-size 1] [--max-batch-size 1024]\n"
               "         [--batch-step 1] [--batches 100]\n"
               "         GPU options: --gpu-dll-dir <dir> (repeat), --cache-dir <dir>, --device 0\n"
               "       mouse search --weights <ONNX> --backend <cpu|cuda|tensorrt>\n"
               "         --precision <fp32|fp16> [--nodes 800] [--movetime <ms>] [--depth "
               "<plies>]\n"
               "         [--fen \"<FEN>\"] [--moves \"e2e4 e7e5\"] [--chess960] [--deterministic]\n"
               "         [--threads 4] [--batch 64] [--ram-limit-mb 512] [--nncache-mb 0]\n"
               "       mouse bench --weights <ONNX> --backend <cpu|cuda|tensorrt>\n"
               "         --precision <fp32|fp16> [--nodes 800] [explicit search/backend options]\n"
               "       mouse perft [divide] <depth> [--fen \"<FEN>\"] [--chess960]\n"
               "  --version  Show build identity and verify the loaded ONNX Runtime.\n"
               "  --help     Show this help.\n"
               "  perft      Count legal move paths; divide lists counts by root move.\n"
               "  describe-net  Validate ONNX metadata and tensor interface on CPU.\n";
        return 0;
    }
    if (arguments.size() >= 2 && std::string_view{arguments[1]} == "backend-eval") {
        return mouse::nn::run_backend_command(mouse::nn::BackendCommandKind::Evaluate,
                                              arguments.subspan(2));
    }
    if (arguments.size() >= 2 && std::string_view{arguments[1]} == "backendbench") {
        return mouse::nn::run_backend_command(mouse::nn::BackendCommandKind::Benchmark,
                                              arguments.subspan(2));
    }
    if (arguments.size() >= 2 && std::string_view{arguments[1]} == "bench") {
        return mouse::uci::run_bench_command(arguments.subspan(2));
    }
    if (arguments.size() >= 2 && std::string_view{arguments[1]} == "search") {
        return mouse::search::run_search_command(arguments.subspan(2));
    }
    if (arguments.size() >= 2 && std::string_view{arguments[1]} == "describe-net") {
        return describe_net(arguments.subspan(2));
    }
    if (arguments.size() >= 2 && std::string_view{arguments[1]} == "perft") {
        return run_perft(arguments.subspan(2));
    }
    if (arguments.size() == 2 && std::string_view{arguments[1]} == "--version") {
        return print_version();
    }
    std::cerr << "Mouse error: Unknown argument. Use mouse --help.\n";
    return 2;
}
}  // namespace

int main(int argument_count, char* arguments[]) {
    try {
        return run({arguments, static_cast<std::size_t>(argument_count)});
    } catch (const std::exception& error) {
        // C stdio keeps exception reporting from throwing another stream exception.
        std::fputs("Mouse error: ", stderr);
        std::fputs(error.what(), stderr);
        std::fputs("\n", stderr);
    } catch (...) {
        std::fputs("Mouse error: Unrecognized exception at the application boundary.\n", stderr);
    }
    return 1;
}
