// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/nn/command_runner.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <utility>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#include "engine/chess/attacks.h"
#include "engine/nn/cached_backend.h"
#include "engine/nn/history.h"
#include "engine/util/build_info.h"
#include "engine/util/sha256.h"

namespace mouse::nn {
namespace {
using Clock = std::chrono::steady_clock;
double milliseconds(Clock::duration duration) {
    return std::chrono::duration<double, std::milli>{duration}.count();
}
Result<std::vector<char>> read_inputs(const std::filesystem::path& path) {
    std::ifstream input{path, std::ios::binary | std::ios::ate};
    if (!input) return tl::unexpected("Cannot open input planes file: " + path.string());
    const auto length = input.tellg();
    constexpr auto kMaximum = std::size_t{16} + std::size_t{1024} * 112U * 12U;
    if (length < 0 || static_cast<std::uint64_t>(length) > kMaximum) {
        return tl::unexpected(std::string{"Input planes file exceeds bounded transport length"});
    }
    std::vector<char> bytes(static_cast<std::size_t>(length));
    input.seekg(0);
    if (!bytes.empty()) input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!input) return tl::unexpected("Cannot read input planes file: " + path.string());
    return bytes;
}
Result<void> verify_identity(const std::filesystem::path& path, std::string_view expected) {
    const auto current = mouse::sha256_file(path);
    if (!current) return tl::unexpected(current.error());
    if (*current != expected)
        return tl::unexpected("Artifact changed during evaluation: " + path.string());
    return {};
}
Result<void> evaluate_command(const BackendCommand& command) {
    const auto bytes = read_inputs(command.inputs_path);
    if (!bytes) return tl::unexpected(bytes.error());
    const auto batch = decode_plane_batch(std::as_bytes(std::span{*bytes}));
    if (!batch) return tl::unexpected(batch.error());
    auto backend = create_backend(command.backend);
    if (!backend) return tl::unexpected(backend.error());
    auto cached = create_cached_backend(std::move(*backend), command.cache_bytes,
                                        command.backend.batch_buckets.back());
    if (!cached) return tl::unexpected(cached.error());
    std::vector<NNOutput> outputs(batch->inputs.size());
    const auto maximum = command.backend.batch_buckets.back();
    for (std::size_t start = 0; start < batch->inputs.size(); start += maximum) {
        const auto count = std::min(maximum, batch->inputs.size() - start);
        const auto evaluated = (*cached)->evaluate(std::span{batch->inputs}.subspan(start, count),
                                                   std::span{outputs}.subspan(start, count));
        if (!evaluated) return evaluated;
    }
    for (const auto& [path, expected] :
         std::array{std::pair{command.inputs_path, batch->input_sha256},
                    std::pair{command.backend.model_path, (*cached)->info().onnx_sha256}}) {
        const auto unchanged = verify_identity(path, expected);
        if (!unchanged) return unchanged;
    }
    const auto wire = encode_backend_outputs(
        outputs,
        {.model_sha256 = (*cached)->info().onnx_sha256, .input_sha256 = batch->input_sha256});
    if (!wire) return tl::unexpected(wire.error());
    std::vector<char> characters(wire->size());
    std::ranges::transform(*wire, characters.begin(),
                           [](std::byte value) { return std::bit_cast<char>(value); });
#ifdef _WIN32
    if (_setmode(_fileno(stdout), _O_BINARY) == -1) {
        return tl::unexpected(std::string{"Cannot set binary backend output mode"});
    }
#endif
    std::cout.write(characters.data(), static_cast<std::streamsize>(characters.size()));
    std::cout.flush();
    if (!std::cout) return tl::unexpected(std::string{"Cannot write complete backend output"});
    return {};
}

Result<void> benchmark_command(const BackendCommand& command) {
    const auto expected_model = mouse::sha256_file(command.backend.model_path);
    if (!expected_model) return tl::unexpected(expected_model.error());
    const chess::AttackTables attacks;
    const auto position = chess::Position::from_fen(chess::kStartFen, attacks);
    if (!position) return tl::unexpected(position.error());
    const std::array history{HistoryFrame::capture(*position)};
    std::unique_ptr<Backend> shared;
    double shared_initialization = 0;
    if (command.backend.kind == BackendKind::Cpu) {
        auto config = command.backend;
        config.batch_buckets = {command.max_batch_size};
        const auto start = Clock::now();
        auto created = create_backend(config);
        shared_initialization = milliseconds(Clock::now() - start);
        if (!created) return tl::unexpected(created.error());
        shared = std::move(*created);
    }
    std::cout << "benchmark_config version " << std::quoted(std::string{build::version}) << " git "
              << std::quoted(std::string{build::git_hash}) << " build "
              << std::quoted(std::string{build::build_type}) << " compiler "
              << std::quoted(std::string{build::compiler}) << " ort_version "
              << build::required_ort_version << " model_sha256 " << *expected_model
              << " start_batch " << command.start_batch_size << " max_batch "
              << command.max_batch_size << " step " << command.batch_step << " batches "
              << command.batches << " device_index " << command.backend.device_id << " cpu_threads "
              << command.backend.cpu_threads << " workspace_bytes "
              << command.backend.workspace_bytes << " cache disabled\n";
    std::cout << "artifact_path " << std::quoted(command.backend.model_path.string())
              << " cache_directory " << std::quoted(command.backend.cache_directory.string())
              << '\n';
    for (const auto& directory : command.backend.gpu_dll_directories)
        std::cout << "gpu_sdk_directory " << std::quoted(directory.string()) << '\n';
    std::vector<InputPlanes> inputs(command.max_batch_size);
    std::vector<NNOutput> outputs(command.max_batch_size);
    double best = 0;
    std::size_t best_batch = 0;
    for (std::size_t count = command.start_batch_size; count <= command.max_batch_size;
         count += command.batch_step) {
        std::unique_ptr<Backend> fixed;
        Backend* backend = shared.get();
        double initialization = shared_initialization;
        if (command.backend.kind != BackendKind::Cpu) {
            auto config = command.backend;
            config.batch_buckets = {count};
            const auto start = Clock::now();
            auto created = create_backend(config);
            initialization = milliseconds(Clock::now() - start);
            if (!created) return tl::unexpected(created.error());
            fixed = std::move(*created);
            backend = fixed.get();
        }
        if (backend->info().onnx_sha256 != *expected_model)
            return tl::unexpected(std::string{"Model changed during backendbench construction"});
        if (count == command.start_batch_size) {
            const auto& info = backend->info();
            std::cout << "device_identity name " << std::quoted(info.device_name) << " uuid "
                      << std::quoted(info.device_uuid) << " driver "
                      << std::quoted(info.driver_version) << " cuda "
                      << std::quoted(info.cuda_version) << " tensorrt "
                      << std::quoted(info.tensorrt_version) << " cache_namespace "
                      << std::quoted(info.cache_namespace) << '\n';
        }
        const auto encoded = encode_classical(history, backend->metadata().input_format);
        if (!encoded) return tl::unexpected(encoded.error());
        std::ranges::fill(inputs, *encoded);
        const auto current_inputs = std::span{inputs}.first(count);
        const auto current_outputs = std::span{outputs}.first(count);
        const auto warming = Clock::now();
        auto warmed = backend->evaluate(current_inputs, current_outputs);
        const auto warm_ms = milliseconds(Clock::now() - warming);
        if (!warmed) return warmed;
        const auto started = Clock::now();
        for (std::uint32_t iteration = 0; iteration < command.batches; ++iteration) {
            const auto evaluated = backend->evaluate(current_inputs, current_outputs);
            if (!evaluated) return evaluated;
        }
        const double seconds = std::chrono::duration<double>{Clock::now() - started}.count();
        if (seconds <= 0) return tl::unexpected(std::string{"Benchmark clock did not advance"});
        const double nps = static_cast<double>(count) * command.batches / seconds;
        if (nps > best) {
            best = nps;
            best_batch = count;
        }
        std::cout << "batch " << count << " physical_batch " << count << " batches "
                  << command.batches << " nps " << std::fixed << std::setprecision(3) << nps
                  << " time_ms " << seconds * 1000 << " initialization_ms " << initialization
                  << " warm_ms " << warm_ms << " provider " << backend->info().provider
                  << " precision " << backend->info().precision << " model_sha256 "
                  << backend->info().onnx_sha256 << '\n';
        std::cout.flush();
        if (!std::cout) return tl::unexpected(std::string{"Cannot write benchmark result"});
    }
    auto unchanged = verify_identity(command.backend.model_path, *expected_model);
    if (!unchanged) return unchanged;
    std::cout << "best_batch " << best_batch << " best_nps " << std::fixed << std::setprecision(3)
              << best << " cache disabled\n";
    return {};
}
}  // namespace

Result<void> resolve_backend_cli_paths(BackendConfig& config) {
    std::error_code error;
    const auto model = std::filesystem::absolute(config.model_path, error);
    if (error) return tl::unexpected("Cannot resolve weights path: " + error.message());
    config.model_path = model.lexically_normal();
    if (!config.cache_directory.empty()) {
        const auto cache = std::filesystem::absolute(config.cache_directory, error);
        if (error) return tl::unexpected("Cannot resolve cache path: " + error.message());
        config.cache_directory = cache.lexically_normal();
    }
    const auto executable = RuntimeLibrary::executable_path();
    if (!executable) return tl::unexpected(executable.error());
    config.runtime_path = executable->parent_path() / build::runtime_filename;
    return {};
}

int run_backend_command(BackendCommandKind kind, std::span<char*> arguments) {
    if (kind != BackendCommandKind::Evaluate && kind != BackendCommandKind::Benchmark) {
        std::cerr << "Mouse error: Configuration parsing profile is not an executable command.\n";
        return 2;
    }
    std::vector<std::string_view> views;
    views.reserve(arguments.size());
    for (const auto* value : arguments) views.emplace_back(value);
    auto command = parse_backend_command(kind, views);
    if (!command) {
        std::cerr << "Mouse error: " << command.error() << '\n';
        return 2;
    }
    const auto paths = resolve_backend_cli_paths(command->backend);
    if (!paths) {
        std::cerr << "Mouse error: " << paths.error() << '\n';
        return 1;
    }
    const auto result = kind == BackendCommandKind::Evaluate ? evaluate_command(*command)
                                                             : benchmark_command(*command);
    if (!result) {
        std::cerr << "Mouse error: " << result.error() << '\n';
        return 1;
    }
    return 0;
}
}  // namespace mouse::nn
