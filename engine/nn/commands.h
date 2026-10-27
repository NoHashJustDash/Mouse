// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "engine/nn/backend.h"

namespace mouse::nn {
enum class BackendCommandKind : std::uint8_t { Evaluate, Benchmark, Configuration };
struct BackendCommand {
    BackendConfig backend;
    std::filesystem::path inputs_path;
    std::size_t cache_bytes = 0;
    std::size_t start_batch_size = 1;
    std::size_t max_batch_size = 1024;
    std::size_t batch_step = 1;
    std::uint32_t batches = 100;
};
struct PlaneBatch {
    std::vector<InputPlanes> inputs;
    std::string input_sha256;
};
struct OutputIdentity {
    std::string_view model_sha256;
    std::string_view input_sha256;
};
// Parsing and transport functions own their returned data and keep no shared
// state. Binary records use explicit little-endian integers/IEEE float32.
[[nodiscard]] Result<BackendCommand> parse_backend_command(
    BackendCommandKind kind, std::span<const std::string_view> arguments);
[[nodiscard]] Result<PlaneBatch> decode_plane_batch(std::span<const std::byte> bytes);
[[nodiscard]] Result<std::vector<std::byte>> encode_backend_outputs(
    std::span<const NNOutput> outputs, OutputIdentity identity);
}  // namespace mouse::nn
