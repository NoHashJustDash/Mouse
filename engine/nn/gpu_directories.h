// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <filesystem>
#include <vector>

#include "engine/nn/backend.h"

namespace mouse::nn {
// The folder beside the executable that scripts/setup-gpu.ps1 fills with NVIDIA's libraries.
inline constexpr const char* kLocalGpuDirectory = "gpu";
// The folders NVIDIA's libraries are loaded from. Configured folders are returned
// unchanged. With none configured, the CUDA and TensorRT backends use
// `<executable_directory>/gpu` when that folder exists. Nothing else is searched: with
// no folder at all the list stays empty and the backend names the library it could not
// find.
[[nodiscard]] std::vector<std::filesystem::path> gpu_directories_or_default(
    std::vector<std::filesystem::path> configured, BackendKind kind,
    const std::filesystem::path& executable_directory);
}  // namespace mouse::nn
