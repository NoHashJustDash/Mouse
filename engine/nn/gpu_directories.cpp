// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/nn/gpu_directories.h"

#include <system_error>
#include <utility>

namespace mouse::nn {
std::vector<std::filesystem::path> gpu_directories_or_default(
    std::vector<std::filesystem::path> configured, BackendKind kind,
    const std::filesystem::path& executable_directory) {
    if (!configured.empty()) return configured;
    if (kind != BackendKind::Cuda && kind != BackendKind::TensorRt) return configured;
    const auto local = executable_directory / kLocalGpuDirectory;
    std::error_code error;
    if (std::filesystem::is_directory(local, error) && !error) configured.push_back(local);
    return configured;
}
}  // namespace mouse::nn
