// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <filesystem>
#include <vector>

#include "engine/nn/backend.h"

namespace mouse::nn::detail {
// The child command line (after the executable) that rebuilds exactly `config`'s
// TensorRT cache identity: model, precision, buckets, cache directory, GPU library
// directories, device, ORT CPU threads, workspace and deterministic compute.
[[nodiscard]] std::vector<std::filesystem::path> tensorrt_builder_arguments(
    const BackendConfig& config);
// Runs `config.tensorrt_builder build-tensorrt-cache ...` and waits for it. Each line
// the child prints on stdout goes to config.initialization_progress. Throws with the
// child's error when it cannot start or exits unsuccessfully. On Windows the child is
// in a kill-on-close job, so it ends with this process; it inherits only its own pipes.
void build_tensorrt_cache(const BackendConfig& config);
}  // namespace mouse::nn::detail
