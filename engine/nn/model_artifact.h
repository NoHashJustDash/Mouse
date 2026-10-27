// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstddef>
#include <span>

#include "engine/util/result.h"

namespace mouse::nn {
// Bounded identity preflight for ONNX 1.18 containers. Rejects external tensors,
// including nested graphs, functions, attributes, sparse tensors and training
// graphs, so a digest of these bytes identifies every weight loaded by ORT.
// This does not replace ORT's model/type/operator validation. Immutable input is
// borrowed only during this call; independent calls may run concurrently.
struct OnnxArtifactInfo {
    std::size_t maximum_graph_name_bytes = 0;
    // Portable cache payload names: ASCII letters/digits/._-, no trailing dot.
    bool windows_safe_graph_names = true;
};
// Same bounded validation, with allocation-free cache-name inspection across
// every graph container. Name safety is informational; ORT validates semantics.
[[nodiscard]] Result<OnnxArtifactInfo> inspect_onnx_artifact(std::span<const std::byte> bytes);
[[nodiscard]] Result<void> validate_self_contained_onnx(std::span<const std::byte> bytes);
}  // namespace mouse::nn
