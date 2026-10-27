// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>

#include "engine/util/result.h"
#include "engine/util/runtime_library.h"

namespace mouse::nn {
using MetadataFields = std::map<std::string, std::string, std::less<>>;
struct ModelMetadata {
    std::string converter_version;
    std::string source_sha256;
    std::string precision = "fp32";
    std::uint32_t input_format = 0;
    std::uint32_t network_format = 0;
    std::uint32_t raw_network_format = 0;
    std::uint64_t parameter_count = 0;
    bool operator==(const ModelMetadata&) const = default;
};

// Metadata is a value after validation and may be read concurrently. Network
// body tags are provenance; input/head contracts choose encoding and outputs.
// These functions do not modify a session or select a fallback encoder.
// By default, batch must remain symbolic. An explicitly specialized session
// must match expected_batch (1..1024) on every input and output.
[[nodiscard]] Result<ModelMetadata> parse_model_metadata(const MetadataFields& fields);
[[nodiscard]] Result<ModelMetadata> read_model_metadata(
    const OrtApi& api, const OrtSession& session,
    std::optional<std::size_t> expected_batch = std::nullopt);
// Explicit CPU-only inspection, with one ORT worker and no inference. Backend
// construction can call read_model_metadata on its own explicitly chosen session
// instead, avoiding a second model load.
[[nodiscard]] Result<ModelMetadata> inspect_onnx_model(const RuntimeLibrary& runtime,
                                                       const std::filesystem::path& path);
}  // namespace mouse::nn
