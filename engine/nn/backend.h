// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "engine/nn/encoder.h"
#include "engine/nn/model_metadata.h"
#include "engine/nn/output.h"

namespace mouse::nn {
enum class BackendKind : std::uint8_t { Cpu, Cuda, TensorRt };
enum class ComputePrecision : std::uint8_t { Float32, Float16 };
// Absolute per-user location; missing/relative environment values are errors.
[[nodiscard]] Result<std::filesystem::path> default_cache_directory();
struct BackendConfig {
    std::filesystem::path model_path;
    std::filesystem::path runtime_path;
    std::filesystem::path cache_directory;
    std::vector<std::filesystem::path> gpu_dll_directories;
    BackendKind kind = BackendKind::TensorRt;
    ComputePrecision precision = ComputePrecision::Float16;
    // ORT requests deterministic GPU kernels where supported; callers also
    // fix search scheduling and CPU worker counts for reproducible analysis.
    bool deterministic_compute = false;
    std::vector<std::size_t> batch_buckets{1, 8, 32, 64};
    int device_id = 0;
    int cpu_threads = 1;
    std::uint64_t workspace_bytes = 1ULL << 30U;
    // Startup-only progress; invoked on the constructing owner, never from an
    // ORT callback or inference. An empty callback suppresses progress reporting.
    std::function<void(std::string_view)> initialization_progress;
};
struct BackendInfo {
    std::string onnx_sha256;
    std::string provider;
    std::string precision;
    std::string device_name;
    std::string device_uuid;
    std::string driver_version;
    std::string cuda_version;
    std::string tensorrt_version;
    std::string cache_namespace;
};
// evaluate serializes session/binding/host-pool use. Callers retain independent
// output storage, must not mutate inputs during a call, and must not overlap
// output spans across calls. On failure all caller outputs remain unchanged.
// All physical rows are validated before nonthrowing publication to caller storage.
// metadata/info are immutable and may be read concurrently. Construction and
// destruction require external startup/shutdown synchronization, like RuntimeLibrary.
class Backend {
   public:
    virtual ~Backend() = default;
    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;
    Backend(Backend&&) = delete;
    Backend& operator=(Backend&&) = delete;
    [[nodiscard]] virtual Result<void> evaluate(std::span<const InputPlanes> inputs,
                                                std::span<NNOutput> outputs) = 0;
    // Physical rows submitted for a validated logical count (including padding).
    [[nodiscard]] virtual std::size_t physical_batch_size(std::size_t logical) const noexcept {
        return logical;
    }
    [[nodiscard]] virtual const ModelMetadata& metadata() const noexcept = 0;
    [[nodiscard]] virtual const BackendInfo& info() const noexcept = 0;

   protected:
    Backend() = default;
};
[[nodiscard]] Result<std::unique_ptr<Backend>> create_backend(const BackendConfig& config);
// Default minibatch for a backend when the UCI MinibatchSize is 0 (auto).
// GPU: lc0 v0.32.1 neural/backends/cuda/network_cuda.cc uses 2 x the device's
// multiprocessor count ("simple heuristic that seems to work for a wide range
// of GPUs"), capped at the 1024 maximum. CPU: Mouse's fixed 64.
[[nodiscard]] Result<std::size_t> recommended_minibatch(const BackendConfig& config);
}  // namespace mouse::nn
