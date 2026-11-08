// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

#include "engine/nn/backend.h"

namespace mouse::nn::detail {
// Pure startup preflight: never creates directories or initializes the GPU.
[[nodiscard]] Result<std::filesystem::path> checked_tensor_rt_cache_root(
    const std::filesystem::path& root, std::size_t graph_name_bytes = 0);
struct GpuIdentity {
    BackendInfo info;
    std::string capability;
    std::string runtime_fingerprint;
};
// Selection is platform independent and testable without a GPU. Only explicit
// absolute directories are searched; distinct matching modules are ambiguous.
[[nodiscard]] std::filesystem::path explicit_gpu_library(
    std::span<const std::filesystem::path> directories, const char* filename);
// Unix has no AddDllDirectory equivalent. Preload the pinned SDK dependency
// closure by absolute name/SONAME, keep it alive through all ORT sessions, and
// never change LD_LIBRARY_PATH. Windows uses RuntimeLibrary's registered dirs.
class GpuDependencies {
   public:
    explicit GpuDependencies(const BackendConfig& config);
    ~GpuDependencies();
    GpuDependencies(const GpuDependencies&) = delete;
    GpuDependencies& operator=(const GpuDependencies&) = delete;
    GpuDependencies(GpuDependencies&&) = delete;
    GpuDependencies& operator=(GpuDependencies&&) = delete;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
// Explicit synchronous transfers for stable IO-bound device buffers. Each call
// selects the configured device on its calling thread; Backend serializes use.
class GpuTransfer {
   public:
    explicit GpuTransfer(const BackendConfig& config);
    ~GpuTransfer();
    GpuTransfer(const GpuTransfer&) = delete;
    GpuTransfer& operator=(const GpuTransfer&) = delete;
    GpuTransfer(GpuTransfer&&) = delete;
    GpuTransfer& operator=(GpuTransfer&&) = delete;
    void upload(std::span<float> device, std::span<const float> host);
    void download(std::span<float> host, std::span<const float> device);

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
// Startup-only helpers: no global state, no current-device changes, no kernels.
[[nodiscard]] GpuIdentity query_gpu_identity(const BackendConfig& config);
[[nodiscard]] std::size_t query_gpu_multiprocessors(const BackendConfig& config);
// DirectML (Windows, DirectML builds): device_id counts hardware adapters in DXGI's
// high-performance order (IDXGIFactory6::EnumAdapterByGpuPreference), so 0 is the
// discrete GPU even on laptops whose display runs on an integrated one. ORT's DirectML
// provider takes the IDXGIFactory::EnumAdapters index instead, matched here by LUID.
struct DirectMlAdapter {
    GpuIdentity identity;
    int enum_adapters_index = -1;
};
[[nodiscard]] DirectMlAdapter query_directml_adapter(const BackendConfig& config);
// After ORT has created a DirectML session: the DirectML.dll it loaded must be the one
// shipped beside the ONNX Runtime library, never an older copy from System32.
void require_bundled_directml(const BackendConfig& config);
[[nodiscard]] std::string backend_identity(const BackendConfig& config, const BackendInfo& info,
                                           const std::string& gpu_fingerprint);
// The Mouse-MoE TensorRT plugin library (engine/nn/trt_moe) beside the ONNX Runtime
// library, for TensorRT backends of builds that include it; nullopt otherwise. A build
// with the plugin fails explicitly when the file is missing.
[[nodiscard]] std::optional<std::filesystem::path> trt_moe_plugin(const BackendConfig& config);
enum class CacheLockMode : std::uint8_t { Wait, Try };
// An OS-held lock serializes cooperating processes. Provider cache paths remain
// stable for the session lifetime. Only a successful complete warm-up publishes
// the identity manifest; incomplete/corrupt prior caches fail explicitly.
class TensorRtCache {
   public:
    // Startup waits for another owner, then validates its published manifest.
    // Try supports fail-fast inspection without weakening exclusive ownership.
    TensorRtCache(const std::filesystem::path& root, std::string identity,
                  CacheLockMode mode = CacheLockMode::Wait);
    ~TensorRtCache();
    TensorRtCache(const TensorRtCache&) = delete;
    TensorRtCache& operator=(const TensorRtCache&) = delete;
    TensorRtCache(TensorRtCache&&) = delete;
    TensorRtCache& operator=(TensorRtCache&&) = delete;
    // Whether the identity's namespace exists, without creating or locking it. A
    // namespace that exists is still validated (and may be incomplete) when opened.
    [[nodiscard]] static bool exists(const std::filesystem::path& root,
                                     const std::string& identity);
    [[nodiscard]] const std::filesystem::path& directory() const noexcept;
    [[nodiscard]] bool reusable() const noexcept;
    void commit();

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace mouse::nn::detail
