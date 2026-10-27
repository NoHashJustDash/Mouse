// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/nn/backend.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "engine/nn/backend_gpu.h"
#include "engine/nn/model_artifact.h"
#include "engine/util/sha256.h"

namespace mouse::nn {
namespace {
template <class T, class Deleter>
using Owner = std::unique_ptr<T, Deleter>;
using SessionOwner = Owner<OrtSession, decltype(OrtApi::ReleaseSession)>;
using BindingOwner = Owner<OrtIoBinding, decltype(OrtApi::ReleaseIoBinding)>;
using ValueOwner = Owner<OrtValue, decltype(OrtApi::ReleaseValue)>;

void check(const OrtApi& api, OrtStatus* raw, std::string_view operation) {
    const Owner<OrtStatus, decltype(api.ReleaseStatus)> status{raw, api.ReleaseStatus};
    if (status)
        throw std::runtime_error(std::string{operation} + ": " + api.GetErrorMessage(status.get()));
}
template <class T>
T require(Result<T> result) {
    if (!result) throw std::runtime_error(result.error());
    return std::move(*result);
}
void require(Result<void> result) {
    if (!result) throw std::runtime_error(result.error());
}
std::string utf8(const std::filesystem::path& path) {
    const auto value = path.u8string();
    return {value.begin(), value.end()};
}
const char* provider_label(BackendKind kind) noexcept {
    switch (kind) {
        case BackendKind::Cpu:
            return "CPU";
        case BackendKind::Cuda:
            return "CUDA";
        case BackendKind::TensorRt:
            return "TensorRT";
    }
    return "unknown";
}
BackendConfig validated(BackendConfig config) {
    if (config.kind != BackendKind::Cpu && config.kind != BackendKind::Cuda &&
        config.kind != BackendKind::TensorRt)
        throw std::runtime_error("Invalid backend provider selection");
    if (config.precision != ComputePrecision::Float32 &&
        config.precision != ComputePrecision::Float16)
        throw std::runtime_error("Invalid backend precision selection");
    if (config.kind == BackendKind::Cpu && config.precision != ComputePrecision::Float32)
        throw std::runtime_error("CPU backend requires float32 precision");
    if (config.batch_buckets.empty()) throw std::runtime_error("Backend batch buckets are empty");
    std::ranges::sort(config.batch_buckets);
    if (config.batch_buckets.front() == 0 || config.batch_buckets.back() > 1024 ||
        std::ranges::adjacent_find(config.batch_buckets) != config.batch_buckets.end())
        throw std::runtime_error("Backend batch buckets must be unique and in 1..1024");
    if (config.cpu_threads <= 0) throw std::runtime_error("Backend CPU threads must be positive");
    if (config.device_id < 0) throw std::runtime_error("Backend GPU device ID must be nonnegative");
    if (config.workspace_bytes == 0 ||
        config.workspace_bytes > std::numeric_limits<std::size_t>::max())
        throw std::runtime_error("Backend workspace bytes must fit a positive size_t");
    if (config.model_path.empty() || !config.model_path.is_absolute())
        throw std::runtime_error("Backend model path must be absolute");
    if (config.runtime_path.empty() || !config.runtime_path.is_absolute())
        throw std::runtime_error("Backend runtime path must be absolute");
    if (config.kind == BackendKind::TensorRt)
        config.cache_directory =
            require(detail::checked_tensor_rt_cache_root(config.cache_directory));
    return config;
}
std::vector<char> snapshot(const std::filesystem::path& path, const BackendConfig& config) {
    if (!std::filesystem::is_regular_file(path))
        throw std::runtime_error("ONNX artifact is not a regular file: " + path.string());
    const auto size = std::filesystem::file_size(path);
    if (size == 0 || size > (512ULL << 20U))
        throw std::runtime_error("ONNX artifact must contain 1..512MiB");
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot open ONNX artifact: " + path.string());
    std::vector<char> bytes(static_cast<std::size_t>(size));
    stream.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!stream || stream.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("ONNX artifact changed or could not be read completely");
    const auto info = require(inspect_onnx_artifact(std::as_bytes(std::span{bytes})));
#ifdef _WIN32
    if (config.kind == BackendKind::TensorRt) {
        if (!info.windows_safe_graph_names)
            throw std::runtime_error(
                "TensorRT ORT1.22 graph name must use portable ASCII cache filename characters "
                "(letters, digits, dot, underscore, hyphen; no trailing dot). Rename the ONNX "
                "graph before using TensorRT");
        require(detail::checked_tensor_rt_cache_root(config.cache_directory,
                                                     info.maximum_graph_name_bytes));
    }
#else
    static_cast<void>(config);
    static_cast<void>(info);
#endif
    return bytes;
}

struct Session {
    explicit Session(const OrtApi& api, std::optional<std::size_t> fixed)
        : batch(fixed), value(nullptr, api.ReleaseSession) {}
    std::optional<std::size_t> batch;
    SessionOwner value;
};
enum class BufferLocation : std::uint8_t { Cpu, Pinned, Device };
class BufferPool {
   public:
    BufferPool(const OrtApi& api, const OrtSession& session, std::size_t capacity,
               BufferLocation location, int device)
        : api_(api),
          memory_(nullptr, api.ReleaseMemoryInfo),
          allocator_(nullptr, api.ReleaseAllocator) {
        OrtMemoryInfo* memory = nullptr;
        if (location == BufferLocation::Device)
            check(api_,
                  api_.CreateMemoryInfo("Cuda", OrtDeviceAllocator, device, OrtMemTypeDefault,
                                        &memory),
                  "Create device memory information");
        else if (location == BufferLocation::Pinned)
            check(api_,
                  api_.CreateMemoryInfo("CudaPinned", OrtDeviceAllocator, 0, OrtMemTypeCPUOutput,
                                        &memory),
                  "Create pinned host memory information");
        else
            check(api_, api_.CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory),
                  "Create CPU memory information");
        memory_.reset(memory);
        constexpr std::size_t kElementsPerRow = kInputElementCount + 1858 + 3 + 1;
        const std::size_t count = capacity * kElementsPerRow;
        if (location != BufferLocation::Cpu) {
            OrtAllocator* allocator = nullptr;
            check(api_, api_.CreateAllocator(&session, memory_.get(), &allocator),
                  "Create inference buffer allocator");
            allocator_.reset(allocator);
            check(api_, api_.AllocatorAlloc(allocator_.get(), count * sizeof(float), &allocation_),
                  "Allocate inference buffers");
            if (allocation_ == nullptr) throw std::bad_alloc{};
        } else {
            cpu_.resize(count);
            allocation_ = cpu_.data();
        }
        const std::span all{static_cast<float*>(allocation_), count};
        input_ = all.first(capacity * kInputElementCount);
        policy_ = all.subspan(input_.size(), capacity * 1858);
        wdl_ = all.subspan(input_.size() + policy_.size(), capacity * 3);
        mlh_ = all.last(capacity);
    }
    ~BufferPool() {
        if (allocator_ && allocation_ != nullptr) {
            // Free cannot fail for an allocation from this live allocator. The
            // destructor still releases any diagnostic status without throwing.
            const Owner<OrtStatus, decltype(api_.ReleaseStatus)> status{
                api_.AllocatorFree(allocator_.get(), allocation_), api_.ReleaseStatus};
        }
    }
    BufferPool(const BufferPool&) = delete;
    BufferPool& operator=(const BufferPool&) = delete;
    BufferPool(BufferPool&&) = delete;
    BufferPool& operator=(BufferPool&&) = delete;
    [[nodiscard]] const OrtMemoryInfo* memory() const noexcept { return memory_.get(); }
    [[nodiscard]] std::span<float> input() const noexcept { return input_; }
    [[nodiscard]] std::span<float> policy() const noexcept { return policy_; }
    [[nodiscard]] std::span<float> wdl() const noexcept { return wdl_; }
    [[nodiscard]] std::span<float> mlh() const noexcept { return mlh_; }

   private:
    std::span<float> input_;
    std::span<float> policy_;
    std::span<float> wdl_;
    std::span<float> mlh_;
    const OrtApi& api_;
    Owner<OrtMemoryInfo, decltype(OrtApi::ReleaseMemoryInfo)> memory_;
    Owner<OrtAllocator, decltype(OrtApi::ReleaseAllocator)> allocator_;
    std::vector<float> cpu_;
    void* allocation_ = nullptr;
};
struct Views {
    explicit Views(const OrtApi& api)
        : input(nullptr, api.ReleaseValue),
          policy(nullptr, api.ReleaseValue),
          wdl(nullptr, api.ReleaseValue),
          mlh(nullptr, api.ReleaseValue) {}
    ValueOwner input;
    ValueOwner policy;
    ValueOwner wdl;
    ValueOwner mlh;
};
ValueOwner tensor(const OrtApi& api, const BufferPool& pool, std::span<float> data,
                  std::span<const std::int64_t> dimensions) {
    OrtValue* value = nullptr;
    const auto status = api.CreateTensorWithDataAsOrtValue(
        pool.memory(), data.data(), data.size_bytes(), dimensions.data(), dimensions.size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &value);
    ValueOwner owner{value, api.ReleaseValue};
    check(api, status, "Create inference buffer view");
    return owner;
}

class OrtBackend final : public Backend {
   public:
    OrtBackend(BackendConfig config, std::vector<char> model, RuntimeLibrary runtime)
        : runtime_(std::move(runtime)),
          api_(runtime_.api()),
          config_(std::move(config)),
          model_(std::move(model)),
          gpu_dependencies_(config_),
          env_(nullptr, api_.ReleaseEnv) {
        OrtEnv* env = nullptr;
        check(api_, api_.CreateEnv(ORT_LOGGING_LEVEL_WARNING, "Mouse", &env),
              "Create ORT environment");
        env_.reset(env);
        initialize_identity();
        const auto identity = detail::backend_identity(config_, info_, gpu_fingerprint_);
        info_.cache_namespace =
            mouse::sha256(std::as_bytes(std::span{identity.data(), identity.size()}));
        if (config_.kind != BackendKind::Cpu) {
            for (const auto batch : config_.batch_buckets) {
                std::unique_ptr<detail::TensorRtCache> cache;
                if (config_.kind == BackendKind::TensorRt) {
                    cache = std::make_unique<detail::TensorRtCache>(
                        config_.cache_directory, identity + "fixed_batch=" + std::to_string(batch));
                    if (!cache->reusable() && config_.initialization_progress)
                        config_.initialization_progress("TensorRT cold startup: building bucket " +
                                                        std::to_string(batch));
                }
                create_session(batch, cache.get());
                caches_.push_back(std::move(cache));
            }
        } else {
            create_session(std::nullopt, nullptr);
        }
        initialize_buffers_and_warm();
        const bool built = std::ranges::any_of(
            caches_, [](const auto& cache) { return cache && !cache->reusable(); });
        if (require(mouse::sha256_file(config_.model_path)) != info_.onnx_sha256)
            throw std::runtime_error("ONNX artifact changed during backend construction");
        for (auto& cache : caches_)
            if (cache) cache->commit();
        if (built) {
            // ORT v1.22 TensorRT keeps builders/parsers in the compiling provider.
            // Discard that provider after committing its verified EPContext;
            // execution-only sessions need no builder and retain much less RAM.
            bindings_.clear();
            views_.clear();
            transfer_.reset();
            device_pool_.reset();
            pool_.reset();
            sessions_.clear();
            for (std::size_t i = 0; i < config_.batch_buckets.size(); ++i)
                create_session(config_.batch_buckets[i], caches_.at(i).get());
            initialize_buffers_and_warm();
            for (const auto& cache : caches_) cache->commit();
        }
        caches_.clear();  // Release startup locks only after every bucket is warm.
        std::vector<char>{}.swap(
            model_);  // Immutable identity retains its hash, not the input snapshot.
    }

    Result<void> evaluate(std::span<const InputPlanes> inputs,
                          std::span<NNOutput> outputs) override {
        try {
            const std::lock_guard lock{mutex_};
            if (inputs.empty() || inputs.size() != outputs.size() ||
                inputs.size() > config_.batch_buckets.back())
                throw std::runtime_error(
                    "Backend batch must be nonempty, match output count and fit maximum batch");
            const auto physical = physical_batch_size(inputs.size());
            for (std::size_t row = 0; row < inputs.size(); ++row)
                require(expand_planes(inputs[row], pool_->input().subspan(row * kInputElementCount,
                                                                          kInputElementCount)));
            std::fill(
                pool_->input().begin() +
                    static_cast<std::ptrdiff_t>(inputs.size() * kInputElementCount),
                pool_->input().begin() + static_cast<std::ptrdiff_t>(physical * kInputElementCount),
                0.0F);
            run(physical);
            // run validates every physical row, including padded tails. No operation
            // below allocates or throws: contiguous float copies and optional<float>
            // assignment publish only after the complete batch has been accepted.
            static_assert(noexcept(std::declval<std::optional<float>&>() = 0.0F));
            for (std::size_t row = 0; row < outputs.size(); ++row) {
                std::copy_n(pool_->policy().data() + row * 1858, 1858,
                            outputs[row].policy_logits.data());
                std::copy_n(pool_->wdl().data() + row * 3, 3, outputs[row].wdl.data());
                outputs[row].moves_left = pool_->mlh()[row];
            }
            return {};
        } catch (const std::bad_alloc&) {
            return tl::unexpected(std::string{"Backend evaluation ran out of memory"});
        } catch (const std::exception& error) {
            return tl::unexpected(std::string{"Backend evaluation: "} + error.what());
        }
    }
    [[nodiscard]] std::size_t physical_batch_size(std::size_t logical) const noexcept override {
        if (config_.kind == BackendKind::Cpu) return logical;
        const auto bucket = std::ranges::lower_bound(config_.batch_buckets, logical);
        return bucket == config_.batch_buckets.end() ? logical : *bucket;
    }
    [[nodiscard]] const ModelMetadata& metadata() const noexcept override { return metadata_; }
    [[nodiscard]] const BackendInfo& info() const noexcept override { return info_; }

   private:
    void initialize_buffers_and_warm() {
        const auto maximum = config_.batch_buckets.back();
        pool_ = std::make_unique<BufferPool>(
            api_, *sessions_.front().value, maximum,
            config_.kind == BackendKind::Cpu ? BufferLocation::Cpu : BufferLocation::Pinned,
            config_.device_id);
        if (config_.kind != BackendKind::Cpu) {
            device_pool_ = std::make_unique<BufferPool>(api_, *sessions_.front().value, maximum,
                                                        BufferLocation::Device, config_.device_id);
            transfer_ = std::make_unique<detail::GpuTransfer>(config_);
        }
        // Views carry only shapes/OrtValue metadata, all sharing this one pool.
        // GPU sessions have fixed shapes and stable addresses for CUDA replay.
        // CPU retains dynamic execution of the exact requested batch size.
        if (config_.kind != BackendKind::Cpu) {
            for (const auto batch : config_.batch_buckets) create_views(batch);
        } else {
            for (std::size_t batch = 1; batch <= maximum; ++batch) create_views(batch);
        }
        const auto warm = [this](std::size_t batch) {
            std::fill_n(pool_->input().begin(), batch * kInputElementCount, 0.0F);
            run(batch);
        };
        // ORT v1.22.0 tensorrt_execution_provider.cc captures graph 0 only,
        // once per provider. Separate fixed sessions prevent stale-shape replay.
        // Three calls cover ordinary execution, capture and replay before readyok.
        for (const auto batch : config_.batch_buckets) {
            warm(batch);
            if (config_.kind == BackendKind::TensorRt) {
                warm(batch);
                warm(batch);
            }
        }
    }
    void initialize_identity() {
        if (config_.kind == BackendKind::Cpu) {
            info_.provider = "CPUExecutionProvider";
            info_.device_name = "CPU";
            info_.device_uuid = "cpu";
        } else {
            auto identity = detail::query_gpu_identity(config_);
            info_ = std::move(identity.info);
            gpu_fingerprint_ = std::move(identity.runtime_fingerprint);
            info_.provider = config_.kind == BackendKind::Cuda ? "CUDAExecutionProvider"
                                                               : "TensorrtExecutionProvider";
        }
        info_.onnx_sha256 = mouse::sha256(std::as_bytes(std::span{model_}));
        info_.precision = config_.precision == ComputePrecision::Float32 ? "fp32" : "fp16";
    }
    void add_provider(OrtSessionOptions& options, std::optional<std::size_t> fixed,
                      const detail::TensorRtCache* cache) {
        if (config_.kind == BackendKind::Cpu) return;
        // Disable all CPU assignments, including shape operators. Only the
        // selected GPU EP is registered: TensorRT never implicitly adds CUDA.
        check(api_, api_.AddSessionConfigEntry(&options, "session.disable_cpu_ep_fallback", "1"),
              "Disable CPU execution fallback");
        const auto device = std::to_string(config_.device_id);
        if (config_.kind == BackendKind::Cuda) {
            OrtCUDAProviderOptionsV2* raw = nullptr;
            check(api_, api_.CreateCUDAProviderOptions(&raw), "Create CUDA provider options");
            const Owner<OrtCUDAProviderOptionsV2, decltype(api_.ReleaseCUDAProviderOptions)>
                provider{raw, api_.ReleaseCUDAProviderOptions};
            const std::array keys{"device_id", "use_tf32", "do_copy_in_default_stream"};
            const std::array values{device.c_str(), "0", "1"};
            check(api_,
                  api_.UpdateCUDAProviderOptions(provider.get(), keys.data(), values.data(),
                                                 keys.size()),
                  "Configure explicit CUDA provider");
            check(api_,
                  api_.SessionOptionsAppendExecutionProvider_CUDA_V2(&options, provider.get()),
                  "Select CUDA provider");
        } else {
            OrtTensorRTProviderOptionsV2* raw = nullptr;
            check(api_, api_.CreateTensorRTProviderOptions(&raw),
                  "Create TensorRT provider options");
            const Owner<OrtTensorRTProviderOptionsV2, decltype(api_.ReleaseTensorRTProviderOptions)>
                provider{raw, api_.ReleaseTensorRTProviderOptions};
            const auto workspace = std::to_string(config_.workspace_bytes);
            // ORT v1.22.0 tensorrt_execution_provider.cc:1621-1640 requires
            // a relative engine path when dumping an external EPContext. The
            // path is resolved against the context file, never process CWD.
            if (!cache || !fixed) throw std::logic_error("TensorRT bucket lacks its cache/profile");
            const auto engine =
                cache->reusable() ? utf8(cache->directory() / "engine") : std::string{"engine"};
            const auto context = utf8(cache->directory() / "context.onnx");
            const auto timing = utf8(cache->directory() / "timing");
            const auto maximum = std::to_string(*fixed);
            const std::string maximum_shape = "planes:" + maximum + "x112x8x8";
            const std::array keys{"device_id",
                                  "trt_fp16_enable",
                                  "trt_max_workspace_size",
                                  "trt_engine_cache_enable",
                                  "trt_engine_cache_path",
                                  "trt_engine_cache_prefix",
                                  "trt_timing_cache_enable",
                                  "trt_timing_cache_path",
                                  "trt_min_subgraph_size",
                                  "trt_profile_min_shapes",
                                  "trt_profile_opt_shapes",
                                  "trt_profile_max_shapes",
                                  "trt_dump_ep_context_model",
                                  "trt_ep_context_file_path",
                                  "trt_ep_context_embed_mode",
                                  "trt_cuda_graph_enable"};
            const std::array values{device.c_str(),
                                    config_.precision == ComputePrecision::Float16 ? "1" : "0",
                                    workspace.c_str(),
                                    "1",
                                    engine.c_str(),
                                    "m",
                                    "1",
                                    timing.c_str(),
                                    "1",
                                    maximum_shape.c_str(),
                                    maximum_shape.c_str(),
                                    maximum_shape.c_str(),
                                    cache->reusable() ? "0" : "1",
                                    context.c_str(),
                                    "0",
                                    "1"};
            check(api_,
                  api_.UpdateTensorRTProviderOptions(provider.get(), keys.data(), values.data(),
                                                     keys.size()),
                  "Configure explicit TensorRT provider");
            check(api_,
                  api_.SessionOptionsAppendExecutionProvider_TensorRT_V2(&options, provider.get()),
                  "Select TensorRT provider");
        }
    }
    void create_session(std::optional<std::size_t> fixed, const detail::TensorRtCache* cache) {
        OrtSessionOptions* raw = nullptr;
        check(api_, api_.CreateSessionOptions(&raw), "Create inference session options");
        const Owner<OrtSessionOptions, decltype(api_.ReleaseSessionOptions)> options{
            raw, api_.ReleaseSessionOptions};
        check(api_, api_.SetIntraOpNumThreads(options.get(), config_.cpu_threads),
              "Set ORT worker count");
        check(api_, api_.SetInterOpNumThreads(options.get(), 1), "Set ORT inter-op worker count");
        check(api_, api_.SetSessionExecutionMode(options.get(), ORT_SEQUENTIAL),
              "Select sequential ORT execution");
        check(api_, api_.SetDeterministicCompute(options.get(), config_.deterministic_compute),
              "Configure deterministic compute");
        check(api_, api_.SetSessionGraphOptimizationLevel(options.get(), ORT_ENABLE_ALL),
              "Enable ORT graph optimization");
        if (fixed)
            check(api_,
                  api_.AddFreeDimensionOverrideByName(options.get(), "batch",
                                                      static_cast<std::int64_t>(*fixed)),
                  "Specialize GPU batch bucket");
        add_provider(*options, fixed, cache);
        Session session{api_, fixed};
        OrtSession* created = nullptr;
        // A file-backed session supplies ModelPath for the context's relative
        // external engine. ORT preserves the original model's metadata (v1.22.0
        // onnx_ctx_model_helper.cc CreateCtxModel); validate the same IO contract
        // below on both cold and warm paths. Corrupt caches fail, never rebuild.
        const auto status =
            cache && cache->reusable()
                ? api_.CreateSession(env_.get(), (cache->directory() / "context.onnx").c_str(),
                                     options.get(), &created)
                : api_.CreateSessionFromArray(env_.get(), model_.data(), model_.size(),
                                              options.get(), &created);
        session.value.reset(created);
        check(api_, status, "Create selected inference session");
        const auto metadata = require(read_model_metadata(api_, *session.value, fixed));
        if (config_.precision == ComputePrecision::Float32 && metadata.precision != "fp32")
            throw std::runtime_error("Float32 precision requires an fp32 ONNX graph");
        if (config_.kind == BackendKind::Cuda && config_.precision == ComputePrecision::Float16 &&
            metadata.precision != "fp16")
            throw std::runtime_error(
                "CUDA float16 precision requires an fp16 ONNX graph; CUDA has no implicit FP16 "
                "conversion");
        if (sessions_.empty())
            metadata_ = metadata;
        else if (metadata_ != metadata)
            throw std::runtime_error("ONNX bucket metadata differs");
        OrtIoBinding* binding = nullptr;
        check(api_, api_.CreateIoBinding(session.value.get(), &binding),
              "Create inference IO binding");
        BindingOwner owned_binding{binding, api_.ReleaseIoBinding};
        sessions_.push_back(std::move(session));
        bindings_.push_back(std::move(owned_binding));
    }
    void create_views(std::size_t batch) {
        Views views{api_};
        const auto count = static_cast<std::int64_t>(batch);
        const std::array<std::int64_t, 4> input_shape{count, 112, 8, 8};
        const std::array<std::int64_t, 2> policy_shape{count, 1858};
        const std::array<std::int64_t, 2> wdl_shape{count, 3};
        const std::array<std::int64_t, 2> mlh_shape{count, 1};
        const auto& buffer = device_pool_ ? *device_pool_ : *pool_;
        views.input =
            tensor(api_, buffer, buffer.input().first(batch * kInputElementCount), input_shape);
        views.policy = tensor(api_, buffer, buffer.policy().first(batch * 1858), policy_shape);
        views.wdl = tensor(api_, buffer, buffer.wdl().first(batch * 3), wdl_shape);
        views.mlh = tensor(api_, buffer, buffer.mlh().first(batch), mlh_shape);
        if (device_pool_) bind(*bindings_.at(views_.size()), views);
        views_.push_back(std::move(views));
    }
    void bind(OrtIoBinding& binding, const Views& views) {
        api_.ClearBoundInputs(&binding);
        api_.ClearBoundOutputs(&binding);
        check(api_, api_.BindInput(&binding, "planes", views.input.get()), "Bind input");
        check(api_, api_.BindOutput(&binding, "policy", views.policy.get()), "Bind policy output");
        check(api_, api_.BindOutput(&binding, "wdl", views.wdl.get()), "Bind WDL output");
        check(api_, api_.BindOutput(&binding, "mlh", views.mlh.get()), "Bind moves-left output");
    }
    void run(std::size_t batch) {
        const bool gpu = config_.kind != BackendKind::Cpu;
        const auto index =
            gpu ? static_cast<std::size_t>(std::ranges::lower_bound(config_.batch_buckets, batch) -
                                           config_.batch_buckets.begin())
                : batch - 1;
        auto& session = sessions_.at(gpu ? index : 0);
        auto* binding = bindings_.at(gpu ? index : 0).get();
        if (gpu) {
            transfer_->upload(device_pool_->input().first(batch * kInputElementCount),
                              pool_->input().first(batch * kInputElementCount));
        } else
            bind(*binding, views_.at(index));
        // Device bindings never change. Explicit copies refresh the same addresses
        // captured by ORT's fixed-shape TensorRT CUDA graph.
        check(api_, api_.RunWithBinding(session.value.get(), nullptr, binding),
              "Run selected provider");
        check(api_, api_.SynchronizeBoundOutputs(binding), "Synchronize provider outputs");
        if (gpu) {
            transfer_->download(pool_->policy().first(batch * 1858),
                                device_pool_->policy().first(batch * 1858));
            transfer_->download(pool_->wdl().first(batch * 3),
                                device_pool_->wdl().first(batch * 3));
            transfer_->download(pool_->mlh().first(batch), device_pool_->mlh().first(batch));
        }
        for (std::size_t row = 0; row < batch; ++row) {
            require(validate_nn_output(
                std::span<const float, 1858>{pool_->policy().data() + row * 1858, 1858},
                std::span<const float, 3>{pool_->wdl().data() + row * 3, 3}, pool_->mlh()[row]));
        }
    }

    // Reverse destruction keeps values/bindings, buffers, allocator, sessions,
    // environment, and finally their runtime module alive in the required order.
    RuntimeLibrary runtime_;
    const OrtApi& api_;
    BackendConfig config_;
    std::vector<char> model_;
    detail::GpuDependencies gpu_dependencies_;
    Owner<OrtEnv, decltype(OrtApi::ReleaseEnv)> env_;
    BackendInfo info_;
    ModelMetadata metadata_;
    std::string gpu_fingerprint_;
    std::vector<std::unique_ptr<detail::TensorRtCache>> caches_;
    std::vector<Session> sessions_;
    std::unique_ptr<BufferPool> pool_;
    std::unique_ptr<BufferPool> device_pool_;
    std::unique_ptr<detail::GpuTransfer> transfer_;
    std::vector<Views> views_;
    std::vector<BindingOwner> bindings_;
    std::mutex mutex_;
};
}  // namespace
Result<std::size_t> recommended_minibatch(const BackendConfig& config) {
    if (config.kind == BackendKind::Cpu) return std::size_t{64};
    try {
        const auto multiprocessors = detail::query_gpu_multiprocessors(config);
        return std::min<std::size_t>(1024, 2 * multiprocessors);
    } catch (const std::exception& error) {
        return tl::unexpected(std::string{"Automatic MinibatchSize ["} +
                              provider_label(config.kind) + "]: " + error.what());
    }
}
Result<std::unique_ptr<Backend>> create_backend(const BackendConfig& requested) {
    try {
        auto config = validated(requested);
        auto bytes = snapshot(config.model_path, config);
#ifdef _WIN32
        auto runtime =
            require(RuntimeLibrary::load(config.runtime_path, config.gpu_dll_directories));
#else
        auto runtime = require(RuntimeLibrary::load(config.runtime_path));
#endif
        std::unique_ptr<Backend> backend =
            std::make_unique<OrtBackend>(std::move(config), std::move(bytes), std::move(runtime));
        return backend;
    } catch (const std::bad_alloc&) {
        return tl::unexpected(std::string{"Backend construction ["} +
                              provider_label(requested.kind) + "]: out of memory");
    } catch (const std::exception& error) {
        return tl::unexpected(std::string{"Backend construction ["} +
                              provider_label(requested.kind) + "]: " + error.what());
    }
}
}  // namespace mouse::nn
