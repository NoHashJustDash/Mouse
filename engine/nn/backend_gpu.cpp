// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/nn/backend_gpu.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <utility>
#ifdef _WIN32
#include <Windows.h>
#else
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

#if MOUSE_HAS_CUDA_HEADERS
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvml.h>
static_assert(CUDART_VERSION == 12080, "NVIDIA bindings require the pinned CUDA12.8 ABI headers");
#endif

#include "engine/util/sha256.h"

namespace mouse::nn {
Result<std::filesystem::path> default_cache_directory() {
    try {
#ifdef _WIN32
        std::array<wchar_t, 32768> value{};
        const auto count = GetEnvironmentVariableW(L"LOCALAPPDATA", value.data(),
                                                   static_cast<DWORD>(value.size()));
        if (count == 0 || count >= value.size())
            return tl::unexpected("LOCALAPPDATA must name an absolute per-user cache directory");
        const std::filesystem::path parent{value.data()};
        if (!parent.is_absolute())
            return tl::unexpected("LOCALAPPDATA must name an absolute per-user cache directory");
        return (parent / "Mouse" / "cache").lexically_normal();
#else
        if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg != nullptr && *xdg != '\0') {
            const std::filesystem::path parent{xdg};
            if (!parent.is_absolute())
                return tl::unexpected("XDG_CACHE_HOME must name an absolute cache directory");
            return (parent / "Mouse" / "cache").lexically_normal();
        }
        const char* home = std::getenv("HOME");
        if (home == nullptr || *home == '\0' || !std::filesystem::path{home}.is_absolute())
            return tl::unexpected("HOME must name an absolute per-user cache directory");
        return (std::filesystem::path{home} / ".cache" / "Mouse" / "cache").lexically_normal();
#endif
    } catch (const std::exception& error) {
        return tl::unexpected("Cannot resolve per-user cache directory: " +
                              std::string{error.what()});
    }
}
}  // namespace mouse::nn

namespace mouse::nn::detail {
namespace {
std::string digest(const std::string& value) {
    return mouse::sha256(std::as_bytes(std::span{value.data(), value.size()}));
}
std::string file_hash(const std::filesystem::path& path) {
    const auto hash = mouse::sha256_file(path);
    if (!hash) throw std::runtime_error(hash.error());
    return *hash;
}
#if MOUSE_HAS_CUDA_HEADERS
struct LibraryNames {
    const char* windows;
    const char* posix;
};
std::filesystem::path driver_library(LibraryNames names) {
#ifdef _WIN32

    std::array<wchar_t, 32768> directory{};
    const auto count = GetSystemDirectoryW(directory.data(), static_cast<UINT>(directory.size()));
    if (count == 0 || count >= directory.size())
        throw std::runtime_error("Cannot resolve Windows GPU driver directory");
    return std::filesystem::path{directory.data()} / names.windows;
#else

    for (const auto* directory : {"/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib"}) {
        const auto path = std::filesystem::path{directory} / names.posix;
        if (std::filesystem::is_regular_file(path)) return std::filesystem::canonical(path);
    }
    throw std::runtime_error("Missing explicit GPU driver library: " + std::string{names.posix});
#endif
}
class Library {
   public:
    explicit Library(const std::filesystem::path& path) {
        if (!path.is_absolute()) throw std::runtime_error("GPU module path must be absolute");
#ifdef _WIN32
        module_ =
            LoadLibraryExW(path.c_str(), nullptr,
                           LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
#else
        module_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
        if (module_ == nullptr)
            throw std::runtime_error("Cannot load GPU module: " + path.string());
    }
    ~Library() {
        if (module_ == nullptr) return;
#ifdef _WIN32
        static_cast<void>(FreeLibrary(module_));
#else
        static_cast<void>(dlclose(module_));
#endif
    }
    Library(const Library&) = delete;
    Library& operator=(const Library&) = delete;
    Library(Library&&) = delete;
    Library& operator=(Library&&) = delete;
    template <class Function>
    Function symbol(const char* name) const {
#ifdef _WIN32
        const auto address = GetProcAddress(module_, name);
#else
        const auto address = dlsym(module_, name);
#endif
        if (address == nullptr) throw std::runtime_error("Missing GPU API: " + std::string{name});
        static_assert(sizeof(Function) == sizeof(address));
        return std::bit_cast<Function>(address);
    }

   private:
#ifdef _WIN32
    HMODULE module_ = nullptr;
#else
    void* module_ = nullptr;
#endif
};
void gpu_status(int status, std::string_view operation) {
    if (status != 0)
        throw std::runtime_error(std::string{operation} + " failed with GPU status " +
                                 std::to_string(status));
}
// CUresult names/descriptions belong only to the driver API. CUDA runtime
// cudaError_t and NVML nvmlReturn_t retain their separate status handling.
class CudaDriverErrors {
   public:
    explicit CudaDriverErrors(const Library& driver)
        : name_(driver.symbol<decltype(&cuGetErrorName)>("cuGetErrorName")),
          description_(driver.symbol<decltype(&cuGetErrorString)>("cuGetErrorString")) {}

    void check(CUresult status, std::string_view operation) const {
        if (status == CUDA_SUCCESS) return;
        std::string failure = std::string{operation} + " failed with CUDA driver status " +
                              std::to_string(static_cast<int>(status));
        const char* name = nullptr;
        const char* description = nullptr;
        if (name_(status, &name) == CUDA_SUCCESS && name != nullptr && *name != '\0')
            failure += " (" + std::string{name} + ')';
        if (description_(status, &description) == CUDA_SUCCESS && description != nullptr &&
            *description != '\0')
            failure += ": " + std::string{description};
        if (status == CUDA_ERROR_NO_DEVICE || status == CUDA_ERROR_INVALID_DEVICE)
            failure += "; check CUDA_VISIBLE_DEVICES and select an available --device ordinal";
        throw std::runtime_error(failure);
    }

   private:
    decltype(&cuGetErrorName) name_;
    decltype(&cuGetErrorString) description_;
};
std::string terminated(const std::array<char, 256>& value) {
    const auto end = std::ranges::find(value, '\0');
    if (end == value.end()) throw std::runtime_error("Unterminated GPU identity response");
    return {value.begin(), end};
}
std::string cuda_version(int version) {
    if (version <= 0) throw std::runtime_error("Invalid CUDA runtime version");
    return std::to_string(version / 1000) + '.' + std::to_string((version % 1000) / 10);
}
class NvmlSession {
   public:
    explicit NvmlSession(const Library& library)
        : shutdown_(library.symbol<decltype(&nvmlShutdown)>("nvmlShutdown")) {
        gpu_status(library.symbol<decltype(&nvmlInit_v2)>("nvmlInit_v2")(), "Initialize NVML");
    }
    ~NvmlSession() { static_cast<void>(shutdown_()); }
    NvmlSession(const NvmlSession&) = delete;
    NvmlSession& operator=(const NvmlSession&) = delete;
    NvmlSession(NvmlSession&&) = delete;
    NvmlSession& operator=(NvmlSession&&) = delete;

   private:
    decltype(&nvmlShutdown) shutdown_;
};
#endif
class FileLock {
   public:
    FileLock(const std::filesystem::path& path, CacheLockMode mode)
#ifdef _WIN32
        : handle_(CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE,
                              FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS,
                              FILE_ATTRIBUTE_NORMAL, nullptr)) {
        if (handle_ == INVALID_HANDLE_VALUE)
            throw std::system_error(static_cast<int>(GetLastError()), std::system_category(),
                                    "Cannot open TensorRT cache lock");
        // A synchronous byte-range lock waits without polling. Sharing the
        // handle permits other owners to queue; the byte lock stays exclusive.
        // https://learn.microsoft.com/windows/win32/api/fileapi/nf-fileapi-lockfileex
        OVERLAPPED offset{};
        const DWORD flags =
            LOCKFILE_EXCLUSIVE_LOCK | (mode == CacheLockMode::Try ? LOCKFILE_FAIL_IMMEDIATELY : 0U);
        if (!LockFileEx(handle_, flags, 0, 1, 0, &offset)) {
            const auto error = GetLastError();
            static_cast<void>(CloseHandle(handle_));
            handle_ = INVALID_HANDLE_VALUE;
            throw std::system_error(static_cast<int>(error), std::system_category(),
                                    "Cannot acquire TensorRT cache lock");
        }
#else
        // POSIX open requires its variadic permissions argument with O_CREAT.
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
        : descriptor_(::open(path.c_str(), O_CREAT | O_RDWR, 0600)) {
        if (descriptor_ < 0)
            throw std::system_error(errno, std::generic_category(), "Open TensorRT cache lock");
        const int flags = LOCK_EX | (mode == CacheLockMode::Try ? LOCK_NB : 0);
        int result = flock(descriptor_, flags);
        while (result != 0 && errno == EINTR) result = flock(descriptor_, flags);
        if (result != 0) {
            const int saved_errno = errno;
            static_cast<void>(::close(descriptor_));
            descriptor_ = -1;
            throw std::system_error(saved_errno, std::generic_category(),
                                    "Cannot acquire TensorRT cache lock");
        }
#endif
    }
    ~FileLock() {
#ifdef _WIN32
        if (handle_ != INVALID_HANDLE_VALUE) {
            OVERLAPPED offset{};
            static_cast<void>(UnlockFileEx(handle_, 0, 1, 0, &offset));
            static_cast<void>(CloseHandle(handle_));
        }
#else
        if (descriptor_ >= 0) static_cast<void>(::close(descriptor_));
#endif
    }
    FileLock(const FileLock&) = delete;
    FileLock& operator=(const FileLock&) = delete;
    FileLock(FileLock&&) = delete;
    FileLock& operator=(FileLock&&) = delete;

   private:
#ifdef _WIN32
    HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
    int descriptor_ = -1;
#endif
};
std::filesystem::path cache_root(const std::filesystem::path& root) {
    const auto checked = checked_tensor_rt_cache_root(root);
    if (!checked) throw std::runtime_error(checked.error());
    std::filesystem::create_directories(*checked);
    return *checked;
}
std::string payload_manifest(const std::filesystem::path& directory, const std::string& identity) {
    std::string result = "Mouse TensorRT cache v3\nsha256=" + digest(identity) + '\n' + identity;
    const auto context = directory / "context.onnx";
    if (std::filesystem::is_symlink(context) || !std::filesystem::is_regular_file(context) ||
        std::filesystem::file_size(context) == 0 ||
        std::filesystem::file_size(context) > (2ULL << 30U))
        throw std::runtime_error("TensorRT cache has no completed external EPContext model");
    std::vector<std::filesystem::path> files{context};
    bool has_engine = false;
    for (const auto* child : {"engine", "timing"}) {
        if (std::filesystem::is_symlink(directory / child))
            throw std::runtime_error("TensorRT cache payload directory must not be a symlink");
        for (const auto& entry : std::filesystem::recursive_directory_iterator(directory / child)) {
            if (entry.is_symlink()) throw std::runtime_error("TensorRT cache contains a symlink");
            if (!entry.is_regular_file()) continue;
            if (files.size() >= 1024 || entry.file_size() > (2ULL << 30U))
                throw std::runtime_error("TensorRT cache exceeds bounded payload limits");
            files.push_back(entry.path());
            has_engine = has_engine || entry.path().extension() == ".engine";
        }
    }
    if (!has_engine) throw std::runtime_error("TensorRT cache has no completed engine payload");
    std::ranges::sort(files);
    for (const auto& file : files) {
        const auto relative = file.lexically_relative(directory).generic_string();
        if (!std::ranges::all_of(relative, [](unsigned char value) {
                return (value >= 'a' && value <= 'z') || (value >= 'A' && value <= 'Z') ||
                       (value >= '0' && value <= '9') || value == '/' || value == '.' ||
                       value == '_' || value == '-';
            }))
            throw std::runtime_error("Unsupported TensorRT cache payload name");
        result += "file " + relative + ' ' + file_hash(file) + '\n';
    }
    return result;
}
}  // namespace

Result<std::filesystem::path> checked_tensor_rt_cache_root(const std::filesystem::path& root,
                                                           std::size_t graph_name_bytes) {
    try {
        if (root.empty() || !root.is_absolute())
            return tl::unexpected("TensorRT requires an explicit absolute cache directory");
#ifdef _WIN32
        const auto check_windows =
            [graph_name_bytes](const std::filesystem::path& path) -> Result<void> {
            const auto& text = path.native();
            if (text.starts_with(L"\\\\?\\") || text.starts_with(L"\\\\.\\"))
                return tl::unexpected(
                    "TensorRT cache does not support extended/device Windows paths; use a shorter "
                    "absolute ASCII CacheDirectory (--cache-dir)");
            if (!std::ranges::all_of(text,
                                     [](wchar_t value) { return value >= 32 && value <= 126; }))
                return tl::unexpected(
                    "TensorRT ORT1.22 cache paths require ASCII; use an absolute ASCII "
                    "CacheDirectory (--cache-dir)");
            // Pinned ORT1.22: tensorrt_execution_provider.cc:3224-3247 and
            // tensorrt_execution_provider_utils.h:749-760. Prefix "m" removes
            // graph names unless its first hash-substring lookup matches within
            // a graph name. Include the whole name conservatively. Bound hash/
            // index to 20 digits each, SM text to 20, plus FP16 and .profile.
            // 1+1+20+1+20+5+3+20+8 = 79 filename characters. Engine and timing
            // directory names are both six characters; engine names dominate.
            constexpr std::size_t maximum_filename = 79;
            constexpr std::size_t path_limit = 260;
            if (graph_name_bytes > path_limit || text.size() > path_limit)
                return tl::unexpected(
                    "TensorRT cache projected path exceeds Windows MAX_PATH limit 260 "
                    "including NUL. Use a shorter graph name and absolute ASCII CacheDirectory "
                    "(--cache-dir)");
            // A name collision can also make ORT erase a name token rather
            // than its duplicate index. Charge another 20-digit index + '_'
            // for every nonempty name, without guessing whether it collides.
            const std::size_t retained_index = graph_name_bytes == 0 ? 0 : 21;
            const auto projected = text.size() + 1 + 16 + 1 + 6 + 1 + maximum_filename +
                                   graph_name_bytes + retained_index + 1;
            if (projected > path_limit)
                return tl::unexpected("TensorRT cache projected path requires " +
                                      std::to_string(projected) +
                                      " characters including NUL; Windows MAX_PATH limit is 260. "
                                      "Use a shorter absolute ASCII CacheDirectory (--cache-dir), "
                                      "or shorten the ONNX graph name");
            return {};
        };
        // Check both spellings: a long unresolved name fails without touching
        // the filesystem, and existing ancestor links cannot hide a deep target.
        if (const auto checked = check_windows(root); !checked)
            return tl::unexpected(checked.error());
#endif
#ifndef _WIN32
        static_cast<void>(graph_name_bytes);
#endif
        auto resolved = std::filesystem::weakly_canonical(root);
#ifdef _WIN32
        if (const auto checked = check_windows(resolved); !checked)
            return tl::unexpected(checked.error());
#endif
        return resolved;
    } catch (const std::exception& error) {
        return tl::unexpected("Invalid TensorRT cache directory: " + std::string{error.what()});
    }
}
std::filesystem::path explicit_gpu_library(std::span<const std::filesystem::path> directories,
                                           const char* filename) {
    if (filename == nullptr || std::filesystem::path{filename}.filename() != filename)
        throw std::runtime_error("GPU library name must be a single filename");
    std::filesystem::path result;
    for (const auto& directory : directories) {
        if (!directory.is_absolute() || !std::filesystem::is_directory(directory))
            throw std::runtime_error("GPU SDK directory must be an existing absolute directory");
        const auto candidate = directory / filename;
        if (!std::filesystem::is_regular_file(candidate)) continue;
        const auto resolved = std::filesystem::canonical(candidate);
        if (!result.empty() && !std::filesystem::equivalent(result, resolved))
            throw std::runtime_error("Ambiguous explicit GPU library: " + std::string{filename});
        result = resolved;
    }
    if (result.empty())
        throw std::runtime_error("Missing explicit GPU library: " + std::string{filename});
    return result;
}
#ifndef _WIN32
struct ModuleCloser {
    void operator()(void* module) const noexcept { static_cast<void>(dlclose(module)); }
};
#endif
struct GpuTransfer::Impl {
#if MOUSE_HAS_CUDA_HEADERS
    explicit Impl(const BackendConfig& config)
        : runtime(explicit_gpu_library(config.gpu_dll_directories,
#ifdef _WIN32
                                       "cudart64_12.dll"
#else
                                       "libcudart.so.12"
#endif
                                       )),
          set_device(runtime.symbol<decltype(&cudaSetDevice)>("cudaSetDevice")),
          copy(runtime.symbol<decltype(&cudaMemcpy)>("cudaMemcpy")),
          device(config.device_id) {
    }
    void transfer(std::span<float> destination, std::span<const float> source,
                  cudaMemcpyKind kind) const {
        if (destination.size() != source.size())
            throw std::runtime_error("GPU transfer buffer sizes differ");
        gpu_status(set_device(device), "Select transfer device");
        gpu_status(copy(destination.data(), source.data(), source.size_bytes(), kind),
                   "Transfer inference buffers");
    }
    Library runtime;
    decltype(&cudaSetDevice) set_device;
    decltype(&cudaMemcpy) copy;
    int device;
#else
    explicit Impl(const BackendConfig&) {
        throw std::runtime_error("GPU transfer requires a build with CUDA headers");
    }
#endif
};
GpuTransfer::GpuTransfer(const BackendConfig& config) : impl_(std::make_unique<Impl>(config)) {}
GpuTransfer::~GpuTransfer() = default;
void GpuTransfer::upload(std::span<float> device, std::span<const float> host) {
#if MOUSE_HAS_CUDA_HEADERS
    impl_->transfer(device, host, cudaMemcpyHostToDevice);
#else
    static_cast<void>(device);
    static_cast<void>(host);
    throw std::runtime_error("GPU transfer requires CUDA headers");
#endif
}
void GpuTransfer::download(std::span<float> host, std::span<const float> device) {
#if MOUSE_HAS_CUDA_HEADERS
    impl_->transfer(host, device, cudaMemcpyDeviceToHost);
#else
    static_cast<void>(host);
    static_cast<void>(device);
    throw std::runtime_error("GPU transfer requires CUDA headers");
#endif
}

struct GpuDependencies::Impl {
    explicit Impl(const BackendConfig& config) {
#ifndef _WIN32
        if (config.kind == BackendKind::Cpu) return;
        // ORT1.22 CUDA12/cuDNN9 and TensorRT10 dependency SONAMEs. Optional
        // libraries are loaded when supplied because CUDA/cuDNN can load them
        // lazily. Missing mandatory libraries and incomplete supplied closures
        // fail at startup. System C/C++/driver libraries use the system loader.
        std::vector<std::filesystem::path> pending;
        const auto add = [&](const char* name, bool required) {
            const bool present =
                std::ranges::any_of(config.gpu_dll_directories, [name](const auto& directory) {
                    return std::filesystem::is_regular_file(directory / name);
                });
            if (required || present)
                pending.push_back(explicit_gpu_library(config.gpu_dll_directories, name));
        };
        for (const auto* name : {"libcudart.so.12", "libcublasLt.so.12", "libcublas.so.12",
                                 "libcufft.so.11", "libcudnn.so.9"})
            add(name, true);
        for (const auto* name :
             {"libnvJitLink.so.12", "libnvrtc-builtins.so.12.8", "libnvrtc.so.12",
              "libcurand.so.10", "libcusparse.so.12", "libcusolver.so.11", "libcudnn_graph.so.9",
              "libcudnn_ops.so.9", "libcudnn_adv.so.9", "libcudnn_cnn.so.9",
              "libcudnn_heuristic.so.9", "libcudnn_engines_precompiled.so.9",
              "libcudnn_engines_runtime_compiled.so.9"})
            add(name, false);
        if (config.kind == BackendKind::TensorRt)
            for (const auto* name :
                 {"libnvinfer.so.10", "libnvinfer_plugin.so.10", "libnvonnxparser.so.10"})
                add(name, true);
        std::string errors;
        while (!pending.empty()) {
            const auto before = pending.size();
            errors.clear();
            for (auto iterator = pending.begin(); iterator != pending.end();) {
                // The dynamic loader resolves each SONAME after its explicit
                // prerequisite has loaded. Revisit unresolved dependencies;
                // this never changes the selected execution provider.
                void* module = dlopen(iterator->c_str(), RTLD_NOW | RTLD_GLOBAL);
                if (module != nullptr) {
                    std::unique_ptr<void, ModuleCloser> owned{module};
                    modules.push_back(std::move(owned));
                    iterator = pending.erase(iterator);
                } else {
                    const char* reason = dlerror();
                    errors += iterator->string() + ": " +
                              (reason == nullptr ? "unknown loader error" : reason) + '\n';
                    ++iterator;
                }
            }
            if (pending.size() == before)
                throw std::runtime_error("Incomplete explicit GPU SDK dependency closure:\n" +
                                         errors);
        }
#else
        static_cast<void>(config);
#endif
    }
    ~Impl() {
#ifndef _WIN32
        for (auto& module : std::views::reverse(modules)) module.reset();
#endif
    }
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&) = delete;
    Impl& operator=(Impl&&) = delete;
#ifndef _WIN32
    std::vector<std::unique_ptr<void, ModuleCloser>> modules;
#endif
};
GpuDependencies::GpuDependencies(const BackendConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}
GpuDependencies::~GpuDependencies() = default;

GpuIdentity query_gpu_identity(const BackendConfig& config) {
#if MOUSE_HAS_CUDA_HEADERS
    // CUDA driver/runtime and NVML public C ABI signatures are loaded explicitly;
    // no CUDA SDK link, static initialization or ambient PATH dependency exists.
    // Device UUID is queried by CUDA ordinal, respecting CUDA_VISIBLE_DEVICES,
    // rather than incorrectly assuming that NVML and CUDA ordinals are equal.
    const Library driver(driver_library({.windows = "nvcuda.dll", .posix = "libcuda.so.1"}));
    const CudaDriverErrors driver_errors{driver};
    driver_errors.check(driver.symbol<decltype(&cuInit)>("cuInit")(0), "Initialize CUDA driver");
    CUdevice device = 0;
    driver_errors.check(
        driver.symbol<decltype(&cuDeviceGet)>("cuDeviceGet")(&device, config.device_id),
        "Select CUDA device");
    std::array<char, 256> name{};
    driver_errors.check(driver.symbol<decltype(&cuDeviceGetName)>("cuDeviceGetName")(
                            name.data(), static_cast<int>(name.size()), device),
                        "Read CUDA device name");
    CUuuid uuid{};
    driver_errors.check(
        driver.symbol<decltype(&cuDeviceGetUuid_v2)>("cuDeviceGetUuid_v2")(&uuid, device),
        "Read CUDA device UUID");
    int major = 0;
    int minor = 0;
    const auto attribute = driver.symbol<decltype(&cuDeviceGetAttribute)>("cuDeviceGetAttribute");
    driver_errors.check(attribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device),
                        "Read CUDA compute major");
    driver_errors.check(attribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device),
                        "Read CUDA compute minor");
    int driver_api = 0;
    driver_errors.check(
        driver.symbol<decltype(&cuDriverGetVersion)>("cuDriverGetVersion")(&driver_api),
        "Read CUDA driver API");
#ifdef _WIN32
    const auto runtime_path = explicit_gpu_library(config.gpu_dll_directories, "cudart64_12.dll");
#else
    const auto runtime_path = explicit_gpu_library(config.gpu_dll_directories, "libcudart.so.12");
#endif
    const Library runtime(runtime_path);
    int runtime_version = 0;
    gpu_status(
        runtime.symbol<decltype(&cudaRuntimeGetVersion)>("cudaRuntimeGetVersion")(&runtime_version),
        "Read CUDA runtime");
    if (runtime_version != 12080) throw std::runtime_error("GPU backend requires CUDA runtime12.8");
    const Library nvml(driver_library({.windows = "nvml.dll", .posix = "libnvidia-ml.so.1"}));
    const NvmlSession initialized(nvml);
    std::array<char, 256> driver_name{};
    gpu_status(nvml.symbol<decltype(&nvmlSystemGetDriverVersion)>("nvmlSystemGetDriverVersion")(
                   driver_name.data(), static_cast<unsigned int>(driver_name.size())),
               "Read NVIDIA driver version");
    GpuIdentity identity;
    identity.info.device_name = terminated(name);
    identity.info.device_uuid = "GPU-";
    constexpr std::string_view hex = "0123456789abcdef";
    const std::span<const char, 16> uuid_bytes{uuid.bytes};
    for (std::size_t index = 0; index < uuid_bytes.size(); ++index) {
        if (index == 4 || index == 6 || index == 8 || index == 10) identity.info.device_uuid += '-';
        identity.info.device_uuid += hex[static_cast<unsigned char>(uuid_bytes[index]) >> 4U];
        identity.info.device_uuid += hex[static_cast<unsigned char>(uuid_bytes[index]) & 15U];
    }
    identity.info.driver_version = terminated(driver_name);
    identity.info.cuda_version = cuda_version(runtime_version);
    identity.capability = std::to_string(major) + '.' + std::to_string(minor);
    identity.runtime_fingerprint = "sm=" + identity.capability +
                                   "\ndriver_api=" + cuda_version(driver_api) +
                                   "\ncudart=" + file_hash(runtime_path);
#ifdef _WIN32
    identity.runtime_fingerprint +=
        "\ncudnn=" + file_hash(explicit_gpu_library(config.gpu_dll_directories, "cudnn64_9.dll"));
    identity.runtime_fingerprint +=
        "\nort_gpu_provider=" +
        file_hash(config.runtime_path.parent_path() / (config.kind == BackendKind::Cuda
                                                           ? "onnxruntime_providers_cuda.dll"
                                                           : "onnxruntime_providers_tensorrt.dll"));
    identity.runtime_fingerprint +=
        "\nort_shared_provider=" +
        file_hash(config.runtime_path.parent_path() / "onnxruntime_providers_shared.dll");
#else
    identity.runtime_fingerprint +=
        "\ncudnn=" + file_hash(explicit_gpu_library(config.gpu_dll_directories, "libcudnn.so.9"));
    identity.runtime_fingerprint +=
        "\nort_gpu_provider=" +
        file_hash(config.runtime_path.parent_path() /
                  (config.kind == BackendKind::Cuda ? "libonnxruntime_providers_cuda.so"
                                                    : "libonnxruntime_providers_tensorrt.so"));
    identity.runtime_fingerprint +=
        "\nort_shared_provider=" +
        file_hash(config.runtime_path.parent_path() / "libonnxruntime_providers_shared.so");
#endif
    if (config.kind == BackendKind::TensorRt) {
#ifdef _WIN32
        const auto trt_path = explicit_gpu_library(config.gpu_dll_directories, "nvinfer_10.dll");
#else
        const auto trt_path = explicit_gpu_library(config.gpu_dll_directories, "libnvinfer.so.10");
#endif
        const Library trt(trt_path);
        const int version = trt.symbol<std::int32_t (*)()>("getInferLibVersion")();
        if (version <= 0) throw std::runtime_error("Invalid TensorRT runtime version");
        identity.info.tensorrt_version = std::to_string(version / 10000) + '.' +
                                         std::to_string((version % 10000) / 100) + '.' +
                                         std::to_string(version % 100);
        identity.runtime_fingerprint += "\ntensorrt=" + file_hash(trt_path);
    }
    return identity;
#else
    static_cast<void>(config);
    throw std::runtime_error(
        "GPU backend unavailable: this build has no pinned CUDA12.8 ABI headers");
#endif
}

std::size_t query_gpu_multiprocessors(const BackendConfig& config) {
#if MOUSE_HAS_CUDA_HEADERS
    // Driver API only (no runtime, NVML or file hashing), so the minibatch can
    // be chosen for this GPU before any backend or TensorRT engine is built.
    const Library driver(driver_library({.windows = "nvcuda.dll", .posix = "libcuda.so.1"}));
    const CudaDriverErrors driver_errors{driver};
    driver_errors.check(driver.symbol<decltype(&cuInit)>("cuInit")(0), "Initialize CUDA driver");
    CUdevice device = 0;
    driver_errors.check(
        driver.symbol<decltype(&cuDeviceGet)>("cuDeviceGet")(&device, config.device_id),
        "Select CUDA device");
    int count = 0;
    driver_errors.check(driver.symbol<decltype(&cuDeviceGetAttribute)>("cuDeviceGetAttribute")(
                            &count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device),
                        "Read CUDA multiprocessor count");
    if (count <= 0) throw std::runtime_error("CUDA device reports no multiprocessors");
    return static_cast<std::size_t>(count);
#else
    static_cast<void>(config);
    throw std::runtime_error(
        "GPU backend unavailable: this build has no pinned CUDA12.8 ABI headers");
#endif
}

std::string backend_identity(const BackendConfig& config, const BackendInfo& info,
                             const std::string& gpu_fingerprint) {
    std::string identity = "provider_options_revision=4\n";
    const auto append = [&identity](std::string_view key, const std::string& value) {
        identity += std::string{key} + '=' + std::to_string(value.size()) + ':' + value + '\n';
    };
    append("onnx", info.onnx_sha256);
    append("provider", info.provider);
    append("precision", info.precision);
    append("device", info.device_uuid);
    append("device_name", info.device_name);
    append("driver", info.driver_version);
    append("cuda", info.cuda_version);
    append("trt", info.tensorrt_version);
    append("ort", RuntimeLibrary::kPinnedVersion);
    append("ort_module", file_hash(config.runtime_path));
    append("gpu_modules", gpu_fingerprint);
    append("workspace", std::to_string(config.workspace_bytes));
    append("cpu_threads", std::to_string(config.cpu_threads));
    append("deterministic_compute", config.deterministic_compute ? "1" : "0");
    append("device_id", std::to_string(config.device_id));
    std::string buckets;
    for (const auto bucket : config.batch_buckets) buckets += std::to_string(bucket) + ',';
    append("buckets", buckets);
    append("execution",
           "sequential;opt_all;gpu_cpu_fallback=disabled;cuda_tf32=0;copy_default=1;trt_min_graph="
           "1;trt_cache_prefix=m;trt_external_ep_context=1;fixed_bucket_profiles=1;device_io=1;trt_"
           "cuda_graph=1;warmup=three_per_profile");
    return identity;
}

struct TensorRtCache::Impl {
    Impl(const std::filesystem::path& requested, std::string expected, CacheLockMode mode)
        : root(cache_root(requested)),
          identity(std::move(expected)),
          path(root / digest(identity).substr(0, 16)),
          lock(root / (digest(identity).substr(0, 16) + ".lock"), mode) {
        const bool created = std::filesystem::create_directory(path);
        if (std::filesystem::is_symlink(path))
            throw std::runtime_error("TensorRT cache namespace must not be a symlink");
        if (created) {
            std::filesystem::create_directory(path / "engine");
            std::filesystem::create_directory(path / "timing");
        } else {
            if (std::filesystem::is_symlink(path / "manifest.txt"))
                throw std::runtime_error("TensorRT cache manifest must not be a symlink");
            std::ifstream input(path / "manifest.txt", std::ios::binary);
            if (!input || std::filesystem::file_size(path / "manifest.txt") > (256ULL << 10U))
                throw std::runtime_error("Incomplete TensorRT cache; inspect/remove namespace: " +
                                         path.string());
            manifest = std::string{std::istreambuf_iterator<char>{input}, {}};
            if (manifest != payload_manifest(path, identity))
                throw std::runtime_error("TensorRT cache identity/payload mismatch: " +
                                         path.string());
        }
    }
    std::filesystem::path root;
    std::string identity;
    std::filesystem::path path;
    FileLock lock;
    std::string manifest;
};
TensorRtCache::TensorRtCache(const std::filesystem::path& root, std::string identity,
                             CacheLockMode mode)
    : impl_(std::make_unique<Impl>(root, std::move(identity), mode)) {}
TensorRtCache::~TensorRtCache() = default;
const std::filesystem::path& TensorRtCache::directory() const noexcept { return impl_->path; }
bool TensorRtCache::reusable() const noexcept { return !impl_->manifest.empty(); }
void TensorRtCache::commit() {
    const auto manifest = payload_manifest(impl_->path, impl_->identity);
    if (!impl_->manifest.empty()) {
        if (manifest != impl_->manifest)
            throw std::runtime_error("TensorRT cache changed during construction");
        return;
    }
    const auto pending = impl_->path / "manifest.pending";
    const auto destination = impl_->path / "manifest.txt";
    if (std::filesystem::exists(pending) || std::filesystem::exists(destination))
        throw std::runtime_error("TensorRT cache manifest already exists");
    std::ofstream output(pending, std::ios::binary);
    output.write(manifest.data(), static_cast<std::streamsize>(manifest.size()));
    output.close();
    if (!output) throw std::runtime_error("Cannot publish TensorRT cache manifest");
    std::filesystem::rename(pending, destination);
    impl_->manifest = manifest;
}
}  // namespace mouse::nn::detail
