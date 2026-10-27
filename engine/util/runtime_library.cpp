// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/util/runtime_library.h"

#include <bit>
#include <cassert>
#include <cstdint>
#include <exception>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif
#endif

namespace mouse {
namespace {
std::string path_text(const std::filesystem::path& path) {
    const auto utf8 = path.generic_u8string();
    return {utf8.begin(), utf8.end()};
}

#ifdef _WIN32
std::string windows_error() {
    return std::system_category().message(static_cast<int>(GetLastError()));
}
#endif
}  // namespace

struct RuntimeLibrary::Impl {
    Impl() = default;
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&) = delete;
    Impl& operator=(Impl&&) = delete;

#ifdef _WIN32
    HMODULE module = nullptr;
    std::vector<DLL_DIRECTORY_COOKIE> directory_cookies;
#else
    void* module = nullptr;
#endif
    const OrtApi* api = nullptr;
    std::string version;

    ~Impl() {
#ifdef _WIN32
        if (module != nullptr) {
            (void)FreeLibrary(module);
        }
        // Dependencies remain discoverable until the runtime finishes unloading.
        for (auto it = directory_cookies.rbegin(); it != directory_cookies.rend(); ++it) {
            (void)RemoveDllDirectory(*it);
        }
#else
        if (module != nullptr) {
            (void)dlclose(module);
        }
#endif
    }
};

RuntimeLibrary::RuntimeLibrary(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
RuntimeLibrary::~RuntimeLibrary() = default;
RuntimeLibrary::RuntimeLibrary(RuntimeLibrary&&) noexcept = default;
RuntimeLibrary& RuntimeLibrary::operator=(RuntimeLibrary&&) noexcept = default;

const std::string& RuntimeLibrary::version() const {
    assert(impl_ != nullptr);
    return impl_->version;
}

const OrtApi& RuntimeLibrary::api() const {
    assert(impl_ != nullptr && impl_->api != nullptr);
    return *impl_->api;
}

Result<RuntimeLibrary> RuntimeLibrary::load(
    const std::filesystem::path& library_path,
    std::span<const std::filesystem::path> gpu_directories) {
    try {
        if (!library_path.is_absolute()) {
            return tl::unexpected("ONNX Runtime library path must be absolute: " +
                                  path_text(library_path));
        }
        std::error_code error;
        const auto resolved_path = std::filesystem::canonical(library_path, error);
        if (error || !std::filesystem::is_regular_file(resolved_path, error)) {
            return tl::unexpected("Cannot load ONNX Runtime library '" + path_text(library_path) +
                                  "': " + (error ? error.message() : "not a regular file"));
        }
        std::vector<std::filesystem::path> resolved_directories;
        resolved_directories.reserve(gpu_directories.size());
        for (const auto& directory : gpu_directories) {
            if (!directory.is_absolute()) {
                return tl::unexpected("GPU DLL directory must be absolute: " +
                                      path_text(directory));
            }
            auto resolved = std::filesystem::canonical(directory, error);
            if (error || !std::filesystem::is_directory(resolved, error)) {
                return tl::unexpected("Invalid GPU DLL directory '" + path_text(directory) +
                                      "': " + (error ? error.message() : "not a directory"));
            }
            resolved_directories.push_back(std::move(resolved));
        }
        auto impl = std::make_unique<Impl>();
#ifdef _WIN32
        // Restrict transitive and delayed provider loads to explicit directories,
        // the application directory and System32. Neither PATH nor cwd is searched.
        if (!SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)) {
            return tl::unexpected("Could not restrict Windows DLL search directories: " +
                                  windows_error());
        }
        impl->directory_cookies.reserve(resolved_directories.size());
        for (const auto& directory : resolved_directories) {
            const auto cookie = AddDllDirectory(directory.c_str());
            if (cookie == nullptr) {
                return tl::unexpected("Could not register GPU DLL directory '" +
                                      path_text(directory) + "': " + windows_error());
            }
            impl->directory_cookies.push_back(cookie);
        }
        impl->module =
            LoadLibraryExW(resolved_path.c_str(), nullptr,
                           LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (impl->module == nullptr) {
            return tl::unexpected("Cannot load ONNX Runtime library '" + path_text(resolved_path) +
                                  "': " + windows_error());
        }
        const auto symbol = GetProcAddress(impl->module, "OrtGetApiBase");
        if (symbol == nullptr) {
            return tl::unexpected("ONNX Runtime library '" + path_text(resolved_path) +
                                  "' does not export OrtGetApiBase: " + windows_error());
        }
        // GetProcAddress returns a generic function pointer. This platform-defined
        // conversion avoids linking an import library that could bind System32 ORT.
        using GetApiBase = const OrtApiBase*(ORT_API_CALL*)() noexcept;
        const auto get_api_base = std::bit_cast<GetApiBase>(symbol);
#else
        if (!resolved_directories.empty()) {
            return tl::unexpected(
                std::string{"Explicit GPU DLL directories are supported on Windows; "
                            "Unix runtime dependencies must use an app-local RUNPATH"});
        }
        impl->module = dlopen(resolved_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (impl->module == nullptr) {
            const char* message = dlerror();
            return tl::unexpected("Cannot load ONNX Runtime library '" + path_text(resolved_path) +
                                  "': " + (message != nullptr ? message : "dlopen failed"));
        }
        using GetApiBase = const OrtApiBase*(ORT_API_CALL*)() noexcept;
        (void)dlerror();
        void* symbol = dlsym(impl->module, "OrtGetApiBase");
        const char* symbol_error = dlerror();
        if (symbol_error != nullptr || symbol == nullptr) {
            return tl::unexpected("ONNX Runtime library '" + path_text(resolved_path) +
                                  "' does not export OrtGetApiBase: " +
                                  (symbol_error != nullptr ? symbol_error : "null symbol"));
        }
        // POSIX specifies dlsym's conversion to the exported function pointer.
        const auto get_api_base = std::bit_cast<GetApiBase>(symbol);
#endif
        const OrtApiBase* base = get_api_base();
        if (base == nullptr || base->GetVersionString == nullptr) {
            return tl::unexpected(std::string{"ONNX Runtime library returned an invalid API base"});
        }
        const char* version = base->GetVersionString();
        if (version == nullptr) {
            return tl::unexpected(
                std::string{"ONNX Runtime library returned a null version string"});
        }
        impl->version = version;
        // This must precede GetApi: even a newer API with the same ABI is outside
        // the pinned and verified toolchain contract (docs/TOOLCHAIN.md).
        if (impl->version != kPinnedVersion) {
            return tl::unexpected("ONNX Runtime version mismatch: loaded " + impl->version +
                                  " from '" + path_text(resolved_path) + "', required " +
                                  kPinnedVersion);
        }
        if (base->GetApi == nullptr) {
            return tl::unexpected(
                std::string{"ONNX Runtime library returned no versioned API accessor"});
        }
        impl->api = base->GetApi(ORT_API_VERSION);
        if (impl->api == nullptr) {
            return tl::unexpected("ONNX Runtime " + impl->version + " does not support C API " +
                                  std::to_string(ORT_API_VERSION));
        }
#ifdef __linux__
        // The pinned ORT image has process-lifetime constructor/dispatch globals.
        // Repeated dlclose/reload abandons them (see the ASan loader regression).
        // Promote only the validated image to process residency; failed version
        // probes still unload. Every loader reference and every Ort object keeps
        // its ordinary RAII release. This policy applies to all Linux builds.
        void* resident =
            dlopen(resolved_path.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD | RTLD_NODELETE);
        if (resident == nullptr) {
            const char* message = dlerror();
            return tl::unexpected("Cannot retain validated ONNX Runtime image: " +
                                  std::string{message != nullptr ? message : "dlopen failed"});
        }
        if (dlclose(resident) != 0) {
            const char* message = dlerror();
            return tl::unexpected("Cannot release ONNX Runtime residency reference: " +
                                  std::string{message != nullptr ? message : "dlclose failed"});
        }
#endif
        return RuntimeLibrary{std::move(impl)};
    } catch (const std::exception& error) {
        return tl::unexpected("Could not initialize ONNX Runtime: " + std::string{error.what()});
    }
}

Result<std::filesystem::path> RuntimeLibrary::executable_path() {
    try {
#ifdef _WIN32
        std::vector<wchar_t> buffer(260);
        while (true) {
            const auto length =
                GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
            if (length == 0) {
                return tl::unexpected("Cannot determine executable path: " + windows_error());
            }
            if (static_cast<std::size_t>(length) < buffer.size()) {
                return std::filesystem::canonical(
                    std::filesystem::path{std::wstring{buffer.data(), length}});
            }
            if (buffer.size() == 32768) {
                return tl::unexpected(
                    std::string{"Executable path exceeds the Windows path length limit"});
            }
            buffer.resize(buffer.size() < 16384 ? buffer.size() * 2 : 32768);
        }
#elif defined(__APPLE__)
        std::uint32_t size = 0;
        (void)_NSGetExecutablePath(nullptr, &size);
        std::vector<char> buffer(size);
        if (_NSGetExecutablePath(buffer.data(), &size) != 0) {
            return tl::unexpected(std::string{"Cannot determine executable path"});
        }
        return std::filesystem::canonical(buffer.data());
#else
        std::vector<char> buffer(256);
        while (true) {
            const auto length = readlink("/proc/self/exe", buffer.data(), buffer.size());
            if (length < 0) {
                return tl::unexpected("Cannot determine executable path: " +
                                      std::error_code{errno, std::generic_category()}.message());
            }
            if (static_cast<std::size_t>(length) < buffer.size()) {
                return std::filesystem::canonical(std::filesystem::path{
                    std::string{buffer.data(), static_cast<std::size_t>(length)}});
            }
            buffer.resize(buffer.size() * 2);
        }
#endif
    } catch (const std::exception& error) {
        return tl::unexpected("Cannot determine executable path: " + std::string{error.what()});
    }
}
}  // namespace mouse
