// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <onnxruntime_c_api.h>

#include <filesystem>
#include <memory>
#include <span>
#include <string>

#include "engine/util/result.h"

namespace mouse {
// Owns the runtime module and its explicitly registered Windows SDK
// directories. The owner must outlive every ORT object and call obtained
// through api(). Reads may run concurrently; load/move/destruction require
// external synchronization. Windows load() establishes the process DLL policy,
// so call it during startup.
// On Linux a validated ORT image stays resident until process exit, preserving
// its process-lifetime globals. Individual handles and Ort objects still release
// normally; rejected images are never promoted to residency.
class RuntimeLibrary {
   public:
    static constexpr const char* kPinnedVersion = "1.22.0";

    [[nodiscard]] static Result<RuntimeLibrary> load(
        const std::filesystem::path& library_path,
        std::span<const std::filesystem::path> gpu_directories = {});
    [[nodiscard]] static Result<std::filesystem::path> executable_path();

    ~RuntimeLibrary();
    RuntimeLibrary(RuntimeLibrary&&) noexcept;
    RuntimeLibrary& operator=(RuntimeLibrary&&) noexcept;
    RuntimeLibrary(const RuntimeLibrary&) = delete;
    RuntimeLibrary& operator=(const RuntimeLibrary&) = delete;

    // Like standard library owners, accessing a moved-from owner is invalid.
    [[nodiscard]] const std::string& version() const;
    [[nodiscard]] const OrtApi& api() const;

   private:
    struct Impl;
    explicit RuntimeLibrary(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};
}  // namespace mouse
