// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/util/runtime_library.h"

#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#ifdef __linux__
#include <dlfcn.h>
namespace {
struct CloseModule {
    void operator()(void* module) const noexcept { static_cast<void>(dlclose(module)); }
};
}  // namespace
#endif

static_assert(!std::is_copy_constructible_v<mouse::RuntimeLibrary>);
static_assert(std::is_nothrow_move_constructible_v<mouse::RuntimeLibrary>);

TEST_CASE("The executable path is absolute and exists", "[runtime]") {
    const auto executable = mouse::RuntimeLibrary::executable_path();
    REQUIRE(executable.has_value());
    CHECK(executable->is_absolute());
    CHECK(std::filesystem::is_regular_file(*executable));
}

TEST_CASE("Relative runtime paths cannot bind an ambient runtime", "[runtime]") {
    const auto runtime = mouse::RuntimeLibrary::load("onnxruntime.dll");
    REQUIRE_FALSE(runtime.has_value());
    CHECK(runtime.error().find("absolute") != std::string::npos);
}

TEST_CASE("A missing runtime produces an actionable load error", "[runtime]") {
    const auto missing = std::filesystem::path{MOUSE_TEST_RUNTIME_PATH}.parent_path() /
                         "mouse-intentionally-missing-ort-library";
    const auto runtime = mouse::RuntimeLibrary::load(missing);
    REQUIRE_FALSE(runtime.has_value());
    CHECK(runtime.error().find("mouse-intentionally-missing-ort-library") != std::string::npos);
}

TEST_CASE("Wrong runtime versions are rejected before accessing the versioned API", "[runtime]") {
    const auto runtime = mouse::RuntimeLibrary::load(MOUSE_TEST_WRONG_RUNTIME_PATH);
    REQUIRE_FALSE(runtime.has_value());
    CHECK(runtime.error().find("0.0.0") != std::string::npos);
    CHECK(runtime.error().find("1.22.0") != std::string::npos);
}

TEST_CASE("Invalid explicit GPU directories fail instead of searching PATH", "[runtime]") {
    const std::filesystem::path relative{"relative-sdk-directory"};
    const auto relative_result =
        mouse::RuntimeLibrary::load(MOUSE_TEST_RUNTIME_PATH, {&relative, 1});
    REQUIRE_FALSE(relative_result.has_value());
    CHECK(relative_result.error().find("absolute") != std::string::npos);

    const auto missing = std::filesystem::path{MOUSE_TEST_RUNTIME_PATH}.parent_path() /
                         "mouse-intentionally-missing-sdk-directory";
    const auto missing_result = mouse::RuntimeLibrary::load(MOUSE_TEST_RUNTIME_PATH, {&missing, 1});
    REQUIRE_FALSE(missing_result.has_value());
    CHECK(missing_result.error().find("mouse-intentionally-missing-sdk-directory") !=
          std::string::npos);
}

TEST_CASE("The pinned runtime API remains callable after ownership moves", "[runtime]") {
    auto source = mouse::RuntimeLibrary::load(MOUSE_TEST_RUNTIME_PATH);
    REQUIRE(source.has_value());
    auto runtime = std::move(*source);
    source = tl::unexpected(std::string{"destroy moved-from owner"});
    REQUIRE(runtime.version() == "1.22.0");
    const auto& api = runtime.api();
    {
        OrtEnv* raw_environment = nullptr;
        OrtStatus* status =
            api.CreateEnv(ORT_LOGGING_LEVEL_ERROR, "mouse-runtime-test", &raw_environment);
        if (status != nullptr) {
            const std::string message{api.GetErrorMessage(status)};
            api.ReleaseStatus(status);
            FAIL(message);
        }
        std::unique_ptr<OrtEnv, decltype(api.ReleaseEnv)> environment{raw_environment,
                                                                      api.ReleaseEnv};
        REQUIRE(environment != nullptr);
    }

    auto other = mouse::RuntimeLibrary::load(MOUSE_TEST_RUNTIME_PATH);
    REQUIRE(other.has_value());
    *other = std::move(runtime);
    CHECK(other->version() == "1.22.0");
    CHECK(other->api().GetErrorCode != nullptr);
}

#ifdef __linux__
TEST_CASE("Linux keeps only validated ORT modules resident across owner destruction",
          "[runtime][linux-loader]") {
    {
        const auto runtime = mouse::RuntimeLibrary::load(MOUSE_TEST_RUNTIME_PATH);
        REQUIRE(runtime.has_value());
    }
    const std::unique_ptr<void, CloseModule> retained{
        dlopen(MOUSE_TEST_RUNTIME_PATH, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD)};
    CHECK(retained != nullptr);
    {
        const auto wrong = mouse::RuntimeLibrary::load(MOUSE_TEST_WRONG_RUNTIME_PATH);
        REQUIRE_FALSE(wrong.has_value());
    }
    const std::unique_ptr<void, CloseModule> rejected{
        dlopen(MOUSE_TEST_WRONG_RUNTIME_PATH, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD)};
    CHECK(rejected == nullptr);
}
#endif
