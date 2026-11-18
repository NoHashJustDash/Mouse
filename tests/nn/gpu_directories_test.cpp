// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/nn/gpu_directories.h"

#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>
#include <vector>

namespace {
using mouse::nn::BackendKind;
using mouse::nn::gpu_directories_or_default;
// A fresh folder standing in for the one that holds mouse.exe.
std::filesystem::path executable_directory(const char* name) {
    const auto root = std::filesystem::temp_directory_path() / "mouse-gpu-directories" / name;
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    return root;
}
}  // namespace

TEST_CASE("NVIDIA backends use the gpu folder beside the executable when none is configured",
          "[nn][gpu-directories]") {
    const auto root = executable_directory("default");
    std::filesystem::create_directories(root / "gpu");
    for (const auto kind : {BackendKind::TensorRt, BackendKind::Cuda}) {
        const auto found = gpu_directories_or_default({}, kind, root);
        REQUIRE(found.size() == 1);
        CHECK(found.front() == root / "gpu");
        CHECK(found.front().is_absolute());
    }
}

TEST_CASE("Configured GPU folders are never replaced by the gpu folder", "[nn][gpu-directories]") {
    const auto root = executable_directory("configured");
    std::filesystem::create_directories(root / "gpu");
    const std::vector<std::filesystem::path> configured{root / "cuda", root / "tensorrt"};
    CHECK(gpu_directories_or_default(configured, BackendKind::TensorRt, root) == configured);
}

TEST_CASE("Without a gpu folder nothing is invented, so the backend reports what is missing",
          "[nn][gpu-directories]") {
    const auto root = executable_directory("absent");
    CHECK(gpu_directories_or_default({}, BackendKind::TensorRt, root).empty());
    // A file called gpu is not a folder of libraries.
    const auto file_root = executable_directory("file");
    {
        std::ofstream{file_root / "gpu"} << "not a folder";
    }
    CHECK(gpu_directories_or_default({}, BackendKind::TensorRt, file_root).empty());
}

TEST_CASE("The CPU and DirectML backends never pick up the gpu folder", "[nn][gpu-directories]") {
    const auto root = executable_directory("other-backends");
    std::filesystem::create_directories(root / "gpu");
    CHECK(gpu_directories_or_default({}, BackendKind::Cpu, root).empty());
    CHECK(gpu_directories_or_default({}, BackendKind::DirectMl, root).empty());
}
