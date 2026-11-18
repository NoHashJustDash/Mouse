// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace mouse::nn {
// What `mouse doctor` found on this computer. Gathered by run_doctor_command; a plain
// value so the report can be tested without a GPU.
struct DoctorFacts {
    std::string version;
    // Which build this is: DirectML (any DirectX 12 GPU), NVIDIA (CUDA and TensorRT), or
    // one whose ONNX Runtime has no GPU provider at all and so runs on the CPU only.
    bool directml_build = false;
    bool cpu_only_build = false;
    // NVIDIA builds.
    bool nvidia_driver = false;
    bool plugin_build = false;
    bool plugin_present = false;
    // The folders searched for NVIDIA's libraries, and whether the user named them.
    std::vector<std::filesystem::path> gpu_directories;
    bool gpu_directories_configured = false;
    std::vector<std::string> missing_libraries;
    // DirectML builds.
    bool directml_present = false;
    // The GPU the default device resolves to, or why it could not be queried.
    std::string gpu_name;
    std::string gpu_problem;
};
struct DoctorReport {
    std::string text;
    bool ready = false;
};
// The NVIDIA libraries a TensorRT backend needs in its GPU folders, by file name. On
// Windows this is the set a fresh engine build and run were verified with, and the set
// scripts/setup-gpu.ps1 installs; on Linux, the mandatory libraries GpuDependencies loads.
[[nodiscard]] std::span<const char* const> required_gpu_libraries();
// The required libraries found in none of `directories`.
[[nodiscard]] std::vector<std::string> missing_gpu_libraries(
    std::span<const std::filesystem::path> directories);
// The report in plain words, ending in "Result: ready" or "Result: not ready" and, when
// not ready, what to do about it.
[[nodiscard]] DoctorReport doctor_report(const DoctorFacts& facts);
// `mouse doctor [--gpu-dll-dir <dir>]...`: prints the report; 0 when ready, 1 when not,
// 2 on a usage error. Changes nothing on the computer.
int run_doctor_command(std::span<char*> arguments);
}  // namespace mouse::nn
