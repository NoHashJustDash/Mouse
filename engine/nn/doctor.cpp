// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/nn/doctor.h"

#include <algorithm>
#include <array>
#include <exception>
#include <iostream>
#include <string_view>
#include <system_error>

#ifdef _WIN32
#include <Windows.h>
#endif

#include "engine/nn/backend.h"
#include "engine/nn/backend_gpu.h"
#include "engine/nn/gpu_directories.h"
#include "engine/util/build_info.h"
#include "engine/util/runtime_library.h"

namespace mouse::nn {
namespace {
#ifdef _WIN32
constexpr std::array kRequiredGpuLibraries{
    "cudart64_12.dll",
    "cublas64_12.dll",
    "cublasLt64_12.dll",
    "cufft64_11.dll",
    "cudnn64_9.dll",
    "cudnn_adv64_9.dll",
    "cudnn_cnn64_9.dll",
    "cudnn_engines_precompiled64_9.dll",
    "cudnn_engines_runtime_compiled64_9.dll",
    "cudnn_graph64_9.dll",
    "cudnn_heuristic64_9.dll",
    "cudnn_ops64_9.dll",
    "nvinfer_10.dll",
    "nvinfer_builder_resource_10.dll",
    "nvinfer_plugin_10.dll",
    "nvonnxparser_10.dll",
};
constexpr const char* kPluginLibrary = "mouse_moe_trt.dll";
constexpr const char* kSetupScript = "setup-gpu.ps1";
#else
constexpr std::array kRequiredGpuLibraries{
    "libcudart.so.12", "libcublasLt.so.12", "libcublas.so.12",         "libcufft.so.11",
    "libcudnn.so.9",   "libnvinfer.so.10",  "libnvinfer_plugin.so.10", "libnvonnxparser.so.10",
};
constexpr const char* kPluginLibrary = "libmouse_moe_trt.so";
constexpr const char* kSetupScript = "";
#endif

// The NVIDIA driver's CUDA library, in the places the backend loads it from
// (driver_library in backend_gpu.cpp).
bool nvidia_driver_installed() {
    std::error_code error;
#ifdef _WIN32
    std::array<wchar_t, 32768> directory{};
    const auto count = GetSystemDirectoryW(directory.data(), static_cast<UINT>(directory.size()));
    if (count == 0 || count >= directory.size()) return false;
    return std::filesystem::is_regular_file(std::filesystem::path{directory.data()} / "nvcuda.dll",
                                            error);
#else
    for (const auto* directory : {"/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib"})
        if (std::filesystem::is_regular_file(std::filesystem::path{directory} / "libcuda.so.1",
                                             error))
            return true;
    return false;
#endif
}

void line(std::string& text, bool ok, std::string_view message) {
    text += ok ? "[ok]      " : "[problem] ";
    text += message;
    text += '\n';
}
std::string setup_advice() {
    std::string advice;
    if (*kSetupScript != '\0')
        advice = std::string{"Run "} + kSetupScript +
                 " (beside the Mouse executable) to download NVIDIA's libraries, or set ";
    else
        advice = "Set ";
    return advice +
           "the GpuDllDirectories option to the folders that hold the CUDA, cuDNN and TensorRT "
           "libraries.";
}

DoctorReport nvidia_report(const DoctorFacts& facts, std::string text) {
    std::vector<std::string> advice;
    line(text, facts.nvidia_driver,
         facts.nvidia_driver ? "NVIDIA driver found" : "No NVIDIA driver found");
    if (!facts.nvidia_driver)
#ifdef _WIN32
        advice.emplace_back(
            "This build needs an NVIDIA GPU and its driver. On an AMD or Intel GPU, or with no "
            "GPU, use Mouse's DirectML build.");
#else
        advice.emplace_back(
            "The CUDA and TensorRT backends need an NVIDIA GPU and its driver. "
            "Without one, set the Backend option to cpu (slow).");
#endif
    if (facts.gpu_directories.empty()) {
        line(text, false,
             "No GPU library folder: GpuDllDirectories is empty and there is no "
             "gpu folder beside the Mouse executable");
    } else {
        for (const auto& directory : facts.gpu_directories)
            line(text, true,
                 std::string{facts.gpu_directories_configured ? "GPU library folder (configured): "
                                                              : "GPU library folder (found beside "
                                                                "the Mouse executable): "} +
                     directory.string());
    }
    for (const auto& name : facts.missing_libraries) line(text, false, "Missing library: " + name);
    if (facts.missing_libraries.empty() && !facts.gpu_directories.empty())
        line(text, true, "All NVIDIA libraries found");
    if (!facts.missing_libraries.empty() || facts.gpu_directories.empty())
        advice.push_back(setup_advice());
    if (facts.plugin_build) {
        line(text, facts.plugin_present,
             std::string{facts.plugin_present ? "Expert plugin found: "
                                              : "Missing expert plugin: "} +
                 kPluginLibrary);
        if (!facts.plugin_present)
            advice.push_back(std::string{"Unpack the Mouse download again: "} + kPluginLibrary +
                             " must be beside the Mouse executable.");
    }
    if (!facts.gpu_name.empty()) line(text, true, "GPU: " + facts.gpu_name);
    if (!facts.gpu_problem.empty()) {
        line(text, false, "GPU check failed: " + facts.gpu_problem);
        advice.emplace_back("Update the NVIDIA driver, then run doctor again.");
    }
    const bool ready = advice.empty() && !facts.gpu_name.empty();
    text += ready ? "Result: ready\n" : "Result: not ready\n";
    for (const auto& item : advice) text += "  " + item + '\n';
    return {.text = std::move(text), .ready = ready};
}

DoctorReport directml_report(const DoctorFacts& facts, std::string text) {
    std::vector<std::string> advice;
    line(text, facts.directml_present,
         facts.directml_present ? "DirectML.dll found beside the Mouse executable"
                                : "DirectML.dll is missing beside the Mouse executable");
    if (!facts.directml_present)
        advice.emplace_back(
            "Unpack the Mouse download again: DirectML.dll must be beside the "
            "Mouse executable.");
    if (!facts.gpu_name.empty()) line(text, true, "GPU: " + facts.gpu_name);
    if (!facts.gpu_problem.empty()) {
        line(text, false, "GPU check failed: " + facts.gpu_problem);
        advice.emplace_back(
            "DirectML needs a DirectX 12 GPU and a current driver. Without one, "
            "set the Backend option to cpu (slow).");
    }
    const bool ready = advice.empty() && !facts.gpu_name.empty();
    text += ready ? "Result: ready\n" : "Result: not ready\n";
    for (const auto& item : advice) text += "  " + item + '\n';
    return {.text = std::move(text), .ready = ready};
}
}  // namespace

std::span<const char* const> required_gpu_libraries() { return kRequiredGpuLibraries; }

std::vector<std::string> missing_gpu_libraries(std::span<const std::filesystem::path> directories) {
    std::vector<std::string> missing;
    for (const auto* name : kRequiredGpuLibraries) {
        const bool found = std::ranges::any_of(directories, [name](const auto& directory) {
            std::error_code error;
            return std::filesystem::is_regular_file(directory / name, error);
        });
        if (!found) missing.emplace_back(name);
    }
    return missing;
}

DoctorReport doctor_report(const DoctorFacts& facts) {
    if (facts.cpu_only_build) {
        std::string text = "Mouse " + facts.version + " doctor (CPU-only build)\n";
        line(text, true, "This build's ONNX Runtime has no GPU provider, so it runs on the CPU");
        text += "Result: ready\n  Set the Backend option to cpu. Expect it to be slow.\n";
        return {.text = std::move(text), .ready = true};
    }
    std::string text = "Mouse " + facts.version + " doctor (" +
                       (facts.directml_build ? "DirectML build: any DirectX 12 GPU"
                                             : "NVIDIA build: CUDA and TensorRT") +
                       ")\n";
    return facts.directml_build ? directml_report(facts, std::move(text))
                                : nvidia_report(facts, std::move(text));
}

int run_doctor_command(std::span<char*> arguments) {
    DoctorFacts facts;
    facts.version = std::string{build::version};
    facts.directml_build = MOUSE_DIRECTML != 0;
    facts.cpu_only_build = MOUSE_DIRECTML == 0 && MOUSE_ORT_GPU_PROVIDERS == 0;
    if (facts.cpu_only_build) {
        const auto report = doctor_report(facts);
        std::cout << report.text;
        return 0;
    }
    for (std::size_t index = 0; index < arguments.size(); index += 2) {
        if (std::string_view{arguments[index]} != "--gpu-dll-dir" ||
            index + 1 == arguments.size()) {
            std::cerr << "Mouse error: Usage: mouse doctor [--gpu-dll-dir <dir>]...\n";
            return 2;
        }
        const std::filesystem::path directory{arguments[index + 1]};
        if (!directory.is_absolute()) {
            std::cerr << "Mouse error: --gpu-dll-dir must be an absolute path\n";
            return 2;
        }
        facts.gpu_directories.push_back(directory);
    }
    const auto executable = RuntimeLibrary::executable_path();
    if (!executable) {
        std::cerr << "Mouse error: " << executable.error() << '\n';
        return 1;
    }
    const auto folder = executable->parent_path();
    BackendConfig config;
    config.runtime_path = folder / build::runtime_filename;
    std::error_code error;
    try {
        if (facts.directml_build) {
            facts.directml_present =
                std::filesystem::is_regular_file(folder / "DirectML.dll", error);
            config.kind = BackendKind::DirectMl;
            if (facts.directml_present) {
                const auto adapter = detail::query_directml_adapter(config);
                facts.gpu_name = adapter.identity.info.device_name + ", driver " +
                                 adapter.identity.info.driver_version;
            }
        } else {
            config.kind = BackendKind::TensorRt;
            facts.gpu_directories_configured = !facts.gpu_directories.empty();
            facts.gpu_directories =
                gpu_directories_or_default(std::move(facts.gpu_directories), config.kind, folder);
            facts.missing_libraries = missing_gpu_libraries(facts.gpu_directories);
            facts.nvidia_driver = nvidia_driver_installed();
#if MOUSE_TRT_MOE_PLUGIN
            facts.plugin_build = true;
            facts.plugin_present = std::filesystem::is_regular_file(folder / kPluginLibrary, error);
#endif
            if (facts.nvidia_driver && facts.missing_libraries.empty()) {
                config.gpu_dll_directories = facts.gpu_directories;
                const auto identity = detail::query_gpu_identity(config);
                facts.gpu_name =
                    identity.info.device_name + ", driver " + identity.info.driver_version;
            }
        }
    } catch (const std::exception& failure) {
        facts.gpu_problem = failure.what();
    }
    const auto report = doctor_report(facts);
    std::cout << report.text;
    return report.ready ? 0 : 1;
}
}  // namespace mouse::nn
