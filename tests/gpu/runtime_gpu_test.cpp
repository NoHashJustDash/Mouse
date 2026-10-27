// SPDX-License-Identifier: GPL-3.0-or-later
#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "engine/nn/backend.h"
#include "engine/util/runtime_library.h"

namespace {
class SanitizedPath {
   public:
    SanitizedPath() {
        if (const char* path = std::getenv("PATH")) {
            original_ = path;
        }
        // No compiler, Python, CUDA, cuDNN or TensorRT directory remains on PATH.
#ifdef _WIN32
        if (_putenv_s("PATH", "") != 0) {
#else
        if (setenv("PATH", "", 1) != 0) {
#endif
            throw std::runtime_error("Could not sanitize PATH for the GPU regression test");
        }
    }

    ~SanitizedPath() {
#ifdef _WIN32
        (void)_putenv_s("PATH", original_ ? original_->c_str() : "");
#else
        if (original_) {
            (void)setenv("PATH", original_->c_str(), 1);
        } else {
            (void)unsetenv("PATH");
        }
#endif
    }

    SanitizedPath(const SanitizedPath&) = delete;
    SanitizedPath& operator=(const SanitizedPath&) = delete;
    SanitizedPath(SanitizedPath&&) = delete;
    SanitizedPath& operator=(SanitizedPath&&) = delete;

   private:
    std::optional<std::string> original_;
};

void check_status(const OrtApi& api, OrtStatus* status) {
    if (status != nullptr) {
        const std::string message{api.GetErrorMessage(status)};
        api.ReleaseStatus(status);
        throw std::runtime_error(message);
    }
}

template <class T>
using OrtHandle = std::unique_ptr<T, void(ORT_API_CALL*)(T*)>;

std::vector<std::filesystem::path> gpu_directories() {
    const char* configured = std::getenv("MOUSE_GPU_DLL_DIRECTORIES");
    if (configured == nullptr || *configured == '\0') {
        throw std::runtime_error(
            "Set MOUSE_GPU_DLL_DIRECTORIES to explicit CUDA, cuDNN and "
            "TensorRT DLL directories before running [gpu] tests");
    }
    const std::string_view value{configured};
    if (value.back() == ';') {
        throw std::runtime_error("MOUSE_GPU_DLL_DIRECTORIES contains an empty directory");
    }
    std::vector<std::filesystem::path> directories;
    std::size_t begin = 0;
    while (begin < value.size()) {
        const auto end = value.find(';', begin);
        const auto entry = value.substr(begin, end == std::string_view::npos ? end : end - begin);
        if (entry.empty()) {
            throw std::runtime_error("MOUSE_GPU_DLL_DIRECTORIES contains an empty directory");
        }
        directories.emplace_back(entry);
        if (end == std::string_view::npos) {
            break;
        }
        begin = end + 1;
    }
    return directories;
}

void run_gpu_model(bool use_tensorrt) {
    const auto directories = gpu_directories();
    const SanitizedPath sanitized_path;
    const char* sanitized_value = std::getenv("PATH");
    REQUIRE((sanitized_value == nullptr || *sanitized_value == '\0'));
    auto runtime_result = mouse::RuntimeLibrary::load(MOUSE_TEST_RUNTIME_PATH, directories);
    if (!runtime_result) {
        FAIL(runtime_result.error());
    }
    auto runtime = std::move(*runtime_result);
    const auto& api = runtime.api();

    OrtEnv* raw_environment = nullptr;
    check_status(api, api.CreateEnv(ORT_LOGGING_LEVEL_WARNING, "mouse-gpu-test", &raw_environment));
    const OrtHandle<OrtEnv> environment{raw_environment, api.ReleaseEnv};
    OrtSessionOptions* raw_options = nullptr;
    check_status(api, api.CreateSessionOptions(&raw_options));
    const OrtHandle<OrtSessionOptions> options{raw_options, api.ReleaseSessionOptions};
    check_status(api,
                 api.AddSessionConfigEntry(options.get(), "session.disable_cpu_ep_fallback", "1"));
    if (use_tensorrt) {
        OrtTensorRTProviderOptions provider{};
        provider.trt_max_partition_iterations = 1000;
        provider.trt_min_subgraph_size = 1;
        provider.trt_max_workspace_size = 1024ULL * 1024ULL * 1024ULL;
        check_status(api,
                     api.SessionOptionsAppendExecutionProvider_TensorRT(options.get(), &provider));
    } else {
        const OrtCUDAProviderOptions provider{};
        check_status(api, api.SessionOptionsAppendExecutionProvider_CUDA(options.get(), &provider));
    }

    const auto unique = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const auto prefix = std::filesystem::temp_directory_path() / ("mouse-gpu-profile-" + unique);
    check_status(api, api.EnableProfiling(options.get(), prefix.c_str()));
    const std::filesystem::path model_path{MOUSE_TEST_MODEL_PATH};
    OrtSession* raw_session = nullptr;
    check_status(
        api, api.CreateSession(environment.get(), model_path.c_str(), options.get(), &raw_session));
    const OrtHandle<OrtSession> session{raw_session, api.ReleaseSession};

    std::array<float, 128> input{};
    for (std::size_t i = 0; i < input.size(); ++i) {
        input.at(i) = static_cast<float>(i) / 16.0F;
    }
    constexpr std::array<std::int64_t, 2> shape{4, 32};
    OrtMemoryInfo* raw_memory = nullptr;
    check_status(api, api.CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &raw_memory));
    const OrtHandle<OrtMemoryInfo> memory{raw_memory, api.ReleaseMemoryInfo};
    OrtValue* raw_input = nullptr;
    check_status(api, api.CreateTensorWithDataAsOrtValue(
                          memory.get(), input.data(), sizeof(input), shape.data(), shape.size(),
                          ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &raw_input));
    const OrtHandle<OrtValue> input_tensor{raw_input, api.ReleaseValue};
    const std::array<const OrtValue*, 1> input_tensors{input_tensor.get()};
    constexpr std::array<const char*, 1> input_names{"input"};
    constexpr std::array<const char*, 1> output_names{"output"};
    OrtValue* raw_output = nullptr;
    check_status(api, api.Run(session.get(), nullptr, input_names.data(), input_tensors.data(), 1,
                              output_names.data(), 1, &raw_output));
    const OrtHandle<OrtValue> output{raw_output, api.ReleaseValue};
    OrtTensorTypeAndShapeInfo* raw_shape = nullptr;
    check_status(api, api.GetTensorTypeAndShape(output.get(), &raw_shape));
    const OrtHandle<OrtTensorTypeAndShapeInfo> output_shape{raw_shape,
                                                            api.ReleaseTensorTypeAndShapeInfo};
    std::size_t count = 0;
    check_status(api, api.GetTensorShapeElementCount(output_shape.get(), &count));
    REQUIRE(count == 64);
    void* raw_data = nullptr;
    check_status(api, api.GetTensorMutableData(output.get(), &raw_data));
    const std::span<const float> actual{static_cast<const float*>(raw_data), count};
    float maximum_error = 0.0F;
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t col = 0; col < 16; ++col) {
            float expected = static_cast<float>(col) / 16.0F;
            for (std::size_t inner = 0; inner < 32; ++inner) {
                const auto weight_index = inner * 16 + col;
                const float weight =
                    static_cast<float>(static_cast<int>(weight_index % 17) - 8) / 16.0F;
                expected += input.at(row * 32 + inner) * weight;
            }
            expected = expected < 0.0F ? 0.0F : expected;
            INFO("output index " << row * 16 + col);
            REQUIRE(std::isfinite(actual[row * 16 + col]));
            const float error = std::abs(actual[row * 16 + col] - expected);
            maximum_error = std::max(maximum_error, error);
            CHECK(error <= 1e-4F);
        }
    }

    OrtAllocator* allocator = nullptr;
    check_status(api, api.GetAllocatorWithDefaultOptions(&allocator));
    char* raw_profile = nullptr;
    check_status(api, api.SessionEndProfiling(session.get(), allocator, &raw_profile));
    const auto free_profile = [allocator](char* text) { allocator->Free(allocator, text); };
    const std::unique_ptr<char, decltype(free_profile)> profile_name{raw_profile, free_profile};
    const std::filesystem::path profile_path{profile_name.get()};
    std::ifstream profile{profile_path};
    REQUIRE(profile.is_open());
    const std::string events{std::istreambuf_iterator<char>{profile},
                             std::istreambuf_iterator<char>{}};
    profile.close();
    const std::regex provider_expression{"\"provider\"\\s*:\\s*\"([^\"]+)\""};
    const std::string expected_provider =
        use_tensorrt ? "TensorrtExecutionProvider" : "CUDAExecutionProvider";
    std::size_t provider_events = 0;
    for (std::sregex_iterator it{events.begin(), events.end(), provider_expression}, end; it != end;
         ++it) {
        CHECK((*it)[1].str() == expected_provider);
        ++provider_events;
    }
    CHECK(provider_events > 0);
    std::cout << "ORT=" << runtime.version() << " expected_provider=" << expected_provider
              << " provider_events=" << provider_events
              << " PATH=empty max_absolute_error=" << maximum_error << '\n';
    CHECK(std::filesystem::remove(profile_path));
}
}  // namespace

TEST_CASE("CUDA executes with an empty PATH and CPU fallback disabled", "[gpu][runtime]") {
    run_gpu_model(false);
}

TEST_CASE("TensorRT executes with an empty PATH and CPU fallback disabled", "[gpu][runtime]") {
    run_gpu_model(true);
}

TEST_CASE("Automatic minibatch is twice the GPU multiprocessor count", "[gpu][runtime]") {
    // lc0 v0.32.1 neural/backends/cuda/network_cuda.cc: 2 x multiprocessors.
    mouse::nn::BackendConfig config;
    config.kind = mouse::nn::BackendKind::TensorRt;
    const auto batch = mouse::nn::recommended_minibatch(config);
    REQUIRE(batch);
    CHECK(*batch % 2 == 0);
    CHECK(*batch >= 2);
    CHECK(*batch <= 1024);
    std::cout << "recommended_minibatch=" << *batch << std::endl;
}
