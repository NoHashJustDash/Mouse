// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/nn/backend.h"

#include <algorithm>
#include <array>
#include <barrier>
#include <bit>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "engine/nn/backend_gpu.h"
#include "engine/util/sha256.h"
#include "tests/required_optional.h"

namespace {
using mouse::nn::BackendConfig;
using mouse::nn::BackendKind;
using mouse::nn::ComputePrecision;
using mouse::nn::InputPlanes;
using mouse::nn::NNOutput;
using mouse::tests::required_optional;

std::filesystem::path fixture(const char* name) {
    return std::filesystem::path{MOUSE_TEST_BACKEND_DIRECTORY} / name;
}
BackendConfig cpu_config() {
    BackendConfig config;
    config.model_path = fixture("valid.onnx");
    config.runtime_path = MOUSE_TEST_RUNTIME_PATH;
    config.kind = BackendKind::Cpu;
    config.precision = ComputePrecision::Float32;
    config.batch_buckets = {1, 4, 8};
    return config;
}
InputPlanes input(float sum) {
    InputPlanes planes{};
    planes[0] = {.mask = 1, .value = sum};
    return planes;
}
void check_row(const NNOutput& output, float sum, float offset = 0) {
    for (std::size_t slot = 0; slot < output.policy_logits.size(); ++slot) {
        const float weight = static_cast<float>(static_cast<int>(slot % 17) - 8) / 16.0F;
        CHECK(output.policy_logits.at(slot) == sum * weight + offset);
    }
    const std::array logits{sum, 0.0F, -sum};
    const double denominator =
        std::exp(static_cast<double>(sum)) + 1 + std::exp(-static_cast<double>(sum));
    for (std::size_t head = 0; head < 3; ++head)
        CHECK(std::abs(output.wdl.at(head) -
                       std::exp(static_cast<double>(logits.at(head))) / denominator) <= 1e-6);
    REQUIRE(output.moves_left.has_value());
    CHECK(required_optional(output.moves_left) == std::max(sum + 2.0F, 0.0F));
}
NNOutput sentinel() {
    NNOutput value;
    value.policy_logits.fill(123.0F);
    value.wdl.fill(0.5F);
    value.moves_left = 17.0F;
    return value;
}
}  // namespace

TEST_CASE("CPU backend validates identity and consumes changed inputs across bucket boundaries",
          "[nn][backend]") {
    const auto config = cpu_config();
    const auto created = mouse::nn::create_backend(config);
    if (!created) FAIL(created.error());
    auto& backend = **created;
    CHECK(backend.info().provider == "CPUExecutionProvider");
    CHECK(backend.info().precision == "fp32");
    CHECK(backend.metadata().input_format == 1);
    const auto hash = mouse::sha256_file(config.model_path);
    REQUIRE(hash.has_value());
    CHECK(backend.info().onnx_sha256 == *hash);
    for (const auto& values :
         {std::vector{0.0F}, std::vector{0.25F, -0.5F, 1.0F},
          std::vector{1.0F, -0.25F, 0.5F, -4.0F, 0.0F}, std::vector{-0.5F, 0.25F, 0.0F}}) {
        std::vector<InputPlanes> inputs;
        inputs.reserve(values.size());
        for (const auto value : values) inputs.push_back(input(value));
        std::vector<NNOutput> outputs(values.size(), sentinel());
        const auto evaluated = backend.evaluate(inputs, outputs);
        if (!evaluated) FAIL(evaluated.error());
        for (std::size_t row = 0; row < values.size(); ++row)
            check_row(outputs.at(row), values.at(row));
    }
}

TEST_CASE("Backend validates explicit configuration without selecting a fallback",
          "[nn][backend]") {
    for (const auto& buckets : {std::vector<std::size_t>{}, {0}, {1, 1}, {1025}}) {
        auto config = cpu_config();
        config.batch_buckets = buckets;
        const auto result = mouse::nn::create_backend(config);
        REQUIRE_FALSE(result);
        CHECK(result.error().find("batch") != std::string::npos);
    }
    SECTION("Invalid provider enum") {
        auto config = cpu_config();
        config.kind = static_cast<BackendKind>(255);
        CHECK_FALSE(mouse::nn::create_backend(config));
    }
    SECTION("Invalid precision enum") {
        auto config = cpu_config();
        config.precision = static_cast<ComputePrecision>(255);
        CHECK_FALSE(mouse::nn::create_backend(config));
    }
    SECTION("CPU is explicitly float32") {
        auto config = cpu_config();
        config.precision = ComputePrecision::Float16;
        const auto result = mouse::nn::create_backend(config);
        REQUIRE_FALSE(result);
        CHECK(result.error().find("precision") != std::string::npos);
    }
    SECTION("Invalid worker count") {
        auto config = cpu_config();
        config.cpu_threads = 0;
        CHECK_FALSE(mouse::nn::create_backend(config));
    }
    SECTION("Invalid device ID") {
        auto config = cpu_config();
        config.device_id = -1;
        CHECK_FALSE(mouse::nn::create_backend(config));
    }
    SECTION("Unsorted unique buckets are normalized") {
        auto config = cpu_config();
        config.batch_buckets = {8, 1, 4};
        CHECK(mouse::nn::create_backend(config).has_value());
    }
}

TEST_CASE("Backend construction reports actual artifact and runtime errors", "[nn][backend]") {
    SECTION("Missing ONNX") {
        auto config = cpu_config();
        config.model_path = fixture("absent.onnx");
        CHECK_FALSE(mouse::nn::create_backend(config));
    }
    SECTION("Unsupported precision metadata") {
        auto config = cpu_config();
        config.model_path = fixture("wrong-precision.onnx");
        const auto result = mouse::nn::create_backend(config);
        REQUIRE_FALSE(result);
        CHECK(result.error().find("precision") != std::string::npos);
    }
    SECTION("Missing runtime") {
        auto config = cpu_config();
        config.runtime_path = fixture("missing-runtime.dll");
        CHECK_FALSE(mouse::nn::create_backend(config));
    }
    SECTION("Wrong runtime version") {
        auto config = cpu_config();
        config.runtime_path = MOUSE_TEST_WRONG_RUNTIME_PATH;
        const auto result = mouse::nn::create_backend(config);
        REQUIRE_FALSE(result);
        CHECK(result.error().find("version") != std::string::npos);
    }
}

TEST_CASE("Backend rejects bad evaluation arguments without publishing partial output",
          "[nn][backend]") {
    const auto created = mouse::nn::create_backend(cpu_config());
    if (!created) FAIL(created.error());
    auto& backend = **created;
    std::vector<NNOutput> outputs(9, sentinel());
    const auto original = outputs;
    CHECK_FALSE(backend.evaluate({}, {}));
    const std::array<InputPlanes, 1> one{input(0.25F)};
    CHECK_FALSE(backend.evaluate(one, {}));
    const std::vector<InputPlanes> oversized(9, input(0.0F));
    CHECK_FALSE(backend.evaluate(oversized, outputs));
    CHECK(outputs == original);
    auto invalid = input(0.25F);
    invalid[1].value = std::numeric_limits<float>::infinity();
    CHECK_FALSE(backend.evaluate(std::span{&invalid, 1}, std::span{outputs}.first(1)));
    CHECK(outputs == original);
    invalid[1].value = std::numeric_limits<float>::quiet_NaN();
    CHECK_FALSE(backend.evaluate(std::span{&invalid, 1}, std::span{outputs}.first(1)));
    CHECK(outputs == original);
}

TEST_CASE("Real nonfinite inference leaves outputs unchanged and later calls recover",
          "[nn][backend]") {
    auto config = cpu_config();
    config.model_path = fixture("nonfinite.onnx");
    const auto created = mouse::nn::create_backend(config);
    if (!created) FAIL(created.error());
    std::array<NNOutput, 2> outputs{sentinel(), sentinel()};
    const auto original = outputs;
    const std::array bad{input(0.25F), input(1.0F)};
    const auto result = (*created)->evaluate(bad, outputs);
    REQUIRE_FALSE(result);
    CHECK(result.error().find("finite") != std::string::npos);
    CHECK(outputs == original);
    const std::array good{input(0.5F), input(0.0F)};
    REQUIRE((*created)->evaluate(good, outputs).has_value());
    check_row(outputs[0], 0.5F, 2.0F);
    check_row(outputs[1], 0.0F, 1.0F);
}

TEST_CASE("Concurrent callers retain independent caller-owned outputs", "[nn][backend]") {
    const auto created = mouse::nn::create_backend(cpu_config());
    if (!created) FAIL(created.error());
    constexpr std::size_t kWorkers = 8;
    std::barrier start(static_cast<std::ptrdiff_t>(kWorkers));
    std::array<std::string, kWorkers> errors;
    std::array<std::array<NNOutput, 2>, kWorkers> outputs;
    std::array<std::jthread, kWorkers> workers;
    for (std::size_t worker = 0; worker < kWorkers; ++worker) {
        workers.at(worker) = std::jthread([&, worker] {
            start.arrive_and_wait();
            const std::array inputs{input(static_cast<float>(worker) / 4.0F), input(-0.5F)};
            for (int repeat = 0; repeat < 30; ++repeat) {
                const auto evaluated = (*created)->evaluate(inputs, outputs.at(worker));
                if (!evaluated) {
                    errors.at(worker) = evaluated.error();
                    return;
                }
            }
        });
    }
    for (auto& worker : workers) worker.join();
    for (std::size_t worker = 0; worker < kWorkers; ++worker) {
        REQUIRE(errors.at(worker).empty());
        check_row(outputs.at(worker)[0], static_cast<float>(worker) / 4.0F);
        check_row(outputs.at(worker)[1], -0.5F);
    }
}

namespace {
class TestDirectory {
   public:
    TestDirectory() {
        parent_ = std::filesystem::canonical(std::filesystem::temp_directory_path());
        const auto unique =
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        path_ = parent_ / ("mouse-backend-test-" + unique);
        if (!std::filesystem::create_directory(path_))
            throw std::runtime_error("Could not create private backend test directory");
    }
    ~TestDirectory() {
        // Only this exclusively created direct child of the canonical temp root
        // is recursively removed; never traverse a caller-provided path.
        if (path_.parent_path() == parent_ &&
            path_.filename().string().starts_with("mouse-backend-test-")) {
            std::error_code ignored;
            std::filesystem::remove_all(path_, ignored);
        }
    }
    TestDirectory(const TestDirectory&) = delete;
    TestDirectory& operator=(const TestDirectory&) = delete;
    TestDirectory(TestDirectory&&) = delete;
    TestDirectory& operator=(TestDirectory&&) = delete;
    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

   private:
    std::filesystem::path parent_;
    std::filesystem::path path_;
};
void write_text(const std::filesystem::path& path, std::string_view text) {
    std::ofstream stream(path, std::ios::binary);
    stream << text;
    stream.close();
    if (!stream) throw std::runtime_error("Cannot write private backend test payload");
}
class EmptyPath {
   public:
    EmptyPath() {
        if (const auto* original = std::getenv("PATH")) original_ = original;
#ifdef _WIN32
        if (_putenv_s("PATH", "") != 0)
#else
        if (setenv("PATH", "", 1) != 0)
#endif
            throw std::runtime_error("Cannot sanitize GPU test PATH");
    }
    ~EmptyPath() {
#ifdef _WIN32
        static_cast<void>(_putenv_s("PATH", original_ ? original_->c_str() : ""));
#else
        if (original_)
            static_cast<void>(setenv("PATH", original_->c_str(), 1));
        else
            static_cast<void>(unsetenv("PATH"));
#endif
    }
    EmptyPath(const EmptyPath&) = delete;
    EmptyPath& operator=(const EmptyPath&) = delete;
    EmptyPath(EmptyPath&&) = delete;
    EmptyPath& operator=(EmptyPath&&) = delete;

   private:
    std::optional<std::string> original_;
};
std::vector<std::filesystem::path> configured_gpu_directories() {
    const char* raw = std::getenv("MOUSE_GPU_DLL_DIRECTORIES");
    if (raw == nullptr || *raw == '\0')
        throw std::runtime_error("Set explicit MOUSE_GPU_DLL_DIRECTORIES before [gpu] tests");
    const std::string_view configured{raw};
    std::vector<std::filesystem::path> result;
    std::size_t begin = 0;
    while (begin <= configured.size()) {
        const auto end = configured.find(';', begin);
        const auto part =
            configured.substr(begin, end == std::string_view::npos ? end : end - begin);
        if (part.empty()) throw std::runtime_error("Empty explicit GPU DLL directory");
        result.emplace_back(part);
        if (end == std::string_view::npos) return result;
        begin = end + 1;
    }
    throw std::runtime_error("Invalid GPU DLL directories");
}
void run_gpu_backend(BackendKind kind) {
    const TestDirectory temporary;
    auto config = cpu_config();
    config.kind = kind;
    config.gpu_dll_directories = configured_gpu_directories();
    config.cache_directory = temporary.path();
    const EmptyPath sanitized;
    const char* sanitized_path = std::getenv("PATH");
    REQUIRE((sanitized_path == nullptr || *sanitized_path == '\0'));
    const auto created = mouse::nn::create_backend(config);
    if (!created) FAIL(created.error());
    CHECK((*created)->physical_batch_size(1) == 1);
    CHECK((*created)->physical_batch_size(3) == 4);
    CHECK((*created)->physical_batch_size(5) == 8);
    const auto& info = (*created)->info();
    CHECK(info.provider ==
          (kind == BackendKind::Cuda ? "CUDAExecutionProvider" : "TensorrtExecutionProvider"));
    REQUIRE_FALSE(info.device_uuid.empty());
    REQUIRE_FALSE(info.cuda_version.empty());
    for (const auto& values :
         {std::vector{0.25F, -0.5F, 1.0F}, std::vector{-0.5F},
          std::vector{0.0F, 0.25F, -0.25F, 0.5F, 1.0F}, std::vector{0.5F, 0.0F, -0.5F},
          std::vector{-0.75F, 0.125F, 0.375F}, std::vector{0.125F},
          std::vector{-0.25F, 0.0F, 0.25F, -0.5F, 0.75F}}) {
        std::vector<InputPlanes> inputs;
        inputs.reserve(values.size());
        for (const auto value : values) inputs.push_back(input(value));
        std::vector<NNOutput> outputs(values.size());
        const auto result = (*created)->evaluate(inputs, outputs);
        if (!result) FAIL(result.error());
        for (std::size_t row = 0; row < values.size(); ++row)
            check_row(outputs.at(row), values.at(row));
    }
    std::cout << "MOUSE_BACKEND_GPU provider=" << info.provider << " precision=" << info.precision
              << " device=" << info.device_name << " uuid=" << info.device_uuid
              << " driver=" << info.driver_version << " cuda=" << info.cuda_version
              << " trt=" << info.tensorrt_version
              << " PATH=empty batches=3,1,5,3,3,1,5 cache=" << info.cache_namespace << '\n';
    if (kind == BackendKind::TensorRt) {
        // A second real construction must validate and consume the published
        // engine/profile/timing cache without changing its identity or payload.
        const auto reused = mouse::nn::create_backend(config);
        if (!reused) FAIL(reused.error());
        CHECK((*reused)->info().cache_namespace == info.cache_namespace);
        const std::array<InputPlanes, 1> inputs{input(-0.25F)};
        std::array<NNOutput, 1> outputs{};
        REQUIRE((*reused)->evaluate(inputs, outputs).has_value());
        check_row(outputs.front(), -0.25F);
    }
}
}  // namespace

TEST_CASE("TensorRT disk cache publishes identity and verifies every payload", "[nn][backend]") {
    const TestDirectory temporary;
    std::filesystem::path cache_path;
    {
        mouse::nn::detail::TensorRtCache cache{temporary.path(), "test_identity=1\n"};
        cache_path = cache.directory();
        CHECK_THROWS(mouse::nn::detail::TensorRtCache(temporary.path(), "test_identity=1\n",
                                                      mouse::nn::detail::CacheLockMode::Try));
        CHECK_THROWS(cache.commit());
        write_text(cache.directory() / "engine" / "fixture.engine",
                   "original test-only engine payload");
        write_text(cache.directory() / "timing" / "fixture.timing",
                   "original test-only timing payload");
        write_text(cache.directory() / "context.onnx", "completed external context");
        cache.commit();
        CHECK(std::filesystem::is_regular_file(cache.directory() / "manifest.txt"));
    }
    {
        mouse::nn::detail::TensorRtCache reused{temporary.path(), "test_identity=1\n"};
        CHECK(reused.directory() == cache_path);
        CHECK_NOTHROW(reused.commit());
    }
    SECTION("Corrupt engine is never silently reused") {
        write_text(cache_path / "engine" / "fixture.engine", "corrupt");
        CHECK_THROWS(mouse::nn::detail::TensorRtCache(temporary.path(), "test_identity=1\n"));
    }
    SECTION("Incomplete namespace fails clearly") {
        {
            const mouse::nn::detail::TensorRtCache incomplete{temporary.path(),
                                                              "test_identity=2\n"};
        }
        CHECK_THROWS(mouse::nn::detail::TensorRtCache(temporary.path(), "test_identity=2\n"));
    }
    SECTION("Runtime identity changes create a distinct namespace") {
        const mouse::nn::detail::TensorRtCache other{temporary.path(), "test_identity=2\n"};
        CHECK(other.directory() != cache_path);
    }
    SECTION("Payload change after validation is rejected before publication") {
        mouse::nn::detail::TensorRtCache reused{temporary.path(), "test_identity=1\n"};
        write_text(reused.directory() / "timing" / "fixture.timing", "changed");
        CHECK_THROWS(reused.commit());
    }
}

TEST_CASE("TensorRT external context is required and bound to the engine manifest",
          "[nn][backend][ep-context-v1]") {
    const TestDirectory temporary;
    std::filesystem::path directory;
    {
        mouse::nn::detail::TensorRtCache cache{temporary.path(), "ep_context_test=1\n"};
        directory = cache.directory();
        write_text(directory / "engine" / "fixture.engine", "completed engine");
        CHECK_THROWS(cache.commit());
        write_text(directory / "context.onnx", "completed external context");
        REQUIRE_NOTHROW(cache.commit());
    }
    SECTION("Missing context is fatal even with a valid engine") {
        REQUIRE(std::filesystem::remove(directory / "context.onnx"));
    }
    SECTION("Corrupt context cannot silently rebuild or reuse an engine") {
        write_text(directory / "context.onnx", "changed external context");
    }
    CHECK_THROWS(mouse::nn::detail::TensorRtCache(temporary.path(), "ep_context_test=1\n"));
}

TEST_CASE("Native CUDA backend refreshes pinned IO across real bucket changes",
          "[nn][backend-gpu][gpu]") {
    run_gpu_backend(BackendKind::Cuda);
}
TEST_CASE("Native TensorRT graph buckets refresh changed inputs and reuse verified engines",
          "[nn][backend-gpu][gpu]") {
    run_gpu_backend(BackendKind::TensorRt);
}

TEST_CASE("Explicit GPU library selection searches only validated directories", "[nn][backend]") {
    const TestDirectory temporary;
    const auto first = temporary.path() / "first";
    const auto second = temporary.path() / "second";
    REQUIRE(std::filesystem::create_directory(first));
    REQUIRE(std::filesystem::create_directory(second));
    std::array directories{first, second};
    CHECK_THROWS(mouse::nn::detail::explicit_gpu_library(directories, "library.test"));
    write_text(first / "library.test", "test-only library selector payload");
    CHECK(mouse::nn::detail::explicit_gpu_library(directories, "library.test") ==
          std::filesystem::canonical(first / "library.test"));
    const std::array same{first, first};
    CHECK(mouse::nn::detail::explicit_gpu_library(same, "library.test") ==
          std::filesystem::canonical(first / "library.test"));
    write_text(second / "library.test", "different test-only payload");
    CHECK_THROWS(mouse::nn::detail::explicit_gpu_library(directories, "library.test"));
    CHECK_THROWS(mouse::nn::detail::explicit_gpu_library(directories, "../library.test"));
    CHECK_THROWS(mouse::nn::detail::explicit_gpu_library(directories, nullptr));
    directories.front() = std::filesystem::path{"relative-sdk"};
    CHECK_THROWS(mouse::nn::detail::explicit_gpu_library(directories, "library.test"));
}

TEST_CASE("CPU construction does not apply Windows directory registration on Unix",
          "[nn][backend]") {
    const TestDirectory temporary;
    auto config = cpu_config();
    config.gpu_dll_directories = {temporary.path()};
    // Before the platform-specific factory fix, this same call is rejected by
    // RuntimeLibrary on Unix, making every explicit Unix GPU configuration
    // impossible before provider setup. Windows registration remains supported.
    const auto created = mouse::nn::create_backend(config);
    if (!created) FAIL(created.error());
    CHECK((*created)->info().provider == "CPUExecutionProvider");
}

TEST_CASE("Backend factory errors identify the explicitly selected provider", "[nn][backend]") {
    SECTION("Actual CPU session metadata failure retains provider context") {
        auto config = cpu_config();
        config.model_path = fixture("wrong-precision.onnx");
        const auto result = mouse::nn::create_backend(config);
        REQUIRE_FALSE(result);
        CHECK(result.error().starts_with("Backend construction [CPU]: "));
        CHECK(result.error().find("precision") != std::string::npos);
    }
    SECTION("Missing runtime identifies every selection before GPU initialization") {
        const std::array selections{std::pair{BackendKind::Cpu, std::string_view{"CPU"}},
                                    std::pair{BackendKind::Cuda, std::string_view{"CUDA"}},
                                    std::pair{BackendKind::TensorRt, std::string_view{"TensorRT"}}};
        for (const auto& [kind, label] : selections) {
            auto config = cpu_config();
            config.kind = kind;
            config.runtime_path = fixture("missing-runtime.dll");
            const auto result = mouse::nn::create_backend(config);
            REQUIRE_FALSE(result);
            CHECK(
                result.error().starts_with("Backend construction [" + std::string{label} + "]: "));
        }
    }
    SECTION("Invalid selection is explicitly unknown") {
        auto config = cpu_config();
        config.kind = static_cast<BackendKind>(255);
        const auto result = mouse::nn::create_backend(config);
        REQUIRE_FALSE(result);
        CHECK(result.error().starts_with("Backend construction [unknown]: "));
    }
}

TEST_CASE("Deterministic compute has a distinct cache identity and repeatable outputs",
          "[nn][backend][deterministic]") {
    auto config = cpu_config();
    const auto normal = mouse::nn::create_backend(config);
    REQUIRE(normal.has_value());
    config.deterministic_compute = true;
    const auto deterministic = mouse::nn::create_backend(config);
    REQUIRE(deterministic.has_value());
    CHECK((*normal)->info().cache_namespace != (*deterministic)->info().cache_namespace);
    const std::array inputs{input(0.25F), input(-0.5F)};
    std::array<NNOutput, 2> first;
    std::array<NNOutput, 2> second;
    REQUIRE((*deterministic)->evaluate(inputs, first).has_value());
    REQUIRE((*deterministic)->evaluate(inputs, second).has_value());
    CHECK(first == second);
    check_row(first[0], 0.25F);
    check_row(first[1], -0.5F);
}

TEST_CASE("TensorRT cache directory keys stay short without discarding full identity",
          "[nn][backend][cache-path]") {
    const TestDirectory temporary;
    const std::string identity = "test_identity=short-key\n";
    const auto full_hash =
        mouse::sha256(std::as_bytes(std::span{identity.data(), identity.size()}));
    mouse::nn::detail::TensorRtCache cache{temporary.path(), identity};
    CHECK(cache.directory().filename().string() == full_hash.substr(0, 16));
    write_text(cache.directory() / "engine" / "fixture.engine", "test-only payload");
    write_text(cache.directory() / "context.onnx", "completed external context");
    cache.commit();
    std::ifstream stream(cache.directory() / "manifest.txt", std::ios::binary);
    const std::string manifest{std::istreambuf_iterator<char>{stream}, {}};
    CHECK(manifest.find("sha256=" + full_hash + '\n') != std::string::npos);
    CHECK(manifest.find(identity) != std::string::npos);
}
#ifdef _WIN32
TEST_CASE("Deep TensorRT paths fail before creating cache files", "[nn][backend][cache-path]") {
    const TestDirectory temporary;
    const auto prefix = temporary.path().native().size() + 1;
    REQUIRE(prefix < 156);
    const auto root = temporary.path() / std::wstring(156 - prefix, L'x');
    REQUIRE(root.native().size() == 156);
    std::string error;
    try {
        const mouse::nn::detail::TensorRtCache cache{root, "test_identity=deep\n"};
    } catch (const std::exception& failure) {
        error = failure.what();
    }
    CHECK(error.find("MAX_PATH") != std::string::npos);
    CHECK(error.find("260") != std::string::npos);
    CHECK(error.find("shorter") != std::string::npos);
    CHECK_FALSE(std::filesystem::exists(root));
}
TEST_CASE("Non-ASCII TensorRT cache paths fail before narrow ORT path conversion",
          "[nn][backend][cache-path]") {
    const TestDirectory temporary;
    const auto root = temporary.path() / L"cache-\u00e9";
    std::string error;
    try {
        const mouse::nn::detail::TensorRtCache cache{root, "test_identity=unicode\n"};
    } catch (const std::exception& failure) {
        error = failure.what();
    }
    CHECK(error.find("ASCII") != std::string::npos);
    CHECK_FALSE(std::filesystem::exists(root));
}
#endif
TEST_CASE("TensorRT shortened-key collisions cannot reuse another full identity",
          "[nn][backend][cache-path]") {
    const TestDirectory temporary;
    const std::string first = "test_identity=first\n";
    const std::string second = "test_identity=second\n";
    std::filesystem::path old_path;
    {
        mouse::nn::detail::TensorRtCache cache{temporary.path(), first};
        old_path = cache.directory();
        write_text(old_path / "engine" / "fixture.engine", "test-only payload");
        write_text(old_path / "context.onnx", "completed external context");
        cache.commit();
    }
    const auto second_hash = mouse::sha256(std::as_bytes(std::span{second.data(), second.size()}));
    // Place the other identity at the requested short key: this exercises the
    // identical constructor state a prefix collision would produce.
    const auto target = temporary.path() / second_hash.substr(0, 16);
    std::filesystem::rename(old_path, target);
    std::string error;
    try {
        const mouse::nn::detail::TensorRtCache cache{temporary.path(), second};
    } catch (const std::exception& failure) {
        error = failure.what();
    }
    CHECK(error.find("identity/payload mismatch") != std::string::npos);
    CHECK(std::filesystem::is_regular_file(target / "engine" / "fixture.engine"));
}
TEST_CASE("TensorRT cache root preflight is absolute and leaves no files",
          "[nn][backend][cache-path]") {
    CHECK_FALSE(mouse::nn::detail::checked_tensor_rt_cache_root({}));
    CHECK_FALSE(mouse::nn::detail::checked_tensor_rt_cache_root("relative/cache"));
    const TestDirectory temporary;
    const auto root = temporary.path() / "unused" / "nested";
    const auto checked = mouse::nn::detail::checked_tensor_rt_cache_root(root);
    REQUIRE(checked);
    CHECK(*checked == root);
    CHECK_FALSE(std::filesystem::exists(root));
}
#ifdef _WIN32
TEST_CASE("TensorRT projected Windows path accepts precisely the legacy limit",
          "[nn][backend][cache-path]") {
    const TestDirectory temporary;
    const auto prefix = temporary.path().native().size() + 1;
    REQUIRE(prefix < 155);
    const auto root = temporary.path() / std::wstring(155 - prefix, L'x');
    const auto checked = mouse::nn::detail::checked_tensor_rt_cache_root(root);
    REQUIRE(checked);
    CHECK_FALSE(std::filesystem::exists(root));
    mouse::nn::detail::TensorRtCache cache{root, "test_identity=boundary\n"};
    const std::string longest = "m_" + std::string(20, '9') + '_' + std::string(20, '9') +
                                "_fp16_sm" + std::string(20, '9') + ".profile";
    REQUIRE(longest.size() == 79);
    const auto longest_path = cache.directory() / "engine" / longest;
    REQUIRE(longest_path.native().size() == 259);
    write_text(longest_path, "test-only maximum length profile");
    write_text(cache.directory() / "engine" / "fixture.engine", "test-only engine");
    write_text(cache.directory() / "context.onnx", "completed external context");
    CHECK_NOTHROW(cache.commit());
    CHECK(std::filesystem::is_regular_file(longest_path));
}
TEST_CASE("TensorRT extended paths cannot bypass the provider path limit",
          "[nn][backend][cache-path]") {
    const TestDirectory temporary;
    for (const auto* prefix : {L"\\\\?\\", L"\\\\.\\"}) {
        const auto root = std::filesystem::path{std::wstring{prefix} + temporary.path().native()};
        const auto checked = mouse::nn::detail::checked_tensor_rt_cache_root(root);
        REQUIRE_FALSE(checked);
        CHECK(checked.error().find("extended/device") != std::string::npos);
    }
}
TEST_CASE("TensorRT deep cache errors precede artifact or runtime access",
          "[nn][backend][cache-path]") {
    const TestDirectory temporary;
    auto config = cpu_config();
    config.kind = BackendKind::TensorRt;
    config.model_path = temporary.path() / "missing-model.onnx";
    config.runtime_path = temporary.path() / "missing-runtime.dll";
    const auto prefix = temporary.path().native().size() + 1;
    REQUIRE(prefix < 156);
    config.cache_directory = temporary.path() / std::wstring(156 - prefix, L'x');
    const auto created = mouse::nn::create_backend(config);
    REQUIRE_FALSE(created);
    CHECK(created.error().find("MAX_PATH") != std::string::npos);
    CHECK_FALSE(std::filesystem::exists(config.cache_directory));
}
#endif
namespace {
std::string named_graph_model(std::string_view name) {
    const auto append_varint = [](std::string& output, std::size_t value) {
        while (value >= 128) {
            output.push_back(static_cast<char>((value & 127U) | 128U));
            value >>= 7U;
        }
        output.push_back(static_cast<char>(value));
    };
    std::string graph{"\x12", 1};
    append_varint(graph, name.size());
    graph.append(name);
    std::string model(1, ':');
    append_varint(model, graph.size());
    model += graph;
    return model;
}
}  // namespace
#ifdef _WIN32
TEST_CASE("TensorRT graph-name suffix projection fails before runtime loading",
          "[nn][backend][cache-path]") {
    const TestDirectory temporary;
    auto config = cpu_config();
    config.kind = BackendKind::TensorRt;
    config.model_path = temporary.path() / "named.onnx";
    config.runtime_path = temporary.path() / "missing-runtime.dll";
    config.cache_directory = temporary.path() / "cache";
    // ORT's first numeric-hash substring match can retain graph-name content.
    write_text(config.model_path, named_graph_model(std::string(160, '9')));
    const auto result = mouse::nn::create_backend(config);
    REQUIRE_FALSE(result);
    CHECK(result.error().find("MAX_PATH") != std::string::npos);
    CHECK_FALSE(std::filesystem::exists(config.cache_directory));
}
TEST_CASE("Unsafe TensorRT graph-name suffixes fail before runtime loading",
          "[nn][backend][cache-path]") {
    const TestDirectory temporary;
    auto config = cpu_config();
    config.kind = BackendKind::TensorRt;
    config.model_path = temporary.path() / "named.onnx";
    config.runtime_path = temporary.path() / "missing-runtime.dll";
    config.cache_directory = temporary.path() / "cache";
    for (const auto name :
         {"123/path", "123\\path", "123:path", "123*path", "123?path", "123\"path", "123<path",
          "123>path", "123|path", "123\npath", "123\xc3\xa9", "123path.", "123path "}) {
        INFO(name);
        write_text(config.model_path, named_graph_model(name));
        const auto result = mouse::nn::create_backend(config);
        REQUIRE_FALSE(result);
        CHECK(result.error().find("graph name") != std::string::npos);
        CHECK_FALSE(std::filesystem::exists(config.cache_directory));
    }
}
#endif
#ifdef _WIN32
TEST_CASE("TensorRT graph-name allowance retains a duplicate numeric suffix safely",
          "[nn][backend][cache-path]") {
    const TestDirectory temporary;
    const auto prefix = temporary.path().native().size() + 1;
    REQUIRE(prefix < 133);
    const auto boundary = temporary.path() / std::wstring(133 - prefix, L'x');
    const auto too_deep = temporary.path() / std::wstring(134 - prefix, L'x');
    CHECK(mouse::nn::detail::checked_tensor_rt_cache_root(boundary, 1));
    const auto failed = mouse::nn::detail::checked_tensor_rt_cache_root(too_deep, 1);
    REQUIRE_FALSE(failed);
    CHECK(failed.error().find("MAX_PATH") != std::string::npos);
    CHECK_FALSE(mouse::nn::detail::checked_tensor_rt_cache_root(
        boundary, std::numeric_limits<std::size_t>::max()));
    CHECK_FALSE(std::filesystem::exists(boundary));
    CHECK_FALSE(std::filesystem::exists(too_deep));
}
#endif
TEST_CASE("Non-TensorRT model preflight leaves graph-name acceptance unchanged",
          "[nn][backend][cache-path]") {
    const TestDirectory temporary;
    for (const auto kind : {BackendKind::Cpu, BackendKind::Cuda}) {
        auto config = cpu_config();
        config.kind = kind;
        config.model_path = temporary.path() / "named.onnx";
        config.runtime_path = temporary.path() / "missing-runtime.dll";
        write_text(config.model_path, named_graph_model("123/unsafe-name"));
        const auto result = mouse::nn::create_backend(config);
        REQUIRE_FALSE(result);
        CHECK(result.error().find("graph name") == std::string::npos);
        CHECK(result.error().find("MAX_PATH") == std::string::npos);
    }
}

namespace {
void check_output_bits(const NNOutput& actual, const NNOutput& expected) {
    for (std::size_t i = 0; i < actual.policy_logits.size(); ++i)
        CHECK(std::bit_cast<std::uint32_t>(actual.policy_logits.at(i)) ==
              std::bit_cast<std::uint32_t>(expected.policy_logits.at(i)));
    for (std::size_t i = 0; i < actual.wdl.size(); ++i)
        CHECK(std::bit_cast<std::uint32_t>(actual.wdl.at(i)) ==
              std::bit_cast<std::uint32_t>(expected.wdl.at(i)));
    REQUIRE(actual.moves_left.has_value() == expected.moves_left.has_value());
    if (actual.moves_left)
        CHECK(std::bit_cast<std::uint32_t>(*actual.moves_left) ==
              std::bit_cast<std::uint32_t>(required_optional(expected.moves_left)));
}
std::vector<std::byte> output_bytes(std::span<const NNOutput> outputs) {
    const auto bytes = std::as_bytes(outputs);
    return {bytes.begin(), bytes.end()};
}
void reset_failure_sentinels(std::span<NNOutput> outputs) {
    for (std::size_t row = 0; row < outputs.size(); ++row) {
        outputs[row] = sentinel();
        outputs[row].policy_logits.fill(123.0F + static_cast<float>(row));
        outputs[row].moves_left = 17.0F + static_cast<float>(row);
    }
}
}  // namespace

TEST_CASE("Borrowed output validation exactly matches owning validation for all heads",
          "[nn][backend][output]") {
    auto verify = [](const NNOutput& value, bool valid) {
        const auto before = output_bytes(std::span{&value, 1});
        const auto owning = mouse::nn::validate_nn_output(value);
        const auto borrowed =
            mouse::nn::validate_nn_output(std::span<const float, 1858>{value.policy_logits},
                                          std::span<const float, 3>{value.wdl}, value.moves_left);
        CHECK(owning.has_value() == valid);
        CHECK(borrowed.has_value() == valid);
        if (!owning && !borrowed) {
            CHECK(owning.error() == borrowed.error());
            CHECK(borrowed.error() ==
                  "NN output requires finite policy, WDL and MLH values; "
                  "WDL must be in [0,1] with positive sum and MLH nonnegative");
        }
        CHECK(output_bytes(std::span{&value, 1}) == before);
    };
    NNOutput good = sentinel();
    verify(good, true);  // Positive unnormalized WDL remains accepted.
    good.moves_left.reset();
    verify(good, true);
    good.moves_left = -0.0F;
    good.policy_logits.front() = std::numeric_limits<float>::max();
    good.policy_logits.back() = -std::numeric_limits<float>::max();
    verify(good, true);
    for (const auto bad :
         {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
          -std::numeric_limits<float>::infinity()}) {
        for (const auto slot : {std::size_t{0}, std::size_t{929}, std::size_t{1857}}) {
            auto value = good;
            value.policy_logits.at(slot) = bad;
            verify(value, false);
        }
        for (std::size_t slot = 0; slot < good.wdl.size(); ++slot) {
            auto value = good;
            value.wdl.at(slot) = bad;
            verify(value, false);
        }
        auto value = good;
        value.moves_left = bad;
        verify(value, false);
    }
    for (const auto bad : {-0.01F, 1.01F}) {
        for (std::size_t slot = 0; slot < good.wdl.size(); ++slot) {
            auto value = good;
            value.wdl.at(slot) = bad;
            verify(value, false);
        }
    }
    good.wdl = {0.0F, -0.0F, 0.0F};
    verify(good, false);
    good = sentinel();
    good.moves_left = -0.01F;
    verify(good, false);
    good.moves_left.reset();
    good.wdl = {0.0F, std::numeric_limits<float>::denorm_min(), 0.0F};
    verify(good, true);
}

TEST_CASE("Backend publishes bit-exact rows across repeated and alternating actual batches",
          "[nn][backend]") {
    const auto created = mouse::nn::create_backend(cpu_config());
    if (!created) FAIL(created.error());
    constexpr std::array values{0.0F, 0.25F, -0.5F, 1.0F, -4.0F, 0.5F, -0.25F, 2.0F};
    std::array<InputPlanes, 8> inputs{};
    std::array<NNOutput, 8> reference{};
    for (std::size_t row = 0; row < values.size(); ++row) inputs.at(row) = input(values.at(row));
    REQUIRE((*created)->evaluate(inputs, reference));
    for (std::size_t pass = 0; pass < 3; ++pass) {
        for (const auto batch : {1U, 3U, 5U, 3U, 8U, 1U}) {
            std::array<NNOutput, 8> outputs{};
            for (std::size_t row = 0; row < batch; ++row) {
                inputs.at(row) = input(values.at((row + pass) % values.size()));
                if (row % 2 == 0)
                    outputs.at(row) = sentinel();
                else
                    REQUIRE_FALSE(outputs.at(row).moves_left);
            }
            REQUIRE((*created)->evaluate(std::span{inputs}.first(batch),
                                         std::span{outputs}.first(batch)));
            for (std::size_t row = 0; row < batch; ++row)
                check_output_bits(outputs.at(row), reference.at((row + pass) % values.size()));
        }
    }
}

TEST_CASE("Last-row failure preserves every caller byte and recovers across actual batches",
          "[nn][backend]") {
    auto config = cpu_config();
    config.model_path = fixture("nonfinite.onnx");
    const auto created = mouse::nn::create_backend(config);
    if (!created) FAIL(created.error());
    std::array<NNOutput, 5> outputs{};
    std::array<InputPlanes, 5> inputs{};
    for (const auto batch : {2U, 5U, 1U, 3U}) {
        for (auto& value : inputs) value = input(0.25F);
        reset_failure_sentinels(outputs);
        outputs.back().moves_left.reset();
        const auto before = output_bytes(outputs);
        inputs.at(batch - 1) = input(1.0F);
        const auto failed =
            (*created)->evaluate(std::span{inputs}.first(batch), std::span{outputs}.first(batch));
        REQUIRE_FALSE(failed);
        CHECK(failed.error().find("finite") != std::string::npos);
        CHECK(output_bytes(outputs) == before);
        inputs.at(batch - 1) = input(0.5F);
        REQUIRE(
            (*created)->evaluate(std::span{inputs}.first(batch), std::span{outputs}.first(batch)));
        for (std::size_t row = 0; row < batch; ++row) {
            const auto sum = row + 1 == batch ? 0.5F : 0.25F;
            check_row(outputs.at(row), sum, 1.0F / (1.0F - sum));
        }
    }
}

TEST_CASE("Backend constructor rejects invalid warmup output before exposing a backend",
          "[nn][backend]") {
    auto config = cpu_config();
    config.model_path = fixture("warmup-nonfinite.onnx");
    const auto created = mouse::nn::create_backend(config);
    REQUIRE_FALSE(created);
    CHECK(created.error().find("Backend construction [CPU]") != std::string::npos);
    CHECK(created.error().find("finite") != std::string::npos);
}

TEST_CASE("CUDA rejects invalid physical padded tail without publishing valid logical rows",
          "[gpu][backend-gpu][backend-padding]") {
    auto config = cpu_config();
    config.model_path = fixture("padding-nonfinite.onnx");
    const std::array logical{input(0.25F), input(0.5F), input(0.75F)};
    std::array<NNOutput, 4> outputs{};
    {
        // CPU executes actual N=3; this is a control, not a simulated padded execution.
        const auto cpu = mouse::nn::create_backend(config);
        if (!cpu) FAIL(cpu.error());
        REQUIRE((*cpu)->evaluate(logical, std::span{outputs}.first(3)));
        for (std::size_t row = 0; row < logical.size(); ++row)
            check_row(outputs.at(row), logical.at(row)[0].value, 1.0F);
    }
    config.kind = BackendKind::Cuda;
    config.precision = ComputePrecision::Float32;
    config.gpu_dll_directories = configured_gpu_directories();
    const EmptyPath sanitized;
    const auto cuda = mouse::nn::create_backend(config);
    if (!cuda) FAIL(cuda.error());  // All-zero bucket warmups must succeed.
    // CPU control outputs equal the values a faulty early CUDA publication would
    // write. Replace every row before taking the failure snapshot.
    reset_failure_sentinels(outputs);
    const auto before = output_bytes(outputs);
    const auto failed = (*cuda)->evaluate(logical, std::span{outputs}.first(3));
    REQUIRE_FALSE(failed);  // Physical N=4, only the padded fourth row is invalid.
    CHECK(failed.error().find("finite") != std::string::npos);
    CHECK(output_bytes(outputs) == before);
    const std::array four{input(0.25F), input(0.5F), input(0.75F), input(1.0F)};
    REQUIRE((*cuda)->evaluate(four, outputs));
    for (std::size_t row = 0; row < four.size(); ++row)
        check_row(outputs.at(row), four.at(row)[0].value, 1.0F);
    REQUIRE((*cuda)->evaluate(std::span{four}.first(1), std::span{outputs}.first(1)));
    reset_failure_sentinels(outputs);
    const auto before_repeated_failure = output_bytes(outputs);
    CHECK_FALSE((*cuda)->evaluate(logical, std::span{outputs}.first(3)));
    CHECK(output_bytes(outputs) == before_repeated_failure);
    std::cout << "MOUSE_BACKEND_PADDING provider=CUDAExecutionProvider precision=fp32 PATH=empty "
                 "logical=3 physical=4 invalid_tail_rejected=true recovery=4,1\n";
}

TEST_CASE("Failure snapshots expose premature publication of each valid logical row",
          "[nn][backend][output-atomicity-sentinel]") {
    auto config = cpu_config();
    config.model_path = fixture("padding-nonfinite.onnx");
    const auto created = mouse::nn::create_backend(config);
    if (!created) FAIL(created.error());
    const std::array inputs{input(0.25F), input(0.5F), input(0.75F)};
    std::array<NNOutput, 3> reference{};
    REQUIRE((*created)->evaluate(inputs, reference));
    auto outputs = reference;
    reset_failure_sentinels(outputs);
    const auto before = output_bytes(outputs);
    for (std::size_t row = 0; row < outputs.size(); ++row) {
        CHECK(outputs.at(row).policy_logits.front() != reference.at(row).policy_logits.front());
        CHECK(outputs.at(row).wdl != reference.at(row).wdl);
        CHECK(outputs.at(row).moves_left != reference.at(row).moves_left);
        auto prematurely_published = outputs;
        prematurely_published.at(row) = reference.at(row);
        const bool changed = output_bytes(prematurely_published) != before;
        CHECK(changed);
    }
}

TEST_CASE("TensorRT cache waiters serialize initialization and validate the owner's result",
          "[nn][backend][cache-wait]") {
    const TestDirectory temporary;
    std::mutex mutex;
    std::condition_variable changed;
    bool entered = false, finished = false, reused = false;
    std::string error;
    // Release the owner before joining on every exceptional exit as well.
    std::jthread waiter;
    auto owner =
        std::make_unique<mouse::nn::detail::TensorRtCache>(temporary.path(), "cache_wait_test=1\n");
    const auto directory = owner->directory();
    bool abandon = false, corrupt = false;
    SECTION("A cold writer publishes before a waiting reader validates") {}
    SECTION("An already committed cache remains exclusive through warm-up") {
        write_text(directory / "engine" / "fixture.engine", "completed engine");
        write_text(directory / "context.onnx", "completed context");
        owner->commit();
    }
    SECTION("An abandoned initialization is still rejected") { abandon = true; }
    SECTION("Corruption while waiting cannot be accepted") { corrupt = true; }
    waiter = std::jthread([&] {
        {
            const std::lock_guard lock(mutex);
            entered = true;
            changed.notify_all();
        }
        try {
            mouse::nn::detail::TensorRtCache reader{temporary.path(), "cache_wait_test=1\n"};
            reused = reader.reusable();
            reader.commit();
        } catch (const std::exception& failure) {
            error = failure.what();
        }
        const std::lock_guard lock(mutex);
        finished = true;
        changed.notify_all();
    });
    {
        std::unique_lock lock(mutex);
        changed.wait(lock, [&] { return entered; });
        CHECK_FALSE(
            changed.wait_for(lock, std::chrono::milliseconds{100}, [&] { return finished; }));
    }
    if (!abandon) {
        write_text(directory / "engine" / "fixture.engine", "completed engine");
        write_text(directory / "context.onnx", "completed context");
        owner->commit();
        if (corrupt) write_text(directory / "engine" / "fixture.engine", "corrupt engine");
    }
    owner.reset();
    waiter.join();
    INFO(error);
    if (abandon) {
        CHECK(error.find("Incomplete TensorRT cache") != std::string::npos);
    } else if (corrupt) {
        CHECK(error.find("identity/payload mismatch") != std::string::npos);
    } else {
        CHECK(error.empty());
        CHECK(reused);
    }
}
