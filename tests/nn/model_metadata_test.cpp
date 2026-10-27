// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/nn/model_metadata.h"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <string>

#include "engine/util/runtime_library.h"

namespace {
mouse::nn::MetadataFields fields() {
    return {{"mouse.schema_version", "1"},
            {"mouse.converter_version", "0.2.0-dev"},
            {"mouse.source_sha256", std::string(64, 'a')},
            {"mouse.input_format", "1"},
            {"mouse.network_format", "6"},
            {"mouse.raw_network_format", "4"},
            {"mouse.policy_format", "3"},
            {"mouse.value_format", "2"},
            {"mouse.moves_left_format", "1"},
            {"mouse.policy_mapping", "lc0-1858-v1"},
            {"mouse.learned_parameter_count", "1"}};
}
std::filesystem::path fixture(std::string_view name) {
    return std::filesystem::path{MOUSE_TEST_METADATA_DIRECTORY} / name;
}
}  // namespace

TEST_CASE("Metadata selects actual input and heads independently of network body",
          "[nn][metadata]") {
    auto data = fields();
    const auto parsed = mouse::nn::parse_model_metadata(data);
    REQUIRE(parsed.has_value());
    CHECK(parsed->input_format == 1);
    CHECK(parsed->network_format == 6);
    CHECK(parsed->raw_network_format == 4);
    CHECK(parsed->parameter_count == 1);
    CHECK(parsed->source_sha256 == std::string(64, 'a'));
    data["mouse.network_format"] = "123";
    data["mouse.raw_network_format"] = "123";
    CHECK(mouse::nn::parse_model_metadata(data).has_value());
}

TEST_CASE("Every required metadata field must be supplied", "[nn][metadata]") {
    const auto complete = fields();
    for (const auto& [key, unused] : complete) {
        static_cast<void>(unused);
        auto missing = complete;
        missing.erase(key);
        const auto parsed = mouse::nn::parse_model_metadata(missing);
        REQUIRE_FALSE(parsed.has_value());
        CHECK(parsed.error().find(key) != std::string::npos);
    }
}

TEST_CASE("Unknown metadata contracts fail rather than choose defaults", "[nn][metadata]") {
    for (const auto& [key, value] :
         std::array{std::pair{"mouse.schema_version", "2"}, std::pair{"mouse.input_format", "2"},
                    std::pair{"mouse.policy_format", "2"}, std::pair{"mouse.value_format", "1"},
                    std::pair{"mouse.moves_left_format", "0"},
                    std::pair{"mouse.policy_mapping", "queen-implicit"},
                    std::pair{"mouse.converter_version", "version with spaces"},
                    std::pair{"mouse.source_sha256", "not-a-hash"},
                    std::pair{"mouse.network_format", "0"}, std::pair{"mouse.network_format", "06"},
                    std::pair{"mouse.raw_network_format", "4294967296"},
                    std::pair{"mouse.learned_parameter_count", "0"},
                    std::pair{"mouse.learned_parameter_count", "18446744073709551616"}}) {
        auto data = fields();
        data[key] = value;
        const auto parsed = mouse::nn::parse_model_metadata(data);
        REQUIRE_FALSE(parsed.has_value());
        CHECK(parsed.error().find(key) != std::string::npos);
    }
    auto large = fields();
    large["mouse.learned_parameter_count"] = std::string(5000, '9');
    CHECK_FALSE(mouse::nn::parse_model_metadata(large).has_value());
    large = fields();
    large["mouse.converter_version"] = std::string(5000, 'a');
    CHECK_FALSE(mouse::nn::parse_model_metadata(large).has_value());
}

TEST_CASE("Real ONNX inspection validates metadata and tensor interface", "[nn][metadata]") {
    const auto runtime = mouse::RuntimeLibrary::load(MOUSE_TEST_RUNTIME_PATH);
    REQUIRE(runtime.has_value());
    const auto valid = mouse::nn::inspect_onnx_model(*runtime, fixture("valid.onnx"));
    REQUIRE(valid.has_value());
    CHECK(valid->parameter_count == 1);
    CHECK(valid->source_sha256.size() == 64);
    for (const auto& [name, error] :
         std::array{std::pair{"missing-metadata.onnx", "mouse.schema_version"},
                    std::pair{"wrong-policy-shape.onnx", "policy"},
                    std::pair{"wrong-input-type.onnx", "float32"},
                    std::pair{"not-onnx.bin", "ONNX"}, std::pair{"missing.onnx", "exist"}}) {
        const auto invalid = mouse::nn::inspect_onnx_model(*runtime, fixture(name));
        REQUIRE_FALSE(invalid.has_value());
        CHECK(invalid.error().find(error) != std::string::npos);
    }
}

namespace {
struct BatchConfiguration {
    std::size_t override_batch = 0;
    std::optional<std::size_t> expected_batch = std::nullopt;
};
mouse::Result<mouse::nn::ModelMetadata> read_specialized(const mouse::RuntimeLibrary& runtime,
                                                         BatchConfiguration configuration) {
    const auto& api = runtime.api();
    const auto check = [&api](OrtStatus* raw) {
        const std::unique_ptr<OrtStatus, decltype(api.ReleaseStatus)> status{raw,
                                                                             api.ReleaseStatus};
        if (status) FAIL(api.GetErrorMessage(status.get()));
    };
    OrtEnv* raw_environment = nullptr;
    auto* status = api.CreateEnv(ORT_LOGGING_LEVEL_ERROR, "metadata-test", &raw_environment);
    const std::unique_ptr<OrtEnv, decltype(api.ReleaseEnv)> environment{raw_environment,
                                                                        api.ReleaseEnv};
    check(status);
    OrtSessionOptions* raw_options = nullptr;
    status = api.CreateSessionOptions(&raw_options);
    const std::unique_ptr<OrtSessionOptions, decltype(api.ReleaseSessionOptions)> options{
        raw_options, api.ReleaseSessionOptions};
    check(status);
    check(api.SetIntraOpNumThreads(options.get(), 1));
    if (configuration.override_batch != 0) {
        check(api.AddFreeDimensionOverrideByName(
            options.get(), "batch", static_cast<std::int64_t>(configuration.override_batch)));
    }
    OrtSession* raw_session = nullptr;
    status = api.CreateSession(environment.get(), fixture("valid.onnx").c_str(), options.get(),
                               &raw_session);
    const std::unique_ptr<OrtSession, decltype(api.ReleaseSession)> session{raw_session,
                                                                            api.ReleaseSession};
    check(status);
    return mouse::nn::read_model_metadata(api, *session, configuration.expected_batch);
}
}  // namespace

TEST_CASE("Model precision is explicit and legacy FP32 remains compatible", "[nn][metadata]") {
    auto data = fields();
    REQUIRE(mouse::nn::parse_model_metadata(data).has_value());
    CHECK(mouse::nn::parse_model_metadata(data)->precision == "fp32");
    data["mouse.precision"] = "fp16";
    REQUIRE(mouse::nn::parse_model_metadata(data).has_value());
    CHECK(mouse::nn::parse_model_metadata(data)->precision == "fp16");
    data["mouse.precision"] = "fp32";
    CHECK(mouse::nn::parse_model_metadata(data).has_value());
    for (const auto value : {"", "auto", "tf32", "FP16"}) {
        data["mouse.precision"] = value;
        CHECK_FALSE(mouse::nn::parse_model_metadata(data).has_value());
    }
}

TEST_CASE("Metadata validates an explicitly specialized batch without weakening dynamic defaults",
          "[nn][metadata]") {
    const auto runtime = mouse::RuntimeLibrary::load(MOUSE_TEST_RUNTIME_PATH);
    REQUIRE(runtime.has_value());
    CHECK(read_specialized(*runtime, {}).has_value());
    CHECK(read_specialized(*runtime, {.override_batch = 1, .expected_batch = 1}).has_value());
    CHECK(read_specialized(*runtime, {.override_batch = 3, .expected_batch = 3}).has_value());
    CHECK_FALSE(read_specialized(*runtime, {.override_batch = 1}).has_value());
    CHECK_FALSE(read_specialized(*runtime, {.override_batch = 1, .expected_batch = 2}).has_value());
    CHECK_FALSE(read_specialized(*runtime, {.expected_batch = 1}).has_value());
    CHECK_FALSE(read_specialized(*runtime, {.expected_batch = 0}).has_value());
}

TEST_CASE("A real ONNX model without an MLH head fails the three-head contract", "[nn][metadata]") {
    const auto runtime = mouse::RuntimeLibrary::load(MOUSE_TEST_RUNTIME_PATH);
    REQUIRE(runtime.has_value());
    const auto missing = mouse::nn::inspect_onnx_model(*runtime, fixture("missing-mlh.onnx"));
    REQUIRE_FALSE(missing.has_value());
    CHECK(missing.error().find("three outputs") != std::string::npos);
}
