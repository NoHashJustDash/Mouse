// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/nn/model_metadata.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <limits>
#include <memory>
#include <set>
#include <span>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <vector>

namespace mouse::nn {
namespace {
constexpr std::array<const char*, 11> kKeys{
    "mouse.schema_version", "mouse.converter_version",      "mouse.source_sha256",
    "mouse.input_format",   "mouse.network_format",         "mouse.raw_network_format",
    "mouse.policy_format",  "mouse.value_format",           "mouse.moves_left_format",
    "mouse.policy_mapping", "mouse.learned_parameter_count"};

void check(const OrtApi& api, OrtStatus* raw, std::string_view operation) {
    const std::unique_ptr<OrtStatus, decltype(api.ReleaseStatus)> status{raw, api.ReleaseStatus};
    if (status) {
        throw std::runtime_error(std::string{operation} + ": " + api.GetErrorMessage(status.get()));
    }
}
struct AllocatedStringDeleter {
    OrtAllocator* allocator;
    void operator()(char* value) const noexcept {
        if (value != nullptr) allocator->Free(allocator, value);
    }
};
std::string owned_string(const OrtApi& api, char* raw, OrtAllocator& allocator, OrtStatus* status,
                         std::string_view operation) {
    const std::unique_ptr<char, AllocatedStringDeleter> value{raw, {&allocator}};
    check(api, status, operation);
    return value ? std::string{value.get()} : std::string{};
}
template <class Integer>
Result<Integer> positive_decimal(std::string_view value, const char* key) {
    Integer parsed{};
    const auto converted = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (value.empty() || value.size() > 20 || value.front() == '0' || converted.ec != std::errc{} ||
        converted.ptr != value.data() + value.size() || parsed == 0) {
        return tl::unexpected(std::string{key} + " must be a positive canonical decimal integer");
    }
    return parsed;
}

void validate_tensor(const OrtApi& api, const OrtSession& session, OrtAllocator& allocator,
                     std::size_t index, bool input, std::set<std::string>& seen,
                     std::optional<std::size_t> expected_batch) {
    char* raw_name = nullptr;
    OrtStatus* status = input ? api.SessionGetInputName(&session, index, &allocator, &raw_name)
                              : api.SessionGetOutputName(&session, index, &allocator, &raw_name);
    const auto name = owned_string(api, raw_name, allocator, status, "Read ONNX tensor name");
    if (!seen.insert(name).second) throw std::runtime_error("Duplicate ONNX tensor name: " + name);
    const std::array<std::int64_t, 3> plane_shape{112, 8, 8};
    const std::array<std::int64_t, 1> policy_shape{1858};
    const std::array<std::int64_t, 1> value_shape{3};
    const std::array<std::int64_t, 1> mlh_shape{1};
    std::span<const std::int64_t> expected;
    if (input && name == "planes")
        expected = plane_shape;
    else if (!input && name == "policy")
        expected = policy_shape;
    else if (!input && name == "wdl")
        expected = value_shape;
    else if (!input && name == "mlh")
        expected = mlh_shape;
    else
        throw std::runtime_error("Unsupported ONNX tensor name: " + name);

    OrtTypeInfo* raw_type = nullptr;
    status = input ? api.SessionGetInputTypeInfo(&session, index, &raw_type)
                   : api.SessionGetOutputTypeInfo(&session, index, &raw_type);
    const std::unique_ptr<OrtTypeInfo, decltype(api.ReleaseTypeInfo)> type{raw_type,
                                                                           api.ReleaseTypeInfo};
    check(api, status, "Read ONNX tensor type");
    const OrtTensorTypeAndShapeInfo* shape = nullptr;
    check(api, api.CastTypeInfoToTensorInfo(type.get(), &shape), "Read ONNX tensor shape");
    if (shape == nullptr) throw std::runtime_error("ONNX value is not a tensor: " + name);
    ONNXTensorElementDataType element{};
    check(api, api.GetTensorElementType(shape, &element), "Read ONNX tensor element type");
    if (element != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error("ONNX tensor must be float32: " + name);
    }
    std::size_t rank = 0;
    check(api, api.GetDimensionsCount(shape, &rank), "Read ONNX tensor rank");
    if (rank != expected.size() + 1) throw std::runtime_error("Wrong ONNX tensor rank: " + name);
    std::vector<std::int64_t> dimensions(rank);
    check(api, api.GetDimensions(shape, dimensions.data(), rank), "Read ONNX tensor dimensions");
    std::vector<const char*> symbols(rank);
    check(api, api.GetSymbolicDimensions(shape, symbols.data(), rank),
          "Read ONNX dynamic dimensions");
    if (expected_batch) {
        if (dimensions.front() != static_cast<std::int64_t>(*expected_batch)) {
            throw std::runtime_error("Wrong specialized ONNX batch dimension: " + name);
        }
    } else if (dimensions.front() != -1 || symbols.front() == nullptr ||
               std::string_view{symbols.front()} != "batch") {
        throw std::runtime_error("ONNX tensor needs dynamic batch dimension named batch: " + name);
    }
    if (!std::equal(expected.begin(), expected.end(), dimensions.begin() + 1)) {
        throw std::runtime_error("Wrong ONNX tensor shape: " + name);
    }
}
}  // namespace

Result<ModelMetadata> parse_model_metadata(const MetadataFields& fields) {
    for (const auto key : kKeys) {
        if (!fields.contains(key) || fields.find(key)->second.empty()) {
            return tl::unexpected("Missing ONNX metadata: " + std::string{key});
        }
    }
    for (const auto& [key, expected] :
         std::array{std::pair{"mouse.schema_version", "1"}, std::pair{"mouse.input_format", "1"},
                    std::pair{"mouse.policy_format", "3"}, std::pair{"mouse.value_format", "2"},
                    std::pair{"mouse.moves_left_format", "1"},
                    std::pair{"mouse.policy_mapping", "lc0-1858-v1"}}) {
        if (fields.at(key) != expected) {
            return tl::unexpected("Unsupported ONNX metadata " + std::string{key} + ": " +
                                  fields.at(key));
        }
    }
    const auto precision_field = fields.find("mouse.precision");
    const std::string precision =
        precision_field == fields.end() ? "fp32" : precision_field->second;
    if (precision != "fp32" && precision != "fp16") {
        return tl::unexpected(std::string{"Unsupported ONNX metadata mouse.precision"});
    }
    const auto& version = fields.at("mouse.converter_version");
    if (version.size() > 64 || !std::ranges::all_of(version, [](unsigned char character) {
            return (character >= 'a' && character <= 'z') ||
                   (character >= 'A' && character <= 'Z') ||
                   (character >= '0' && character <= '9') || character == '.' || character == '+' ||
                   character == '-';
        })) {
        return tl::unexpected(std::string{"Invalid ONNX metadata mouse.converter_version"});
    }
    const auto& hash = fields.at("mouse.source_sha256");
    if (hash.size() != 64 || !std::ranges::all_of(hash, [](unsigned char character) {
            return (character >= '0' && character <= '9') || (character >= 'a' && character <= 'f');
        })) {
        return tl::unexpected(std::string{"Invalid ONNX metadata mouse.source_sha256"});
    }
    const auto network =
        positive_decimal<std::uint32_t>(fields.at("mouse.network_format"), "mouse.network_format");
    if (!network) return tl::unexpected(network.error());
    const auto raw_network = positive_decimal<std::uint32_t>(fields.at("mouse.raw_network_format"),
                                                             "mouse.raw_network_format");
    if (!raw_network) return tl::unexpected(raw_network.error());
    const auto count = positive_decimal<std::uint64_t>(fields.at("mouse.learned_parameter_count"),
                                                       "mouse.learned_parameter_count");
    if (!count) return tl::unexpected(count.error());
    return ModelMetadata{.converter_version = version,
                         .source_sha256 = hash,
                         .precision = precision,
                         .input_format = 1,
                         .network_format = *network,
                         .raw_network_format = *raw_network,
                         .parameter_count = *count};
}

Result<ModelMetadata> read_model_metadata(const OrtApi& api, const OrtSession& session,
                                          std::optional<std::size_t> expected_batch) {
    if (expected_batch && (*expected_batch == 0 || *expected_batch > 1024)) {
        return tl::unexpected(std::string{"Specialized ONNX batch must be in 1..1024"});
    }
    try {
        OrtAllocator* allocator = nullptr;
        check(api, api.GetAllocatorWithDefaultOptions(&allocator), "Get ORT metadata allocator");
        OrtModelMetadata* raw_metadata = nullptr;
        OrtStatus* status = api.SessionGetModelMetadata(&session, &raw_metadata);
        const std::unique_ptr<OrtModelMetadata, decltype(api.ReleaseModelMetadata)> metadata{
            raw_metadata, api.ReleaseModelMetadata};
        check(api, status, "Read ONNX metadata");
        MetadataFields fields;
        for (const auto key : kKeys) {
            char* raw = nullptr;
            status = api.ModelMetadataLookupCustomMetadataMap(metadata.get(), allocator, key, &raw);
            auto value = owned_string(api, raw, *allocator, status, "Read ONNX custom metadata");
            if (!value.empty()) fields.emplace(key, std::move(value));
        }
        char* raw_precision = nullptr;
        status = api.ModelMetadataLookupCustomMetadataMap(metadata.get(), allocator,
                                                          "mouse.precision", &raw_precision);
        const bool precision_present = raw_precision != nullptr;
        auto precision_value =
            owned_string(api, raw_precision, *allocator, status, "Read ONNX precision metadata");
        if (precision_present) fields.emplace("mouse.precision", std::move(precision_value));
        auto parsed = parse_model_metadata(fields);
        if (!parsed) return parsed;
        std::size_t inputs = 0;
        std::size_t outputs = 0;
        check(api, api.SessionGetInputCount(&session, &inputs), "Read ONNX input count");
        check(api, api.SessionGetOutputCount(&session, &outputs), "Read ONNX output count");
        if (inputs != 1 || outputs != 3) {
            return tl::unexpected(std::string{"ONNX model requires one input and three outputs"});
        }
        std::set<std::string> names;
        validate_tensor(api, session, *allocator, 0, true, names, expected_batch);
        for (std::size_t index = 0; index < outputs; ++index) {
            validate_tensor(api, session, *allocator, index, false, names, expected_batch);
        }
        return parsed;
    } catch (const std::exception& error) {
        return tl::unexpected(std::string{"ONNX model validation failed: "} + error.what());
    }
}

Result<ModelMetadata> inspect_onnx_model(const RuntimeLibrary& runtime,
                                         const std::filesystem::path& path) {
    try {
        std::error_code error;
        if (!std::filesystem::is_regular_file(path, error)) {
            return tl::unexpected(
                std::string{"ONNX model does not exist or is not a regular file: "} +
                path.string());
        }
        const auto& api = runtime.api();
        OrtEnv* raw_environment = nullptr;
        OrtStatus* status =
            api.CreateEnv(ORT_LOGGING_LEVEL_ERROR, "mouse-model-inspection", &raw_environment);
        const std::unique_ptr<OrtEnv, decltype(api.ReleaseEnv)> environment{raw_environment,
                                                                            api.ReleaseEnv};
        check(api, status, "Create ONNX inspection environment");
        OrtSessionOptions* raw_options = nullptr;
        status = api.CreateSessionOptions(&raw_options);
        const std::unique_ptr<OrtSessionOptions, decltype(api.ReleaseSessionOptions)> options{
            raw_options, api.ReleaseSessionOptions};
        check(api, status, "Create ONNX inspection options");
        check(api, api.SetIntraOpNumThreads(options.get(), 1), "Set inspection worker count");
        check(api, api.SetInterOpNumThreads(options.get(), 1), "Set inspection worker count");
        check(api, api.SetSessionGraphOptimizationLevel(options.get(), ORT_DISABLE_ALL),
              "Disable optimization for ONNX metadata inspection");
        OrtSession* raw_session = nullptr;
        status = api.CreateSession(environment.get(), path.c_str(), options.get(), &raw_session);
        const std::unique_ptr<OrtSession, decltype(api.ReleaseSession)> session{raw_session,
                                                                                api.ReleaseSession};
        check(api, status, "Open ONNX model for inspection");
        return read_model_metadata(api, *session);
    } catch (const std::exception& exception) {
        return tl::unexpected(std::string{"Cannot inspect ONNX model: "} + exception.what());
    }
}
}  // namespace mouse::nn
