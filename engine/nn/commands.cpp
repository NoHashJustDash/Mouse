// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/nn/commands.h"

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <limits>
#include <set>

#include "engine/util/sha256.h"

namespace mouse::nn {
namespace {
struct NumberRange {
    std::uint64_t minimum;
    std::uint64_t maximum;
};
Result<std::uint64_t> number(std::string_view text, const char* name, NumberRange range) {
    std::uint64_t value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || text.size() > 20 || parsed.ec != std::errc{} ||
        parsed.ptr != text.data() + text.size() || value < range.minimum || value > range.maximum) {
        return tl::unexpected(std::string{name} + " requires an integer in " +
                              std::to_string(range.minimum) + ".." + std::to_string(range.maximum));
    }
    return value;
}
Result<std::vector<std::size_t>> buckets(std::string_view text) {
    std::vector<std::size_t> result;
    if (text.empty() || text.size() > 5120 || text.back() == ',') {
        return tl::unexpected(std::string{"--buckets requires a nonempty increasing list"});
    }
    while (!text.empty()) {
        const auto comma = text.find(',');
        const auto value =
            number(text.substr(0, comma), "--buckets", {.minimum = 1, .maximum = 1024});
        if (!value) return tl::unexpected(value.error());
        if (!result.empty() && *value <= result.back()) {
            return tl::unexpected(std::string{"--buckets must be strictly increasing"});
        }
        result.push_back(static_cast<std::size_t>(*value));
        text = comma == std::string_view::npos ? std::string_view{} : text.substr(comma + 1);
    }
    return result;
}
template <class Integer>
Integer read_little(std::span<const std::byte> bytes) {
    Integer value = 0;
    for (std::size_t index = 0; index < sizeof(Integer); ++index) {
        value |= static_cast<Integer>(std::to_integer<unsigned int>(bytes[index])) << (index * 8);
    }
    return value;
}
template <class Integer>
void append_little(std::vector<std::byte>& bytes, Integer value) {
    for (std::size_t index = 0; index < sizeof(Integer); ++index) {
        bytes.push_back(static_cast<std::byte>((value >> (index * 8)) & 255U));
    }
}
void append_text(std::vector<std::byte>& bytes, std::string_view text) {
    for (const char value : text) bytes.push_back(static_cast<std::byte>(value));
}
bool is_hash(std::string_view hash) {
    return hash.size() == 64 && std::ranges::all_of(hash, [](unsigned char value) {
               return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
           });
}
bool has_magic(std::span<const std::byte> bytes, std::string_view text) {
    if (bytes.size() < text.size()) return false;
    for (std::size_t index = 0; index < text.size(); ++index) {
        if (bytes[index] != static_cast<std::byte>(text[index])) return false;
    }
    return true;
}
}  // namespace

Result<BackendCommand> parse_backend_command(BackendCommandKind kind,
                                             std::span<const std::string_view> arguments) {
    if (kind != BackendCommandKind::Evaluate && kind != BackendCommandKind::Benchmark &&
        kind != BackendCommandKind::Configuration)
        return tl::unexpected(std::string{"Invalid backend command kind"});
    BackendCommand command;
    std::set<std::string_view> seen;
    for (std::size_t index = 0; index < arguments.size(); ++index) {
        const auto option = arguments[index];
        if (option != "--gpu-dll-dir" && !seen.insert(option).second) {
            return tl::unexpected("Duplicate backend option: " + std::string{option});
        }
        if (index + 1 >= arguments.size()) {
            return tl::unexpected("Missing value after backend option: " + std::string{option});
        }
        const auto value = arguments[++index];
        if (value.empty()) return tl::unexpected("Empty value after " + std::string{option});
        if (option == "--weights") {
            command.backend.model_path = std::filesystem::path{value};
        } else if (option == "--backend") {
            if (value == "cpu")
                command.backend.kind = BackendKind::Cpu;
            else if (value == "cuda")
                command.backend.kind = BackendKind::Cuda;
            else if (value == "tensorrt")
                command.backend.kind = BackendKind::TensorRt;
            else
                return tl::unexpected(std::string{"--backend must be cpu, cuda or tensorrt"});
        } else if (option == "--precision") {
            if (value == "fp32")
                command.backend.precision = ComputePrecision::Float32;
            else if (value == "fp16")
                command.backend.precision = ComputePrecision::Float16;
            else
                return tl::unexpected(std::string{"--precision must be fp32 or fp16"});
        } else if (option == "--cache-dir") {
            command.backend.cache_directory = std::filesystem::path{value};
        } else if (option == "--gpu-dll-dir") {
            const std::filesystem::path path{value};
            if (std::ranges::find(command.backend.gpu_dll_directories, path) !=
                command.backend.gpu_dll_directories.end()) {
                return tl::unexpected(std::string{"Duplicate --gpu-dll-dir"});
            }
            command.backend.gpu_dll_directories.push_back(path);
        } else if (option == "--inputs" && kind == BackendCommandKind::Evaluate) {
            command.inputs_path = std::filesystem::path{value};
        } else if (option == "--buckets" && kind != BackendCommandKind::Benchmark) {
            const auto parsed = buckets(value);
            if (!parsed) return tl::unexpected(parsed.error());
            command.backend.batch_buckets = *parsed;
        } else if (option == "--device") {
            const auto parsed = number(value, "--device", {.minimum = 0, .maximum = 1023});
            if (!parsed) return tl::unexpected(parsed.error());
            command.backend.device_id = static_cast<int>(*parsed);
        } else if (option == "--cpu-threads") {
            const auto parsed = number(value, "--cpu-threads", {.minimum = 1, .maximum = 128});
            if (!parsed) return tl::unexpected(parsed.error());
            command.backend.cpu_threads = static_cast<int>(*parsed);
        } else if (option == "--workspace-mb") {
            const auto parsed = number(value, "--workspace-mb", {.minimum = 1, .maximum = 16384});
            if (!parsed) return tl::unexpected(parsed.error());
            command.backend.workspace_bytes = *parsed * 1024U * 1024U;
        } else if (option == "--nncache-mb" && kind != BackendCommandKind::Benchmark) {
            const auto parsed = number(value, "--nncache-mb", {.minimum = 0, .maximum = 16384});
            if (!parsed) return tl::unexpected(parsed.error());
            command.cache_bytes = static_cast<std::size_t>(*parsed * 1024U * 1024U);
        } else if (option == "--batches" && kind == BackendCommandKind::Benchmark) {
            const auto parsed = number(value, "--batches", {.minimum = 1, .maximum = 1000000});
            if (!parsed) return tl::unexpected(parsed.error());
            command.batches = static_cast<std::uint32_t>(*parsed);
        } else if ((option == "--start-batch-size" || option == "--max-batch-size" ||
                    option == "--batch-step") &&
                   kind == BackendCommandKind::Benchmark) {
            const auto parsed =
                number(value, "--batch-size/step",
                       {.minimum = 1, .maximum = option == "--batch-step" ? 256U : 1024U});
            if (!parsed) return tl::unexpected(parsed.error());
            if (option == "--start-batch-size")
                command.start_batch_size = static_cast<std::size_t>(*parsed);
            else if (option == "--max-batch-size")
                command.max_batch_size = static_cast<std::size_t>(*parsed);
            else
                command.batch_step = static_cast<std::size_t>(*parsed);
        } else {
            return tl::unexpected("Unknown option for backend command: " + std::string{option});
        }
    }
    for (const auto required : {"--weights", "--backend", "--precision"}) {
        if (!seen.contains(required))
            return tl::unexpected("Missing required option: " + std::string{required});
    }
    if (kind == BackendCommandKind::Evaluate && command.inputs_path.empty()) {
        return tl::unexpected(std::string{"backend-eval requires --inputs"});
    }
    if (command.start_batch_size > command.max_batch_size) {
        return tl::unexpected(std::string{"Start batch size exceeds maximum"});
    }
    return command;
}

Result<PlaneBatch> decode_plane_batch(std::span<const std::byte> bytes) {
    constexpr std::size_t kRecordBytes = std::size_t{112} * 12;
    if (bytes.size() < 16 || !has_magic(bytes, "MOUSEIN1")) {
        return tl::unexpected(std::string{"Invalid MOUSEIN1 input header"});
    }
    const auto count = read_little<std::uint32_t>(bytes.subspan(8, 4));
    const auto planes = read_little<std::uint32_t>(bytes.subspan(12, 4));
    if (count < 1 || count > 1024 || planes != 112 ||
        bytes.size() != 16 + static_cast<std::size_t>(count) * kRecordBytes) {
        return tl::unexpected(std::string{"Input requires exactly 1..1024 rows of 112 planes"});
    }
    PlaneBatch result{.inputs = std::vector<InputPlanes>(count),
                      .input_sha256 = mouse::sha256(bytes)};
    std::size_t position = 16;
    for (auto& input : result.inputs) {
        for (auto& plane : input) {
            plane.mask = read_little<std::uint64_t>(bytes.subspan(position, 8));
            plane.value =
                std::bit_cast<float>(read_little<std::uint32_t>(bytes.subspan(position + 8, 4)));
            if (!std::isfinite(plane.value)) {
                return tl::unexpected(std::string{"Input planes contain a nonfinite value"});
            }
            position += 12;
        }
    }
    return result;
}

Result<std::vector<std::byte>> encode_backend_outputs(std::span<const NNOutput> outputs,
                                                      OutputIdentity identity) {
    if (outputs.empty() || outputs.size() > 1024 || !is_hash(identity.model_sha256) ||
        !is_hash(identity.input_sha256)) {
        return tl::unexpected(std::string{"Invalid backend output count or artifact identities"});
    }
    for (const auto& output : outputs) {
        const auto valid = validate_nn_output(output);
        if (!valid) return tl::unexpected(valid.error());
        if (!output.moves_left) return tl::unexpected(std::string{"Backend output is missing MLH"});
    }
    std::vector<std::byte> bytes;
    bytes.reserve(144 + outputs.size() * 1862 * 4);
    append_text(bytes, "MOUSEOU1");
    append_little(bytes, static_cast<std::uint32_t>(outputs.size()));
    append_little(bytes, std::uint32_t{1862});
    append_text(bytes, identity.model_sha256);
    append_text(bytes, identity.input_sha256);
    for (const auto& output : outputs) {
        // Keep the optional invariant local to its use, including in static analysis.
        if (!output.moves_left.has_value())
            return tl::unexpected(std::string{"Backend output is missing MLH"});
        for (const float value : output.policy_logits)
            append_little(bytes, std::bit_cast<std::uint32_t>(value));
        for (const float value : output.wdl)
            append_little(bytes, std::bit_cast<std::uint32_t>(value));
        append_little(bytes, std::bit_cast<std::uint32_t>(*output.moves_left));
    }
    return bytes;
}
}  // namespace mouse::nn
