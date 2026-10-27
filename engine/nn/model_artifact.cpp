// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/nn/model_artifact.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

namespace mouse::nn {
namespace {
// Field numbers from the installed, pinned ONNX 1.18 schema, independently
// interpreted here; no generated parser or upstream source is vendored.
// https://github.com/onnx/onnx/blob/v1.18.0/onnx/onnx.in.proto
enum class Container : std::uint8_t {
    Model,
    Graph,
    Node,
    Attribute,
    Tensor,
    Sparse,
    Function,
    Training
};
struct Field {
    std::uint32_t number;
    std::uint32_t wire;
    std::uint64_t integer = 0;
    std::span<const std::byte> payload;
};
class WireReader {
   public:
    explicit WireReader(std::span<const std::byte> bytes) : bytes_(bytes) {}
    [[nodiscard]] std::optional<Field> next() {
        if (cursor_ == bytes_.size()) return std::nullopt;
        const auto key = varint();
        if ((key >> 3U) == 0 || (key >> 3U) > 0x1fffffffU) {
            throw std::runtime_error("Invalid ONNX protobuf field number");
        }
        Field field{.number = static_cast<std::uint32_t>(key >> 3U),
                    .wire = static_cast<std::uint32_t>(key & 7U),
                    .integer = 0,
                    .payload = {}};
        switch (field.wire) {
            case 0:
                field.integer = varint();
                break;
            case 1:
                field.payload = payload(8);
                break;
            case 2:
                field.payload = payload(varint());
                break;
            case 5:
                field.payload = payload(4);
                break;
            default:
                throw std::runtime_error("Unsupported ONNX protobuf wire type");
        }
        return field;
    }

   private:
    [[nodiscard]] std::uint64_t varint() {
        std::uint64_t value = 0;
        for (unsigned int index = 0; index < 10; ++index) {
            if (cursor_ == bytes_.size()) throw std::runtime_error("Truncated ONNX varint");
            const auto byte = std::to_integer<std::uint32_t>(bytes_[cursor_++]);
            if (index == 9 && byte > 1) throw std::runtime_error("Overflowing ONNX varint");
            value |= std::uint64_t{byte & 127U} << (index * 7U);
            if ((byte & 128U) == 0) return value;
        }
        throw std::runtime_error("Overflowing ONNX varint");
    }
    [[nodiscard]] std::span<const std::byte> payload(std::uint64_t length) {
        if (length > bytes_.size() - cursor_) {
            throw std::runtime_error("Truncated ONNX protobuf field");
        }
        const auto bounded = static_cast<std::size_t>(length);
        const auto result = bytes_.subspan(cursor_, bounded);
        cursor_ += bounded;
        return result;
    }
    std::span<const std::byte> bytes_;
    std::size_t cursor_ = 0;
};

std::optional<Container> child_container(Container parent, std::uint32_t field) {
    switch (parent) {
        case Container::Model:
            if (field == 7) return Container::Graph;
            if (field == 20) return Container::Training;
            if (field == 25) return Container::Function;
            break;
        case Container::Graph:
            if (field == 1) return Container::Node;
            if (field == 5) return Container::Tensor;
            if (field == 15) return Container::Sparse;
            break;
        case Container::Node:
            if (field == 5) return Container::Attribute;
            break;
        case Container::Attribute:
            if (field == 5 || field == 10) return Container::Tensor;
            if (field == 6 || field == 11) return Container::Graph;
            if (field == 22 || field == 23) return Container::Sparse;
            break;
        case Container::Sparse:
            if (field == 1 || field == 2) return Container::Tensor;
            break;
        case Container::Function:
            if (field == 7) return Container::Node;
            if (field == 11) return Container::Attribute;
            break;
        case Container::Training:
            if (field == 1 || field == 2) return Container::Graph;
            break;
        case Container::Tensor:
            break;
    }
    return std::nullopt;
}

void validate_container(std::span<const std::byte> bytes, Container kind, unsigned int depth,
                        std::size_t& remaining_fields, OnnxArtifactInfo& info) {
    if (depth > 32) throw std::runtime_error("ONNX artifact exceeds nesting limit 32");
    WireReader reader{bytes};
    std::size_t graphs = 0;
    while (const auto field = reader.next()) {
        if (remaining_fields == 0) throw std::runtime_error("ONNX artifact exceeds field limit");
        --remaining_fields;
        if (kind == Container::Graph && field->number == 2) {
            info.maximum_graph_name_bytes =
                std::max(info.maximum_graph_name_bytes, field->payload.size());
            const bool safe = field->wire == 2 &&
                              std::ranges::all_of(
                                  field->payload,
                                  [](std::byte value) {
                                      const auto ch = std::to_integer<unsigned char>(value);
                                      return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
                                             (ch >= '0' && ch <= '9') || ch == '.' || ch == '_' ||
                                             ch == '-';
                                  }) &&
                              (field->payload.empty() || field->payload.back() != std::byte{'.'});
            info.windows_safe_graph_names = info.windows_safe_graph_names && safe;
        }
        if (kind == Container::Tensor) {
            if (field->number == 13) {
                throw std::runtime_error("ONNX external tensor data is unsupported");
            }
            if (field->number == 14 && (field->wire != 0 || field->integer != 0)) {
                throw std::runtime_error("ONNX external or unknown tensor data location");
            }
        }
        if (kind == Container::Model && field->number == 7) ++graphs;
        if (const auto child = child_container(kind, field->number)) {
            if (field->wire != 2) throw std::runtime_error("Wrong ONNX container field wire type");
            validate_container(field->payload, *child, depth + 1, remaining_fields, info);
        }
    }
    if (kind == Container::Model && graphs != 1) {
        throw std::runtime_error("ONNX artifact requires exactly one model graph");
    }
}
}  // namespace

Result<OnnxArtifactInfo> inspect_onnx_artifact(std::span<const std::byte> bytes) {
    try {
        if (bytes.empty()) return tl::unexpected(std::string{"ONNX artifact is empty"});
        if (bytes.size() > std::size_t{512} * 1024U * 1024U) {
            return tl::unexpected(std::string{"ONNX artifact exceeds 512 MiB limit"});
        }
        std::size_t remaining_fields = 1000000;
        OnnxArtifactInfo info;
        validate_container(bytes, Container::Model, 0, remaining_fields, info);
        return info;
    } catch (const std::exception& error) {
        return tl::unexpected(std::string{"Invalid ONNX artifact: "} + error.what());
    }
}
Result<void> validate_self_contained_onnx(std::span<const std::byte> bytes) {
    const auto inspected = inspect_onnx_artifact(bytes);
    if (!inspected) return tl::unexpected(inspected.error());
    return {};
}
}  // namespace mouse::nn
