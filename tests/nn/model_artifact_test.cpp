// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/nn/model_artifact.h"

#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

namespace {
using Bytes = std::vector<std::byte>;
void varint(Bytes& output, std::uint64_t value) {
    while (value >= 128) {
        output.push_back(static_cast<std::byte>((value & 127U) | 128U));
        value >>= 7U;
    }
    output.push_back(static_cast<std::byte>(value));
}
Bytes field(std::uint32_t number, const Bytes& contents) {
    Bytes output;
    varint(output, (std::uint64_t{number} << 3U) | 2U);
    varint(output, contents.size());
    output.insert(output.end(), contents.begin(), contents.end());
    return output;
}
struct ScalarField {
    std::uint32_t number;
    std::uint64_t value;
};
Bytes scalar(ScalarField item) {
    Bytes output;
    varint(output, std::uint64_t{item.number} << 3U);
    varint(output, item.value);
    return output;
}
Bytes graph(const Bytes& tensor) { return field(5, tensor); }
Bytes model(const Bytes& body) { return field(7, body); }
Bytes attribute_graph(const Bytes& attribute) { return field(1, field(5, attribute)); }
}  // namespace

TEST_CASE(
    "Artifact validation accepts self-contained tensors without scanning raw payload as proto",
    "[nn][artifact]") {
    const Bytes raw{std::byte{0x6a}, std::byte{0}, std::byte{0x70}, std::byte{1}};
    auto tensor = field(9, raw);
    const auto location = scalar({.number = 14, .value = 0});
    tensor.insert(tensor.end(), location.begin(), location.end());
    CHECK(mouse::nn::validate_self_contained_onnx(model(graph(tensor))).has_value());
    std::ifstream input{std::filesystem::path{MOUSE_TEST_METADATA_DIRECTORY} / "valid.onnx",
                        std::ios::binary};
    REQUIRE(input.good());
    const std::vector<char> characters{std::istreambuf_iterator<char>{input}, {}};
    Bytes actual;
    for (const char value : characters) actual.push_back(static_cast<std::byte>(value));
    CHECK(mouse::nn::validate_self_contained_onnx(actual).has_value());
}

TEST_CASE("External tensors are rejected in every supported ONNX tensor container",
          "[nn][artifact]") {
    const auto external = scalar({.number = 14, .value = 1});
    CHECK_FALSE(mouse::nn::validate_self_contained_onnx(model(graph(external))).has_value());
    CHECK_FALSE(mouse::nn::validate_self_contained_onnx(model(graph(field(13, {})))).has_value());
    CHECK_FALSE(
        mouse::nn::validate_self_contained_onnx(model(graph(scalar({.number = 14, .value = 2}))))
            .has_value());
    CHECK_FALSE(
        mouse::nn::validate_self_contained_onnx(model(field(15, field(1, external)))).has_value());
    CHECK_FALSE(
        mouse::nn::validate_self_contained_onnx(model(field(15, field(2, external)))).has_value());
    for (const auto number : {5U, 10U}) {
        CHECK_FALSE(
            mouse::nn::validate_self_contained_onnx(model(attribute_graph(field(number, external))))
                .has_value());
    }
    for (const auto number : {6U, 11U}) {
        CHECK_FALSE(mouse::nn::validate_self_contained_onnx(
                        model(attribute_graph(field(number, graph(external)))))
                        .has_value());
    }
    for (const auto number : {22U, 23U}) {
        CHECK_FALSE(mouse::nn::validate_self_contained_onnx(
                        model(attribute_graph(field(number, field(1, external)))))
                        .has_value());
    }
    auto function = field(25, field(7, field(5, field(5, external))));
    auto valid_model = model({});
    function.insert(function.end(), valid_model.begin(), valid_model.end());
    CHECK_FALSE(mouse::nn::validate_self_contained_onnx(function).has_value());
    function = field(25, field(11, field(5, external)));
    function.insert(function.end(), valid_model.begin(), valid_model.end());
    CHECK_FALSE(mouse::nn::validate_self_contained_onnx(function).has_value());
    for (const auto number : {1U, 2U}) {
        auto training = field(20, field(number, graph(external)));
        training.insert(training.end(), valid_model.begin(), valid_model.end());
        CHECK_FALSE(mouse::nn::validate_self_contained_onnx(training).has_value());
    }
}

TEST_CASE("ONNX artifact wire boundaries and resource limits fail clearly", "[nn][artifact]") {
    for (const auto& bytes : std::vector<Bytes>{{},
                                                {std::byte{0}},
                                                {std::byte{0x3a}},
                                                {std::byte{0x3a}, std::byte{10}},
                                                {std::byte{0x3b}},
                                                {std::byte{0x0f}},
                                                scalar({.number = 7, .value = 0}),
                                                Bytes(11, std::byte{0x80})}) {
        CHECK_FALSE(mouse::nn::validate_self_contained_onnx(bytes).has_value());
    }
    auto duplicate = model({});
    const auto other = model({});
    duplicate.insert(duplicate.end(), other.begin(), other.end());
    CHECK_FALSE(mouse::nn::validate_self_contained_onnx(duplicate).has_value());
    Bytes deep;
    for (int depth = 0; depth < 16; ++depth) deep = attribute_graph(field(6, deep));
    const auto nested = mouse::nn::validate_self_contained_onnx(model(deep));
    REQUIRE_FALSE(nested.has_value());
    CHECK(nested.error().find("nesting") != std::string::npos);
    Bytes many;
    const auto unknown = scalar({.number = 99, .value = 0});
    for (int index = 0; index < 1000000; ++index) {
        many.insert(many.end(), unknown.begin(), unknown.end());
    }
    const auto excessive = mouse::nn::validate_self_contained_onnx(model(many));
    REQUIRE_FALSE(excessive.has_value());
    CHECK(excessive.error().find("field") != std::string::npos);
}

TEST_CASE("Artifact inspection bounds every graph name without changing model acceptance",
          "[nn][artifact][cache-path]") {
    const auto text = [](std::string_view value) {
        Bytes bytes;
        for (const auto ch : value) bytes.push_back(static_cast<std::byte>(ch));
        return bytes;
    };
    auto body = field(2, text("main_graph"));
    const auto nested = attribute_graph(field(6, field(2, text(std::string(170, '9')))));
    body.insert(body.end(), nested.begin(), nested.end());
    const auto bytes = model(body);
    const auto info = mouse::nn::inspect_onnx_artifact(bytes);
    REQUIRE(info);
    CHECK(info->maximum_graph_name_bytes == 170);
    CHECK(info->windows_safe_graph_names);
    CHECK(mouse::nn::validate_self_contained_onnx(bytes));
    const auto absent = mouse::nn::inspect_onnx_artifact(model({}));
    REQUIRE(absent);
    CHECK(absent->maximum_graph_name_bytes == 0);
    CHECK(absent->windows_safe_graph_names);
    for (const auto name :
         {"bad/name", "bad\\name", "bad:name", "bad*name", "bad?name", "bad\"name", "bad<name",
          "bad>name", "bad|name", "bad\nname", "bad\xc3\xa9", "badname.", "badname "}) {
        const auto unsafe = model(field(2, text(name)));
        const auto inspected = mouse::nn::inspect_onnx_artifact(unsafe);
        REQUIRE(inspected);
        CHECK_FALSE(inspected->windows_safe_graph_names);
        CHECK(mouse::nn::validate_self_contained_onnx(unsafe));
    }
    CHECK_FALSE(mouse::nn::inspect_onnx_artifact(model(graph(field(13, {})))));
    // Preserve the original validator's division of responsibilities: ORT
    // validates graph-name wire semantics; only Windows TRT needs safe names.
    const auto wrong_wire = model(scalar({.number = 2, .value = 1}));
    CHECK(mouse::nn::validate_self_contained_onnx(wrong_wire));
    const auto inspected = mouse::nn::inspect_onnx_artifact(wrong_wire);
    REQUIRE(inspected);
    CHECK_FALSE(inspected->windows_safe_graph_names);
}
TEST_CASE("Artifact cache-name inspection includes function and training graphs",
          "[nn][artifact][cache-path]") {
    const Bytes unsafe_name{std::byte{'/'}};
    const auto unsafe_graph = field(2, unsafe_name);
    for (const auto& nested :
         {field(25, field(7, field(5, field(6, unsafe_graph)))), field(20, field(1, unsafe_graph)),
          field(20, field(2, unsafe_graph))}) {
        auto bytes = model({});
        bytes.insert(bytes.end(), nested.begin(), nested.end());
        const auto inspected = mouse::nn::inspect_onnx_artifact(bytes);
        REQUIRE(inspected);
        CHECK(inspected->maximum_graph_name_bytes == 1);
        CHECK_FALSE(inspected->windows_safe_graph_names);
        CHECK(mouse::nn::validate_self_contained_onnx(bytes));
    }
}