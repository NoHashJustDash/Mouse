// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/nn/commands.h"

#include <array>
#include <bit>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "engine/util/sha256.h"

namespace {
using mouse::nn::BackendCommandKind;
using mouse::nn::InputPlanes;
using mouse::nn::NNOutput;
std::vector<std::byte> encoded_input() {
    std::vector<std::byte> bytes;
    for (const char character : std::string_view{"MOUSEIN1"}) {
        bytes.push_back(static_cast<std::byte>(character));
    }
    for (const auto value : {2U, 112U}) {
        for (unsigned int shift = 0; shift < 32; shift += 8) {
            bytes.push_back(static_cast<std::byte>((value >> shift) & 255U));
        }
    }
    for (unsigned int row = 0; row < 2; ++row) {
        for (unsigned int plane = 0; plane < 112; ++plane) {
            const std::uint64_t mask = std::uint64_t{1} << (plane % 64);
            const auto value = std::bit_cast<std::uint32_t>(static_cast<float>(row + plane));
            for (unsigned int shift = 0; shift < 64; shift += 8) {
                bytes.push_back(static_cast<std::byte>((mask >> shift) & 255U));
            }
            for (unsigned int shift = 0; shift < 32; shift += 8) {
                bytes.push_back(static_cast<std::byte>((value >> shift) & 255U));
            }
        }
    }
    return bytes;
}
mouse::Result<mouse::nn::BackendCommand> parse(std::string_view options, BackendCommandKind kind) {
    std::vector<std::string_view> arguments;
    while (!options.empty()) {
        const auto end = options.find(' ');
        arguments.push_back(options.substr(0, end));
        options = end == std::string_view::npos ? std::string_view{} : options.substr(end + 1);
    }
    return mouse::nn::parse_backend_command(kind, arguments);
}
}  // namespace

TEST_CASE("Backend commands require explicit model provider and precision", "[nn][commands]") {
    const auto evaluation = parse(
        "--weights model.onnx --backend cpu --precision fp32 --inputs planes.bin "
        "--buckets 1,4,8 --cpu-threads 2 --nncache-mb 1",
        BackendCommandKind::Evaluate);
    REQUIRE(evaluation.has_value());
    CHECK(evaluation->backend.model_path == "model.onnx");
    CHECK(evaluation->backend.kind == mouse::nn::BackendKind::Cpu);
    CHECK(evaluation->backend.precision == mouse::nn::ComputePrecision::Float32);
    CHECK(evaluation->backend.batch_buckets == std::vector<std::size_t>{1, 4, 8});
    CHECK(evaluation->cache_bytes == 1048576);
    const auto benchmark =
        parse("--weights model --backend tensorrt --precision fp16 --cache-dir cache",
              BackendCommandKind::Benchmark);
    REQUIRE(benchmark.has_value());
    CHECK(benchmark->start_batch_size == 1);
    CHECK(benchmark->max_batch_size == 1024);
    CHECK(benchmark->batch_step == 1);
    CHECK(benchmark->batches == 100);
    CHECK_FALSE(
        parse("--weights model --backend cpu --precision fp32", BackendCommandKind::Evaluate)
            .has_value());
    for (const auto options :
         {"--backend cpu --precision fp32 --inputs x", "--weights x --precision fp32 --inputs x",
          "--weights x --backend cpu --inputs x"}) {
        CHECK_FALSE(parse(options, BackendCommandKind::Evaluate).has_value());
    }
}

TEST_CASE("Backend option errors are reported rather than ignored", "[nn][commands]") {
    constexpr std::string_view base =
        "--weights model --backend cuda --precision fp16 --inputs planes ";
    for (const auto suffix :
         {"--unknown x", "--device -1", "--device 999999999999999999999999", "--cpu-threads 0",
          "--workspace-mb 0", "--buckets 4,1", "--buckets 1,1", "--buckets 0", "--buckets 1025",
          "--buckets 1,", "--weights duplicate", "--precision fp32", "--backend", "--nncache-mb -1",
          "--inputs duplicate"}) {
        CHECK_FALSE(parse(std::string{base} + suffix, BackendCommandKind::Evaluate).has_value());
    }
    CHECK_FALSE(parse("--weights x --backend auto --precision fp16 --inputs x",
                      BackendCommandKind::Evaluate)
                    .has_value());
    CHECK_FALSE(
        parse("--weights x --backend cpu --precision auto --inputs x", BackendCommandKind::Evaluate)
            .has_value());
    CHECK_FALSE(parse("--weights x --backend cpu --precision fp32 --max-batch-size 0",
                      BackendCommandKind::Benchmark)
                    .has_value());
    CHECK_FALSE(parse("--weights x --backend cpu --precision fp32 --start-batch-size 3 "
                      "--max-batch-size 2",
                      BackendCommandKind::Benchmark)
                    .has_value());
    CHECK_FALSE(parse("--weights x --backend cpu --precision fp32 --nncache-mb 1",
                      BackendCommandKind::Benchmark)
                    .has_value());
}

TEST_CASE("Backend input transport preserves every mask value bit and row", "[nn][commands]") {
    const auto bytes = encoded_input();
    const auto batch = mouse::nn::decode_plane_batch(bytes);
    REQUIRE(batch.has_value());
    REQUIRE(batch->inputs.size() == 2);
    CHECK(batch->input_sha256.size() == 64);
    for (std::size_t row = 0; row < 2; ++row) {
        for (std::size_t plane = 0; plane < 112; ++plane) {
            CHECK(batch->inputs.at(row).at(plane).mask == (std::uint64_t{1} << (plane % 64)));
            CHECK(batch->inputs.at(row).at(plane).value == static_cast<float>(row + plane));
        }
    }
    for (std::size_t length = 0; length < 16; ++length) {
        CHECK_FALSE(mouse::nn::decode_plane_batch(std::span{bytes}.first(length)).has_value());
    }
    auto corrupt = bytes;
    corrupt.at(0) = std::byte{0};
    CHECK_FALSE(mouse::nn::decode_plane_batch(corrupt).has_value());
    corrupt = bytes;
    corrupt.at(8) = std::byte{0};
    CHECK_FALSE(mouse::nn::decode_plane_batch(corrupt).has_value());
    corrupt = bytes;
    corrupt.at(12) = std::byte{111};
    CHECK_FALSE(mouse::nn::decode_plane_batch(corrupt).has_value());
    corrupt = bytes;
    corrupt.pop_back();
    CHECK_FALSE(mouse::nn::decode_plane_batch(corrupt).has_value());
    corrupt = bytes;
    corrupt.push_back(std::byte{0});
    CHECK_FALSE(mouse::nn::decode_plane_batch(corrupt).has_value());
    corrupt = bytes;
    corrupt.at(26) = std::byte{0xc0};
    corrupt.at(27) = std::byte{0x7f};
    CHECK_FALSE(mouse::nn::decode_plane_batch(corrupt).has_value());
}

TEST_CASE("Backend output transport carries identities and every finite output", "[nn][commands]") {
    NNOutput row;
    row.policy_logits.fill(-0.25F);
    row.wdl = {0.25F, 0.5F, 0.25F};
    row.moves_left = 10.0F;
    const std::array rows{row};
    const auto output = mouse::nn::encode_backend_outputs(
        rows, {.model_sha256 = std::string(64, 'a'), .input_sha256 = std::string(64, 'b')});
    REQUIRE(output.has_value());
    CHECK(output->size() == 144 + 1862 * 4);
    CHECK(output->at(0) == std::byte{'M'});
    CHECK(output->at(7) == std::byte{'1'});
    CHECK(output->at(8) == std::byte{1});
    CHECK(output->at(16) == std::byte{'a'});
    CHECK(output->at(80) == std::byte{'b'});
    CHECK(output->at(147) == std::byte{0xbe});
    CHECK_FALSE(mouse::nn::encode_backend_outputs(
                    rows, {.model_sha256 = "bad", .input_sha256 = std::string(64, 'b')})
                    .has_value());
    row.moves_left.reset();
    CHECK_FALSE(mouse::nn::encode_backend_outputs(
                    std::array{row},
                    {.model_sha256 = std::string(64, 'a'), .input_sha256 = std::string(64, 'b')})
                    .has_value());
    row.moves_left = 10.0F;
    row.policy_logits.at(1800) = std::numeric_limits<float>::infinity();
    CHECK_FALSE(mouse::nn::encode_backend_outputs(
                    std::array{row},
                    {.model_sha256 = std::string(64, 'a'), .input_sha256 = std::string(64, 'b')})
                    .has_value());
}

TEST_CASE("Reusable backend configuration does not invent an evaluation input",
          "[nn][commands][configuration]") {
    const auto config = parse(
        "--weights model --backend cpu --precision fp32 --buckets 1,16 "
        "--nncache-mb 2",
        BackendCommandKind::Configuration);
    REQUIRE(config.has_value());
    CHECK(config->inputs_path.empty());
    CHECK(config->backend.batch_buckets == std::vector<std::size_t>{1, 16});
    CHECK(config->cache_bytes == 2097152);
    CHECK_FALSE(parse("--weights model --backend cpu --precision fp32",
                      static_cast<BackendCommandKind>(255))
                    .has_value());
    CHECK_FALSE(parse("--weights model --backend cpu --precision fp32 --inputs unused",
                      BackendCommandKind::Configuration)
                    .has_value());
    CHECK_FALSE(parse("--weights model --backend cpu --precision fp32 --batches 1",
                      BackendCommandKind::Configuration)
                    .has_value());
}

TEST_CASE("Backend output transport preserves all bits across rows and rejects missing MLH",
          "[nn][commands]") {
    NNOutput first;
    first.policy_logits.fill(-0.25F);
    first.wdl = {0.25F, 0.5F, 0.25F};
    first.moves_left = 10.0F;
    NNOutput second;
    second.policy_logits.fill(-0.0F);
    second.wdl = {1.0F, 0.0F, 0.0F};
    second.moves_left = 0.0F;
    std::array rows{first, second};
    const auto encode = [&rows] {
        return mouse::nn::encode_backend_outputs(
            rows, {.model_sha256 = std::string(64, 'a'), .input_sha256 = std::string(64, 'b')});
    };
    const auto output = encode();
    REQUIRE(output.has_value());
    // Independent Python struct.pack("<II"/"<f") protocol calculation. It covers
    // the complete header and both 1862-float rows, including negative zero.
    CHECK(mouse::sha256(*output) ==
          "3ce309942bad8b2d5eb38de7d7e89653bd152ec9ab45bd8f8f23ab2a9e60b9eb");
    for (std::size_t missing = 0; missing < rows.size(); ++missing) {
        rows = {first, second};
        rows.at(missing).moves_left.reset();
        const auto invalid = encode();
        REQUIRE_FALSE(invalid.has_value());
        CHECK(invalid.error().find("missing MLH") != std::string::npos);
    }
}
