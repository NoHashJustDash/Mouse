// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/util/hash128.h"

#include <algorithm>
#include <array>
#include <bit>
#include <catch2/catch_test_macros.hpp>
namespace {
// Byte-at-a-time MurmurHash3 x64_128 reference (aappleby/smhasher
// src/MurmurHash3.cpp), assembling every word from explicit little-endian
// bytes so it is independent of how the engine loads its 8-byte blocks.
mouse::Hash128 bytewise(std::span<const std::byte> bytes, std::uint32_t seed) {
    const auto word = [&](std::size_t at, std::size_t count) {
        std::uint64_t value = 0;
        for (std::size_t i = 0; i < count; ++i)
            value |= std::to_integer<std::uint64_t>(bytes[at + i]) << (8 * i);
        return value;
    };
    const auto mix = [](std::uint64_t value) {
        value ^= value >> 33;
        value *= 0xff51afd7ed558ccdULL;
        value ^= value >> 33;
        value *= 0xc4ceb9fe1a85ec53ULL;
        return value ^ (value >> 33);
    };
    constexpr std::uint64_t c1 = 0x87c37b91114253d5ULL, c2 = 0x4cf5ad432745937fULL;
    std::uint64_t h1 = seed, h2 = seed;
    const std::size_t blocks = bytes.size() / 16, tail = bytes.size() % 16;
    for (std::size_t block = 0; block < blocks; ++block) {
        h1 ^= std::rotl(word(16 * block, 8) * c1, 31) * c2;
        h1 = (std::rotl(h1, 27) + h2) * 5 + 0x52dce729ULL;
        h2 ^= std::rotl(word(16 * block + 8, 8) * c2, 33) * c1;
        h2 = (std::rotl(h2, 31) + h1) * 5 + 0x38495ab5ULL;
    }
    if (tail > 8) h2 ^= std::rotl(word(16 * blocks + 8, tail - 8) * c2, 33) * c1;
    if (tail > 0) h1 ^= std::rotl(word(16 * blocks, std::min(tail, std::size_t{8})) * c1, 31) * c2;
    const auto size = static_cast<std::uint64_t>(bytes.size());
    h1 ^= size;
    h2 ^= size;
    h1 += h2;
    h2 += h1;
    h1 = mix(h1);
    h2 = mix(h2);
    h1 += h2;
    h2 += h1;
    return {h1, h2};
}
}  // namespace
TEST_CASE("Cache hash block loads match byte-wise assembly at every offset and length",
          "[util][hash128]") {
    // High-bit bytes in every position catch sign extension or byte-order errors
    // in the 8-byte block loads; offsets 0..15 cover every alignment.
    std::array<std::byte, 320> data{};
    for (std::size_t i = 0; i < data.size(); ++i)
        data.at(i) = static_cast<std::byte>((i * 151 + 0x80) & 0xffU);
    bool identical = true;
    for (const std::uint32_t seed : {0U, 0x9747b28cU, 0xffffffffU})
        for (std::size_t offset = 0; offset < 16; ++offset)
            for (std::size_t size = 0; offset + size <= data.size() && size <= 300; ++size) {
                const auto view = std::span{data}.subspan(offset, size);
                identical = identical && mouse::hash128(view, seed) == bytewise(view, seed);
            }
    CHECK(identical);
}
TEST_CASE("Cache hash matches upstream SMHasher x64_128 verification", "[util][hash128]") {
    // Independent published oracle: main.cpp's Murmur3F verification 0x6384BA69.
    // Protocol: src/KeysetTest.cpp::VerificationTest in aappleby/smhasher.
    std::array<std::byte, 256> key{};
    std::array<std::byte, std::size_t{256} * 16> hashes{};
    for (std::size_t size = 0; size < key.size(); ++size) {
        key.at(size) = static_cast<std::byte>(size);
        const auto digest =
            mouse::hash128(std::span{key}.first(size), static_cast<std::uint32_t>(256 - size));
        for (std::size_t byte = 0; byte < 16; ++byte)
            hashes.at(size * 16 + byte) =
                static_cast<std::byte>((digest.at(byte / 8) >> (8 * (byte % 8))) & 0xffU);
    }
    CHECK(static_cast<std::uint32_t>(mouse::hash128(hashes)[0]) == 0x6384ba69U);
    CHECK(mouse::hash128({}) == mouse::Hash128{});
}
TEST_CASE("Cache hash handles unaligned spans and every tail boundary", "[util][hash128]") {
    std::array<std::byte, 256> key{};
    for (std::size_t i = 0; i < key.size(); ++i) key.at(i) = static_cast<std::byte>(i);
    for (std::size_t offset = 0; offset < 16; ++offset) {
        std::array<std::byte, 272> unaligned{};
        std::ranges::copy(key, unaligned.begin() + static_cast<std::ptrdiff_t>(offset));
        for (std::size_t size = 0; size <= key.size(); ++size)
            CHECK(mouse::hash128(std::span{key}.first(size)) ==
                  mouse::hash128(std::span{unaligned}.subspan(offset, size)));
    }
}
