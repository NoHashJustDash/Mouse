// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/util/hash128.h"

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
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
