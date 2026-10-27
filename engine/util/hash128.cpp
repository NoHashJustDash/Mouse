// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/util/hash128.h"

#include <algorithm>
#include <bit>
namespace mouse {
namespace {
// Independent, bounds-safe implementation of Austin Appleby's public-domain
// MurmurHash3 x64_128 algorithm. Constants, rotations and finalization:
// https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp
constexpr std::uint64_t kFirst = 0x87c37b91114253d5ULL, kSecond = 0x4cf5ad432745937fULL;
std::uint64_t little_endian(std::span<const std::byte> bytes) noexcept {
    std::uint64_t result = 0;
    for (std::size_t i = 0; i < bytes.size(); ++i)
        result |= std::to_integer<std::uint64_t>(bytes[i]) << (8 * i);
    return result;
}
std::uint64_t scramble(std::uint64_t word, std::uint64_t first, int rotation,
                       std::uint64_t second) noexcept {
    return std::rotl(word * first, rotation) * second;
}
std::uint64_t avalanche(std::uint64_t value) noexcept {
    value = (value ^ (value >> 33)) * 0xff51afd7ed558ccdULL;
    value = (value ^ (value >> 33)) * 0xc4ceb9fe1a85ec53ULL;
    return value ^ (value >> 33);
}
}  // namespace
Hash128 hash128(std::span<const std::byte> bytes, std::uint32_t seed) noexcept {
    const auto size = static_cast<std::uint64_t>(bytes.size());
    std::uint64_t first = seed, second = seed;
    while (bytes.size() >= 16) {
        first ^= scramble(little_endian(bytes.first(8)), kFirst, 31, kSecond);
        first = (std::rotl(first, 27) + second) * 5 + 0x52dce729ULL;
        second ^= scramble(little_endian(bytes.subspan(8, 8)), kSecond, 33, kFirst);
        second = (std::rotl(second, 31) + first) * 5 + 0x38495ab5ULL;
        bytes = bytes.subspan(16);
    }
    if (bytes.size() > 8) second ^= scramble(little_endian(bytes.subspan(8)), kSecond, 33, kFirst);
    if (!bytes.empty())
        first ^= scramble(little_endian(bytes.first(std::min(bytes.size(), std::size_t{8}))),
                          kFirst, 31, kSecond);
    first ^= size;
    second ^= size;
    first += second;
    second += first;
    first = avalanche(first);
    second = avalanche(second);
    first += second;
    second += first;
    return {first, second};
}
}  // namespace mouse
