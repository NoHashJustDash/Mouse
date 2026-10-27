// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
namespace mouse {
using Hash128 = std::array<std::uint64_t, 2>;
// Portable MurmurHash3 x64_128, with explicit little-endian input/output order.
// This is a cache fingerprint, not cryptographic artifact authentication.
[[nodiscard]] Hash128 hash128(std::span<const std::byte>, std::uint32_t seed = 0) noexcept;
}  // namespace mouse
