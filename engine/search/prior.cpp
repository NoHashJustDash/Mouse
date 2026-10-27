// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/search/prior.h"

#include <algorithm>
#include <array>
#include <bit>
#include <bitset>
#include <cassert>
#include <cmath>
#include <limits>

#include "engine/nn/policy_map.h"

namespace mouse::search {
namespace {
// The caller uses shifts 13..24. Retain the low bit for an exact ties-to-even
// decision, including rounding a largest subnormal into the first normal.
std::uint32_t rounded_significand(std::uint32_t value, unsigned shift) noexcept {
    const auto retained = value >> shift;
    const auto remainder = value & ((std::uint32_t{1} << shift) - 1U);
    const auto midpoint = std::uint32_t{1} << (shift - 1U);
    return retained + static_cast<std::uint32_t>(remainder > midpoint ||
                                                 (remainder == midpoint && (retained & 1U) != 0));
}
}  // namespace

Result<std::uint16_t> encode_prior(float value) {
    static_assert(sizeof(float) == sizeof(std::uint32_t));
    static_assert(std::numeric_limits<float>::is_iec559 &&
                  std::numeric_limits<float>::digits == 24);
    if (!std::isfinite(value) || value < 0.0F || value > 1.0F)
        return tl::unexpected("Prior must be finite and in [0,1]");
    if (value == 0.0F) return std::uint16_t{0};
    const auto bits = std::bit_cast<std::uint32_t>(value);
    const auto exponent = (bits >> 23U) & 0xffU;
    if (exponent < 102U) return std::uint16_t{0};
    const auto significand = (bits & 0x7fffffU) | 0x800000U;
    if (exponent < 113U)
        return static_cast<std::uint16_t>(rounded_significand(significand, 126U - exponent));
    const auto result = ((exponent - 113U) << 10U) + rounded_significand(significand, 13U);
    return static_cast<std::uint16_t>(result);
}

float decode_prior(std::uint16_t bits) noexcept {
    assert(bits <= 0x3c00);
    const auto exponent = static_cast<std::uint32_t>(bits) >> 10U;
    const auto fraction = static_cast<std::uint32_t>(bits) & 0x3ffU;
    if (exponent != 0) return std::bit_cast<float>(((exponent + 112U) << 23U) | (fraction << 13U));
    if (fraction == 0) return 0.0F;
    const auto highest_bit = 31U - static_cast<unsigned>(std::countl_zero(fraction));
    const auto normalized_fraction = (fraction ^ (std::uint32_t{1} << highest_bit))
                                     << (23U - highest_bit);
    return std::bit_cast<float>(((highest_bit + 103U) << 23U) | normalized_fraction);
}

Result<void> encode_legal_logits(std::span<const float> logits, double temperature,
                                 std::span<std::uint16_t> output) {
    if (logits.empty() || logits.size() > chess::MoveList::kCapacity ||
        output.size() != logits.size())
        return tl::unexpected("Prior output requires a complete nonempty bounded legal move list");
    if (!std::isfinite(temperature) || temperature <= 0)
        return tl::unexpected("Policy temperature must be positive and finite");
    std::array<std::uint16_t, chess::MoveList::kCapacity> staged{};
    std::array<double, chess::MoveList::kCapacity> weights{};
    double maximum = -std::numeric_limits<double>::infinity();
    for (const float value : logits) {
        if (!std::isfinite(value)) return tl::unexpected("Legal policy logit must be finite");
        maximum = std::max(maximum, static_cast<double>(value));
    }
    // lc0 neural/wrapper.cc: mask first, softmax at the configured temperature.
    double sum = 0;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        weights.at(i) = std::exp((static_cast<double>(logits[i]) - maximum) / temperature);
        sum += weights.at(i);
    }
    for (std::size_t i = 0; i < logits.size(); ++i) {
        const auto prior = encode_prior(static_cast<float>(weights.at(i) / sum));
        if (!prior) return tl::unexpected(prior.error());
        staged.at(i) = *prior;
    }
    std::ranges::copy(std::span{staged}.first(logits.size()), output.begin());
    return {};
}
Result<void> encode_legal_priors(std::span<const chess::Move> moves, chess::Color side,
                                 std::span<const float, 1858> logits, double temperature,
                                 std::span<std::uint16_t> output) {
    if (moves.empty() || moves.size() > chess::MoveList::kCapacity || output.size() != moves.size())
        return tl::unexpected("Prior output requires a complete nonempty bounded legal move list");
    std::array<float, chess::MoveList::kCapacity> legal{};
    std::bitset<1858> seen;
    for (std::size_t i = 0; i < moves.size(); ++i) {
        const auto slot = nn::PolicyMap::index(moves[i], side);
        if (!slot) return tl::unexpected(slot.error());
        if (seen.test(*slot)) return tl::unexpected("Legal moves have duplicate policy indices");
        seen.set(*slot);
        legal.at(i) = logits[*slot];
    }
    return encode_legal_logits(std::span{legal}.first(moves.size()), temperature, output);
}
}  // namespace mouse::search
