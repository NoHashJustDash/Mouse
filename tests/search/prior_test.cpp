// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/search/prior.h"

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdint>
#include <limits>

#include "engine/chess/attacks.h"
#include "engine/chess/movegen.h"
#include "engine/nn/policy_map.h"

TEST_CASE("Binary16 priors preserve endpoints and subnormal boundaries", "[search][prior]") {
    using mouse::search::decode_prior;
    using mouse::search::encode_prior;
    REQUIRE(encode_prior(0.0F) == 0x0000);
    REQUIRE(encode_prior(-0.0F) == 0x0000);
    REQUIRE(encode_prior(1.0F) == 0x3c00);
    REQUIRE(encode_prior(0.5F) == 0x3800);
    REQUIRE(encode_prior(std::ldexp(1.0F, -14)) == 0x0400);
    REQUIRE(encode_prior(std::ldexp(1.0F, -24)) == 0x0001);
    REQUIRE(encode_prior(std::ldexp(1.0F, -25)) == 0x0000);
    REQUIRE(encode_prior(std::nextafter(std::ldexp(1.0F, -25), 1.0F)) == 0x0001);
    CHECK(decode_prior(0x0001) == std::ldexp(1.0F, -24));
    CHECK(decode_prior(0x03ff) == std::ldexp(1023.0F, -24));
    CHECK(decode_prior(0x0400) == std::ldexp(1.0F, -14));
    for (const float bad :
         {-1.0F, std::nextafter(1.0F, 2.0F), std::numeric_limits<float>::quiet_NaN(),
          std::numeric_limits<float>::infinity()}) {
        CHECK_FALSE(encode_prior(bad));
    }
}

TEST_CASE("Every nonnegative binary16 prior roundtrips and midpoint ties round even",
          "[search][prior]") {
    using mouse::search::decode_prior;
    using mouse::search::encode_prior;
    for (std::uint32_t bits = 0; bits <= 0x3c00; ++bits) {
        const auto exponent = bits >> 10U;
        const auto fraction = bits & 0x3ffU;
        const float independent = exponent == 0 ? std::ldexp(static_cast<float>(fraction), -24)
                                                : std::ldexp(static_cast<float>(1024U + fraction),
                                                             static_cast<int>(exponent) - 25);
        const auto code = static_cast<std::uint16_t>(bits);
        CHECK(decode_prior(code) == independent);
        REQUIRE(encode_prior(independent) == code);
        if (bits == 0x3c00) continue;
        const float lower = decode_prior(code);
        const float upper = decode_prior(static_cast<std::uint16_t>(bits + 1));
        const float midpoint = (lower + upper) * 0.5F;
        const auto even = static_cast<std::uint16_t>(bits + (bits & 1U));
        CHECK(encode_prior(midpoint) == even);
        CHECK(encode_prior(std::nextafter(midpoint, lower)) == code);
        CHECK(encode_prior(std::nextafter(midpoint, upper)) == bits + 1U);
    }
}

TEST_CASE("Legal policy normalization masks illegal slots and respects piece and color indices",
          "[search][prior]") {
    const mouse::chess::AttackTables attacks;
    constexpr std::array fens{mouse::chess::kStartFen,
                              std::string_view{"4k2r/6P1/8/8/8/8/8/4K3 w k - 0 1"},
                              std::string_view{"4k3/8/8/8/8/8/6p1/4K2R b K - 0 1"},
                              std::string_view{"r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"},
                              std::string_view{"r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1"},
                              std::string_view{"4k3/8/8/8/8/8/8/5KR1 w G - 0 1"}};
    for (const auto fen : fens) {
        auto position = mouse::chess::Position::from_fen(fen, attacks);
        REQUIRE(position);
        const auto legal = mouse::chess::legal_moves(*position, attacks);
        const std::span<const mouse::chess::Move> moves{legal.begin(), legal.end()};
        std::array<float, 1858> logits{};
        logits.fill(std::numeric_limits<float>::quiet_NaN());
        std::array<double, mouse::chess::MoveList::kCapacity> numerators{};
        double sum = 0;
        constexpr double temperature = 1.359;
        for (std::size_t i = 0; i < moves.size(); ++i) {
            const auto slot = mouse::nn::PolicyMap::index(moves[i], position->side_to_move());
            REQUIRE(slot);
            logits.at(*slot) = static_cast<float>(i) * 0.3F;
            numerators.at(i) = std::exp(static_cast<double>(logits.at(*slot)) / temperature);
            sum += numerators.at(i);
        }
        std::array<std::uint16_t, mouse::chess::MoveList::kCapacity> storage{};
        auto output = std::span{storage}.first(moves.size());
        REQUIRE(mouse::search::encode_legal_priors(moves, position->side_to_move(), logits,
                                                   temperature, output));
        double decoded_sum = 0;
        for (std::size_t i = 0; i < moves.size(); ++i) {
            const double expected = numerators.at(i) / sum;
            const double observed = mouse::search::decode_prior(output[i]);
            CHECK(std::abs(observed - expected) <=
                  std::max(std::ldexp(1.0, -25), expected * 0.0005));
            decoded_sum += observed;
        }
        CHECK(std::abs(decoded_sum - 1.0) < 0.001);
    }
}

TEST_CASE("Legal priors reject invalid input atomically and remain stable at extreme temperature",
          "[search][prior]") {
    using mouse::chess::Color;
    using mouse::chess::Move;
    using mouse::chess::square;
    const std::array moves{Move{square(0, 1), square(0, 2)}, Move{square(1, 1), square(1, 2)}};
    std::array<float, 1858> logits{};
    std::array<std::uint16_t, 2> out{0xdead, 0xbeef};
    const auto original = out;
    const auto first = mouse::nn::PolicyMap::index(moves[0], Color::White);
    const auto second = mouse::nn::PolicyMap::index(moves[1], Color::White);
    REQUIRE(first);
    REQUIRE(second);
    for (const double temperature : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                                     std::numeric_limits<double>::quiet_NaN()}) {
        CHECK_FALSE(
            mouse::search::encode_legal_priors(moves, Color::White, logits, temperature, out));
        CHECK(out == original);
    }
    logits.at(*second) = std::numeric_limits<float>::quiet_NaN();
    CHECK_FALSE(mouse::search::encode_legal_priors(moves, Color::White, logits, 1.0, out));
    CHECK(out == original);
    logits.at(*second) = 0;
    const std::array duplicate{moves[0], moves[0]};
    CHECK_FALSE(mouse::search::encode_legal_priors(duplicate, Color::White, logits, 1.0, out));
    CHECK_FALSE(mouse::search::encode_legal_priors(moves, static_cast<Color>(2), logits, 1.0, out));
    CHECK_FALSE(mouse::search::encode_legal_priors(moves, Color::White, logits, 1.0,
                                                   std::span{out}.first(1)));
    CHECK_FALSE(mouse::search::encode_legal_priors({}, Color::White, logits, 1.0, {}));
    CHECK(out == original);
    logits.at(*first) = std::numeric_limits<float>::max();
    logits.at(*second) = -std::numeric_limits<float>::max();
    REQUIRE(mouse::search::encode_legal_priors(moves, Color::White, logits,
                                               std::numeric_limits<double>::denorm_min(), out));
    CHECK(out == std::array<std::uint16_t, 2>{0x3c00, 0});
}
