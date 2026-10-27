// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/uci/score.h"

#include <array>
#include <bit>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {
struct ReferenceCase {
    const char* name;
    std::uint32_t q_bits, draw_bits;
    int cp;
    std::array<int, 3> wdl;
};
constexpr std::array<ReferenceCase, 28> kReferenceCases{{
#include "../data/uci/score-v1/cases.inc"
}};
}  // namespace

TEST_CASE("Default UCI score matches all repeated official binary float32 fixtures",
          "[uci][score]") {
    for (const auto& reference : kReferenceCases) {
        CAPTURE(reference.name);
        const auto score =
            mouse::uci::format_score({.q = std::bit_cast<float>(reference.q_bits),
                                      .draw = std::bit_cast<float>(reference.draw_bits)});
        REQUIRE(score);
        CHECK(score->centipawns == reference.cp);
        CHECK(score->wdl == reference.wdl);
    }
}

TEST_CASE("Default UCI score truncates cp and preserves asymmetric WDL half ties", "[uci][score]") {
    const auto positive = mouse::uci::format_score({.q = 0.125F, .draw = 0});
    const auto negative = mouse::uci::format_score({.q = -0.125F, .draw = 0});
    REQUIRE(positive);
    REQUIRE(negative);
    CHECK(positive->centipawns == 8);
    CHECK(negative->centipawns == -8);
    CHECK(positive->wdl == std::array<int, 3>{563, 0, 437});
    CHECK(negative->wdl == std::array<int, 3>{438, 0, 562});
    const auto old_control = mouse::uci::format_score({.q = 0.5F, .draw = 0});
    REQUIRE(old_control);
    CHECK(old_control->centipawns == 44);
    CHECK(old_control->centipawns != 89);  // Actual legacy centipawn control.
}

TEST_CASE("Default UCI score rejects nonfinite and nonprobability input", "[uci][score]") {
    using mouse::uci::format_score;
    for (const auto invalid :
         {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
          -std::numeric_limits<float>::infinity()}) {
        CHECK_FALSE(format_score({invalid, 0}));
        CHECK_FALSE(format_score({0, invalid}));
    }
    for (const auto invalid : {std::nextafter(1.0F, 2.0F), std::nextafter(-1.0F, -2.0F)})
        CHECK_FALSE(format_score({invalid, 0}));
    CHECK_FALSE(format_score({0, -std::numeric_limits<float>::denorm_min()}));
    CHECK_FALSE(format_score({0, std::nextafter(1.0F, 2.0F)}));
    CHECK_FALSE(format_score({0.75F, 0.5F}));
    CHECK_FALSE(format_score({-0.75F, 0.5F}));
    // A normalized float32 WDL subtraction can round to this valid float sum.
    CHECK(format_score({0.1F, 0.9F}));
    CHECK(format_score({-0.1F, 0.9F}));
    CHECK(format_score({-0.0F, -0.0F}));
    CHECK(format_score({.q = 0.8F, .draw = 0.2F}));
    CHECK(format_score({.q = -0.8F, .draw = 0.2F}));
}

TEST_CASE("Default UCI score keeps every valid coarse simplex triple representable",
          "[uci][score]") {
    for (int q = -128; q <= 128; ++q) {
        for (int draw = 0; draw <= 128 - std::abs(q); ++draw) {
            const auto score = mouse::uci::format_score(
                {.q = static_cast<float>(q) / 128.0F, .draw = static_cast<float>(draw) / 128.0F});
            REQUIRE(score);
            CHECK(score->wdl[0] >= 0);
            CHECK(score->wdl[1] >= 0);
            CHECK(score->wdl[2] >= 0);
            CHECK(score->wdl[0] + score->wdl[1] + score->wdl[2] == 1000);
        }
    }
}

TEST_CASE("Inverse WDL cancellation is an explicit error before any integer conversion",
          "[uci][score]") {
    const auto score = mouse::uci::format_score(
        {.q = std::bit_cast<float>(0x3db0f27cU), .draw = std::bit_cast<float>(0x38e6afcdU)});
    REQUIRE_FALSE(score);
    CHECK(score.error().find("not representable") != std::string::npos);
}
