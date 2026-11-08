// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/search/draw_rate.h"

#include <array>
#include <bit>
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>

#include "engine/uci/score.h"

namespace {
using mouse::search::draw_rate_ratio;
using mouse::search::DrawRateParameters;
using mouse::search::rescale_wdl;
// The logistic WDL model in exact double arithmetic: an equal position (mu 0)
// at width s draws with probability 1 - 2 / (1 + e^(1/s)).
double width_for_draw_rate(double draw) { return 1.0 / std::log((1.0 + draw) / (1.0 - draw)); }
double draw_rate_at_width(double width) { return 1.0 - 2.0 / (1.0 + std::exp(1.0 / width)); }
struct Lc0Case {
    std::uint32_t q_bits, d_bits;
    double target, reference, max_s;
    int cp, win, draw, loss;
    const char *wl, *d;
};
// Observed from the pinned lc0 v0.32.1 executable by draw-rate-v1/capture.py.
constexpr Lc0Case kLc0Cases[] = {
#include "tests/data/search/draw-rate-v1/cases.inc"
};
std::string fixed(float value, int precision) {
    std::ostringstream text;
    text << std::fixed << std::setprecision(precision) << value;
    return text.str();
}
}  // namespace

TEST_CASE("Draw-rate rescaling matches the lc0 executable", "[search][draw-rate][lc0-parity]") {
    for (const auto& c : kLc0Cases) {
        auto q = std::bit_cast<float>(c.q_bits), draw = std::bit_cast<float>(c.d_bits);
        INFO("q " << q << " d " << draw << " target " << c.target << " reference " << c.reference
                  << " max_s " << c.max_s);
        const DrawRateParameters parameters{
            .target = c.target, .reference = c.reference, .max_s = c.max_s};
        rescale_wdl(q, draw, draw_rate_ratio(parameters), static_cast<float>(parameters.max_s));
        // The UCI score passes the stored values through lc0's display inverse.
        const auto score = mouse::uci::format_score({.q = q, .draw = draw});
        REQUIRE(score);
        CHECK(score->centipawns == c.cp);
        CHECK(score->wdl == std::array{c.win, c.draw, c.loss});
        // Verbose stats print the same inverse, which keeps the width: a unit rescale.
        float shown_q = q, shown_draw = draw;
        rescale_wdl(shown_q, shown_draw, 1.0F, 10.0F);
        CHECK(fixed(shown_q, 5) == c.wl);
        CHECK(fixed(shown_draw, 3) == c.d);
    }
}

TEST_CASE("Draw-rate ratio is exactly one without a target and the model's width ratio with one",
          "[search][draw-rate]") {
    CHECK(draw_rate_ratio({.target = 0.0}) == 1.0F);
    CHECK(draw_rate_ratio({.target = 0.0, .reference = 0.3}) == 1.0F);
    const auto ratio = draw_rate_ratio({.target = 0.7, .reference = 0.5});
    CHECK(std::abs(ratio - width_for_draw_rate(0.7) / width_for_draw_rate(0.5)) < 1e-6);
    // Lc0 lifts an accidental tiny positive target to 0.001 rather than dividing by ~0.
    CHECK(draw_rate_ratio({.target = 1e-6, .reference = 0.5}) ==
          draw_rate_ratio({.target = 0.001, .reference = 0.5}));
}

TEST_CASE("Draw-rate rescaling moves an equal position's draw rate from reference to target",
          "[search][draw-rate]") {
    for (const double target : {0.3, 0.6, 0.7, 0.9}) {
        float q = 0.0F;
        auto draw = static_cast<float>(draw_rate_at_width(width_for_draw_rate(0.5)));
        REQUIRE(rescale_wdl(q, draw, draw_rate_ratio({.target = target, .reference = 0.5}), 10.0F));
        // FastLog/FastLogistic are approximations; lc0 carries the same error.
        CHECK(std::abs(draw - target) < 0.01);
        CHECK(std::abs(q) < 1e-6F);
    }
}

TEST_CASE("Draw-rate rescaling is colour symmetric and keeps the reported WDL_mu score",
          "[search][draw-rate]") {
    const auto ratio = draw_rate_ratio({.target = 0.75, .reference = 0.5});
    struct Case {
        float q, draw;
        bool beyond_one;  // |mu| > 1: the favourite already wins more often than it draws
    };
    // mu = (a - b)/(a + b) with a = ln(1/L - 1), b = ln(1/W - 1): 0.69, 1.53 and -0.23.
    for (const auto [q0, d0, beyond_one] :
         {Case{0.3F, 0.5F, false}, Case{0.6F, 0.3F, true}, Case{-0.1F, 0.7F, false}}) {
        float q = q0, draw = d0, mirrored_q = -q0, mirrored_draw = d0;
        REQUIRE(rescale_wdl(q, draw, ratio, 10.0F));
        REQUIRE(rescale_wdl(mirrored_q, mirrored_draw, ratio, 10.0F));
        // W and L swap, so only float rounding of 1+q versus 1-q differs.
        CHECK(std::abs(mirrored_q + q) < 1e-6F);
        CHECK(std::abs(mirrored_draw - draw) < 1e-6F);
        // A narrower logistic draws more inside |mu| < 1 and is more decisive beyond it.
        if (beyond_one) {
            CHECK(draw < d0);
            CHECK(std::abs(q) > std::abs(q0));
        } else {
            CHECK(draw > d0);
        }
        // The centipawn score is derived from mu, which the width rescale leaves alone.
        const auto before = mouse::uci::format_score({.q = q0, .draw = d0});
        const auto after = mouse::uci::format_score({.q = q, .draw = draw});
        REQUIRE(before);
        REQUIRE(after);
        CHECK(std::abs(before->centipawns - after->centipawns) <= 1);
    }
}

TEST_CASE("Draw-rate rescaling caps a broad network WDL width at WDLMaxS", "[search][draw-rate]") {
    // W 0.3 D 0.4 L 0.3 has width ~1.18; a 0.5 cap makes it decisive-leaning before scaling.
    const auto ratio = draw_rate_ratio({.target = 0.6, .reference = 0.5});
    float capped_q = 0.0F, capped_draw = 0.4F, open_q = 0.0F, open_draw = 0.4F;
    REQUIRE(rescale_wdl(capped_q, capped_draw, ratio, 0.5F));
    REQUIRE(rescale_wdl(open_q, open_draw, ratio, 10.0F));
    CHECK(capped_draw > open_draw);
    CHECK(std::abs(capped_draw - draw_rate_at_width(0.5 * ratio)) < 0.01);
}

TEST_CASE("Draw-rate rescaling leaves near-certain WDL unchanged", "[search][draw-rate]") {
    const auto ratio = draw_rate_ratio({.target = 0.7, .reference = 0.5});
    for (const auto [q0, d0] : {std::pair{1.0F, 0.0F},
                                {0.99995F, 0.00004F},
                                {0.0F, 1.0F},
                                {-0.5F, 0.5F},
                                {0.4F, 0.00005F}}) {
        float q = q0, draw = d0;
        CHECK_FALSE(rescale_wdl(q, draw, ratio, 1.4F));
        CHECK(q == q0);
        CHECK(draw == d0);
    }
}

TEST_CASE("Draw-rate options reject values outside lc0's ranges", "[search][draw-rate]") {
    using mouse::search::validate_draw_rate_parameters;
    CHECK(validate_draw_rate_parameters({}));
    CHECK(validate_draw_rate_parameters({.target = 0.999, .reference = 0.001, .max_s = 10}));
    CHECK_FALSE(validate_draw_rate_parameters({.target = -0.01}));
    CHECK_FALSE(validate_draw_rate_parameters({.target = 1.0}));
    CHECK_FALSE(validate_draw_rate_parameters({.reference = 0.0}));
    CHECK_FALSE(validate_draw_rate_parameters({.reference = 1.0}));
    CHECK_FALSE(validate_draw_rate_parameters({.max_s = -1}));
    CHECK_FALSE(validate_draw_rate_parameters({.max_s = 10.5}));
    CHECK_FALSE(validate_draw_rate_parameters({.target = std::nan("")}));
}
