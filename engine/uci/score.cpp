// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/uci/score.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <string>

namespace mouse::uci {
namespace {
static_assert(std::numeric_limits<float>::is_iec559 && std::numeric_limits<float>::digits == 24);

// Preserve the reference's float32 operation boundaries across compilers and
// prevent a contracted multiply/add from changing the integer wire result.
float rounded(float value) {
    volatile const float stored = value;
    return stored;
}

// Original arithmetic implementation of the pinned approximations, using
// frexp/ldexp rather than copying the upstream float-bit manipulation.
// Reference: Lc0 fd71a2d src/utils/fastmath.h FastLog/FastExp2/FastLogistic.
float fast_log(float value) {
    int exponent = 0;
    const float mantissa = std::frexp(value, &exponent);
    const float fraction = rounded(rounded(mantissa * 2.0F) - 1.0F);
    const float shape = rounded(1.3465552F - rounded(0.34655523F * fraction));
    const float curve = rounded(fraction * shape);
    const float binary_log = rounded(rounded(curve - 127.0F) + static_cast<float>(exponent + 126));
    return rounded(std::numbers::ln2_v<float> * binary_log);
}
float fast_power2(float value) {
    // Only called by logistic after saturation: |value| <= 20*log2(e).
    int exponent = static_cast<int>(std::floor(value));
    if (value < 0 && value == static_cast<float>(exponent)) --exponent;
    const float fraction = rounded(value - static_cast<float>(exponent));
    const float shape = rounded(0.6602339F + rounded(0.33976606F * fraction));
    const float polynomial = rounded(1.0F + rounded(fraction * shape));
    return std::ldexp(polynomial, exponent);
}
float logistic(float value) {
    if (value > 20.0F) return 1.0F;
    if (value < -20.0F) return 0.0F;
    const float power = fast_power2(rounded(-value * std::numbers::log2e_v<float>));
    return rounded(1.0F / rounded(1.0F + power));
}
bool representable_int(double value) {
    return std::isfinite(value) && value >= std::numeric_limits<int>::min() &&
           value <= std::numeric_limits<int>::max();
}
}  // namespace

Result<UciScore> format_score(ScoreInput input) {
    float q = input.q;
    float draw = input.draw;
    if (!std::isfinite(q) || !std::isfinite(draw) || q < -1 || q > 1 || draw < 0 || draw > 1 ||
        rounded(std::abs(q) + draw) > 1)
        return tl::unexpected(std::string{"UCI score requires finite Q/D probabilities"});

    // Lc0 fd71a2d src/search/classic/search.cc WDLRescale (inverse),
    // SendUciInfo, and src/search/classic/params.cc default ScoreType=WDL_mu.
    // Default ratio=1/difference=0 still reconstructs WDL with fast arithmetic.
    float mu = 0;
    const float win = rounded(rounded(rounded(1.0F + q) - draw) / 2.0F);
    const float loss = rounded(rounded(rounded(1.0F - q) - draw) / 2.0F);
    constexpr float epsilon = 0.0001F;
    constexpr float upper = 1.0F - epsilon;
    if (win > epsilon && draw > epsilon && loss > epsilon && win < upper && draw < upper &&
        loss < upper) {
        const float a = fast_log(rounded(rounded(1.0F / loss) - 1.0F));
        const float b = fast_log(rounded(rounded(1.0F / win) - 1.0F));
        const float denominator = rounded(a + b);
        if (!std::isfinite(denominator) || denominator == 0)
            return tl::unexpected(std::string{"UCI score inverse WDL is not representable"});
        const float width = rounded(2.0F / denominator);
        mu = rounded(rounded(a - b) / denominator);
        if (!std::isfinite(width) || width == 0 || !std::isfinite(mu))
            return tl::unexpected(std::string{"UCI score inverse WDL is not representable"});
        const float new_win = logistic(rounded(rounded(mu - 1.0F) / width));
        const float new_loss = logistic(rounded(rounded(-mu - 1.0F) / width));
        q = rounded(new_win - new_loss);
        draw = std::max(0.0F, rounded(rounded(1.0F - new_win) - new_loss));
    }
    const auto fallback = static_cast<float>(45.0 * std::tan(1.56728071628 * q));
    const float mu_score = rounded(100.0F * mu);
    const bool use_mu = mu != 0 && rounded(std::abs(q) + draw) < 0.996F &&
                        (std::abs(mu) < 1.0F || std::abs(fallback) < std::abs(mu_score));
    const float cp = use_mu ? mu_score : fallback;
    const double rounded_win = std::round(500.0 * (1.0 + static_cast<double>(q) - draw));
    const double rounded_loss = std::round(500.0 * (1.0 - static_cast<double>(q) - draw));
    if (!representable_int(cp) || !representable_int(rounded_win) ||
        !representable_int(rounded_loss))
        return tl::unexpected(std::string{"UCI score integer output is not representable"});

    int w = std::max(0, static_cast<int>(rounded_win));
    int l = std::max(0, static_cast<int>(rounded_loss));
    // Valid reconstructed probabilities bound each component to 0..1000.
    if (w > 1000 || l > 1000)
        return tl::unexpected(std::string{"UCI score WDL output is outside probability range"});
    int d = 1000 - w - l;
    if (d < 0) {
        w = std::clamp(w + d / 2, 0, 1000);
        l = 1000 - w;
        d = 0;
    }
    return UciScore{.centipawns = static_cast<int>(cp), .wdl = {w, d, l}};
}
}  // namespace mouse::uci
