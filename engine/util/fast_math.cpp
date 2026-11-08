// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/util/fast_math.h"

#include <cmath>
#include <limits>
#include <numbers>

namespace mouse::util {
namespace {
static_assert(std::numeric_limits<float>::is_iec559 && std::numeric_limits<float>::digits == 24);

float fast_power2(float value) noexcept {
    // Only called by the logistic after saturation: |value| <= 20*log2(e).
    int exponent = static_cast<int>(std::floor(value));
    if (value < 0 && value == static_cast<float>(exponent)) --exponent;
    const float fraction = rounded(value - static_cast<float>(exponent));
    const float shape = rounded(0.6602339F + rounded(0.33976606F * fraction));
    const float polynomial = rounded(1.0F + rounded(fraction * shape));
    return std::ldexp(polynomial, exponent);
}
}  // namespace

float rounded(float value) noexcept {
    volatile const float stored = value;
    return stored;
}
// Original arithmetic implementation of the pinned approximations, using
// frexp/ldexp rather than copying the upstream float-bit manipulation.
float fast_log(float value) noexcept {
    int exponent = 0;
    const float mantissa = std::frexp(value, &exponent);
    const float fraction = rounded(rounded(mantissa * 2.0F) - 1.0F);
    const float shape = rounded(1.3465552F - rounded(0.34655523F * fraction));
    const float curve = rounded(fraction * shape);
    const float binary_log = rounded(rounded(curve - 127.0F) + static_cast<float>(exponent + 126));
    return rounded(std::numbers::ln2_v<float> * binary_log);
}
float fast_logistic(float value) noexcept {
    if (value > 20.0F) return 1.0F;
    if (value < -20.0F) return 0.0F;
    const float power = fast_power2(rounded(-value * std::numbers::log2e_v<float>));
    return rounded(1.0F / rounded(1.0F + power));
}
}  // namespace mouse::util
