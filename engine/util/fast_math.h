// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

namespace mouse::util {
// Lc0's float32 approximations, kept bit-exact so that values derived from them
// (UCI scores and draw-rate rescaled WDL) match the reference executable.
// Reference: Lc0 fd71a2d src/utils/fastmath.h FastLog/FastExp2/FastLogistic.

// Stores through a volatile float so compilers keep each float32 boundary and
// cannot contract a multiply and add into one fused, differently rounded step.
[[nodiscard]] float rounded(float value) noexcept;
// FastLog, for positive finite input.
[[nodiscard]] float fast_log(float value) noexcept;
// FastLogistic: 1 / (1 + 2^(-value*log2(e))), saturated outside [-20, 20].
[[nodiscard]] float fast_logistic(float value) noexcept;
}  // namespace mouse::util
