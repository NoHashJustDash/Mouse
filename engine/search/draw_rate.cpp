// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/search/draw_rate.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <string>
#include <utility>

#include "engine/util/fast_math.h"

namespace mouse::search {
namespace {
using util::fast_log;
using util::fast_logistic;
using util::rounded;
// Lc0 fd71a2d src/search/classic/params.cc: WDLDrawRateTarget 0..0.999,
// WDLDrawRateReference 0.001..0.999, WDLMaxS 0..10.
constexpr double kMaximumTarget = 0.999, kMinimumReference = 0.001, kMaximumReference = 0.999;
constexpr double kMaximumS = 10.0;
float scale(float draw_rate) noexcept {
    return rounded(1.0F / std::log(rounded(rounded(1.0F + draw_rate) / rounded(1.0F - draw_rate))));
}
}  // namespace

Result<void> validate_draw_rate_parameters(const DrawRateParameters& parameters) {
    if (!std::isfinite(parameters.target) || parameters.target < 0 ||
        parameters.target > kMaximumTarget)
        return tl::unexpected(std::string{"WDLDrawRateTarget requires 0..0.999"});
    if (!std::isfinite(parameters.reference) || parameters.reference < kMinimumReference ||
        parameters.reference > kMaximumReference)
        return tl::unexpected(std::string{"WDLDrawRateReference requires 0.001..0.999"});
    if (!std::isfinite(parameters.max_s) || parameters.max_s < 0 || parameters.max_s > kMaximumS)
        return tl::unexpected(std::string{"WDLMaxS requires 0..10"});
    if (!std::isfinite(parameters.contempt))
        return tl::unexpected(std::string{"Contempt requires a finite value"});
    if (!std::isfinite(parameters.contempt_max) || parameters.contempt_max < 0 ||
        parameters.contempt_max > 10000)
        return tl::unexpected(std::string{"ContemptMaxValue requires 0..10000"});
    if (!std::isfinite(parameters.attenuation) || parameters.attenuation < -10 ||
        parameters.attenuation > 10)
        return tl::unexpected(std::string{"WDLContemptAttenuation requires -10..10"});
    if (!std::isfinite(parameters.book_exit_bias) || parameters.book_exit_bias < -2 ||
        parameters.book_exit_bias > 2)
        return tl::unexpected(std::string{"WDLBookExitBias requires -2..2"});
    if (!std::isfinite(parameters.calibration_elo) || parameters.calibration_elo < 0 ||
        parameters.calibration_elo > 10000)
        return tl::unexpected(std::string{"WDLCalibrationElo requires 0..10000"});
    if (!std::isfinite(parameters.eval_objectivity) || parameters.eval_objectivity < 0 ||
        parameters.eval_objectivity > 1)
        return tl::unexpected(std::string{"WDLEvalObjectivity requires 0..1"});
    return {};
}

// Original implementation of the pinned behaviour, not copied code:
// Lc0 fd71a2d src/search/classic/params.cc AccurateWDLRescaleParams (ratio only;
// with contempt 0 its diff is 0) and src/search/classic/search.cc WDLRescale.
float draw_rate_ratio(const DrawRateParameters& parameters) noexcept {
    auto target = static_cast<float>(parameters.target);
    // Lc0 lifts accidental tiny positive targets to 0.001.
    if (target > 0.0F && target < 0.001F) target = 0.001F;
    const float reference = scale(static_cast<float>(parameters.reference));
    const float rescaled = target == 0.0F ? reference : scale(target);
    return rounded(rescaled / reference);
}

namespace {
// Lc0 fd71a2d classic/params.cc AccurateWDLRescaleParams: the diff term.
float accurate_diff(const DrawRateParameters& parameters) noexcept {
    auto target = static_cast<float>(parameters.target);
    if (target > 0.0F && target < 0.001F) target = 0.001F;
    const float reference = scale(static_cast<float>(parameters.reference));
    const float rescaled = target == 0.0F ? reference : scale(target);
    const auto bias = static_cast<float>(parameters.book_exit_bias);
    const float low = std::cosh(0.5F * (1.0F - bias) / rescaled);
    const float high = std::cosh(0.5F * (1.0F + bias) / rescaled);
    const auto limit = static_cast<float>(parameters.contempt_max);
    return rescaled / (reference * reference) / (1.0F / (low * low) + 1.0F / (high * high)) *
           std::log(10.0F) / 200.0F *
           std::clamp(static_cast<float>(parameters.contempt), -limit, limit) *
           static_cast<float>(parameters.attenuation);
}
// Lc0 classic/params.cc ConvertRegularToGamePairElo.
float game_pair_elo(float elo) noexcept {
    constexpr float sharpness = 250.0F, midpoint = 2737.0F;
    return elo + 0.5F * sharpness * std::log(1.0F + std::exp((midpoint - elo) / sharpness));
}
// Lc0 classic/params.cc SimplifiedWDLRescaleParams, used when WDLCalibrationElo is set.
WdlRescale simplified_rescale(const DrawRateParameters& parameters) noexcept {
    constexpr float scale_zero = 15.0F, elo_slope = 425.0F, offset = 6.75F;
    const float reference = scale(static_cast<float>(parameters.reference));
    const auto limit = static_cast<float>(parameters.contempt_max);
    const auto calibration = static_cast<float>(parameters.calibration_elo);
    const float opponent =
        calibration - std::clamp(static_cast<float>(parameters.contempt), -limit, limit);
    const float active_elo = game_pair_elo(calibration);
    const float opponent_elo = game_pair_elo(opponent);
    const float scale_active =
        1.0F / (1.0F / scale_zero + std::exp(active_elo / elo_slope - offset));
    const float scale_opponent =
        1.0F / (1.0F / scale_zero + std::exp(opponent_elo / elo_slope - offset));
    const float target = std::sqrt(
        (scale_active * scale_active + scale_opponent * scale_opponent) / 2.0F);
    const float mu_active = -std::log(10.0F) / 200.0F * scale_zero * elo_slope *
                            std::log(1.0F + std::exp(-active_elo / elo_slope + offset) / scale_zero);
    const float mu_opponent =
        -std::log(10.0F) / 200.0F * scale_zero * elo_slope *
        std::log(1.0F + std::exp(-opponent_elo / elo_slope + offset) / scale_zero);
    return {.ratio = target / reference,
            .diff = 1.0F / (reference * reference) * (mu_active - mu_opponent) *
                    static_cast<float>(parameters.attenuation)};
}
}  // namespace

WdlRescale wdl_rescale_parameters(const DrawRateParameters& parameters) noexcept {
    if (parameters.calibration_elo != 0.0) return simplified_rescale(parameters);
    return {.ratio = draw_rate_ratio(parameters), .diff = accurate_diff(parameters)};
}

Result<double> resolve_contempt(std::string_view opponent, std::string_view contempt,
                                double rating_advantage) {
    const auto parse = [](std::string_view text) -> Result<double> {
        double value = 0;
        const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
        if (text.empty() || parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
            !std::isfinite(value))
            return tl::unexpected("Invalid contempt value: " + std::string{text});
        return value;
    };
    const auto lower = [](char c) {
        return c >= 'A' && c <= 'Z' ? static_cast<char>(c + ('a' - 'A')) : c;
    };
    double result = rating_advantage;
    while (!contempt.empty()) {
        const auto comma = contempt.find(',');
        const auto entry = contempt.substr(0, comma);
        contempt = comma == std::string_view::npos ? std::string_view{} : contempt.substr(comma + 1);
        if (entry.empty()) continue;
        const auto equals = entry.find('=');
        if (equals == std::string_view::npos) {
            auto value = parse(entry);
            if (!value) return value;
            result = *value;
            continue;
        }
        const auto name = entry.substr(0, equals);
        const auto number = entry.substr(equals + 1);
        if (number.find('=') != std::string_view::npos)
            return tl::unexpected("Invalid contempt entry: " + std::string{entry});
        const auto found = std::search(opponent.begin(), opponent.end(), name.begin(), name.end(),
                                       [&](char a, char b) { return lower(a) == lower(b); });
        if (found == opponent.end()) continue;
        auto value = parse(number);
        if (!value) return value;
        return *value;
    }
    return result;
}

bool rescale_wdl(float& q, float& draw, const WdlTransform& transform, float* mu) noexcept {
    const float diff = transform.invert ? -transform.diff : transform.diff;
    const float ratio = transform.invert ? rounded(1.0F / transform.ratio) : transform.ratio;
    const float win = rounded(rounded(rounded(1.0F + q) - draw) / 2.0F);
    const float loss = rounded(rounded(rounded(1.0F - q) - draw) / 2.0F);
    constexpr float epsilon = 0.0001F;
    constexpr float upper = 1.0F - epsilon;
    if (!(win > epsilon && draw > epsilon && loss > epsilon && win < upper && draw < upper &&
          loss < upper))
        return false;
    const float a = fast_log(rounded(rounded(1.0F / loss) - 1.0F));
    const float b = fast_log(rounded(rounded(1.0F / win) - 1.0F));
    float width = rounded(2.0F / rounded(a + b));
    if (!transform.invert) width = std::min(transform.max_s, width);
    const float centre = rounded(rounded(a - b) / rounded(a + b));
    float new_width = rounded(width * ratio);
    if (transform.invert) {
        std::swap(width, new_width);
        width = std::min(transform.max_s, width);
    }
    const float new_centre =
        rounded(centre + rounded(rounded(transform.sign * rounded(width * width)) * diff));
    const float new_win = fast_logistic(rounded(rounded(-1.0F + new_centre) / new_width));
    const float new_loss = fast_logistic(rounded(rounded(-1.0F - new_centre) / new_width));
    q = rounded(new_win - new_loss);
    draw = std::max(0.0F, rounded(rounded(1.0F - new_win) - new_loss));
    if (mu) *mu = new_centre;
    return true;
}

bool rescale_wdl(float& q, float& draw, float ratio, float max_s) noexcept {
    const float win = rounded(rounded(rounded(1.0F + q) - draw) / 2.0F);
    const float loss = rounded(rounded(rounded(1.0F - q) - draw) / 2.0F);
    constexpr float epsilon = 0.0001F;
    constexpr float upper = 1.0F - epsilon;
    if (!(win > epsilon && draw > epsilon && loss > epsilon && win < upper && draw < upper &&
          loss < upper))
        return false;
    const float a = fast_log(rounded(rounded(1.0F / loss) - 1.0F));
    const float b = fast_log(rounded(rounded(1.0F / win) - 1.0F));
    const float width = std::min(max_s, rounded(2.0F / rounded(a + b)));
    const float mu = rounded(rounded(a - b) / rounded(a + b));
    const float new_width = rounded(width * ratio);
    const float new_win = fast_logistic(rounded(rounded(-1.0F + mu) / new_width));
    const float new_loss = fast_logistic(rounded(rounded(-1.0F - mu) / new_width));
    q = rounded(new_win - new_loss);
    draw = std::max(0.0F, rounded(rounded(1.0F - new_win) - new_loss));
    return true;
}
}  // namespace mouse::search
