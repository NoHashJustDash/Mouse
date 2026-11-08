// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/uci/score.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <utility>

#include "engine/search/draw_rate.h"
#include "engine/util/fast_math.h"

namespace mouse::uci {
namespace {
using util::fast_log;
using util::fast_logistic;
using util::rounded;
bool representable_int(double value) {
    return std::isfinite(value) && value >= std::numeric_limits<int>::min() &&
           value <= std::numeric_limits<int>::max();
}
constexpr std::array<std::pair<std::string_view, ScoreType>, 8> kScoreTypes{{
    {"centipawn", ScoreType::Centipawn},
    {"centipawn_with_drawscore", ScoreType::CentipawnWithDrawScore},
    {"centipawn_2019", ScoreType::Centipawn2019},
    {"centipawn_2018", ScoreType::Centipawn2018},
    {"win_percentage", ScoreType::WinPercentage},
    {"Q", ScoreType::Q},
    {"W-L", ScoreType::WL},
    {"WDL_mu", ScoreType::WdlMu},
}};
}  // namespace

std::optional<ScoreType> parse_score_type(std::string_view text) noexcept {
    for (const auto& [name, type] : kScoreTypes)
        if (name == text) return type;
    return std::nullopt;
}
std::string_view score_type_name(ScoreType type) noexcept {
    for (const auto& [name, value] : kScoreTypes)
        if (value == type) return name;
    return "WDL_mu";
}

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
    const bool contempt = input.rescale_diff != 0.0F;
    const float draw_score_q = input.draw_score_q.value_or(input.q);
    if (contempt) {
        search::rescale_wdl(q, draw,
                            {.ratio = input.rescale_ratio,
                             .diff = input.rescale_diff,
                             .sign = input.rescale_sign,
                             .invert = true,
                             .max_s = input.max_s},
                            &mu);
    } else if (input.type == ScoreType::WdlMu) {
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
            const float new_win = fast_logistic(rounded(rounded(mu - 1.0F) / width));
            const float new_loss = fast_logistic(rounded(rounded(-mu - 1.0F) / width));
            q = rounded(new_win - new_loss);
            draw = std::max(0.0F, rounded(rounded(1.0F - new_win) - new_loss));
        }
    }
    float cp = 0;
    switch (input.type) {
        case ScoreType::CentipawnWithDrawScore:
            cp = static_cast<float>(90.0 * std::tan(1.5637541897 * draw_score_q));
            break;
        case ScoreType::Centipawn:
            cp = static_cast<float>(90.0 * std::tan(1.5637541897 * q));
            break;
        case ScoreType::Centipawn2019:
            cp = static_cast<float>(295.0 * q / (1.0 - 0.976953126 * std::pow(q, 14)));
            break;
        case ScoreType::Centipawn2018:
            cp = static_cast<float>(290.680623072 * std::tan(1.548090806 * q));
            break;
        case ScoreType::WinPercentage:
            cp = rounded(rounded(q * 5000.0F) + 5000.0F);
            break;
        case ScoreType::Q:
            cp = rounded(draw_score_q * 10000.0F);
            break;
        case ScoreType::WL:
            cp = rounded(q * 10000.0F);
            break;
        case ScoreType::WdlMu: {
            const auto fallback = static_cast<float>(45.0 * std::tan(1.56728071628 * q));
            const float mu_score = rounded(100.0F * mu);
            const bool use_mu = mu != 0 && rounded(std::abs(q) + draw) < 0.996F &&
                                (std::abs(mu) < 1.0F || std::abs(fallback) < std::abs(mu_score));
            cp = use_mu ? mu_score : fallback;
            break;
        }
    }
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
