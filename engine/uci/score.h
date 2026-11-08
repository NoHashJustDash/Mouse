// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <array>
#include <cstdint>
#include <optional>
#include <string_view>

#include "engine/util/result.h"

namespace mouse::uci {
// Lc0 ScoreType (classic/params.cc); WDL_mu is its default.
enum class ScoreType : std::uint8_t {
    Centipawn,
    CentipawnWithDrawScore,
    Centipawn2019,
    Centipawn2018,
    WinPercentage,
    Q,
    WL,
    WdlMu
};
[[nodiscard]] std::optional<ScoreType> parse_score_type(std::string_view text) noexcept;
[[nodiscard]] std::string_view score_type_name(ScoreType type) noexcept;
struct ScoreInput {
    float q;
    float draw;
    ScoreType type = ScoreType::WdlMu;
    // Q + DrawScore * D, for the score types that report it. Defaults to q when unset.
    std::optional<float> draw_score_q = std::nullopt;
    // Contempt to take back out of the reported value: WDLRescaleDiff * WDLEvalObjectivity,
    // zero with no contempt side, signed for the root player (lc0 SendUciInfo).
    float rescale_ratio = 1.0F;
    float rescale_diff = 0.0F;
    float rescale_sign = 1.0F;
    float max_s = 1.4F;
};
struct UciScore {
    int centipawns;
    std::array<int, 3> wdl;
};
// Default pinned Lc0 WDL_mu reporting for a WDL value in the root player's
// perspective. Inputs remain unchanged; the inverse reporting transform is local.
// Finite Q in [-1,1], D in [0,1], and float32 abs(Q)+D<=1 are required.
// Terminal mate reporting belongs to the caller; Q endpoints here remain cp.
// Stateless and thread-safe, with checked integer conversions on every output.
[[nodiscard]] Result<UciScore> format_score(ScoreInput input);
}  // namespace mouse::uci
