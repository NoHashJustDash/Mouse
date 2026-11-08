// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <string_view>

#include "engine/util/result.h"

namespace mouse::search {
// WDL rescaling of the network's value before it enters the tree: Lc0's WDLDrawRateTarget,
// WDLDrawRateReference, WDLMaxS, the contempt settings and WDLCalibrationElo.
// A target of 0 keeps the raw WDL. target and reference are Mouse's tuned defaults (see
// SelectionParameters); Lc0's are 0 (off) and 0.5. The others carry Lc0's defaults.
struct DrawRateParameters {
    double target = 0.779;
    double reference = 0.75;
    double max_s = 1.4;
    // Resolved Elo advantage, from Contempt, UCI_Opponent and UCI_RatingAdv.
    double contempt = 0.0;
    double contempt_max = 420.0;      // ContemptMaxValue 0..10000
    double attenuation = 1.0;         // WDLContemptAttenuation -10..10
    double book_exit_bias = 0.65;     // WDLBookExitBias -2..2
    double calibration_elo = 0.0;     // WDLCalibrationElo 0..10000, 0 = use the draw-rate target
    double eval_objectivity = 1.0;    // WDLEvalObjectivity 0..1, used only for reported scores
    bool operator==(const DrawRateParameters&) const = default;
};
[[nodiscard]] Result<void> validate_draw_rate_parameters(const DrawRateParameters& parameters);
// Lc0 BaseSearchParams::WDLRescaleParams.
struct WdlRescale {
    float ratio = 1.0F;
    float diff = 0.0F;
};
[[nodiscard]] WdlRescale wdl_rescale_parameters(const DrawRateParameters& parameters) noexcept;
// Width ratio of the rescaled logistic WDL; exactly 1 when target is 0.
[[nodiscard]] float draw_rate_ratio(const DrawRateParameters& parameters) noexcept;
// Rescales side-to-move Q (W-L) and D in place. Returns false, leaving them
// unchanged, when a component is too close to 0 or 1 to rescale safely.
bool rescale_wdl(float& q, float& draw, float ratio, float max_s) noexcept;
// The full transform of Lc0 search.cc WDLRescale. sign is +1 when q's side holds the contempt.
// invert undoes the diff for reported scores and keeps the width. Writes the rescaled mu when
// a transform happened and leaves it alone otherwise.
struct WdlTransform {
    float ratio = 1.0F;
    float diff = 0.0F;
    float sign = 1.0F;
    bool invert = false;
    float max_s = 1.4F;
};
bool rescale_wdl(float& q, float& draw, const WdlTransform& transform, float* mu = nullptr) noexcept;
// Lc0 classic/params.cc GetContempt: comma separated [name=]value entries. A bare value replaces
// the UCI_RatingAdv default; the first entry whose name occurs in the opponent's name (ignoring
// case) wins. Malformed text is an error, never a silent 0.
[[nodiscard]] Result<double> resolve_contempt(std::string_view opponent, std::string_view contempt,
                                              double rating_advantage);
}  // namespace mouse::search
