// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <optional>

#include "engine/search/search.h"
namespace mouse::uci {
// Roots may come from a later observation than the authoritative result/PV.
// Keep its chosen move first without changing search or emergency selection.
void rank_report_roots(search::RootStatistics&, std::optional<chess::Move> preferred);
}  // namespace mouse::uci
