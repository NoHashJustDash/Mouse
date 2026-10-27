// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "engine/chess/position.h"
#include "engine/nn/commands.h"
#include "engine/search/search.h"

namespace mouse::search {
struct SearchCommand {
    nn::BackendCommand neural;
    SearchConfig config;
    SearchLimits limits;
    std::string fen{chess::kStartFen};
    std::vector<std::string> moves;
    bool chess960 = false;
};
// Moves-left selection utility defaults on; --no-moves-left-utility disables it.
// Owns all retained data. History accepts at most 131072 UCI moves in 1 MiB;
// legality and FEN consistency are checked before creating a backend.
[[nodiscard]] Result<SearchCommand> parse_search_command(
    std::span<const std::string_view> arguments);
}  // namespace mouse::search
