// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "engine/chess/position.h"
#include "engine/util/result.h"

namespace mouse::uci {
inline constexpr std::size_t kMaximumLineBytes = std::size_t{1024} * 1024;
inline constexpr std::size_t kMaximumHistoryMoves = 131072;
struct EmptyCommand {};
struct UciCommand {};
struct IsReadyCommand {};
struct NewGameCommand {};
struct StopCommand {};
struct PonderHitCommand {};
struct QuitCommand {};
struct DebugCommand {
    bool enabled = false;
};
struct SetOptionCommand {
    std::string name;
    std::optional<std::string> value;
};
struct PositionCommand {
    std::string fen{chess::kStartFen};
    std::vector<std::string> moves;
};
struct GoCommand {
    std::optional<std::uint64_t> nodes, movetime, wtime, btime, winc, binc, movestogo;
    std::optional<std::size_t> depth, mate;
    std::vector<std::string> searchmoves;
    bool infinite = false, ponder = false;
};
using Command = std::variant<EmptyCommand, UciCommand, IsReadyCommand, NewGameCommand, StopCommand,
                             PonderHitCommand, QuitCommand, DebugCommand, SetOptionCommand,
                             PositionCommand, GoCommand>;
// Stateless grammar parsing. Every retained byte is owned; chess legality and
// option/configuration semantics are validated by the session before mutation.
[[nodiscard]] Result<Command> parse_command(std::string_view line);
}  // namespace mouse::uci
