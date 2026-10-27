// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/uci/commands.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <limits>
#include <set>
#include <span>
#include <utility>

namespace mouse::uci {
namespace {
bool space(char value) {
    return value == ' ' || value == '\t' || value == '\r' || value == '\n' || value == '\v' ||
           value == '\f';
}
char lower(char value) {
    return value >= 'A' && value <= 'Z' ? static_cast<char>(value + ('a' - 'A')) : value;
}
bool keyword(std::string_view token, const char* expected) {
    return std::ranges::equal(token, std::string_view{expected},
                              [](char left, char right) { return lower(left) == right; });
}
std::string_view trim(std::string_view text) {
    while (!text.empty() && space(text.front())) text.remove_prefix(1);
    while (!text.empty() && space(text.back())) text.remove_suffix(1);
    return text;
}
std::vector<std::string_view> tokenize(std::string_view text) {
    std::vector<std::string_view> result;
    while (!(text = trim(text)).empty()) {
        const auto end = std::ranges::find_if(text, space);
        const auto size = static_cast<std::size_t>(end - text.begin());
        result.push_back(text.substr(0, size));
        text.remove_prefix(size);
    }
    return result;
}
std::string join(std::span<const std::string_view> tokens) {
    std::string result;
    for (const auto token : tokens) {
        if (!result.empty()) result += ' ';
        result += token;
    }
    return result;
}
bool square(std::string_view text) {
    return text.size() == 2 && text[0] >= 'a' && text[0] <= 'h' && text[1] >= '1' && text[1] <= '8';
}
bool move_token(std::string_view text) {
    return (text.size() == 4 || text.size() == 5) && square(text.substr(0, 2)) &&
           square(text.substr(2, 2)) &&
           (text.size() == 4 || std::string_view{"qrbn"}.find(text[4]) != std::string_view::npos);
}
struct Bounds {
    std::uint64_t minimum, maximum;
};
Result<std::uint64_t> number(std::string_view text, Bounds bounds, const char* name) {
    std::uint64_t value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || text.size() > 20 || parsed.ec != std::errc{} ||
        parsed.ptr != text.data() + text.size() || value < bounds.minimum || value > bounds.maximum)
        return tl::unexpected(std::string{name} + " requires an integer in " +
                              std::to_string(bounds.minimum) + ".." +
                              std::to_string(bounds.maximum));
    return value;
}
Result<std::uint64_t> remaining_clock(std::string_view text, const char* name) {
    std::int64_t value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || text.size() > 20 || parsed.ec != std::errc{} ||
        parsed.ptr != text.data() + text.size())
        return tl::unexpected(std::string{name} + " requires signed 64-bit integer milliseconds");
    // A GUI may report an overstepped clock. It still requires a valid go reply;
    // only remaining time is clamped, never increments or other search limits.
    return value < 0 ? std::uint64_t{0} : static_cast<std::uint64_t>(value);
}
const char* go_key(std::string_view token) {
    constexpr std::array keys{"nodes", "movetime",    "wtime",     "btime",
                              "winc",  "binc",        "movestogo", "depth",
                              "mate",  "searchmoves", "infinite",  "ponder"};
    for (const auto* key : keys)
        if (keyword(token, key)) return key;
    return nullptr;
}
Result<GoCommand> parse_go(std::span<const std::string_view> tokens) {
    GoCommand command;
    std::set<std::string_view> seen;
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        const auto* key = go_key(tokens[index]);
        if (!key) return tl::unexpected("Unknown go token: " + std::string{tokens[index]});
        const std::string_view option{key};
        if (!seen.insert(option).second)
            return tl::unexpected("Duplicate go token: " + std::string{key});
        if (option == "infinite")
            command.infinite = true;
        else if (option == "ponder")
            command.ponder = true;
        else if (option == "searchmoves") {
            while (index + 1 < tokens.size() && !go_key(tokens[index + 1])) {
                const auto move = tokens[++index];
                if (!move_token(move))
                    return tl::unexpected("Invalid searchmoves token: " + std::string{move});
                if (command.searchmoves.size() == chess::MoveList::kCapacity ||
                    std::ranges::find(command.searchmoves, move) != command.searchmoves.end())
                    return tl::unexpected(std::string{"Duplicate or excessive searchmoves"});
                command.searchmoves.emplace_back(move);
            }
            if (command.searchmoves.empty())
                return tl::unexpected(std::string{"searchmoves requires a move"});
        } else {
            if (index + 1 == tokens.size())
                return tl::unexpected("Missing value after go " + std::string{key});
            const bool clock =
                option == "wtime" || option == "btime" || option == "winc" || option == "binc";
            const auto maximum =
                option == "nodes" || option == "movestogo"
                    ? std::numeric_limits<std::uint64_t>::max()
                : option == "depth" ? std::uint64_t{512}
                : option == "mate"
                    ? std::uint64_t{256}
                    : static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
            const auto token = tokens[++index];
            const auto value =
                option == "wtime" || option == "btime"
                    ? remaining_clock(token, key)
                    : number(token, {.minimum = clock ? 0U : 1U, .maximum = maximum}, key);
            if (!value) return tl::unexpected(value.error());
            if (option == "nodes")
                command.nodes = *value;
            else if (option == "movetime")
                command.movetime = *value;
            else if (option == "wtime")
                command.wtime = *value;
            else if (option == "btime")
                command.btime = *value;
            else if (option == "winc")
                command.winc = *value;
            else if (option == "binc")
                command.binc = *value;
            else if (option == "movestogo")
                command.movestogo = *value;
            else if (option == "depth")
                command.depth = static_cast<std::size_t>(*value);
            else
                command.mate = static_cast<std::size_t>(*value);
        }
    }
    return command;
}
Result<PositionCommand> parse_position(std::span<const std::string_view> tokens) {
    PositionCommand command;
    if (tokens.empty()) return tl::unexpected(std::string{"position requires startpos or fen"});
    std::size_t next = 1;
    if (keyword(tokens[0], "fen")) {
        if (tokens.size() < 7)
            return tl::unexpected(std::string{"position fen requires all six FEN fields"});
        command.fen = join(tokens.subspan(1, 6));
        if (command.fen.size() > 256)
            return tl::unexpected(std::string{"FEN exceeds 256 characters"});
        next = 7;
    } else if (!keyword(tokens[0], "startpos")) {
        return tl::unexpected(std::string{"position requires startpos or fen"});
    }
    if (next < tokens.size()) {
        if (!keyword(tokens[next++], "moves"))
            return tl::unexpected(std::string{"Expected moves after position"});
        if (tokens.size() - next > kMaximumHistoryMoves)
            return tl::unexpected(std::string{"Position history exceeds 131072 plies"});
        for (; next < tokens.size(); ++next) {
            if (!move_token(tokens[next]))
                return tl::unexpected("Invalid position move: " + std::string{tokens[next]});
            command.moves.emplace_back(tokens[next]);
        }
    }
    return command;
}
}  // namespace
Result<Command> parse_command(std::string_view line) {
    if (line.size() > kMaximumLineBytes)
        return tl::unexpected(std::string{"UCI line exceeds 1 MiB"});
    if (std::ranges::any_of(line, [](unsigned char value) {
            return (value < 32 && !space(static_cast<char>(value))) || value == 127;
        }))
        return tl::unexpected(std::string{"UCI line contains a control character"});
    const auto tokens = tokenize(line);
    if (tokens.empty()) return EmptyCommand{};
    if (keyword(tokens[0], "setoption")) {
        if (tokens.size() < 3 || !keyword(tokens[1], "name"))
            return tl::unexpected(std::string{"setoption requires name"});
        std::size_t end = 2;
        while (end < tokens.size() && !keyword(tokens[end], "value")) ++end;
        if (end == 2) return tl::unexpected(std::string{"setoption requires a nonempty name"});
        SetOptionCommand command{.name = join(std::span{tokens}.subspan(2, end - 2)),
                                 .value = std::nullopt};
        if (command.name.size() > 256)
            return tl::unexpected(std::string{"Option name exceeds 256 characters"});
        if (end < tokens.size()) {
            command.value = end + 1 == tokens.size()
                                ? std::string{}
                                : std::string{trim(line.substr(static_cast<std::size_t>(
                                      tokens[end + 1].data() - line.data())))};
        }
        return command;
    }
    if (keyword(tokens[0], "position")) {
        auto command = parse_position(std::span{tokens}.subspan(1));
        if (!command) return tl::unexpected(command.error());
        return std::move(*command);
    }
    if (keyword(tokens[0], "go")) {
        auto command = parse_go(std::span{tokens}.subspan(1));
        if (!command) return tl::unexpected(command.error());
        return std::move(*command);
    }
    if (keyword(tokens[0], "debug")) {
        if (tokens.size() == 2 && (keyword(tokens[1], "on") || keyword(tokens[1], "off")))
            return DebugCommand{.enabled = keyword(tokens[1], "on")};
        return tl::unexpected(std::string{"debug requires on or off"});
    }
    if (tokens.size() != 1)
        return tl::unexpected(std::string{"Unexpected arguments after UCI command"});
    if (keyword(tokens[0], "uci")) return UciCommand{};
    if (keyword(tokens[0], "isready")) return IsReadyCommand{};
    if (keyword(tokens[0], "ucinewgame")) return NewGameCommand{};
    if (keyword(tokens[0], "stop")) return StopCommand{};
    if (keyword(tokens[0], "ponderhit")) return PonderHitCommand{};
    if (keyword(tokens[0], "quit")) return QuitCommand{};
    return tl::unexpected("Unknown UCI command: " + std::string{tokens[0]});
}
}  // namespace mouse::uci
