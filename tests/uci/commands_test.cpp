// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/uci/commands.h"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <string>
#include <variant>

using namespace mouse::uci;
TEST_CASE("UCI framing accepts complete commands and harmless whitespace", "[uci][parser]") {
    CHECK(std::holds_alternative<EmptyCommand>(*parse_command(" \t\r")));
    CHECK(std::holds_alternative<UciCommand>(*parse_command(" UCI ")));
    CHECK(std::holds_alternative<IsReadyCommand>(*parse_command("isready")));
    CHECK(std::holds_alternative<NewGameCommand>(*parse_command("ucinewgame")));
    CHECK(std::holds_alternative<StopCommand>(*parse_command("stop")));
    CHECK(std::holds_alternative<PonderHitCommand>(*parse_command("ponderhit")));
    CHECK(std::holds_alternative<QuitCommand>(*parse_command("quit")));
    CHECK(std::get<DebugCommand>(*parse_command("debug on")).enabled);
    CHECK_FALSE(std::get<DebugCommand>(*parse_command("debug off")).enabled);
}
TEST_CASE("UCI position preserves the complete owning legal-input history", "[uci][parser]") {
    auto parsed = parse_command("position startpos moves e2e4 e7e5 g1f3");
    REQUIRE(parsed);
    const auto& command = std::get<PositionCommand>(*parsed);
    CHECK(command.fen == mouse::chess::kStartFen);
    CHECK(command.moves == std::vector<std::string>{"e2e4", "e7e5", "g1f3"});
    const auto fen = parse_command("position fen 7k/6Q1/5K2/8/8/8/8/8 b - - 0 1");
    REQUIRE(fen);
    CHECK(std::get<PositionCommand>(*fen).fen == "7k/6Q1/5K2/8/8/8/8/8 b - - 0 1");
}
TEST_CASE("UCI option values preserve spaces within paths", "[uci][parser]") {
    std::string input = "setoption name WeightsFile value C:/net  files/T1 model.onnx  ";
    auto result = parse_command(input);
    REQUIRE(result);
    input.assign("replaced");
    const auto& option = std::get<SetOptionCommand>(*result);
    CHECK(option.name == "WeightsFile");
    CHECK(option.value == "C:/net  files/T1 model.onnx");
    const auto empty = parse_command("setoption name WeightsFile value");
    REQUIRE(empty);
    CHECK(std::get<SetOptionCommand>(*empty).value == "");
    const auto button = parse_command("setoption name Clear Hash");
    REQUIRE(button);
    CHECK_FALSE(std::get<SetOptionCommand>(*button).value);
}
TEST_CASE("UCI go owns every bound without an implicit node budget", "[uci][parser]") {
    const auto result = parse_command(
        "go wtime 0 btime 12000 winc 10 binc 20 movestogo 31 nodes 18446744073709551615 "
        "movetime 300 depth 5 mate 3 searchmoves e2e4 d2d4 ponder infinite");
    REQUIRE(result);
    const auto& go = std::get<GoCommand>(*result);
    CHECK(go.wtime == 0);
    CHECK(go.btime == 12000);
    CHECK(go.winc == 10);
    CHECK(go.binc == 20);
    CHECK(go.movestogo == 31);
    CHECK(go.nodes == UINT64_MAX);
    CHECK(go.movetime == 300);
    CHECK(go.depth == 5);
    CHECK(go.mate == 3);
    CHECK(go.searchmoves == std::vector<std::string>{"e2e4", "d2d4"});
    CHECK(go.ponder);
    CHECK(go.infinite);
    CHECK_FALSE(std::get<GoCommand>(*parse_command("go")).nodes);
}
TEST_CASE("UCI malformed commands never become partial valid commands", "[uci][parser]") {
    constexpr std::array invalid{"unknown",
                                 "uci extra",
                                 "isready now",
                                 "stop now",
                                 "quit extra",
                                 "debug",
                                 "debug maybe",
                                 "setoption",
                                 "setoption name",
                                 "setoption value x",
                                 "position",
                                 "position startpos bad",
                                 "position fen incomplete",
                                 "position startpos moves broken",
                                 "go nonsense",
                                 "go nodes 0",
                                 "go nodes -1",
                                 "go nodes 18446744073709551616",
                                 "go movetime 0",
                                 "go btime 9223372036854775808",
                                 "go depth 0",
                                 "go depth 513",
                                 "go mate 0",
                                 "go mate 257",
                                 "go movestogo 0",
                                 "go nodes 3 nodes 4",
                                 "go ponder ponder",
                                 "go infinite infinite",
                                 "go searchmoves",
                                 "go searchmoves e2e4 e2e4",
                                 "go searchmoves e2e4 searchmoves d2d4",
                                 "go nodes",
                                 "go searchmoves 0000",
                                 "position startpos moves e7e8x"};
    for (const auto* input : invalid) {
        INFO(input);
        CHECK_FALSE(parse_command(input));
    }
    CHECK_FALSE(parse_command(std::string(kMaximumLineBytes + 1, ' ')));
    CHECK_FALSE(parse_command(std::string{"uci\0quit", 8}));
    CHECK_FALSE(parse_command(std::string{"uci\x1b", 4}));
}
TEST_CASE("UCI negative remaining clocks are valid bounded zero clocks",
          "[uci][parser][clock-regression]") {
    for (const auto* token : {"-20", "-1", "-0", "-9223372036854775808"}) {
        INFO(token);
        const auto result = parse_command(std::string{"go wtime "} + token + " btime " + token +
                                          " winc 100 binc 100");
        REQUIRE(result);
        const auto& go = std::get<GoCommand>(*result);
        CHECK(go.wtime == 0);
        CHECK(go.btime == 0);
        CHECK(go.winc == 100);
        CHECK(go.binc == 100);
    }
    const auto mixed = parse_command("go wtime -20 btime 1000 winc 100 binc 100");
    REQUIRE(mixed);
    CHECK(std::get<GoCommand>(*mixed).wtime == 0);
    CHECK(std::get<GoCommand>(*mixed).btime == 1000);
}
TEST_CASE("UCI remaining clock clamping preserves syntax and overflow rejection",
          "[uci][parser][clock-regression]") {
    constexpr std::array invalid{"go wtime -",
                                 "go btime --1",
                                 "go wtime -+1",
                                 "go btime +1",
                                 "go wtime -1.5",
                                 "go btime -1ms",
                                 "go wtime -9223372036854775809",
                                 "go btime -18446744073709551615",
                                 "go wtime 9223372036854775808",
                                 "go btime 18446744073709551616",
                                 "go wtime -00000000000000000001",
                                 "go winc -1",
                                 "go binc -1",
                                 "go movetime -1",
                                 "go nodes -1",
                                 "go wtime -1 wtime 3",
                                 "go btime -1 btime -2"};
    for (const auto* input : invalid) {
        INFO(input);
        CHECK_FALSE(parse_command(input));
    }
    const auto boundary = parse_command("go wtime 9223372036854775807 btime 9223372036854775807");
    REQUIRE(boundary);
    CHECK(std::get<GoCommand>(*boundary).wtime == 9223372036854775807ULL);
    CHECK(std::get<GoCommand>(*boundary).btime == 9223372036854775807ULL);
}
