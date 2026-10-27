// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/nn/encoder.h"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include "engine/chess/attacks.h"
#include "engine/chess/movegen.h"

namespace {
using namespace mouse::chess;
using namespace mouse::nn;
constexpr std::uint64_t kFull = std::numeric_limits<std::uint64_t>::max();
Position parse(std::string_view fen, const AttackTables& attacks) {
    auto position = Position::from_fen(fen, attacks);
    if (!position) FAIL(position.error());
    return *position;
}
void play(Position& position, std::string_view token, const AttackTables& attacks) {
    const auto move = parse_uci_move(position, token, attacks);
    if (!move) FAIL(move.error());
    static_cast<void>(position.make_move(*move, attacks));
}
}  // namespace

TEST_CASE("Classical encoder rejects unsupported inputs without fallback", "[nn][encoder]") {
    const AttackTables attacks;
    const std::array history{HistoryFrame::capture(parse(kStartFen, attacks))};
    CHECK_FALSE(encode_classical({}).has_value());
    CHECK_FALSE(encode_classical(history, 2).has_value());
    CHECK_FALSE(encode_classical(history, 133).has_value());
    CHECK_FALSE(encode_classical(history, 1, static_cast<HistoryFill>(255)).has_value());
    const std::array duplicate{history[0], history[0]};
    CHECK_FALSE(encode_classical(duplicate).has_value());
}

TEST_CASE("Initial classical planes have exact masks and auxiliary values", "[nn][encoder]") {
    const AttackTables attacks;
    const std::array history{HistoryFrame::capture(parse(kStartFen, attacks))};
    const auto encoded = encode_classical(history);
    REQUIRE(encoded.has_value());
    CHECK((*encoded)[0].mask == 0x000000000000ff00ULL);
    CHECK((*encoded)[1].mask == 0x0000000000000042ULL);
    CHECK((*encoded)[2].mask == 0x0000000000000024ULL);
    CHECK((*encoded)[3].mask == 0x0000000000000081ULL);
    CHECK((*encoded)[4].mask == 0x0000000000000008ULL);
    CHECK((*encoded)[5].mask == 0x0000000000000010ULL);
    CHECK((*encoded)[6].mask == 0x00ff000000000000ULL);
    CHECK((*encoded)[11].mask == 0x1000000000000000ULL);
    CHECK((*encoded)[12].mask == 0);
    for (std::size_t plane = 13; plane < 104; ++plane) CHECK(encoded->at(plane).mask == 0);
    for (std::size_t plane = 104; plane < 108; ++plane) CHECK(encoded->at(plane).mask == kFull);
    CHECK((*encoded)[108].mask == 0);
    CHECK((*encoded)[109].mask == kFull);
    CHECK((*encoded)[109].value == 0.0F);
    CHECK((*encoded)[110].mask == 0);
    CHECK((*encoded)[111].mask == kFull);
}

TEST_CASE("Black orientation and missing history use current perspective", "[nn][encoder]") {
    const AttackTables attacks;
    const std::array history{HistoryFrame::capture(
        parse("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 17 1", attacks))};
    CHECK_FALSE(history[0].is_initial_board());
    const auto encoded = encode_classical(history);
    REQUIRE(encoded.has_value());
    CHECK((*encoded)[0].mask == 0xff00ULL);
    CHECK((*encoded)[1].mask == 0x42ULL);
    CHECK((*encoded)[6].mask == 0x00ff000000000000ULL);
    CHECK((*encoded)[13].mask == 0xff00ULL);
    CHECK((*encoded)[108].mask == kFull);
    CHECK((*encoded)[109].value == 17.0F);
}

TEST_CASE("Every historical board retains the current players piece ownership", "[nn][encoder]") {
    const AttackTables attacks;
    auto position = parse(kStartFen, attacks);
    std::vector<HistoryFrame> history{HistoryFrame::capture(position)};
    play(position, "g1f3", attacks);
    history.push_back(HistoryFrame::capture(position));
    play(position, "g8f6", attacks);
    history.push_back(HistoryFrame::capture(position));
    const auto encoded = encode_classical(history);
    REQUIRE(encoded.has_value());
    const auto white_knights = bit(square(1, 0)) | bit(square(5, 2));
    CHECK((*encoded)[1].mask == white_knights);
    CHECK((*encoded)[14].mask == white_knights);
    CHECK((*encoded)[27].mask == 0x42ULL);
    CHECK((*encoded)[7].mask == (bit(square(1, 7)) | bit(square(5, 5))));
    CHECK((*encoded)[20].mask == 0x4200000000000000ULL);
    CHECK((*encoded)[39].mask == 0);
}

TEST_CASE("History fill is explicit and repetitions are input context", "[nn][encoder]") {
    const AttackTables attacks;
    const std::array history{HistoryFrame::capture(parse(kStartFen, attacks), 2)};
    const auto no = encode_classical(history, 1, HistoryFill::No);
    const auto always = encode_classical(history, 1, HistoryFill::Always);
    REQUIRE(no.has_value());
    REQUIRE(always.has_value());
    CHECK((*no)[12].mask == kFull);
    CHECK((*no)[13].mask == 0);
    for (std::size_t step = 0; step < 8; ++step) {
        CHECK(always->at(step * 13).mask == 0xff00ULL);
        CHECK(always->at(step * 13 + 12).mask == kFull);
    }
    auto repeated_clock =
        parse("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 9 1", attacks);
    const std::array clock_history{HistoryFrame::capture(repeated_clock)};
    CHECK(clock_history[0].is_initial_board());
    const auto fen_only = encode_classical(clock_history);
    REQUIRE(fen_only.has_value());
    CHECK((*fen_only)[13].mask == 0);
    CHECK((*fen_only)[109].value == 9.0F);
}

TEST_CASE("Missing EP history reconstructs the preceding double pawn push", "[nn][encoder]") {
    const AttackTables attacks;
    auto position = parse(kStartFen, attacks);
    const auto initial = HistoryFrame::capture(position);
    play(position, "e2e4", attacks);
    const auto current = HistoryFrame::capture(position);
    const std::array fen_history{current};
    const std::array actual_history{initial, current};
    const auto no = encode_classical(fen_history, 1, HistoryFill::No);
    const auto full = encode_classical(actual_history, 1, HistoryFill::No);
    const auto filled = encode_classical(fen_history);
    REQUIRE(no.has_value());
    REQUIRE(full.has_value());
    REQUIRE(filled.has_value());
    CHECK(*no == *full);
    CHECK((*no)[6].mask == ((0x00ff000000000000ULL ^ bit(square(4, 6))) | bit(square(4, 4))));
    CHECK((*no)[19].mask == 0x00ff000000000000ULL);
    CHECK((*no)[26].mask == 0);
    CHECK((*filled)[97].mask == 0x00ff000000000000ULL);
    const std::array black_push{HistoryFrame::capture(
        parse("rnbqkbnr/ppp1pppp/8/3p4/8/8/PPPPPPPP/RNBQKBNR w KQkq d6 0 2", attacks))};
    const auto white = encode_classical(black_push, 1, HistoryFill::No);
    REQUIRE(white.has_value());
    CHECK((*white)[19].mask == 0x00ff000000000000ULL);
}

TEST_CASE("Classical rights and raw rule50 are encoded without scaling", "[nn][encoder]") {
    const AttackTables attacks;
    const std::array history{
        HistoryFrame::capture(parse("4k3/8/8/8/8/8/8/R3K3 w Q - 99 1", attacks))};
    const auto encoded = encode_classical(history);
    REQUIRE(encoded.has_value());
    CHECK((*encoded)[104].mask == kFull);
    CHECK((*encoded)[105].mask == 0);
    CHECK((*encoded)[106].mask == 0);
    CHECK((*encoded)[107].mask == 0);
    CHECK((*encoded)[109].value == 99.0F);
}

TEST_CASE("Dense expansion preserves masks and validates its boundary", "[nn][encoder]") {
    const AttackTables attacks;
    const std::array history{
        HistoryFrame::capture(parse("4k3/8/8/8/8/8/8/4K3 w - - 47 1", attacks))};
    auto encoded = encode_classical(history);
    REQUIRE(encoded.has_value());
    std::array<float, kInputElementCount> dense{};
    REQUIRE(expand_planes(*encoded, dense).has_value());
    CHECK(dense[5U * 64U + 4U] == 1.0F);
    CHECK(dense[5U * 64U + 3U] == 0.0F);
    CHECK(dense[109U * 64U + 63U] == 47.0F);
    CHECK(dense[std::size_t{111} * 64U] == 1.0F);
    CHECK_FALSE(expand_planes(*encoded, std::span<float>{dense}.first(1)).has_value());
    (*encoded)[0].value = std::numeric_limits<float>::quiet_NaN();
    CHECK_FALSE(expand_planes(*encoded, dense).has_value());
}
