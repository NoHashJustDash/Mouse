// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/chess/position.h"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <charconv>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>

#include "engine/chess/attacks.h"
#include "engine/chess/movegen.h"
#include "engine/chess/perft.h"

using namespace mouse::chess;

TEST_CASE("Initial position round-trips and has the published perft tree", "[chess]") {
    const AttackTables attacks;
    auto parsed = Position::from_fen(kStartFen, attacks);
    REQUIRE(parsed.has_value());
    auto position = std::move(*parsed);
    CHECK(position.fen() == kStartFen);
    CHECK(position.key() == position.recompute_key(attacks));
    CHECK(legal_moves(position, attacks).size() == 20);
    CHECK(perft(position, attacks, 0) == 1);
    CHECK(perft(position, attacks, 1) == 20);
    CHECK(perft(position, attacks, 2) == 400);
    CHECK(perft(position, attacks, 3) == 8902);
    CHECK(perft(position, attacks, 4) == 197281);
}

TEST_CASE("Malformed and locally illegal FENs return reasons", "[chess]") {
    const AttackTables attacks;
    const std::array invalid{"",
                             "8/8/8/8/8/8/8/8 w - - 0 1",
                             "4k3/8/8/8/8/8/8/4K3 x - - 0 1",
                             "4k3/8/8/8/8/8/8/4K3 w - - -1 1",
                             "4k3/8/8/8/8/8/8/4K3 w - - 0 0",
                             "4k3/8/8/8/8/8/8/4K3 w K - 0 1",
                             "4k3/8/8/8/8/8/8/4K3 w - e6 0 1",
                             "4k3/8/8/8/8/8/4R3/K7 w - - 0 1",
                             "4k3/8/8/8/8/8/4K3/8 w - - 0 1 extra",
                             "4k3/8/8/8/8/8/4K3/P7 w - - 0 1"};
    for (const auto* fen : invalid) {
        INFO(fen);
        const auto parsed = Position::from_fen(fen, attacks);
        REQUIRE_FALSE(parsed.has_value());
        CHECK_FALSE(parsed.error().empty());
    }
}

TEST_CASE("Every initial move restores board, counters, history and hash", "[chess]") {
    const AttackTables attacks;
    auto parsed = Position::from_fen(kStartFen, attacks);
    REQUIRE(parsed.has_value());
    auto position = std::move(*parsed);
    const Position before = position;
    for (const Move move : legal_moves(position, attacks)) {
        const auto undo = position.make_move(move, attacks);
        CHECK(position.key() == position.recompute_key(attacks));
        CHECK(Position::from_fen(position.fen(), attacks).has_value());
        position.unmake_move(move, undo);
        CHECK(position == before);
    }
}

TEST_CASE("En passant discovered check is illegal and does not alter repetition identity",
          "[chess]") {
    const AttackTables attacks;
    auto pinned = Position::from_fen("k7/8/8/r4pPK/8/8/8/8 w - f6 0 1", attacks);
    auto no_ep = Position::from_fen("k7/8/8/r4pPK/8/8/8/8 w - - 0 1", attacks);
    REQUIRE(pinned.has_value());
    REQUIRE(no_ep.has_value());
    CHECK_FALSE(parse_uci_move(*pinned, "g5f6", attacks).has_value());
    CHECK(pinned->key() == no_ep->key());
    CHECK(pinned->fen() == "k7/8/8/r4pPK/8/8/8/8 w - f6 0 1");
    auto legal = Position::from_fen("k7/8/8/5pPK/8/8/8/8 w - f6 0 1", attacks);
    auto legal_no_ep = Position::from_fen("k7/8/8/5pPK/8/8/8/8 w - - 0 1", attacks);
    REQUIRE(legal.has_value());
    REQUIRE(legal_no_ep.has_value());
    CHECK(parse_uci_move(*legal, "g5f6", attacks).has_value());
    CHECK(legal->key() != legal_no_ep->key());
}

TEST_CASE("Castling is stored king to rook and restores overlapping Chess960 squares", "[chess]") {
    const AttackTables attacks;
    const std::array cases{std::pair{"4k3/8/8/8/8/8/8/4K2R w K - 7 12", "e1h1"},
                           std::pair{"4k3/8/8/8/8/8/8/6KR w H - 7 12", "g1h1"},
                           std::pair{"4k3/8/8/8/8/8/8/4KR2 w F - 7 12", "e1f1"},
                           std::pair{"4k3/8/8/8/8/8/8/5KR1 w G - 7 12", "f1g1"},
                           std::pair{"4k3/8/8/8/8/8/8/2RK4 w C - 7 12", "d1c1"}};
    for (const auto& [fen, uci] : cases) {
        INFO(fen);
        auto parsed = Position::from_fen(fen, attacks);
        REQUIRE(parsed.has_value());
        const Position before = *parsed;
        const auto move = parse_uci_move(*parsed, uci, attacks, true);
        REQUIRE(move.has_value());
        REQUIRE(move->kind() == MoveKind::Castle);
        const auto undo = parsed->make_move(*move, attacks);
        CHECK(parsed->key() == parsed->recompute_key(attacks));
        CHECK(parsed->piece_at(square(move->to() > move->from() ? 6 : 2, 0)).type ==
              PieceType::King);
        parsed->unmake_move(*move, undo);
        CHECK(*parsed == before);
    }
}

TEST_CASE("Promotions and en passant captures restore captured state", "[chess]") {
    const AttackTables attacks;
    const std::array cases{std::pair{"4k2r/6P1/8/8/8/8/8/4K3 w k - 0 1", "g7h8n"},
                           std::pair{"4k3/8/8/4pP2/8/8/8/4K3 w - e6 0 1", "f5e6"}};
    for (const auto& [fen, uci] : cases) {
        auto parsed = Position::from_fen(fen, attacks);
        REQUIRE(parsed.has_value());
        const Position before = *parsed;
        const auto move = parse_uci_move(*parsed, uci, attacks);
        REQUIRE(move.has_value());
        const auto undo = parsed->make_move(*move, attacks);
        CHECK(parsed->key() == parsed->recompute_key(attacks));
        CHECK(parsed->halfmove_clock() == 0);
        if (move->is_promotion()) CHECK(parsed->castling_rook(Color::Black, true) == kNoSquare);
        parsed->unmake_move(*move, undo);
        CHECK(*parsed == before);
    }
}

TEST_CASE("Repetition requires equal legal options and supports the tree twofold threshold",
          "[chess]") {
    const AttackTables attacks;
    auto parsed = Position::from_fen(kStartFen, attacks);
    REQUIRE(parsed.has_value());
    for (int cycle = 0; cycle < 2; ++cycle) {
        for (const auto uci : {"g1f3", "g8f6", "f3g1", "f6g8"}) {
            auto move = parse_uci_move(*parsed, uci, attacks);
            REQUIRE(move.has_value());
            (void)parsed->make_move(*move, attacks);
        }
        CHECK(parsed->is_repetition(2));
        CHECK(parsed->is_repetition(3) == (cycle == 1));
    }
}

TEST_CASE("Game status gives checkmate precedence over the fifty-move clock", "[chess]") {
    const AttackTables attacks;
    auto mate = Position::from_fen("7k/6Q1/5K2/8/8/8/8/8 b - - 100 50", attacks);
    auto stale = Position::from_fen("7k/5K2/6Q1/8/8/8/8/8 b - - 0 50", attacks);
    auto dead = Position::from_fen("7k/8/5K2/8/8/8/8/8 w - - 0 1", attacks);
    REQUIRE(mate.has_value());
    REQUIRE(stale.has_value());
    REQUIRE(dead.has_value());
    CHECK(game_status(*mate, attacks) == GameStatus::Checkmate);
    CHECK(game_status(*stale, attacks) == GameStatus::Stalemate);
    CHECK(game_status(*dead, attacks) == GameStatus::DrawInsufficientMaterial);
}

TEST_CASE("Perft rejects depth errors and reports count overflow", "[chess]") {
    const AttackTables attacks;
    auto position = Position::from_fen(kStartFen, attacks);
    REQUIRE(position.has_value());
    const Position before = *position;
    CHECK_FALSE(perft(*position, attacks, -1).has_value());
    CHECK_FALSE(perft(*position, attacks, kMaxPerftDepth + 1).has_value());
    CHECK_FALSE(perft_divide(*position, attacks, 0).has_value());
    const auto max = std::numeric_limits<std::uint64_t>::max();
    CHECK(detail::checked_node_sum(max - 1, 1) == max);
    CHECK_FALSE(detail::checked_node_sum(max, 1).has_value());
    const auto divided = perft_divide(*position, attacks, 2);
    REQUIRE(divided.has_value());
    REQUIRE(divided->size() == 20);
    std::uint64_t total = 0;
    for (const auto& entry : *divided) total += entry.nodes;
    CHECK(total == 400);
    CHECK(*position == before);
}

TEST_CASE("Counter range and overflow preserve the original position at every recursion level",
          "[chess]") {
    const AttackTables attacks;
    const std::string near_limit = "7k/8/8/8/8/8/8/K7 b - - 0 18446744073709551614";
    auto parsed = Position::from_fen(near_limit, attacks);
    REQUIRE(parsed.has_value());
    const Position before = *parsed;
    const auto overflow = perft(*parsed, attacks, 3);
    REQUIRE_FALSE(overflow.has_value());
    CHECK(overflow.error().find("counter") != std::string::npos);
    CHECK(*parsed == before);
    CHECK_FALSE(perft_divide(*parsed, attacks, 3).has_value());
    CHECK(*parsed == before);
    auto maximum = Position::from_fen("7k/8/8/8/8/8/8/K7 b - - 0 18446744073709551615", attacks);
    REQUIRE(maximum.has_value());
    const Position before_maximum = *maximum;
    CHECK_THROWS_AS(maximum->make_move(Move{square(7, 7), square(6, 7)}, attacks),
                    std::overflow_error);
    CHECK(*maximum == before_maximum);
    CHECK_FALSE(parse_uci_move(*maximum, "h8g8", attacks).has_value());
    auto halfmove = Position::from_fen("7k/8/8/8/8/8/8/K7 w - - 18446744073709551615 1", attacks);
    REQUIRE(halfmove.has_value());
    const Position before_halfmove = *halfmove;
    CHECK_THROWS_AS(halfmove->make_move(Move{square(0, 0), square(1, 0)}, attacks),
                    std::overflow_error);
    CHECK(*halfmove == before_halfmove);
    CHECK_FALSE(
        Position::from_fen("7k/8/8/8/8/8/8/K7 w - - 18446744073709551616 1", attacks).has_value());
}

TEST_CASE("FEN promotion budget includes the square colors of bishops", "[chess]") {
    const AttackTables attacks;
    for (const auto fen :
         {"7k/8/8/8/8/8/PPPPPPPP/3QKQ2 w - - 0 1", "7k/8/8/8/8/4B3/PPPPPPPP/2B1K3 w - - 0 1"}) {
        const auto parsed = Position::from_fen(fen, attacks);
        REQUIRE_FALSE(parsed.has_value());
        CHECK_FALSE(parsed.error().empty());
    }
    CHECK(Position::from_fen("7k/8/8/8/8/4B3/1PPPPPPP/2B1K3 w - - 0 1", attacks).has_value());
}

TEST_CASE("Castling rejects attacked transit squares and newly exposed Chess960 lines", "[chess]") {
    const AttackTables attacks;
    auto transit = Position::from_fen("4kr2/8/8/8/8/8/8/4K2R w K - 0 1", attacks);
    auto pinned_rook = Position::from_fen("4k3/8/8/8/8/8/8/rRK5 w B - 0 1", attacks);
    REQUIRE(transit.has_value());
    REQUIRE(pinned_rook.has_value());
    CHECK_FALSE(parse_uci_move(*transit, "e1g1", attacks).has_value());
    CHECK_FALSE(parse_uci_move(*pinned_rook, "c1b1", attacks, true).has_value());
    auto standard = Position::from_fen("4k3/8/8/8/8/8/8/4K2R w K - 0 1", attacks);
    REQUIRE(standard.has_value());
    auto move = parse_uci_move(*standard, "e1g1", attacks);
    REQUIRE(move.has_value());
    CHECK(move_to_uci(*move) == "e1g1");
    CHECK(move_to_uci(*move, true) == "e1h1");
    auto outer = Position::from_fen("4k3/8/8/8/8/8/8/4KR1R w H - 0 1", attacks);
    auto inner = Position::from_fen("4k3/8/8/8/8/8/8/4KR1R w F - 0 1", attacks);
    REQUIRE(outer.has_value());
    REQUIRE(inner.has_value());
    CHECK(outer->key() != inner->key());
}

namespace {
struct FuzzShard {
    std::uint64_t index = 0;
    std::uint64_t count = 1;
    std::uint64_t begin = 0;
    std::uint64_t end = 1'000'000;
};
struct FuzzSettings {
    std::optional<std::string_view> index_text;
    std::optional<std::string_view> count_text;
};
mouse::Result<FuzzShard> parse_fuzz_shard(FuzzSettings settings) {
    auto parse = [](std::optional<std::string_view> text,
                    std::uint64_t fallback) -> mouse::Result<std::uint64_t> {
        if (!text) return fallback;
        if (text->empty())
            return tl::unexpected(std::string{"Fuzz shard settings cannot be empty"});
        std::uint64_t value = 0;
        const auto parsed =
            std::from_chars(std::to_address(text->begin()), std::to_address(text->end()), value);
        if (parsed.ec != std::errc{} || parsed.ptr != std::to_address(text->end()) || text->empty())
            return tl::unexpected(
                std::string{"Fuzz shard settings must be unsigned decimal integers"});
        return value;
    };
    const auto index_value = parse(settings.index_text, 0);
    const auto count_value = parse(settings.count_text, 1);
    if (!index_value) return tl::unexpected(index_value.error());
    if (!count_value) return tl::unexpected(count_value.error());
    if (*count_value < 1 || *count_value > 16 || *index_value >= *count_value)
        return tl::unexpected(
            std::string{"Fuzz shard count must be 1..16 and index must be less than count"});
    constexpr std::uint64_t total = 1'000'000;
    return FuzzShard{.index = *index_value,
                     .count = *count_value,
                     .begin = total * *index_value / *count_value,
                     .end = total * (*index_value + 1) / *count_value};
}
struct FuzzResult {
    std::uint64_t completed = 0;
    std::uint64_t plies = 0;
};
FuzzResult run_complete_games(std::uint64_t begin, std::uint64_t end) {
    const AttackTables attacks;
    FuzzResult result;
    for (auto game = begin; game < end; ++game) {
        // Each game is independently reproducible regardless of shard count/order.
        std::mt19937_64 random{0x6d6f757365ULL ^ (game * 0x9e3779b97f4a7c15ULL)};
        auto parsed = Position::from_fen(kStartFen, attacks);
        REQUIRE(parsed.has_value());
        auto position = std::move(*parsed);
        const Position initial = position;
        std::vector<std::pair<Move, Undo>> history;
        while (game_status(position, attacks) == GameStatus::Ongoing) {
            INFO("game " << game << " ply " << history.size());
            const auto moves = legal_moves(position, attacks);
            REQUIRE_FALSE(moves.empty());
            const Move move =
                moves[std::uniform_int_distribution<std::size_t>{0, moves.size() - 1}(random)];
            const Position before = position;
            const Undo undo = position.make_move(move, attacks);
            CHECK(position.key() == position.recompute_key(attacks));
            const auto roundtrip = Position::from_fen(position.fen(), attacks);
            REQUIRE(roundtrip.has_value());
            CHECK(roundtrip->fen() == position.fen());
            CHECK(roundtrip->key() == position.key());
            position.unmake_move(move, undo);
            REQUIRE(position == before);
            history.emplace_back(move, position.make_move(move, attacks));
            ++result.plies;
        }
        while (!history.empty()) {
            position.unmake_move(history.back().first, history.back().second);
            history.pop_back();
        }
        CHECK(position == initial);
        ++result.completed;
    }
    return result;
}
}  // namespace

TEST_CASE("Complete random games preserve hash, FEN and reversible state", "[chess][property]") {
    const auto result = run_complete_games(0, 200);
    CHECK(result.completed == 200);
}

TEST_CASE("Million complete random games preserve all position invariants",
          "[chess][property-long][.long]") {
    const auto environment = [](const char* name) -> std::optional<std::string_view> {
        if (const char* value = std::getenv(name)) return value;
        return std::nullopt;
    };
    const auto shard = parse_fuzz_shard({.index_text = environment("MOUSE_FUZZ_SHARD_INDEX"),
                                         .count_text = environment("MOUSE_FUZZ_SHARD_COUNT")});
    if (!shard) FAIL(shard.error());
    const auto result = run_complete_games(shard->begin, shard->end);
    REQUIRE(result.completed == shard->end - shard->begin);
    std::cout << "MOUSE_FUZZ_RESULT total=1000000 shard=" << shard->index
              << " shards=" << shard->count << " begin=" << shard->begin << " end=" << shard->end
              << " completed=" << result.completed << " plies=" << result.plies << '\n';
}

TEST_CASE("Fuzz shards reject malformed settings and partition the fixed million games",
          "[chess]") {
    const auto defaults = parse_fuzz_shard({});
    REQUIRE(defaults.has_value());
    CHECK(defaults->begin == 0);
    CHECK(defaults->end == 1'000'000);
    for (const auto& [index_text, count_text] : std::array{
             std::pair{"0", "0"}, std::pair{"0", "17"}, std::pair{"-1", "2"}, std::pair{"2", "2"},
             std::pair{"x", "2"}, std::pair{"0", ""}, std::pair{"", "2"}, std::pair{"0", "1x"}}) {
        const auto shard = parse_fuzz_shard({.index_text = index_text, .count_text = count_text});
        CHECK_FALSE(shard.has_value());
    }
    for (std::uint64_t count = 1; count <= 16; ++count) {
        std::uint64_t previous_end = 0;
        std::uint64_t total = 0;
        for (std::uint64_t index_value = 0; index_value < count; ++index_value) {
            const auto index_text = std::to_string(index_value);
            const auto count_text = std::to_string(count);
            const auto shard =
                parse_fuzz_shard({.index_text = index_text, .count_text = count_text});
            REQUIRE(shard.has_value());
            CHECK(shard->begin == previous_end);
            total += shard->end - shard->begin;
            previous_end = shard->end;
        }
        CHECK(total == 1'000'000);
        CHECK(previous_end == 1'000'000);
    }
}

TEST_CASE("FEN rejects more than two individual checking pieces", "[chess]") {
    const AttackTables attacks;
    const auto parsed = Position::from_fen("k3r3/8/8/8/1b5b/8/8/4K3 w - - 0 1", attacks);
    REQUIRE_FALSE(parsed.has_value());
    CHECK(parsed.error().find("check") != std::string::npos);
}

TEST_CASE("An asserted en-passant push cannot follow an already checked moving king", "[chess]") {
    const AttackTables attacks;
    const auto knight = Position::from_fen("k7/8/8/3p4/8/5n2/8/4K3 w - d6 0 2", attacks);
    REQUIRE_FALSE(knight.has_value());
    CHECK(knight.error().find("en-passant") != std::string::npos);
    const auto pawn_origin = Position::from_fen("k7/8/2K5/3p4/8/8/8/8 w - d6 0 2", attacks);
    CHECK_FALSE(pawn_origin.has_value());
    CHECK(Position::from_fen("k7/8/8/3p4/4K3/8/8/8 w - d6 0 2", attacks).has_value());
    CHECK(Position::from_fen("k1b5/8/8/3p4/6K1/8/8/8 w - d6 0 2", attacks).has_value());
    CHECK(Position::from_fen("8/8/8/3k4/4P3/8/8/K7 b - e3 0 27", attacks).has_value());
}

TEST_CASE("Orthodox UCI mode rejects nonstandard and ambiguous castling tokens", "[chess]") {
    const AttackTables attacks;
    auto parsed = Position::from_fen("4k3/8/8/8/8/8/8/5K1R w H - 0 1", attacks);
    REQUIRE(parsed.has_value());
    CHECK_FALSE(parse_uci_move(*parsed, "f1g1", attacks, false).has_value());
    auto step = parse_uci_move(*parsed, "f1g1", attacks, true);
    auto castle = parse_uci_move(*parsed, "f1h1", attacks, true);
    REQUIRE(step.has_value());
    REQUIRE(castle.has_value());
    CHECK(step->kind() == MoveKind::Normal);
    CHECK(castle->kind() == MoveKind::Castle);
    CHECK(move_to_uci(*step, true) == "f1g1");
    CHECK(move_to_uci(*castle, true) == "f1h1");
    CHECK_THROWS_AS(move_to_uci(*castle, false), std::invalid_argument);
    auto stationary = Position::from_fen("4k3/8/8/8/8/8/8/6KR w H - 0 1", attacks);
    REQUIRE(stationary.has_value());
    CHECK_FALSE(parse_uci_move(*stationary, "g1g1", attacks, false).has_value());
}
