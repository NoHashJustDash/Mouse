// SPDX-License-Identifier: GPL-3.0-or-later
#include <algorithm>
#include <array>
#include <bit>
#include <catch2/catch_test_macros.hpp>
#include <stdexcept>
#include <string_view>

#include "engine/chess/attacks.h"
#include "engine/chess/movegen.h"
#include "engine/nn/policy_map.h"
#include "engine/search/search.h"
#include "tests/required_optional.h"

namespace {
using mouse::search::ProofOutcome;
using mouse::tests::required_optional;
using enum ProofOutcome;
struct UniformBackend final : mouse::nn::Backend {
    mouse::nn::ModelMetadata model;
    mouse::nn::BackendInfo identity;
    std::array<std::uint64_t, 8192> trace{};
    std::array<std::size_t, 8192> batches{};
    std::size_t rows = 0, calls = 0;
    bool endpoints = false, cycle = false;
    UniformBackend() {
        model.input_format = 1;
        identity.onnx_sha256 = "proof-uniform";
    }
    mouse::Result<void> evaluate(std::span<const mouse::nn::InputPlanes> inputs,
                                 std::span<mouse::nn::NNOutput> outputs) override {
        batches.at(calls++) = inputs.size();
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            outputs[i] = {};
            outputs[i].wdl =
                endpoints ? std::array{1.0F, 0.0F, 0.0F} : std::array{0.6F, 0.2F, 0.2F};
            if (cycle) {
                outputs[i].policy_logits.fill(-100.0F);
                for (const auto move :
                     {mouse::chess::Move{mouse::chess::square(1, 0), mouse::chess::square(2, 2)},
                      mouse::chess::Move{mouse::chess::square(2, 2), mouse::chess::square(1, 0)}}) {
                    const auto index =
                        mouse::nn::PolicyMap::index(move, mouse::chess::Color::White);
                    if (!index) return tl::unexpected(index.error());
                    outputs[i].policy_logits.at(*index) = 100.0F;
                }
            }
            std::uint64_t hash = 1469598103934665603ULL;
            for (const auto& plane : inputs[i]) {
                hash = (hash ^ plane.mask) * 1099511628211ULL;
                hash = (hash ^ std::bit_cast<std::uint32_t>(plane.value)) * 1099511628211ULL;
            }
            trace.at(rows++) = hash;
        }
        return {};
    }
    [[nodiscard]] const mouse::nn::BackendInfo& info() const noexcept override { return identity; }
    [[nodiscard]] const mouse::nn::ModelMetadata& metadata() const noexcept override {
        return model;
    }
};
mouse::chess::Position position(std::string_view fen, const mouse::chess::AttackTables& attacks) {
    auto root = mouse::chess::Position::from_fen(fen, attacks);
    if (!root) throw std::runtime_error(root.error());
    return std::move(*root);
}
struct Fixture {
    const mouse::chess::AttackTables attacks;
    UniformBackend backend;
    mouse::chess::Position root;
    mouse::nn::EncodingHistory history;
    std::unique_ptr<mouse::search::SearchEngine> engine;
    explicit Fixture(std::string_view fen = mouse::chess::kStartFen, bool enabled = true,
                     std::size_t depth = 32)
        : root(position(fen, attacks)) {
        history.reset(root);
        mouse::search::SearchConfig config;
        config.threads = 1;
        config.maximum_batch = 4;
        config.max_depth = depth;
        config.ram_limit_bytes = 8U << 20U;
        config.deterministic = true;
        config.terminal_proof = enabled;
        auto created = mouse::search::SearchEngine::create(config, attacks, backend);
        if (!created) throw std::runtime_error(created.error());
        engine = std::move(*created);
        auto configured = engine->set_position(root, history);
        if (!configured) throw std::runtime_error(configured.error());
    }
    mouse::search::SearchResult run(std::uint64_t nodes) {
        mouse::search::SearchLimits limit;
        // Exhaustively verify proof propagation/reuse at the prescribed visit
        // count. Source-compatible early win pruning is tested separately.
        limit.smart_pruning = false;
        limit.nodes = nodes;
        const auto result = engine->run(limit);
        if (!result) throw std::runtime_error(result.error());
        return *result;
    }
};
mouse::chess::MoveList mate_in_one(mouse::chess::Position root,
                                   const mouse::chess::AttackTables& attacks) {
    mouse::chess::MoveList result;
    const auto legal = mouse::chess::legal_moves(root, attacks);
    for (const auto move : legal) {
        const auto undo = root.make_move(move, attacks);
        if (mouse::chess::legal_moves(root, attacks).empty() && root.in_check(attacks))
            result.push_back(move);
        root.unmake_move(move, undo);
    }
    return result;
}
constexpr std::string_view kMate = "7k/8/5KQ1/8/8/8/8/8 w - - 0 1";
}  // namespace

TEST_CASE("Terminal proof replaces diluted means while retaining real child visits",
          "[search][proof][proof-coordinator]") {
    Fixture enabled{kMate}, disabled{kMate, false};
    const auto mates = mate_in_one(enabled.root, enabled.attacks);
    REQUIRE_FALSE(mates.empty());
    const auto result = enabled.run(128), old = disabled.run(128);
    CHECK(result.info.proof.outcome == Win);
    CHECK(result.info.proof.mate_upper_bound == 1);
    CHECK(result.info.q == 1);
    CHECK(result.info.draw == 0);
    CHECK(result.reason == mouse::search::StopReason::NodeLimit);
    CHECK(result.info.nodes == 128);
    REQUIRE(result.best_move);
    CHECK(std::ranges::find(mates, required_optional(result.best_move)) != mates.end());
    CHECK(old.info.proof.outcome == Unknown);
    CHECK(old.info.q < 1);
    const auto roots = enabled.engine->root_moves();
    std::uint64_t best = 0;
    for (const auto& row : std::span{roots.moves}.first(roots.size))
        if (row.move == required_optional(result.best_move)) best = row.visits;
    for (const auto& row : std::span{roots.moves}.first(roots.size)) {
        CHECK(best >= row.visits);
        if (std::ranges::find(mates, row.move) != mates.end() && row.visits != 0) {
            CHECK(row.child_proof.outcome == Loss);
            CHECK(row.child_proof.mate_upper_bound == 0);
            CHECK(row.q == 1);
        }
    }
    const auto continued = enabled.run(256);
    CHECK(continued.info.nodes == 256);
    CHECK(continued.info.new_playouts == 128);
    CHECK(continued.info.q == 1);
    CHECK(continued.info.draw == 0);
    CHECK(continued.info.proof == result.info.proof);
    CHECK(continued.reason == mouse::search::StopReason::NodeLimit);
    std::uint64_t child_visits = 0;
    const auto later = enabled.engine->root_moves();
    for (const auto& row : std::span{later.moves}.first(later.size)) child_visits += row.visits;
    CHECK(child_visits == 255);  // Proven nonterminal root still traverses real children.
}

TEST_CASE("Proof retains exact context on reuse and resets with different history",
          "[search][proof][proof-coordinator]") {
    Fixture fixture{kMate};
    const auto first = fixture.run(128);
    REQUIRE(first.info.proof.outcome == Win);
    REQUIRE(fixture.engine->set_position(fixture.root, fixture.history));
    CHECK(fixture.engine->snapshot().proof == first.info.proof);
    CHECK(fixture.engine->snapshot().nodes == 128);
    REQUIRE(first.best_move);
    const auto retained = fixture.engine->advance(required_optional(first.best_move));
    REQUIRE(retained);
    CHECK_FALSE(retained->discarded_context);
    CHECK(fixture.engine->snapshot().proof.outcome == Loss);
    CHECK(fixture.engine->snapshot().proof.mate_upper_bound == 0);
    CHECK(fixture.run(256).reason == mouse::search::StopReason::Terminal);
    auto changed = position("7k/8/5KQ1/8/8/8/8/8 w - - 3 2", fixture.attacks);
    mouse::nn::EncodingHistory history;
    history.reset(changed);
    REQUIRE(fixture.engine->set_position(changed, history));
    CHECK(fixture.engine->snapshot().proof.outcome == Unknown);
    CHECK(fixture.engine->snapshot().nodes == 0);
}

TEST_CASE("NN endpoints and depth frontiers never establish proof", "[search][proof]") {
    Fixture fixture{mouse::chess::kStartFen, true, 1};
    fixture.backend.endpoints = true;
    const auto result = fixture.run(257);
    CHECK(result.info.proof.outcome == Unknown);
    CHECK(result.info.seldepth == 1);
    const auto roots = fixture.engine->root_moves();
    REQUIRE(roots.size == 20);
    for (const auto& row : std::span{roots.moves}.first(roots.size))
        CHECK(row.child_proof.outcome == Unknown);
}

TEST_CASE("Genuine mate at a depth boundary still proves while other frontiers remain unknown",
          "[search][proof]") {
    Fixture fixture{kMate, true, 1};
    const auto result = fixture.run(512);
    CHECK(result.info.proof.outcome == Win);
    CHECK(result.info.proof.mate_upper_bound == 1);
    CHECK(result.info.q == 1);
    CHECK(result.info.nodes == 512);
    CHECK(result.reason == mouse::search::StopReason::NodeLimit);
}

TEST_CASE("Proof feature-off scheduling is unchanged before genuine terminals", "[search][proof]") {
    Fixture first, second{mouse::chess::kStartFen, false};
    const auto a = first.run(257), b = second.run(257);
    CHECK(a.info.proof.outcome == Unknown);
    CHECK(b.info.proof.outcome == Unknown);
    CHECK(a.best_move == b.best_move);
    CHECK(a.info.q == b.info.q);
    CHECK(a.info.draw == b.info.draw);
    REQUIRE(first.backend.rows == second.backend.rows);
    REQUIRE(first.backend.calls == second.backend.calls);
    CHECK(std::equal(first.backend.trace.begin(),
                     first.backend.trace.begin() + static_cast<std::ptrdiff_t>(first.backend.rows),
                     second.backend.trace.begin()));
    CHECK(
        std::equal(first.backend.batches.begin(),
                   first.backend.batches.begin() + static_cast<std::ptrdiff_t>(first.backend.calls),
                   second.backend.batches.begin()));
}

TEST_CASE("Genuine root terminals expose proof while tree-twofold remains only heuristic",
          "[search][proof]") {
    for (const auto fen : {"7k/6Q1/6K1/8/8/8/8/8 b - - 100 1", "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"}) {
        Fixture fixture{fen};
        const auto result = fixture.run(128);
        CHECK(result.reason == mouse::search::StopReason::Terminal);
        CHECK(result.info.proof.outcome == (fixture.root.in_check(fixture.attacks) ? Loss : Draw));
        CHECK(result.info.proof.mate_upper_bound == 0);
        CHECK(fixture.backend.calls == 0);
    }
    Fixture repeated;
    repeated.backend.cycle = true;
    const auto result = repeated.run(128);
    CHECK(result.info.proof.outcome == Unknown);
    REQUIRE(result.best_move);
    const auto moved = repeated.engine->advance(required_optional(result.best_move));
    REQUIRE(moved);
    CHECK(moved->discarded_context);
    CHECK(repeated.engine->snapshot().proof.outcome == Unknown);
}
TEST_CASE("A nonterminal forced loss and retained winning child keep real child visits",
          "[search][proof][proof-coordinator]") {
    Fixture fixture{"7k/8/5K2/6Q1/8/8/8/8 b - - 0 1"};
    auto probe = fixture.root;
    const auto replies = mouse::chess::legal_moves(probe, fixture.attacks);
    REQUIRE(replies.size() == 1);
    for (const auto move : replies) {
        const auto undo = probe.make_move(move, fixture.attacks);
        CHECK_FALSE(mate_in_one(probe, fixture.attacks).empty());
        probe.unmake_move(move, undo);
    }
    const auto result = fixture.run(128);
    CHECK(result.info.proof.outcome == Loss);
    CHECK(result.info.proof.mate_upper_bound == 2);
    CHECK(result.info.q == -1);
    CHECK(result.info.draw == 0);
    CHECK(result.reason == mouse::search::StopReason::NodeLimit);
    REQUIRE(result.best_move);
    REQUIRE(fixture.engine->advance(required_optional(result.best_move)));
    const auto before = fixture.engine->snapshot();
    CHECK(before.proof.outcome == Win);
    CHECK(before.proof.mate_upper_bound == 1);
    const auto next = fixture.run(before.nodes + 128);
    CHECK(next.reason == mouse::search::StopReason::NodeLimit);
    CHECK(next.info.inherited_visits == before.nodes);
    CHECK(next.info.new_playouts == 128);
    CHECK(next.info.proof == before.proof);
    CHECK(next.info.q == 1);
    const auto roots = fixture.engine->root_moves();
    std::uint64_t visits = 0;
    for (const auto& row : std::span{roots.moves}.first(roots.size)) visits += row.visits;
    CHECK(visits > 128);
}

TEST_CASE("Draw proof consumes every legal edge rather than only a visited subset",
          "[search][proof][proof-coordinator]") {
    Fixture fixture{"7k/8/8/8/8/2K5/8/R7 w - - 99 70"};
    const auto partial = fixture.run(2);
    CHECK(partial.info.proof.outcome == Unknown);
    const auto all = fixture.run(128);
    CHECK(all.info.proof.outcome == Draw);
    CHECK(all.info.proof.mate_upper_bound == 0);
    CHECK(all.info.q == 0);
    CHECK(all.info.draw == 1);
    CHECK(all.reason == mouse::search::StopReason::NodeLimit);
    const auto roots = fixture.engine->root_moves();
    REQUIRE(roots.size > 1);
    for (const auto& row : std::span{roots.moves}.first(roots.size))
        CHECK(row.child_proof.outcome == Draw);
}

TEST_CASE("FIDE threefold root still expands legal moves instead of seeding terminal proof",
          "[search][proof]") {
    Fixture fixture;
    for (unsigned cycle = 0; cycle < 2; ++cycle) {
        for (const auto token : {"b1c3", "b8c6", "c3b1", "c6b8"}) {
            const auto move = mouse::chess::parse_uci_move(fixture.root, token, fixture.attacks);
            REQUIRE(move);
            static_cast<void>(fixture.root.make_move(*move, fixture.attacks));
            REQUIRE(fixture.history.append(fixture.root));
        }
    }
    REQUIRE(fixture.root.is_repetition(3));
    REQUIRE(fixture.engine->set_position(fixture.root, fixture.history));
    const auto result = fixture.run(128);
    CHECK(result.reason == mouse::search::StopReason::NodeLimit);
    CHECK(result.info.nodes == 128);
    REQUIRE(result.best_move);
    CHECK(fixture.backend.calls >= 1);
}

TEST_CASE("Unknown siblings remain PUCT-explorable after an established win", "[search][proof]") {
    Fixture fixture{kMate};
    REQUIRE(fixture.run(128).info.proof.outcome == Win);
    const auto before = fixture.engine->root_moves();
    std::optional<mouse::chess::Move> unknown;
    std::uint64_t previous = 0;
    for (const auto& row : std::span{before.moves}.first(before.size)) {
        if (row.child_proof.outcome == Unknown) {
            unknown = row.move;
            previous = row.visits;
            break;
        }
    }
    REQUIRE(unknown);
    const auto result = fixture.run(2048);
    CHECK(result.info.proof.outcome == Win);
    const auto after = fixture.engine->root_moves();
    bool found = false;
    for (const auto& row : std::span{after.moves}.first(after.size)) {
        if (row.move == required_optional(unknown)) {
            found = true;
            CHECK(row.visits > previous);
        }
    }
    CHECK(found);
}
TEST_CASE("Root search restrictions cannot omit unknown legal defenses from proof",
          "[search][proof][root-filter]") {
    Fixture fixture{"4k3/8/8/8/8/8/8/R3K3 w - - 99 1"};
    auto root = fixture.root;
    const auto legal = mouse::chess::legal_moves(root, fixture.attacks);
    REQUIRE(legal.size() > 1);
    mouse::search::SearchLimits limit;
    limit.nodes = 128;
    limit.searchmoves[0] = legal[0];
    limit.searchmoves_count = 1;
    const auto restricted = fixture.engine->run(limit);
    REQUIRE(restricted);
    CHECK(restricted->best_move == legal[0]);
    CHECK(restricted->info.proof.outcome == Unknown);
    const auto full = fixture.run(2048);
    CHECK(full.info.proof.outcome == Draw);
    CHECK(full.info.q == 0);
    CHECK(full.info.draw == 1);
}

TEST_CASE("Known mate outranks a more visited stalemate when smart pruning stops",
          "[search][proof][terminal-choice-v1]") {
    Fixture fixture{kMate};
    const auto mates = mate_in_one(fixture.root, fixture.attacks);
    REQUIRE_FALSE(mates.empty());
    std::optional<mouse::chess::Move> draw;
    auto probe = fixture.root;
    for (const auto move : mouse::chess::legal_moves(probe, fixture.attacks)) {
        const auto undo = probe.make_move(move, fixture.attacks);
        if (mouse::chess::game_status(probe, fixture.attacks) ==
            mouse::chess::GameStatus::Stalemate)
            draw = move;
        probe.unmake_move(move, undo);
        if (draw) break;
    }
    REQUIRE(draw);
    mouse::search::SearchLimits limits;
    limits.smart_pruning = false;
    limits.nodes = 16;
    limits.searchmoves_count = 1;
    limits.searchmoves[0] = required_optional(draw);
    auto result = fixture.engine->run(limits);
    REQUIRE(result);
    REQUIRE(result->info.nodes == 16);
    limits.nodes = 17;
    limits.searchmoves[0] = mates[0];
    result = fixture.engine->run(limits);
    REQUIRE(result);
    REQUIRE(result->info.proof.outcome == Win);
    REQUIRE(fixture.engine->set_position(fixture.root, fixture.history));
    const auto snapshot = fixture.engine->result_snapshot();
    REQUIRE(snapshot.best_move);
    CHECK(std::ranges::find(mates, required_optional(snapshot.best_move)) != mates.end());
    limits.nodes = 1000;
    limits.searchmoves_count = 0;
    limits.smart_pruning = true;
    result = fixture.engine->run(limits);
    REQUIRE(result);
    REQUIRE(result->reason == mouse::search::StopReason::SmartPruning);
    REQUIRE(result->best_move);
    CHECK(std::ranges::find(mates, required_optional(result->best_move)) != mates.end());
    const auto roots = fixture.engine->root_moves();
    std::uint64_t draw_visits = 0, mate_visits = 0;
    for (const auto& row : std::span{roots.moves}.first(roots.size)) {
        if (row.move == required_optional(draw)) draw_visits = row.visits;
        if (row.move == mates[0]) mate_visits = row.visits;
    }
    CHECK(draw_visits > mate_visits);
}

TEST_CASE("Ordinary neural backups do not rescan unchanged sibling proofs",
          "[search][proof][proof-refresh]") {
    Fixture fixture;
    mouse::search::SearchLimits limits;
    limits.nodes = 257;
    limits.smart_pruning = false;
    limits.log_search_stats = true;
    const auto result = fixture.engine->run(limits);
    REQUIRE(result);
    CHECK(result->info.nodes == 257);
    CHECK(result->info.proof.outcome == Unknown);
    CHECK(result->stats.out_of_order == 0);
    CHECK(result->stats.proof_scans == 0);
}

TEST_CASE("Parallel gather reuses unchanged PUCT terms with a deterministic trace",
          "[search][selection-cache]") {
    Fixture fixture;
    mouse::search::SearchLimits limits;
    limits.nodes = 3000;
    limits.smart_pruning = false;
    limits.log_search_stats = true;
    const auto result = fixture.engine->run(limits);
    REQUIRE(result);
    std::uint64_t digest = 1469598103934665603ULL;
    for (std::size_t i = 0; i < fixture.backend.rows; ++i)
        digest = (digest ^ fixture.backend.trace.at(i)) * 1099511628211ULL;
    for (std::size_t i = 0; i < fixture.backend.calls; ++i)
        digest = (digest ^ fixture.backend.batches.at(i)) * 1099511628211ULL;
    const auto root = fixture.engine->root_moves();
    for (std::size_t i = 0; i < root.size; ++i) {
        digest = (digest ^ root.moves.at(i).visits) * 1099511628211ULL;
        digest = (digest ^ std::bit_cast<std::uint64_t>(root.moves.at(i).q)) * 1099511628211ULL;
    }
    // Every encoded input, batch boundary, root visit and Q bit is covered, not
    // merely the final selected move. Re-captured for the lc0 single-walk gather
    // (PickNodesToExtendTask), which changes batch composition by design; the
    // per-row descent trace was 14677433757627993344 (recorded 2026-10-18).
    CHECK(digest == 8352298212493842131ULL);
    CHECK(result->info.nodes == 3000);
    CHECK(result->stats.selection_hits > 0);
}
