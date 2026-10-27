// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/search/proof.h"

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <limits>
#include <type_traits>

#include "engine/search/rules.h"
#include "engine/search/tree.h"
#include "tests/required_optional.h"

namespace {
using mouse::search::NodeProof;
using mouse::search::ProofAccumulator;
using mouse::search::ProofOutcome;
using mouse::tests::required_optional;
using enum ProofOutcome;
NodeProof proof(ProofOutcome outcome, std::uint16_t bound = 0) {
    return {.outcome = outcome, .mate_upper_bound = bound};
}
mouse::Result<NodeProof> combine(std::span<const NodeProof> children) {
    auto accumulator = ProofAccumulator::create(children.size());
    if (!accumulator) return tl::unexpected(accumulator.error());
    for (const auto child : children) {
        const auto added = accumulator->add(child);
        if (!added) return tl::unexpected(added.error());
    }
    return accumulator->finish();
}
}  // namespace

TEST_CASE("Only genuine rule terminals seed exact proof", "[search][proof]") {
    using enum mouse::search::TerminalReason;
    const auto seed = mouse::search::proof_from_terminal;
    CHECK(seed(Checkmate) == proof(Loss));
    for (const auto reason : {Stalemate, FiftyMove, Threefold, InsufficientMaterial})
        CHECK(seed(reason) == proof(Draw));
    CHECK(seed(None) == proof(Unknown));
    CHECK(seed(TreeRepetition) == proof(Unknown));
    static_assert(std::is_trivially_copyable_v<NodeProof>);
    static_assert(std::is_trivially_copyable_v<ProofAccumulator>);
}

TEST_CASE("Proof AND OR two-child truth table includes every unknown case", "[search][proof]") {
    const std::array outcomes{Unknown, Win, Draw, Loss};
    const std::array matrix{std::array{Unknown, Unknown, Unknown, Win},
                            std::array{Unknown, Loss, Draw, Win},
                            std::array{Unknown, Draw, Draw, Win}, std::array{Win, Win, Win, Win}};
    for (std::size_t i = 0; i < outcomes.size(); ++i) {
        for (std::size_t j = 0; j < outcomes.size(); ++j) {
            const auto a = outcomes.at(i), b = outcomes.at(j);
            const std::array children{proof(a, a == Win ? 1 : 0), proof(b, b == Win ? 1 : 0)};
            const auto result = combine(children);
            REQUIRE(result);
            CHECK(result->outcome == matrix.at(i).at(j));
            const auto bound = result->outcome == Win ? 1 : (result->outcome == Loss ? 2 : 0);
            CHECK(result->mate_upper_bound == bound);
        }
    }
    const auto empty = combine({});
    REQUIRE(empty);
    CHECK(*empty == proof(Unknown));
}

TEST_CASE("Proof mate upper bounds are permutation independent min and max", "[search][proof]") {
    std::array<unsigned, 5> order{0, 1, 2, 3, 4};
    const std::array values{proof(Win, 3), proof(Loss, 4), proof(Unknown), proof(Draw),
                            proof(Loss, 2)};
    for (bool more = true; more; more = std::ranges::next_permutation(order).found) {
        std::array<NodeProof, 5> children{};
        for (std::size_t i = 0; i < order.size(); ++i) children.at(i) = values.at(order.at(i));
        const auto result = combine(children);
        REQUIRE(result);
        CHECK(*result == proof(Win, 3));
    }
    std::array<unsigned, 3> loss_order{0, 1, 2};
    const std::array wins{proof(Win, 1), proof(Win, 5), proof(Win, 3)};
    for (bool more = true; more; more = std::ranges::next_permutation(loss_order).found) {
        const std::array children{wins.at(loss_order.at(0)), wins.at(loss_order.at(1)),
                                  wins.at(loss_order.at(2))};
        const auto result = combine(children);
        REQUIRE(result);
        CHECK(*result == proof(Loss, 6));
    }
}

TEST_CASE("Proof accumulation requires the complete legal set without silent truncation",
          "[search][proof]") {
    auto accumulator = ProofAccumulator::create(3);
    REQUIRE(accumulator);
    REQUIRE(accumulator->add(proof(Win, 1)));
    REQUIRE(accumulator->add(proof(Draw)));
    CHECK_FALSE(accumulator->finish());  // A root filter cannot turn a partial set into Draw.
    REQUIRE(accumulator->add(proof(Unknown)));
    CHECK(*accumulator->finish() == proof(Unknown));
    CHECK_FALSE(accumulator->add(proof(Loss)));
    CHECK(*accumulator->finish() == proof(Unknown));
    auto winning = ProofAccumulator::create(2);
    REQUIRE(winning);
    REQUIRE(winning->add(proof(Loss)));
    CHECK_FALSE(winning->finish());  // Even a known winning child needs complete edge enumeration.
    REQUIRE(winning->add(proof(Unknown)));
    CHECK(*winning->finish() == proof(Win, 1));
    CHECK_FALSE(ProofAccumulator::create(mouse::chess::MoveList::kCapacity + 1));
    CHECK_FALSE(ProofAccumulator::create(std::numeric_limits<std::size_t>::max()));
    std::array<NodeProof, mouse::chess::MoveList::kCapacity> maximum{};
    CHECK(*combine(maximum) == proof(Unknown));
}

TEST_CASE("Malformed proof and mate-bound overflow reject atomically", "[search][proof]") {
    for (const auto invalid :
         {proof(static_cast<ProofOutcome>(255)), proof(Unknown, 1), proof(Draw, 1), proof(Win),
          proof(Loss, 513), proof(Win, std::numeric_limits<std::uint16_t>::max())}) {
        auto accumulator = ProofAccumulator::create(1);
        REQUIRE(accumulator);
        CHECK_FALSE(accumulator->add(invalid));
        REQUIRE(accumulator->add(proof(Draw)));
        CHECK(*accumulator->finish() == proof(Draw));
    }
    const std::array too_long_win{proof(Loss, 512)};
    const std::array too_long_loss{proof(Win, 512)};
    CHECK_FALSE(combine(too_long_win));
    CHECK_FALSE(combine(too_long_loss));
    const std::array exactly_max{proof(Win, 511)};
    CHECK(*combine(exactly_max) == proof(Loss, 512));
    const std::array shorter_known{proof(Loss, 512), proof(Loss)};
    CHECK(*combine(shorter_known) == proof(Win, 1));
    const std::array unknown_with_long_win{proof(Unknown), proof(Win, 512)};
    CHECK(*combine(unknown_with_long_win) == proof(Unknown));
}
TEST_CASE("Proof upper bounds tighten when newly established alternatives improve strategy",
          "[search][proof]") {
    const std::array first{proof(Loss, 4), proof(Unknown)};
    const std::array refined{proof(Loss, 4), proof(Loss, 2)};
    CHECK(*combine(first) == proof(Win, 5));
    CHECK(*combine(refined) == proof(Win, 3));
    const std::array first_loss{proof(Win, 1), proof(Win, 5)};
    const std::array refined_loss{proof(Win, 1), proof(Win, 3)};
    CHECK(*combine(first_loss) == proof(Loss, 6));
    CHECK(*combine(refined_loss) == proof(Loss, 4));
}

TEST_CASE("Tree compaction preserves proven expanded nodes and discards contextual heuristics",
          "[search][proof][tree]") {
    auto created = mouse::search::Tree::create(64U << 10U);
    REQUIRE(created);
    auto& tree = **created;
    for (unsigned i = 1; i <= 5; ++i) {
        const auto node = tree.allocate_node();
        REQUIRE(node);
        CHECK(*node == i);
    }
    const mouse::chess::Move a{mouse::chess::square(0, 1), mouse::chess::square(0, 2)};
    const mouse::chess::Move b{mouse::chess::square(1, 1), mouse::chess::square(1, 2)};
    const std::array root_seeds{mouse::search::EdgeSeed{.move = a},
                                mouse::search::EdgeSeed{.move = b}};
    const std::array one_seed{mouse::search::EdgeSeed{.move = a}};
    REQUIRE(tree.expand(0, root_seeds));
    REQUIRE(tree.expand(1, one_seed));
    REQUIRE(tree.expand(3, root_seeds));
    tree.edges(0)[0].child = 1;
    tree.edges(0)[1].child = 2;
    tree.edges(1)[0].child = 3;
    tree.edges(3)[0].child = 4;
    tree.edges(3)[1].child = 5;
    tree.node(0).proof = proof(Win, 3);
    tree.node(1).proof = proof(Loss, 2);
    tree.node(1).visits = 7;
    tree.node(1).value_sum = -7;
    tree.node(3).proof = proof(Win, 1);
    tree.node(4).proof = proof(Loss);
    tree.node(4).state = mouse::search::NodeState::Terminal;
    bool contextual = false;
    SECTION("Quiescent complete-history subtree keeps proof") {}
    SECTION("A reachable heuristic twofold invalidates retained context") {
        contextual = true;
        tree.node(5).state = mouse::search::NodeState::Terminal;
        tree.node(5).contextual_draw = true;
        tree.node(5).leaf_draw = 1;
    }
    const auto reused = tree.re_root(1);
    REQUIRE(reused);
    CHECK(reused->discarded_context == contextual);
    if (contextual) {
        CHECK(tree.node(0).proof == proof(Unknown));
        CHECK(tree.node(0).visits == 0);
        CHECK(tree.stats().node_count == 1);
    } else {
        CHECK(tree.node(0).proof == proof(Loss, 2));
        CHECK(tree.node(0).state == mouse::search::NodeState::Expanded);
        CHECK(tree.node(0).visits == 7);
        CHECK(tree.node(0).value_sum == -7);
        CHECK(tree.node(1).proof == proof(Win, 1));
        CHECK(tree.node(1).state == mouse::search::NodeState::Expanded);
        CHECK(tree.node(2).proof == proof(Loss));
        CHECK(tree.node(3).proof == proof(Unknown));
        CHECK(tree.stats().node_count == 4);
        CHECK(tree.stats().edge_count == 3);
        REQUIRE(tree.reset());
        CHECK(tree.node(0).proof == proof(Unknown));
    }
    CHECK(tree.stats().charged_bytes <= tree.stats().budget_bytes);
}
TEST_CASE("Soft proof utility prefers established bounds without excluding refinement",
          "[search][proof][proof-utility]") {
    const auto utility = mouse::search::proof_selection_utility;
    const auto best = utility(proof(Win, 5), proof(Loss, 4));
    const auto longer = utility(proof(Win, 5), proof(Loss, 6));
    REQUIRE(best);
    REQUIRE(longer);
    REQUIRE(best->has_value());
    REQUIRE(longer->has_value());
    CHECK(required_optional(*best) == 0.0345);
    CHECK(required_optional(*longer) > 0);
    CHECK(required_optional(*longer) == 0.0345 / 3.0);
    // The formerly longer branch is still traversable and may establish a
    // shorter bound. Recomputing the parent then reverses the relative bonus.
    const auto improved = utility(proof(Win, 3), proof(Loss, 2));
    const auto previous = utility(proof(Win, 3), proof(Loss, 4));
    REQUIRE(improved);
    REQUIRE(previous);
    CHECK(required_optional(*improved) == required_optional(*best));
    CHECK(required_optional(*previous) == required_optional(*longer));
    const auto delay = utility(proof(Loss, 6), proof(Win, 5));
    const auto shorter_loss = utility(proof(Loss, 6), proof(Win, 1));
    REQUIRE(delay);
    REQUIRE(shorter_loss);
    CHECK(required_optional(*delay) == 0.0345);
    CHECK(required_optional(*shorter_loss) == 0.0345 / 5.0);
    for (const auto parent : {proof(Unknown), proof(Draw), proof(Loss)}) {
        const auto none = utility(parent, proof(Win, 1));
        REQUIRE(none);
        CHECK_FALSE(none->has_value());
    }
    for (const auto child : {proof(Unknown), proof(Draw), proof(Win, 1)}) {
        const auto none = utility(proof(Win, 3), child);
        REQUIRE(none);
        CHECK_FALSE(none->has_value());
    }
    const auto invalid = utility(proof(Win), proof(Loss));
    CHECK_FALSE(invalid);
    const auto overflowing = utility(proof(Loss, 512), proof(Win, 65535));
    CHECK_FALSE(overflowing);
}