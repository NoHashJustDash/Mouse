// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/search/tree.h"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace {
using namespace mouse::search;
using mouse::chess::Move;

std::unique_ptr<Tree> make_tree(std::size_t budget = 65536) {
    auto result = Tree::create(budget);
    REQUIRE(result.has_value());
    return std::move(*result);
}
NodeId allocate(Tree& tree) {
    auto result = tree.allocate_node();
    REQUIRE(result.has_value());
    return *result;
}
EdgeSeed seed(unsigned int tag, std::uint16_t bits = 0x3800) {
    return {.move = Move{static_cast<mouse::chess::Square>(tag % 64),
                         static_cast<mouse::chess::Square>((tag / 64 + 1) % 64)},
            .prior_bits = bits};
}
bool same_node(const Node& left, const Node& right) {
    return std::tie(left.value_sum, left.draw_sum, left.visits, left.in_flight, left.first_edge,
                    left.edge_count, left.state, left.contextual_draw, left.leaf_q,
                    left.leaf_draw) == std::tie(right.value_sum, right.draw_sum, right.visits,
                                                right.in_flight, right.first_edge, right.edge_count,
                                                right.state, right.contextual_draw, right.leaf_q,
                                                right.leaf_draw);
}
struct Snapshot {
    TreeStats stats;
    NodeId root;
    std::uint64_t generation;
    std::vector<Node> nodes;
    std::vector<std::vector<Edge>> edges;
};
Snapshot snapshot(const Tree& tree) {
    Snapshot result{.stats = tree.stats(),
                    .root = tree.root(),
                    .generation = tree.generation(),
                    .nodes = {},
                    .edges = {}};
    for (NodeId id = 0; id < result.stats.node_count; ++id) {
        result.nodes.push_back(tree.node(id));
        const auto edges = tree.edges(id);
        result.edges.emplace_back(edges.begin(), edges.end());
    }
    return result;
}
void check_snapshot(const Tree& tree, const Snapshot& expected) {
    CHECK(tree.root() == expected.root);
    CHECK(tree.generation() == expected.generation);
    CHECK(tree.stats().node_count == expected.stats.node_count);
    CHECK(tree.stats().edge_count == expected.stats.edge_count);
    CHECK(tree.stats().charged_bytes == expected.stats.charged_bytes);
    for (NodeId id = 0; id < expected.stats.node_count; ++id) {
        CHECK(same_node(tree.node(id), expected.nodes.at(id)));
        const auto edges = tree.edges(id);
        REQUIRE(edges.size() == expected.edges.at(id).size());
        for (std::size_t index = 0; index < edges.size(); ++index) {
            CHECK(edges[index].move == expected.edges.at(id).at(index).move);
            CHECK(edges[index].child == expected.edges.at(id).at(index).child);
            CHECK(edges[index].prior_bits == expected.edges.at(id).at(index).prior_bits);
        }
    }
}
void tag_node(Node& node, unsigned int tag) {
    node.value_sum = static_cast<double>(tag) / 4;
    node.draw_sum = static_cast<double>(tag) / 8;
    node.visits = tag + 1;
    node.leaf_q = static_cast<float>(tag) / 16;
    node.leaf_draw = static_cast<float>(tag) / 32;
}
}  // namespace

TEST_CASE("Tree rejects impossible budgets and charges all fixed storage", "[search][tree]") {
    CHECK_FALSE(Tree::create(0));
    CHECK_FALSE(Tree::create(sizeof(Tree)));
    CHECK_FALSE(Tree::create(std::numeric_limits<std::size_t>::max()));
    auto tree = make_tree();
    const auto stats = tree->stats();
    CHECK(stats.budget_bytes == 65536);
    CHECK(stats.charged_bytes <= stats.budget_bytes);
    CHECK(stats.node_count == 1);
    CHECK(stats.edge_count == 0);
    CHECK(stats.node_capacity > 1);
    CHECK(stats.edge_capacity > 1);
    CHECK(stats.charged_bytes >=
          sizeof(Tree) + stats.node_capacity * sizeof(Node) + stats.edge_capacity * sizeof(Edge));
    CHECK(tree->root() == 0);
    CHECK(same_node(tree->node(0), Node{}));
}

TEST_CASE("Tree node exhaustion is atomic and reset recovers every slot", "[search][tree]") {
    auto tree = make_tree(4096);
    while (tree->stats().node_count < tree->stats().node_capacity) {
        const auto expected = tree->stats().node_count;
        CHECK(allocate(*tree) == expected);
    }
    const auto full = snapshot(*tree);
    CHECK_FALSE(tree->allocate_node());
    check_snapshot(*tree, full);
    REQUIRE(tree->reset());
    CHECK(tree->generation() == full.generation + 1);
    CHECK(tree->stats().node_count == 1);
    CHECK(tree->stats().edge_count == 0);
    CHECK(tree->stats().charged_bytes == full.stats.charged_bytes);
    CHECK(same_node(tree->node(0), Node{}));
    CHECK(allocate(*tree) == 1);
}

TEST_CASE("Tree expansion validates the complete seed block before publishing", "[search][tree]") {
    auto tree = make_tree();
    const auto before = snapshot(*tree);
    const std::array invalid_bits{std::uint16_t{0x3c01}, std::uint16_t{0x8000},
                                  std::uint16_t{0x7c00}, std::uint16_t{0x7e00}};
    for (const auto bits : invalid_bits) {
        const std::array seeds{seed(1), seed(2, bits)};
        CHECK_FALSE(tree->expand(0, seeds));
        check_snapshot(*tree, before);
    }
    const std::array duplicate{seed(1), seed(1)};
    CHECK_FALSE(tree->expand(0, duplicate));
    CHECK_FALSE(tree->expand(kNoNode, std::array{seed(1)}));
    CHECK_FALSE(tree->expand(0, {}));
    check_snapshot(*tree, before);
    const std::array valid{seed(1, 0), seed(2, 1), seed(3, 0x3c00)};
    tree->node(0).state = NodeState::Pending;
    tree->node(0).in_flight = 1;
    REQUIRE(tree->expand(0, valid));
    CHECK(tree->node(0).state == NodeState::Expanded);
    CHECK(tree->node(0).in_flight == 1);
    REQUIRE(tree->edges(0).size() == 3);
    for (std::size_t index = 0; index < valid.size(); ++index) {
        CHECK(tree->edges(0)[index].move == valid.at(index).move);
        CHECK(tree->edges(0)[index].prior_bits == valid.at(index).prior_bits);
        CHECK(tree->edges(0)[index].child == kNoNode);
    }
    const auto expanded = snapshot(*tree);
    CHECK_FALSE(tree->expand(0, valid));
    check_snapshot(*tree, expanded);
}

TEST_CASE("Tree edge exhaustion does not partially expand a node", "[search][tree]") {
    auto tree = make_tree(4096);
    const auto other = allocate(*tree);
    const auto capacity = tree->stats().edge_capacity;
    REQUIRE(capacity < 4096);
    std::vector<EdgeSeed> too_many;
    for (unsigned int index = 0; index <= capacity; ++index) too_many.push_back(seed(index));
    const auto before = snapshot(*tree);
    CHECK_FALSE(tree->expand(0, too_many));
    check_snapshot(*tree, before);
    too_many.pop_back();
    REQUIRE(tree->expand(0, too_many));
    const auto exhausted = snapshot(*tree);
    CHECK_FALSE(tree->expand(other, std::array{seed(1)}));
    check_snapshot(*tree, exhausted);
}

TEST_CASE("Tree accessor and nonexpandable state failures are explicit", "[search][tree]") {
    auto tree = make_tree();
    const auto& immutable = *tree;
    CHECK_THROWS_AS(tree->node(1), std::out_of_range);
    CHECK_THROWS_AS(immutable.node(kNoNode), std::out_of_range);
    CHECK_THROWS_AS(tree->edges(kNoNode), std::out_of_range);
    CHECK_THROWS_AS(immutable.edges(1), std::out_of_range);
    for (const auto state : {NodeState::Terminal, NodeState::Frontier}) {
        tree->node(0).state = state;
        const auto before = snapshot(*tree);
        CHECK_FALSE(tree->expand(0, std::array{seed(1)}));
        check_snapshot(*tree, before);
    }
}

TEST_CASE("Tree reset and reroot reject reservations anywhere without changes", "[search][tree]") {
    auto tree = make_tree();
    const auto orphan = allocate(*tree);
    for (const auto pending : {false, true}) {
        tree->node(orphan).in_flight = pending ? 0 : 1;
        tree->node(orphan).state = pending ? NodeState::Pending : NodeState::Unexpanded;
        const auto before = snapshot(*tree);
        CHECK_FALSE(tree->reset());
        CHECK_FALSE(tree->re_root(0));
        check_snapshot(*tree, before);
    }
    tree->node(orphan) = Node{};
    REQUIRE(tree->reset());
}

TEST_CASE("Tree reroot compacts edges in allocation order rather than node order",
          "[search][tree]") {
    auto tree = make_tree();
    const auto keep = allocate(*tree);       // old1 -> new0
    const auto discarded = allocate(*tree);  // old2
    const auto left = allocate(*tree);       // old3 -> new1
    const auto right = allocate(*tree);      // old4 -> new2
    const auto leaf = allocate(*tree);       // old5 -> new3
    REQUIRE(tree->expand(0, std::array{seed(10), seed(11)}));
    tree->edges(0)[0].child = keep;
    tree->edges(0)[1].child = discarded;
    REQUIRE(tree->expand(right, std::array{seed(40), seed(41)}));
    tree->edges(right)[0].child = leaf;
    REQUIRE(tree->expand(discarded, std::array{seed(20)}));
    REQUIRE(tree->expand(left, std::array{seed(30)}));
    REQUIRE(tree->expand(keep, std::array{seed(12), seed(13)}));
    tree->edges(keep)[0].child = left;
    tree->edges(keep)[1].child = right;
    tag_node(tree->node(keep), 4);
    tag_node(tree->node(left), 8);
    tag_node(tree->node(right), 12);
    tag_node(tree->node(leaf), 16);
    tree->node(leaf).state = NodeState::Terminal;
    const std::array saved{tree->node(keep), tree->node(left), tree->node(right), tree->node(leaf)};
    const auto generation = tree->generation();
    const auto result = tree->re_root(keep);
    REQUIRE(result);
    CHECK(result->kept_nodes == 4);
    CHECK(result->kept_edges == 5);
    CHECK_FALSE(result->discarded_context);
    CHECK(tree->generation() == generation + 1);
    CHECK(tree->root() == 0);
    CHECK(tree->stats().node_count == 4);
    CHECK(tree->stats().edge_count == 5);
    for (NodeId id = 0; id < 4; ++id) {
        auto expected = saved.at(id);
        expected.first_edge = tree->node(id).first_edge;
        CHECK(same_node(tree->node(id), expected));
    }
    CHECK(tree->node(2).first_edge == 0);
    CHECK(tree->node(1).first_edge == 2);
    CHECK(tree->node(0).first_edge == 3);
    CHECK(tree->edges(0)[0].move == seed(12).move);
    CHECK(tree->edges(0)[0].child == 1);
    CHECK(tree->edges(0)[1].child == 2);
    CHECK(tree->edges(1)[0].move == seed(30).move);
    CHECK(tree->edges(2)[0].move == seed(40).move);
    CHECK(tree->edges(2)[0].child == 3);
    CHECK(tree->edges(2)[1].move == seed(41).move);
    CHECK(tree->edges(2)[1].child == kNoNode);
    CHECK(allocate(*tree) == 4);
}

TEST_CASE("Tree contextual descendants discard every retained statistic", "[search][tree]") {
    auto tree = make_tree();
    const auto keep = allocate(*tree);
    const auto leaf = allocate(*tree);
    REQUIRE(tree->expand(keep, std::array{seed(1)}));
    tree->edges(keep)[0].child = leaf;
    tag_node(tree->node(keep), 4);
    tree->node(leaf).state = NodeState::Terminal;
    tree->node(leaf).contextual_draw = true;
    const auto result = tree->re_root(keep);
    REQUIRE(result);
    CHECK(result->discarded_context);
    CHECK(result->kept_nodes == 1);
    CHECK(result->kept_edges == 0);
    CHECK(same_node(tree->node(0), Node{}));
    CHECK(tree->stats().node_count == 1);
    CHECK(tree->stats().edge_count == 0);
}

TEST_CASE("Tree ignores discarded contextual nodes and supports repeated compaction",
          "[search][tree]") {
    auto tree = make_tree();
    const auto discarded = allocate(*tree);
    tree->node(discarded).contextual_draw = true;
    auto current = allocate(*tree);
    for (unsigned int depth = 0; depth < 30; ++depth) {
        const auto next = allocate(*tree);
        REQUIRE(tree->expand(current, std::array{seed(depth)}));
        tree->edges(current)[0].child = next;
        current = next;
    }
    auto result = tree->re_root(2);
    REQUIRE(result);
    CHECK_FALSE(result->discarded_context);
    CHECK(result->kept_nodes == 31);
    for (unsigned int depth = 0; depth < 30; ++depth) {
        REQUIRE(tree->edges(0).size() == 1);
        CHECK(tree->edges(0)[0].move == seed(depth).move);
        CHECK(tree->edges(0)[0].child == 1);
        result = tree->re_root(1);
        REQUIRE(result);
        CHECK(result->kept_nodes == 30 - depth);
        CHECK(result->kept_edges == 29 - depth);
    }
    CHECK(same_node(tree->node(0), Node{}));
}

TEST_CASE("Tree malformed reroot graph is rejected before observable mutation", "[search][tree]") {
    for (unsigned int fault = 0; fault < 5; ++fault) {
        auto tree = make_tree();
        const auto child = allocate(*tree);
        REQUIRE(tree->expand(0, std::array{seed(1), seed(2)}));
        tree->edges(0)[0].child = child;
        if (fault == 0) tree->edges(0)[1].child = child;  // shared child
        if (fault == 1) tree->edges(0)[0].child = 0;      // backward/cycle
        if (fault == 2) tree->edges(0)[0].child = 9;      // invalid allocated ID
        if (fault == 3) tree->edges(0)[1].prior_bits = 0x8000;
        if (fault == 4) {  // overlapping edge blocks
            tree->node(child).first_edge = 0;
            tree->node(child).edge_count = 1;
        }
        const auto before = snapshot(*tree);
        CHECK_FALSE(tree->re_root(0));
        check_snapshot(*tree, before);
    }
    auto tree = make_tree();
    const auto before = snapshot(*tree);
    CHECK_FALSE(tree->re_root(kNoNode));
    check_snapshot(*tree, before);
}

TEST_CASE("Tree minimum complete budget has an exact failure boundary", "[search][tree]") {
    std::size_t lower = sizeof(Tree);
    std::size_t upper = 4096;
    while (lower < upper) {
        const auto middle = lower + (upper - lower) / 2;
        if (Tree::create(middle))
            upper = middle;
        else
            lower = middle + 1;
    }
    CHECK_FALSE(Tree::create(lower - 1));
    auto tree = make_tree(lower);
    CHECK(tree->stats().node_capacity == 1);
    CHECK(tree->stats().charged_bytes == lower);
    CHECK_FALSE(tree->allocate_node());
    REQUIRE(tree->expand(0, std::array{seed(1)}));
    const auto before = tree->generation();
    REQUIRE(tree->reset());
    CHECK(tree->generation() == before + 1);
    REQUIRE(tree->expand(0, std::array{seed(2)}));
    CHECK(tree->edges(0)[0].move == seed(2).move);
    CHECK(tree->edges(0)[0].child == kNoNode);
}

TEST_CASE("Tree oversized seed spans cannot truncate the published edge count", "[search][tree]") {
    auto tree = make_tree(2ULL * 1024 * 1024);
    const std::vector<EdgeSeed> oversized(65536, seed(1));
    const auto before = snapshot(*tree);
    CHECK_FALSE(tree->expand(0, oversized));
    check_snapshot(*tree, before);
}

TEST_CASE("Tree malformed edge ranges fail access and reroot without mutation", "[search][tree]") {
    for (unsigned int fault = 0; fault < 3; ++fault) {
        auto tree = make_tree();
        REQUIRE(tree->expand(0, std::array{seed(1)}));
        if (fault == 0) tree->node(0).first_edge = kNoEdge;
        if (fault == 1) tree->node(0).edge_count = 2;
        if (fault == 2) tree->node(0).edge_count = 0;
        const auto before = tree->node(0);
        const auto stats = tree->stats();
        const auto generation = tree->generation();
        const auto& immutable = *tree;
        CHECK_THROWS_AS(tree->edges(0), std::out_of_range);
        CHECK_THROWS_AS(immutable.edges(0), std::out_of_range);
        CHECK_FALSE(tree->re_root(0));
        CHECK(same_node(tree->node(0), before));
        CHECK(tree->generation() == generation);
        CHECK(tree->stats().node_count == stats.node_count);
        CHECK(tree->stats().edge_count == stats.edge_count);
    }
}

TEST_CASE("Tree wide reverse-allocated edge blocks preserve all retained branches",
          "[search][tree]") {
    auto tree = make_tree();
    for (NodeId id = 1; id < 127; ++id) REQUIRE(allocate(*tree) == id);
    for (NodeId id = 127; id-- > 0;) {
        tag_node(tree->node(id), id);
        if (id < 63) {
            REQUIRE(tree->expand(id, std::array{seed(id * 2), seed(id * 2 + 1)}));
            tree->edges(id)[0].child = id * 2 + 1;
            tree->edges(id)[1].child = id * 2 + 2;
        } else {
            tree->node(id).state = NodeState::Frontier;
        }
    }
    const auto result = tree->re_root(1);
    REQUIRE(result);
    CHECK(result->kept_nodes == 63);
    CHECK(result->kept_edges == 62);
    std::vector<std::pair<NodeId, NodeId>> pending{{0, 1}};
    for (std::size_t index = 0; index < pending.size(); ++index) {
        const auto [id, original] = pending.at(index);
        CHECK(tree->node(id).visits == original + 1);
        CHECK(tree->node(id).value_sum == static_cast<double>(original) / 4);
        CHECK(tree->node(id).draw_sum == static_cast<double>(original) / 8);
        CHECK(tree->node(id).leaf_q == static_cast<float>(original) / 16);
        CHECK(tree->node(id).leaf_draw == static_cast<float>(original) / 32);
        if (original < 63) {
            REQUIRE(tree->edges(id).size() == 2);
            for (unsigned int side = 0; side < 2; ++side) {
                const auto edge = tree->edges(id)[side];
                CHECK(edge.move == seed(original * 2 + side).move);
                CHECK(edge.prior_bits == 0x3800);
                REQUIRE(edge.child != kNoNode);
                pending.emplace_back(edge.child, original * 2 + side + 1);
            }
        } else {
            CHECK(tree->node(id).state == NodeState::Frontier);
            CHECK(tree->edges(id).empty());
        }
    }
    CHECK(pending.size() == 63);
}

TEST_CASE("Tree deep reverse-allocated blocks compact without recursion", "[search][tree]") {
    auto tree = make_tree(262144);
    for (NodeId id = 1; id <= 512; ++id) REQUIRE(allocate(*tree) == id);
    tree->node(512).state = NodeState::Terminal;
    tree->node(512).leaf_q = -1;
    tree->node(512).visits = 7;
    for (NodeId id = 512; id-- > 0;) {
        REQUIRE(tree->expand(id, std::array{seed(id)}));
        tree->edges(id)[0].child = id + 1;
    }
    const auto result = tree->re_root(1);
    REQUIRE(result);
    CHECK(result->kept_nodes == 512);
    CHECK(result->kept_edges == 511);
    for (NodeId id = 0; id < 511; ++id) {
        REQUIRE(tree->edges(id).size() == 1);
        CHECK(tree->edges(id)[0].move == seed(id + 1).move);
        CHECK(tree->edges(id)[0].child == id + 1);
    }
    CHECK(tree->node(511).state == NodeState::Terminal);
    CHECK(tree->node(511).leaf_q == -1);
    CHECK(tree->node(511).visits == 7);
}

TEST_CASE("Tree reroot distinguishes root context from context in discarded siblings",
          "[search][tree]") {
    auto tree = make_tree();
    const auto keep = allocate(*tree);
    const auto discard = allocate(*tree);
    REQUIRE(tree->expand(0, std::array{seed(1), seed(2)}));
    tree->edges(0)[0].child = keep;
    tree->edges(0)[1].child = discard;
    tree->node(discard).contextual_draw = true;
    tree->node(keep).state = NodeState::Terminal;
    tree->node(keep).leaf_q = -1;
    tree->node(keep).visits = 13;
    auto result = tree->re_root(keep);
    REQUIRE(result);
    CHECK_FALSE(result->discarded_context);
    CHECK(tree->node(0).visits == 13);
    CHECK(tree->node(0).leaf_q == -1);
    CHECK(tree->node(0).state == NodeState::Terminal);
    tree->node(0).contextual_draw = true;
    result = tree->re_root(0);
    REQUIRE(result);
    CHECK(result->discarded_context);
    CHECK(same_node(tree->node(0), Node{}));
}
