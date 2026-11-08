// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/search/moves_left.h"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <limits>

namespace {
mouse::search::Node completed(double q, std::optional<double> moves_left) {
    mouse::search::Node node;
    node.visits = 1;
    node.value_sum = q;
    if (moves_left) {
        node.moves_left_samples = 1;
        node.moves_left_sum = *moves_left;
    }
    return node;
}
}  // namespace
TEST_CASE("Moves-left utility requires a strict parent threshold and complete samples",
          "[search][moves-left]") {
    auto parent = completed(0.95, 30);
    auto child = completed(-0.95, 20);
    CHECK(mouse::search::moves_left_utility(parent, child) > 0);
    parent.value_sum = 0.8;
    CHECK(mouse::search::moves_left_utility(parent, child) == 0);
    parent.value_sum = -0.8;
    CHECK(mouse::search::moves_left_utility(parent, child) == 0);
    parent.value_sum = std::nextafter(0.8, 1.0);
    CHECK(mouse::search::moves_left_utility(parent, child) > 0);
    parent.value_sum = std::nextafter(-0.8, -1.0);
    CHECK(mouse::search::moves_left_utility(parent, child) > 0);
    parent.value_sum = 0.95;
    child.moves_left_samples = 0;
    CHECK(mouse::search::moves_left_utility(parent, child) == 0);
    child.moves_left_samples = 1;
    parent.moves_left_samples = 0;
    CHECK(mouse::search::moves_left_utility(parent, child) == 0);
    parent.moves_left_samples = 1;
    child.visits = 2;
    CHECK(mouse::search::moves_left_utility(parent, child) == 0);
    child = completed(-0.95, 20);
    parent.visits = 2;
    CHECK(mouse::search::moves_left_utility(parent, child) == 0);
    parent = completed(0.95, 30);
    child = mouse::search::Node{};
    CHECK(mouse::search::moves_left_utility(parent, child) == 0);
}
TEST_CASE("Moves-left utility has parent-view sign, cap and quadratic confidence",
          "[search][moves-left]") {
    const auto parent = completed(0.95, 30);
    auto child = completed(-0.95, 20);
    CHECK(std::abs(mouse::search::moves_left_utility(parent, child) - 0.02355125625) < 1e-14);
    child = completed(-1, 20);
    CHECK(std::abs(mouse::search::moves_left_utility(parent, child) - 0.027) < 1e-15);
    child = completed(-1, 0);
    CHECK(std::abs(mouse::search::moves_left_utility(parent, child) - 0.0345) < 1e-15);
    child = completed(-1, 100);
    CHECK(std::abs(mouse::search::moves_left_utility(parent, child) + 0.0345) < 1e-15);
    child = completed(1, 100);
    CHECK(std::abs(mouse::search::moves_left_utility(parent, child) - 0.0345) < 1e-15);
    child = completed(1, 0);
    CHECK(std::abs(mouse::search::moves_left_utility(parent, child) + 0.0345) < 1e-15);
    child = completed(-0.8, 0);
    CHECK(mouse::search::moves_left_utility(parent, child) == 0);
    child.value_sum = std::nextafter(-0.8, -1.0);
    CHECK(mouse::search::moves_left_utility(parent, child) > 0);
    child = completed(0, 0);
    CHECK(mouse::search::moves_left_utility(parent, child) == 0);
    child = completed(-0.95, 30);
    CHECK(mouse::search::moves_left_utility(parent, child) == 0);
    auto averaged_parent = completed(0.95, 30), averaged_child = completed(-0.95, 20);
    averaged_parent.visits = averaged_parent.moves_left_samples = 4;
    averaged_parent.value_sum *= 4;
    averaged_parent.moves_left_sum *= 4;
    averaged_child.visits = averaged_child.moves_left_samples = 2;
    averaged_child.value_sum *= 2;
    averaged_child.moves_left_sum *= 2;
    CHECK(std::abs(mouse::search::moves_left_utility(averaged_parent, averaged_child) -
                   0.02355125625) < 1e-14);
}
TEST_CASE("Moves-left utility follows lc0's MovesLeft* options", "[search][moves-left]") {
    // Edge Q 0.95 from the parent's view, child M 10 plies shorter: base term
    // clamp(slope * -10, -cap, cap) times the sign of -Q, then the confidence factor.
    const auto parent = completed(0.95, 30);
    const auto child = completed(-0.95, 20);
    using mouse::search::MovesLeftParameters;
    const auto utility = [&](const MovesLeftParameters& parameters) {
        return mouse::search::moves_left_utility(parent, child, parameters);
    };
    CHECK(utility({}) == mouse::search::moves_left_utility(parent, child));
    CHECK(std::abs(utility({.constant_factor = 0.5, .scaled_factor = 0, .quadratic_factor = 0}) -
                   0.0135) < 1e-15);
    CHECK(std::abs(utility({.max_effect = 0.01}) - 0.0087226875) < 1e-15);
    CHECK(std::abs(utility({.slope = 0.002}) - 0.017445375) < 1e-15);
    CHECK(std::abs(utility({.threshold = 0.9}) - 0.017901675) < 1e-12);
    // A threshold of 0 uses the raw Q (1.6521*0.95 - 0.6521*0.95^2).
    CHECK(std::abs(utility({.threshold = 0}) - 0.02648631825) < 1e-14);
    // The parent must exceed the threshold.
    CHECK(utility({.threshold = 0.96}) == 0);
    CHECK(utility({.threshold = 1}) == 0);
    CHECK(utility({.max_effect = 0}) == 0);
}
TEST_CASE("Moves-left options reject values outside lc0's ranges", "[search][moves-left]") {
    using mouse::search::validate_moves_left_parameters;
    CHECK(validate_moves_left_parameters({}));
    CHECK(validate_moves_left_parameters({.max_effect = 1,
                                          .threshold = 0,
                                          .slope = 1,
                                          .constant_factor = -1,
                                          .scaled_factor = 2,
                                          .quadratic_factor = 1}));
    CHECK_FALSE(validate_moves_left_parameters({.max_effect = -0.1}));
    CHECK_FALSE(validate_moves_left_parameters({.threshold = 1.5}));
    CHECK_FALSE(validate_moves_left_parameters({.slope = 2}));
    CHECK_FALSE(validate_moves_left_parameters({.constant_factor = 1.5}));
    CHECK_FALSE(validate_moves_left_parameters({.scaled_factor = -2.5}));
    CHECK_FALSE(validate_moves_left_parameters({.quadratic_factor = std::nan("")}));
}
TEST_CASE("Moves-left samples back up one ply without sign flips or invented heads",
          "[search][moves-left]") {
    std::array<mouse::search::Node, 3> path{};
    std::optional<double> sample = 0.0;
    for (std::size_t i = path.size(); i > 0; --i) {
        auto& node = path.at(i - 1);
        ++node.visits;
        mouse::search::backup_moves_left_sample(node, sample);
    }
    CHECK(path[2].moves_left_sum == 0);
    CHECK(path[1].moves_left_sum == 1);
    CHECK(path[0].moves_left_sum == 2);
    CHECK(sample == 3.0);
    for (const auto& node : path) {
        CHECK(node.moves_left_samples == 1);
        CHECK(node.visits == 1);
    }
    sample.reset();
    for (auto& node : path) {
        ++node.visits;
        mouse::search::backup_moves_left_sample(node, sample);
        CHECK(node.moves_left_samples == 1);
        CHECK(node.visits == 2);
    }
    CHECK_FALSE(sample);
    CHECK(path[0].moves_left_sum == 2);
    auto frontier = completed(-1, 4.5);
    frontier.state = mouse::search::NodeState::Frontier;
    frontier.leaf_moves_left = 4.5F;
    sample = frontier.leaf_moves_left;
    ++frontier.visits;
    mouse::search::backup_moves_left_sample(frontier, sample);
    CHECK(frontier.moves_left_samples == 2);
    CHECK(frontier.moves_left_sum == 9.0);
    CHECK(sample == 5.5);
}
TEST_CASE("Tree compaction retains complete and unknown moves-left statistics exactly",
          "[search][moves-left]") {
    auto created = mouse::search::Tree::create(65536);
    REQUIRE(created);
    auto& tree = **created;
    const auto child = tree.allocate_node();
    REQUIRE(child);
    const std::array seed{
        mouse::search::EdgeSeed{.move = mouse::chess::Move{0, 1}, .prior_bits = 0x3c00}};
    REQUIRE(tree.expand(0, seed));
    tree.edges(0)[0].child = *child;
    auto& node = tree.node(*child);
    node = completed(-0.95, 7.25);
    node.state = mouse::search::NodeState::Frontier;
    node.leaf_moves_left = 7.25F;
    SECTION("Complete known head") {}
    SECTION("Mixed missing head") {
        node.visits = 2;
        node.leaf_moves_left.reset();
    }
    const auto cached = node.leaf_moves_left;
    const auto visits = node.visits;
    REQUIRE(tree.re_root(*child));
    CHECK(tree.node(0).moves_left_sum == 7.25);
    CHECK(tree.node(0).moves_left_samples == 1);
    CHECK(tree.node(0).leaf_moves_left == cached);
    CHECK(tree.node(0).visits == visits);
    REQUIRE(tree.reset());
    CHECK(tree.node(0).moves_left_sum == 0);
    CHECK(tree.node(0).moves_left_samples == 0);
    CHECK_FALSE(tree.node(0).leaf_moves_left);
    CHECK(tree.stats().charged_bytes <= tree.stats().budget_bytes);
}

TEST_CASE("Rule terminals start at zero remaining plies and tree twofolds at their cycle",
          "[search][moves-left]") {
    // Lc0 fd71a2d classic/search.cc ExtendNode: rule terminals have M = 0 and a
    // tree twofold uses its cycle length (plies since the earlier occurrence).
    for (const std::uint16_t cycle : {std::uint16_t{0}, std::uint16_t{4}}) {
        mouse::search::Node terminal, parent;
        terminal.state = mouse::search::NodeState::Terminal;
        terminal.twofold = cycle != 0;
        terminal.leaf_moves_left = static_cast<float>(cycle);
        for (std::size_t i = 0; i < 300; ++i) {
            std::optional<double> sample = terminal.leaf_moves_left;
            ++terminal.visits;
            mouse::search::backup_moves_left_sample(terminal, sample);
            ++parent.visits;
            mouse::search::backup_moves_left_sample(parent, sample);
        }
        CHECK(terminal.moves_left_sum == 300.0 * cycle);
        CHECK(terminal.moves_left_samples == 300);
        CHECK(parent.moves_left_sum == 300.0 * (cycle + 1));
        CHECK(parent.moves_left_samples == 300);
    }
}
