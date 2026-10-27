// SPDX-License-Identifier: GPL-3.0-or-later
#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "engine/chess/attacks.h"
#include "engine/chess/perft.h"

namespace {
struct PerftCase {
    std::string fen;
    std::string identifier;
    std::string variant;
    std::vector<std::pair<int, std::uint64_t>> counts;
};

std::string_view trim(std::string_view text) {
    constexpr std::string_view kWhitespace = " \t\r\n";
    const auto first = text.find_first_not_of(kWhitespace);
    if (first == std::string_view::npos) {
        return {};
    }
    return text.substr(first, text.find_last_not_of(kWhitespace) - first + 1);
}

template <typename Integer>
Integer integer(std::string_view text) {
    Integer result{};
    const auto parsed =
        std::from_chars(std::to_address(text.begin()), std::to_address(text.end()), result);
    if (text.empty() || parsed.ec != std::errc{} || parsed.ptr != std::to_address(text.end())) {
        throw std::runtime_error("Invalid perft fixture integer: " + std::string{text});
    }
    return result;
}

std::vector<PerftCase> read_suite() {
    std::ifstream input{MOUSE_TEST_PERFT_SUITE_PATH};
    if (!input) {
        throw std::runtime_error("Cannot open perft suite: " MOUSE_TEST_PERFT_SUITE_PATH);
    }
    std::vector<PerftCase> result;
    std::set<std::string> identifiers;
    for (std::string line; std::getline(input, line);) {
        const auto text = trim(line);
        if (text.empty() || text.front() == '#') {
            continue;
        }
        const auto separator = text.find(';');
        if (separator == std::string_view::npos || text.back() != ';') {
            throw std::runtime_error("Perft fixture requires semicolon-delimited fields: " + line);
        }
        PerftCase entry;
        entry.fen = trim(text.substr(0, separator));
        std::istringstream fields{std::string{text.substr(separator + 1)}};
        for (std::string field; std::getline(fields, field, ';');) {
            std::istringstream tokens{std::string{trim(field)}};
            std::string key;
            if (!(tokens >> key)) {
                continue;
            }
            if (key == "id") {
                tokens >> std::ws;
                if (!entry.identifier.empty() || tokens.peek() != '"' ||
                    !(tokens >> std::quoted(entry.identifier)) || entry.identifier.empty()) {
                    throw std::runtime_error("Invalid or duplicate perft fixture id: " + line);
                }
            } else if (key == "variant") {
                if (!entry.variant.empty() || !(tokens >> entry.variant) ||
                    (entry.variant != "standard" && entry.variant != "chess960")) {
                    throw std::runtime_error("Invalid or duplicate perft variant: " + line);
                }
            } else if (key.front() == 'D') {
                const auto depth = integer<int>(std::string_view{key}.substr(1));
                std::string count;
                if (depth < 1 || depth > mouse::chess::kMaxPerftDepth || !(tokens >> count) ||
                    std::ranges::any_of(
                        entry.counts, [depth](const auto& item) { return item.first == depth; })) {
                    throw std::runtime_error("Invalid or duplicate perft depth: " + line);
                }
                entry.counts.emplace_back(depth, integer<std::uint64_t>(count));
            } else {
                throw std::runtime_error("Unknown perft fixture field: " + key);
            }
            std::string extra;
            if (tokens >> extra) {
                throw std::runtime_error("Unexpected perft fixture token: " + extra);
            }
        }
        if (entry.identifier.empty() || entry.variant.empty() || entry.counts.empty() ||
            !identifiers.insert(entry.identifier).second) {
            throw std::runtime_error("Missing or repeated perft fixture metadata: " + line);
        }
        result.push_back(std::move(entry));
    }
    if (input.bad()) {
        throw std::runtime_error("Error reading perft suite");
    }
    return result;
}
}  // namespace

TEST_CASE("The full sourced perft suite matches every stored depth", "[chess][perft-gate]") {
    const mouse::chess::AttackTables attacks;
    const auto cases = read_suite();
    REQUIRE(cases.size() >= 100);
    struct Required {
        std::string_view identifier;
        int depth;
        std::uint64_t nodes;
    };
    constexpr std::array kRequired{
        Required{.identifier = "spec_start", .depth = 5, .nodes = 4865609},
        Required{.identifier = "spec_start", .depth = 6, .nodes = 119060324},
        Required{.identifier = "spec_kiwipete", .depth = 4, .nodes = 4085603},
        Required{.identifier = "spec_kiwipete", .depth = 5, .nodes = 193690690},
        Required{.identifier = "spec_rook_endgame", .depth = 6, .nodes = 11030083},
        Required{.identifier = "spec_promotions_castling", .depth = 5, .nodes = 15833292},
        Required{.identifier = "spec_promotion_checks", .depth = 4, .nodes = 2103487},
        Required{.identifier = "spec_middlegame", .depth = 5, .nodes = 164075551},
    };
    for (const auto& required : kRequired) {
        bool found = false;
        for (const auto& entry : cases) {
            if (entry.identifier == required.identifier) {
                const auto count = std::ranges::find_if(
                    entry.counts,
                    [&required](const auto& item) { return item.first == required.depth; });
                REQUIRE(count != entry.counts.end());
                REQUIRE(count->second == required.nodes);
                found = true;
            }
        }
        REQUIRE(found);
    }
    std::set<std::uint64_t> unique_positions;
    std::size_t chess960_cases = 0;
    std::size_t checks = 0;
    const auto start = std::chrono::steady_clock::now();
    for (const auto& entry : cases) {
        INFO("case " << entry.identifier << " FEN " << entry.fen);
        auto parsed = mouse::chess::Position::from_fen(entry.fen, attacks);
        if (!parsed) {
            FAIL(parsed.error());
        }
        auto position = std::move(*parsed);
        const auto before = position;
        unique_positions.insert(position.key());
        if (entry.variant == "chess960") {
            ++chess960_cases;
        }
        for (const auto& [depth, expected] : entry.counts) {
            INFO("depth " << depth << " expected " << expected);
            const auto actual = mouse::chess::perft(position, attacks, depth);
            if (!actual) {
                FAIL(actual.error());
            }
            CHECK(*actual == expected);
            REQUIRE(position == before);
            ++checks;
        }
    }
    REQUIRE(unique_positions.size() >= 100);
    REQUIRE(chess960_cases > 0);
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
    std::cout << "MOUSE_PERFT_RESULT positions=" << cases.size()
              << " unique=" << unique_positions.size() << " chess960=" << chess960_cases
              << " checks=" << checks << " elapsed_seconds=" << elapsed.count() << '\n';
}
