// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/nn/policy_map.h"

#include <array>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>

#include "engine/chess/movegen.h"

namespace mouse::nn {
namespace {
constexpr int magnitude(int value) noexcept { return value < 0 ? -value : value; }

struct PolicyIndices {
    std::array<std::array<std::int16_t, 64>, 64> ordinary{};
    std::array<std::array<std::array<std::int16_t, 3>, 8>, 8> promotions{};
    std::uint16_t count = 0;
};

// Original geometric construction, not a copy of Lc0's 1858-entry move table.
// Ordering/packing reference: lc0 fd71a2d921b689c5f479d3227c3806c8e272d9c5,
// src/neural/encoder.cc kMoveStrs/MoveToNNIndex and src/chess/types.h piece IDs.
// That pinned format has 1792 ordinary queen-ray/knight slots followed by 66
// explicit Q/R/B promotions. Knight promotion uses its ordinary coordinate slot.
consteval PolicyIndices make_policy_indices() {
    PolicyIndices result;
    for (auto& destinations : result.ordinary) destinations.fill(-1);
    for (auto& destinations : result.promotions)
        for (auto& pieces : destinations) pieces.fill(-1);
    for (chess::Square from = 0; from < 64; ++from) {
        for (chess::Square to = 0; to < 64; ++to) {
            const int file_distance = magnitude(chess::file_of(to) - chess::file_of(from));
            const int rank_distance = magnitude(chess::rank_of(to) - chess::rank_of(from));
            const bool queen =
                file_distance == 0 || rank_distance == 0 || file_distance == rank_distance;
            const bool knight = (file_distance == 1 && rank_distance == 2) ||
                                (file_distance == 2 && rank_distance == 1);
            if (from != to && (queen || knight))
                result.ordinary.at(from).at(to) = static_cast<std::int16_t>(result.count++);
        }
    }
    if (result.count != 1792) throw "Policy ordinary geometry has an incorrect size";
    for (std::size_t from_file = 0; from_file < 8; ++from_file) {
        for (std::size_t to_file = 0; to_file < 8; ++to_file) {
            if (magnitude(static_cast<int>(to_file) - static_cast<int>(from_file)) > 1) continue;
            for (auto& piece : result.promotions.at(from_file).at(to_file))
                piece = static_cast<std::int16_t>(result.count++);
        }
    }
    return result;
}
inline constexpr PolicyIndices kIndices = make_policy_indices();
static_assert(kIndices.count == PolicyMap::kSize);

chess::Square canonical_square(chess::Square square, chess::Color side) {
    return side == chess::Color::White ? square : static_cast<chess::Square>(square ^ 56U);
}
}  // namespace

Result<std::uint16_t> PolicyMap::index(chess::Move move, chess::Color side) {
    if (side != chess::Color::White && side != chess::Color::Black)
        return tl::unexpected(std::string{"Policy move has an invalid side-to-move enum"});
    const auto kind = move.kind();
    if (kind != chess::MoveKind::Normal && kind != chess::MoveKind::EnPassant &&
        kind != chess::MoveKind::Castle)
        return tl::unexpected(std::string{"Policy move has an invalid move-kind enum"});
    const auto promotion = move.promotion();
    if (promotion != chess::PieceType::None && promotion != chess::PieceType::Knight &&
        promotion != chess::PieceType::Bishop && promotion != chess::PieceType::Rook &&
        promotion != chess::PieceType::Queen)
        return tl::unexpected(std::string{"Policy move has an invalid promotion piece"});
    if (move.is_promotion() && kind != chess::MoveKind::Normal)
        return tl::unexpected(
            std::string{"Policy promotion cannot also be castling or en passant"});
    const auto from = canonical_square(move.from(), side);
    const auto to = canonical_square(move.to(), side);
    const int file_distance = magnitude(chess::file_of(to) - chess::file_of(from));
    if (move.is_promotion() &&
        (chess::rank_of(from) != 6 || chess::rank_of(to) != 7 || file_distance > 1))
        return tl::unexpected(std::string{
            "Policy promotion requires a rank-seven pawn move to an adjacent rank-eight square"});
    if (kind == chess::MoveKind::Castle &&
        (chess::rank_of(from) != 0 || chess::rank_of(to) != 0 || from == to))
        return tl::unexpected(
            std::string{"Policy castling must use distinct home-rank king and rook squares"});
    if (kind == chess::MoveKind::EnPassant &&
        (chess::rank_of(from) != 4 || chess::rank_of(to) != 5 || file_distance != 1))
        return tl::unexpected(
            std::string{"Policy en-passant move has invalid pawn-capture geometry"});
    std::int16_t slot = -1;
    if (promotion == chess::PieceType::Queen || promotion == chess::PieceType::Rook ||
        promotion == chess::PieceType::Bishop) {
        const std::size_t piece = promotion == chess::PieceType::Queen  ? 0
                                  : promotion == chess::PieceType::Rook ? 1
                                                                        : 2;
        slot = kIndices.promotions.at(static_cast<std::size_t>(chess::file_of(from)))
                   .at(static_cast<std::size_t>(chess::file_of(to)))
                   .at(piece);
    } else
        slot = kIndices.ordinary.at(from).at(to);
    if (slot < 0)
        return tl::unexpected(std::string{"Move has no entry in the 1858-slot policy geometry"});
    return static_cast<std::uint16_t>(slot);
}

Result<chess::Move> PolicyMap::move(std::uint16_t policy_index, chess::Position& position,
                                    const chess::AttackTables& attacks) {
    if (policy_index >= kSize)
        return tl::unexpected("Policy index must be less than " + std::to_string(kSize));
    try {
        std::optional<chess::Move> match;
        for (const auto candidate : chess::legal_moves(position, attacks)) {
            const auto slot = index(candidate, position.side_to_move());
            if (!slot)
                return tl::unexpected("Legal move cannot be policy-encoded: " + slot.error());
            if (*slot != policy_index) continue;
            if (match)
                return tl::unexpected(
                    std::string{"Policy index is ambiguous among the position's legal moves"});
            match = candidate;
        }
        if (!match)
            return tl::unexpected(
                std::string{"Policy index does not identify a legal move in this position"});
        return *match;
    } catch (const std::overflow_error& error) {
        return tl::unexpected(std::string{"Cannot decode policy move: "} + error.what());
    }
}
}  // namespace mouse::nn
