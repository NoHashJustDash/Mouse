// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstdint>

#include "engine/chess/position.h"
#include "engine/util/result.h"

namespace mouse::nn {
// Stateless access to immutable compile-time geometry; all calls are safe to run
// concurrently on independent Positions. index() validates policy encoding, not
// chess legality. move() accepts only a unique legal move in its board context;
// its caller must exclusively own the Position during the legality query.
class PolicyMap {
   public:
    static constexpr std::uint16_t kSize = 1858;

    [[nodiscard]] static Result<std::uint16_t> index(chess::Move move, chess::Color side);
    [[nodiscard]] static Result<chess::Move> move(std::uint16_t policy_index,
                                                  chess::Position& position,
                                                  const chess::AttackTables& attacks);
};
}  // namespace mouse::nn
