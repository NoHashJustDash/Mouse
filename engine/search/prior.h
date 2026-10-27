// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cstdint>
#include <span>

#include "engine/chess/types.h"
#include "engine/util/result.h"

namespace mouse::search {
// Stateless original IEEE binary16 conversion, round-to-nearest/ties-to-even.
// Priors are finite in [0,1]; negative zero is canonicalized to positive zero.
[[nodiscard]] Result<std::uint16_t> encode_prior(float value);
// Internal precondition: validated nonnegative prior bits <= 0x3c00.
[[nodiscard]] float decode_prior(std::uint16_t bits) noexcept;
// Legal moves must be unique, nonempty and <= MoveList::kCapacity. Only legal
// logit slots participate; the caller validates the complete NNOutput first.
// Output must match move count, and remains unchanged on every failure.
// Calls are independent/thread-safe when output buffers do not overlap.
// Already mapped legal logits, in exactly the caller's move order.
[[nodiscard]] Result<void> encode_legal_logits(std::span<const float> logits, double temperature,
                                               std::span<std::uint16_t> output);
[[nodiscard]] Result<void> encode_legal_priors(std::span<const chess::Move> moves,
                                               chess::Color side,
                                               std::span<const float, 1858> logits,
                                               double temperature, std::span<std::uint16_t> output);
}  // namespace mouse::search
