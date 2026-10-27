// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cstddef>
#include <span>
#include <vector>

#include "engine/nn/encoder.h"

namespace mouse::nn {
// One mutating owner stores consecutive legal plies. Immutable views may be
// shared by readers while the owner is not being mutated. reset/append/pop
// invalidate outstanding views. No Position or mutable global state is retained.
class EncodingHistory {
   public:
    // A FEN root retains its stated EP, even without a capturer. Allocation
    // failure leaves the previous history intact.
    void reset(const chess::Position& root);
    // The caller supplies the Position after a legal move from the last frame.
    // Alternation is checked; this is not a second legal move validator.
    // Appended EP is retained for geometric capturers, including pinned pawns.
    [[nodiscard]] Result<void> append(const chess::Position& position);
    [[nodiscard]] Result<void> pop();
    // Additional frames from the current sequence; reserve before replay. Failure
    // preserves contents. Charge actual capacity payload plus one 64-byte allocation
    // allowance, excluding object control which Search accounts separately.
    [[nodiscard]] Result<void> reserve_history(std::size_t additional_plies);
    [[nodiscard]] std::size_t history_storage_bytes() const noexcept;
    bool operator==(const EncodingHistory&) const = default;
    [[nodiscard]] std::span<const HistoryFrame> frames() const noexcept { return frames_; }

   private:
    [[nodiscard]] static bool same_identity(const HistoryFrame& left,
                                            const HistoryFrame& right) noexcept;
    [[nodiscard]] static chess::Square normalized_ep(const HistoryFrame& frame) noexcept;
    std::vector<HistoryFrame> frames_;
};
}  // namespace mouse::nn
