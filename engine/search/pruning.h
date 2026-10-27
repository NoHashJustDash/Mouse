// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>

namespace mouse::search {
struct PruningSample {
    std::uint64_t new_nodes = 0, completed_batches = 0, remaining_nodes = 1;
    std::chrono::milliseconds elapsed{}, remaining_time{};
    std::span<const std::uint64_t> root_visits;
    std::size_t losing_moves = 0;
    bool win_found = false, may_resign = true;
};
// Source behavior: lc0 v0.32.1 classic/stoppers/stoppers.cc SmartPruningStopper.
// One search owner supplies explicit samples; no wall clock or allocation here.
class SmartPruning {
   public:
    explicit SmartPruning(float factor = 1.33F, std::uint64_t minimum_batches = 0);
    [[nodiscard]] bool should_stop(const PruningSample&) noexcept;
    [[nodiscard]] std::uint64_t estimated_remaining_playouts() const noexcept {
        return remaining_playouts_;
    }

   private:
    std::uint64_t remaining_playouts_ = std::numeric_limits<std::uint64_t>::max();
    float factor_;
    std::uint64_t minimum_batches_;
    std::optional<std::chrono::milliseconds> first_eval_;
};
}  // namespace mouse::search
