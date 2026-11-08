// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace mouse::search {

// Takes `count` slots of a run of `capacity` that threads share, or nothing if they do not fit. A
// failed attempt never moves `used`: adding first and taking it back could hand out held slots.
[[nodiscard]] inline std::optional<std::size_t> reserve_range(std::atomic_ref<std::size_t> used,
                                                              std::size_t count,
                                                              std::size_t capacity) noexcept {
    auto offset = used.load(std::memory_order_relaxed);
    do {
        if (count > capacity || offset > capacity - count) return std::nullopt;
    } while (!used.compare_exchange_weak(offset, offset + count, std::memory_order_relaxed));
    return offset;
}

// The tasks of one pick round, which threads take in order. The round number shares a word with
// the next task's index: a thread preempted between peek and claim can resume in any later round,
// and its claim must then fail. Bare counters reset each round would let it succeed.
class PickQueue {
   public:
    struct Ticket {
        std::uint64_t state = 0;
    };
    // Only with no task running. A thread that reads the new state also sees the emptied counters.
    void reset() noexcept {
        added_.store(0, std::memory_order_relaxed);
        done_.store(0, std::memory_order_relaxed);
        const auto round = state_.load(std::memory_order_relaxed) >> kIndexBits;
        state_.store((round + 1) << kIndexBits, std::memory_order_release);
    }
    // Makes the first `count` tasks available. Their contents must already be written.
    void publish(std::size_t count) noexcept { added_.store(count, std::memory_order_release); }
    [[nodiscard]] std::size_t size() const noexcept {
        return added_.load(std::memory_order_acquire);
    }
    [[nodiscard]] std::optional<Ticket> peek() const noexcept {
        const auto state = state_.load(std::memory_order_acquire);
        if ((state & kIndexMask) >= added_.load(std::memory_order_acquire)) return std::nullopt;
        return Ticket{state};
    }
    // The task's index, unless another thread moved the queue on since the ticket's peek.
    [[nodiscard]] std::optional<std::size_t> claim(Ticket ticket) noexcept {
        auto expected = ticket.state;
        if (!state_.compare_exchange_strong(expected, expected + 1, std::memory_order_acq_rel))
            return std::nullopt;
        return static_cast<std::size_t>(ticket.state & kIndexMask);
    }
    void complete() noexcept { done_.fetch_add(1, std::memory_order_release); }
    // Done is read first. Done never passes added, and a task can only be added by the coordinator
    // or a running task, so finding them equal in this order means nothing is left running.
    [[nodiscard]] bool drained() const noexcept {
        const auto done = done_.load(std::memory_order_acquire);
        return done == added_.load(std::memory_order_acquire);
    }

   private:
    static constexpr unsigned kIndexBits = 32;
    static constexpr std::uint64_t kIndexMask = (std::uint64_t{1} << kIndexBits) - 1;
    std::atomic<std::uint64_t> state_{0};
    std::atomic<std::size_t> added_{0}, done_{0};
};

}  // namespace mouse::search
