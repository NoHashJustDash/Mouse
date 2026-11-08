// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/search/pick_queue.h"

#include <array>
#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

using mouse::search::PickQueue;

// Threads of different sizes crowding a nearly full run: a failed attempt must not let a later one
// take slots that are already held.
TEST_CASE("Reserved ranges never overlap when the run is nearly full", "[search][pick]") {
    constexpr std::size_t kCapacity = 64, kThreads = 8;
    std::atomic<std::size_t> overlaps{0};
    for (int round = 0; round < 4000 && overlaps.load() == 0; ++round) {
        std::size_t used = 0;
        std::array<std::atomic<std::uint8_t>, kCapacity> held{};
        std::atomic<bool> start{false};
        std::vector<std::thread> threads;
        for (std::size_t t = 0; t < kThreads; ++t)
            threads.emplace_back([&, t] {
                while (!start.load(std::memory_order_acquire)) {
                }
                for (std::size_t attempt = 0; attempt < 40; ++attempt) {
                    const auto size = 1 + (t * 7 + attempt * 13) % 23;
                    const auto offset =
                        mouse::search::reserve_range(std::atomic_ref{used}, size, kCapacity);
                    if (!offset) continue;
                    for (std::size_t i = 0; i < size; ++i)
                        if (held.at(*offset + i).exchange(1) != 0) ++overlaps;
                }
            });
        start.store(true, std::memory_order_release);
        for (auto& thread : threads) thread.join();
    }
    CHECK(overlaps.load() == 0);
}

// The race behind game 28 of the c4108b1 match. A thread preempted between looking at the queue
// and claiming a task resumes in a later round. It took a stale task, and tasks done passed tasks
// added, so the coordinator waited forever for the two to meet.
TEST_CASE("A pick queue ticket from an earlier round claims nothing", "[search][pick]") {
    PickQueue queue;
    queue.reset();
    queue.publish(1);
    const auto stale = queue.peek();
    REQUIRE(stale);
    const auto running = queue.peek();
    REQUIRE(running);
    REQUIRE(queue.claim(*running) == 0);
    queue.complete();
    REQUIRE(queue.drained());

    SECTION("in a round with no tasks") {
        queue.reset();
        CHECK_FALSE(queue.claim(*stale));
        CHECK(queue.drained());
        CHECK_FALSE(queue.peek());
    }
    SECTION("in a round that has tasks") {
        queue.reset();
        queue.publish(2);
        CHECK_FALSE(queue.claim(*stale));
        const auto fresh = queue.peek();
        REQUIRE(fresh);
        CHECK(queue.claim(*fresh) == 0);
        CHECK_FALSE(queue.drained());
    }
}

TEST_CASE("A pick queue hands each task to one thread and drains at the end", "[search][pick]") {
    PickQueue queue;
    queue.reset();
    CHECK(queue.drained());
    CHECK_FALSE(queue.peek());
    queue.publish(2);
    CHECK(queue.size() == 2);
    CHECK_FALSE(queue.drained());

    const auto first = queue.peek();
    const auto rival = queue.peek();
    REQUIRE(first);
    REQUIRE(rival);
    CHECK(queue.claim(*first) == 0);
    CHECK_FALSE(queue.claim(*rival));

    // A running task may add another; a task claimed but not finished keeps the queue open.
    queue.publish(3);
    const auto second = queue.peek();
    REQUIRE(second);
    CHECK(queue.claim(*second) == 1);
    const auto third = queue.peek();
    REQUIRE(third);
    CHECK(queue.claim(*third) == 2);
    CHECK_FALSE(queue.peek());
    queue.complete();
    queue.complete();
    CHECK_FALSE(queue.drained());
    queue.complete();
    CHECK(queue.drained());
}
