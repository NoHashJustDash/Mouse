// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/uci/time.h"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>

#include "tests/required_optional.h"

namespace {
using mouse::chess::Color;
using mouse::tests::required_optional;
using mouse::uci::GoCommand;
using Milliseconds = std::chrono::milliseconds;
constexpr auto kMaximum = std::numeric_limits<std::uint64_t>::max();
mouse::Result<mouse::uci::TimeBudget> allocate_time(const GoCommand& command, Color side,
                                                    std::uint64_t overhead) {
    return mouse::uci::LegacyTimeManager{}.allocate(
        command, {.side_to_move = side, .overhead_ms = overhead});
}
mouse::Result<mouse::uci::TimeBudget> allocate_after_ponderhit(const GoCommand& command, Color side,
                                                               std::uint64_t overhead) {
    auto active = command;
    active.ponder = false;
    return allocate_time(active, side, overhead);
}
}  // namespace

TEST_CASE("Legacy allocation uses the active clock and native book-move budget", "[uci][time]") {
    GoCommand command;
    command.wtime = 3030;
    command.btime = 10030;
    command.winc = 10;
    command.binc = 100;
    const auto white = allocate_time(command, Color::White, 30);
    const auto black = allocate_time(command, Color::Black, 30);
    REQUIRE(white);
    REQUIRE(black);
    CHECK(white->budget == Milliseconds{209});
    CHECK(black->budget == Milliseconds{897});
    CHECK_FALSE(white->hold_until_signal);
    command.movestogo = 10;
    CHECK(allocate_time(command, Color::White, 30)->budget == Milliseconds{942});
    command.movestogo = 1;
    CHECK(allocate_time(command, Color::White, 30)->budget == Milliseconds{3000});
    command.movestogo = 10;
    command.winc = 1;
    CHECK(allocate_time(command, Color::White, 30)->budget == Milliseconds{917});
    command.winc = 5;
    CHECK(allocate_time(command, Color::White, 30)->budget == Milliseconds{928});
    command.winc = 6;
    CHECK(allocate_time(command, Color::White, 30)->budget == Milliseconds{931});
    command.btime = kMaximum;
    command.binc = kMaximum;
    CHECK(allocate_time(command, Color::White, 30)->budget == Milliseconds{931});
}

TEST_CASE("Fixed movetime subtracts overhead and coexists with the clock stopper", "[uci][time]") {
    GoCommand command;
    command.movetime = 500;
    command.wtime = 1;
    command.binc = kMaximum;
    command.nodes = 1;
    command.depth = 2;
    command.mate = 3;
    const auto result = allocate_time(command, Color::White, kMaximum);
    REQUIRE(result);
    CHECK(result->budget == Milliseconds{1});
    command.wtime.reset();
    command.binc.reset();
    CHECK(allocate_time(command, Color::White, 30)->budget == Milliseconds{470});
    CHECK_FALSE(result->hold_until_signal);
    command.movetime = kMaximum;
    CHECK(allocate_time(command, Color::Black, 30)->budget == Milliseconds::max());
    command.movetime = 1;
    CHECK(allocate_time(command, Color::Black, 30)->budget == Milliseconds{1});
}

TEST_CASE("Zero and overhead-exhausted clocks get exactly one emergency millisecond",
          "[uci][time]") {
    GoCommand command;
    for (const auto remaining : {0U, 1U, 29U, 30U, 31U}) {
        command.wtime = remaining;
        const auto result = allocate_time(command, Color::White, 30);
        REQUIRE(result);
        CHECK(result->budget == Milliseconds{1});
        CHECK_FALSE(result->hold_until_signal);
    }
    command.wtime = kMaximum;
    command.winc = kMaximum;
    CHECK(allocate_time(command, Color::White, kMaximum)->budget == Milliseconds{1});
}

TEST_CASE("Unbounded go and independent compute limits have no invented node or time bound",
          "[uci][time]") {
    GoCommand command;
    const auto unbounded = allocate_time(command, Color::White, 30);
    REQUIRE(unbounded);
    CHECK_FALSE(unbounded->budget);
    CHECK(unbounded->hold_until_signal);
    CHECK_FALSE(command.nodes);
    command.nodes = 1;
    const auto nodes = allocate_time(command, Color::White, 30);
    REQUIRE(nodes);
    CHECK_FALSE(nodes->budget);
    CHECK_FALSE(nodes->hold_until_signal);
    command.nodes.reset();
    command.depth = 1;
    CHECK_FALSE(allocate_time(command, Color::White, 30)->hold_until_signal);
    command.depth.reset();
    command.mate = 1;
    CHECK_FALSE(allocate_time(command, Color::White, 30)->hold_until_signal);
    command.wtime = 3030;
    CHECK(allocate_time(command, Color::White, 30)->budget == Milliseconds{179});
}

TEST_CASE("Ponder suspends timing until hit without mutating the original request", "[uci][time]") {
    GoCommand command;
    command.ponder = true;
    command.wtime = 3030;
    command.winc = 10;
    command.nodes = 100;
    const auto waiting = allocate_time(command, Color::White, 30);
    REQUIRE(waiting);
    CHECK_FALSE(waiting->budget);
    CHECK(waiting->hold_until_signal);
    const auto hit = allocate_after_ponderhit(command, Color::White, 30);
    REQUIRE(hit);
    CHECK(hit->budget == Milliseconds{209});
    CHECK_FALSE(hit->hold_until_signal);
    CHECK(command.ponder);
    CHECK(command.nodes == 100);
    CHECK(command.wtime == 3030);
    command.movetime = 7;
    CHECK(allocate_after_ponderhit(command, Color::White, 30)->budget == Milliseconds{1});
    command.movetime.reset();
    command.wtime.reset();
    command.winc.reset();
    CHECK_FALSE(allocate_after_ponderhit(command, Color::White, 30)->hold_until_signal);
}

TEST_CASE("Infinite precedence survives ponderhit and suspends every timed bound", "[uci][time]") {
    GoCommand command;
    command.infinite = true;
    command.ponder = true;
    command.movetime = 7;
    command.wtime = 3030;
    command.nodes = 100;
    for (const auto& result : {allocate_time(command, Color::White, 30),
                               allocate_after_ponderhit(command, Color::White, 30)}) {
        REQUIRE(result);
        CHECK_FALSE(result->budget);
        CHECK(result->hold_until_signal);
    }
}

TEST_CASE("Clock allocation rejects missing active clocks and invalid semantic time values",
          "[uci][time]") {
    GoCommand command;
    command.btime = 1000;
    CHECK_FALSE(allocate_time(command, Color::White, 30));
    command.btime.reset();
    command.winc = 1;
    CHECK_FALSE(allocate_time(command, Color::White, 30));
    command.winc.reset();
    command.movestogo = 10;
    CHECK_FALSE(allocate_time(command, Color::White, 30));
    command.wtime = 1000;
    command.movestogo = 0;
    CHECK_FALSE(allocate_time(command, Color::White, 30));
    command.movestogo.reset();
    command.movetime = 0;
    CHECK_FALSE(allocate_time(command, Color::White, 30));
    command.movetime.reset();
    CHECK_FALSE(allocate_time(command, static_cast<Color>(2), 30));
    command.wtime.reset();
    command.btime = 1000;
    command.ponder = true;
    CHECK(allocate_time(command, Color::White, 30));
    CHECK_FALSE(allocate_after_ponderhit(command, Color::White, 30));
}

TEST_CASE("Clock arithmetic stays positive bounded and overflow-safe at uint64 extremes",
          "[uci][time]") {
    constexpr std::array<std::uint64_t, 7> remaining{0, 1, 29, 30, 31, 1000, kMaximum};
    constexpr std::array<std::uint64_t, 3> overheads{0, 30, kMaximum};
    constexpr std::array<std::uint64_t, 5> increments{0, 1, 4, 5, kMaximum};
    constexpr std::array<std::uint64_t, 3> horizons{1, 30, kMaximum};
    for (const auto rem : remaining)
        for (const auto overhead : overheads)
            for (const auto inc : increments)
                for (const auto horizon : horizons) {
                    GoCommand command;
                    command.wtime = rem;
                    command.winc = inc;
                    command.movestogo = horizon;
                    const auto result = allocate_time(command, Color::White, overhead);
                    REQUIRE(result);
                    const auto budget = result->budget;
                    REQUIRE(budget);
                    if (budget) {
                        const auto duration = *budget;
                        CHECK(duration.count() >= 1);
                        CHECK_FALSE(result->hold_until_signal);
                        if (rem <= overhead)
                            CHECK(duration == Milliseconds{1});
                        else
                            CHECK(static_cast<std::uint64_t>(duration.count()) <= rem - overhead);
                    }
                }
    GoCommand command;
    command.wtime = kMaximum;
    command.movestogo = 1;
    CHECK(allocate_time(command, Color::White, 0)->budget == Milliseconds::max());
    command.movestogo = 30;
    CHECK(allocate_time(command, Color::White, 0)->budget == Milliseconds{937709558216785920});
    command.winc = kMaximum;
    CHECK(allocate_time(command, Color::White, 0)->budget == Milliseconds::max());
    command.winc = 0;
    command.movestogo = kMaximum;
    // Out-of-reference uint64 clocks saturate to int64 before native float math.
    CHECK(allocate_time(command, Color::White, 0)->budget == Milliseconds{551593819372519424});
}

TEST_CASE("Timed batch forecasts retain the absolute allocation and bounded bucket",
          "[uci][time][deadline-schedule]") {
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::time_point{std::chrono::seconds{100}};
    constexpr std::array<std::size_t, 4> buckets{1, 8, 32, 64};
    const mouse::uci::TimeBudget budget{.budget = Milliseconds{100}, .hold_until_signal = false};
    const auto schedule = mouse::uci::schedule_search(budget, start, start + Milliseconds{20},
                                                      Milliseconds{20}, buckets, 64);
    CHECK_FALSE(schedule.exhausted);
    CHECK(schedule.maximum_batch == 1);
    REQUIRE(schedule.deadline);
    CHECK(required_optional(schedule.deadline) == start + Milliseconds{100});
    const auto late = mouse::uci::schedule_search(budget, start, start + Milliseconds{61},
                                                  Milliseconds{20}, buckets, 64);
    CHECK(late.exhausted);
    const auto expired = mouse::uci::schedule_search(budget, start, start + Milliseconds{101},
                                                     Milliseconds{1}, buckets, 64);
    CHECK(expired.exhausted);
    const auto throughput = mouse::uci::schedule_search(budget, start, start + Milliseconds{1},
                                                        Milliseconds{1}, buckets, 64);
    CHECK_FALSE(throughput.exhausted);
    CHECK(throughput.maximum_batch == 32);
    CHECK(throughput.deadline == start + Milliseconds{100});
    const auto untimed = mouse::uci::schedule_search({}, start, start + std::chrono::hours{1},
                                                     Milliseconds{20}, buckets, 64);
    CHECK_FALSE(untimed.exhausted);
    CHECK_FALSE(untimed.deadline);
    CHECK(untimed.maximum_batch == 64);
    const auto immediate =
        mouse::uci::schedule_search({.budget = Milliseconds{1}, .hold_until_signal = false}, start,
                                    start, Clock::duration{}, buckets, 64);
    CHECK(immediate.exhausted);
    const auto saturated = mouse::uci::schedule_search(
        {.budget = Milliseconds::max(), .hold_until_signal = false},
        Clock::time_point::max() - Milliseconds{100}, Clock::time_point::max() - Milliseconds{90},
        Milliseconds{1}, buckets, 64);
    REQUIRE(saturated.deadline);
    CHECK(required_optional(saturated.deadline) <= Clock::time_point::max());
    CHECK_FALSE(saturated.exhausted);
}

TEST_CASE("Recent physical root paths bound batches without shortening the allocation",
          "[uci][time][latency-recovery]") {
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::time_point{std::chrono::seconds{100}};
    constexpr std::array<std::size_t, 4> buckets{1, 8, 32, 64};
    const mouse::uci::TimeBudget budget{.budget = Milliseconds{41}, .hold_until_signal = false};
    mouse::uci::LatencyWindow latency;
    CHECK(latency.estimate() == Clock::duration::zero());
    latency.observe(Milliseconds{100});
    CHECK(mouse::uci::schedule_search(budget, start, start + Milliseconds{1}, latency.estimate(),
                                      buckets, 64)
              .maximum_batch == 1);
    for (unsigned sample = 0; sample < 7; ++sample) {
        latency.observe(Milliseconds{1});
        CHECK(latency.estimate() == Milliseconds{100});
    }
    // No physical observation means no eviction; invalid durations also cannot
    // turn a known slow path into optimistic evidence.
    for (unsigned sample = 0; sample < 20; ++sample) {
        latency.observe(Clock::duration::zero());
        latency.observe(Milliseconds{-1});
        CHECK(latency.estimate() == Milliseconds{100});
    }
    latency.observe(Milliseconds{1});
    const auto recovered = mouse::uci::schedule_search(budget, start, start + Milliseconds{1},
                                                       latency.estimate(), buckets, 64);
    CHECK_FALSE(recovered.exhausted);
    CHECK(recovered.maximum_batch == 8);
    CHECK(recovered.deadline == start + Milliseconds{41});
    latency.observe(Milliseconds{100});
    CHECK(mouse::uci::schedule_search(budget, start, start + Milliseconds{1}, latency.estimate(),
                                      buckets, 64)
              .maximum_batch == 1);
    latency.clear();
    CHECK(latency.estimate() == Clock::duration::zero());
    latency.observe(Clock::duration::max());
    CHECK(mouse::uci::schedule_search(budget, start, start + Milliseconds{1}, latency.estimate(),
                                      buckets, 64)
              .maximum_batch == 1);
    for (unsigned sample = 0; sample < 8; ++sample) latency.observe(Milliseconds{1});
    CHECK(latency.estimate() == Milliseconds{1});
    const auto saturated = mouse::uci::schedule_search(
        {.budget = Milliseconds::max(), .hold_until_signal = false},
        Clock::time_point::max() - Milliseconds{100}, Clock::time_point::max() - Milliseconds{90},
        latency.estimate(), buckets, 64);
    CHECK_FALSE(saturated.exhausted);
    REQUIRE(saturated.deadline);
    CHECK(required_optional(saturated.deadline) <= Clock::time_point::max());
}

TEST_CASE("Legacy first move matches the real lc0 allocation rather than a fixed horizon",
          "[uci][time][legacy-v1]") {
    GoCommand command;
    command.wtime = 1000;
    // Official v0.32.1 native logfile, internal game ply1, SmartPruningFactor0:
    // 59.8039ms without increment,89.7059ms with10ms increment.
    CHECK(allocate_time(command, Color::White, 0)->budget == Milliseconds{59});
    command.winc = 10;
    CHECK(allocate_time(command, Color::White, 0)->budget == Milliseconds{89});
    command.movestogo = 40;
    CHECK(allocate_time(command, Color::White, 0)->budget == Milliseconds{105});
}

TEST_CASE("Legacy saved time is used once and new games restore the book bonus",
          "[uci][time][legacy-v1]") {
    mouse::uci::LegacyTimeManager manager;
    GoCommand command;
    command.wtime = 1000;
    command.winc = 10;
    const auto first =
        manager.allocate(command, {.side_to_move = Color::White, .overhead_ms = 0, .game_ply = 1});
    REQUIRE(first);
    REQUIRE(first->clock_budget == Milliseconds{89});
    manager.complete(*first, Milliseconds{20});
    // 69ms saved. Native curve at ply2 is50.5: (1495-69)/50.5+69=97.2376.
    const auto next =
        manager.allocate(command, {.side_to_move = Color::White, .overhead_ms = 0, .game_ply = 2});
    REQUIRE(next);
    CHECK(next->budget == Milliseconds{97});
    manager.complete(*next, Milliseconds{120});
    const auto indebted =
        manager.allocate(command, {.side_to_move = Color::White, .overhead_ms = 0, .game_ply = 3});
    REQUIRE(indebted);
    CHECK(indebted->budget == Milliseconds{29});
    manager.reset();
    CHECK(
        manager.allocate(command, {.side_to_move = Color::White, .overhead_ms = 0, .game_ply = 1})
            ->budget == Milliseconds{89});
}
TEST_CASE("Legacy keeps its own limit when movetime ends a search early",
          "[uci][time][legacy-v1]") {
    mouse::uci::LegacyTimeManager manager;
    GoCommand command;
    command.wtime = 1000;
    command.winc = 10;
    command.movetime = 30;
    const auto first =
        manager.allocate(command, {.side_to_move = Color::White, .overhead_ms = 0, .game_ply = 1});
    REQUIRE(first);
    CHECK(first->budget == Milliseconds{30});
    CHECK(first->clock_budget == Milliseconds{89});
    manager.complete(*first, Milliseconds{20});
    command.movetime.reset();
    CHECK(
        manager.allocate(command, {.side_to_move = Color::White, .overhead_ms = 0, .game_ply = 2})
            ->budget == Milliseconds{97});
}

TEST_CASE("Legacy allocation matches96 actual lc0 clock and book-ply cases",
          "[uci][time][legacy-v1]") {
    struct Case {
        std::uint64_t clock, increment, movestogo, ply;
        double logged_ms;
    };
    static constexpr auto cases = std::to_array<Case>({
#include "tests/data/uci/time-legacy-v1/allocations.inc"
    });
    STATIC_REQUIRE(std::size(cases) == 96);
    for (const auto& item : cases) {
        mouse::uci::LegacyTimeManager manager;
        GoCommand command;
        command.wtime = item.clock;
        command.winc = item.increment;
        if (item.movestogo != 0) command.movestogo = item.movestogo;
        const auto allocated = manager.allocate(
            command, {.side_to_move = Color::White, .overhead_ms = 0, .game_ply = item.ply});
        REQUIRE(allocated);
        REQUIRE(allocated->clock_budget);
        const auto logged_limit = std::min(item.logged_ms, static_cast<double>(item.clock));
        INFO("clock=" << item.clock << " inc=" << item.increment << " mgt=" << item.movestogo
                      << " native internal ply=" << item.ply);
        // Native logger uses6 significant decimal digits; allocations reach
        // 375769ms, so its printed values can round by a millisecond. This
        // scalar capture check does not replace the required elapsed-time gate.
        CHECK(std::abs(static_cast<double>(required_optional(allocated->clock_budget).count()) -
                       logged_limit) <= 1.0);
    }
}

TEST_CASE("Batch forecasting preserves the full legacy deadline with a slow accepted provider",
          "[uci][time][legacy-v1]") {
    using Clock = std::chrono::steady_clock;
    const auto accepted = Clock::time_point{std::chrono::seconds{100}};
    constexpr std::array<std::size_t, 4> buckets{1, 8, 32, 64};
    const mouse::uci::TimeBudget budget{.budget = Milliseconds{100}, .hold_until_signal = false};
    const auto forecast = mouse::uci::schedule_search(budget, accepted, accepted + Milliseconds{20},
                                                      Milliseconds{20}, buckets, 64);
    CHECK(forecast.deadline == accepted + Milliseconds{100});
    const auto slow = mouse::uci::schedule_search(budget, accepted, accepted + Milliseconds{61},
                                                  Milliseconds{100}, buckets, 64);
    CHECK(slow.exhausted);
    CHECK(slow.maximum_batch == 1);
}

TEST_CASE("Legacy widened clock state stays bounded outside the reference integer domain",
          "[uci][time][legacy-v1]") {
    mouse::uci::LegacyTimeManager manager;
    GoCommand command;
    command.wtime = 1000;
    const mouse::uci::TimeContext context{
        .side_to_move = Color::White, .overhead_ms = 0, .game_ply = kMaximum};
    CHECK(manager.allocate(command, context)->budget == Milliseconds{1});
    manager.reset();
    command.winc = kMaximum;
    CHECK(manager.allocate(command, context)->budget == Milliseconds{1000});
    command.winc = 10;
    manager.reset();
    const mouse::uci::TimeBudget negative{
        .budget = Milliseconds{1}, .hold_until_signal = false, .clock_budget = Milliseconds::min()};
    manager.complete(negative, Milliseconds::max());
    manager.complete(negative, Milliseconds::max());
    CHECK(manager.allocate(command, {.overhead_ms = 0})->budget == Milliseconds{89});
    manager.reset();
    const mouse::uci::TimeBudget positive{.budget = Milliseconds::max(),
                                          .hold_until_signal = false,
                                          .clock_budget = Milliseconds::max()};
    manager.complete(positive, Milliseconds{0});
    manager.complete(positive, Milliseconds{0});
    CHECK(manager.allocate(command, {.overhead_ms = 0})->budget == Milliseconds{1000});
}

TEST_CASE("Publication includes output and wake costs before the unchanged clock deadline",
          "[uci][time][delivery-deadline-v1]") {
    using Clock = std::chrono::steady_clock;
    using Duration = Clock::duration;
    const auto accepted = Clock::time_point{std::chrono::seconds{10}};
    const auto deadline = accepted + Milliseconds{100};
    for (const auto cost :
         {Duration::zero(), Duration{std::chrono::microseconds{350}},
          Duration{std::chrono::microseconds{6400}}, Duration{Milliseconds{25}}}) {
        const auto wake = mouse::uci::publication_start(accepted, deadline, deadline, cost);
        // A periodic report can begin just before the wake, followed by the
        // final snapshot/report/bestmove. Both must fit, including one clock tick.
        CHECK(wake >= accepted);
        CHECK(wake + cost + cost < deadline);
    }
    CHECK(mouse::uci::publication_start(accepted, deadline, deadline, Duration{-1}) < deadline);
    CHECK(mouse::uci::publication_start(accepted, accepted, accepted, Duration::zero()) ==
          accepted);
    CHECK(mouse::uci::publication_start(accepted, accepted + Milliseconds{1},
                                        accepted + Milliseconds{1}, Milliseconds{6}) == accepted);
    CHECK(mouse::uci::publication_start(accepted, deadline, deadline, Duration::max()) == accepted);
    const auto high = Clock::time_point::max();
    CHECK(mouse::uci::publication_start(high - Milliseconds{100}, high, high, Milliseconds{6}) <
          high);
    const auto low = Clock::time_point::min();
    CHECK(mouse::uci::publication_start(low, low + Milliseconds{1}, low + Milliseconds{1},
                                        Milliseconds{6}) == low);
}

TEST_CASE("Publication preserves legacy allocations with room in the raw GUI clock",
          "[uci][time][delivery-deadline-v1]") {
    using Clock = std::chrono::steady_clock;
    const auto accepted = Clock::time_point{std::chrono::seconds{10}};
    const auto allocated = accepted + Milliseconds{50};
    CHECK(mouse::uci::publication_start(accepted, allocated, accepted + Milliseconds{80},
                                        Milliseconds{6}) == allocated);
    CHECK(mouse::uci::publication_start(accepted, allocated, std::nullopt, Milliseconds{100}) ==
          allocated);
}
