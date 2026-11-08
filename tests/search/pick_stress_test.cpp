// SPDX-License-Identifier: GPL-3.0-or-later
// Hidden from ordinary runs: it oversubscribes the CPU for as long as MOUSE_PICK_STRESS_SECONDS.
#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>

#include "engine/chess/attacks.h"
#include "engine/nn/history.h"
#include "engine/search/search.h"

namespace {
struct StressBackend final : mouse::nn::Backend {
    mouse::nn::ModelMetadata model;
    mouse::nn::BackendInfo identity;
    StressBackend() {
        model.input_format = 1;
        identity.onnx_sha256 = "pick-stress";
    }
    [[nodiscard]] mouse::Result<void> evaluate(std::span<const mouse::nn::InputPlanes> inputs,
                                               std::span<mouse::nn::NNOutput> outputs) override {
        for (std::size_t row = 0; row < inputs.size(); ++row) {
            std::uint64_t mix = 0xcbf29ce484222325ULL;
            for (const auto& plane : inputs[row]) mix = (mix ^ plane.mask) * 0x100000001b3ULL;
            mouse::nn::NNOutput value;
            auto seed = mix;
            for (auto& logit : value.policy_logits) {
                seed = seed * 0x9E3779B97F4A7C15ULL + 1;
                logit = static_cast<float>(seed >> 54U) / 256.0F;
            }
            value.wdl = {0.35F, 0.35F, 0.3F};
            value.moves_left = 30.0F;
            outputs[row] = value;
        }
        return {};
    }
    [[nodiscard]] const mouse::nn::ModelMetadata& metadata() const noexcept override {
        return model;
    }
    [[nodiscard]] const mouse::nn::BackendInfo& info() const noexcept override { return identity; }
};
}  // namespace

TEST_CASE("Parallel picking never hangs under preemption", "[.pick-stress]") {
    using Clock = std::chrono::steady_clock;
    const auto* text = std::getenv("MOUSE_PICK_STRESS_SECONDS");
    const auto seconds = std::chrono::seconds{text ? std::atoi(text) : 60};
    std::atomic<std::uint64_t> searches{0};
    std::atomic<bool> finished{false};
    // A search that stops making progress holds no lock anyone else needs, so only a watchdog
    // notices. It exits outright, as the hung thread cannot be joined.
    std::jthread watchdog([&] {
        auto seen = searches.load();
        auto since = Clock::now();
        while (!finished.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds{200});
            const auto now = searches.load();
            if (now != seen) {
                seen = now;
                since = Clock::now();
            } else if (Clock::now() - since > std::chrono::seconds{20}) {
                std::printf("PICK_STRESS hang after %llu searches\n",
                            static_cast<unsigned long long>(now));
                std::fflush(stdout);
                std::_Exit(3);
            }
        }
    });
    // A failed assertion leaves by exception, and the watchdog must still stop.
    const struct FinishOnExit {
        std::atomic<bool>& flag;
        ~FinishOnExit() { flag.store(true); }
    } finish_on_exit{finished};
    const mouse::chess::AttackTables attacks;
    auto root = *mouse::chess::Position::from_fen(mouse::chess::kStartFen, attacks);
    mouse::nn::EncodingHistory history;
    history.reset(root);
    StressBackend backend;
    mouse::search::SearchConfig config;
    // Far more threads than cores: the race needs a helper preempted between reading and claiming.
    config.threads = 64;
    config.maximum_batch = 64;
    config.ram_limit_bytes = 256ULL << 20U;
    config.gather.minimum_picking_work = 1;
    config.gather.minimum_remaining_picking_work = 0;
    auto created = mouse::search::SearchEngine::create(config, attacks, backend, nullptr);
    if (!created) FAIL(created.error());
    auto& engine = **created;
    const auto end = Clock::now() + seconds;
    while (Clock::now() < end) {
        REQUIRE(engine.set_position(root, history, false));
        mouse::search::SearchLimits limits;
        limits.nodes = 3000;
        limits.smart_pruning = false;
        const auto result = engine.run(limits);
        if (!result) FAIL(result.error());
        ++searches;
    }
    std::printf("PICK_STRESS ok after %llu searches\n",
                static_cast<unsigned long long>(searches.load()));
    CHECK(searches.load() > 0);
}
