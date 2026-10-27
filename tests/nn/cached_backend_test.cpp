// SPDX-License-Identifier: GPL-3.0-or-later
// FakeBackend is only a test utility; production always owns a real Backend.
#include "engine/nn/cached_backend.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <bit>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include "engine/nn/cache.h"

namespace {
using namespace mouse::nn;
InputPlanes cached_key(std::size_t identifier) {
    InputPlanes input{};
    input[0].mask = static_cast<std::uint64_t>(identifier) + 1;
    input[13].value = static_cast<float>(identifier % 7);
    input[109].value = static_cast<float>(identifier % 100);
    return input;
}
NNOutput expected(const InputPlanes& input) {
    NNOutput output;
    const float tag = static_cast<float>(input[0].mask) + input[13].value + input[109].value;
    output.policy_logits.fill(tag);
    output.wdl = {0.25F, 0.5F, 0.25F};
    output.moves_left = std::abs(tag);
    return output;
}
NNOutput sentinel() {
    NNOutput output;
    output.policy_logits.fill(-1000.0F);
    output.wdl = {0.5F, 0.25F, 0.25F};
    output.moves_left = 999.0F;
    return output;
}
class FakeBackend final : public Backend {
   public:
    enum class Mode : std::uint8_t { Good, Fail, InvalidLast, PartialLast, NoMlh, Throw };
    FakeBackend() {
        metadata_.source_sha256.assign(64, 'a');
        metadata_.input_format = 1;
        info_.onnx_sha256.assign(64, 'b');
        info_.provider = "CPUExecutionProvider";
        info_.precision = "fp32";
        info_.cache_namespace = "actual-onnx:fixture|CPU|fp32|runtime:fixture";
    }
    mouse::Result<void> evaluate(std::span<const InputPlanes> inputs,
                                 std::span<NNOutput> outputs) override {
        const auto active_now = active_.fetch_add(1) + 1;
        auto maximum = max_active_.load();
        while (maximum < active_now && !max_active_.compare_exchange_weak(maximum, active_now)) {
        }
        class ActiveGuard {
           public:
            explicit ActiveGuard(std::atomic<std::size_t>& count) : count_(&count) {}
            ~ActiveGuard() { --*count_; }
            ActiveGuard(const ActiveGuard&) = delete;
            ActiveGuard& operator=(const ActiveGuard&) = delete;
            ActiveGuard(ActiveGuard&&) = delete;
            ActiveGuard& operator=(ActiveGuard&&) = delete;

           private:
            std::atomic<std::size_t>* count_;
        } guard{active_};
        const auto call = calls_.fetch_add(1);
        batch_sizes_.at(call).store(inputs.size());
        std::this_thread::yield();
        for (std::size_t row = 0; row < inputs.size(); ++row) {
            if (mode_ == Mode::PartialLast && row + 1 == inputs.size()) continue;
            outputs[row] = expected(inputs[row]);
            if (mode_ == Mode::NoMlh) outputs[row].moves_left.reset();
        }
        if (mode_ == Mode::Fail) return tl::unexpected(std::string{"deliberate provider failure"});
        if (mode_ == Mode::Throw) throw std::runtime_error("deliberate provider exception");
        if (mode_ == Mode::InvalidLast && !outputs.empty()) {
            outputs.back().policy_logits[1857] = std::numeric_limits<float>::quiet_NaN();
        }
        return {};
    }
    [[nodiscard]] const ModelMetadata& metadata() const noexcept override { return metadata_; }
    [[nodiscard]] const BackendInfo& info() const noexcept override { return info_; }
    void set_mode(Mode mode) noexcept { mode_ = mode; }
    [[nodiscard]] std::size_t calls() const noexcept { return calls_.load(); }
    [[nodiscard]] std::size_t batch_size(std::size_t index) const {
        return batch_sizes_.at(index).load();
    }
    [[nodiscard]] std::size_t max_active() const noexcept { return max_active_.load(); }

   private:
    Mode mode_ = Mode::Good;
    std::atomic<std::size_t> calls_{0};
    std::atomic<std::size_t> active_{0};
    std::atomic<std::size_t> max_active_{0};
    std::array<std::atomic<std::size_t>, 4096> batch_sizes_{};
    ModelMetadata metadata_;
    BackendInfo info_;
};
struct Fixture {
    FakeBackend* raw = nullptr;
    std::unique_ptr<Backend> wrapped;
};
Fixture wrap(std::size_t max_batch = 8, std::size_t bytes = 4ULL * 1024ULL * 1024ULL) {
    auto raw = std::make_unique<FakeBackend>();
    auto* view = raw.get();
    auto wrapped = create_cached_backend(std::move(raw), bytes, max_batch);
    if (!wrapped) FAIL(wrapped.error());
    return {.raw = view, .wrapped = std::move(*wrapped)};
}
}  // namespace

TEST_CASE("Cached backend validates construction and zero budget returns the original backend",
          "[nn][cached-backend]") {
    CHECK_FALSE(create_cached_backend(nullptr, 0, 1).has_value());
    CHECK_FALSE(create_cached_backend(std::make_unique<FakeBackend>(), 65536, 0).has_value());
    CHECK_FALSE(create_cached_backend(std::make_unique<FakeBackend>(), 65536, 1025).has_value());
    CHECK_FALSE(create_cached_backend(std::make_unique<FakeBackend>(), 1, 8).has_value());
    CHECK_FALSE(create_cached_backend(std::make_unique<FakeBackend>(),
                                      std::numeric_limits<std::size_t>::max(), 1)
                    .has_value());
    auto raw = std::make_unique<FakeBackend>();
    auto* original = raw.get();
    auto result = create_cached_backend(std::move(raw), 0, 1024);
    REQUIRE(result.has_value());
    CHECK(result->get() == original);
}

TEST_CASE("Cached backend delegates immutable identity and metadata", "[nn][cached-backend]") {
    const auto fixture = wrap();
    CHECK(&fixture.wrapped->metadata() == &fixture.raw->metadata());
    CHECK(&fixture.wrapped->info() == &fixture.raw->info());
    CHECK(fixture.wrapped->info().onnx_sha256 == std::string(64, 'b'));
    CHECK(fixture.wrapped->info().provider == "CPUExecutionProvider");
}

TEST_CASE("Cached backend batches only unique misses and preserves caller row order",
          "[nn][cached-backend]") {
    const auto fixture = wrap();
    const std::array first{cached_key(1), cached_key(2), cached_key(1), cached_key(3)};
    std::array<NNOutput, 4> outputs{};
    REQUIRE(fixture.wrapped->evaluate(first, outputs).has_value());
    CHECK(fixture.raw->calls() == 1);
    CHECK(fixture.raw->batch_size(0) == 3);
    for (std::size_t row = 0; row < first.size(); ++row)
        CHECK(outputs.at(row) == expected(first.at(row)));
    REQUIRE(fixture.wrapped->evaluate(first, outputs).has_value());
    CHECK(fixture.raw->calls() == 1);
    const std::array next{cached_key(3), cached_key(4), cached_key(2), cached_key(4)};
    REQUIRE(fixture.wrapped->evaluate(next, outputs).has_value());
    CHECK(fixture.raw->calls() == 2);
    CHECK(fixture.raw->batch_size(1) == 1);
    for (std::size_t row = 0; row < next.size(); ++row)
        CHECK(outputs.at(row) == expected(next.at(row)));
}

TEST_CASE("Cached backend exact miss dedup includes all112 masks values and signed zero",
          "[nn][cached-backend]") {
    const auto fixture = wrap(227, 8ULL * 1024ULL * 1024ULL);
    const auto original = cached_key(0);
    std::vector<InputPlanes> inputs{original};
    for (std::size_t plane = 0; plane < original.size(); ++plane) {
        auto changed = original;
        changed.at(plane).mask ^= 1;
        inputs.push_back(changed);
        changed = original;
        changed.at(plane).value += 1;
        inputs.push_back(changed);
    }
    auto changed = original;
    changed[50].value = 0.0F;
    inputs.push_back(changed);
    changed[50].value = -0.0F;
    inputs.push_back(changed);
    std::vector<NNOutput> outputs(inputs.size());
    REQUIRE(fixture.wrapped->evaluate(inputs, outputs).has_value());
    CHECK(fixture.raw->calls() == 1);
    CHECK(fixture.raw->batch_size(0) == 227);
    REQUIRE(fixture.wrapped->evaluate(inputs, outputs).has_value());
    CHECK(fixture.raw->calls() == 1);
}

TEST_CASE("Cached backend rejects malformed batches before touching caller output",
          "[nn][cached-backend]") {
    const auto fixture = wrap(2);
    const std::array inputs{cached_key(1), cached_key(2), cached_key(3)};
    const std::array before{sentinel(), sentinel(), sentinel()};
    auto outputs = before;
    CHECK_FALSE(fixture.wrapped->evaluate(inputs, outputs).has_value());
    CHECK(outputs == before);
    CHECK_FALSE(fixture.wrapped->evaluate(std::span{inputs}.first(2), outputs).has_value());
    CHECK(outputs == before);
    CHECK_FALSE(fixture.wrapped->evaluate({}, {}).has_value());
    CHECK(fixture.raw->calls() == 0);
}

TEST_CASE("Cached backend validates all input rows before evaluating or publishing",
          "[nn][cached-backend]") {
    const auto fixture = wrap();
    std::array inputs{cached_key(1), cached_key(2)};
    inputs[1][109].value = std::numeric_limits<float>::infinity();
    const std::array before{sentinel(), sentinel()};
    auto outputs = before;
    CHECK_FALSE(fixture.wrapped->evaluate(inputs, outputs).has_value());
    CHECK(outputs == before);
    CHECK(fixture.raw->calls() == 0);
    inputs[1] = cached_key(2);
    REQUIRE(fixture.wrapped->evaluate(inputs, outputs).has_value());
    CHECK(fixture.raw->batch_size(0) == 2);
}

TEST_CASE("Cached backend provider failures and exceptions preserve all caller outputs",
          "[nn][cached-backend]") {
    for (const auto mode : {FakeBackend::Mode::Fail, FakeBackend::Mode::Throw}) {
        const auto fixture = wrap();
        fixture.raw->set_mode(mode);
        const std::array inputs{cached_key(1), cached_key(2)};
        const std::array before{sentinel(), sentinel()};
        auto outputs = before;
        CHECK_FALSE(fixture.wrapped->evaluate(inputs, outputs).has_value());
        CHECK(outputs == before);
        fixture.raw->set_mode(FakeBackend::Mode::Good);
        REQUIRE(fixture.wrapped->evaluate(inputs, outputs).has_value());
        CHECK(fixture.raw->calls() == 2);
        CHECK(fixture.raw->batch_size(1) == 2);
    }
}

TEST_CASE("Cached backend validates every miss before inserting any result",
          "[nn][cached-backend]") {
    const auto fixture = wrap();
    fixture.raw->set_mode(FakeBackend::Mode::InvalidLast);
    const std::array inputs{cached_key(1), cached_key(2), cached_key(3)};
    const std::array before{sentinel(), sentinel(), sentinel()};
    auto outputs = before;
    CHECK_FALSE(fixture.wrapped->evaluate(inputs, outputs).has_value());
    CHECK(outputs == before);
    fixture.raw->set_mode(FakeBackend::Mode::Good);
    REQUIRE(fixture.wrapped->evaluate(inputs, outputs).has_value());
    CHECK(fixture.raw->calls() == 2);
    CHECK(fixture.raw->batch_size(1) == 3);
}

TEST_CASE("Cached backend failed mixed hit miss batch cannot publish staged hits",
          "[nn][cached-backend]") {
    const auto fixture = wrap();
    const std::array<InputPlanes, 1> cached{cached_key(1)};
    std::array<NNOutput, 1> seed{};
    REQUIRE(fixture.wrapped->evaluate(cached, seed).has_value());
    fixture.raw->set_mode(FakeBackend::Mode::Fail);
    const std::array mixed{cached_key(1), cached_key(2)};
    const std::array before{sentinel(), sentinel()};
    auto outputs = before;
    CHECK_FALSE(fixture.wrapped->evaluate(mixed, outputs).has_value());
    CHECK(outputs == before);
    CHECK(fixture.raw->batch_size(1) == 1);
    REQUIRE(fixture.wrapped->evaluate(cached, seed).has_value());
    CHECK(seed[0] == expected(cached[0]));
    CHECK(fixture.raw->calls() == 2);
}

TEST_CASE("Cached backend serializes concurrent callers and does not share output buffers",
          "[nn][cached-backend]") {
    const auto fixture = wrap();
    std::atomic<bool> failed{false};
    constexpr std::size_t workers = 8;
    std::barrier start{static_cast<std::ptrdiff_t>(workers)};
    std::vector<std::jthread> threads;
    threads.reserve(workers);
    for (std::size_t worker = 0; worker < workers; ++worker) {
        threads.emplace_back([&, worker] {
            start.arrive_and_wait();
            for (std::size_t iteration = 0; iteration < 100; ++iteration) {
                const std::array<InputPlanes, 1> inputs{cached_key(worker * 16 + iteration % 16)};
                std::array<NNOutput, 1> outputs{};
                if (!fixture.wrapped->evaluate(inputs, outputs) ||
                    outputs[0] != expected(inputs[0])) {
                    failed.store(true);
                    return;
                }
            }
        });
    }
    threads.clear();
    CHECK_FALSE(failed.load());
    CHECK(fixture.raw->max_active() == 1);
    CHECK(fixture.raw->calls() == 128);
}

TEST_CASE(
    "Cached backend cannot reuse stale miss scratch after a successful partial provider write",
    "[nn][cached-backend]") {
    const auto fixture = wrap();
    const std::array initial{cached_key(1), cached_key(2)};
    std::array<NNOutput, 2> outputs{};
    REQUIRE(fixture.wrapped->evaluate(initial, outputs).has_value());
    fixture.raw->set_mode(FakeBackend::Mode::PartialLast);
    const std::array next{cached_key(3), cached_key(4)};
    const std::array before{sentinel(), sentinel()};
    outputs = before;
    CHECK_FALSE(fixture.wrapped->evaluate(next, outputs).has_value());
    CHECK(outputs == before);
    fixture.raw->set_mode(FakeBackend::Mode::Good);
    REQUIRE(fixture.wrapped->evaluate(next, outputs).has_value());
    CHECK(fixture.raw->calls() == 3);
    CHECK(fixture.raw->batch_size(2) == 2);
}

TEST_CASE("Cached backend maximum batch preserves staged hits during cache eviction",
          "[nn][cached-backend]") {
    constexpr std::size_t rows = 1024;
    constexpr std::size_t budget = 65536;
    const auto fixture = wrap(rows, budget);
    auto capacity_probe = NNCache::create(budget, fixture.raw->info().cache_namespace);
    REQUIRE(capacity_probe.has_value());
    const auto capacity = (*capacity_probe)->stats().capacity;
    REQUIRE(capacity > 0);
    REQUIRE(capacity < rows);
    std::vector<InputPlanes> inputs;
    inputs.reserve(rows);
    for (std::size_t row = 0; row < rows; ++row) inputs.push_back(cached_key(row));
    std::vector<NNOutput> outputs(rows);
    REQUIRE(fixture.wrapped->evaluate(inputs, outputs).has_value());
    CHECK(fixture.raw->batch_size(0) == rows);
    REQUIRE(fixture.wrapped->evaluate(inputs, outputs).has_value());
    CHECK(fixture.raw->calls() == 2);
    CHECK(fixture.raw->batch_size(1) == rows - capacity);
    for (std::size_t row = 0; row < rows; ++row) CHECK(outputs.at(row) == expected(inputs.at(row)));
}

TEST_CASE("Cached backend retains an explicitly absent optional moves-left head",
          "[nn][cached-backend]") {
    const auto fixture = wrap();
    fixture.raw->set_mode(FakeBackend::Mode::NoMlh);
    const std::array<InputPlanes, 1> inputs{cached_key(1)};
    std::array<NNOutput, 1> outputs{};
    REQUIRE(fixture.wrapped->evaluate(inputs, outputs).has_value());
    CHECK_FALSE(outputs[0].moves_left.has_value());
    fixture.raw->set_mode(FakeBackend::Mode::Fail);
    outputs[0] = sentinel();
    REQUIRE(fixture.wrapped->evaluate(inputs, outputs).has_value());
    CHECK_FALSE(outputs[0].moves_left.has_value());
    CHECK(fixture.raw->calls() == 1);
}
