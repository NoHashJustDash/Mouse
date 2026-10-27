// SPDX-License-Identifier: GPL-3.0-or-later
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <array>
#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>

// Test a separately named copy of the actual implementation, retaining native
// ReadFile/CancelSynchronousIo calls. The gates control OS-call interleavings,
// rather than a product hook or copied reader algorithm. The ordinary input
// suite separately exercises the untouched production class with real pipes.
namespace input_request_probe {
using namespace std::chrono_literals;
struct Gate {
    enum class Phase : std::uint8_t { BeforeIssue, AfterResult };
    explicit Gate(Phase selected) : phase(selected) {}
    Gate(const Gate&) = delete;
    Gate& operator=(const Gate&) = delete;
    Gate(Gate&&) = delete;
    Gate& operator=(Gate&&) = delete;
    ~Gate() = default;
    std::mutex mutex;
    std::condition_variable changed;
    Phase phase;
    std::atomic<bool> armed{true};
    bool entered = false, released = false;
    unsigned cancel_attempts = 0, cancel_misses = 0, cancel_successes = 0;
    DWORD native_error = ERROR_SUCCESS, native_bytes = 0;
    void pause() {
        std::unique_lock lock(mutex);
        entered = true;
        changed.notify_all();
        changed.wait(lock, [&] { return released; });
    }
    bool wait_entered() {
        std::unique_lock lock(mutex);
        return changed.wait_for(lock, 3s, [&] { return entered; });
    }
    bool wait_cancels(unsigned count) {
        std::unique_lock lock(mutex);
        return changed.wait_for(lock, 3s, [&] { return cancel_attempts >= count; });
    }
    void release() {
        const std::lock_guard lock(mutex);
        released = true;
        changed.notify_all();
    }
};
// A scoped registration outlives the reader and its joined native worker.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::atomic<Gate*> active_gate{nullptr};
BOOL WINAPI gated_read(HANDLE handle, LPVOID buffer, DWORD size, LPDWORD received,
                       LPOVERLAPPED overlapped) {
    auto* gate = active_gate.load(std::memory_order_acquire);
    const bool intercept = gate && gate->armed.exchange(false, std::memory_order_acq_rel);
    if (intercept && gate->phase == Gate::Phase::BeforeIssue) gate->pause();
    const BOOL result = ReadFile(handle, buffer, size, received, overlapped);
    const DWORD error = result ? ERROR_SUCCESS : GetLastError();
    if (intercept && gate->phase == Gate::Phase::AfterResult) {
        {
            const std::lock_guard lock(gate->mutex);
            gate->native_error = error;
            gate->native_bytes = *received;
        }
        gate->pause();
    }
    SetLastError(error);
    return result;
}
BOOL WINAPI counted_cancel(HANDLE thread) {
    const BOOL result = CancelSynchronousIo(thread);
    const DWORD error = result ? ERROR_SUCCESS : GetLastError();
    if (auto* gate = active_gate.load(std::memory_order_acquire)) {
        const std::lock_guard lock(gate->mutex);
        ++gate->cancel_attempts;
        if (result)
            ++gate->cancel_successes;
        else if (error == ERROR_NOT_FOUND)
            ++gate->cancel_misses;
        gate->changed.notify_all();
    }
    SetLastError(error);
    return result;
}
struct Registration {
    explicit Registration(Gate& gate) {
        REQUIRE(active_gate.exchange(&gate, std::memory_order_acq_rel) == nullptr);
    }
    ~Registration() { active_gate.store(nullptr, std::memory_order_release); }
    Registration(const Registration&) = delete;
    Registration& operator=(const Registration&) = delete;
    Registration(Registration&&) = delete;
    Registration& operator=(Registration&&) = delete;
};
}  // namespace input_request_probe

// These macros are scoped only across the actual input.cpp include. windows.h
// is already included, so native SDK declarations are never macro-renamed.
// NOLINTBEGIN(cppcoreguidelines-macro-usage)
#define InputReader InstrumentedInputReader
#define ReadFile ::input_request_probe::gated_read
#define CancelSynchronousIo ::input_request_probe::counted_cancel
#include "engine/uci/input.cpp"  // NOLINT(bugprone-suspicious-include)
#undef CancelSynchronousIo
#undef ReadFile
#undef InputReader
// NOLINTEND(cppcoreguidelines-macro-usage)

namespace {
using namespace std::chrono_literals;
using input_request_probe::Gate;
using mouse::uci::InstrumentedInputReader;
using mouse::uci::ReadEvent;
using mouse::uci::ReadKind;
class RequestPipe {
   public:
    RequestPipe() { REQUIRE(CreatePipe(&input, &output, nullptr, 0)); }
    ~RequestPipe() {
        if (input) CloseHandle(input);
        finish();
    }
    RequestPipe(const RequestPipe&) = delete;
    RequestPipe& operator=(const RequestPipe&) = delete;
    RequestPipe(RequestPipe&&) = delete;
    RequestPipe& operator=(RequestPipe&&) = delete;
    void finish() {
        if (output) CloseHandle(std::exchange(output, nullptr));
    }
    void write(std::string_view text) const {
        DWORD received = 0;
        REQUIRE(
            WriteFile(output, text.data(), static_cast<DWORD>(text.size()), &received, nullptr));
        REQUIRE(received == text.size());
    }
    HANDLE input = nullptr, output = nullptr;
};
class ReadCall {
   public:
    ReadCall(InstrumentedInputReader& reader, Gate& gate, RequestPipe& pipe, bool rescue = true)
        : gate_(gate), pipe_(pipe), rescue_(rescue), thread_([&reader, this] {
              result_ = reader.read(stop_.get_token());
              {
                  const std::lock_guard lock(mutex_);
                  returned_ = true;
              }
              changed_.notify_all();
          }) {}
    ~ReadCall() {
        stop_.request_stop();
        gate_.release();
        // Only owned writer handles are closed as a last-resort test rescue.
        // The borrowed input remains valid until the reader destructor joins.
        if (rescue_) pipe_.finish();
        if (thread_.joinable()) thread_.join();
    }
    ReadCall(const ReadCall&) = delete;
    ReadCall& operator=(const ReadCall&) = delete;
    ReadCall(ReadCall&&) = delete;
    ReadCall& operator=(ReadCall&&) = delete;
    bool wait() {
        std::unique_lock lock(mutex_);
        return changed_.wait_for(lock, 3s, [&] { return returned_; });
    }
    bool returned() {
        const std::lock_guard lock(mutex_);
        return returned_;
    }
    void join() { thread_.join(); }
    void request_stop() { static_cast<void>(stop_.request_stop()); }
    [[nodiscard]] const mouse::Result<ReadEvent>& outcome() const {
        if (!result_) throw std::logic_error("Read call did not publish a result");
        return *result_;
    }

   private:
    std::stop_source stop_;
    std::optional<mouse::Result<ReadEvent>> result_;
    Gate& gate_;
    RequestPipe& pipe_;
    bool rescue_;
    std::mutex mutex_;
    std::condition_variable changed_;
    bool returned_ = false;
    std::jthread thread_;
};
}  // namespace

TEST_CASE("Pipe cancellation retries an unissued synchronous read until its request drains",
          "[uci][input][input-request]") {
    Gate gate{Gate::Phase::BeforeIssue};
    input_request_probe::Registration registered{gate};
    RequestPipe pipe;
    InstrumentedInputReader reader{pipe.input};
    ReadCall pending{reader, gate, pipe};
    // Functional mechanism assertion, not a microsecond scheduling threshold:
    // an idle anonymous pipe must have a cancellable owned blocking request.
    REQUIRE(gate.wait_entered());
    pending.request_stop();
    REQUIRE(gate.wait_cancels(2));
    CHECK_FALSE(pending.returned());
    gate.release();
    REQUIRE(pending.wait());
    pending.join();
    const auto& pending_result = pending.outcome();
    REQUIRE(pending_result);
    CHECK(pending_result->kind == ReadKind::Cancelled);
    CHECK(gate.cancel_misses >= 2);
    CHECK(gate.cancel_successes >= 1);
    pipe.write("next-generation\n");
    ReadCall fresh{reader, gate, pipe};
    REQUIRE(fresh.wait());
    fresh.join();
    const auto& fresh_result = fresh.outcome();
    REQUIRE(fresh_result);
    CHECK(fresh_result->kind == ReadKind::Line);
    CHECK(fresh_result->line == "next-generation");
}

TEST_CASE("Cancellation racing a native completion retains its data EOF or error for a fresh read",
          "[uci][input][input-request]") {
    Gate gate{Gate::Phase::AfterResult};
    input_request_probe::Registration registered{gate};
    RequestPipe pipe;
    bool error = false, eof = false;
    SECTION("Data") { pipe.write("uci\n"); }
    SECTION("EOF") {
        eof = true;
        pipe.finish();
    }
    SECTION("Native error") { error = true; }
    InstrumentedInputReader reader{error ? pipe.output : pipe.input};
    ReadCall pending{reader, gate, pipe, !error};
    REQUIRE(gate.wait_entered());
    pending.request_stop();
    gate.release();
    REQUIRE(pending.wait());
    pending.join();
    const auto& pending_result = pending.outcome();
    REQUIRE(pending_result);
    CHECK(pending_result->kind == ReadKind::Cancelled);
    ReadCall fresh{reader, gate, pipe, !error};
    REQUIRE(fresh.wait());
    fresh.join();
    const auto& fresh_result = fresh.outcome();
    if (error) {
        REQUIRE_FALSE(fresh_result);
        CHECK(gate.native_error != ERROR_SUCCESS);
        CHECK(fresh_result.error().find("(" + std::to_string(gate.native_error) + ")") !=
              std::string::npos);
    } else {
        REQUIRE(fresh_result);
        CHECK(fresh_result->kind == (eof ? ReadKind::Eof : ReadKind::Line));
        if (!eof) CHECK(fresh_result->line == "uci");
    }
}
#endif
