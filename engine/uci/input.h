// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stop_token>
#include <string>

#include "engine/util/result.h"

namespace mouse::uci {
#ifdef _WIN32
using NativeInput = void*;  // Native Windows HANDLE, without exposing windows.h.
#else
using NativeInput = int;
#endif

enum class ReadKind : std::uint8_t { Line, Eof, Cancelled, RejectedLine };
struct ReadEvent {
    ReadKind kind;
    std::string line;  // Line payload or RejectedLine diagnostic; empty for EOF/cancellation.
};

// One owner calls read or destroys this object, never concurrently. The borrowed
// handle/fd stays open through destruction and has no other reader. Another
// thread may request stop. Native handles, console modes and fd flags are unchanged.
// Windows pipes lazily own one bounded synchronous-reader thread: each request
// is acknowledged or canceled and drained before read returns; there is no idle
// prefetch. Destruction cancels/drains and joins before releasing its own state.
// Pipe data/completion wakes an event-driven wait. Cancellation retries cover
// CancelSynchronousIo missing a not-yet-issued ReadFile; Windows driver completion
// and OS scheduling have no universal hard latency bound. Console/POSIX stop
// polling remains at most 20 ms, excluding native device/scheduling latency.
// Cancellation precedes buffered input and retains partial/discard state plus
// completed data/EOF/errors for a later call with a fresh token.
// Lines contain at most commands.h kMaximumLineBytes raw bytes before LF; a CR
// immediately before LF is removed and counts toward that limit. Embedded NUL
// is preserved for the parser. EOF returns a final unterminated line once (a
// trailing CR is preserved), then Eof. An oversized line is discarded through
// LF/EOF before one RejectedLine diagnostic, preserving the next command.
// Result errors indicate native I/O/Unicode conversion failure and are fatal;
// consumers must not retry them as malformed protocol commands. Console input
// uses queued Unicode key events, supports backspace and Ctrl-Z EOF, and does
// not echo/write output. POSIX terminal editing remains the terminal's job.
class InputReader {
   public:
    InputReader();
    explicit InputReader(NativeInput input);
    InputReader(const InputReader&) = delete;
    InputReader& operator=(const InputReader&) = delete;
    InputReader(InputReader&&) = delete;
    InputReader& operator=(InputReader&&) = delete;
    ~InputReader();

    [[nodiscard]] Result<ReadEvent> read(const std::stop_token& stop);

   private:
    enum class NativeKind : std::uint8_t { Data, Eof, Cancelled, Erase };
    [[nodiscard]] Result<NativeKind> fill(const std::stop_token& stop);
#ifdef _WIN32
    struct PipeWorker;
    std::unique_ptr<PipeWorker> pipe_worker_;
    [[nodiscard]] Result<NativeKind> fill_console(const std::stop_token& stop);
    std::uint16_t console_character_ = 0;
    std::uint16_t console_repeats_ = 0;
    std::uint16_t high_surrogate_ = 0;
#endif
    NativeInput input_;
    std::array<char, 4096> bytes_{};
    std::size_t cursor_ = 0;
    std::size_t size_ = 0;
    std::string partial_;
    bool discarding_ = false;
    bool eof_ = false;
};
}  // namespace mouse::uci
