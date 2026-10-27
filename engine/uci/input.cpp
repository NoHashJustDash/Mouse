// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/uci/input.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <exception>
#include <mutex>
#include <system_error>
#include <thread>
#include <utility>

#include "engine/uci/commands.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <poll.h>
#include <unistd.h>

#include <cerrno>
#endif

namespace mouse::uci {
namespace {
#ifdef _WIN32
constexpr auto kPollInterval = std::chrono::milliseconds{20};
constexpr auto kCancelRetry = std::chrono::milliseconds{1};
#else
constexpr int kPollMilliseconds = 20;
#endif
std::string input_error(int code) {
    return "UCI input error: " + std::error_code(code, std::system_category()).message() + " (" +
           std::to_string(code) + ")";
}
}  // namespace

#ifdef _WIN32
// Anonymous Windows pipes do not support overlapped I/O. Keep one synchronous
// request on an owned thread and acknowledge completion before queuing another.
// https://learn.microsoft.com/windows/win32/ipc/anonymous-pipe-operations
// https://learn.microsoft.com/windows/win32/fileio/cancelsynchronousio-func
struct InputReader::PipeWorker {
    explicit PipeWorker(NativeInput handle) : input(handle), thread([this] { run(); }) {}
    PipeWorker(const PipeWorker&) = delete;
    PipeWorker& operator=(const PipeWorker&) = delete;
    PipeWorker(PipeWorker&&) = delete;
    PipeWorker& operator=(PipeWorker&&) = delete;
    ~PipeWorker() noexcept {
        try {
            {
                std::unique_lock lock(mutex);
                drain(lock);
                quit = true;
                work.notify_one();
            }
            thread.join();
        } catch (...) {
            // A broken synchronization primitive cannot be recovered by
            // abandoning a thread that still owns this object's storage.
            std::fputs("Mouse fatal: cannot drain/join the UCI input pipe worker.\n", stderr);
            std::terminate();
        }
    }
    [[nodiscard]] Result<NativeKind> read(const std::stop_token& stop,
                                          std::array<char, 4096>& destination, std::size_t& size) {
        std::unique_lock lock(mutex);
        if (stop.stop_requested()) return NativeKind::Cancelled;
        if (!done) {
            if (failed) return tl::unexpected(input_error(ERROR_UNHANDLED_EXCEPTION));
            cancel_requested = false;
            pending = true;
            work.notify_one();
        }
        static_cast<void>(completed.wait(lock, stop, [&] { return done; }));
        if (stop.stop_requested()) {
            drain(lock);
            // Cancellation may race normal data, EOF or error completion. Only
            // an empty operation that actually aborted can be dropped. Other
            // completions stay unacknowledged for the next fresh-token call.
            if (!success && error == ERROR_OPERATION_ABORTED && received == 0) done = false;
            return NativeKind::Cancelled;
        }
        done = false;
        if (!success &&
            !((error == ERROR_MORE_DATA || error == ERROR_OPERATION_ABORTED) && received != 0)) {
            if (error == ERROR_BROKEN_PIPE || error == ERROR_NO_DATA) return NativeKind::Eof;
            return tl::unexpected(input_error(static_cast<int>(error)));
        }
        std::copy_n(buffer.begin(), received, destination.begin());
        size = received;
        return size == 0 ? NativeKind::Eof : NativeKind::Data;
    }

   private:
    void drain(std::unique_lock<std::mutex>& lock) {
        cancel_requested = true;
        while (pending) {
            // CancelSynchronousIo does not wait for completion and can miss the
            // gap before ReadFile issues. Retrying is restricted to this exact
            // pending request; no callback can cancel a later generation.
            lock.unlock();
            static_cast<void>(CancelSynchronousIo(thread.native_handle()));
            lock.lock();
            completed.wait_for(lock, kCancelRetry, [&] { return !pending; });
        }
    }
    void run() noexcept {
        try {
            std::unique_lock lock(mutex);
            for (;;) {
                work.wait(lock, [&] { return pending || quit; });
                if (quit) return;
                DWORD count = 0;
                BOOL ok = FALSE;
                DWORD code = ERROR_OPERATION_ABORTED;
                if (!cancel_requested) {
                    lock.unlock();
                    ok = ReadFile(input, buffer.data(), static_cast<DWORD>(buffer.size()), &count,
                                  nullptr);
                    code = ok ? ERROR_SUCCESS : GetLastError();
                    lock.lock();
                }
                success = ok != FALSE;
                received = count;
                error = code;
                pending = false;
                done = true;
                completed.notify_all();
            }
        } catch (...) {
            // Never let a C++ failure escape the owned thread. Fatal consumers
            // get one bounded native diagnostic, including after a stop race.
            const std::lock_guard lock(mutex);
            failed = true;
            success = false;
            received = 0;
            error = ERROR_UNHANDLED_EXCEPTION;
            pending = false;
            done = true;
            completed.notify_all();
        }
    }
    const NativeInput input;
    std::array<char, 4096> buffer{};
    std::mutex mutex;
    std::condition_variable work;
    std::condition_variable_any completed;
    bool pending = false, done = false, cancel_requested = false, quit = false;
    bool failed = false, success = false;
    DWORD received = 0, error = ERROR_SUCCESS;
    // Last member: every state field is initialized before this thread starts.
    std::thread thread;
};
InputReader::InputReader() : InputReader(GetStdHandle(STD_INPUT_HANDLE)) {}
#else
InputReader::InputReader() : InputReader(STDIN_FILENO) {}
#endif
InputReader::InputReader(NativeInput input) : input_(input) {}
InputReader::~InputReader() = default;

Result<ReadEvent> InputReader::read(const std::stop_token& stop) {
    while (true) {
        if (stop.stop_requested()) {
            return ReadEvent{.kind = ReadKind::Cancelled, .line = {}};
        }
        if (cursor_ == size_) {
            if (eof_) {
                return ReadEvent{.kind = ReadKind::Eof, .line = {}};
            }
            const auto native = fill(stop);
            if (!native) {
                return tl::unexpected(native.error());
            }
            if (*native == NativeKind::Cancelled) {
                return ReadEvent{.kind = ReadKind::Cancelled, .line = {}};
            }
            if (*native == NativeKind::Erase) {
                if (!discarding_ && !partial_.empty()) {
                    // Remove one whole UTF-8 character, not an isolated suffix.
                    while (!partial_.empty()) {
                        const auto byte = static_cast<unsigned char>(partial_.back());
                        partial_.pop_back();
                        if ((byte & 0xc0U) != 0x80U) {
                            break;
                        }
                    }
                }
                continue;
            }
            if (*native == NativeKind::Eof) {
                eof_ = true;
                if (discarding_) {
                    discarding_ = false;
                    return ReadEvent{.kind = ReadKind::RejectedLine,
                                     .line = "UCI input line exceeds maximum byte length " +
                                             std::to_string(kMaximumLineBytes)};
                }
                if (!partial_.empty()) {
                    return ReadEvent{.kind = ReadKind::Line, .line = std::exchange(partial_, {})};
                }
                return ReadEvent{.kind = ReadKind::Eof, .line = {}};
            }
            continue;
        }
        const char byte = bytes_.at(cursor_++);
        if (byte == '\n') {
            if (discarding_) {
                discarding_ = false;
                return ReadEvent{.kind = ReadKind::RejectedLine,
                                 .line = "UCI input line exceeds maximum byte length " +
                                         std::to_string(kMaximumLineBytes)};
            }
            if (!partial_.empty() && partial_.back() == '\r') {
                partial_.pop_back();
            }
            return ReadEvent{.kind = ReadKind::Line, .line = std::exchange(partial_, {})};
        }
        if (!discarding_) {
            if (partial_.size() == kMaximumLineBytes) {
                partial_.clear();
                discarding_ = true;
            } else {
                // Every append is preceded by the bound; there is no unbounded
                // getline allocation even when a peer never sends a newline.
                partial_.push_back(byte);
            }
        }
    }
}

#ifdef _WIN32
Result<InputReader::NativeKind> InputReader::fill_console(const std::stop_token& stop) {
    // ReadConsole/ReadFile in cooked mode can wait for Enter after a key event.
    // Consume only queued records instead: Microsoft's ReadConsoleInput contract
    // https://learn.microsoft.com/windows/console/readconsoleinput
    while (!stop.stop_requested()) {
        if (console_repeats_ == 0) {
            DWORD available = 0;
            if (!GetNumberOfConsoleInputEvents(input_, &available)) {
                return tl::unexpected(input_error(static_cast<int>(GetLastError())));
            }
            if (available == 0) {
                std::this_thread::sleep_for(kPollInterval);
                continue;
            }
            INPUT_RECORD record{};
            DWORD received = 0;
            if (!ReadConsoleInputW(input_, &record, 1, &received)) {
                return tl::unexpected(input_error(static_cast<int>(GetLastError())));
            }
            if (received != 1 || record.EventType != KEY_EVENT) {
                continue;
            }
            // The Windows ABI selects this tagged union member with KEY_EVENT.
            // UnicodeChar is the documented wide variant of the SDK union.
            // NOLINTBEGIN(cppcoreguidelines-pro-type-union-access)
            const auto key = record.Event.KeyEvent;
            console_character_ = static_cast<std::uint16_t>(key.uChar.UnicodeChar);
            // NOLINTEND(cppcoreguidelines-pro-type-union-access)
            if (!key.bKeyDown) {
                continue;
            }
            console_repeats_ = key.wRepeatCount;
            if (console_repeats_ == 0 || console_character_ == 0) {
                console_repeats_ = 0;
                continue;
            }
        }
        --console_repeats_;
        const auto character = console_character_;
        if (character == 0x08) {
            if (high_surrogate_ != 0) {
                high_surrogate_ = 0;
                continue;
            }
            return NativeKind::Erase;
        }
        if (character == 0x1a) {
            if (high_surrogate_ != 0) {
                high_surrogate_ = 0;
                return tl::unexpected("UCI input console has an incomplete UTF-16 character");
            }
            return NativeKind::Eof;
        }
        std::uint32_t codepoint = character;
        if (character >= 0xd800 && character <= 0xdbff) {
            if (high_surrogate_ != 0) {
                high_surrogate_ = 0;
                return tl::unexpected("UCI input console has an invalid UTF-16 surrogate pair");
            }
            high_surrogate_ = character;
            continue;
        }
        if (character >= 0xdc00 && character <= 0xdfff) {
            if (high_surrogate_ == 0) {
                return tl::unexpected("UCI input console has an unpaired UTF-16 low surrogate");
            }
            codepoint = 0x10000U +
                        ((static_cast<std::uint32_t>(high_surrogate_) - 0xd800U) << 10U) +
                        (static_cast<std::uint32_t>(character) - 0xdc00U);
            high_surrogate_ = 0;
        } else if (high_surrogate_ != 0) {
            high_surrogate_ = 0;
            return tl::unexpected("UCI input console has an incomplete UTF-16 character");
        }
        if (codepoint <= 0x7fU) {
            bytes_[0] = codepoint == '\r' ? '\n' : static_cast<char>(codepoint);
            size_ = 1;
        } else if (codepoint <= 0x7ffU) {
            bytes_[0] = static_cast<char>(0xc0U | (codepoint >> 6U));
            bytes_[1] = static_cast<char>(0x80U | (codepoint & 0x3fU));
            size_ = 2;
        } else if (codepoint <= 0xffffU) {
            bytes_[0] = static_cast<char>(0xe0U | (codepoint >> 12U));
            bytes_[1] = static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU));
            bytes_[2] = static_cast<char>(0x80U | (codepoint & 0x3fU));
            size_ = 3;
        } else {
            bytes_[0] = static_cast<char>(0xf0U | (codepoint >> 18U));
            bytes_[1] = static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3fU));
            bytes_[2] = static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU));
            bytes_[3] = static_cast<char>(0x80U | (codepoint & 0x3fU));
            size_ = 4;
        }
        cursor_ = 0;
        return NativeKind::Data;
    }
    return NativeKind::Cancelled;
}
#endif

Result<InputReader::NativeKind> InputReader::fill(const std::stop_token& stop) {
    cursor_ = 0;
    size_ = 0;
    while (!stop.stop_requested()) {
#ifdef _WIN32
        const DWORD type = GetFileType(input_);
        if (type == FILE_TYPE_CHAR) {
            DWORD mode = 0;
            if (!GetConsoleMode(input_, &mode)) {
                return tl::unexpected("UCI input is an unsupported non-console character device");
            }
            return fill_console(stop);
        }
        if (type == FILE_TYPE_PIPE) {
            try {
                if (!pipe_worker_) pipe_worker_ = std::make_unique<PipeWorker>(input_);
                return pipe_worker_->read(stop, bytes_, size_);
            } catch (const std::exception& error) {
                // This also drains any request left by an exceptional wait.
                pipe_worker_.reset();
                return tl::unexpected(std::string{"Cannot read UCI pipe: "} + error.what());
            }
        }
        if (type != FILE_TYPE_DISK)
            return tl::unexpected("UCI input has an invalid or unsupported native handle");
        const auto request = static_cast<DWORD>(bytes_.size());
        if (stop.stop_requested()) {
            return NativeKind::Cancelled;
        }
        DWORD received = 0;
        if (!ReadFile(input_, bytes_.data(), request, &received, nullptr)) {
            const DWORD error = GetLastError();
            if (error == ERROR_BROKEN_PIPE || error == ERROR_NO_DATA) {
                return NativeKind::Eof;
            }
            if (error != ERROR_MORE_DATA || received == 0) {
                return tl::unexpected(input_error(static_cast<int>(error)));
            }
        }
        size_ = received;
#else
        if (input_ < 0) {
            return tl::unexpected("UCI input has an invalid native descriptor");
        }
        pollfd descriptor{.fd = input_, .events = POLLIN, .revents = 0};
        const int ready = poll(&descriptor, 1, kPollMilliseconds);
        if (ready < 0) {
            if (errno == EINTR) {
                continue;
            }
            return tl::unexpected(input_error(errno));
        }
        if (ready == 0) {
            continue;
        }
        if (stop.stop_requested()) {
            return NativeKind::Cancelled;
        }
        if ((descriptor.revents & POLLNVAL) != 0) {
            return tl::unexpected("UCI input has an invalid native descriptor");
        }
        if ((descriptor.revents & (POLLIN | POLLHUP)) == 0) {
            return tl::unexpected("UCI input poll reported an input device error");
        }
        const auto received = ::read(input_, bytes_.data(), bytes_.size());
        if (received < 0) {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
                continue;
            }
            return tl::unexpected(input_error(errno));
        }
        size_ = static_cast<std::size_t>(received);
#endif
        return size_ == 0 ? NativeKind::Eof : NativeKind::Data;
    }
    return NativeKind::Cancelled;
}
}  // namespace mouse::uci
