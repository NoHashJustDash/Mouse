// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/uci/input.h"

#include <array>
#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include "engine/uci/commands.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace {
using namespace std::chrono_literals;
using mouse::uci::InputReader;
using mouse::uci::NativeInput;
using mouse::uci::ReadKind;

void close_native(NativeInput handle) {
#ifdef _WIN32
    if (handle != nullptr && handle != INVALID_HANDLE_VALUE) {
        CloseHandle(handle);
    }
#else
    if (handle >= 0) {
        close(handle);
    }
#endif
}
class Pipe {
   public:
    Pipe() {
#ifdef _WIN32
        REQUIRE(CreatePipe(&read_handle, &write_handle, nullptr, 0));
#else
        std::array<int, 2> handles{};
        REQUIRE(pipe(handles.data()) == 0);
        read_handle = handles[0];
        write_handle = handles[1];
#endif
    }
    Pipe(const Pipe&) = delete;
    Pipe& operator=(const Pipe&) = delete;
    Pipe(Pipe&&) = delete;
    Pipe& operator=(Pipe&&) = delete;
    ~Pipe() {
        close_native(read_handle);
        close_native(write_handle);
    }
    void write(std::string_view bytes) const {
#ifdef _WIN32
        DWORD written = 0;
        REQUIRE(WriteFile(write_handle, bytes.data(), static_cast<DWORD>(bytes.size()), &written,
                          nullptr));
        REQUIRE(written == bytes.size());
#else
        REQUIRE(::write(write_handle, bytes.data(), bytes.size()) ==
                static_cast<ssize_t>(bytes.size()));
#endif
    }
    void finish() {
        close_native(write_handle);
#ifdef _WIN32
        write_handle = nullptr;
#else
        write_handle = -1;
#endif
    }
#ifdef _WIN32
    NativeInput read_handle = nullptr;
    NativeInput write_handle = nullptr;
#else
    NativeInput read_handle = -1;
    NativeInput write_handle = -1;
#endif
};
class File {
   public:
    explicit File(std::string_view contents) {
        static std::atomic<unsigned int> sequence{0};
        path = std::filesystem::temp_directory_path() /
               ("mouse-uci-input-" +
                std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" +
                std::to_string(sequence.fetch_add(1)));
        {
            std::ofstream output(path, std::ios::binary);
            REQUIRE(output);
            output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
            REQUIRE(output);
        }
#ifdef _WIN32
        // The file must be written and closed before the native read handle opens.
        // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
        handle = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE,
                             nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        REQUIRE(handle != INVALID_HANDLE_VALUE);
#else
        // The file must be written and closed before the native read handle opens.
        // POSIX open is variadic even when O_RDONLY requires no mode argument.
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
        const auto descriptor = open(path.c_str(), O_RDONLY);
        REQUIRE(descriptor >= 0);
        handle = descriptor;
#endif
    }
    File(const File&) = delete;
    File& operator=(const File&) = delete;
    File(File&&) = delete;
    File& operator=(File&&) = delete;
    ~File() {
        close_native(handle);
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
    }
    std::filesystem::path path;
#ifdef _WIN32
    NativeInput handle = nullptr;
#else
    NativeInput handle = -1;
#endif
};
void expect_line(InputReader& reader, std::string_view expected) {
    const auto event = reader.read({});
    REQUIRE(event);
    CHECK(event->kind == ReadKind::Line);
    CHECK(event->line == expected);
}
void expect_rejected(InputReader& reader) {
    const auto event = reader.read({});
    REQUIRE(event);
    CHECK(event->kind == ReadKind::RejectedLine);
    CHECK(event->line.find("maximum") != std::string::npos);
    CHECK(event->line.size() < 256);
}
void expect_eof(InputReader& reader) {
    const auto event = reader.read({});
    REQUIRE(event);
    CHECK(event->kind == ReadKind::Eof);
    CHECK(event->line.empty());
}
}  // namespace

TEST_CASE("Native input frames pipe CRLF empty and final unterminated lines", "[uci][input]") {
    Pipe pipe;
    pipe.write("uci\r\n\nisready\nquit");
    pipe.finish();
    InputReader reader(pipe.read_handle);
    expect_line(reader, "uci");
    expect_line(reader, "");
    expect_line(reader, "isready");
    expect_line(reader, "quit");
    expect_eof(reader);
    expect_eof(reader);
}
TEST_CASE("Native input preserves embedded NUL for strict parser rejection", "[uci][input]") {
    File file(std::string("uc\0i\n", 5));
    InputReader reader(file.handle);
    const auto event = reader.read({});
    REQUIRE(event);
    REQUIRE(event->kind == ReadKind::Line);
    CHECK(event->line == std::string("uc\0i", 4));
    CHECK_FALSE(mouse::uci::parse_command(event->line));
    expect_eof(reader);
}
TEST_CASE("Native input accepts the exact raw byte limit including final CR", "[uci][input]") {
    const std::string maximum(mouse::uci::kMaximumLineBytes - 1, 'x');
    File file(maximum + "\r\nuci\n");
    InputReader reader(file.handle);
    expect_line(reader, maximum);
    expect_line(reader, "uci");
    expect_eof(reader);
}
TEST_CASE("Native input discards the whole oversized line and retains the next command",
          "[uci][input]") {
    File file(std::string(mouse::uci::kMaximumLineBytes * 3, 'x') + "\nisready\nquit\n");
    InputReader reader(file.handle);
    expect_rejected(reader);
    expect_line(reader, "isready");
    expect_line(reader, "quit");
    expect_eof(reader);
}
TEST_CASE("Native input reports oversized EOF once without exposing a truncated line",
          "[uci][input]") {
    File file(std::string(mouse::uci::kMaximumLineBytes + 1, 'x'));
    InputReader reader(file.handle);
    expect_rejected(reader);
    expect_eof(reader);
}
TEST_CASE("Native input distinguishes pre-requested cancellation and retains buffered commands",
          "[uci][input]") {
    File file("uci\nquit\n");
    InputReader reader(file.handle);
    expect_line(reader, "uci");
    std::stop_source stop;
    stop.request_stop();
    const auto cancelled = reader.read(stop.get_token());
    REQUIRE(cancelled);
    CHECK(cancelled->kind == ReadKind::Cancelled);
    CHECK(cancelled->line.empty());
    expect_line(reader, "quit");
    expect_eof(reader);
}
TEST_CASE("Native input cancellation wakes an idle anonymous pipe without EOF", "[uci][input]") {
    Pipe pipe;
    InputReader reader(pipe.read_handle);
    std::stop_source fatal_backend;
    std::jthread cancel([&] {
        std::this_thread::sleep_for(10ms);
        fatal_backend.request_stop();
    });
    const auto start = std::chrono::steady_clock::now();
    const auto cancelled = reader.read(fatal_backend.get_token());
    REQUIRE(cancelled);
    CHECK(cancelled->kind == ReadKind::Cancelled);
    CHECK(std::chrono::steady_clock::now() - start < 200ms);
    pipe.write("isready\n");
    expect_line(reader, "isready");
}
TEST_CASE("Native input cancellation preserves a partial line for a later read", "[uci][input]") {
    Pipe pipe;
    pipe.write("isrea");
    InputReader reader(pipe.read_handle);
    std::stop_source stop;
    std::jthread cancel([&] {
        std::this_thread::sleep_for(10ms);
        stop.request_stop();
    });
    const auto event = reader.read(stop.get_token());
    REQUIRE(event);
    CHECK(event->kind == ReadKind::Cancelled);
    pipe.write("dy\n");
    expect_line(reader, "isready");
}
TEST_CASE("Native input cancellation during oversized discard still preserves framing",
          "[uci][input]") {
    Pipe pipe;
    InputReader reader(pipe.read_handle);
    std::stop_source stop;
    std::jthread producer([&] {
        const std::string chunk(1024, 'x');
        for (std::size_t i = 0; i < mouse::uci::kMaximumLineBytes / chunk.size() + 1; ++i) {
#ifdef _WIN32
            DWORD written = 0;
            if (!WriteFile(pipe.write_handle, chunk.data(), static_cast<DWORD>(chunk.size()),
                           &written, nullptr) ||
                written != chunk.size()) {
                return;
            }
#else
            if (::write(pipe.write_handle, chunk.data(), chunk.size()) !=
                static_cast<ssize_t>(chunk.size())) {
                return;
            }
#endif
        }
        stop.request_stop();
    });
    const auto event = reader.read(stop.get_token());
    REQUIRE(event);
    CHECK(event->kind == ReadKind::Cancelled);
    producer.join();
    // Cancellation can leave a full native pipe. Resume writing concurrently
    // with the resumed read so this test cannot deadlock its own producer.
    std::atomic<bool> completed{false};
    std::jthread continuation([&] {
#ifdef _WIN32
        DWORD written = 0;
        completed.store(WriteFile(pipe.write_handle, "\nuci\n", 5, &written, nullptr) &&
                        written == 5);
#else
        completed.store(::write(pipe.write_handle, "\nuci\n", 5) == 5);
#endif
    });
    expect_rejected(reader);
    expect_line(reader, "uci");
    continuation.join();
    CHECK(completed.load());
}
TEST_CASE("Native input reports invalid handles as errors rather than EOF", "[uci][input]") {
#ifdef _WIN32
    InputReader reader(INVALID_HANDLE_VALUE);
#else
    InputReader reader(-1);
#endif
    const auto event = reader.read({});
    REQUIRE_FALSE(event);
    CHECK(event.error().find("input") != std::string::npos);
}
TEST_CASE("Native input borrows and never closes the supplied handle", "[uci][input]") {
    Pipe pipe;
    {
        InputReader reader(pipe.read_handle);
        pipe.write("uci\n");
        expect_line(reader, "uci");
    }
    InputReader next(pipe.read_handle);
    pipe.write("quit\n");
    expect_line(next, "quit");
}
TEST_CASE("Native input preserves a final CR without a newline", "[uci][input]") {
    File file("quit\r");
    InputReader reader(file.handle);
    expect_line(reader, "quit\r");
    expect_eof(reader);
}
TEST_CASE("Native default input binds stdin without owning or changing it", "[uci][input]") {
    Pipe pipe;
#ifdef _WIN32
    const auto original = GetStdHandle(STD_INPUT_HANDLE);
    struct Restore {
        explicit Restore(NativeInput value) : handle(value) {}
        Restore(const Restore&) = delete;
        Restore& operator=(const Restore&) = delete;
        Restore(Restore&&) = delete;
        Restore& operator=(Restore&&) = delete;
        NativeInput handle;
        ~Restore() { SetStdHandle(STD_INPUT_HANDLE, handle); }
    } restore{original};
    REQUIRE(SetStdHandle(STD_INPUT_HANDLE, pipe.read_handle));
#else
    const int original = dup(STDIN_FILENO);
    REQUIRE(original >= 0);
    struct Restore {
        explicit Restore(int value) : descriptor(value) {}
        Restore(const Restore&) = delete;
        Restore& operator=(const Restore&) = delete;
        Restore(Restore&&) = delete;
        Restore& operator=(Restore&&) = delete;
        int descriptor;
        ~Restore() {
            dup2(descriptor, STDIN_FILENO);
            close(descriptor);
        }
    } restore{original};
    REQUIRE(dup2(pipe.read_handle, STDIN_FILENO) == STDIN_FILENO);
#endif
    InputReader reader;
    pipe.write("uci\n");
    expect_line(reader, "uci");
}
#ifdef _WIN32
namespace {
void console_text(NativeInput input, std::wstring_view text) {
    for (const wchar_t character : text) {
        INPUT_RECORD record{};
        record.EventType = KEY_EVENT;
        // SDK KEY_EVENT/UnicodeChar tagged union initialization.
        // NOLINTBEGIN(cppcoreguidelines-pro-type-union-access)
        record.Event.KeyEvent.bKeyDown = TRUE;
        record.Event.KeyEvent.wRepeatCount = 1;
        record.Event.KeyEvent.uChar.UnicodeChar = character;
        // NOLINTEND(cppcoreguidelines-pro-type-union-access)
        DWORD written = 0;
        REQUIRE(WriteConsoleInputW(input, &record, 1, &written));
        REQUIRE(written == 1);
    }
}
struct ConsoleHandle {
    NativeInput handle =
        CreateFileW(L"CONIN$", GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                    nullptr, OPEN_EXISTING, 0, nullptr);
    ConsoleHandle() { REQUIRE(handle != INVALID_HANDLE_VALUE); }
    ConsoleHandle(const ConsoleHandle&) = delete;
    ConsoleHandle& operator=(const ConsoleHandle&) = delete;
    ConsoleHandle(ConsoleHandle&&) = delete;
    ConsoleHandle& operator=(ConsoleHandle&&) = delete;
    ~ConsoleHandle() { close_native(handle); }
};
}  // namespace
TEST_CASE("Native console hidden child validates queued Unicode editing and cancellation",
          "[.input-console-child]") {
    ConsoleHandle console;
    REQUIRE(FlushConsoleInputBuffer(console.handle));
    InputReader reader(console.handle);
    console_text(console.handle, L"ucx\bi\r");
    expect_line(reader, "uci");
    console_text(console.handle, L"\xd83d\xde42\b\xe9\r");
    expect_line(reader, "\xc3\xa9");
    std::stop_source stop;
    std::jthread cancel([&] {
        std::this_thread::sleep_for(10ms);
        stop.request_stop();
    });
    console_text(console.handle, L"isrea");
    const auto start = std::chrono::steady_clock::now();
    const auto cancelled = reader.read(stop.get_token());
    REQUIRE(cancelled);
    CHECK(cancelled->kind == ReadKind::Cancelled);
    CHECK(std::chrono::steady_clock::now() - start < 200ms);
    console_text(console.handle, L"dy\r");
    expect_line(reader, "isready");
    INPUT_RECORD repeated{};
    repeated.EventType = KEY_EVENT;
    // SDK KEY_EVENT/UnicodeChar tagged union initialization.
    // NOLINTBEGIN(cppcoreguidelines-pro-type-union-access)
    repeated.Event.KeyEvent.bKeyDown = TRUE;
    repeated.Event.KeyEvent.wRepeatCount = 3;
    repeated.Event.KeyEvent.uChar.UnicodeChar = L'q';
    // NOLINTEND(cppcoreguidelines-pro-type-union-access)
    DWORD written = 0;
    REQUIRE(WriteConsoleInputW(console.handle, &repeated, 1, &written));
    console_text(console.handle, L"\r");
    expect_line(reader, "qqq");
    console_text(console.handle, L"\xd800x");
    CHECK_FALSE(reader.read({}));
    console_text(console.handle, L"uci\r");
    expect_line(reader, "uci");
    console_text(console.handle, L"quit\x1a");
    expect_line(reader, "quit");
    expect_eof(reader);
}
TEST_CASE("Native console transport is exercised in a hidden independent console", "[uci][input]") {
    Pipe capture;
    REQUIRE(SetHandleInformation(capture.write_handle, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT));
    REQUIRE(SetHandleInformation(capture.read_handle, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT));
    std::array<wchar_t, 32768> path{};
    const auto length = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
    REQUIRE(length != 0);
    REQUIRE(length < path.size());
    std::wstring command =
        L"\"" + std::wstring(path.data(), length) + L"\" [input-console-child] --reporter compact";
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESHOWWINDOW | STARTF_USESTDHANDLES;
    startup.wShowWindow = SW_HIDE;
    startup.hStdInput = capture.read_handle;
    startup.hStdOutput = capture.write_handle;
    startup.hStdError = capture.write_handle;
    struct Process {
        Process() = default;
        Process(const Process&) = delete;
        Process& operator=(const Process&) = delete;
        Process(Process&&) = delete;
        Process& operator=(Process&&) = delete;
        PROCESS_INFORMATION info{};
        ~Process() {
            close_native(info.hThread);
            close_native(info.hProcess);
        }
    } process;
    REQUIRE(CreateProcessW(nullptr, command.data(), nullptr, nullptr, TRUE, CREATE_NEW_CONSOLE,
                           nullptr, nullptr, &startup, &process.info));
    capture.finish();
    const auto waited = WaitForSingleObject(process.info.hProcess, 5000);
    if (waited != WAIT_OBJECT_0) {
        TerminateProcess(process.info.hProcess, 1);
        WaitForSingleObject(process.info.hProcess, 1000);
    }
    std::string output;
    std::array<char, 4096> buffer{};
    DWORD received = 0;
    while (ReadFile(capture.read_handle, buffer.data(), static_cast<DWORD>(buffer.size()),
                    &received, nullptr) &&
           received != 0) {
        output.append(buffer.data(), received);
    }
    INFO(output);
    CHECK(waited == WAIT_OBJECT_0);
    DWORD code = 1;
    REQUIRE(GetExitCodeProcess(process.info.hProcess, &code));
    CHECK(code == 0);
    CHECK(output.find("All tests passed") != std::string::npos);
}
#endif

TEST_CASE("Native pipe write cancellation races conserve every command", "[uci][input][race]") {
    Pipe pipe;
    InputReader reader(pipe.read_handle);
    for (unsigned sequence = 0; sequence < 192; ++sequence) {
        const auto expected = "command-" + std::to_string(sequence);
        const auto frame = expected + "\n";
        std::stop_source stop;
        std::atomic<bool> wrote{false};
        std::jthread producer([&] {
            if (sequence % 3 == 0) stop.request_stop();
#ifdef _WIN32
            DWORD written = 0;
            wrote.store(WriteFile(pipe.write_handle, frame.data(), static_cast<DWORD>(frame.size()),
                                  &written, nullptr) &&
                        written == frame.size());
#else
            wrote.store(::write(pipe.write_handle, frame.data(), frame.size()) ==
                        static_cast<ssize_t>(frame.size()));
#endif
            if (sequence % 3 != 0) stop.request_stop();
        });
        const auto first = reader.read(stop.get_token());
        producer.join();
        REQUIRE(wrote.load());
        REQUIRE(first);
        if (first->kind == ReadKind::Line) {
            CHECK(first->line == expected);
        } else {
            REQUIRE(first->kind == ReadKind::Cancelled);
            expect_line(reader, expected);
        }
        // A cancellation aimed at the old read must never reach this new one.
        pipe.write("following\n");
        expect_line(reader, "following");
    }
    pipe.finish();
    expect_eof(reader);
}

TEST_CASE("Native EOF cancellation races retain the final unterminated bytes",
          "[uci][input][race]") {
    for (unsigned sequence = 0; sequence < 64; ++sequence) {
        Pipe pipe;
        InputReader reader(pipe.read_handle);
        pipe.write("final\r");
        std::stop_source stop;
        std::jthread closer([&] {
            if (sequence % 2 == 0) stop.request_stop();
            pipe.finish();
            if (sequence % 2 != 0) stop.request_stop();
        });
        const auto first = reader.read(stop.get_token());
        closer.join();
        REQUIRE(first);
        if (first->kind == ReadKind::Line) {
            CHECK(first->line == "final\r");
        } else {
            REQUIRE(first->kind == ReadKind::Cancelled);
            expect_line(reader, "final\r");
        }
        expect_eof(reader);
        expect_eof(reader);
    }
}

TEST_CASE("Repeated idle pipe cancellations drain before reuse and destruction",
          "[uci][input][race]") {
    Pipe pipe;
    for (unsigned sequence = 0; sequence < 64; ++sequence) {
        InputReader reader(pipe.read_handle);
        std::stop_source stop;
        std::jthread cancel([&] {
            std::this_thread::yield();
            stop.request_stop();
        });
        const auto cancelled = reader.read(stop.get_token());
        cancel.join();
        REQUIRE(cancelled);
        CHECK(cancelled->kind == ReadKind::Cancelled);
        pipe.write("alive\n");
        expect_line(reader, "alive");
    }
    InputReader next(pipe.read_handle);
    pipe.write("quit\n");
    expect_line(next, "quit");
}

#ifdef _WIN32
TEST_CASE("Windows pipe readers release their owned thread handles before destruction returns",
          "[uci][input][lifetime]") {
    Pipe pipe;
    const auto once = [&] {
        InputReader reader{pipe.read_handle};
        pipe.write("owned\n");
        expect_line(reader, "owned");
    };
    // Exclude one-time CRT/thread initialization from the resource baseline.
    once();
    DWORD before = 0, after = 0;
    REQUIRE(GetProcessHandleCount(GetCurrentProcess(), &before));
    for (unsigned iteration = 0; iteration < 128; ++iteration) once();
    REQUIRE(GetProcessHandleCount(GetCurrentProcess(), &after));
    CHECK(after == before);
    // The borrowed handle remains reusable after every joined worker.
    once();
}
#endif
