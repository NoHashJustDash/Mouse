// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/nn/tensorrt_builder.h"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <Windows.h>
#else
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

namespace mouse::nn::detail {
namespace {
constexpr std::uint64_t kMiB = 1ULL << 20U;
// Enough of the child's stderr for its error; TensorRT also logs plugin registration there.
constexpr std::size_t kErrorTail = 16384;

void keep_tail(std::string& tail, std::string_view text) {
    tail.append(text);
    if (tail.size() > kErrorTail) tail.erase(0, tail.size() - kErrorTail);
}
// The child's own "Mouse error:" line when it printed one, else its last stderr line.
std::string child_error(const std::string& tail) {
    std::string last;
    std::size_t begin = 0;
    std::string chosen;
    while (begin < tail.size()) {
        auto end = tail.find('\n', begin);
        if (end == std::string::npos) end = tail.size();
        auto line = tail.substr(begin, end - begin);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.starts_with("Mouse error: ")) chosen = line.substr(13);
        if (!line.empty()) last = line;
        begin = end + 1;
    }
    return chosen.empty() ? last : chosen;
}
// Splits complete lines out of `pending` and reports them.
void report_lines(std::string& pending, const BackendConfig& config) {
    for (auto end = pending.find('\n'); end != std::string::npos; end = pending.find('\n')) {
        auto line = pending.substr(0, end);
        pending.erase(0, end + 1);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty() && config.initialization_progress) config.initialization_progress(line);
    }
}

#ifdef _WIN32
class Handle {
   public:
    Handle() = default;
    explicit Handle(HANDLE value) : value_(value) {}
    ~Handle() { reset(); }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    Handle(Handle&& other) noexcept : value_(std::exchange(other.value_, nullptr)) {}
    Handle& operator=(Handle&& other) noexcept {
        if (this != &other) {
            reset();
            value_ = std::exchange(other.value_, nullptr);
        }
        return *this;
    }
    [[nodiscard]] HANDLE get() const noexcept { return value_; }
    void reset() noexcept {
        if (value_ != nullptr && value_ != INVALID_HANDLE_VALUE) CloseHandle(value_);
        value_ = nullptr;
    }

   private:
    HANDLE value_ = nullptr;
};
[[noreturn]] void fail(const char* what) {
    throw std::runtime_error(std::string{"TensorRT build helper: "} + what + " (Windows error " +
                             std::to_string(GetLastError()) + ")");
}
// CommandLineToArgvW quoting.
void append_argument(std::wstring& line, const std::wstring& argument) {
    if (!line.empty()) line.push_back(L' ');
    if (!argument.empty() && argument.find_first_of(L" \t\n\v\"") == std::wstring::npos) {
        line.append(argument);
        return;
    }
    line.push_back(L'"');
    std::size_t slashes = 0;
    for (const wchar_t value : argument) {
        if (value == L'\\') {
            ++slashes;
            continue;
        }
        line.append(value == L'"' ? 2 * slashes + 1 : slashes, L'\\');
        slashes = 0;
        line.push_back(value);
    }
    line.append(2 * slashes, L'\\');
    line.push_back(L'"');
}
void pipe(Handle& read, Handle& write) {
    SECURITY_ATTRIBUTES inherit{.nLength = sizeof(SECURITY_ATTRIBUTES),
                                .lpSecurityDescriptor = nullptr,
                                .bInheritHandle = TRUE};
    HANDLE r = nullptr, w = nullptr;
    if (CreatePipe(&r, &w, &inherit, 0) == 0) fail("cannot create a pipe");
    read = Handle{r};
    write = Handle{w};
    if (SetHandleInformation(read.get(), HANDLE_FLAG_INHERIT, 0) == 0)
        fail("cannot keep a pipe end private");
}
#else
class Descriptor {
   public:
    Descriptor() = default;
    explicit Descriptor(int value) : value_(value) {}
    ~Descriptor() { reset(); }
    Descriptor(const Descriptor&) = delete;
    Descriptor& operator=(const Descriptor&) = delete;
    [[nodiscard]] int get() const noexcept { return value_; }
    void reset() noexcept {
        if (value_ >= 0) close(value_);
        value_ = -1;
    }

   private:
    int value_ = -1;
};
#endif
}  // namespace

std::vector<std::filesystem::path> tensorrt_builder_arguments(const BackendConfig& config) {
    if (config.kind != BackendKind::TensorRt)
        throw std::logic_error("Only TensorRT engines are built in a helper process");
    if (config.workspace_bytes % kMiB != 0)
        throw std::logic_error("TensorRT workspace must be a whole number of MiB");
    std::string buckets;
    for (const auto bucket : config.batch_buckets) {
        if (!buckets.empty()) buckets.push_back(',');
        buckets += std::to_string(bucket);
    }
    std::vector<std::filesystem::path> arguments{
        "build-tensorrt-cache",
        "--weights",
        config.model_path,
        "--backend",
        "tensorrt",
        "--precision",
        config.precision == ComputePrecision::Float16 ? "fp16" : "fp32",
        "--buckets",
        buckets,
        "--cache-dir",
        config.cache_directory};
    for (const auto& directory : config.gpu_dll_directories) {
        arguments.emplace_back("--gpu-dll-dir");
        arguments.push_back(directory);
    }
    for (const auto& [option, value] :
         {std::pair{"--device", std::to_string(config.device_id)},
          std::pair{"--cpu-threads", std::to_string(config.cpu_threads)},
          std::pair{"--workspace-mb", std::to_string(config.workspace_bytes / kMiB)},
          std::pair{"--deterministic-compute",
                    std::string{config.deterministic_compute ? "true" : "false"}}}) {
        arguments.emplace_back(option);
        arguments.emplace_back(value);
    }
    return arguments;
}

void build_tensorrt_cache(const BackendConfig& config) {
    const auto arguments = tensorrt_builder_arguments(config);
    if (!config.tensorrt_builder.is_absolute())
        throw std::logic_error("The TensorRT build helper needs an absolute executable path");
    std::string out, err;
    int status = 0;
#ifdef _WIN32
    std::wstring line;
    append_argument(line, config.tensorrt_builder.native());
    for (const auto& argument : arguments) append_argument(line, argument.native());
    Handle out_read, out_write, err_read, err_write;
    pipe(out_read, out_write);
    pipe(err_read, err_write);
    SECURITY_ATTRIBUTES inherit{.nLength = sizeof(SECURITY_ATTRIBUTES),
                                .lpSecurityDescriptor = nullptr,
                                .bInheritHandle = TRUE};
    Handle input{CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &inherit,
                             OPEN_EXISTING, 0, nullptr)};
    if (input.get() == INVALID_HANDLE_VALUE) fail("cannot open NUL");
    // Inherit exactly these three handles, never the engine's own UCI pipes.
    std::array<HANDLE, 3> inherited{input.get(), out_write.get(), err_write.get()};
    SIZE_T size = 0;
    InitializeProcThreadAttributeList(nullptr, 1, 0, &size);
    std::vector<std::byte> storage(size);
    auto* attributes = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(storage.data());
    if (InitializeProcThreadAttributeList(attributes, 1, 0, &size) == 0)
        fail("cannot prepare process attributes");
    struct Attributes {
        LPPROC_THREAD_ATTRIBUTE_LIST list;
        ~Attributes() { DeleteProcThreadAttributeList(list); }
    } owned{attributes};
    if (UpdateProcThreadAttribute(attributes, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                                  inherited.data(), sizeof(inherited), nullptr, nullptr) == 0)
        fail("cannot restrict inherited handles");
    STARTUPINFOEXW startup{};
    startup.StartupInfo.cb = sizeof(startup);
    startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    startup.StartupInfo.hStdInput = input.get();
    startup.StartupInfo.hStdOutput = out_write.get();
    startup.StartupInfo.hStdError = err_write.get();
    startup.lpAttributeList = attributes;
    Handle job{CreateJobObjectW(nullptr, nullptr)};
    if (job.get() == nullptr) fail("cannot create a job object");
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if (SetInformationJobObject(job.get(), JobObjectExtendedLimitInformation, &limits,
                                sizeof(limits)) == 0)
        fail("cannot limit the job object");
    PROCESS_INFORMATION created{};
    if (CreateProcessW(config.tensorrt_builder.c_str(), line.data(), nullptr, nullptr, TRUE,
                       EXTENDED_STARTUPINFO_PRESENT | CREATE_SUSPENDED | CREATE_NO_WINDOW, nullptr,
                       nullptr, &startup.StartupInfo, &created) == 0)
        fail(("cannot start " + config.tensorrt_builder.string()).c_str());
    Handle process{created.hProcess}, thread{created.hThread};
    if (AssignProcessToJobObject(job.get(), process.get()) == 0) {
        TerminateProcess(process.get(), 1);
        fail("cannot place the child in its job");
    }
    ResumeThread(thread.get());
    // The child holds the write ends now; ours must close for the reads to end.
    out_write.reset();
    err_write.reset();
    input.reset();
    std::thread errors([&err, &err_read] {
        std::array<char, 4096> buffer{};
        DWORD count = 0;
        while (ReadFile(err_read.get(), buffer.data(), static_cast<DWORD>(buffer.size()), &count,
                        nullptr) != 0 &&
               count != 0)
            keep_tail(err, {buffer.data(), count});
    });
    std::array<char, 4096> buffer{};
    DWORD count = 0;
    while (ReadFile(out_read.get(), buffer.data(), static_cast<DWORD>(buffer.size()), &count,
                    nullptr) != 0 &&
           count != 0) {
        out.append(buffer.data(), count);
        report_lines(out, config);
    }
    errors.join();
    WaitForSingleObject(process.get(), INFINITE);
    DWORD code = 1;
    if (GetExitCodeProcess(process.get(), &code) == 0) fail("cannot read the child's exit code");
    status = static_cast<int>(code);
#else
    int out_pipe[2], err_pipe[2];
    if (::pipe(out_pipe) != 0)
        throw std::runtime_error("TensorRT build helper: cannot create a pipe");
    Descriptor out_read{out_pipe[0]}, out_write{out_pipe[1]};
    if (::pipe(err_pipe) != 0)
        throw std::runtime_error("TensorRT build helper: cannot create a pipe");
    Descriptor err_read{err_pipe[0]}, err_write{err_pipe[1]};
    for (const int descriptor : {out_read.get(), err_read.get()})
        fcntl(descriptor, F_SETFD, FD_CLOEXEC);
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addopen(&actions, 0, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_adddup2(&actions, out_write.get(), 1);
    posix_spawn_file_actions_adddup2(&actions, err_write.get(), 2);
    std::vector<std::string> text{config.tensorrt_builder.native()};
    for (const auto& argument : arguments) text.push_back(argument.native());
    std::vector<char*> argv;
    for (auto& value : text) argv.push_back(value.data());
    argv.push_back(nullptr);
    pid_t child = 0;
    const int spawned =
        posix_spawn(&child, text.front().c_str(), &actions, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    if (spawned != 0)
        throw std::runtime_error("TensorRT build helper: cannot start " + text.front());
    out_write.reset();
    err_write.reset();
    std::thread errors([&err, &err_read] {
        std::array<char, 4096> buffer{};
        for (ssize_t count; (count = read(err_read.get(), buffer.data(), buffer.size())) > 0;)
            keep_tail(err, {buffer.data(), static_cast<std::size_t>(count)});
    });
    std::array<char, 4096> buffer{};
    for (ssize_t count; (count = read(out_read.get(), buffer.data(), buffer.size())) > 0;) {
        out.append(buffer.data(), static_cast<std::size_t>(count));
        report_lines(out, config);
    }
    errors.join();
    int wait_status = 0;
    if (waitpid(child, &wait_status, 0) != child)
        throw std::runtime_error("TensorRT build helper: cannot wait for the child");
    status = WIFEXITED(wait_status) ? WEXITSTATUS(wait_status) : 128 + WTERMSIG(wait_status);
#endif
    out.push_back('\n');
    report_lines(out, config);
    if (status != 0) {
        const auto error = child_error(err);
        throw std::runtime_error("TensorRT build helper exited " + std::to_string(status) +
                                 (error.empty() ? std::string{} : ": " + error));
    }
}
}  // namespace mouse::nn::detail
