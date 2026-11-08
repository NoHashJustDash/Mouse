// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/uci/command_runner.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifdef _WIN32
#include <Windows.h>
#endif

#include "engine/uci/input.h"
#include "engine/uci/session.h"
namespace mouse::uci {
namespace {
std::filesystem::path executable_directory() {
#ifdef _WIN32
    std::wstring buffer(32768, L'\0');
    const auto length = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (length != 0 && length < buffer.size()) {
        buffer.resize(length);
        return std::filesystem::path{buffer}.parent_path();
    }
    return {};
#else
    std::error_code error;
    const auto self = std::filesystem::read_symlink("/proc/self/exe", error);
    return error ? std::filesystem::path{} : self.parent_path();
#endif
}
// Lc0 fd71a2d utils/configfile.cc: --config=<path> or -c <path> names the file and an empty path
// disables it. By default it is mouse.config beside the executable, and its absence is normal.
// A file that was named but cannot be opened, or whose content is invalid, is an error.
Result<void> load_config(Options& options, std::optional<std::string> named) {
    if (named && named->empty()) return {};
    const auto path = named ? std::filesystem::path(std::u8string(named->begin(), named->end()))
                            : executable_directory() / options.config_file;
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        if (named)
            return tl::unexpected("Could not open configuration file: " + path.generic_string());
        return {};
    }
    std::ostringstream text;
    text << input.rdbuf();
    auto applied = apply_config_text(options, text.str());
    if (!applied) return tl::unexpected(path.generic_string() + ": " + applied.error());
    options = std::move(*applied);
    if (named) options.config_file = *named;
    return {};
}
}  // namespace
int run_uci_command(std::span<char*> arguments, SearchAlgorithm algorithm) {
    std::vector<std::string_view> views;
    views.reserve(arguments.size());
    std::optional<std::string> config;
    for (std::size_t i = 0; i < arguments.size(); ++i) {
        const std::string_view text = arguments[i];
        if (text.starts_with("--config="))
            config = std::string{text.substr(9)};
        else if (text == "-c" && i + 1 < arguments.size())
            config = std::string{arguments[++i]};
        else
            views.emplace_back(text);
    }
    auto options = startup_options(views);
    if (!options) {
        std::cerr << "Mouse error: " << options.error() << '\n';
        return 2;
    }
    if (auto loaded = load_config(*options, std::move(config)); !loaded) {
        std::cerr << "Mouse error: " << loaded.error() << '\n';
        return 2;
    }
    options->algorithm = algorithm;
    Session session(*options, [](std::string_view line) {
        std::cout << line << '\n' << std::flush;
        if (!std::cout) throw std::runtime_error("UCI output stream failed");
    });
    InputReader input;
    bool input_failed = false;
    while (true) {
        const auto event = input.read(session.cancellation_token());
        if (!event) {
            session.shutdown();
            std::cerr << "Mouse error: " << event.error() << '\n';
            input_failed = true;
            break;
        }
        if (event->kind == ReadKind::Cancelled || event->kind == ReadKind::Eof) break;
        if (event->kind == ReadKind::RejectedLine) {
            if (!session.report_input_error(event->line)) break;
        } else if (!session.handle(event->line))
            break;
    }
    session.shutdown();
    return input_failed ? 1 : session.exit_code();
}
}  // namespace mouse::uci
