// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/uci/command_runner.h"

#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "engine/uci/input.h"
#include "engine/uci/session.h"
namespace mouse::uci {
int run_uci_command(std::span<char*> arguments) {
    std::vector<std::string_view> views;
    views.reserve(arguments.size());
    for (const char* text : arguments) views.emplace_back(text);
    const auto options = startup_options(views);
    if (!options) {
        std::cerr << "Mouse error: " << options.error() << '\n';
        return 2;
    }
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
