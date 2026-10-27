// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <span>
namespace mouse::uci {
[[nodiscard]] int run_uci_command(std::span<char*> arguments);
}  // namespace mouse::uci
