// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <span>

#include "engine/uci/options.h"
namespace mouse::uci {
[[nodiscard]] int run_uci_command(std::span<char*> arguments,
                                  SearchAlgorithm algorithm = SearchAlgorithm::Classic);
}  // namespace mouse::uci
