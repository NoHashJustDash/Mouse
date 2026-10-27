// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <span>
namespace mouse::search {
// Application boundary: success publishes one complete JSON object; errors go
// to stderr and never publish a partial search result.
int run_search_command(std::span<char*> arguments);
}  // namespace mouse::search
