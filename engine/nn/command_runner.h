// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <span>

#include "engine/nn/commands.h"

namespace mouse::nn {
// Resolve user-selected model/cache paths against the current directory and
// bind the app-local runtime. Never substitutes an artifact or provider.
[[nodiscard]] Result<void> resolve_backend_cli_paths(BackendConfig& config);
// Standalone process commands. Output streams and the Windows stdout binary
// mode are process resources; call only from main before starting other work.
int run_backend_command(BackendCommandKind kind, std::span<char*> arguments);
}  // namespace mouse::nn
