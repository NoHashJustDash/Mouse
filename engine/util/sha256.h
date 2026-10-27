// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstddef>
#include <filesystem>
#include <span>
#include <string>

#include "engine/util/result.h"

namespace mouse {
// Independent SHA-256 artifact identity, FIPS 180-4. All state is local to the
// call; concurrent calls are independent. This utility does not authenticate a
// publisher. The caller must pin trusted expected digests where required.
[[nodiscard]] std::string sha256(std::span<const std::byte> bytes);
[[nodiscard]] Result<std::string> sha256_file(const std::filesystem::path& path);
}  // namespace mouse
