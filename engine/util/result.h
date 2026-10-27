// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <string>
#include <tl/expected.hpp>

namespace mouse {
template <class T>
using Result = tl::expected<T, std::string>;
}  // namespace mouse
