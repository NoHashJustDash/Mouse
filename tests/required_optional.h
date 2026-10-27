// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <optional>
#include <stdexcept>

namespace mouse::tests {
// Test assertions must fail on a missing value rather than dereference it.
template <typename T>
const T& required_optional(const std::optional<T>& value) {
    if (!value.has_value()) throw std::logic_error("required optional test value is absent");
    return *value;
}
}  // namespace mouse::tests
