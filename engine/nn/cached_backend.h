// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstddef>
#include <memory>

#include "engine/nn/backend.h"

namespace mouse::nn {
// Owns a Backend and an independent exact-input cache scoped to its immutable
// cache_namespace. Evaluate serializes all cache/staging/provider use. Caller
// outputs publish only after the entire batch succeeds. Immutable info/metadata
// delegate to the owned backend and remain valid for this owner's lifetime.
//
// max_batch is 1..1024. Scratch for inputs, row indices, misses and staged outputs
// is preallocated separately from the NN LRU byte budget. Zero cache bytes
// returns the original backend after checking null/max_batch, with no wrapper.
[[nodiscard]] Result<std::unique_ptr<Backend>> create_cached_backend(
    std::unique_ptr<Backend> backend, std::size_t budget_bytes, std::size_t max_batch);
}  // namespace mouse::nn
