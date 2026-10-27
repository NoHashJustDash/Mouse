// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "engine/nn/backend.h"
#include "engine/search/search.h"
#include "engine/uci/commands.h"
namespace mouse::uci {
struct Options {
    explicit Options(std::optional<std::filesystem::path> cache_override = std::nullopt);
    nn::BackendConfig backend;
    search::SearchConfig search;
    // MinibatchSize 0 (lc0's default): the session resolves the batch for the
    // loaded backend (nn::recommended_minibatch). Until then search and backend
    // carry the CPU default shape.
    bool auto_minibatch = true;
    std::size_t nncache_entries = 2'000'000;
    // lc0 CacheHistoryLength: earlier positions included in the NN cache key.
    std::size_t cache_history_length = 0;
    std::size_t multipv = 1;
    std::uint64_t move_overhead_ms = 30;
    double smart_pruning_factor = 1.33;
    std::uint64_t minimum_smart_pruning_batches = 0;
    bool log_search_stats = false;
    bool show_wdl = false, chess960 = false, ponder = false;
};
enum class Reconfigure : std::uint8_t { None, Search, Backend };
struct OptionUpdate {
    Options options;
    Reconfigure impact = Reconfigure::None;
};
// Pure configuration transactions: invalid values leave the caller unchanged.
// Artifact existence/provider compatibility are checked at explicit readiness.
[[nodiscard]] Result<OptionUpdate> apply_option(const Options&, const SetOptionCommand&);
[[nodiscard]] Result<Options> startup_options(std::span<const std::string_view> arguments);
[[nodiscard]] std::vector<std::string> option_lines(const Options&);
// TensorRT/CUDA batch buckets used for a minibatch: 1, 8 and 32 below it, then it.
[[nodiscard]] std::vector<std::size_t> minibatch_buckets(std::size_t batch);
}  // namespace mouse::uci
