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
#include "engine/uci/score.h"
#include "engine/uci/time.h"
namespace mouse::uci {
// Lc0 ContemptMode (classic/params.cc).
enum class ContemptMode : std::uint8_t { Play, WhiteSideAnalysis, BlackSideAnalysis, Disable };
// Lc0's search algorithms, chosen on the command line: "mouse policyhead" plays the policy's first
// choice, "mouse valuehead" the move whose position the value head likes best, and
// "mouse dag-preview" searches a graph in which transpositions share their evaluation and edges.
enum class SearchAlgorithm : std::uint8_t { Classic, PolicyHead, ValueHead, DagPreview };
struct Options {
    explicit Options(std::optional<std::filesystem::path> cache_override = std::nullopt);
    SearchAlgorithm algorithm = SearchAlgorithm::Classic;
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
    // Lc0 TaskWorkers: helper threads for a round's host work. -1 keeps Threads, the count
    // measured best for the two-slot pipeline (see kDefaultThreads in options.cpp).
    std::int64_t task_workers = -1;
    // Lc0 MaxConcurrentSearchers and SearchSpinBackoff. Mouse has one gathering coordinator, so
    // any limit of one or more already holds, and that coordinator never spins for a turn.
    std::uint64_t max_concurrent_searchers = 1;
    bool search_spin_backoff = false;
    // Lc0 contempt inputs; search.draw_rate.contempt holds the value they resolve to.
    ContemptMode contempt_mode = ContemptMode::Play;
    std::string contempt, uci_opponent;
    double uci_rating_advantage = 0.0;
    ScoreType score_type = ScoreType::WdlMu;
    bool per_pv_counters = false, show_moves_left = false, verbose_move_stats = false;
    std::string log_file;
    std::string config_file = "mouse.config";
    std::string time_manager_text = "legacy";
    TimeManagerConfig time_manager;
    std::string backend_options;
    // Lc0 SyzygyPath; the session opens the tables into search.tablebase.
    std::string syzygy_path;
    bool syzygy_fast_play = false;
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
// Lc0 utils/configfile.cc format: one parameter per line, "--name=value" or "name=value", blank
// lines and "#" comments ignored. A name is an option name or Lc0's long flag for it, ignoring
// case and dashes. Applied in order; anything unknown or invalid is an error naming its line.
[[nodiscard]] Result<Options> apply_config_text(Options options, std::string_view text);
}  // namespace mouse::uci
