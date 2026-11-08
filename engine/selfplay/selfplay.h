// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "engine/selfplay/training_data.h"
#include "engine/uci/options.h"

namespace mouse::selfplay {
// Lc0 chess/pgn.h Opening.
struct Opening {
    std::string start_fen{chess::kStartFen};
    std::vector<chess::Move> moves;
};
// Lc0 chess/pgn.h PgnReader, for what an opening book needs: FEN tags and SAN movetext.
[[nodiscard]] Result<std::vector<Opening>> parse_pgn(std::string_view text,
                                                     const chess::AttackTables& attacks);
enum class OpeningsMode : std::uint8_t { Sequential, Shuffled, Random };
// What lc0 selfplay reads per player and colour (--player1.visits=, --black.resign-percentage=).
struct PlayerSettings {
    uci::Options options;
    std::int64_t playouts = -1, visits = -1, movetime_ms = -1;
    bool reuse_tree = false, resign_wdl_style = false;
    double resign_percentage = 0.0;
    std::uint32_t resign_earliest_move = 0, minimum_allowed_visits = 0;
    double opening_stop_probability = 0.0;
    bool verbose_thinking = false, move_thinking = false;
};
struct Settings {
    bool share_trees = true;
    std::int64_t games = -1;
    std::size_t parallelism = 8;
    bool training = false;
    std::filesystem::path training_directory;
    std::size_t policy_mode_size = 0, value_mode_size = 0;
    std::string tournament_results_file;
    double resign_playthrough = 0.0, discarded_start_chance = 0.0;
    std::string openings_file;
    bool mirror_openings = false;
    OpeningsMode openings_mode = OpeningsMode::Sequential;
    std::uint64_t random_seed = 0;
    // player1 and player2, each as white and as black.
    std::array<std::array<PlayerSettings, 2>, 2> players;
};
// Lc0's selfplay defaults, then its flags: --name=value or --name value, with a player1., player2.,
// white. or black. prefix applying to those contexts. A more specific context wins.
[[nodiscard]] Result<Settings> parse_settings(std::span<const std::string_view> arguments);
struct TournamentInfo {
    // Results of player 1 as [win, draw, loss][white, black].
    std::array<std::array<std::uint64_t, 2>, 3> results{};
    std::uint64_t move_count = 0, nodes_total = 0;
    bool finished = false;
};
[[nodiscard]] std::string tournament_status_line(const TournamentInfo& info);
using BackendFactory =
    std::function<Result<std::unique_ptr<nn::Backend>>(const nn::BackendConfig&)>;
// Lc0 selfplay/tournament.cc SelfPlayTournament. Writes each response line through `output`, from
// any thread, but one line at a time.
[[nodiscard]] Result<TournamentInfo> run_tournament(
    Settings settings, const BackendFactory& factory,
    const std::function<void(std::string_view)>& output);
[[nodiscard]] int run_selfplay_command(std::span<char*> arguments);
}  // namespace mouse::selfplay
