// SPDX-License-Identifier: GPL-3.0-or-later
// Behaviour follows lc0 fd71a2d src/selfplay/{loop,tournament,game,multigame}.cc.
#include "engine/selfplay/selfplay.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>

#include "engine/chess/attacks.h"
#include "engine/chess/movegen.h"
#include "engine/nn/backend.h"
#include "engine/nn/command_runner.h"
#include "engine/nn/history.h"
#include "engine/nn/legal_cache.h"
#include "engine/nn/policy_map.h"
#include "engine/search/search.h"

namespace mouse::selfplay {
namespace {
std::string lower_no_dashes(std::string_view text) {
    std::string result;
    for (const char c : text)
        if (c != '-')
            result.push_back(static_cast<char>(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c));
    return result;
}
// Lc0 selfplay/tournament.cc PopulateOptions defaults, then game.cc's and stoppers' for selfplay.
constexpr std::array<std::pair<const char*, const char*>, 17> kSelfplayDefaults{{
    {"MinibatchSize", "32"},
    {"CPuct", "1.2"},
    {"CPuctFactor", "0"},
    {"PolicyTemperature", "1"},
    {"MaxCollisionVisits", "1"},
    {"MaxCollisionEvents", "1"},
    {"CacheHistoryLength", "7"},
    {"OutOfOrderEval", "false"},
    {"Temperature", "1"},
    {"DirichletNoiseEpsilon", "0.25"},
    {"FpuValue", "0"},
    {"HistoryFill", "no"},
    {"StickyEndgames", "false"},
    {"TwoFoldDraws", "false"},
    {"TaskWorkers", "0"},
    {"SmartPruningFactor", "0"},
    {"SmartPruningMinimumBatches", "0"},
}};
Result<void> apply_uci(uci::Options& options, std::string_view name, std::string_view value) {
    auto updated = uci::apply_config_text(options, std::string{name} + "=" + std::string{value});
    if (!updated) return tl::unexpected(updated.error());
    options = std::move(*updated);
    return {};
}
Result<double> real_in(std::string_view name, std::string_view value, double low, double high) {
    double parsed = 0;
    std::istringstream in{std::string{value}};
    in >> parsed;
    if (in.fail() || !in.eof() || !std::isfinite(parsed) || parsed < low || parsed > high) {
        std::ostringstream message;
        message << name << " requires a number in " << low << ".." << high;
        return tl::unexpected(message.str());
    }
    return parsed;
}
Result<std::int64_t> integer_in(std::string_view name, std::string_view value, std::int64_t low,
                                std::int64_t high) {
    std::int64_t parsed = 0;
    std::istringstream in{std::string{value}};
    in >> parsed;
    if (in.fail() || !in.eof() || parsed < low || parsed > high)
        return tl::unexpected(std::string{name} + " requires an integer in " +
                              std::to_string(low) + ".." + std::to_string(high));
    return parsed;
}
Result<bool> boolean_of(std::string_view name, std::string_view value) {
    if (value == "true" || value == "1" || value == "on") return true;
    if (value == "false" || value == "0" || value == "off") return false;
    return tl::unexpected(std::string{name} + " requires true or false");
}
// Options lc0 reads from each player's dictionary. Returns whether `key` was one.
Result<bool> apply_player(PlayerSettings& player, std::string_view key, std::string_view value) {
    const auto set_integer = [&](std::int64_t& target, std::int64_t low,
                                 std::int64_t high) -> Result<bool> {
        auto parsed = integer_in(key, value, low, high);
        if (!parsed) return tl::unexpected(parsed.error());
        target = *parsed;
        return true;
    };
    const auto set_boolean = [&](bool& target) -> Result<bool> {
        auto parsed = boolean_of(key, value);
        if (!parsed) return tl::unexpected(parsed.error());
        target = *parsed;
        return true;
    };
    const auto set_real = [&](double& target, double low, double high) -> Result<bool> {
        auto parsed = real_in(key, value, low, high);
        if (!parsed) return tl::unexpected(parsed.error());
        target = *parsed;
        return true;
    };
    if (key == "playouts") return set_integer(player.playouts, -1, 999999999);
    if (key == "visits") return set_integer(player.visits, -1, 999999999);
    if (key == "movetime") return set_integer(player.movetime_ms, -1, 999999999);
    if (key == "reusetree") return set_boolean(player.reuse_tree);
    if (key == "resignwdlstyle") return set_boolean(player.resign_wdl_style);
    if (key == "resignpercentage") return set_real(player.resign_percentage, 0, 100);
    if (key == "openingstopprob") return set_real(player.opening_stop_probability, 0, 1);
    if (key == "verbosethinking") return set_boolean(player.verbose_thinking);
    if (key == "movethinking") return set_boolean(player.move_thinking);
    if (key == "resignearliestmove") {
        std::int64_t parsed = 0;
        auto done = set_integer(parsed, 0, 1000);
        if (done) player.resign_earliest_move = static_cast<std::uint32_t>(parsed);
        return done;
    }
    if (key == "minimumallowedvisits") {
        std::int64_t parsed = 0;
        auto done = set_integer(parsed, 0, 1000000);
        if (done) player.minimum_allowed_visits = static_cast<std::uint32_t>(parsed);
        return done;
    }
    return false;
}
Result<bool> apply_global(Settings& settings, std::string_view key, std::string_view value) {
    if (key == "sharetrees") {
        auto parsed = boolean_of(key, value);
        if (!parsed) return tl::unexpected(parsed.error());
        settings.share_trees = *parsed;
    } else if (key == "games") {
        auto parsed = integer_in(key, value, -2, 999999);
        if (!parsed) return tl::unexpected(parsed.error());
        settings.games = *parsed;
    } else if (key == "parallelism") {
        auto parsed = integer_in(key, value, 1, 256);
        if (!parsed) return tl::unexpected(parsed.error());
        settings.parallelism = static_cast<std::size_t>(*parsed);
    } else if (key == "training") {
        auto parsed = boolean_of(key, value);
        if (!parsed) return tl::unexpected(parsed.error());
        settings.training = *parsed;
    } else if (key == "trainingdirectory") {
        settings.training_directory =
            std::filesystem::path(std::u8string(value.begin(), value.end()));
    } else if (key == "policymodesize") {
        auto parsed = integer_in(key, value, 0, 1024);
        if (!parsed) return tl::unexpected(parsed.error());
        settings.policy_mode_size = static_cast<std::size_t>(*parsed);
    } else if (key == "valuemodesize") {
        auto parsed = integer_in(key, value, 0, 64);
        if (!parsed) return tl::unexpected(parsed.error());
        settings.value_mode_size = static_cast<std::size_t>(*parsed);
    } else if (key == "tournamentresultsfile") {
        settings.tournament_results_file = std::string{value};
    } else if (key == "resignplaythrough") {
        auto parsed = real_in(key, value, 0, 100);
        if (!parsed) return tl::unexpected(parsed.error());
        settings.resign_playthrough = *parsed;
    } else if (key == "discardedstartchance") {
        auto parsed = real_in(key, value, 0, 100);
        if (!parsed) return tl::unexpected(parsed.error());
        settings.discarded_start_chance = *parsed;
    } else if (key == "openingspgn" || key == "openingspgnfile") {
        settings.openings_file = std::string{value};
    } else if (key == "mirroropenings") {
        auto parsed = boolean_of(key, value);
        if (!parsed) return tl::unexpected(parsed.error());
        settings.mirror_openings = *parsed;
    } else if (key == "openingsmode") {
        if (value == "sequential")
            settings.openings_mode = OpeningsMode::Sequential;
        else if (value == "shuffled")
            settings.openings_mode = OpeningsMode::Shuffled;
        else if (value == "random")
            settings.openings_mode = OpeningsMode::Random;
        else
            return tl::unexpected(std::string{"OpeningsMode requires sequential, shuffled or random"});
    } else if (key == "seed") {
        auto parsed = integer_in(key, value, 0, std::numeric_limits<std::int64_t>::max());
        if (!parsed) return tl::unexpected(parsed.error());
        settings.random_seed = static_cast<std::uint64_t>(*parsed);
    } else {
        return false;
    }
    return true;
}
Result<void> apply_context(PlayerSettings& player, std::string_view name, std::string_view value) {
    const auto key = lower_no_dashes(name);
    auto handled = apply_player(player, key, value);
    if (!handled) return tl::unexpected(handled.error());
    if (*handled) return {};
    return apply_uci(player.options, name, value);
}
}  // namespace

Result<Settings> parse_settings(std::span<const std::string_view> arguments) {
    uci::Options base;
    for (const auto& [name, value] : kSelfplayDefaults) {
        auto applied = apply_uci(base, name, value);
        if (!applied) return tl::unexpected(applied.error());
    }
    Settings settings;
    for (auto& per_player : settings.players)
        for (auto& player : per_player) player.options = base;
    struct Flag {
        std::string context;
        std::string name, value;
    };
    std::vector<Flag> flags;
    for (std::size_t i = 0; i < arguments.size(); ++i) {
        auto argument = arguments[i];
        if (!argument.starts_with("--"))
            return tl::unexpected("Selfplay flags start with --: " + std::string{argument});
        argument.remove_prefix(2);
        Flag flag;
        const auto equals = argument.find('=');
        if (equals != std::string_view::npos) {
            flag.name = std::string{argument.substr(0, equals)};
            flag.value = std::string{argument.substr(equals + 1)};
        } else {
            flag.name = std::string{argument};
            if (i + 1 < arguments.size() && !arguments[i + 1].starts_with("--"))
                flag.value = std::string{arguments[++i]};
            else
                flag.value = "true";
        }
        for (const std::string_view prefix : {"player1.", "player2."})
            if (flag.name.starts_with(prefix)) {
                flag.context = std::string{prefix.substr(0, 7)};
                flag.name.erase(0, prefix.size());
            }
        for (const std::string_view prefix : {"white.", "black."})
            if (flag.name.starts_with(prefix)) {
                flag.context +=
                    (flag.context.empty() ? "" : ".") + std::string{prefix.substr(0, 5)};
                flag.name.erase(0, prefix.size());
            }
        flags.push_back(std::move(flag));
    }
    // Lc0 reads the most specific dictionary first: flags apply from the general to the specific.
    const auto rank = [](const Flag& flag) {
        if (flag.context.empty()) return 0;
        if (flag.context.find('.') != std::string::npos) return 3;
        return flag.context.starts_with("player") ? 1 : 2;
    };
    std::stable_sort(flags.begin(), flags.end(),
                     [&](const Flag& a, const Flag& b) { return rank(a) < rank(b); });
    for (const auto& flag : flags) {
        if (flag.context.empty()) {
            auto global = apply_global(settings, lower_no_dashes(flag.name), flag.value);
            if (!global) return tl::unexpected(global.error());
            if (*global) continue;
        }
        for (std::size_t p = 0; p < 2; ++p) {
            for (std::size_t c = 0; c < 2; ++c) {
                const auto player = std::string{"player"} + static_cast<char>('1' + p);
                const std::string color = c == 0 ? "white" : "black";
                if (!flag.context.empty() && flag.context != player && flag.context != color &&
                    flag.context != player + "." + color)
                    continue;
                auto applied =
                    apply_context(settings.players.at(p).at(c), flag.name, flag.value);
                if (!applied) return tl::unexpected("--" + flag.name + ": " + applied.error());
            }
        }
    }
    return settings;
}

std::string tournament_status_line(const TournamentInfo& info) {
    const auto win = info.results[0][0] + info.results[0][1];
    const auto loss = info.results[2][0] + info.results[2][1];
    const auto draws = info.results[1][0] + info.results[1][1];
    float percentage = -1.0F;
    if (win + loss + draws > 0)
        percentage = (static_cast<float>(draws) / 2.0F + static_cast<float>(win)) /
                     static_cast<float>(win + loss + draws);
    std::ostringstream out;
    out << "tournamentstatus";
    if (info.finished) out << " final";
    out << " P1: +" << win << " -" << loss << " =" << draws;
    if (percentage > 0)
        out << " Win: " << std::fixed << std::setw(5) << std::setprecision(2)
            << percentage * 100.0F << "%";
    if (percentage < 1 && percentage > 0)
        out << " Elo: " << std::fixed << std::setw(5) << std::setprecision(2)
            << -400.0F * std::log(1.0F / percentage - 1.0F) / std::log(10.0F);
    if (win + loss > 0)
        out << " LOS: " << std::fixed << std::setw(5) << std::setprecision(2)
            << (0.5F + 0.5F * static_cast<float>(std::erf(
                                  (static_cast<double>(win) - static_cast<double>(loss)) /
                                  std::sqrt(2.0 * static_cast<double>(win + loss))))) *
                   100.0F
            << "%";
    out << " P1-W: +" << info.results[0][0] << " -" << info.results[2][0] << " ="
        << info.results[1][0];
    out << " P1-B: +" << info.results[0][1] << " -" << info.results[2][1] << " ="
        << info.results[1][1];
    out << " npm "
        << std::to_string(static_cast<double>(info.nodes_total) /
                          static_cast<double>(info.move_count));
    out << " nodes " << info.nodes_total;
    out << " moves " << info.move_count;
    return out.str();
}

namespace {
// Lc0 PositionHistory::ComputeGameResult.
GameResult compute_result(chess::Position& position, const chess::AttackTables& attacks) {
    switch (chess::game_status(position, attacks)) {
        case chess::GameStatus::Checkmate:
            return position.side_to_move() == chess::Color::White ? GameResult::BlackWon
                                                                  : GameResult::WhiteWon;
        case chess::GameStatus::Ongoing:
            return GameResult::Undecided;
        default:
            return GameResult::Draw;
    }
}
// Lc0 neural/backends/multiplexing: searches running at once share one batch on the device.
class MultiplexingBackend final : public nn::Backend {
   public:
    MultiplexingBackend(std::unique_ptr<nn::Backend> inner, std::size_t maximum_rows)
        : inner_(std::move(inner)), maximum_rows_(maximum_rows), thread_([this] { serve(); }) {}
    ~MultiplexingBackend() override {
        {
            const std::lock_guard lock(mutex_);
            shutdown_ = true;
        }
        changed_.notify_all();
        thread_.join();
    }
    [[nodiscard]] Result<void> evaluate(std::span<const nn::InputPlanes> inputs,
                                        std::span<nn::NNOutput> outputs) override {
        // A value game can ask for more rows than the device's largest batch: lc0 splits those too.
        for (std::size_t at = 0; at < inputs.size(); at += maximum_rows_) {
            const auto rows = std::min(maximum_rows_, inputs.size() - at);
            Request request{.inputs = inputs.subspan(at, rows), .outputs = outputs.subspan(at, rows)};
            std::unique_lock lock(mutex_);
            queue_.push_back(&request);
            changed_.notify_all();
            finished_.wait(lock, [&] { return request.done; });
            if (!request.result) return request.result;
        }
        return {};
    }
    [[nodiscard]] std::size_t physical_batch_size(std::size_t logical) const noexcept override {
        return inner_->physical_batch_size(logical);
    }
    [[nodiscard]] const nn::ModelMetadata& metadata() const noexcept override {
        return inner_->metadata();
    }
    [[nodiscard]] const nn::BackendInfo& info() const noexcept override { return inner_->info(); }

   private:
    struct Request {
        std::span<const nn::InputPlanes> inputs;
        std::span<nn::NNOutput> outputs;
        Result<void> result;
        bool done = false;
    };
    void serve() {
        std::vector<Request*> taken;
        std::vector<nn::InputPlanes> inputs;
        std::vector<nn::NNOutput> outputs;
        for (;;) {
            taken.clear();
            std::size_t rows = 0;
            {
                std::unique_lock lock(mutex_);
                changed_.wait(lock, [&] { return shutdown_ || !queue_.empty(); });
                if (shutdown_ && queue_.empty()) return;
                while (!queue_.empty() &&
                       (taken.empty() || rows + queue_.front()->inputs.size() <= maximum_rows_)) {
                    rows += queue_.front()->inputs.size();
                    taken.push_back(queue_.front());
                    queue_.pop_front();
                }
            }
            Result<void> result;
            if (taken.size() == 1) {
                result = inner_->evaluate(taken[0]->inputs, taken[0]->outputs);
            } else {
                inputs.clear();
                for (const auto* request : taken)
                    inputs.insert(inputs.end(), request->inputs.begin(), request->inputs.end());
                outputs.resize(rows);
                result = inner_->evaluate(inputs, outputs);
                if (result) {
                    std::size_t at = 0;
                    for (auto* request : taken) {
                        std::copy_n(outputs.begin() + static_cast<std::ptrdiff_t>(at),
                                    request->outputs.size(), request->outputs.begin());
                        at += request->outputs.size();
                    }
                }
            }
            const std::lock_guard lock(mutex_);
            for (auto* request : taken) {
                request->result = result;
                request->done = true;
            }
            finished_.notify_all();
        }
    }
    std::unique_ptr<nn::Backend> inner_;
    std::size_t maximum_rows_;
    std::mutex mutex_;
    std::condition_variable changed_, finished_;
    std::deque<Request*> queue_;
    bool shutdown_ = false;
    std::thread thread_;
};
struct SharedBackend {
    std::unique_ptr<nn::Backend> backend;
    std::unique_ptr<nn::LegalCache> cache;
};
class Tournament {
   public:
    Tournament(Settings settings, const BackendFactory& factory,
               const std::function<void(std::string_view)>& output)
        : settings_(std::move(settings)),
          factory_(factory),
          output_(output),
          random_(settings_.random_seed != 0 ? settings_.random_seed : std::random_device{}()) {
    }
    Result<TournamentInfo> run() {
        auto prepared = prepare();
        if (!prepared) return tl::unexpected(prepared.error());
        std::vector<std::thread> threads;
        threads.reserve(settings_.parallelism);
        for (std::size_t i = 0; i < settings_.parallelism; ++i)
            threads.emplace_back([this, i] { guarded_worker(i); });
        for (auto& thread : threads) thread.join();
        if (!failure_.empty()) return tl::unexpected(failure_);
        save_results();
        info_.finished = true;
        output_(tournament_status_line(info_));
        return info_;
    }

   private:
    Result<void> prepare() {
        const auto& first = settings_.players[0][0];
        multi_games_size_ = std::max(settings_.policy_mode_size, settings_.value_mode_size);
        if (settings_.policy_mode_size > 0 && settings_.value_mode_size > 0)
            return tl::unexpected(std::string{"Can't do both policy and value games at the same time."});
        if (!settings_.openings_file.empty()) {
            std::ifstream in(std::filesystem::path(std::u8string(
                                 settings_.openings_file.begin(), settings_.openings_file.end())),
                             std::ios::binary);
            if (!in) return tl::unexpected("Cannot open opening book " + settings_.openings_file);
            std::ostringstream text;
            text << in.rdbuf();
            auto parsed = parse_pgn(text.str(), attacks_);
            if (!parsed) return tl::unexpected(parsed.error());
            openings_ = std::move(*parsed);
            if (settings_.openings_mode == OpeningsMode::Shuffled)
                std::shuffle(openings_.begin(), openings_.end(), random_);
        }
        if (multi_games_size_ > 0 && openings_.empty())
            return tl::unexpected(std::string{
                "Policy/Value games are deterministic, needs opening book to be useful."});
        if (multi_games_size_ > 0 &&
            (settings_.games == -1 ||
             (settings_.games > 0 &&
              static_cast<std::size_t>(settings_.games) > openings_.size() * 2)))
            return tl::unexpected(std::string{
                "Policy/Value games are deterministic, you do not want to go through the opening "
                "book more than once."});
        if (multi_games_size_ > 0 && !settings_.mirror_openings)
            return tl::unexpected(
                std::string{"Policy/Value multi games mode only supports mirrored openings."});
        if (multi_games_size_ > 0 && settings_.games > 0 && settings_.games % 2 == 1)
            return tl::unexpected(std::string{
                "Policy/Value multi games can't support mirrored with an odd number of games."});
        // If playing just one game, the player1 is white, otherwise randomize.
        if (settings_.games != 1) first_game_black_ = std::bernoulli_distribution(0.5)(random_);
        for (std::size_t p = 0; p < 2; ++p) {
            for (std::size_t c = 0; c < 2; ++c) {
                const auto& player = settings_.players.at(p).at(c);
                const bool has_limit =
                    player.playouts != -1 || player.visits != -1 || player.movetime_ms != -1;
                if (multi_games_size_ == 0 && !has_limit)
                    return tl::unexpected(std::string{
                        "Please define --visits, --playouts or --movetime, otherwise it's not "
                        "clear when to stop search."});
                if (multi_games_size_ > 0 && has_limit)
                    return tl::unexpected(std::string{
                        "Policy mode does not need --visits, --playouts or --movetime as it has "
                        "fixed 1 node behaviour."});
                if (player.opening_stop_probability !=
                        settings_.players.at(p).at(1 - c).opening_stop_probability &&
                    player.opening_stop_probability != 0.0 &&
                    settings_.players.at(p).at(1 - c).opening_stop_probability != 0.0)
                    return tl::unexpected(
                        std::string{"Stop probabilities must be both equal or zero!"});
            }
        }
        if (!first.options.syzygy_path.empty()) {
            auto opened = search::Tablebase::open(first.options.syzygy_path);
            if (!opened) return tl::unexpected(opened.error());
            tablebase_ = std::move(*opened);
        }
        if (settings_.training) {
            if (settings_.training_directory.empty()) {
                static constexpr std::string_view kLetters = "abcdefghijklmnopqrstuvwxyz";
                std::string name = "data-";
                std::uniform_int_distribution<std::size_t> pick(0, kLetters.size() - 1);
                for (int i = 0; i < 12; ++i) name.push_back(kLetters[pick(random_)]);
                settings_.training_directory = first.options.backend.cache_directory / name;
            }
            std::error_code error;
            std::filesystem::create_directories(settings_.training_directory, error);
            if (error)
                return tl::unexpected("Cannot create training directory " +
                                      settings_.training_directory.generic_string() + ": " +
                                      error.message());
        }
        return {};
    }
    // Lc0 Backend::IsSameConfiguration: the same net on the same device shares everything.
    Result<SharedBackend*> backend_for(const uci::Options& options) {
        const auto& config = options.backend;
        std::string key = config.model_path.generic_string() + "|" +
                          std::to_string(static_cast<int>(config.kind)) + "|" +
                          std::to_string(static_cast<int>(config.precision)) + "|" +
                          std::to_string(config.device_id);
        const std::lock_guard lock(backends_mutex_);
        auto found = backends_.find(key);
        if (found != backends_.end()) return &found->second;
        if (config.model_path.empty())
            return tl::unexpected(std::string{"WeightsFile is required for selfplay"});
        auto resolved = config;
        auto paths = nn::resolve_backend_cli_paths(resolved);
        if (!paths) return tl::unexpected(paths.error());
        auto made = factory_(resolved);
        if (!made) return tl::unexpected(made.error());
        SharedBackend shared;
        const auto rows = resolved.batch_buckets.empty() ? options.search.maximum_batch
                                                         : resolved.batch_buckets.back();
        shared.backend = std::make_unique<MultiplexingBackend>(std::move(*made), rows);
        if (options.nncache_entries != 0) {
            auto cache = nn::LegalCache::create(options.nncache_entries,
                                                shared.backend->info().cache_namespace,
                                                options.cache_history_length);
            if (!cache) return tl::unexpected(cache.error());
            shared.cache = std::move(*cache);
        }
        return &backends_.emplace(std::move(key), std::move(shared)).first->second;
    }
    struct WorkerEngine {
        std::string key;
        std::unique_ptr<search::SearchEngine> engine;
        SharedBackend* backend = nullptr;
    };
    // Each worker keeps its engines between games, so their memory is touched only once.
    Result<WorkerEngine*> engine_for(std::list<WorkerEngine>& engines, const PlayerSettings& player,
                                     std::size_t role) {
        std::string key = std::to_string(role);
        for (const auto& line : uci::option_lines(player.options)) key += line;
        for (auto& engine : engines)
            if (engine.key == key) return &engine;
        auto shared = backend_for(player.options);
        if (!shared) return tl::unexpected(shared.error());
        auto config = player.options.search;
        if (player.options.task_workers >= 0)
            config.threads = static_cast<std::size_t>(player.options.task_workers);
        config.cache_history_length =
            static_cast<std::uint32_t>(player.options.cache_history_length);
        config.tablebase = tablebase_;
        // The memory limit is for the whole tournament; every game may hold two trees.
        config.ram_limit_bytes =
            player.options.search.ram_limit_bytes / (2 * settings_.parallelism);
        {
            const std::lock_guard lock(mutex_);
            config.random_seed = settings_.random_seed != 0 ? random_() | 1U : 0;
        }
        auto made =
            search::SearchEngine::create(config, attacks_, *(*shared)->backend, (*shared)->cache.get());
        if (!made) return tl::unexpected(made.error());
        engines.push_back({.key = std::move(key), .engine = std::move(*made), .backend = *shared});
        return &engines.back();
    }
    void guarded_worker(std::size_t) {
        try {
            worker();
        } catch (const std::exception& error) {
            fail(error.what());
        }
    }
    void fail(std::string_view text) {
        const std::lock_guard lock(mutex_);
        if (failure_.empty()) failure_ = std::string{text};
        abort_ = true;
    }
    void worker() {
        std::list<WorkerEngine> engines;
        for (;;) {
            std::int64_t game_id = 0, count = 0;
            {
                const std::lock_guard lock(mutex_);
                if (abort_) return;
                if (multi_games_size_ != 0) {
                    const auto maximum = static_cast<std::int64_t>(2 * multi_games_size_);
                    auto take = maximum;
                    if (settings_.games != -1) {
                        const auto cap = settings_.games == -2
                                             ? static_cast<std::int64_t>(openings_.size() * 2)
                                             : settings_.games;
                        take = std::min(maximum, cap - games_count_);
                    }
                    if (take <= 0) return;
                    game_id = games_count_;
                    count = take;
                    games_count_ += take;
                } else {
                    if ((settings_.games >= 0 && games_count_ >= settings_.games) ||
                        (settings_.games == -2 && !openings_.empty() &&
                         games_count_ >= static_cast<std::int64_t>(openings_.size()) *
                                             (settings_.mirror_openings ? 2 : 1)))
                        return;
                    game_id = games_count_++;
                }
            }
            auto played = multi_games_size_ != 0 ? play_multi_games(game_id, count)
                                                 : play_one_game(engines, game_id);
            if (!played) {
                fail(played.error());
                return;
            }
        }
    }
    struct Side {
        const PlayerSettings* settings = nullptr;
        WorkerEngine* engine = nullptr;
        std::size_t player = 0;
    };
    float random_float(float maximum) {
        const std::lock_guard lock(mutex_);
        return std::uniform_real_distribution<float>(0.0F, maximum)(random_);
    }
    Result<void> play_one_game(std::list<WorkerEngine>& engines, std::int64_t game_id) {
        bool player1_black = false;
        Opening opening;
        {
            const std::lock_guard lock(mutex_);
            player1_black = (game_id % 2 == 1) != first_game_black_;
            if (!openings_.empty()) {
                const auto size = static_cast<std::int64_t>(openings_.size());
                if (settings_.mirror_openings)
                    opening = openings_[static_cast<std::size_t>((game_id / 2) % size)];
                else if (settings_.openings_mode == OpeningsMode::Random)
                    opening = openings_[std::uniform_int_distribution<std::size_t>(
                        0, openings_.size() - 1)(random_)];
                else
                    opening = openings_[static_cast<std::size_t>(game_id % size)];
            }
            if (!discard_pile_.empty() &&
                std::uniform_real_distribution<float>(0.0F, 100.0F)(random_) <
                    settings_.discarded_start_chance) {
                const auto at = std::uniform_int_distribution<std::size_t>(
                    0, discard_pile_.size() - 1)(random_);
                std::swap(discard_pile_[at], discard_pile_.back());
                opening = std::move(discard_pile_.back());
                discard_pile_.pop_back();
            }
        }
        // Index 0 plays white.
        std::array<Side, 2> sides;
        for (std::size_t color = 0; color < 2; ++color) {
            const std::size_t player = (color == 0) == player1_black ? 1 : 0;
            sides.at(color).settings = &settings_.players.at(player).at(color);
            sides.at(color).player = player;
        }
        // One tree for both sides only if they search alike, since a Mouse engine holds one
        // configuration. It matters only when a tree survives between moves.
        const bool same = uci::option_lines(sides[0].settings->options) ==
                          uci::option_lines(sides[1].settings->options);
        const bool shared_tree =
            same && (settings_.share_trees ||
                     (!sides[0].settings->reuse_tree && !sides[1].settings->reuse_tree));
        for (std::size_t color = 0; color < 2; ++color) {
            auto engine = engine_for(engines, *sides.at(color).settings, shared_tree ? 0 : color);
            if (!engine) return tl::unexpected(engine.error());
            sides.at(color).engine = *engine;
        }
        auto parsed = chess::Position::from_fen(opening.start_fen, attacks_);
        if (!parsed) return tl::unexpected("Invalid opening FEN: " + parsed.error());
        auto position = std::move(*parsed);
        nn::EncodingHistory history;
        history.reset(position);
        std::vector<chess::Move> played_moves;
        const auto initial_ply = position.fullmove_number();
        // Lc0 SelfPlayGame::SelfPlayGame: maybe leave the opening early.
        const auto white_probability =
            static_cast<float>(sides[0].settings->opening_stop_probability);
        const auto black_probability =
            static_cast<float>(sides[1].settings->opening_stop_probability);
        std::size_t start_ply = 0;
        for (const auto move : opening.moves) {
            const bool black = position.side_to_move() == chess::Color::Black;
            const auto now = black ? black_probability : white_probability;
            const auto next = black ? white_probability : black_probability;
            const auto positions = static_cast<float>(opening.moves.size() - start_ply + 1);
            if (now > 0.0F &&
                random_float(1.0F) <
                    std::max(now, now / (now * std::floor((positions + 1.0F) / 2.0F) +
                                         next * std::floor(positions / 2.0F))))
                break;
            static_cast<void>(position.make_move(move, attacks_));
            auto appended = history.append(position);
            if (!appended) return appended;
            played_moves.push_back(move);
            ++start_ply;
        }
        const bool enable_resign =
            random_float(100.0F) >= static_cast<float>(settings_.resign_playthrough);
        for (auto& side : sides) {
            auto set = side.engine->engine->set_position(position, history, false);
            if (!set) return set;
        }
        TrainingGame training;
        GameResult result = GameResult::Undecided;
        bool adjudicated = false;
        std::uint64_t move_count = 0, nodes_total = 0;
        std::array<float, 2> min_eval{1.0F, 1.0F};
        std::array<float, 3> max_eval{0.0F, 0.0F, 0.0F};
        bool stop = false;
        while (!stop) {
            {
                const std::lock_guard lock(mutex_);
                if (abort_) return {};
            }
            result = compute_result(position, attacks_);
            if (result != GameResult::Undecided) break;
            const auto game_ply = initial_ply + played_moves.size();
            if (game_ply >= 450) {
                adjudicated = true;
                break;
            }
            const bool black = position.side_to_move() == chess::Color::Black;
            const auto color_index = black ? std::size_t{1} : std::size_t{0};
            const auto& side = sides.at(color_index);
            const auto& player = *side.settings;
            auto& engine = *side.engine->engine;
            // Lc0 TrimTreeAtHead: with no reuse the tree starts over at this position.
            if (!player.reuse_tree) {
                auto set = engine.set_position(position, history, false);
                if (!set) return set;
            }
            const auto inherited = engine.snapshot().nodes;
            search::SearchLimits limits;
            limits.nodes = std::nullopt;
            if (player.visits >= 0) limits.nodes = static_cast<std::uint64_t>(player.visits);
            if (player.playouts >= 0) {
                const auto limit = inherited + static_cast<std::uint64_t>(player.playouts);
                limits.nodes = limits.nodes ? std::min(*limits.nodes, limit) : limit;
            }
            if (limits.nodes && *limits.nodes == 0) limits.nodes = 1;
            if (player.movetime_ms >= 0)
                limits.movetime = std::chrono::milliseconds{std::max<std::int64_t>(1, player.movetime_ms)};
            limits.smart_pruning = player.options.smart_pruning_factor > 0.0;
            limits.smart_pruning_factor = player.options.smart_pruning_factor;
            limits.minimum_smart_pruning_batches = player.options.minimum_smart_pruning_batches;
            limits.contempt_side =
                black ? search::ContemptSide::Black : search::ContemptSide::White;
            auto searched = engine.run(limits);
            if (!searched) return tl::unexpected(searched.error());
            ++move_count;
            nodes_total += searched->info.new_playouts;
            const auto roots = engine.all_root_moves();
            if (!searched->best_move) return tl::unexpected(std::string{"Search found no move"});
            const auto best_move = *searched->best_move;
            const auto root_q = static_cast<float>(searched->info.q);
            const auto root_d = static_cast<float>(searched->info.draw);
            const auto root_m = static_cast<float>(searched->info.moves_left.value_or(0.0));
            const auto eval_of = [&](chess::Move move) {
                Eval out{.q = root_q, .d = root_d, .m = root_m};
                for (std::size_t i = 0; i < roots.size; ++i) {
                    const auto& stat = roots.moves.at(i);
                    if (stat.move != move) continue;
                    if (stat.visits != 0) {
                        out.q = static_cast<float>(stat.q);
                        out.d = static_cast<float>(stat.draw);
                        if (stat.moves_left) out.m = static_cast<float>(*stat.moves_left) + 1.0F;
                    }
                    break;
                }
                return out;
            };
            const auto best_eval = eval_of(best_move);
            auto eval = (best_eval.q + 1.0F) / 2.0F;
            min_eval.at(color_index) = std::min(min_eval.at(color_index), eval);
            const auto best_w = (best_eval.q + 1.0F - best_eval.d) / 2.0F;
            const auto best_l = best_w - best_eval.q;
            max_eval[0] = std::max(max_eval[0], black ? best_l : best_w);
            max_eval[1] = std::max(max_eval[1], best_eval.d);
            max_eval[2] = std::max(max_eval[2], black ? best_w : best_l);
            const auto move_number = (played_moves.size() + 1) / 2 + 1;
            if (enable_resign && move_number >= player.resign_earliest_move) {
                const auto resign = static_cast<float>(player.resign_percentage) / 100.0F;
                if (player.resign_wdl_style) {
                    const auto threshold = 1.0F - resign;
                    if (best_w > threshold) {
                        result = black ? GameResult::BlackWon : GameResult::WhiteWon;
                        adjudicated = true;
                    } else if (best_l > threshold) {
                        result = black ? GameResult::WhiteWon : GameResult::BlackWon;
                        adjudicated = true;
                    } else if (best_eval.d > threshold) {
                        result = GameResult::Draw;
                        adjudicated = true;
                    }
                } else if (eval < resign) {
                    result = black ? GameResult::WhiteWon : GameResult::BlackWon;
                    adjudicated = true;
                }
                if (adjudicated) break;
            }
            // Lc0 SelfPlayGame::Play: a pick with too few visits goes back and is drawn again.
            chess::Move move = best_move;
            Eval played_eval = best_eval;
            std::uint64_t max_n = 0;
            for (std::size_t i = 0; i < roots.size; ++i)
                max_n = std::max(max_n, roots.moves.at(i).visits);
            for (;;) {
                const auto choice = engine.final_move(game_ply);
                if (!choice) return tl::unexpected(std::string{"Search found no move"});
                move = *choice;
                std::uint64_t cur_n = 0;
                for (std::size_t i = 0; i < roots.size; ++i)
                    if (roots.moves.at(i).move == move) cur_n = roots.moves.at(i).visits;
                played_eval = eval_of(move);
                if (cur_n == max_n || cur_n >= player.minimum_allowed_visits) break;
                auto discarded = position;
                static_cast<void>(discarded.make_move(move, attacks_));
                if (compute_result(discarded, attacks_) == GameResult::Undecided &&
                    settings_.discarded_start_chance != 0.0) {
                    Opening entry{.start_fen = opening.start_fen, .moves = played_moves};
                    entry.moves.push_back(move);
                    const std::lock_guard lock(mutex_);
                    discard_pile_.push_back(std::move(entry));
                    // 10k seems it should be enough to keep a good mix and avoid running out of ram.
                    if (discard_pile_.size() > 10000) {
                        const auto at = std::uniform_int_distribution<std::size_t>(
                            0, discard_pile_.size() - 1)(random_);
                        std::swap(discard_pile_[at], discard_pile_.back());
                        discard_pile_.pop_back();
                    }
                }
            }
            if (player.move_thinking) {
                std::ostringstream line;
                line << "bestmove " << chess::move_to_uci(move, player.options.chess960)
                     << " player " << side.player + 1 << " gameid " << game_id << " side "
                     << (black ? "black" : "white");
                emit(line.str());
            }
            if (player.verbose_thinking) {
                std::ostringstream line;
                line << "info player " << side.player + 1 << " gameid " << game_id << " side "
                     << (black ? "black" : "white") << " nodes " << searched->info.nodes
                     << " score q " << searched->info.q;
                emit(line.str());
            }
            if (settings_.training) {
                auto added = add_training(training, position, history, *side.engine, player,
                                          roots, searched->info, best_eval, played_eval, best_move,
                                          move);
                if (!added) return added;
            }
            // Add the move to every tree.
            const auto applied_to = [&](WorkerEngine* engine_entry) -> Result<void> {
                auto advanced = engine_entry->engine->advance(move);
                if (!advanced) return tl::unexpected(advanced.error());
                return {};
            };
            static_cast<void>(position.make_move(move, attacks_));
            auto appended = history.append(position);
            if (!appended) return appended;
            played_moves.push_back(move);
            auto advanced = applied_to(sides[0].engine);
            if (!advanced) return advanced;
            if (sides[1].engine != sides[0].engine) {
                advanced = applied_to(sides[1].engine);
                if (!advanced) return advanced;
            }
        }
        // Lc0 PlayOneGame reports only a decided game, so one that reached 450 plies vanishes.
        if (result == GameResult::Undecided) return {};
        std::string training_file;
        if (settings_.training && start_ply < played_moves.size() && training.size() != 0) {
            std::ostringstream name;
            name << "game_" << std::setfill('0') << std::setw(6) << game_id << ".gz";
            const auto path = settings_.training_directory / name.str();
            auto written = write_training_file(path, training.finish(result, adjudicated));
            if (!written) return written;
            training_file = path.generic_string();
        }
        if (!enable_resign) {
            float worst = 0.0F;
            if (sides[0].settings->resign_wdl_style)
                worst = result == GameResult::WhiteWon   ? std::max(max_eval[1], max_eval[2])
                        : result == GameResult::BlackWon ? std::max(max_eval[1], max_eval[0])
                                                         : std::max(max_eval[2], max_eval[0]);
            else
                worst = result == GameResult::WhiteWon   ? min_eval[0]
                        : result == GameResult::BlackWon ? min_eval[1]
                                                         : std::min(min_eval[0], min_eval[1]);
            emit("resign_report fp_threshold " + std::to_string(worst));
        }
        emit_game(training_file, game_id, start_ply, player1_black, result, played_moves,
                  opening.start_fen);
        {
            const std::lock_guard lock(mutex_);
            int slot = result == GameResult::Draw ? 1 : result == GameResult::WhiteWon ? 0 : 2;
            if (player1_black) slot = 2 - slot;
            ++info_.results.at(static_cast<std::size_t>(slot)).at(player1_black ? 1 : 0);
            info_.move_count += move_count;
            info_.nodes_total += nodes_total;
            emit(tournament_status_line(info_));
        }
        return {};
    }
    Result<void> add_training(TrainingGame& training, const chess::Position& position,
                              const nn::EncodingHistory& history, const WorkerEngine& engine,
                              const PlayerSettings& player, const search::RootStatistics& roots,
                              const search::SearchInfo& info, Eval best_eval, Eval played_eval,
                              chess::Move best_move, chess::Move played_move) {
        auto planes = nn::encode_classical(history.frames(), 1, player.options.search.history_fill);
        if (!planes) return tl::unexpected(planes.error());
        // Lc0 checks the best move against every other edge: a bound above it means it isn't proven.
        bool proven = false;
        for (std::size_t i = 0; i < roots.size; ++i) {
            const auto& stat = roots.moves.at(i);
            if (stat.move != best_move) continue;
            proven = stat.visits != 0 && stat.child_proof.outcome != search::ProofOutcome::Unknown;
        }
        if (proven && best_eval.q < 1.0F) {
            for (std::size_t i = 0; i < roots.size; ++i) {
                const auto& stat = roots.moves.at(i);
                if (stat.child_proof.outcome == search::ProofOutcome::Unknown ||
                    stat.child_proof.outcome == search::ProofOutcome::Loss)
                    proven = false;
                if (best_eval.q < 0.0F && stat.child_proof.outcome == search::ProofOutcome::Draw)
                    proven = false;
            }
        }
        std::optional<nn::LegalOutput> evaluation;
        const nn::LegalOutput* cached = nullptr;
        if (engine.backend->cache) {
            nn::PolicyIndices policy;
            policy.size = roots.size;
            for (std::size_t i = 0; i < roots.size; ++i) {
                const auto index =
                    nn::PolicyMap::index(roots.moves.at(i).move, position.side_to_move());
                if (!index) return tl::unexpected(index.error());
                policy.values.at(i) = *index;
            }
            auto found = engine.backend->cache->lookup(*planes, policy);
            if (!found) return tl::unexpected(found.error());
            evaluation = *found;
            if (evaluation) cached = &*evaluation;
        }
        const Eval root{.q = static_cast<float>(info.q),
                        .d = static_cast<float>(info.draw),
                        .m = static_cast<float>(info.moves_left.value_or(0.0))};
        return training.add({.position = &position,
                             .planes = &*planes,
                             .roots = &roots,
                             .root_visits = info.nodes,
                             .root = root,
                             .best = best_eval,
                             .played = played_eval,
                             .best_is_proven = proven,
                             .best_move = best_move,
                             .played_move = played_move,
                             .evaluation = cached,
                             .policy_temperature = player.options.search.selection.policy_temperature});
    }
    void emit(std::string_view line) {
        const std::lock_guard lock(output_mutex_);
        output_(line);
    }
    void emit_game(const std::string& training_file, std::int64_t game_id, std::size_t start_ply,
                   bool player1_black, GameResult result, std::span<const chess::Move> moves,
                   std::string_view start_fen) {
        std::string line = "gameready";
        if (!training_file.empty()) line += " trainingfile " + training_file;
        line += " gameid " + std::to_string(game_id);
        line += " play_start_ply " + std::to_string(start_ply);
        line += " player1 " + std::string(player1_black ? "black" : "white");
        if (result != GameResult::Undecided)
            line += std::string(" result ") + (result == GameResult::Draw       ? "draw"
                                               : result == GameResult::WhiteWon ? "whitewon"
                                                                                : "blackwon");
        if (!moves.empty()) {
            // Lc0 SelfPlayLoop::SendGameInfo always writes castling as king takes rook.
            line += " moves";
            for (const auto move : moves) line += " " + chess::move_to_uci(move, true);
        }
        if (start_fen != chess::kStartFen) line += " from_fen " + std::string{start_fen};
        emit(line);
    }
    // Lc0 MultiSelfPlayGames: many games at once with no search, one network batch per colour.
    struct MultiGame {
        std::optional<chess::Position> position;
        nn::EncodingHistory history;
        GameResult result = GameResult::Undecided;
        std::vector<chess::Move> moves;
        std::size_t start_ply = 0;
    };
    Result<void> play_multi_round(std::span<MultiGame> games, std::array<nn::Backend*, 2> backends,
                                  std::array<nn::HistoryFill, 2> fills) {
        const bool use_value = settings_.value_mode_size > 0;
        for (;;) {
            {
                const std::lock_guard lock(mutex_);
                if (abort_) return {};
            }
            bool all_done = true, blacks_move = false;
            for (auto& game : games) {
                if (game.result != GameResult::Undecided) continue;
                game.result = compute_result(*game.position, attacks_);
                if (game.result != GameResult::Undecided) continue;
                if (tablebase_ && tablebase_->can_probe(*game.position)) {
                    if (const auto wdl = tablebase_->probe_wdl(*game.position, attacks_)) {
                        const bool black = game.position->side_to_move() == chess::Color::Black;
                        game.result =
                            *wdl == search::TablebaseWdl::Win
                                ? (black ? GameResult::BlackWon : GameResult::WhiteWon)
                            : *wdl == search::TablebaseWdl::Loss
                                ? (black ? GameResult::WhiteWon : GameResult::BlackWon)
                                : GameResult::Draw;
                        continue;
                    }
                }
                if (all_done) {
                    all_done = false;
                    blacks_move = game.position->side_to_move() == chess::Color::Black;
                }
            }
            if (all_done) return {};
            const auto color = blacks_move ? std::size_t{1} : std::size_t{0};
            std::vector<nn::InputPlanes> inputs;
            struct Row {
                MultiGame* game = nullptr;
                chess::MoveList legal;
                // Rows of network input per move, or a known value in `values`.
                std::vector<std::size_t> rows;
                std::vector<float> values;
            };
            std::vector<Row> pending;
            for (auto& game : games) {
                if (game.result != GameResult::Undecided ||
                    (game.position->side_to_move() == chess::Color::Black) != blacks_move)
                    continue;
                Row row;
                row.game = &game;
                row.legal = chess::legal_moves(*game.position, attacks_);
                if (!use_value) {
                    auto encoded = nn::encode_classical(game.history.frames(), 1, fills.at(color));
                    if (!encoded) return tl::unexpected(encoded.error());
                    row.rows.push_back(inputs.size());
                    inputs.push_back(*encoded);
                } else {
                    for (const auto move : row.legal) {
                        auto next = *game.position;
                        static_cast<void>(next.make_move(move, attacks_));
                        const auto outcome = compute_result(next, attacks_);
                        if (outcome != GameResult::Undecided) {
                            row.values.push_back(outcome == GameResult::Draw ? 0.0F : 1.0F);
                            row.rows.push_back(std::numeric_limits<std::size_t>::max());
                            continue;
                        }
                        auto after = game.history;
                        auto appended = after.append(next);
                        if (!appended) return appended;
                        auto encoded = nn::encode_classical(after.frames(), 1, fills.at(color));
                        if (!encoded) return tl::unexpected(encoded.error());
                        row.values.push_back(0.0F);
                        row.rows.push_back(inputs.size());
                        inputs.push_back(*encoded);
                    }
                }
                pending.push_back(std::move(row));
            }
            std::vector<nn::NNOutput> outputs(inputs.size());
            auto evaluated = backends.at(color)->evaluate(inputs, outputs);
            if (!evaluated) return evaluated;
            for (auto& row : pending) {
                std::size_t best = 0;
                float best_score = -std::numeric_limits<float>::infinity();
                const auto side = row.game->position->side_to_move();
                for (std::size_t i = 0; i < row.legal.size(); ++i) {
                    float score = 0.0F;
                    if (!use_value) {
                        const auto index = nn::PolicyMap::index(row.legal[i], side);
                        if (!index) return tl::unexpected(index.error());
                        score = outputs.at(row.rows[0]).policy_logits.at(*index);
                    } else if (row.rows.at(i) == std::numeric_limits<std::size_t>::max()) {
                        score = row.values.at(i);
                    } else {
                        // The opponent is to move in the position the network sees.
                        const auto& wdl = outputs.at(row.rows.at(i)).wdl;
                        score = (wdl[2] - wdl[0]) / (wdl[0] + wdl[1] + wdl[2]);
                    }
                    if (score > best_score) {
                        best_score = score;
                        best = i;
                    }
                }
                const auto move = row.legal[best];
                static_cast<void>(row.game->position->make_move(move, attacks_));
                auto appended = row.game->history.append(*row.game->position);
                if (!appended) return appended;
                row.game->moves.push_back(move);
            }
        }
    }
    Result<void> play_multi_games(std::int64_t game_id, std::int64_t game_count) {
        std::vector<Opening> openings;
        {
            const std::lock_guard lock(mutex_);
            const auto basis = static_cast<std::size_t>(game_id / 2);
            for (std::size_t i = 0; i < static_cast<std::size_t>(game_count / 2); ++i)
                openings.push_back(openings_[(basis + i) % openings_.size()]);
        }
        const auto prepare_games = [&]() -> Result<std::vector<MultiGame>> {
            std::vector<MultiGame> games(openings.size());
            for (std::size_t i = 0; i < openings.size(); ++i) {
                auto parsed = chess::Position::from_fen(openings[i].start_fen, attacks_);
                if (!parsed) return tl::unexpected("Invalid opening FEN: " + parsed.error());
                games[i].position = std::move(*parsed);
                games[i].history.reset(*games[i].position);
                for (const auto move : openings[i].moves) {
                    static_cast<void>(games[i].position->make_move(move, attacks_));
                    auto appended = games[i].history.append(*games[i].position);
                    if (!appended) return tl::unexpected(appended.error());
                    games[i].moves.push_back(move);
                }
                games[i].start_ply = openings[i].moves.size();
            }
            return games;
        };
        const auto backend_of = [&](std::size_t player, std::size_t color) -> Result<nn::Backend*> {
            auto shared = backend_for(settings_.players.at(player).at(color).options);
            if (!shared) return tl::unexpected(shared.error());
            return (*shared)->backend.get();
        };
        const auto fill_of = [&](std::size_t player, std::size_t color) {
            return settings_.players.at(player).at(color).options.search.history_fill;
        };
        auto p1_white = backend_of(0, 0), p2_black = backend_of(1, 1);
        auto p1_black = backend_of(0, 1), p2_white = backend_of(1, 0);
        if (!p1_white) return tl::unexpected(p1_white.error());
        if (!p2_black) return tl::unexpected(p2_black.error());
        if (!p1_black) return tl::unexpected(p1_black.error());
        if (!p2_white) return tl::unexpected(p2_white.error());
        auto first = prepare_games();
        if (!first) return tl::unexpected(first.error());
        auto second = prepare_games();
        if (!second) return tl::unexpected(second.error());
        auto played = play_multi_round(*first, {*p1_white, *p2_black}, {fill_of(0, 0), fill_of(1, 1)});
        if (!played) return played;
        played = play_multi_round(*second, {*p2_white, *p1_black}, {fill_of(1, 0), fill_of(0, 1)});
        if (!played) return played;
        for (std::size_t i = 0; i < openings.size(); ++i) {
            for (const bool black : {false, true}) {
                const auto& game = (black ? *second : *first)[i];
                if (game.result == GameResult::Undecided) continue;
                emit_game({}, game_id + 2 * static_cast<std::int64_t>(i) + (black ? 1 : 0),
                          game.start_ply, black, game.result, game.moves, openings[i].start_fen);
                const std::lock_guard lock(mutex_);
                int slot = game.result == GameResult::Draw ? 1
                           : game.result == GameResult::WhiteWon ? 0 : 2;
                if (black) slot = 2 - slot;
                ++info_.results.at(static_cast<std::size_t>(slot)).at(black ? 1 : 0);
                emit(tournament_status_line(info_));
            }
        }
        return {};
    }
    // Lc0 SelfPlayTournament::SaveResults: appended in a fake pgn format.
    void save_results() {
        if (settings_.tournament_results_file.empty()) return;
        std::ofstream out(std::filesystem::path(std::u8string(
                              settings_.tournament_results_file.begin(),
                              settings_.tournament_results_file.end())),
                          std::ios::app);
        const auto name = [&](std::size_t player) {
            return settings_.players.at(player)[0].options.backend.model_path.generic_string();
        };
        out << "\n[White \"" << name(0) << "\"]\n[Black \"" << name(1) << "\"]\n[Results \""
            << info_.results[0][0] << " " << info_.results[2][0] << " " << info_.results[1][0]
            << "\"]\n\n[White \"" << name(1) << "\"]\n[Black \"" << name(0) << "\"]\n[Results \""
            << info_.results[2][1] << " " << info_.results[0][1] << " " << info_.results[1][1]
            << "\"]\n";
    }
    Settings settings_;
    const BackendFactory& factory_;
    const std::function<void(std::string_view)>& output_;
    chess::AttackTables attacks_;
    std::mutex mutex_, output_mutex_, backends_mutex_;
    std::mt19937_64 random_;
    std::map<std::string, SharedBackend> backends_;
    std::vector<Opening> openings_;
    std::vector<Opening> discard_pile_;
    std::shared_ptr<search::Tablebase> tablebase_;
    std::size_t multi_games_size_ = 0;
    bool first_game_black_ = false, abort_ = false;
    std::int64_t games_count_ = 0;
    TournamentInfo info_;
    std::string failure_;
};
Result<chess::Move> san_to_move(chess::Position& position, std::string_view token,
                                const chess::AttackTables& attacks) {
    std::string san{token};
    while (!san.empty() && std::string_view{"+#!?"}.find(san.back()) != std::string_view::npos)
        san.pop_back();
    const auto legal = chess::legal_moves(position, attacks);
    const auto side = position.side_to_move();
    if (san == "O-O" || san == "O-O-O" || san == "0-0" || san == "0-0-0") {
        const bool kingside = san == "O-O" || san == "0-0";
        for (const auto move : legal)
            if (move.kind() == chess::MoveKind::Castle &&
                (chess::file_of(move.to()) > chess::file_of(move.from())) == kingside)
                return move;
        return tl::unexpected("Illegal castling in opening: " + san);
    }
    auto promotion = chess::PieceType::None;
    if (const auto equals = san.find('='); equals != std::string::npos && equals + 1 < san.size()) {
        switch (san[equals + 1]) {
            case 'Q': promotion = chess::PieceType::Queen; break;
            case 'R': promotion = chess::PieceType::Rook; break;
            case 'B': promotion = chess::PieceType::Bishop; break;
            case 'N': promotion = chess::PieceType::Knight; break;
            default: return tl::unexpected("Unknown promotion in opening: " + san);
        }
        san.erase(equals);
    }
    auto type = chess::PieceType::Pawn;
    if (!san.empty() && std::string_view{"NBRQK"}.find(san.front()) != std::string_view::npos) {
        type = san.front() == 'N'   ? chess::PieceType::Knight
               : san.front() == 'B' ? chess::PieceType::Bishop
               : san.front() == 'R' ? chess::PieceType::Rook
               : san.front() == 'Q' ? chess::PieceType::Queen
                                    : chess::PieceType::King;
        san.erase(0, 1);
    }
    std::erase(san, 'x');
    if (san.size() < 2 || san[san.size() - 2] < 'a' || san[san.size() - 2] > 'h' ||
        san.back() < '1' || san.back() > '8')
        return tl::unexpected("Unreadable move in opening: " + std::string{token});
    const auto to = chess::square(san[san.size() - 2] - 'a', san.back() - '1');
    std::optional<int> from_file, from_rank;
    for (const char c : std::string_view{san}.substr(0, san.size() - 2)) {
        if (c >= 'a' && c <= 'h') from_file = c - 'a';
        if (c >= '1' && c <= '8') from_rank = c - '1';
    }
    std::optional<chess::Move> found;
    for (const auto move : legal) {
        if (move.to() != to || move.promotion() != promotion ||
            move.kind() == chess::MoveKind::Castle ||
            (position.pieces(side, type) & chess::bit(move.from())) == 0 ||
            (from_file && chess::file_of(move.from()) != *from_file) ||
            (from_rank && chess::rank_of(move.from()) != *from_rank))
            continue;
        if (found) return tl::unexpected("Ambiguous move in opening: " + std::string{token});
        found = move;
    }
    if (!found) return tl::unexpected("Illegal move in opening: " + std::string{token});
    return *found;
}
}  // namespace

Result<std::vector<Opening>> parse_pgn(std::string_view text, const chess::AttackTables& attacks) {
    std::vector<Opening> result;
    std::optional<chess::Position> position;
    Opening current;
    const auto begin_game = [&]() -> Result<void> {
        if (position) return {};
        auto parsed = chess::Position::from_fen(current.start_fen, attacks);
        if (!parsed) return tl::unexpected("Invalid FEN in opening book: " + parsed.error());
        position = std::move(*parsed);
        return {};
    };
    const auto end_game = [&] {
        if (position) result.push_back(current);
        current = {};
        position.reset();
    };
    std::size_t at = 0;
    while (at < text.size()) {
        const char c = text[at];
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            ++at;
        } else if (c == '{') {
            at = text.find('}', at);
            if (at == std::string_view::npos) break;
            ++at;
        } else if (c == ';' || c == '%') {
            at = text.find('\n', at);
            if (at == std::string_view::npos) break;
        } else if (c == '[') {
            // A tag after movetext begins the next game even without a result token.
            if (position) end_game();
            const auto end = text.find('\n', at);
            const auto line =
                text.substr(at, end == std::string_view::npos ? std::string_view::npos : end - at);
            if (line.starts_with("[FEN \"")) {
                const auto close = line.rfind('"');
                if (close > 6) current.start_fen = std::string{line.substr(6, close - 6)};
            }
            if (end == std::string_view::npos) break;
            at = end;
        } else if (c == '(') {
            int depth = 0;
            for (; at < text.size(); ++at) {
                if (text[at] == '(') ++depth;
                if (text[at] == ')' && --depth == 0) break;
            }
            ++at;
        } else {
            auto end = text.find_first_of(" \t\r\n", at);
            if (end == std::string_view::npos) end = text.size();
            auto token = text.substr(at, end - at);
            at = end;
            if (token == "1-0" || token == "0-1" || token == "1/2-1/2" || token == "*") {
                auto begun = begin_game();
                if (!begun) return tl::unexpected(begun.error());
                end_game();
                continue;
            }
            if (token.front() == '$') continue;
            if (const auto dot = token.rfind('.'); dot != std::string_view::npos)
                token.remove_prefix(dot + 1);
            if (token.empty()) continue;
            auto begun = begin_game();
            if (!begun) return tl::unexpected(begun.error());
            auto move = san_to_move(*position, token, attacks);
            if (!move) return tl::unexpected(move.error());
            current.moves.push_back(*move);
            static_cast<void>(position->make_move(*move, attacks));
        }
    }
    if (position) end_game();
    return result;
}

Result<TournamentInfo> run_tournament(Settings settings, const BackendFactory& factory,
                                      const std::function<void(std::string_view)>& output) {
    Tournament tournament(std::move(settings), factory, output);
    return tournament.run();
}

int run_selfplay_command(std::span<char*> arguments) {
    std::vector<std::string_view> views;
    views.reserve(arguments.size());
    for (const char* argument : arguments) views.emplace_back(argument);
    auto settings = parse_settings(views);
    if (!settings) {
        std::cerr << "Mouse error: " << settings.error() << '\n';
        return 2;
    }
    std::cout << "id name Mouse\nid author Mouse authors, with Lc0's search design\n" << std::flush;
    const BackendFactory factory = [](const nn::BackendConfig& config) {
        return nn::create_backend(config);
    };
    const std::function<void(std::string_view)> output = [](std::string_view line) {
        std::cout << line << '\n' << std::flush;
    };
    auto done = run_tournament(std::move(*settings), factory, output);
    if (!done) {
        std::cerr << "Mouse error: " << done.error() << '\n';
        return 1;
    }
    return 0;
}
}  // namespace mouse::selfplay
