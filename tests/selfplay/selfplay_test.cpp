// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/selfplay/selfplay.h"

#include <atomic>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>

#include "engine/chess/attacks.h"
#include "engine/chess/movegen.h"
#include "engine/nn/policy_map.h"

#include "tests/data/selfplay/lc0-game-v1.inc"

namespace {
using namespace mouse::selfplay;
struct SelfplayBackend final : mouse::nn::Backend {
    mouse::nn::ModelMetadata model;
    mouse::nn::BackendInfo identity;
    std::atomic<std::size_t>* largest = nullptr;
    SelfplayBackend() {
        model.input_format = 1;
        identity.onnx_sha256 = "selfplay-test";
        identity.cache_namespace = "selfplay-test";
    }
    [[nodiscard]] mouse::Result<void> evaluate(std::span<const mouse::nn::InputPlanes> inputs,
                                               std::span<mouse::nn::NNOutput> outputs) override {
        if (largest) {
            auto seen = largest->load();
            while (seen < inputs.size() && !largest->compare_exchange_weak(seen, inputs.size())) {
            }
        }
        for (std::size_t row = 0; row < inputs.size(); ++row) {
            std::uint64_t mix = 0xcbf29ce484222325ULL;
            for (const auto& plane : inputs[row]) mix = (mix ^ plane.mask) * 0x100000001b3ULL;
            mouse::nn::NNOutput value;
            auto seed = mix;
            for (auto& logit : value.policy_logits) {
                seed = seed * 0x9E3779B97F4A7C15ULL + 1;
                logit = static_cast<float>(seed >> 54U) / 256.0F;
            }
            value.wdl = {0.35F, 0.35F, 0.3F};
            value.moves_left = 40.0F;
            outputs[row] = value;
        }
        return {};
    }
    [[nodiscard]] const mouse::nn::ModelMetadata& metadata() const noexcept override {
        return model;
    }
    [[nodiscard]] const mouse::nn::BackendInfo& info() const noexcept override { return identity; }
};
std::vector<std::string_view> views(std::initializer_list<std::string_view> list) { return list; }
Settings settings_for(std::initializer_list<std::string_view> flags) {
    auto all = views(flags);
    all.emplace_back("--weights=unused.onnx");
    all.emplace_back("--backend=cpu");
    auto parsed = parse_settings(all);
    if (!parsed) FAIL(parsed.error());
    return std::move(*parsed);
}
struct Run {
    std::vector<std::string> lines;
    TournamentInfo info;
};
const BackendFactory kFactory = [](const mouse::nn::BackendConfig&)
    -> mouse::Result<std::unique_ptr<mouse::nn::Backend>> {
    return std::unique_ptr<mouse::nn::Backend>(std::make_unique<SelfplayBackend>());
};
Run run(Settings settings) {
    Run out;
    std::mutex mutex;
    const std::function<void(std::string_view)> output = [&](std::string_view line) {
        const std::lock_guard lock(mutex);
        out.lines.emplace_back(line);
    };
    auto done = run_tournament(std::move(settings), kFactory, output);
    if (!done) FAIL(done.error());
    out.info = *done;
    return out;
}
std::filesystem::path scratch(std::string_view name) {
    auto path = std::filesystem::temp_directory_path() / "mouse-selfplay-test" / name;
    std::filesystem::remove_all(path);
    std::filesystem::create_directories(path);
    return path;
}
// A lone pawn's games end soon by rule, long before lc0's 450-ply limit discards one.
std::string lone_pawn_book(std::string_view name) {
    const auto book = scratch(name) / "book.pgn";
    std::ofstream out(book);
    out << "[FEN \"8/8/8/4k3/8/8/4P3/4K3 w - - 0 1\"]\n\n*\n";
    return "--openings-pgn=" + book.generic_string();
}
std::uint64_t games_in(const TournamentInfo& info) {
    std::uint64_t total = 0;
    for (const auto& result : info.results) total += result[0] + result[1];
    return total;
}
}  // namespace

TEST_CASE("Selfplay starts from lc0's defaults", "[selfplay]") {
    const auto settings = settings_for({"--visits=10"});
    const auto& player = settings.players[0][0];
    CHECK(player.options.search.maximum_batch == 32);
    CHECK(player.options.search.selection.cpuct == Catch::Approx(1.2));
    CHECK(player.options.search.selection.cpuct_factor == 0.0);
    CHECK(player.options.search.selection.policy_temperature == 1.0);
    CHECK(player.options.search.gather.max_collision_visits == 1);
    CHECK(player.options.search.gather.max_collision_events == 1);
    CHECK(player.options.cache_history_length == 7);
    CHECK_FALSE(player.options.search.gather.out_of_order_eval);
    CHECK(player.options.search.temperature.temperature == 1.0);
    CHECK(player.options.search.noise_epsilon == 0.25);
    CHECK(player.options.search.selection.fpu_reduction == 0.0);
    CHECK(player.options.search.history_fill == mouse::nn::HistoryFill::No);
    CHECK_FALSE(player.options.search.terminal_proof);
    CHECK_FALSE(player.options.search.two_fold_draws);
    CHECK(player.options.task_workers == 0);
    CHECK(player.options.smart_pruning_factor == 0.0);
    CHECK(settings.share_trees);
    CHECK(settings.games == -1);
    CHECK(settings.parallelism == 8);
    CHECK(player.visits == 10);
}

TEST_CASE("A more specific selfplay context wins", "[selfplay]") {
    const auto settings = settings_for({"--black.visits=30", "--visits=10", "--player1.visits=20",
                                        "--player1.black.visits=40", "--player2.cpuct=2.5",
                                        "--games=4", "--parallelism", "2"});
    CHECK(settings.players[0][0].visits == 20);
    CHECK(settings.players[0][1].visits == 40);
    CHECK(settings.players[1][0].visits == 10);
    CHECK(settings.players[1][1].visits == 30);
    CHECK(settings.players[1][0].options.search.selection.cpuct == 2.5);
    CHECK(settings.players[0][0].options.search.selection.cpuct == Catch::Approx(1.2));
    CHECK(settings.games == 4);
    CHECK(settings.parallelism == 2);
}

TEST_CASE("Selfplay refuses bad settings", "[selfplay]") {
    CHECK_FALSE(parse_settings(views({"--nonsense=1"})).has_value());
    CHECK_FALSE(parse_settings(views({"--games=-3"})).has_value());
    CHECK_FALSE(parse_settings(views({"--openings-mode=wild"})).has_value());
    CHECK_FALSE(parse_settings(views({"not-a-flag"})).has_value());
    std::vector<std::string> lines;
    const std::function<void(std::string_view)> output = [&](std::string_view line) {
        lines.emplace_back(line);
    };
    // No search limit.
    CHECK_FALSE(run_tournament(settings_for({"--games=1"}), kFactory, output).has_value());
    // Policy mode needs a book.
    CHECK_FALSE(
        run_tournament(settings_for({"--games=2", "--policy-mode-size=2", "--mirror-openings=true"}),
                       kFactory, output)
            .has_value());
}

TEST_CASE("PGN openings read FEN tags, SAN and comments", "[selfplay]") {
    const mouse::chess::AttackTables attacks;
    const auto games = parse_pgn(
        "[Event \"x\"]\n\n1. e4 {best} e5 2. Nf3 $1 (2. f4 exf4) Nc6 3. Bb5 a6 *\n\n"
        "[FEN \"4k3/P7/8/8/8/8/8/4K2R w K - 0 1\"]\n\n1. a8=Q+ Ke7 2. O-O 1-0\n",
        attacks);
    REQUIRE(games.has_value());
    REQUIRE(games->size() == 2);
    CHECK((*games)[0].start_fen == mouse::chess::kStartFen);
    REQUIRE((*games)[0].moves.size() == 6);
    CHECK(mouse::chess::move_to_uci((*games)[0].moves[4]) == "f1b5");
    REQUIRE((*games)[1].moves.size() == 3);
    CHECK(mouse::chess::move_to_uci((*games)[1].moves[0]) == "a7a8q");
    CHECK(mouse::chess::move_to_uci((*games)[1].moves[2]) == "e1g1");
    CHECK_FALSE(parse_pgn("1. e5 *", attacks).has_value());
}

TEST_CASE("The tournament status line matches lc0's", "[selfplay]") {
    TournamentInfo info;
    info.results[0] = {3, 1};
    info.results[1] = {1, 2};
    info.results[2] = {0, 1};
    info.move_count = 400;
    info.nodes_total = 40000;
    const auto line = tournament_status_line(info);
    CHECK(line.starts_with("tournamentstatus P1: +4 -1 =3 Win: 68.75% Elo: 136.97 LOS: 91.01%"));
    CHECK(line.find(" P1-W: +3 -0 =1 P1-B: +1 -1 =2 npm 100.000000 nodes 40000 moves 400") !=
          std::string::npos);
}

TEST_CASE("Training records take their result and plies left at the end", "[selfplay]") {
    const mouse::chess::AttackTables attacks;
    auto position = *mouse::chess::Position::from_fen(mouse::chess::kStartFen, attacks);
    mouse::nn::EncodingHistory history;
    history.reset(position);
    auto planes = mouse::nn::encode_classical(history.frames());
    REQUIRE(planes.has_value());
    const auto legal = mouse::chess::legal_moves(position, attacks);
    mouse::search::RootStatistics roots;
    for (const auto move : legal) {
        auto& stat = roots.moves.at(roots.size++);
        stat.move = move;
    }
    roots.moves.at(0).visits = 3;
    roots.moves.at(1).visits = 1;
    TrainingGame game;
    TrainingInput input;
    input.position = &position;
    input.planes = &*planes;
    input.roots = &roots;
    input.root_visits = 5;
    input.root = {.q = 0.1F, .d = 0.5F, .m = 30.0F};
    input.best = {.q = 0.2F, .d = 0.4F, .m = 20.0F};
    input.played = {.q = 0.1F, .d = 0.3F, .m = 25.0F};
    input.best_move = legal[0];
    input.played_move = legal[1];
    REQUIRE(game.add(input));
    REQUIRE(game.add(input));
    auto zero = input;
    zero.root_visits = 1;
    roots.moves.at(0).visits = 0;
    roots.moves.at(1).visits = 0;
    CHECK_FALSE(game.add(zero).has_value());
    roots.moves.at(0).visits = 3;
    roots.moves.at(1).visits = 1;
    const auto records = game.finish(GameResult::WhiteWon, true);
    REQUIRE(records.size() == 2);
    CHECK(records[0].version == 6);
    CHECK(records[0].input_format == 1);
    CHECK(records[0].result_q == 1.0F);
    CHECK(records[0].result_d == 0.0F);
    CHECK((records[0].invariance_info & (1U << 5U)) != 0);
    CHECK((records[0].invariance_info & (1U << 4U)) == 0);
    CHECK(records[0].plies_left == 21.0F);
    CHECK(records[1].plies_left == 20.0F);
    CHECK(records[0].visits == 5);
    CHECK(records[0].castling_us_oo == 1);
    CHECK(records[0].side_to_move_or_enpassant == 0);
    CHECK(std::isnan(records[0].orig_q));
    const auto first = *mouse::nn::PolicyMap::index(legal[0], mouse::chess::Color::White);
    CHECK(records[0].probabilities[first] == 0.75F);
    CHECK(records[0].played_idx == *mouse::nn::PolicyMap::index(legal[1], mouse::chess::Color::White));
    float sum = 0.0F;
    int negative = 0;
    for (const auto value : records[0].probabilities) {
        if (value < 0.0F)
            ++negative;
        else
            sum += value;
    }
    CHECK(sum == 1.0F);
    CHECK(negative == 1858 - 20);
    const auto undecided = game.finish(GameResult::Undecided, true);
    CHECK((undecided[0].invariance_info & (1U << 4U)) != 0);
}

TEST_CASE("Training files round-trip through gzip", "[selfplay]") {
    const auto directory = scratch("roundtrip");
    std::vector<V6TrainingData> records(3);
    records[1].visits = 123;
    records[2].planes[7] = 0x1234;
    const auto path = directory / "game.gz";
    REQUIRE(write_training_file(path, records));
    const auto read = read_training_file(path);
    REQUIRE(read.has_value());
    REQUIRE(read->size() == 3);
    CHECK((*read)[1].visits == 123);
    CHECK((*read)[2].planes[7] == 0x1234);
    CHECK(std::filesystem::file_size(path) > 3 * sizeof(V6TrainingData));
}

TEST_CASE("A selfplay tournament plays its games and writes training data", "[selfplay]") {
    const auto directory = scratch("games");
    const auto training = "--training-directory=" + directory.generic_string();
    const auto openings = lone_pawn_book("games-book");
    auto settings = settings_for({"--games=4", "--parallelism=2", "--visits=12", "--training=true",
                                  "--seed=7", "--ram-limit-mb=256", training, openings});
    const auto result = run(std::move(settings));
    CHECK(games_in(result.info) == 4);
    CHECK(result.info.finished);
    CHECK(result.info.move_count > 0);
    std::size_t ready = 0;
    for (const auto& line : result.lines)
        if (line.starts_with("gameready")) {
            ++ready;
            CHECK(line.find(" trainingfile ") != std::string::npos);
            CHECK(line.find(" result ") != std::string::npos);
        }
    CHECK(ready == 4);
    CHECK(result.lines.back().starts_with("tournamentstatus final"));
    std::size_t files = 0;
    const mouse::chess::AttackTables attacks;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        ++files;
        const auto records = read_training_file(entry.path());
        REQUIRE(records.has_value());
        REQUIRE_FALSE(records->empty());
        for (const auto& record : *records) {
            CHECK(record.version == 6);
            CHECK(record.visits >= 12);
            CHECK(record.probabilities[record.played_idx] >= 0.0F);
            CHECK(record.probabilities[record.best_idx] >= 0.0F);
        }
        // The last record's plies left is its best_m; each earlier one counts up from there.
        CHECK(records->back().plies_left == Catch::Approx(records->back().best_m).margin(0.001F));
        // The cache holds what the search just evaluated, so the original values are present.
        CHECK_FALSE(std::isnan(records->front().orig_q));
        CHECK(records->front().policy_kld > 0.0F);
    }
    CHECK(files == 4);
}

TEST_CASE("Selfplay games follow their opening book", "[selfplay]") {
    const auto directory = scratch("book");
    const auto book = directory / "book.pgn";
    {
        std::ofstream out(book);
        out << "1. d4 d5 *\n\n1. c4 c5 *\n";
    }
    const auto flag = "--openings-pgn=" + book.generic_string();
    auto settings = settings_for({"--games=4", "--parallelism=1", "--visits=4", flag,
                                  "--mirror-openings=true", "--seed=3", "--ram-limit-mb=128"});
    const auto result = run(std::move(settings));
    std::vector<std::string> openings;
    for (const auto& line : result.lines)
        if (line.starts_with("gameready")) {
            CHECK(line.find("play_start_ply 2") != std::string::npos);
            const auto at = line.find(" moves ");
            REQUIRE(at != std::string::npos);
            openings.push_back(line.substr(at + 7, 9));
        }
    REQUIRE(openings.size() == 4);
    CHECK(openings[0] == "d2d4 d7d5");
    CHECK(openings[1] == "d2d4 d7d5");
    CHECK(openings[2] == "c2c4 c7c5");
    CHECK(openings[3] == "c2c4 c7c5");
}

TEST_CASE("Selfplay reuses a tree only if asked", "[selfplay]") {
    const auto openings = lone_pawn_book("reuse-book");
    auto reuse = settings_for({"--games=1", "--parallelism=1", "--visits=64", "--reuse-tree=true",
                               "--temperature=0", "--dirichlet-noise-epsilon=0", "--seed=9",
                               "--ram-limit-mb=128", openings});
    auto fresh = settings_for({"--games=1", "--parallelism=1", "--visits=64", "--temperature=0",
                               "--dirichlet-noise-epsilon=0", "--seed=9", "--ram-limit-mb=128",
                               openings});
    const auto reused = run(std::move(reuse));
    const auto restarted = run(std::move(fresh));
    const auto per_move = [](const TournamentInfo& info) {
        return static_cast<double>(info.nodes_total) / static_cast<double>(info.move_count);
    };
    // Visits counts what the tree already holds, so a reused tree needs fewer new ones.
    CHECK(per_move(reused.info) < per_move(restarted.info));
    CHECK(per_move(restarted.info) == Catch::Approx(64.0).margin(1.0));
}

TEST_CASE("Selfplay resigns as told", "[selfplay]") {
    // Q is about zero throughout, so resigning below 60% ends the game at once.
    auto settings = settings_for({"--games=1", "--parallelism=1", "--visits=8",
                                  "--resign-percentage=60", "--seed=2", "--ram-limit-mb=128"});
    const auto result = run(std::move(settings));
    CHECK(result.info.move_count == 1);
    const auto openings = lone_pawn_book("resign-book");
    auto ignored = settings_for({"--games=1", "--parallelism=1", "--visits=2", "--seed=2",
                                 "--resign-percentage=60", "--resign-playthrough=100",
                                 "--opening-stop-prob=0", "--ram-limit-mb=128", openings});
    const auto played = run(std::move(ignored));
    CHECK(played.info.move_count > 1);
    bool reported = false;
    for (const auto& line : played.lines) reported |= line.starts_with("resign_report fp_threshold");
    CHECK(reported);
}

TEST_CASE("Parallel selfplay games share batches", "[selfplay]") {
    std::atomic<std::size_t> largest{0};
    const BackendFactory factory = [&](const mouse::nn::BackendConfig&)
        -> mouse::Result<std::unique_ptr<mouse::nn::Backend>> {
        auto backend = std::make_unique<SelfplayBackend>();
        backend->largest = &largest;
        return std::unique_ptr<mouse::nn::Backend>(std::move(backend));
    };
    auto settings = settings_for({"--games=8", "--parallelism=8", "--visits=2", "--minibatch-size=8",
                                  "--seed=4", "--ram-limit-mb=256"});
    const std::function<void(std::string_view)> output = [](std::string_view) {};
    const auto done = run_tournament(std::move(settings), factory, output);
    if (!done) FAIL(done.error());
    CHECK(largest.load() > 1);
}

TEST_CASE("Policy and value games need no search", "[selfplay]") {
    const auto directory = scratch("multi");
    const auto book = directory / "book.pgn";
    {
        std::ofstream out(book);
        out << "1. e4 *\n\n1. d4 *\n";
    }
    const auto flag = "--openings-pgn=" + book.generic_string();
    for (const auto mode : {std::string_view{"--policy-mode-size=2"},
                            std::string_view{"--value-mode-size=2"}}) {
        auto settings = settings_for({"--games=-2", "--parallelism=1", mode, flag,
                                      "--mirror-openings=true", "--ram-limit-mb=128"});
        const auto result = run(std::move(settings));
        CHECK(games_in(result.info) == 4);
    }
}

TEST_CASE("Training records hold what lc0's hold for the same game", "[selfplay]") {
    const mouse::chess::AttackTables attacks;
    auto position = *mouse::chess::Position::from_fen(mouse::chess::kStartFen, attacks);
    mouse::nn::EncodingHistory history;
    history.reset(position);
    std::size_t index = 0;
    for (const auto& expected : kLc0Records) {
        INFO("record " << index);
        // Lc0's selfplay default.
        auto planes = mouse::nn::encode_classical(history.frames(), 1, mouse::nn::HistoryFill::No);
        REQUIRE(planes.has_value());
        const auto move =
            mouse::chess::parse_uci_move(position, kLc0GameMoves[index], attacks, true);
        REQUIRE(move.has_value());
        mouse::search::RootStatistics roots;
        for (const auto legal : mouse::chess::legal_moves(position, attacks)) {
            auto& stat = roots.moves.at(roots.size++);
            stat.move = legal;
            stat.visits = 1;
        }
        TrainingGame game;
        TrainingInput input;
        input.position = &position;
        input.planes = &*planes;
        input.roots = &roots;
        input.root_visits = roots.size + 1;
        input.best_move = *move;
        input.played_move = *move;
        REQUIRE(game.add(input));
        const auto record = game.finish(GameResult::Draw, false).front();
        for (std::size_t plane = 0; plane < 104; ++plane) {
            INFO("plane " << plane);
            CHECK(record.planes[plane] == expected.planes.at(plane));
        }
        CHECK(record.played_idx == expected.played);
        CHECK(record.side_to_move_or_enpassant == expected.stm);
        CHECK(record.rule50_count == expected.rule50);
        CHECK(record.castling_us_ooo == expected.castling[0]);
        CHECK(record.castling_us_oo == expected.castling[1]);
        CHECK(record.castling_them_ooo == expected.castling[2]);
        CHECK(record.castling_them_oo == expected.castling[3]);
        std::vector<unsigned> legal;
        for (unsigned i = 0; i < 1858; ++i)
            if (record.probabilities[i] >= 0.0F) legal.push_back(i);
        CHECK(legal == expected.legal);
        static_cast<void>(position.make_move(*move, attacks));
        REQUIRE(history.append(position));
        ++index;
    }
    CHECK(index == 12);
}
