// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/search/command_runner.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <locale>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "engine/chess/attacks.h"
#include "engine/chess/movegen.h"
#include "engine/nn/cached_backend.h"
#include "engine/nn/command_runner.h"
#include "engine/search/commands.h"
#include "engine/util/build_info.h"
#include "engine/util/sha256.h"

namespace mouse::search {
namespace {
std::string json_string(std::string_view text) {
    constexpr std::string_view hex = "0123456789abcdef";
    std::string result{"\""};
    for (const char character : text) {
        const auto value = static_cast<unsigned char>(character);
        if (value == '"' || value == '\\') {
            result += '\\';
            result += static_cast<char>(value);
        } else if (value < 32) {
            result += "\\u00";
            result += hex[value >> 4U];
            result += hex[value & 15U];
        } else {
            result += static_cast<char>(value);
        }
    }
    result += '"';
    return result;
}
std::string_view reason_name(StopReason reason) {
    switch (reason) {
        case StopReason::NodeLimit:
            return "nodes";
        case StopReason::TimeLimit:
            return "time";
        case StopReason::DepthLimit:
            return "depth";
        case StopReason::ExternalStop:
            return "stop";
        case StopReason::MemoryLimit:
            return "memory";
        case StopReason::Terminal:
            return "terminal";
        case StopReason::SmartPruning:
            return "smart-pruning";
    }
    return "invalid";
}
Result<std::string> execute(SearchCommand command) {
    const chess::AttackTables attacks;
    auto position = chess::Position::from_fen(command.fen, attacks);
    if (!position) return tl::unexpected("Invalid FEN: " + position.error());
    const auto variant = chess::validate_uci_castling_mode(*position, command.chess960);
    if (!variant) return tl::unexpected(variant.error());
    nn::EncodingHistory history;
    history.reset(*position);
    std::string identity =
        "MouseSearchInput1\nchess960=" + std::to_string(command.chess960 ? 1 : 0) + "\n" +
        position->fen(command.chess960) + "\n";
    const auto position_space = position->reserve_history(command.moves.size());
    if (!position_space) return tl::unexpected(position_space.error());
    const auto history_space = history.reserve_history(command.moves.size());
    if (!history_space) return tl::unexpected(history_space.error());
    for (const auto& text : command.moves) {
        const auto move = chess::parse_uci_move(*position, text, attacks, command.chess960);
        if (!move) return tl::unexpected("Invalid move " + text + ": " + move.error());
        identity += chess::move_to_uci(*move, command.chess960) + "\n";
        const auto undo = position->make_move(*move, attacks);
        static_cast<void>(undo);
        const auto appended = history.append(*position);
        if (!appended) return tl::unexpected(appended.error());
    }
    const auto input_sha256 = sha256(std::as_bytes(std::span{identity}));
    const auto paths = nn::resolve_backend_cli_paths(command.neural.backend);
    if (!paths) return tl::unexpected(paths.error());
    auto backend = nn::create_backend(command.neural.backend);
    if (!backend) return tl::unexpected(backend.error());
    auto cached = nn::create_cached_backend(std::move(*backend), command.neural.cache_bytes,
                                            command.neural.backend.batch_buckets.back());
    if (!cached) return tl::unexpected(cached.error());
    auto engine = SearchEngine::create(command.config, attacks, **cached);
    if (!engine) return tl::unexpected(engine.error());
    const auto configured = (*engine)->set_position(*position, history);
    if (!configured) return tl::unexpected(configured.error());
    const auto result = (*engine)->run(command.limits);
    if (!result) return tl::unexpected(result.error());
    const auto& info = result->info;
    if (info.pv.size > info.pv.moves.size() || !std::isfinite(info.q) ||
        !std::isfinite(info.draw) || info.charged_bytes > info.budget_bytes ||
        reason_name(result->reason) == "invalid") {
        return tl::unexpected(std::string{"Search returned invalid statistics"});
    }
    std::vector<std::string> pv;
    auto replay = *position;
    for (std::size_t index = 0; index < info.pv.size; ++index) {
        const auto moves = chess::legal_moves(replay, attacks);
        const auto move = info.pv.moves.at(index);
        if (std::ranges::find(moves, move) == moves.end()) {
            return tl::unexpected(std::string{"Search returned an illegal PV"});
        }
        pv.push_back(chess::move_to_uci(move, command.chess960));
        const auto undo = replay.make_move(move, attacks);
        static_cast<void>(undo);
    }
    const auto best_move = result->best_move;
    if (best_move.has_value()) {
        const auto legal = chess::legal_moves(*position, attacks);
        if (std::ranges::find(legal, *best_move) == legal.end() ||
            (!pv.empty() && info.pv.moves[0] != *best_move)) {
            return tl::unexpected(std::string{"Search returned an invalid best move"});
        }
    } else if (!pv.empty() || result->reason != StopReason::Terminal) {
        return tl::unexpected(std::string{"Search omitted a nonterminal best move"});
    }
    const auto unchanged = sha256_file(command.neural.backend.model_path);
    if (!unchanged) return tl::unexpected(unchanged.error());
    if (*unchanged != (*cached)->info().onnx_sha256) {
        return tl::unexpected(std::string{"Model changed during search"});
    }
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << std::setprecision(17) << R"({"schema":"MouseSearch1","version":)"
           << json_string(build::version) << ",\"git\":" << json_string(build::git_hash)
           << ",\"provider\":" << json_string((*cached)->info().provider)
           << ",\"precision\":" << json_string((*cached)->info().precision)
           << ",\"model_sha256\":" << json_string(*unchanged)
           << ",\"input_sha256\":" << json_string(input_sha256)
           << ",\"fen\":" << json_string(position->fen(command.chess960))
           << ",\"deterministic\":" << (command.config.deterministic ? "true" : "false")
           << ",\"moves_left_utility\":" << (command.config.moves_left_utility ? "true" : "false")
           << ",\"threads\":" << (command.config.deterministic ? 1 : command.config.threads)
           << ",\"batch\":" << command.config.maximum_batch
           << ",\"nncache_bytes\":" << command.neural.cache_bytes
           << ",\"stop_reason\":" << json_string(reason_name(result->reason))
           << ",\"nodes\":" << info.nodes << ",\"new_playouts\":" << info.new_playouts
           << ",\"inherited_visits\":" << info.inherited_visits << ",\"depth\":" << info.depth
           << ",\"seldepth\":" << info.seldepth << ",\"q\":" << info.q << ",\"draw\":" << info.draw
           << ",\"elapsed_ms\":" << info.elapsed.count() << ",\"nps\":"
           << (info.elapsed.count() > 0 ? static_cast<double>(info.new_playouts) * 1000.0 /
                                              static_cast<double>(info.elapsed.count())
                                        : 0.0)
           << ",\"charged_bytes\":" << info.charged_bytes
           << ",\"budget_bytes\":" << info.budget_bytes << ",\"tree_nodes\":" << info.tree_nodes
           << ",\"tree_capacity\":" << info.tree_capacity << ",\"bestmove\":";
    if (best_move.has_value())
        output << json_string(chess::move_to_uci(*best_move, command.chess960));
    else
        output << "null";
    output << ",\"pv\":[";
    for (std::size_t index = 0; index < pv.size(); ++index) {
        if (index != 0) output << ',';
        output << json_string(pv[index]);
    }
    output << "],\"root_moves\":[";
    const auto roots = (*engine)->root_moves();
    for (std::size_t index = 0; index < roots.size; ++index) {
        const auto& item = roots.moves.at(index);
        if (index != 0) output << ',';
        output << "{\"move\":" << json_string(chess::move_to_uci(item.move, command.chess960))
               << ",\"visits\":" << item.visits << ",\"q\":" << item.q << ",\"draw\":" << item.draw
               << ",\"prior\":" << item.prior << '}';
    }
    output << "]}\n";
    return output.str();
}
}  // namespace
int run_search_command(std::span<char*> arguments) {
    std::vector<std::string_view> views;
    views.reserve(arguments.size());
    for (const auto* value : arguments) views.emplace_back(value);
    auto command = parse_search_command(views);
    if (!command) {
        std::cerr << "Mouse error: " << command.error() << '\n';
        return 2;
    }
    const auto result = execute(std::move(*command));
    if (!result) {
        std::cerr << "Mouse error: " << result.error() << '\n';
        return 1;
    }
    std::cout << *result;
    std::cout.flush();
    if (!std::cout) {
        std::cerr << "Mouse error: Cannot write complete search result.\n";
        return 1;
    }
    return 0;
}
}  // namespace mouse::search
