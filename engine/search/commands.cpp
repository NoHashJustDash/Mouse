// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/search/commands.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>

namespace mouse::search {
namespace {
struct OptionValue {
    std::string_view option;
    std::string_view text;
};
Result<std::uint64_t> integer(OptionValue argument, std::uint64_t maximum) {
    const auto text = argument.text;
    std::uint64_t value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || text.size() > 20 || parsed.ec != std::errc{} ||
        parsed.ptr != text.data() + text.size() || value == 0 || value > maximum) {
        return tl::unexpected(std::string{argument.option} + " requires an integer in 1.." +
                              std::to_string(maximum));
    }
    return value;
}
Result<double> decimal(OptionValue argument) {
    const auto text = argument.text;
    double value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || text.size() > 128 || parsed.ec != std::errc{} ||
        parsed.ptr != text.data() + text.size() || !std::isfinite(value)) {
        return tl::unexpected(std::string{argument.option} + " requires a finite decimal");
    }
    return value;
}
bool is_neural_option(std::string_view option) {
    constexpr std::array options{"--weights",     "--backend", "--precision",   "--cache-dir",
                                 "--gpu-dll-dir", "--device",  "--cpu-threads", "--workspace-mb",
                                 "--nncache-mb",  "--buckets"};
    return std::ranges::find(options, option) != options.end();
}
bool square(std::string_view text) {
    return text.size() == 2 && text[0] >= 'a' && text[0] <= 'h' && text[1] >= '1' && text[1] <= '8';
}
bool whitespace(char value) {
    return value == ' ' || value == '\t' || value == '\r' || value == '\n' || value == '\v' ||
           value == '\f';
}
Result<std::vector<std::string>> move_list(std::string_view text) {
    if (text.size() > std::size_t{1024} * 1024U) {
        return tl::unexpected(std::string{"--moves exceeds the 1 MiB input limit"});
    }
    std::vector<std::string> moves;
    while (!text.empty()) {
        while (!text.empty() && whitespace(text.front())) text.remove_prefix(1);
        if (text.empty()) break;
        const auto end = std::ranges::find_if(text, whitespace);
        const auto length = static_cast<std::size_t>(end - text.begin());
        const auto move = text.substr(0, length);
        if ((move.size() != 4 && move.size() != 5) || !square(move.substr(0, 2)) ||
            !square(move.substr(2, 2)) ||
            (move.size() == 5 &&
             std::string_view{"qrbn"}.find(move[4]) == std::string_view::npos)) {
            return tl::unexpected(std::string{"--moves requires UCI coordinate moves"});
        }
        if (moves.size() == 131072) {
            return tl::unexpected(std::string{"--moves exceeds the 131072-ply input limit"});
        }
        moves.emplace_back(move);
        text.remove_prefix(length);
    }
    return moves;
}
}  // namespace

Result<SearchCommand> parse_search_command(std::span<const std::string_view> arguments) {
    SearchCommand command;
    std::set<std::string_view> seen;
    std::vector<std::string_view> neural;
    bool explicit_nodes = false;
    bool explicit_buckets = false;
    for (std::size_t index = 0; index < arguments.size(); ++index) {
        const auto option = arguments[index];
        if (is_neural_option(option)) {
            if (index + 1 >= arguments.size()) {
                return tl::unexpected("Missing value after " + std::string{option});
            }
            explicit_buckets |= option == "--buckets";
            neural.push_back(option);
            neural.push_back(arguments[++index]);
            continue;
        }
        if (!seen.insert(option).second) {
            return tl::unexpected("Duplicate search option: " + std::string{option});
        }
        if (option == "--deterministic") {
            command.config.deterministic = true;
            continue;
        }
        if (option == "--chess960") {
            command.chess960 = true;
            continue;
        }
        if (option == "--no-moves-left-utility") {
            command.config.moves_left_utility = false;
            continue;
        }
        if (option == "--no-smart-pruning") {
            command.limits.smart_pruning = false;
            continue;
        }
        if (index + 1 >= arguments.size()) {
            return tl::unexpected("Missing value after " + std::string{option});
        }
        const auto value = arguments[++index];
        if (option == "--fpu-strategy-at-root") {
            if (value == "same")
                command.config.selection.root_fpu_strategy = RootFpuStrategy::Same;
            else if (value == "reduction")
                command.config.selection.root_fpu_strategy = RootFpuStrategy::Reduction;
            else if (value == "absolute")
                command.config.selection.root_fpu_strategy = RootFpuStrategy::Absolute;
            else
                return tl::unexpected(
                    std::string{"--fpu-strategy-at-root requires same, reduction or absolute"});
        } else if (option == "--fen") {
            if (value.empty() || value.size() > 256) {
                return tl::unexpected(std::string{"--fen requires 1..256 characters"});
            }
            command.fen = value;
        } else if (option == "--moves") {
            auto moves = move_list(value);
            if (!moves) return tl::unexpected(moves.error());
            command.moves = std::move(*moves);
        } else if (option == "--threads" || option == "--batch" || option == "--ram-limit-mb" ||
                   option == "--depth" || option == "--nodes" || option == "--movetime") {
            const auto maximum =
                option == "--threads"        ? std::uint64_t{128}
                : option == "--batch"        ? std::uint64_t{1024}
                : option == "--ram-limit-mb" ? std::uint64_t{16384}
                : option == "--depth"        ? std::uint64_t{512}
                : option == "--movetime"
                    ? static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())
                    : std::numeric_limits<std::uint64_t>::max();
            const auto parsed = integer({.option = option, .text = value}, maximum);
            if (!parsed) return tl::unexpected(parsed.error());
            if (option == "--threads")
                command.config.threads = static_cast<std::size_t>(*parsed);
            else if (option == "--batch")
                command.config.maximum_batch = static_cast<std::size_t>(*parsed);
            else if (option == "--ram-limit-mb")
                command.config.ram_limit_bytes = static_cast<std::size_t>(*parsed) * 1024U * 1024U;
            else if (option == "--depth")
                command.limits.depth = static_cast<std::size_t>(*parsed);
            else if (option == "--nodes") {
                command.limits.nodes = *parsed;
                explicit_nodes = true;
            } else
                command.limits.movetime =
                    std::chrono::milliseconds{static_cast<std::int64_t>(*parsed)};
        } else if (option == "--cpuct" || option == "--cpuct-base" || option == "--cpuct-factor" ||
                   option == "--fpu-value" || option == "--fpu-value-at-root" ||
                   option == "--policy-temperature") {
            const auto parsed = decimal({.option = option, .text = value});
            if (!parsed) return tl::unexpected(parsed.error());
            if (option == "--cpuct")
                command.config.selection.cpuct = *parsed;
            else if (option == "--cpuct-base")
                command.config.selection.cpuct_base = *parsed;
            else if (option == "--cpuct-factor")
                command.config.selection.cpuct_factor = *parsed;
            else if (option == "--fpu-value")
                command.config.selection.fpu_reduction = *parsed;
            else if (option == "--fpu-value-at-root")
                command.config.selection.root_fpu = *parsed;
            else
                command.config.selection.policy_temperature = *parsed;
        } else {
            return tl::unexpected("Unknown search option: " + std::string{option});
        }
    }
    auto parsed = nn::parse_backend_command(nn::BackendCommandKind::Configuration, neural);
    if (!parsed) return tl::unexpected(parsed.error());
    command.neural = std::move(*parsed);
    if (!explicit_buckets) {
        auto& buckets = command.neural.backend.batch_buckets;
        buckets.clear();
        for (const std::size_t bucket : {std::size_t{1}, std::size_t{8}, std::size_t{32}}) {
            if (bucket < command.config.maximum_batch) buckets.push_back(bucket);
        }
        buckets.push_back(command.config.maximum_batch);
    }
    if (command.neural.backend.batch_buckets.back() < command.config.maximum_batch) {
        return tl::unexpected(std::string{"--buckets must cover --batch"});
    }
    if (command.config.deterministic && command.neural.backend.cpu_threads != 1) {
        return tl::unexpected(std::string{"--deterministic requires --cpu-threads 1"});
    }
    command.neural.backend.deterministic_compute = command.config.deterministic;
    if (!explicit_nodes && (command.limits.movetime || command.limits.depth)) {
        command.limits.nodes.reset();
    }
    const auto valid = validate_selection_parameters(command.config.selection);
    if (!valid) return tl::unexpected(valid.error());
    return command;
}
}  // namespace mouse::search
