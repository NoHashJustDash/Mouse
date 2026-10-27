// SPDX-License-Identifier: GPL-3.0-or-later
// Local cache gate: the same harness links against old/new cache builds.
// CPU-only actual-model evaluation leaves the GPU available for the TM SPRT.
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <locale>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "engine/chess/attacks.h"
#include "engine/chess/movegen.h"
#include "engine/nn/command_runner.h"
#include "engine/nn/history.h"
#include "engine/nn/legal_cache.h"
#include "engine/search/search.h"
#include "engine/util/build_info.h"
#include "engine/util/sha256.h"
namespace {
std::filesystem::path utf8_path(std::string_view text) {
    std::u8string value;
    value.reserve(text.size());
    for (const char byte : text)
        value.push_back(static_cast<char8_t>(static_cast<unsigned char>(byte)));
    return {value};
}
struct Case {
    std::string id, group;
    mouse::chess::Position position;
    mouse::nn::EncodingHistory history;
};
template <class T>
T checked(mouse::Result<T> value) {
    if (!value) throw std::runtime_error(value.error());
    return std::move(*value);
}
void check(const mouse::Result<void>& value) {
    if (!value) throw std::runtime_error(value.error());
}
std::vector<Case> cases(const std::filesystem::path& path,
                        const mouse::chess::AttackTables& attacks) {
    if (std::filesystem::file_size(path) > std::uintmax_t{4} * 1024 * 1024)
        throw std::runtime_error("Equality corpus exceeds 4 MiB");
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open equality corpus");
    std::vector<Case> result;
    std::set<std::string> ids, positions;
    std::array<std::size_t, 3> groups{};
    bool schema = false;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line == "# MouseCacheEquality1") schema = true;
        if (line.empty() || line.front() == '#') continue;
        if (!schema || line.size() > 16384 || result.size() >= 1000)
            throw std::runtime_error("Invalid equality corpus framing/bound");
        std::array<std::string, 4> fields;
        std::size_t offset = 0;
        for (std::size_t field = 0; field < 3; ++field) {
            const auto tab = line.find('\t', offset);
            if (tab == std::string::npos) throw std::runtime_error("Missing equality corpus field");
            fields.at(field) = line.substr(offset, tab - offset);
            offset = tab + 1;
        }
        fields[3] = line.substr(offset);
        if (fields[3].find('\t') != std::string::npos || fields[0].empty() ||
            fields[0].size() > 64 ||
            !std::ranges::all_of(fields[0],
                                 [](char c) {
                                     return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
                                            c == '_' || c == '-';
                                 }) ||
            !ids.insert(fields[0]).second)
            throw std::runtime_error("Invalid/duplicate equality case identifier");
        const std::size_t group = fields[1] == "book" ? 0 : fields[1] == "middle" ? 1 : 2;
        if (fields[1] != "book" && fields[1] != "middle" && fields[1] != "end")
            throw std::runtime_error("Unknown equality position group");
        ++groups.at(group);
        auto position = checked(mouse::chess::Position::from_fen(fields[2], attacks));
        mouse::nn::EncodingHistory history;
        history.reset(position);
        std::istringstream moves(fields[3]);
        std::string text;
        std::size_t plies = 0;
        while (moves >> text) {
            if (++plies > 512) throw std::runtime_error("Equality history exceeds 512 plies");
            const auto move = checked(mouse::chess::parse_uci_move(position, text, attacks, false));
            static_cast<void>(position.make_move(move, attacks));
            check(history.append(position));
        }
        std::istringstream fen(position.fen());
        std::string identity;
        for (unsigned field = 0; field < 4; ++field) {
            fen >> text;
            identity += text + ' ';
        }
        if (!positions.insert(identity).second)
            throw std::runtime_error("Duplicate root position cannot pad the equality gate");
        result.push_back({fields[0], fields[1], std::move(position), std::move(history)});
    }
    if (!input.eof() || result.size() < 200 ||
        std::ranges::any_of(groups, [](auto n) { return n == 0; }))
        throw std::runtime_error(
            "Equality gate requires at least 200 distinct book/middle/end positions");
    return result;
}
}  // namespace
int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: mouse_cache_equivalence <FP32.onnx> <positions.tsv>\n";
        return 2;
    }
    try {
        std::cout.imbue(std::locale::classic());
        const std::span arguments{argv, static_cast<std::size_t>(argc)};
        const auto model = utf8_path(arguments[1]);
        const auto corpus = utf8_path(arguments[2]);
        const auto model_hash = checked(mouse::sha256_file(model));
        const auto corpus_hash = checked(mouse::sha256_file(corpus));
        const mouse::chess::AttackTables attacks;
        const auto positions = cases(corpus, attacks);
        mouse::nn::BackendConfig backend_config;
        backend_config.model_path = std::filesystem::absolute(model);
        backend_config.kind = mouse::nn::BackendKind::Cpu;
        backend_config.precision = mouse::nn::ComputePrecision::Float32;
        backend_config.cpu_threads = 1;
        backend_config.deterministic_compute = true;
        backend_config.batch_buckets = {1, 8};
        check(mouse::nn::resolve_backend_cli_paths(backend_config));
        auto backend = checked(mouse::nn::create_backend(backend_config));
        if (backend->info().onnx_sha256 != model_hash)
            throw std::runtime_error("Loaded model identity differs");
        auto cache =
            checked(mouse::nn::LegalCache::create(2'000'000, backend->info().cache_namespace));
        mouse::search::SearchConfig config;
        config.threads = 1;
        config.maximum_batch = 8;
        config.deterministic = true;
        auto engine =
            checked(mouse::search::SearchEngine::create(config, attacks, *backend, cache.get()));
        mouse::search::SearchLimits limits;
        limits.nodes = 800;
        limits.smart_pruning = false;
        std::cout << R"json({"type":"header","schema":"MouseCacheEquality1","git":")json"
                  << mouse::build::git_hash << R"json(","model_sha256":")json" << model_hash
                  << R"json(","corpus_sha256":")json" << corpus_hash << R"json(","positions":)json"
                  << positions.size()
                  << R"json(,"nodes_per_search":800,"threads":1,"batch":8,"cache_capacity":)json"
                     "2000000,\"passes\":2}\n";
        std::array<std::uint64_t, 2> total_nodes{}, total_ns{};
        for (const auto& row : positions) {
            cache->clear();
            for (std::size_t pass = 0; pass < 2; ++pass) {
                check(engine->set_position(row.position, row.history, false));
                const auto before = cache->stats();
                const auto started = std::chrono::steady_clock::now();
                const auto result = checked(engine->run(limits));
                const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now() - started)
                                         .count();
                const auto after = cache->stats();
                const auto roots = engine->root_moves();
                const auto best = result.best_move;
                if (!best.has_value() || !std::isfinite(result.info.q) ||
                    !std::isfinite(result.info.draw) || elapsed <= 0 ||
                    (pass == 1 && after.hits == before.hits))
                    throw std::runtime_error("Incomplete/nonfinite/unused-cache equality search");
                total_nodes.at(pass) += result.info.nodes;
                total_ns.at(pass) += static_cast<std::uint64_t>(elapsed);
                std::cout << R"json({"type":"case","id":")json" << row.id
                          << R"json(","group":")json" << row.group << R"json(","pass":)json" << pass
                          << R"json(,"bestmove":")json" << mouse::chess::move_to_uci(best.value())
                          << R"json(","nodes":)json" << result.info.nodes << R"json(,"q_bits":)json"
                          << std::bit_cast<std::uint64_t>(result.info.q)
                          << R"json(,"draw_bits":)json"
                          << std::bit_cast<std::uint64_t>(result.info.draw)
                          << R"json(,"elapsed_ns":)json" << elapsed << R"json(,"cache_hits":)json"
                          << after.hits - before.hits << R"json(,"cache_misses":)json"
                          << after.misses - before.misses << R"json(,"charged_bytes":)json"
                          << after.charged_bytes << R"json(,"roots":[)json";
                for (std::size_t i = 0; i < roots.size; ++i) {
                    const auto& root = roots.moves.at(i);
                    if (!std::isfinite(root.q) || !std::isfinite(root.draw) ||
                        !std::isfinite(root.prior))
                        throw std::runtime_error("Nonfinite root statistics");
                    if (i != 0) std::cout << ',';
                    std::cout << "[\"" << mouse::chess::move_to_uci(root.move) << "\","
                              << root.visits << ',' << std::bit_cast<std::uint64_t>(root.q) << ','
                              << std::bit_cast<std::uint64_t>(root.draw) << ','
                              << std::bit_cast<std::uint32_t>(root.prior) << ']';
                }
                std::cout << "]}\n" << std::flush;
            }
        }
        if (checked(mouse::sha256_file(model)) != model_hash ||
            checked(mouse::sha256_file(corpus)) != corpus_hash)
            throw std::runtime_error("Gate inputs changed during execution");
        std::cout << R"json({"type":"complete","positions":)json" << positions.size()
                  << R"json(,"records":)json" << positions.size() * 2 << R"json(,"nodes":[)json"
                  << total_nodes[0] << ',' << total_nodes[1] << R"json(],"elapsed_ns":[)json"
                  << total_ns[0] << ',' << total_ns[1] << "]}\n"
                  << std::flush;
        if (!std::cout) throw std::runtime_error("Cannot publish complete equality output");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Cache equality gate: " << error.what() << '\n';
        return 1;
    }
}