// SPDX-License-Identifier: GPL-3.0-or-later
#include <fcntl.h>

#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <memory>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>
#ifdef _WIN32
#include <io.h>
#include <share.h>
#include <sys/stat.h>
#else
#include <unistd.h>
#endif

#include "engine/chess/attacks.h"
#include "engine/chess/movegen.h"
#include "engine/nn/history.h"
#include "engine/nn/policy_map.h"

namespace {
constexpr std::size_t kPositions = 10000;
constexpr std::uint64_t kSeed = 0x4d4f555345020001ULL;
constexpr std::size_t kHistorySlices = 8;
using mouse::chess::AttackTables;
using mouse::chess::Position;
using mouse::nn::EncodingHistory;

struct CloseFile {
    // unique_ptr passes its owned FILE to this legacy C resource deleter.
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    void operator()(std::FILE* file) const noexcept { static_cast<void>(std::fclose(file)); }
};

// The destination is created atomically and exclusively. A failed generation
// retains its newly created incomplete file for inspection; it never deletes or
// overwrites a path which another process might replace. Only finish() permits
// printing the success marker.
class NewOutput {
   public:
    explicit NewOutput(const std::filesystem::path& path) {
        int descriptor = -1;
#ifdef _WIN32
        const auto error =
            _wsopen_s(&descriptor, path.c_str(), _O_CREAT | _O_EXCL | _O_WRONLY | _O_BINARY,
                      _SH_DENYRW, _S_IREAD | _S_IWRITE);
        if (error != 0)
            throw std::system_error(error, std::generic_category(), "create new output");
        auto* file = _fdopen(descriptor, "wb");
        if (file == nullptr) {
            const int saved_errno = errno;
            static_cast<void>(_close(descriptor));
            throw std::system_error(saved_errno, std::generic_category(), "open new output stream");
        }
#else
        // POSIX open requires the mode argument with O_CREAT. Preserve the
        // atomic O_EXCL contract; C++20 has no equivalent noreplace stream mode.
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
        descriptor = ::open(path.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0666);
        if (descriptor < 0)
            throw std::system_error(errno, std::generic_category(), "create new output");
        auto* file = ::fdopen(descriptor, "wb");
        if (file == nullptr) {
            const int saved_errno = errno;
            static_cast<void>(::close(descriptor));
            throw std::system_error(saved_errno, std::generic_category(), "open new output stream");
        }
#endif
        file_.reset(file);
    }
    void write(const std::string& line) {
        if (std::fwrite(line.data(), 1, line.size(), file_.get()) != line.size())
            throw std::system_error(errno, std::generic_category(), "write incomplete output");
    }
    void finish() {
        if (std::fflush(file_.get()) != 0)
            throw std::system_error(errno, std::generic_category(), "flush incomplete output");
        if (std::fclose(file_.release()) != 0)
            throw std::system_error(errno, std::generic_category(), "close incomplete output");
    }

   private:
    std::unique_ptr<std::FILE, CloseFile> file_;
};

std::string json_string(std::string_view value) {
    std::string result{"\""};
    constexpr std::string_view hex = "0123456789abcdef";
    for (const char byte : value) {
        const auto character = static_cast<unsigned char>(byte);
        if (character == '"' || character == '\\') {
            result += '\\';
            result += static_cast<char>(character);
        } else if (character < 0x20) {
            result += "\\u00";
            result += hex[character >> 4U];
            result += hex[character & 15U];
        } else {
            result += static_cast<char>(character);
        }
    }
    result += '"';
    return result;
}

Position parse(std::string_view fen, const AttackTables& attacks) {
    auto position = Position::from_fen(fen, attacks);
    if (!position) throw std::runtime_error(position.error());
    return std::move(*position);
}

std::string captured_fen(const Position& position, const mouse::nn::HistoryFrame& frame) {
    std::istringstream original(position.fen());
    std::array<std::string, 6> fields;
    for (auto& field : fields) {
        if (!(original >> field)) throw std::runtime_error("internal FEN serialization failed");
    }
    fields[3] = frame.en_passant_square() == mouse::chess::kNoSquare
                    ? "-"
                    : mouse::chess::square_name(frame.en_passant_square());
    std::string result;
    for (const auto& field : fields) {
        if (!result.empty()) result += ' ';
        result += field;
    }
    return result;
}

void remember(const Position& position, const EncodingHistory& history,
              std::deque<std::string>& fens) {
    fens.push_back(captured_fen(position, history.frames().back()));
    if (fens.size() > kHistorySlices) fens.pop_front();
}

void play(Position& position, EncodingHistory& history, std::deque<std::string>& fens,
          mouse::chess::Move move, const AttackTables& attacks) {
    static_cast<void>(position.make_move(move, attacks));
    const auto appended = history.append(position);
    if (!appended) throw std::runtime_error(appended.error());
    remember(position, history, fens);
}

void emit(NewOutput& output, std::size_t id, Position& position, const EncodingHistory& history,
          const std::deque<std::string>& fens, const AttackTables& attacks) {
    const auto frames = history.frames().last(fens.size());
    const auto planes = mouse::nn::encode_classical(frames);
    if (!planes) throw std::runtime_error(planes.error());
    std::ostringstream row;
    row.imbue(std::locale::classic());
    row << std::setprecision(std::numeric_limits<float>::max_digits10);
    row << "{\"id\":" << id << ",\"frames\":[";
    for (std::size_t frame = 0; frame < frames.size(); ++frame) {
        if (frame != 0) row << ',';
        row << "{\"fen\":" << json_string(fens[frame])
            << ",\"repetitions\":" << frames[frame].repetitions() << '}';
    }
    row << "],\"planes\":[";
    for (std::size_t plane = 0; plane < planes->size(); ++plane) {
        if (plane != 0) row << ',';
        row << '[' << planes->at(plane).mask << ',' << planes->at(plane).value << ']';
    }
    row << "],\"moves\":[";
    std::array<bool, mouse::nn::PolicyMap::kSize> seen{};
    bool first_move = true;
    for (const auto move : mouse::chess::legal_moves(position, attacks)) {
        const auto index = mouse::nn::PolicyMap::index(move, position.side_to_move());
        if (!index) throw std::runtime_error(index.error());
        if (seen.at(*index)) throw std::runtime_error("duplicate legal policy index");
        seen.at(*index) = true;
        if (!first_move) row << ',';
        first_move = false;
        row << '[' << json_string(mouse::chess::move_to_uci(move, true)) << ',' << *index << ']';
    }
    row << "]}\n";
    output.write(row.str());
}

struct Scenario {
    std::string_view fen;
    std::vector<std::string_view> moves;
};

void generate(const std::filesystem::path& path) {
    NewOutput output(path);
    const AttackTables attacks;
    EncodingHistory history;
    std::deque<std::string> fens;
    std::array<std::size_t, 2> colors{};
    std::size_t emitted = 0;
    const auto record = [&](Position& position) {
        if (emitted >= kPositions) throw std::runtime_error("corpus exceeded fixed position count");
        emit(output, emitted, position, history, fens, attacks);
        ++colors.at(mouse::chess::index(position.side_to_move()));
        ++emitted;
    };
    const auto reset = [&](const Position& position) {
        history.reset(position);
        fens.clear();
        remember(position, history, fens);
    };
    // Fixed edge histories precede seeded walks. FENs are absolute, while every
    // encoded history slice uses the final position's player/rank perspective.
    const std::array scenarios{
        Scenario{.fen = mouse::chess::kStartFen, .moves = {"e2e4", "e7e5", "g1f3", "b8c6", "f1b5"}},
        Scenario{.fen = "1n2r1k1/8/8/3pP3/8/8/8/1N2K3 w - d6 0 2",
                 .moves = {"b1c3", "b8c6", "c3b1", "c6b8", "b1c3", "b8c6", "c3b1", "c6b8"}},
        Scenario{.fen = "1n2r1k1/3p4/8/4P3/8/8/8/1N2K3 b - - 0 1", .moves = {"d7d5"}},
        Scenario{.fen = "4k3/8/8/3p4/8/8/8/4K3 w - d6 0 2", .moves = {}},
        Scenario{.fen = "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 2", .moves = {"e4d3"}},
        Scenario{.fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", .moves = {"e1h1", "e8a8"}},
        Scenario{.fen = "4k3/8/8/8/8/8/8/5K1R w H - 0 1", .moves = {"f1h1"}},
        Scenario{.fen = "5k1r/8/8/8/8/8/8/4K3 b h - 0 1", .moves = {"f8h8"}},
        Scenario{.fen = "4k3/8/8/8/8/8/8/6KR w H - 0 1", .moves = {"g1h1"}},
        Scenario{.fen = "4k3/8/8/8/8/8/8/5KR1 w G - 0 1", .moves = {"f1g1"}},
        Scenario{.fen = "7k/P7/8/8/8/8/8/4K3 w - - 0 1", .moves = {"a7a8n"}},
        Scenario{.fen = "4k3/8/8/8/8/8/p7/7K b - - 0 1", .moves = {"a2a1b"}},
        Scenario{.fen = "4k3/8/8/8/8/8/7R/4K3 w - - 123 37", .moves = {}},
        Scenario{.fen = mouse::chess::kStartFen,
                 .moves = {"b1c3", "b8c6", "c3b1", "c6b8", "b1c3", "b8c6", "c3b1", "c6b8", "b1c3",
                           "b8c6", "c3b1", "c6b8"}}};
    for (const auto& scenario : scenarios) {
        auto position = parse(scenario.fen, attacks);
        reset(position);
        record(position);
        for (const auto text : scenario.moves) {
            const auto move = mouse::chess::parse_uci_move(position, text, attacks, true);
            if (!move) throw std::runtime_error(move.error());
            play(position, history, fens, *move, attacks);
            record(position);
        }
    }
    const auto edge_positions = emitted;
    std::mt19937_64 random(kSeed);
    auto position = parse(mouse::chess::kStartFen, attacks);
    reset(position);
    while (emitted < kPositions) {
        record(position);
        if (emitted == kPositions) break;
        if (mouse::chess::game_status(position, attacks) != mouse::chess::GameStatus::Ongoing) {
            position = parse(mouse::chess::kStartFen, attacks);
            reset(position);
        } else {
            const auto moves = mouse::chess::legal_moves(position, attacks);
            const auto selected = static_cast<std::size_t>(random() % moves.size());
            play(position, history, fens, moves[selected], attacks);
        }
    }
    if (emitted != kPositions || colors[0] == 0 || colors[1] == 0)
        throw std::runtime_error("incomplete fixed corpus");
    output.finish();
    std::cout << "MOUSE_ENCODER_PROBE_RESULT version=1 seed=" << kSeed << " positions=" << emitted
              << " white=" << colors[0] << " black=" << colors[1]
              << " edge_positions=" << edge_positions << '\n';
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("usage: mouse_encoder_probe <new-output.jsonl>");
        const std::span<char*> arguments{argv, static_cast<std::size_t>(argc)};
        generate(std::filesystem::path(arguments[1]));
        return 0;
    } catch (const std::exception& error) {
        std::fputs("mouse_encoder_probe failed: ", stderr);
        std::fputs(error.what(), stderr);
        std::fputs("; any newly created output is incomplete and must not be accepted\n", stderr);
        return 1;
    } catch (...) {
        std::fputs("mouse_encoder_probe failed with an unknown exception; output is incomplete\n",
                   stderr);
        return 1;
    }
}
