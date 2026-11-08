// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/selfplay/training_data.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <tuple>

#include "engine/nn/policy_map.h"

namespace mouse::selfplay {
namespace {
// Lc0 utils/bititer.h ReverseBitsInBytes.
std::uint64_t reverse_bits_in_bytes(std::uint64_t value) noexcept {
    value = ((value >> 1U) & 0x5555555555555555ULL) | ((value & 0x5555555555555555ULL) << 1U);
    value = ((value >> 2U) & 0x3333333333333333ULL) | ((value & 0x3333333333333333ULL) << 2U);
    value = ((value >> 4U) & 0x0F0F0F0F0F0F0F0FULL) | ((value & 0x0F0F0F0F0F0F0F0FULL) << 4U);
    return value;
}
// Lc0 trainingdata.cc DriftCorrect, without its console warnings.
std::tuple<float, float> drift_correct(float q, float d) noexcept {
    q = std::clamp(q, -1.0F, 1.0F);
    d = std::clamp(d, 0.0F, 1.0F);
    const float w = (1.0F - d + q) / 2.0F;
    const float l = w - q;
    if (w < 0.0F || l < 0.0F) d = std::max(0.0F, d + 2.0F * std::min(w, l));
    return {q, d};
}
std::uint32_t crc32_of(std::span<const unsigned char> bytes, std::uint32_t crc) noexcept {
    static const auto table = [] {
        std::array<std::uint32_t, 256> values{};
        for (std::uint32_t i = 0; i < values.size(); ++i) {
            auto value = i;
            for (int bit = 0; bit < 8; ++bit)
                value = (value & 1U) != 0 ? 0xEDB88320U ^ (value >> 1U) : value >> 1U;
            values.at(i) = value;
        }
        return values;
    }();
    crc = ~crc;
    for (const auto byte : bytes) crc = table.at((crc ^ byte) & 0xFFU) ^ (crc >> 8U);
    return ~crc;
}
}  // namespace

Result<void> TrainingGame::add(const TrainingInput& input) {
    const auto& position = *input.position;
    const auto side = position.side_to_move();
    const auto their_side = chess::opposite(side);
    const auto count = input.roots->size;
    const auto total = input.root_visits > 0 ? input.root_visits - 1 : 0;
    // Lc0 throws too: only a position with one legal move can stop before any child has a visit.
    if (total == 0 && count != 1)
        return tl::unexpected(std::string{"Search generated invalid training data"});
    V6TrainingData result;
    for (std::size_t i = 0; i < result.planes[0] * 0 + 104; ++i)
        result.planes[i] = reverse_bits_in_bytes(input.planes->at(i).mask);
    std::fill(std::begin(result.probabilities), std::end(result.probabilities), -1.0F);
    float kld_sum = 0.0F, prior_total = 0.0F;
    // The priors as the network gave them: the cached logits with the softmax temperature undone.
    std::array<float, chess::MoveList::kCapacity> priors{};
    if (input.evaluation) {
        const auto temperature = static_cast<float>(input.policy_temperature);
        double sum = 0.0;
        for (std::size_t i = 0; i < count; ++i)
            sum += std::exp(input.evaluation->logits.at(i) / temperature);
        for (std::size_t i = 0; i < count; ++i) {
            const auto soft =
                static_cast<float>(std::exp(input.evaluation->logits.at(i) / temperature) / sum);
            priors.at(i) = std::pow(soft, temperature);
        }
    }
    for (std::size_t i = 0; i < count; ++i) {
        const auto& move = input.roots->moves.at(i);
        const float fraction = total > 0 ? static_cast<float>(move.visits) /
                                               static_cast<float>(total)
                                         : 1.0F;
        if (input.evaluation) {
            if (fraction > 0.0F) kld_sum += fraction * std::log(fraction / priors.at(i));
            prior_total += priors.at(i);
        }
        const auto index = nn::PolicyMap::index(move.move, side);
        if (!index) return tl::unexpected(index.error());
        result.probabilities[*index] = fraction;
    }
    if (input.evaluation)
        kld_sum = std::max(kld_sum + std::log(prior_total), 0.0F) +
                  std::numeric_limits<float>::min();
    result.policy_kld = kld_sum;
    const auto can = [&](chess::Color color, bool kingside) -> std::uint8_t {
        return position.castling_rook(color, kingside) != chess::kNoSquare ? 1 : 0;
    };
    result.castling_us_ooo = can(side, false);
    result.castling_us_oo = can(side, true);
    result.castling_them_ooo = can(their_side, false);
    result.castling_them_oo = can(their_side, true);
    result.side_to_move_or_enpassant = side == chess::Color::Black ? 1 : 0;
    result.invariance_info = input.best_is_proven ? 1U << 3U : 0U;
    result.rule50_count = static_cast<std::uint8_t>(std::min<std::uint64_t>(
        position.halfmove_clock(), std::numeric_limits<std::uint8_t>::max()));
    result.result_q = 0.0F;
    result.result_d = 1.0F;
    result.root_q = input.root.q;
    result.best_q = input.best.q;
    result.played_q = input.played.q;
    result.root_d = input.root.d;
    result.best_d = input.best.d;
    result.played_d = input.played.d;
    constexpr auto kNaN = std::numeric_limits<float>::quiet_NaN();
    result.orig_q = kNaN;
    result.orig_d = kNaN;
    result.orig_m = kNaN;
    if (input.evaluation) {
        const auto& wdl = input.evaluation->wdl;
        const auto sum = wdl[0] + wdl[1] + wdl[2];
        result.orig_q = (wdl[0] - wdl[2]) / sum;
        result.orig_d = wdl[1] / sum;
        result.orig_m = input.evaluation->moves_left.value_or(0.0F);
    }
    std::tie(result.best_q, result.best_d) = drift_correct(result.best_q, result.best_d);
    std::tie(result.root_q, result.root_d) = drift_correct(result.root_q, result.root_d);
    std::tie(result.played_q, result.played_d) = drift_correct(result.played_q, result.played_d);
    result.root_m = input.root.m;
    result.best_m = input.best.m;
    result.played_m = input.played.m;
    result.visits = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(input.root_visits, std::numeric_limits<std::uint32_t>::max()));
    const auto best_index = nn::PolicyMap::index(input.best_move, side);
    if (!best_index) return tl::unexpected(best_index.error());
    const auto played_index = nn::PolicyMap::index(input.played_move, side);
    if (!played_index) return tl::unexpected(played_index.error());
    result.best_idx = *best_index;
    result.played_idx = *played_index;
    records_.push_back(result);
    return {};
}

std::vector<V6TrainingData> TrainingGame::finish(GameResult game_result, bool adjudicated) const {
    auto out = records_;
    if (out.empty()) return out;
    // Lc0 bases the estimate on best_m and leaves any other approach to external processing.
    float estimate = out.back().best_m + static_cast<float>(out.size()) - 1.0F;
    for (auto& chunk : out) {
        const bool black_to_move = chunk.side_to_move_or_enpassant != 0;
        if (game_result == GameResult::WhiteWon) {
            chunk.result_q = black_to_move ? -1.0F : 1.0F;
            chunk.result_d = 0.0F;
        } else if (game_result == GameResult::BlackWon) {
            chunk.result_q = black_to_move ? 1.0F : -1.0F;
            chunk.result_d = 0.0F;
        } else {
            chunk.result_q = 0.0F;
            chunk.result_d = 1.0F;
        }
        if (adjudicated) chunk.invariance_info |= 1U << 5U;
        if (adjudicated && game_result == GameResult::Undecided) chunk.invariance_info |= 1U << 4U;
        chunk.plies_left = estimate;
        estimate -= 1.0F;
    }
    return out;
}

Result<void> write_training_file(const std::filesystem::path& path,
                                 std::span<const V6TrainingData> records) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) return tl::unexpected("Cannot create training file " + path.generic_string());
    constexpr std::array<unsigned char, 10> header{0x1F, 0x8B, 8, 0, 0, 0, 0, 0, 0, 0xFF};
    out.write(reinterpret_cast<const char*>(header.data()), header.size());
    const auto bytes = std::span{reinterpret_cast<const unsigned char*>(records.data()),
                                 records.size() * sizeof(V6TrainingData)};
    std::uint32_t crc = 0;
    constexpr std::size_t kBlock = 65535;
    std::size_t offset = 0;
    do {
        const auto size = std::min(kBlock, bytes.size() - offset);
        const auto length = static_cast<std::uint16_t>(size);
        const auto inverse = static_cast<std::uint16_t>(~length);
        const std::array<unsigned char, 5> block{
            static_cast<unsigned char>(offset + size == bytes.size() ? 1 : 0),
            static_cast<unsigned char>(length & 0xFFU),
            static_cast<unsigned char>(length >> 8U),
            static_cast<unsigned char>(inverse & 0xFFU),
            static_cast<unsigned char>(inverse >> 8U)};
        out.write(reinterpret_cast<const char*>(block.data()), block.size());
        const auto part = bytes.subspan(offset, size);
        out.write(reinterpret_cast<const char*>(part.data()),
                  static_cast<std::streamsize>(part.size()));
        crc = crc32_of(part, crc);
        offset += size;
    } while (offset < bytes.size());
    const auto put32 = [&](std::uint32_t value) {
        const std::array<unsigned char, 4> word{
            static_cast<unsigned char>(value & 0xFFU), static_cast<unsigned char>((value >> 8U) & 0xFFU),
            static_cast<unsigned char>((value >> 16U) & 0xFFU),
            static_cast<unsigned char>((value >> 24U) & 0xFFU)};
        out.write(reinterpret_cast<const char*>(word.data()), word.size());
    };
    put32(crc);
    put32(static_cast<std::uint32_t>(bytes.size() & 0xFFFFFFFFU));
    out.flush();
    if (!out) return tl::unexpected("Unable to write into " + path.generic_string());
    return {};
}

Result<std::vector<V6TrainingData>> read_training_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return tl::unexpected("Cannot open training file " + path.generic_string());
    std::vector<unsigned char> file((std::istreambuf_iterator<char>(in)),
                                    std::istreambuf_iterator<char>());
    if (file.size() < 18 || file[0] != 0x1F || file[1] != 0x8B || file[2] != 8 || file[3] != 0)
        return tl::unexpected("Not a plain gzip file: " + path.generic_string());
    std::vector<unsigned char> data;
    std::size_t at = 10;
    for (bool last = false; !last;) {
        if (at + 5 > file.size() || (file[at] & 0xFEU) != 0)
            return tl::unexpected("Only stored gzip blocks can be read: " + path.generic_string());
        last = (file[at] & 1U) != 0;
        const std::size_t size = file[at + 1] | (std::size_t{file[at + 2]} << 8U);
        at += 5;
        if (at + size > file.size())
            return tl::unexpected("Truncated training file: " + path.generic_string());
        data.insert(data.end(), file.begin() + static_cast<std::ptrdiff_t>(at),
                    file.begin() + static_cast<std::ptrdiff_t>(at + size));
        at += size;
    }
    const std::uint32_t stored = file[at] | (std::uint32_t{file[at + 1]} << 8U) |
                                 (std::uint32_t{file[at + 2]} << 16U) |
                                 (std::uint32_t{file[at + 3]} << 24U);
    if (crc32_of(data, 0) != stored)
        return tl::unexpected("Training file fails its checksum: " + path.generic_string());
    if (data.size() % sizeof(V6TrainingData) != 0)
        return tl::unexpected("Training file holds a partial record: " + path.generic_string());
    std::vector<V6TrainingData> records(data.size() / sizeof(V6TrainingData));
    if (!data.empty()) std::memcpy(static_cast<void*>(records.data()), data.data(), data.size());
    return records;
}
}  // namespace mouse::selfplay
