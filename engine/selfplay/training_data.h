// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <vector>

#include "engine/chess/position.h"
#include "engine/nn/encoder.h"
#include "engine/nn/legal_output.h"
#include "engine/search/search.h"
#include "engine/util/result.h"

namespace mouse::selfplay {
// Lc0 fd71a2d trainingdata/trainingdata_v6.h, field for field. Little-endian on disk.
#pragma pack(push, 1)
struct V6TrainingData {
    std::uint32_t version = 6;
    std::uint32_t input_format = 1;
    float probabilities[1858] = {};
    std::uint64_t planes[104] = {};
    std::uint8_t castling_us_ooo = 0;
    std::uint8_t castling_us_oo = 0;
    std::uint8_t castling_them_ooo = 0;
    std::uint8_t castling_them_oo = 0;
    std::uint8_t side_to_move_or_enpassant = 0;
    std::uint8_t rule50_count = 0;
    // bit 5: game adjudicated, bit 4: max game length exceeded, bit 3: best_q is proven best.
    std::uint8_t invariance_info = 0;
    std::uint8_t dummy = 0;
    float root_q = 0, best_q = 0, root_d = 0, best_d = 0, root_m = 0, best_m = 0;
    float plies_left = 0;
    float result_q = 0, result_d = 0;
    float played_q = 0, played_d = 0, played_m = 0;
    // NaN when the evaluation has left the cache.
    float orig_q = 0, orig_d = 0, orig_m = 0;
    std::uint32_t visits = 0;
    std::uint16_t played_idx = 0, best_idx = 0;
    float policy_kld = 0;
    std::uint32_t reserved = 0;
};
#pragma pack(pop)
static_assert(sizeof(V6TrainingData) == 8356, "Wrong struct size");

enum class GameResult : std::uint8_t { Undecided, WhiteWon, Draw, BlackWon };

// A search's Q, D and M, from the side to move.
struct Eval {
    float q = 0, d = 0, m = 0;
};
// What one searched position contributes (lc0 V6TrainingDataArray::Add).
struct TrainingInput {
    const chess::Position* position = nullptr;
    const nn::InputPlanes* planes = nullptr;
    const search::RootStatistics* roots = nullptr;
    std::uint64_t root_visits = 0;
    Eval root;
    Eval best;
    Eval played;
    bool best_is_proven = false;
    chess::Move best_move;
    chess::Move played_move;
    // The network's output for this position in root-edge order, when still in the cache.
    const nn::LegalOutput* evaluation = nullptr;
    double policy_temperature = 1.0;
};
class TrainingGame {
   public:
    [[nodiscard]] Result<void> add(const TrainingInput& input);
    // Lc0 V6TrainingDataArray::Write: the records with their game result and plies left.
    [[nodiscard]] std::vector<V6TrainingData> finish(GameResult result, bool adjudicated) const;
    [[nodiscard]] std::size_t size() const noexcept { return records_.size(); }

   private:
    std::vector<V6TrainingData> records_;
};
// A gzip file of the records. Its deflate stream uses stored blocks: every gzip reader accepts it,
// and lc0's trainers read it the same as one that compressed the data, at about 8 KiB per record.
[[nodiscard]] Result<void> write_training_file(const std::filesystem::path& path,
                                               std::span<const V6TrainingData> records);
[[nodiscard]] Result<std::vector<V6TrainingData>> read_training_file(
    const std::filesystem::path& path);
}  // namespace mouse::selfplay
