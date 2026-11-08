// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "engine/chess/attacks.h"
#include "engine/chess/position.h"
#include "engine/chess/types.h"
#include "engine/util/result.h"

namespace mouse::search {
// Lc0 WDLScore. Cursed wins and blessed losses are draws under the 50-move rule.
enum class TablebaseWdl : std::int8_t { Loss = -2, BlessedLoss = -1, Draw = 0, CursedWin = 1, Win = 2 };
// Syzygy tablebases through Lc0's probing code (third_party/lc0_syzygy). Safe to probe from
// several threads; replacing tables means opening a new object.
class Tablebase {
   public:
    // `paths` lists directories separated by ';' on Windows and ':' elsewhere. An error means no
    // usable table was found: nothing falls back to searching without saying so.
    [[nodiscard]] static Result<std::shared_ptr<Tablebase>> open(const std::string& paths);
    ~Tablebase();
    Tablebase(const Tablebase&) = delete;
    Tablebase& operator=(const Tablebase&) = delete;
    Tablebase(Tablebase&&) = delete;
    Tablebase& operator=(Tablebase&&) = delete;
    [[nodiscard]] int max_cardinality() const noexcept;
    // Lc0 ExtendNode's conditions: no castling left, 50-move counter at 0, few enough pieces.
    [[nodiscard]] bool can_probe(const chess::Position& position) const noexcept;
    [[nodiscard]] std::optional<TablebaseWdl> probe_wdl(const chess::Position& position,
                                                        const chess::AttackTables& attacks);
    [[nodiscard]] std::optional<int> probe_dtz(const chess::Position& position,
                                               const chess::AttackTables& attacks);
    struct RootFilter {
        std::vector<chess::Move> moves;
        // Lc0 root_is_in_dtz_: DTZ ranked the moves, so the search below them needs no probes.
        bool used_dtz = false;
    };
    // Lc0 MakeRootMoveFilter. Nothing when the root is outside the tables.
    [[nodiscard]] std::optional<RootFilter> root_filter(const chess::Position& position,
                                                        const chess::AttackTables& attacks,
                                                        bool has_repeated, bool fast_play);
    // The DTZ and WDL rankings behind root_filter, separately. Empty when a probe fails.
    [[nodiscard]] std::vector<chess::Move> root_moves_dtz(const chess::Position& position,
                                                          const chess::AttackTables& attacks,
                                                          bool has_repeated);
    [[nodiscard]] std::vector<chess::Move> root_moves_wdl(const chess::Position& position,
                                                          const chess::AttackTables& attacks);

   private:
    struct Impl;
    explicit Tablebase(std::unique_ptr<Impl> implementation);
    std::unique_ptr<Impl> impl_;
};
}  // namespace mouse::search
