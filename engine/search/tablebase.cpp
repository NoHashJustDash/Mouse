// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/search/tablebase.h"

#include <bit>
#include <exception>

#include "third_party/lc0_syzygy/syzygy.h"

namespace mouse::search {
struct Tablebase::Impl {
    lczero::SyzygyTablebase tables;
};
Tablebase::Tablebase(std::unique_ptr<Impl> implementation) : impl_(std::move(implementation)) {}
Tablebase::~Tablebase() = default;
Result<std::shared_ptr<Tablebase>> Tablebase::open(const std::string& paths) {
    try {
        auto implementation = std::make_unique<Impl>();
        if (!implementation->tables.init(paths))
            return tl::unexpected("No Syzygy tablebases found in: " + paths);
        return std::shared_ptr<Tablebase>(new Tablebase(std::move(implementation)));
    } catch (const std::exception& error) {
        return tl::unexpected("Cannot load Syzygy tablebases: " + std::string{error.what()});
    }
}
int Tablebase::max_cardinality() const noexcept { return impl_->tables.max_cardinality(); }
bool Tablebase::can_probe(const chess::Position& position) const noexcept {
    if (position.halfmove_clock() != 0) return false;
    if (std::popcount(position.occupancy()) > max_cardinality()) return false;
    for (const auto color : {chess::Color::White, chess::Color::Black})
        for (const bool side : {false, true})
            if (position.castling_rook(color, side) != chess::kNoSquare) return false;
    return true;
}
std::optional<TablebaseWdl> Tablebase::probe_wdl(const chess::Position& position,
                                                 const chess::AttackTables& attacks) {
    lczero::ProbeState state{};
    const lczero::Position view{position, attacks};
    const auto wdl = impl_->tables.probe_wdl(view, &state);
    if (state == lczero::FAIL) return std::nullopt;
    return static_cast<TablebaseWdl>(static_cast<int>(wdl));
}
std::optional<int> Tablebase::probe_dtz(const chess::Position& position,
                                        const chess::AttackTables& attacks) {
    lczero::ProbeState state{};
    const lczero::Position view{position, attacks};
    const auto dtz = impl_->tables.probe_dtz(view, &state);
    if (state == lczero::FAIL) return std::nullopt;
    return dtz;
}
std::optional<Tablebase::RootFilter> Tablebase::root_filter(const chess::Position& position,
                                                            const chess::AttackTables& attacks,
                                                            bool has_repeated, bool fast_play) {
    // lc0 fd71a2d classic/search.cc MakeRootMoveFilter: only castling rights and the piece count
    // rule a root out; its 50-move counter is what root_probe ranks with.
    if (std::popcount(position.occupancy()) > max_cardinality()) return std::nullopt;
    for (const auto color : {chess::Color::White, chess::Color::Black})
        for (const bool side : {false, true})
            if (position.castling_rook(color, side) != chess::kNoSquare) return std::nullopt;
    const lczero::Position view{position, attacks};
    std::vector<lczero::Move> safe;
    RootFilter filter;
    if (impl_->tables.root_probe(view, fast_play || has_repeated, false, &safe)) {
        filter.used_dtz = true;
    } else {
        safe.clear();
        if (!impl_->tables.root_probe_wdl(view, &safe)) return std::nullopt;
    }
    filter.moves.reserve(safe.size());
    for (const auto& move : safe) filter.moves.push_back(move.raw());
    return filter;
}
std::vector<chess::Move> Tablebase::root_moves_dtz(const chess::Position& position,
                                                   const chess::AttackTables& attacks,
                                                   bool has_repeated) {
    const lczero::Position view{position, attacks};
    std::vector<lczero::Move> safe;
    std::vector<chess::Move> moves;
    if (!impl_->tables.root_probe(view, has_repeated, false, &safe)) return moves;
    for (const auto& move : safe) moves.push_back(move.raw());
    return moves;
}
std::vector<chess::Move> Tablebase::root_moves_wdl(const chess::Position& position,
                                                   const chess::AttackTables& attacks) {
    const lczero::Position view{position, attacks};
    std::vector<lczero::Move> safe;
    std::vector<chess::Move> moves;
    if (!impl_->tables.root_probe_wdl(view, &safe)) return moves;
    for (const auto& move : safe) moves.push_back(move.raw());
    return moves;
}
}  // namespace mouse::search
