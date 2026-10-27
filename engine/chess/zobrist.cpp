// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/chess/zobrist.h"

#include <array>
namespace mouse::chess::zobrist {
namespace {
// Deterministic SplitMix64 stream; immutable compile-time keys avoid shared mutable
// initialization. These are Mouse's own keys, not Lc0's board hash representation.
consteval auto make_keys() {
    std::array<std::uint64_t, 1089> keys{};
    std::uint64_t seed = 0x4d6f757365323032ULL;
    for (auto& key : keys) {
        seed += 0x9e3779b97f4a7c15ULL;
        auto value = seed;
        value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
        key = value ^ (value >> 31U);
    }
    return keys;
}
inline constexpr auto kKeys = make_keys();
}  // namespace
std::uint64_t piece(Color color, PieceType type, Square at) {
    return kKeys.at(index(color) * 384 + index(type) * 64 + at);
}
std::uint64_t side() { return kKeys.at(768); }
std::uint64_t castling(Color color, bool kingside, Square rook) {
    return kKeys.at(769 + index(color) * 128 + (kingside ? 64 : 0) + rook);
}
std::uint64_t en_passant(Square target) { return kKeys.at(1025 + target); }
}  // namespace mouse::chess::zobrist
