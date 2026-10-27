// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/chess/attacks.h"

#include <bit>
#include <cassert>
#include <cstddef>
#include <stdexcept>

#include "engine/chess/magic_numbers.h"

#if defined(MOUSE_USE_PEXT) && MOUSE_USE_PEXT
#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#elif defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#else
#error "MOUSE_USE_PEXT requires an x86-64 compiler with BMI2 intrinsics."
#endif
#endif

namespace mouse::chess {
namespace {
constexpr std::size_t kSlidingTableEntries = 107648;

bool inside(int file, int rank) noexcept { return file >= 0 && file < 8 && rank >= 0 && rank < 8; }

// Only construction uses this walk. Search uses the precomputed hash table.
// The terminal endpoint cannot affect a ray, so masks omit exactly that square.
enum class RayFamily : std::uint8_t { Orthogonal, Diagonal };
enum class RayMode : std::uint8_t { Attacks, RelevantMask };

Bitboard ray(Square from, RayFamily family, Bitboard occupied, RayMode mode) noexcept {
    Bitboard result = 0;
    for (int file_step = -1; file_step <= 1; ++file_step) {
        for (int rank_step = -1; rank_step <= 1; ++rank_step) {
            if ((file_step == 0 && rank_step == 0) ||
                (family == RayFamily::Diagonal ? file_step == 0 || rank_step == 0
                                               : file_step != 0 && rank_step != 0)) {
                continue;
            }
            auto file = static_cast<int>(file_of(from)) + file_step;
            auto rank = static_cast<int>(rank_of(from)) + rank_step;
            while (inside(file, rank)) {
                if (mode == RayMode::RelevantMask && !inside(file + file_step, rank + rank_step)) {
                    break;
                }
                const Bitboard destination = Bitboard{1} << (rank * 8 + file);
                result |= destination;
                if ((occupied & destination) != 0) {
                    break;
                }
                file += file_step;
                rank += rank_step;
            }
        }
    }
    return result;
}

#if defined(MOUSE_USE_PEXT) && MOUSE_USE_PEXT
bool cpu_has_pext() noexcept {
#if defined(_MSC_VER)
    std::array<int, 4> registers{};
    __cpuid(registers.data(), 0);
    if (registers[0] < 7) {
        return false;
    }
    __cpuidex(registers.data(), 7, 0);
    return (static_cast<unsigned>(registers[1]) & (1U << 8U)) != 0;
#else
    return __builtin_cpu_supports("bmi2") != 0;
#endif
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("bmi2")))
#endif
std::size_t
extract_bits(Bitboard occupied, Bitboard mask) noexcept {
    return static_cast<std::size_t>(_pext_u64(occupied, mask));
}
#endif
}  // namespace

AttackTables::AttackTables() {
#if defined(MOUSE_USE_PEXT) && MOUSE_USE_PEXT
    if (!cpu_has_pext()) {
        throw std::runtime_error(
            "Mouse was built with MOUSE_USE_PEXT, but this CPU does not support BMI2/PEXT.");
    }
#endif
    sliding_.reserve(kSlidingTableEntries);
    for (int origin = 0; origin < 64; ++origin) {
        const auto from = static_cast<Square>(origin);
        const auto file = origin % 8;
        const auto rank = origin / 8;
        for (int file_step = -2; file_step <= 2; ++file_step) {
            for (int rank_step = -2; rank_step <= 2; ++rank_step) {
                const auto target_file = file + file_step;
                const auto target_rank = rank + rank_step;
                if (!inside(target_file, target_rank)) {
                    continue;
                }
                const Bitboard destination = Bitboard{1} << (8 * target_rank + target_file);
                const auto absolute_file = file_step < 0 ? -file_step : file_step;
                const auto absolute_rank = rank_step < 0 ? -rank_step : rank_step;
                if ((absolute_file == 1 && absolute_rank == 2) ||
                    (absolute_file == 2 && absolute_rank == 1)) {
                    knights_.at(from) |= destination;
                }
                if (absolute_file <= 1 && absolute_rank <= 1 &&
                    (absolute_file != 0 || absolute_rank != 0)) {
                    kings_.at(from) |= destination;
                }
                if (absolute_file == 1 && rank_step == 1) {
                    pawns_.at(static_cast<std::size_t>(Color::White)).at(from) |= destination;
                }
                if (absolute_file == 1 && rank_step == -1) {
                    pawns_.at(static_cast<std::size_t>(Color::Black)).at(from) |= destination;
                }
            }
        }
        initialize_slider(from, false, rooks_.at(from));
        initialize_slider(from, true, bishops_.at(from));
    }
    if (sliding_.size() != kSlidingTableEntries) {
        throw std::logic_error("Internal sliding attack table size is inconsistent.");
    }
}

void AttackTables::initialize_slider(Square from, bool diagonal, Slider& slider) {
    const auto family = diagonal ? RayFamily::Diagonal : RayFamily::Orthogonal;
    slider.mask = ray(from, family, 0, RayMode::RelevantMask);
    slider.multiplier = diagonal ? detail::kBishopMagics.at(from) : detail::kRookMagics.at(from);
    slider.shift = static_cast<unsigned>(64 - std::popcount(slider.mask));
    slider.offset = sliding_.size();
    const auto count = std::size_t{1} << std::popcount(slider.mask);
    sliding_.resize(slider.offset + count);
    Bitboard subset = 0;
    for (std::size_t entry = 0; entry < count; ++entry) {
        const auto slot = slider.offset + index(slider, subset);
        const auto attacks = ray(from, family, subset, RayMode::Attacks);
        if (sliding_[slot] != 0 && sliding_[slot] != attacks) {
            throw std::logic_error("Internal magic attack table contains a destructive collision.");
        }
        sliding_[slot] = attacks;
        subset = (subset - slider.mask) & slider.mask;
    }
}

std::size_t AttackTables::index(const Slider& slider, Bitboard occupied) noexcept {
#if defined(MOUSE_USE_PEXT) && MOUSE_USE_PEXT
    return extract_bits(occupied, slider.mask);
#else
    return static_cast<std::size_t>(((occupied & slider.mask) * slider.multiplier) >> slider.shift);
#endif
}

Bitboard AttackTables::pawn(Color color, Square from) const noexcept {
    assert(from < 64);
    assert(color == Color::White || color == Color::Black);
    return pawns_.at(static_cast<std::size_t>(color)).at(from);
}

Bitboard AttackTables::knight(Square from) const noexcept {
    assert(from < 64);
    return knights_.at(from);
}

Bitboard AttackTables::king(Square from) const noexcept {
    assert(from < 64);
    return kings_.at(from);
}

// Numeric aliases keep the agreed square/occupancy API consistent; exhaustive tests cover ordering.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Bitboard AttackTables::bishop(Square from, Bitboard occupied) const noexcept {
    assert(from < 64);
    const auto& slider = bishops_.at(from);
    return sliding_[slider.offset + index(slider, occupied)];
}

// Numeric aliases keep the agreed square/occupancy API consistent; exhaustive tests cover ordering.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Bitboard AttackTables::rook(Square from, Bitboard occupied) const noexcept {
    assert(from < 64);
    const auto& slider = rooks_.at(from);
    return sliding_[slider.offset + index(slider, occupied)];
}

Bitboard AttackTables::queen(Square from, Bitboard occupied) const noexcept {
    return bishop(from, occupied) | rook(from, occupied);
}
}  // namespace mouse::chess
