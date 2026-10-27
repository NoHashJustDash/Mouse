// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/chess/types.h"
namespace mouse::chess {
std::string square_name(Square value) {
    assert(value < 64);
    return {static_cast<char>('a' + file_of(value)), static_cast<char>('1' + rank_of(value))};
}
}  // namespace mouse::chess
