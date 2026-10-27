# Reproducing Mouse's magic multipliers

These constants are produced by Mouse's own deterministic sparse random search.
No multipliers, search seeds or source code were copied from another engine.

The algorithm hashes each subset of a square's relevant occupancy with an
unsigned 64-bit multiply and right shift. A candidate is accepted only when any
colliding subsets have identical attacks. Terminal edge squares are omitted
from the relevant mask because their occupancy cannot change the ray result.
The immutable production attack tables are populated once; no magic search runs
when Mouse starts.

The conceptual algorithm reference is
[Stockfish sf_17.1 src/bitboard.cpp](https://github.com/official-stockfish/Stockfish/blob/sf_17.1/src/bitboard.cpp).
The a1=0 square convention agrees with
[Lc0 src/chess/bitboard.h](https://github.com/LeelaChessZero/lc0/blob/master/src/chess/bitboard.h).

The generator uses seed `0x4d6f7573654d6167` and a fixed per-square/per-piece seed
derivation. All arithmetic is unsigned. This makes the result independent of
compiler, operating system and execution time.

From a VS 2022 x64 developer PowerShell, at the repository root:

```powershell
cl /nologo /std:c++20 /O2 /W4 /WX /permissive- /EHsc tools/magic/generate.cpp /Febuild/generate-magics.exe /Fobuild/generate-magics.obj
cmd /d /c "build\generate-magics.exe > build\magic_numbers.h"
clang-format -i build/magic_numbers.h
git diff --no-index engine/chess/magic_numbers.h build/magic_numbers.h
```

On Linux with GCC 13+ or Clang 17+:

```sh
c++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Werror tools/magic/generate.cpp -o build/generate-magics
build/generate-magics > build/magic_numbers.h
clang-format -i build/magic_numbers.h
diff -u engine/chess/magic_numbers.h build/magic_numbers.h
```

Progress and candidate counts are written to stderr; stdout contains the header.
The generator exits with an error if any square exceeds one billion candidates.
The attack tests exhaust all 107,648 relevant rook/bishop occupancies against a
coordinate ray oracle, then check irrelevant bits and random full occupancies.
Both the magic and PEXT configurations must run this same suite.
