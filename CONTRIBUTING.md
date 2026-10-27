# Contributing to Mouse

Bug reports, test results and pull requests are welcome. For behaviour that
should match lc0, include the lc0 version, the position and the command you ran.

Mouse is developed with AI coding assistants (see the README). Contributions
written with AI help are welcome; say so in the pull request, and hold them to
the same tests and evidence as any other change. When code follows another
engine's behaviour, cite the source file it follows, and do not copy code whose
licence is not compatible with GPL-3.0-or-later.

Use C++20 with warnings as errors and RAII. Recoverable failures use
`mouse::Result<T>`; catch exceptions at module boundaries. Dependency direction is
uci -> search -> nn -> chess -> util. Chess code must not depend on ORT.
All source files carry SPDX-License-Identifier: GPL-3.0-or-later.

Write meaningful regression tests before bug fixes. Keep GPU tests tagged
`[gpu]`; run them locally with explicitly selected providers and CPU fallback
disabled. Serialize GPU benchmarks and matches. Never weaken a test or reduce a
required match count to obtain a passing result.

Use CMake presets and the pinned vcpkg manifest. Configure Linux GCC release,
Clang ASan/UBSan, and Clang TSan with their named presets. CI checks both
sanitizers, warnings, formatting, tidy, and Python tests/types/lint.

Python tools target 3.12. Install dependencies from the tracked hash locks and
the package without dependency resolution; see tools/README.md. Run downloaded
file readers with `python -I`. Keep modules importable with a main function.

Use small Conventional Commits, and put the commands you ran and their output in
the pull request. A claimed speed-up needs before/after benchmark numbers, and a
claimed strength change needs a match result. Do not commit SDKs, virtual
environments, downloaded networks, or generated build artifacts.
