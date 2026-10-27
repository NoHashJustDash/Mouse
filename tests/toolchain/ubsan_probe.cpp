// SPDX-License-Identifier: GPL-3.0-or-later
#include <cstdlib>
#include <iostream>
#include <limits>

// Intentional runtime UB belongs only to this child-process policy sentinel.
// LLVM documents that UBSan normally recovers; the parent must reject recovery:
// https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html
int main(int argc, char**) {
    if (argc > 2) {
        return EXIT_FAILURE;
    }
    volatile int increment = argc - 1;
    std::cout << "UBSAN_PROBE_ENTERED\n" << std::flush;
    [[maybe_unused]] volatile int observed = std::numeric_limits<int>::max() + increment;
    std::cout << "UBSAN_PROBE_COMPLETED\n" << std::flush;
    return EXIT_SUCCESS;
}
