// SPDX-License-Identifier: GPL-3.0-or-later
#include <array>
#include <cstddef>
#include <iostream>
#include <memory>

#include "engine/util/runtime_library.h"

namespace {
// An intentional lost allocation exists only in this sanitizer policy child.
// The parent requires LSan to detect it even after validated ORT residency.
[[gnu::noinline]] void lose_allocation() {
    auto allocation = std::make_unique<std::array<std::byte, 256>>();
    const auto* leaked = allocation.release();
    std::cerr << "LSAN_PROBE_ALLOCATION " << static_cast<const void*>(leaked) << '\n';
}
}  // namespace
int main() {
    {
        const auto runtime = mouse::RuntimeLibrary::load(MOUSE_TEST_RUNTIME_PATH);
        if (!runtime) {
            std::cerr << runtime.error() << '\n';
            return 2;
        }
    }
    lose_allocation();
    std::cout << "LSAN_PROBE_COMPLETED\n" << std::flush;
    return 0;
}
