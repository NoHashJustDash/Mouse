// SPDX-License-Identifier: GPL-3.0-or-later
#include <onnxruntime_c_api.h>

#include <cstdint>
#include <cstdlib>

namespace {
const OrtApi* ORT_API_CALL unexpected_api_access(std::uint32_t) noexcept {
    // A version mismatch must be rejected before calling an incompatible API.
    std::abort();
}

const char* ORT_API_CALL wrong_version() noexcept { return "0.0.0"; }
}  // namespace

extern "C" const OrtApiBase* ORT_API_CALL OrtGetApiBase() noexcept {
    static const OrtApiBase base{unexpected_api_access, wrong_version};
    return &base;
}
