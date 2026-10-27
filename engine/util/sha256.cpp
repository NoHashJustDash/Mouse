// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/util/sha256.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace mouse {
namespace {
// SHA-256 constants and operations specified by NIST FIPS 180-4 §§4.2.2,
// 5.1.1, 5.3.3 and 6.2.2:
// https://nvlpubs.nist.gov/nistpubs/fips/nist.fips.180-4.pdf
constexpr std::array<std::uint32_t, 64> kRound{
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U,
    0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU,
    0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU,
    0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
    0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
    0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U,
    0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U,
    0xc67178f2U};

class Sha256 {
   public:
    void update(std::span<const std::byte> bytes) {
        constexpr auto kMaxBytes = std::numeric_limits<std::uint64_t>::max() / 8U;
        if (bytes.size() > kMaxBytes - count_) {
            throw std::length_error("SHA256 input exceeds its 64-bit bit-length limit");
        }
        count_ += bytes.size();
        while (!bytes.empty()) {
            const auto take = std::min(buffer_.size() - used_, bytes.size());
            for (std::size_t index = 0; index < take; ++index) {
                buffer_.at(used_ + index) = bytes[index];
            }
            used_ += take;
            bytes = bytes.subspan(take);
            if (used_ == buffer_.size()) {
                compress();
                used_ = 0;
            }
        }
    }

    [[nodiscard]] std::string finish() {
        const auto bits = count_ * 8U;
        buffer_.at(used_++) = std::byte{0x80};
        if (used_ > 56) {
            while (used_ < buffer_.size()) buffer_.at(used_++) = std::byte{0};
            compress();
            used_ = 0;
        }
        while (used_ < 56) buffer_.at(used_++) = std::byte{0};
        for (std::size_t index = 0; index < 8; ++index) {
            buffer_.at(56 + index) = static_cast<std::byte>((bits >> ((7 - index) * 8)) & 0xffU);
        }
        compress();
        constexpr std::string_view kHex = "0123456789abcdef";
        std::string result(64, '0');
        for (std::size_t index = 0; index < state_.size(); ++index) {
            for (std::size_t nibble = 0; nibble < 8; ++nibble) {
                result.at(index * 8 + nibble) =
                    kHex.at((state_.at(index) >> ((7 - nibble) * 4)) & 0xfU);
            }
        }
        return result;
    }

   private:
    void compress() {
        std::array<std::uint32_t, 64> words{};
        for (std::size_t index = 0; index < 16; ++index) {
            std::uint32_t word = 0;
            for (std::size_t byte = 0; byte < 4; ++byte) {
                word = (word << 8U) | std::to_integer<std::uint32_t>(buffer_.at(index * 4 + byte));
            }
            words.at(index) = word;
        }
        for (std::size_t index = 16; index < words.size(); ++index) {
            const auto x = words.at(index - 15);
            const auto y = words.at(index - 2);
            const auto s0 = std::rotr(x, 7) ^ std::rotr(x, 18) ^ (x >> 3U);
            const auto s1 = std::rotr(y, 17) ^ std::rotr(y, 19) ^ (y >> 10U);
            words.at(index) = words.at(index - 16) + s0 + words.at(index - 7) + s1;
        }
        auto [a, b, c, d, e, f, g, h] = state_;
        for (std::size_t round = 0; round < words.size(); ++round) {
            const auto s1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
            const auto choose = (e & f) ^ (~e & g);
            const auto t1 = h + s1 + choose + kRound.at(round) + words.at(round);
            const auto s0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
            const auto majority = (a & b) ^ (a & c) ^ (b & c);
            const auto t2 = s0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        const std::array working{a, b, c, d, e, f, g, h};
        for (std::size_t index = 0; index < state_.size(); ++index) {
            state_.at(index) += working.at(index);
        }
    }

    std::array<std::uint32_t, 8> state_{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                                        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
    std::array<std::byte, 64> buffer_{};
    std::uint64_t count_ = 0;
    std::size_t used_ = 0;
};
}  // namespace

std::string sha256(std::span<const std::byte> bytes) {
    Sha256 hash;
    hash.update(bytes);
    return hash.finish();
}

Result<std::string> sha256_file(const std::filesystem::path& path) {
    try {
        std::ifstream input{path, std::ios::binary};
        if (!input) return tl::unexpected("Cannot open file for SHA256: " + path.string());
        Sha256 hash;
        std::array<char, 65536> buffer{};
        while (input) {
            input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
            const auto count = input.gcount();
            if (count > 0) {
                hash.update(
                    std::as_bytes(std::span{buffer.data(), static_cast<std::size_t>(count)}));
            }
        }
        if (!input.eof()) return tl::unexpected("Cannot read file for SHA256: " + path.string());
        return hash.finish();
    } catch (const std::exception& error) {
        return tl::unexpected(std::string{"SHA256 file error: "} + error.what());
    }
}
}  // namespace mouse
