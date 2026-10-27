// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/util/sha256.h"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
std::string digest(std::string_view text) {
    return mouse::sha256(std::as_bytes(std::span{text.data(), text.size()}));
}
std::vector<std::byte> pattern(std::size_t length) {
    std::vector<std::byte> bytes(length);
    for (std::size_t index = 0; index < length; ++index) {
        bytes.at(index) = static_cast<std::byte>((index * 17) % 256);
    }
    return bytes;
}
class TemporaryDirectory {
   public:
    TemporaryDirectory() {
        const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
        const auto parent = std::filesystem::temp_directory_path();
        for (int attempt = 0; attempt < 1000; ++attempt) {
            path_ =
                parent / ("mouse-sha256-" + std::to_string(nonce) + "-" + std::to_string(attempt));
            if (std::filesystem::create_directory(path_)) return;
        }
        throw std::runtime_error("Cannot create SHA test directory");
    }
    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }
    TemporaryDirectory(const TemporaryDirectory&) = delete;
    TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;
    TemporaryDirectory(TemporaryDirectory&&) = delete;
    TemporaryDirectory& operator=(TemporaryDirectory&&) = delete;
    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

   private:
    std::filesystem::path path_;
};
}  // namespace

TEST_CASE("SHA256 matches standard independent known answers", "[util][sha256]") {
    CHECK(digest("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK(digest("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    CHECK(digest("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    CHECK(digest(std::string(1000000, 'a')) ==
          "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

TEST_CASE("SHA256 handles padding and block boundaries", "[util][sha256]") {
    // Independent CPython hashlib answers for bytes[i] = (17*i) modulo 256.
    for (const auto& [length, expected] : std::array{
             std::pair{55U, "be6fa81b6dd5fba2459dfae9647b535b754b1e593fc5e538d4b092275f7d2d00"},
             std::pair{56U, "9288dbcbf62343e076e614b71d1e584f3f0e1976134b86921bc79be3f414e6b7"},
             std::pair{63U, "751eb0dee0145c6fd9645e58f26dedfbba223074973df31f931ecdba3830c3e1"},
             std::pair{64U, "4331b5485c880a6a8b91828f37c23e3fe847adc55335a8edb14fef8531212eb3"},
             std::pair{65U, "c5bbf822a70255c93801b952a12b007f6c032d6e1b11f95f5e6d47a1270af113"}}) {
        CHECK(mouse::sha256(pattern(length)) == expected);
    }
}

TEST_CASE("File SHA256 streams exact binary data and reports I/O failures", "[util][sha256]") {
    const TemporaryDirectory temporary;
    const auto file = temporary.path() / "binary.dat";
    {
        const auto bytes = pattern(131085);
        std::ofstream output{file, std::ios::binary};
        REQUIRE(output.is_open());
        const auto characters = std::as_bytes(std::span{bytes});
        for (const auto value : characters) output.put(static_cast<char>(value));
        REQUIRE(output.good());
    }
    const auto hashed = mouse::sha256_file(file);
    REQUIRE(hashed.has_value());
    CHECK(*hashed == "4597277e72b187f8524471c4b487013f3fb062a3ba8ac35ccb9c5d5af6f66731");
    std::ofstream{file, std::ios::binary | std::ios::trunc}.close();
    REQUIRE(mouse::sha256_file(file).has_value());
    CHECK(*mouse::sha256_file(file) == digest(""));
    CHECK_FALSE(mouse::sha256_file(temporary.path() / "missing").has_value());
    CHECK_FALSE(mouse::sha256_file(temporary.path()).has_value());
}
