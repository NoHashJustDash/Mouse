// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/uci/options.h"

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string_view>

namespace {
mouse::Result<mouse::uci::OptionUpdate> apply(const mouse::uci::Options& old, std::string name,
                                              std::string value) {
    return mouse::uci::apply_option(old, {.name = std::move(name), .value = std::move(value)});
}
}  // namespace
TEST_CASE("UCI options advertise actual conservative defaults", "[uci][options]") {
    const mouse::uci::Options options;
    CHECK(options.backend.model_path.empty());
    CHECK(options.search.threads == 2);
    CHECK(options.auto_minibatch);
    CHECK(options.search.maximum_batch == 64);
    CHECK(options.nncache_entries == 2000000);
    CHECK(options.search.ram_limit_bytes == 536870912);
    CHECK(options.move_overhead_ms == 30);
    const auto lines = mouse::uci::option_lines(options);
    for (const auto name : {"WeightsFile",
                            "Backend",
                            "Threads",
                            "MinibatchSize",
                            "NNCacheSize",
                            "RamLimitMb",
                            "CPuct",
                            "CPuctBase",
                            "CPuctFactor",
                            "FpuValue",
                            "PolicyTemperature",
                            "MultiPV",
                            "UCI_ShowWDL",
                            "UCI_Chess960",
                            "Ponder",
                            "MoveOverheadMs",
                            "Precision",
                            "GpuDllDirectories",
                            "CacheDirectory",
                            "Device",
                            "CpuThreads",
                            "WorkspaceMb",
                            "Deterministic",
                            "CacheHistoryLength"}) {
        CHECK(std::ranges::any_of(lines, [&](const std::string& line) {
            return line.starts_with("option name " + std::string{name} + " type ");
        }));
    }
}
TEST_CASE("UCI configuration updates are atomic and identify owner rebuilds", "[uci][options]") {
    const mouse::uci::Options original;
    const auto threads = apply(original, "threads", "8");
    REQUIRE(threads);
    CHECK(threads->options.search.threads == 8);
    CHECK(threads->impact == mouse::uci::Reconfigure::Search);
    CHECK(original.search.threads == 2);
    const auto repeated = apply(threads->options, "Threads", "8");
    REQUIRE(repeated);
    CHECK(repeated->impact == mouse::uci::Reconfigure::None);
    const auto batch = apply(original, "MinibatchSize", "17");
    REQUIRE(batch);
    CHECK(batch->impact == mouse::uci::Reconfigure::Backend);
    CHECK(batch->options.backend.batch_buckets == std::vector<std::size_t>{1, 8, 17});
    const auto cache = apply(original, "NNCacheSize", "2");
    REQUIRE(cache);
    CHECK(cache->options.nncache_entries == 2);
    CHECK(cache->impact == mouse::uci::Reconfigure::Search);
    const auto reporting = apply(original, "UCI_ShowWDL", "true");
    REQUIRE(reporting);
    CHECK(reporting->options.show_wdl);
    CHECK(reporting->impact == mouse::uci::Reconfigure::None);
    const auto det = apply(original, "Deterministic", "true");
    REQUIRE(det);
    CHECK(det->options.search.deterministic);
    CHECK(det->options.backend.deterministic_compute);
    CHECK(det->impact == mouse::uci::Reconfigure::Backend);
    CHECK_FALSE(apply(det->options, "CpuThreads", "2"));
}
TEST_CASE("Every advertised UCI option parses a valid value", "[uci][options]") {
    constexpr std::array<std::array<std::string_view, 2>, 23> valid{
        {{"WeightsFile", "T1 model.onnx"},
         {"Backend", "cpu"},
         {"Precision", "fp32"},
         {"Threads", "8"},
         {"MinibatchSize", "32"},
         {"NNCacheSize", "0"},
         {"CacheHistoryLength", "7"},
         {"RamLimitMb", "64"},
         {"CPuct", "1.8"},
         {"CPuctBase", "50000"},
         {"CPuctFactor", "0"},
         {"FpuValue", "0.2"},
         {"PolicyTemperature", "1.2"},
         {"MultiPV", "3"},
         {"UCI_ShowWDL", "true"},
         {"UCI_Chess960", "true"},
         {"Ponder", "true"},
         {"MoveOverheadMs", "5"},
         {"CacheDirectory", "local cache"},
         {"Device", "2"},
         {"CpuThreads", "3"},
         {"WorkspaceMb", "128"},
         {"Deterministic", "true"}}};
    const mouse::uci::Options original;
    for (const auto& pair : valid) {
        INFO(pair[0]);
        CHECK(apply(original, std::string{pair[0]}, std::string{pair[1]}));
    }
    CHECK(apply(original, "GpuDllDirectories", "<empty>"));
}
TEST_CASE("Invalid UCI options never mutate active settings", "[uci][options]") {
    constexpr std::array<std::array<std::string_view, 2>, 22> invalid{{{"Unknown", "3"},
                                                                       {"Threads", "0"},
                                                                       {"Threads", "129"},
                                                                       {"CacheHistoryLength", "8"},
                                                                       {"CacheHistoryLength", "-1"},
                                                                       {"MinibatchSize", "1025"},
                                                                       {"NNCacheSize", "-1"},
                                                                       {"RamLimitMb", "0"},
                                                                       {"MultiPV", "0"},
                                                                       {"MultiPV", "257"},
                                                                       {"CPuct", "nan"},
                                                                       {"CPuctBase", "0"},
                                                                       {"CPuctFactor", "-1"},
                                                                       {"FpuValue", "-1"},
                                                                       {"PolicyTemperature", "0"},
                                                                       {"MoveOverheadMs", "-1"},
                                                                       {"Device", "1024"},
                                                                       {"CpuThreads", "129"},
                                                                       {"WorkspaceMb", "0"},
                                                                       {"Backend", "auto"},
                                                                       {"Precision", "auto"},
                                                                       {"Ponder", "maybe"}}};
    const mouse::uci::Options original;
    for (const auto& pair : invalid) {
        INFO(pair[0]);
        CHECK_FALSE(apply(original, std::string{pair[0]}, std::string{pair[1]}));
    }
    CHECK_FALSE(mouse::uci::apply_option(original, {.name = "Threads", .value = std::nullopt}));
    CHECK_FALSE(apply(original, "WeightsFile", std::string{"bad\npath"}));
    CHECK(original.search.threads == 2);
}
TEST_CASE("UCI startup options use explicit model provider and precision", "[uci][options]") {
    const std::array<std::string_view, 8> args{"--weights",   "T1.onnx", "--backend", "cpu",
                                               "--precision", "fp32",    "--threads", "2"};
    const auto result = mouse::uci::startup_options(args);
    REQUIRE(result);
    CHECK(result->search.threads == 2);
    CHECK(result->nncache_entries == 2000000);
    const std::array<std::string_view, 8> analysis{"--weights",   "T1.onnx", "--backend", "cpu",
                                                   "--precision", "fp32",    "--nodes",   "1"};
    CHECK_FALSE(mouse::uci::startup_options(analysis));
    CHECK(mouse::uci::startup_options({}));
}

#ifdef _WIN32
TEST_CASE("Malformed UTF8 paths return errors without escaping option transactions",
          "[uci][options]") {
    const mouse::uci::Options original;
    for (const auto* name : {"WeightsFile", "CacheDirectory", "GpuDllDirectories"}) {
        const auto invalid = std::string{"C:/nets/"} + std::string(1, static_cast<char>(0xff));
        CHECK_NOTHROW(apply(original, name, invalid));
        CHECK_FALSE(apply(original, name, invalid));
        CHECK(original.backend.model_path.empty());
    }
    const auto unicode =
        apply(original, "WeightsFile", std::string{"C:/nets/"} + "\xe2\x99\x9e.onnx");
    REQUIRE(unicode);
    CHECK(unicode->options.backend.model_path.generic_u8string() == u8"C:/nets/\u265e.onnx");
}
#endif

TEST_CASE("UCI root FPU options advertise matched defaults and accept explicit strategies",
          "[uci][options][root-fpu]") {
    const mouse::uci::Options original;
    const auto lines = mouse::uci::option_lines(original);
    CHECK(std::ranges::find(lines,
                            "option name FpuStrategyAtRoot type combo default same var same var "
                            "reduction var absolute") != lines.end());
    CHECK(std::ranges::find(lines, "option name FpuValueAtRoot type string default 1") !=
          lines.end());
    for (const auto* strategy : {"same", "reduction", "absolute"}) {
        const auto changed = apply(original, "FpuStrategyAtRoot", strategy);
        REQUIRE(changed);
        const auto value = apply(changed->options, "FpuValueAtRoot", "-0.5");
        REQUIRE(value);
        CHECK(value->impact == mouse::uci::Reconfigure::Search);
    }
    CHECK_FALSE(apply(original, "FpuStrategyAtRoot", "automatic"));
    CHECK_FALSE(apply(original, "FpuValueAtRoot", "nan"));
    CHECK_FALSE(apply(original, "FpuValueAtRoot", "100.00001"));
    CHECK_FALSE(apply(original, "FpuValueAtRoot", "-100.00001"));
}

TEST_CASE("Root FPU UCI changes are atomic search rebuilds and startup settings agree",
          "[uci][options][root-fpu]") {
    using mouse::search::RootFpuStrategy;
    const mouse::uci::Options original;
    CHECK(original.search.selection.root_fpu_strategy == RootFpuStrategy::Same);
    CHECK(original.search.selection.fpu_reduction == 0.33);
    CHECK(original.search.selection.root_fpu == 1.0);
    const auto changed = apply(original, "FpuStrategyAtRoot", "absolute");
    REQUIRE(changed);
    CHECK(changed->impact == mouse::uci::Reconfigure::Search);
    CHECK(changed->options.search.selection.root_fpu_strategy == RootFpuStrategy::Absolute);
    CHECK(original.search.selection.root_fpu_strategy == RootFpuStrategy::Same);
    const auto unchanged = apply(changed->options, "FpuStrategyAtRoot", "absolute");
    REQUIRE(unchanged);
    CHECK(unchanged->impact == mouse::uci::Reconfigure::None);
    const std::array<std::string_view, 10> arguments{"--weights",
                                                     "model",
                                                     "--backend",
                                                     "cpu",
                                                     "--precision",
                                                     "fp32",
                                                     "--fpu-strategy-at-root",
                                                     "reduction",
                                                     "--fpu-value-at-root",
                                                     "0.5"};
    const auto startup = mouse::uci::startup_options(arguments);
    REQUIRE(startup);
    CHECK(startup->search.selection.root_fpu_strategy == RootFpuStrategy::Reduction);
    CHECK(startup->search.selection.root_fpu == 0.5);
}

TEST_CASE("Default UCI disk cache is absolute and outside the GUI working directory",
          "[uci][options][cache-path]") {
    const mouse::uci::Options options;
    CHECK(options.backend.cache_directory.is_absolute());
#ifdef _WIN32
    const auto* local = _wgetenv(L"LOCALAPPDATA");
    REQUIRE(local != nullptr);
    CHECK(options.backend.cache_directory == std::filesystem::path{local} / "Mouse" / "cache");
#endif
}
#ifdef _WIN32
namespace {
class LocalAppData {
   public:
    explicit LocalAppData(const wchar_t* value) {
        if (const auto* old = _wgetenv(L"LOCALAPPDATA")) original_ = old;
        if (_wputenv_s(L"LOCALAPPDATA", value) != 0)
            throw std::runtime_error("Cannot set private LOCALAPPDATA test value");
    }
    ~LocalAppData() {
        static_cast<void>(_wputenv_s(L"LOCALAPPDATA", original_ ? original_->c_str() : L""));
    }
    LocalAppData(const LocalAppData&) = delete;
    LocalAppData& operator=(const LocalAppData&) = delete;
    LocalAppData(LocalAppData&&) = delete;
    LocalAppData& operator=(LocalAppData&&) = delete;

   private:
    std::optional<std::wstring> original_;
};
}  // namespace
TEST_CASE("Explicit startup cache overrides do not require LOCALAPPDATA",
          "[uci][options][cache-path]") {
    const LocalAppData missing{L""};
    const auto location =
        (std::filesystem::temp_directory_path() / "mouse explicit cache").string();
    const std::array<std::string_view, 8> args{"--weights",   "T1.onnx", "--backend",   "cpu",
                                               "--precision", "fp32",    "--cache-dir", location};
    mouse::Result<mouse::uci::Options> result = tl::unexpected(std::string{"not called"});
    CHECK_NOTHROW(result = mouse::uci::startup_options(args));
    INFO((result ? std::string{} : result.error()));
    REQUIRE(result);
    CHECK(result->backend.cache_directory == location);
}
TEST_CASE("Missing default cache environment becomes a startup Result error",
          "[uci][options][cache-path]") {
    const LocalAppData missing{L""};
    mouse::Result<mouse::uci::Options> result = tl::unexpected(std::string{"not called"});
    CHECK_NOTHROW(result = mouse::uci::startup_options({}));
    REQUIRE_FALSE(result);
    CHECK(result.error().find("LOCALAPPDATA") != std::string::npos);
}
TEST_CASE("Default cache environment is absolute and Unicode is retained",
          "[uci][options][cache-path]") {
    SECTION("Relative values fail explicitly") {
        const LocalAppData relative{L"relative-cache"};
        const auto result = mouse::nn::default_cache_directory();
        REQUIRE_FALSE(result);
        CHECK(result.error().find("absolute") != std::string::npos);
        CHECK_THROWS(mouse::uci::Options{});
    }
    SECTION("Wide Windows environment survives without lossy conversion") {
        const auto location = std::filesystem::temp_directory_path() / L"mouse-\u00e9";
        const LocalAppData unicode{location.c_str()};
        const auto result = mouse::nn::default_cache_directory();
        REQUIRE(result);
        CHECK(*result == location / "Mouse" / "cache");
        const mouse::uci::Options options;
        CHECK(options.backend.cache_directory == *result);
    }
}
#endif

TEST_CASE("Smart pruning can be disabled without reloading the model or tree",
          "[uci][options][legacy-v1]") {
    const mouse::uci::Options original;
    const auto disabled = apply(original, "SmartPruningFactor", "0");
    REQUIRE(disabled);
    CHECK(disabled->impact == mouse::uci::Reconfigure::None);
    CHECK_FALSE(apply(original, "SmartPruningFactor", "-1"));
    CHECK_FALSE(apply(original, "SmartPruningFactor", "10.1"));
    CHECK_FALSE(apply(original, "SmartPruningMinimumBatches", "10001"));
    CHECK(apply(original, "SmartPruningMinimumBatches", "10000"));
}

TEST_CASE("Default NN cache advertises the native two million entry capacity",
          "[uci][options][cache-capacity-v1]") {
    const mouse::uci::Options options;
    CHECK(options.nncache_entries == 2'000'000);
    const auto lines = mouse::uci::option_lines(options);
    CHECK(std::ranges::find(
              lines, "option name NNCacheSize type spin default 2000000 min 0 max 16000000") !=
          lines.end());
}

TEST_CASE("NN cache keys on the current position by default like lc0",
          "[uci][options][cache-history]") {
    const mouse::uci::Options options;
    // lc0 classic/params.cc kCacheHistoryLengthId: range 0..7, default 0.
    CHECK(options.cache_history_length == 0);
    const auto lines = mouse::uci::option_lines(options);
    CHECK(std::ranges::find(lines,
                            "option name CacheHistoryLength type spin default 0 min 0 max 7") !=
          lines.end());
    const auto changed = apply(options, "CacheHistoryLength", "3");
    REQUIRE(changed);
    CHECK(changed->options.cache_history_length == 3);
    CHECK(changed->impact == mouse::uci::Reconfigure::Search);
}

TEST_CASE("MinibatchSize 0 selects the backend-recommended batch like lc0",
          "[uci][options][minibatch-auto]") {
    const mouse::uci::Options options;
    const auto lines = mouse::uci::option_lines(options);
    // lc0 classic/params.cc kMiniBatchSizeId: 0..1024, default 0.
    CHECK(
        std::ranges::find(lines, "option name MinibatchSize type spin default 0 min 0 max 1024") !=
        lines.end());
    CHECK(std::ranges::find(lines, "option name Threads type spin default 2 min 1 max 128") !=
          lines.end());
    const auto fixed = apply(options, "MinibatchSize", "96");
    REQUIRE(fixed);
    CHECK_FALSE(fixed->options.auto_minibatch);
    CHECK(fixed->options.search.maximum_batch == 96);
    CHECK(fixed->options.backend.batch_buckets == std::vector<std::size_t>{1, 8, 32, 96});
    CHECK(fixed->impact == mouse::uci::Reconfigure::Backend);
    const auto automatic = apply(fixed->options, "MinibatchSize", "0");
    REQUIRE(automatic);
    CHECK(automatic->options.auto_minibatch);
    CHECK(automatic->impact == mouse::uci::Reconfigure::Backend);
    const auto unchanged = apply(options, "MinibatchSize", "0");
    REQUIRE(unchanged);
    CHECK(unchanged->impact == mouse::uci::Reconfigure::None);
    CHECK(mouse::uci::minibatch_buckets(4) == std::vector<std::size_t>{1, 4});
}
