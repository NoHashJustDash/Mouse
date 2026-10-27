// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/uci/options.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <system_error>
#include <utility>

#include "engine/nn/legal_cache.h"
#include "engine/search/commands.h"

namespace mouse::uci {
namespace {
constexpr std::size_t kMiB = std::size_t{1} << 20U;
constexpr std::uint64_t kMaximumMultiPv = 256;
struct Bounds {
    std::uint64_t minimum = 0, maximum = 0;
};
enum class Field : std::uint8_t {
    Weights,
    Backend,
    Precision,
    Threads,
    Batch,
    Cache,
    CacheHistory,
    Ram,
    Cpuct,
    Base,
    Factor,
    Fpu,
    RootFpuStrategy,
    RootFpuValue,
    Temperature,
    MultiPv,
    Wdl,
    LogSearchStats,
    Chess960,
    Ponder,
    Overhead,
    SmartPruning,
    MinimumBatches,
    Directories,
    CacheDirectory,
    Device,
    CpuThreads,
    Workspace,
    Deterministic
};
enum class Kind : std::uint8_t {
    Text,
    Integer,
    Real,
    Boolean,
    Backend,
    Precision,
    RootFpuStrategy
};
struct Definition {
    const char* name = "";
    Field field = Field::Weights;
    Kind kind = Kind::Text;
    Bounds bounds{};
};
constexpr std::array kDefinitions{
    Definition{.name = "WeightsFile", .field = Field::Weights, .kind = Kind::Text},
    Definition{.name = "Backend", .field = Field::Backend, .kind = Kind::Backend},
    Definition{.name = "Precision", .field = Field::Precision, .kind = Kind::Precision},
    Definition{.name = "Threads",
               .field = Field::Threads,
               .kind = Kind::Integer,
               .bounds = {.minimum = 1, .maximum = 128}},
    // lc0 classic/params.cc kMiniBatchSizeId: 0..1024, 0 = backend recommended.
    Definition{.name = "MinibatchSize",
               .field = Field::Batch,
               .kind = Kind::Integer,
               .bounds = {.minimum = 0, .maximum = 1024}},
    Definition{.name = "NNCacheSize",
               .field = Field::Cache,
               .kind = Kind::Integer,
               .bounds = {.minimum = 0, .maximum = 16000000}},
    // lc0 classic/params.cc kCacheHistoryLengthId: 0..7, default 0.
    Definition{.name = "CacheHistoryLength",
               .field = Field::CacheHistory,
               .kind = Kind::Integer,
               .bounds = {.minimum = 0, .maximum = 7}},
    Definition{.name = "RamLimitMb",
               .field = Field::Ram,
               .kind = Kind::Integer,
               .bounds = {.minimum = 1, .maximum = 16384}},
    Definition{.name = "CPuct", .field = Field::Cpuct, .kind = Kind::Real},
    Definition{.name = "CPuctBase", .field = Field::Base, .kind = Kind::Real},
    Definition{.name = "CPuctFactor", .field = Field::Factor, .kind = Kind::Real},
    Definition{.name = "FpuValue", .field = Field::Fpu, .kind = Kind::Real},
    Definition{.name = "FpuStrategyAtRoot",
               .field = Field::RootFpuStrategy,
               .kind = Kind::RootFpuStrategy},
    Definition{.name = "FpuValueAtRoot", .field = Field::RootFpuValue, .kind = Kind::Real},
    Definition{.name = "PolicyTemperature", .field = Field::Temperature, .kind = Kind::Real},
    Definition{.name = "MultiPV",
               .field = Field::MultiPv,
               .kind = Kind::Integer,
               .bounds = {.minimum = 1, .maximum = kMaximumMultiPv}},
    Definition{.name = "LogSearchStats", .field = Field::LogSearchStats, .kind = Kind::Boolean},
    Definition{.name = "UCI_ShowWDL", .field = Field::Wdl, .kind = Kind::Boolean},
    Definition{.name = "UCI_Chess960", .field = Field::Chess960, .kind = Kind::Boolean},
    Definition{.name = "Ponder", .field = Field::Ponder, .kind = Kind::Boolean},
    Definition{.name = "MoveOverheadMs",
               .field = Field::Overhead,
               .kind = Kind::Integer,
               .bounds = {.minimum = 0, .maximum = 60000}},
    Definition{.name = "SmartPruningFactor", .field = Field::SmartPruning, .kind = Kind::Real},
    Definition{.name = "SmartPruningMinimumBatches",
               .field = Field::MinimumBatches,
               .kind = Kind::Integer,
               .bounds = {.minimum = 0, .maximum = 10000}},
    Definition{.name = "GpuDllDirectories", .field = Field::Directories, .kind = Kind::Text},
    Definition{.name = "CacheDirectory", .field = Field::CacheDirectory, .kind = Kind::Text},
    Definition{.name = "Device",
               .field = Field::Device,
               .kind = Kind::Integer,
               .bounds = {.minimum = 0, .maximum = 1023}},
    Definition{.name = "CpuThreads",
               .field = Field::CpuThreads,
               .kind = Kind::Integer,
               .bounds = {.minimum = 1, .maximum = 128}},
    Definition{.name = "WorkspaceMb",
               .field = Field::Workspace,
               .kind = Kind::Integer,
               .bounds = {.minimum = 1, .maximum = 16384}},
    Definition{.name = "Deterministic", .field = Field::Deterministic, .kind = Kind::Boolean}};
char lower(char value) {
    return value >= 'A' && value <= 'Z' ? static_cast<char>(value + ('a' - 'A')) : value;
}
bool same(std::string_view text, const char* expected) {
    return std::ranges::equal(text, std::string_view{expected},
                              [](char left, char right) { return lower(left) == lower(right); });
}
bool control(std::string_view text) {
    return std::ranges::any_of(text,
                               [](unsigned char value) { return value < 32 || value == 127; });
}
Result<std::uint64_t> integer(std::string_view text, const Definition& definition) {
    std::uint64_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || text.size() > 20 || result.ec != std::errc{} ||
        result.ptr != text.data() + text.size() || value < definition.bounds.minimum ||
        value > definition.bounds.maximum)
        return tl::unexpected(std::string{definition.name} + " requires an integer in " +
                              std::to_string(definition.bounds.minimum) + ".." +
                              std::to_string(definition.bounds.maximum));
    return value;
}
Result<double> real(std::string_view text, const Definition& definition) {
    double value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || text.size() > 128 || result.ec != std::errc{} ||
        result.ptr != text.data() + text.size() || !std::isfinite(value))
        return tl::unexpected(std::string{definition.name} + " requires a finite decimal");
    return value;
}
std::filesystem::path path(std::string_view text) {
    std::u8string bytes;
    bytes.reserve(text.size());
    for (const char value : text)
        bytes.push_back(static_cast<char8_t>(static_cast<unsigned char>(value)));
    return std::filesystem::path{bytes};
}
std::string path_text(const std::filesystem::path& value) {
    std::string result;
    for (const char8_t byte : value.generic_u8string()) result.push_back(static_cast<char>(byte));
    if (control(result)) throw std::invalid_argument("UCI path contains a control character");
    return result.empty() ? "<empty>" : result;
}
Result<std::vector<std::filesystem::path>> directories(std::string_view text) {
    std::vector<std::filesystem::path> result;
    if (text.empty() || text == "<empty>") return result;
    for (;;) {
        const auto separator = text.find(';');
        const auto piece = text.substr(0, separator);
        if (piece.empty() || piece.size() > 32767 || result.size() == 16)
            return tl::unexpected(
                std::string{"GpuDllDirectories requires at most16 nonempty absolute paths"});
        auto current = path(piece);
        if (!current.is_absolute() || std::ranges::find(result, current) != result.end())
            return tl::unexpected(
                std::string{"GpuDllDirectories contains a relative or duplicate path"});
        result.push_back(std::move(current));
        if (separator == std::string_view::npos) break;
        text.remove_prefix(separator + 1);
    }
    return result;
}
std::string decimal_text(double value) {
    std::array<char, 64> bytes{};
    const auto converted = std::to_chars(bytes.data(), std::to_address(bytes.end()), value);
    if (converted.ec != std::errc{}) throw std::logic_error("Cannot render UCI decimal default");
    return {bytes.data(), converted.ptr};
}
const char* boolean_text(bool value) { return value ? "true" : "false"; }
std::string default_value(const Options& options, Field field) {
    switch (field) {
        case Field::Weights:
            return path_text(options.backend.model_path);
        case Field::Backend:
            switch (options.backend.kind) {
                case nn::BackendKind::Cpu:
                    return "cpu";
                case nn::BackendKind::Cuda:
                    return "cuda";
                case nn::BackendKind::TensorRt:
                    return "tensorrt";
            }
            throw std::logic_error("Invalid configured backend");
        case Field::Precision:
            switch (options.backend.precision) {
                case nn::ComputePrecision::Float32:
                    return "fp32";
                case nn::ComputePrecision::Float16:
                    return "fp16";
            }
            throw std::logic_error("Invalid configured precision");
        case Field::Threads:
            return std::to_string(options.search.threads);
        case Field::Batch:
            return options.auto_minibatch ? std::string{"0"}
                                          : std::to_string(options.search.maximum_batch);
        case Field::Cache:
            return std::to_string(options.nncache_entries);
        case Field::CacheHistory:
            return std::to_string(options.cache_history_length);
        case Field::Ram:
            return std::to_string(options.search.ram_limit_bytes / kMiB);
        case Field::Cpuct:
            return decimal_text(options.search.selection.cpuct);
        case Field::Base:
            return decimal_text(options.search.selection.cpuct_base);
        case Field::Factor:
            return decimal_text(options.search.selection.cpuct_factor);
        case Field::Fpu:
            return decimal_text(options.search.selection.fpu_reduction);
        case Field::RootFpuStrategy:
            switch (options.search.selection.root_fpu_strategy) {
                case search::RootFpuStrategy::Same:
                    return "same";
                case search::RootFpuStrategy::Reduction:
                    return "reduction";
                case search::RootFpuStrategy::Absolute:
                    return "absolute";
            }
            throw std::logic_error("Invalid configured root FPU strategy");
        case Field::RootFpuValue:
            return decimal_text(options.search.selection.root_fpu);
        case Field::Temperature:
            return decimal_text(options.search.selection.policy_temperature);
        case Field::MultiPv:
            return std::to_string(options.multipv);
        case Field::LogSearchStats:
            return boolean_text(options.log_search_stats);
        case Field::Wdl:
            return boolean_text(options.show_wdl);
        case Field::Chess960:
            return boolean_text(options.chess960);
        case Field::Ponder:
            return boolean_text(options.ponder);
        case Field::Overhead:
            return std::to_string(options.move_overhead_ms);
        case Field::SmartPruning:
            return decimal_text(options.smart_pruning_factor);
        case Field::MinimumBatches:
            return std::to_string(options.minimum_smart_pruning_batches);
        case Field::CacheDirectory:
            return path_text(options.backend.cache_directory);
        case Field::Device:
            return std::to_string(options.backend.device_id);
        case Field::CpuThreads:
            return std::to_string(options.backend.cpu_threads);
        case Field::Workspace:
            return std::to_string(options.backend.workspace_bytes / kMiB);
        case Field::Deterministic:
            return boolean_text(options.search.deterministic);
        case Field::Directories: {
            std::string result;
            for (const auto& directory : options.backend.gpu_dll_directories) {
                if (!result.empty()) result += ';';
                result += path_text(directory);
            }
            return result.empty() ? "<empty>" : result;
        }
    }
    throw std::logic_error("Invalid internal UCI option field");
}
// Shape carried while MinibatchSize is automatic and no backend is resolved.
constexpr std::size_t kUnresolvedMinibatch = 64;
// Search threads: two matched or beat four and eight on the RTX 5070 at the
// automatic minibatch (sweep 2026-10-18); lc0 likewise starts two search
// workers on a single-stream GPU backend (classic/search.cc StartThreads).
constexpr std::size_t kDefaultThreads = 2;
void set_batch(Options& options, std::size_t batch) {
    options.search.maximum_batch = batch;
    options.backend.batch_buckets = minibatch_buckets(batch);
}
}  // namespace
std::vector<std::size_t> minibatch_buckets(std::size_t batch) {
    std::vector<std::size_t> buckets;
    for (const std::size_t value : {std::size_t{1}, std::size_t{8}, std::size_t{32}})
        if (value < batch) buckets.push_back(value);
    buckets.push_back(batch);
    return buckets;
}
Options::Options(std::optional<std::filesystem::path> cache_override) {
    const auto cache = cache_override ? Result<std::filesystem::path>{std::move(*cache_override)}
                                      : nn::default_cache_directory();
    if (!cache) throw std::runtime_error(cache.error());
    backend.cache_directory = *cache;
    search.threads = kDefaultThreads;
    set_batch(*this, kUnresolvedMinibatch);
}
Result<OptionUpdate> apply_option(const Options& original, const SetOptionCommand& command) {
    if (command.name.size() > 256 || control(command.name) ||
        (command.value && (command.value->size() > kMaximumLineBytes || control(*command.value))))
        return tl::unexpected(std::string{"Invalid option text or length"});
    const auto definition = std::ranges::find_if(
        kDefinitions, [&](const Definition& entry) { return same(command.name, entry.name); });
    if (definition == kDefinitions.end()) return tl::unexpected("Unknown option: " + command.name);
    if (!command.value) return tl::unexpected("Missing value for " + command.name);
    const std::string_view value{*command.value};
    std::uint64_t numeric = 0;
    double decimal = 0;
    bool boolean = false;
    if (definition->kind == Kind::Integer) {
        const auto parsed = integer(value, *definition);
        if (!parsed) return tl::unexpected(parsed.error());
        numeric = *parsed;
    } else if (definition->kind == Kind::Real) {
        const auto parsed = real(value, *definition);
        if (!parsed) return tl::unexpected(parsed.error());
        decimal = *parsed;
    } else if (definition->kind == Kind::Boolean) {
        if (!same(value, "true") && !same(value, "false"))
            return tl::unexpected(command.name + " requires true or false");
        boolean = same(value, "true");
    }
    try {
        OptionUpdate result{.options = original};
        auto& options = result.options;
        const auto change = [&](auto& target, auto replacement, Reconfigure impact) {
            if (target != replacement) {
                target = std::move(replacement);
                result.impact = impact;
            }
        };
        switch (definition->field) {
            case Field::Weights:
            case Field::CacheDirectory: {
                if (value.size() > 32767)
                    return tl::unexpected(command.name + " path exceeds32767 bytes");
                auto replacement = path(value == "<empty>" ? std::string_view{} : value);
                if (definition->field == Field::Weights)
                    change(options.backend.model_path, std::move(replacement),
                           Reconfigure::Backend);
                else
                    change(options.backend.cache_directory, std::move(replacement),
                           Reconfigure::Backend);
                break;
            }
            case Field::Backend: {
                nn::BackendKind kind{};
                if (same(value, "cpu"))
                    kind = nn::BackendKind::Cpu;
                else if (same(value, "cuda"))
                    kind = nn::BackendKind::Cuda;
                else if (same(value, "tensorrt"))
                    kind = nn::BackendKind::TensorRt;
                else
                    return tl::unexpected(std::string{"Backend requires cpu, cuda or tensorrt"});
                change(options.backend.kind, kind, Reconfigure::Backend);
                break;
            }
            case Field::Precision: {
                nn::ComputePrecision precision{};
                if (same(value, "fp32"))
                    precision = nn::ComputePrecision::Float32;
                else if (same(value, "fp16"))
                    precision = nn::ComputePrecision::Float16;
                else
                    return tl::unexpected(std::string{"Precision requires fp32 or fp16"});
                change(options.backend.precision, precision, Reconfigure::Backend);
                break;
            }
            case Field::Threads:
                change(options.search.threads, static_cast<std::size_t>(numeric),
                       Reconfigure::Search);
                break;
            case Field::Batch: {
                const auto batch = static_cast<std::size_t>(numeric);
                if (batch == 0) {
                    if (!options.auto_minibatch) {
                        options.auto_minibatch = true;
                        set_batch(options, kUnresolvedMinibatch);
                        result.impact = Reconfigure::Backend;
                    }
                } else if (options.auto_minibatch || options.search.maximum_batch != batch) {
                    options.auto_minibatch = false;
                    set_batch(options, batch);
                    result.impact = Reconfigure::Backend;
                }
                break;
            }
            case Field::Cache:
                change(options.nncache_entries, static_cast<std::size_t>(numeric),
                       Reconfigure::Search);
                break;
            case Field::CacheHistory:
                change(options.cache_history_length, static_cast<std::size_t>(numeric),
                       Reconfigure::Search);
                break;
            case Field::Ram:
                change(options.search.ram_limit_bytes, static_cast<std::size_t>(numeric) * kMiB,
                       Reconfigure::Search);
                break;
            case Field::Cpuct:
                change(options.search.selection.cpuct, decimal, Reconfigure::Search);
                break;
            case Field::Base:
                change(options.search.selection.cpuct_base, decimal, Reconfigure::Search);
                break;
            case Field::Factor:
                change(options.search.selection.cpuct_factor, decimal, Reconfigure::Search);
                break;
            case Field::Fpu:
                change(options.search.selection.fpu_reduction, decimal, Reconfigure::Search);
                break;
            case Field::RootFpuStrategy: {
                search::RootFpuStrategy strategy{};
                if (same(value, "same"))
                    strategy = search::RootFpuStrategy::Same;
                else if (same(value, "reduction"))
                    strategy = search::RootFpuStrategy::Reduction;
                else if (same(value, "absolute"))
                    strategy = search::RootFpuStrategy::Absolute;
                else
                    return tl::unexpected(
                        std::string{"FpuStrategyAtRoot requires same, reduction or absolute"});
                change(options.search.selection.root_fpu_strategy, strategy, Reconfigure::Search);
                break;
            }
            case Field::RootFpuValue:
                change(options.search.selection.root_fpu, decimal, Reconfigure::Search);
                break;
            case Field::Temperature:
                change(options.search.selection.policy_temperature, decimal, Reconfigure::Search);
                break;
            case Field::MultiPv:
                change(options.multipv, static_cast<std::size_t>(numeric), Reconfigure::None);
                break;
            case Field::LogSearchStats:
                change(options.log_search_stats, boolean, Reconfigure::None);
                break;
            case Field::Wdl:
                change(options.show_wdl, boolean, Reconfigure::None);
                break;
            case Field::Chess960:
                change(options.chess960, boolean, Reconfigure::Search);
                break;
            case Field::Ponder:
                change(options.ponder, boolean, Reconfigure::None);
                break;
            case Field::Overhead:
                change(options.move_overhead_ms, numeric, Reconfigure::None);
                break;
            case Field::SmartPruning:
                if (decimal < 0.0 || decimal > 10.0)
                    return tl::unexpected(std::string{"SmartPruningFactor requires0..10"});
                change(options.smart_pruning_factor, decimal, Reconfigure::None);
                break;
            case Field::MinimumBatches:
                change(options.minimum_smart_pruning_batches, numeric, Reconfigure::None);
                break;
            case Field::Directories: {
                auto parsed = directories(value);
                if (!parsed) return tl::unexpected(parsed.error());
                change(options.backend.gpu_dll_directories, std::move(*parsed),
                       Reconfigure::Backend);
                break;
            }
            case Field::Device:
                change(options.backend.device_id, static_cast<int>(numeric), Reconfigure::Backend);
                break;
            case Field::CpuThreads:
                change(options.backend.cpu_threads, static_cast<int>(numeric),
                       Reconfigure::Backend);
                break;
            case Field::Workspace:
                change(options.backend.workspace_bytes, numeric * kMiB, Reconfigure::Backend);
                break;
            case Field::Deterministic:
                change(options.search.deterministic, boolean, Reconfigure::Backend);
                change(options.backend.deterministic_compute, boolean, Reconfigure::Backend);
                break;
        }
        const auto valid = search::validate_selection_parameters(options.search.selection);
        if (!valid) return tl::unexpected(valid.error());
        if (options.search.deterministic && options.backend.cpu_threads != 1)
            return tl::unexpected(std::string{"Deterministic requires CpuThreads1"});
        return result;
    } catch (const std::system_error& error) {
        return tl::unexpected(std::string{"Invalid option path: "} + error.what());
    }
}
Result<Options> startup_options(std::span<const std::string_view> arguments) {
    if (arguments.empty()) {
        const auto cache = nn::default_cache_directory();
        if (!cache) return tl::unexpected(cache.error());
        return Options{*cache};
    }
    constexpr std::array pairs{"--weights",           "--backend",
                               "--precision",         "--cache-dir",
                               "--gpu-dll-dir",       "--device",
                               "--cpu-threads",       "--workspace-mb",
                               "--nncache-mb",        "--buckets",
                               "--threads",           "--batch",
                               "--ram-limit-mb",      "--cpuct",
                               "--cpuct-base",        "--cpuct-factor",
                               "--fpu-value",         "--fpu-strategy-at-root",
                               "--fpu-value-at-root", "--policy-temperature"};
    bool explicit_cache = false, explicit_disk_cache = false;
    bool explicit_batch = false, explicit_threads = false;
    for (std::size_t index = 0; index < arguments.size(); ++index) {
        const auto option = arguments[index];
        if (option == "--deterministic" || option == "--chess960") continue;
        if (std::ranges::find(pairs, option) == pairs.end())
            return tl::unexpected("Not a UCI startup option: " + std::string{option});
        if (index + 1 == arguments.size())
            return tl::unexpected("Missing value after " + std::string{option});
        const auto value = arguments[++index];
        if (value.size() > kMaximumLineBytes || control(value))
            return tl::unexpected(std::string{"Invalid UCI startup option text"});
        explicit_cache |= option == "--nncache-mb";
        explicit_batch |= option == "--batch" || option == "--buckets";
        explicit_threads |= option == "--threads";
        explicit_disk_cache |= option == "--cache-dir";
    }
    auto parsed = search::parse_search_command(arguments);
    if (!parsed) return tl::unexpected(parsed.error());
    const auto cache = explicit_disk_cache
                           ? Result<std::filesystem::path>{parsed->neural.backend.cache_directory}
                           : nn::default_cache_directory();
    if (!cache) return tl::unexpected(cache.error());
    Options result{*cache};
    result.backend = std::move(parsed->neural.backend);
    result.search = parsed->config;
    result.chess960 = parsed->chess960;
    // Flags not given keep the UCI defaults rather than the CLI search defaults.
    result.auto_minibatch = !explicit_batch;
    if (!explicit_batch) set_batch(result, kUnresolvedMinibatch);
    if (!explicit_threads) result.search.threads = kDefaultThreads;
    if (explicit_cache) {
        // Legacy CLI MiB flag retains its meaning; UCI NNCacheSize uses entries.
        // Production cache namespaces are at most a SHA256 string (64 bytes).
        const auto capacity = nn::LegalCache::capacity_for_budget(
            {.bytes = parsed->neural.cache_bytes, .identity_bytes = 64});
        if (!capacity) return tl::unexpected(capacity.error());
        result.nncache_entries = *capacity;
    }
    if (!explicit_disk_cache) result.backend.cache_directory = *cache;
    return result;
}
std::vector<std::string> option_lines(const Options& options) {
    std::vector<std::string> lines;
    lines.reserve(kDefinitions.size());
    for (const auto& definition : kDefinitions) {
        std::string line = "option name " + std::string{definition.name} + " type ";
        switch (definition.kind) {
            case Kind::Integer:
                line += "spin";
                break;
            case Kind::Boolean:
                line += "check";
                break;
            case Kind::Backend:
            case Kind::Precision:
            case Kind::RootFpuStrategy:
                line += "combo";
                break;
            case Kind::Text:
            case Kind::Real:
                line += "string";
                break;
        }
        line += " default " + default_value(options, definition.field);
        if (definition.kind == Kind::Integer)
            line += " min " + std::to_string(definition.bounds.minimum) + " max " +
                    std::to_string(definition.bounds.maximum);
        else if (definition.kind == Kind::Backend)
            line += " var cpu var cuda var tensorrt";
        else if (definition.kind == Kind::Precision)
            line += " var fp32 var fp16";
        else if (definition.kind == Kind::RootFpuStrategy)
            line += " var same var reduction var absolute";
        lines.push_back(std::move(line));
    }
    return lines;
}
}  // namespace mouse::uci
