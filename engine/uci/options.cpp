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
    MovesLeftMaxEffect,
    MovesLeftThreshold,
    MovesLeftSlope,
    MovesLeftConstantFactor,
    MovesLeftScaledFactor,
    MovesLeftQuadraticFactor,
    DrawRateTarget,
    DrawRateReference,
    MaxS,
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
    Deterministic,
    MaxCollisionEvents,
    MaxCollisionVisits,
    CollisionScalingStart,
    CollisionScalingEnd,
    CollisionScalingPower,
    OutOfOrderEval,
    OutOfOrderFactor,
    MaxPrefetch,
    IdlingMinimumWork,
    ThreadIdlingThreshold,
    MinimumProcessingWork,
    MinimumPerTaskProcessing,
    MinimumPickingWork,
    MinimumRemainingPickingWork,
    SolidTreeThreshold,
    SearchTemperature,
    TempDecayMoves,
    TempDecayDelayMoves,
    TempCutoffMove,
    TempEndgame,
    TempValueCutoff,
    TempVisitOffset,
    NoiseEpsilon,
    NoiseAlpha,
    TaskWorkers,
    MaxConcurrentSearchers,
    SearchSpinBackoff,
    TwoFoldDraws,
    StickyEndgames,
    FpuStrategy,
    HistoryFill,
    DrawScore,
    NpsLimit,
    Contempt,
    ContemptMode,
    ContemptMax,
    Attenuation,
    BookExitBias,
    CalibrationElo,
    EvalObjectivity,
    Opponent,
    RatingAdvantage,
    ScoreType,
    PerPvCounters,
    ShowMovesLeft,
    VerboseMoveStats,
    LogFile,
    ConfigFile,
    TimeManager,
    BackendOptions,
    SyzygyPath,
    SyzygyFastPlay
};
enum class Kind : std::uint8_t {
    Text,
    Integer,
    SignedInteger,
    Real,
    Boolean,
    Backend,
    Precision,
    RootFpuStrategy,
    Choice
};
struct Definition {
    const char* name = "";
    Field field = Field::Weights;
    Kind kind = Kind::Text;
    Bounds bounds{};
    // Kind::Choice: the allowed values, separated by '|'.
    const char* choices = "";
    std::int64_t signed_minimum = 0;
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
    // lc0 classic/params.cc: MovesLeft* utility and WDLDrawRate*/WDLMaxS rescaling,
    // ranges checked by search validation.
    Definition{
        .name = "MovesLeftMaxEffect", .field = Field::MovesLeftMaxEffect, .kind = Kind::Real},
    Definition{
        .name = "MovesLeftThreshold", .field = Field::MovesLeftThreshold, .kind = Kind::Real},
    Definition{.name = "MovesLeftSlope", .field = Field::MovesLeftSlope, .kind = Kind::Real},
    Definition{.name = "MovesLeftConstantFactor",
               .field = Field::MovesLeftConstantFactor,
               .kind = Kind::Real},
    Definition{
        .name = "MovesLeftScaledFactor", .field = Field::MovesLeftScaledFactor, .kind = Kind::Real},
    Definition{.name = "MovesLeftQuadraticFactor",
               .field = Field::MovesLeftQuadraticFactor,
               .kind = Kind::Real},
    Definition{.name = "WDLDrawRateTarget", .field = Field::DrawRateTarget, .kind = Kind::Real},
    Definition{
        .name = "WDLDrawRateReference", .field = Field::DrawRateReference, .kind = Kind::Real},
    Definition{.name = "WDLMaxS", .field = Field::MaxS, .kind = Kind::Real},
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
    Definition{.name = "Deterministic", .field = Field::Deterministic, .kind = Kind::Boolean},
    // lc0 fd71a2d classic/params.cc Populate: names, kinds and ranges.
    Definition{.name = "MaxCollisionEvents",
               .field = Field::MaxCollisionEvents,
               .kind = Kind::Integer,
               .bounds = {.minimum = 1, .maximum = 65536}},
    Definition{.name = "MaxCollisionVisits",
               .field = Field::MaxCollisionVisits,
               .kind = Kind::Integer,
               .bounds = {.minimum = 1, .maximum = 100000000}},
    Definition{.name = "MaxCollisionVisitsScalingStart",
               .field = Field::CollisionScalingStart,
               .kind = Kind::Integer,
               .bounds = {.minimum = 1, .maximum = 100000}},
    Definition{.name = "MaxCollisionVisitsScalingEnd",
               .field = Field::CollisionScalingEnd,
               .kind = Kind::Integer,
               .bounds = {.minimum = 0, .maximum = 100000000}},
    Definition{.name = "MaxCollisionVisitsScalingPower",
               .field = Field::CollisionScalingPower,
               .kind = Kind::Real},
    Definition{.name = "OutOfOrderEval", .field = Field::OutOfOrderEval, .kind = Kind::Boolean},
    Definition{.name = "MaxOutOfOrderEvalsFactor",
               .field = Field::OutOfOrderFactor,
               .kind = Kind::Real},
    Definition{.name = "MaxPrefetch",
               .field = Field::MaxPrefetch,
               .kind = Kind::Integer,
               .bounds = {.minimum = 0, .maximum = 1024}},
    Definition{.name = "IdlingMinimumWork",
               .field = Field::IdlingMinimumWork,
               .kind = Kind::Integer,
               .bounds = {.minimum = 0, .maximum = 10000}},
    Definition{.name = "ThreadIdlingThreshold",
               .field = Field::ThreadIdlingThreshold,
               .kind = Kind::Integer,
               .bounds = {.minimum = 0, .maximum = 128}},
    Definition{.name = "MinimumProcessingWork",
               .field = Field::MinimumProcessingWork,
               .kind = Kind::Integer,
               .bounds = {.minimum = 2, .maximum = 100000}},
    Definition{.name = "MinimumPerTaskProcessing",
               .field = Field::MinimumPerTaskProcessing,
               .kind = Kind::Integer,
               .bounds = {.minimum = 1, .maximum = 100000}},
    Definition{.name = "MinimumPickingWork",
               .field = Field::MinimumPickingWork,
               .kind = Kind::Integer,
               .bounds = {.minimum = 1, .maximum = 100000}},
    Definition{.name = "MinimumRemainingPickingWork",
               .field = Field::MinimumRemainingPickingWork,
               .kind = Kind::Integer,
               .bounds = {.minimum = 0, .maximum = 100000}},
    // Lc0 classic/params.cc kSolidTreeThresholdId.
    Definition{.name = "SolidTreeThreshold",
               .field = Field::SolidTreeThreshold,
               .kind = Kind::Integer,
               .bounds = {.minimum = 1, .maximum = 2000000000}},
    // Lc0 classic/params.cc kTemperatureId and its companions, then the Dirichlet noise.
    Definition{.name = "Temperature", .field = Field::SearchTemperature, .kind = Kind::Real},
    Definition{.name = "TempDecayMoves",
               .field = Field::TempDecayMoves,
               .kind = Kind::Integer,
               .bounds = {.minimum = 0, .maximum = 640}},
    Definition{.name = "TempDecayDelayMoves",
               .field = Field::TempDecayDelayMoves,
               .kind = Kind::Integer,
               .bounds = {.minimum = 0, .maximum = 100}},
    Definition{.name = "TempCutoffMove",
               .field = Field::TempCutoffMove,
               .kind = Kind::Integer,
               .bounds = {.minimum = 0, .maximum = 1000}},
    Definition{.name = "TempEndgame", .field = Field::TempEndgame, .kind = Kind::Real},
    Definition{.name = "TempValueCutoff", .field = Field::TempValueCutoff, .kind = Kind::Real},
    Definition{.name = "TempVisitOffset", .field = Field::TempVisitOffset, .kind = Kind::Real},
    Definition{.name = "DirichletNoiseEpsilon", .field = Field::NoiseEpsilon, .kind = Kind::Real},
    Definition{.name = "DirichletNoiseAlpha", .field = Field::NoiseAlpha, .kind = Kind::Real},
    Definition{.name = "TaskWorkers",
               .field = Field::TaskWorkers,
               .kind = Kind::SignedInteger,
               .bounds = {.minimum = 0, .maximum = 128},
               .signed_minimum = -1},
    Definition{.name = "MaxConcurrentSearchers",
               .field = Field::MaxConcurrentSearchers,
               .kind = Kind::Integer,
               .bounds = {.minimum = 0, .maximum = 128}},
    Definition{.name = "SearchSpinBackoff",
               .field = Field::SearchSpinBackoff,
               .kind = Kind::Boolean},
    Definition{.name = "TwoFoldDraws", .field = Field::TwoFoldDraws, .kind = Kind::Boolean},
    Definition{.name = "StickyEndgames", .field = Field::StickyEndgames, .kind = Kind::Boolean},
    Definition{.name = "FpuStrategy",
               .field = Field::FpuStrategy,
               .kind = Kind::Choice,
               .choices = "reduction|absolute"},
    Definition{.name = "HistoryFill",
               .field = Field::HistoryFill,
               .kind = Kind::Choice,
               .choices = "no|fen_only|always"},
    Definition{.name = "DrawScore", .field = Field::DrawScore, .kind = Kind::Real},
    Definition{.name = "NodesPerSecondLimit", .field = Field::NpsLimit, .kind = Kind::Real},
    Definition{.name = "Contempt", .field = Field::Contempt, .kind = Kind::Text},
    Definition{.name = "ContemptMode",
               .field = Field::ContemptMode,
               .kind = Kind::Choice,
               .choices = "play|white_side_analysis|black_side_analysis|disable"},
    Definition{.name = "ContemptMaxValue", .field = Field::ContemptMax, .kind = Kind::Real},
    Definition{.name = "WDLContemptAttenuation", .field = Field::Attenuation, .kind = Kind::Real},
    Definition{.name = "WDLBookExitBias", .field = Field::BookExitBias, .kind = Kind::Real},
    Definition{.name = "WDLCalibrationElo", .field = Field::CalibrationElo, .kind = Kind::Real},
    Definition{.name = "WDLEvalObjectivity", .field = Field::EvalObjectivity, .kind = Kind::Real},
    Definition{.name = "UCI_Opponent", .field = Field::Opponent, .kind = Kind::Text},
    Definition{.name = "UCI_RatingAdv", .field = Field::RatingAdvantage, .kind = Kind::Real},
    Definition{.name = "ScoreType",
               .field = Field::ScoreType,
               .kind = Kind::Choice,
               .choices = "centipawn|centipawn_with_drawscore|centipawn_2019|centipawn_2018|"
                          "win_percentage|Q|W-L|WDL_mu"},
    Definition{.name = "PerPVCounters", .field = Field::PerPvCounters, .kind = Kind::Boolean},
    Definition{.name = "UCI_ShowMovesLeft", .field = Field::ShowMovesLeft, .kind = Kind::Boolean},
    Definition{.name = "VerboseMoveStats",
               .field = Field::VerboseMoveStats,
               .kind = Kind::Boolean},
    Definition{.name = "LogFile", .field = Field::LogFile, .kind = Kind::Text},
    Definition{.name = "ConfigFile", .field = Field::ConfigFile, .kind = Kind::Text},
    Definition{.name = "TimeManager", .field = Field::TimeManager, .kind = Kind::Text},
    Definition{.name = "BackendOptions", .field = Field::BackendOptions, .kind = Kind::Text},
    Definition{.name = "SyzygyPath", .field = Field::SyzygyPath, .kind = Kind::Text},
    Definition{.name = "SyzygyFastPlay", .field = Field::SyzygyFastPlay, .kind = Kind::Boolean}};
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
                case nn::BackendKind::DirectMl:
                    return "directml";
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
        case Field::MovesLeftMaxEffect:
            return decimal_text(options.search.moves_left.max_effect);
        case Field::MovesLeftThreshold:
            return decimal_text(options.search.moves_left.threshold);
        case Field::MovesLeftSlope:
            return decimal_text(options.search.moves_left.slope);
        case Field::MovesLeftConstantFactor:
            return decimal_text(options.search.moves_left.constant_factor);
        case Field::MovesLeftScaledFactor:
            return decimal_text(options.search.moves_left.scaled_factor);
        case Field::MovesLeftQuadraticFactor:
            return decimal_text(options.search.moves_left.quadratic_factor);
        case Field::DrawRateTarget:
            return decimal_text(options.search.draw_rate.target);
        case Field::DrawRateReference:
            return decimal_text(options.search.draw_rate.reference);
        case Field::MaxS:
            return decimal_text(options.search.draw_rate.max_s);
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
        case Field::MaxCollisionEvents:
            return std::to_string(options.search.gather.max_collision_events);
        case Field::MaxCollisionVisits:
            return std::to_string(options.search.gather.max_collision_visits);
        case Field::CollisionScalingStart:
            return std::to_string(options.search.gather.collision_scaling_start);
        case Field::CollisionScalingEnd:
            return std::to_string(options.search.gather.collision_scaling_end);
        case Field::CollisionScalingPower:
            return decimal_text(options.search.gather.collision_scaling_power);
        case Field::OutOfOrderEval:
            return boolean_text(options.search.gather.out_of_order_eval);
        case Field::OutOfOrderFactor:
            return decimal_text(options.search.gather.max_out_of_order_factor);
        case Field::MaxPrefetch:
            return std::to_string(options.search.gather.max_prefetch);
        case Field::IdlingMinimumWork:
            return std::to_string(options.search.gather.idling_minimum_work);
        case Field::ThreadIdlingThreshold:
            return std::to_string(options.search.gather.thread_idling_threshold);
        case Field::MinimumProcessingWork:
            return std::to_string(options.search.gather.minimum_processing_work);
        case Field::MinimumPerTaskProcessing:
            return std::to_string(options.search.gather.minimum_per_task_processing);
        case Field::MinimumPickingWork:
            return std::to_string(options.search.gather.minimum_picking_work);
        case Field::MinimumRemainingPickingWork:
            return std::to_string(options.search.gather.minimum_remaining_picking_work);
        case Field::SolidTreeThreshold:
            return std::to_string(options.search.solid_tree_threshold);
        case Field::SearchTemperature:
            return decimal_text(options.search.temperature.temperature);
        case Field::TempDecayMoves:
            return std::to_string(options.search.temperature.decay_moves);
        case Field::TempDecayDelayMoves:
            return std::to_string(options.search.temperature.decay_delay_moves);
        case Field::TempCutoffMove:
            return std::to_string(options.search.temperature.cutoff_move);
        case Field::TempEndgame:
            return decimal_text(options.search.temperature.endgame);
        case Field::TempValueCutoff:
            return decimal_text(options.search.temperature.value_cutoff);
        case Field::TempVisitOffset:
            return decimal_text(options.search.temperature.visit_offset);
        case Field::NoiseEpsilon:
            return decimal_text(options.search.noise_epsilon);
        case Field::NoiseAlpha:
            return decimal_text(options.search.noise_alpha);
        case Field::TaskWorkers:
            return std::to_string(options.task_workers);
        case Field::MaxConcurrentSearchers:
            return std::to_string(options.max_concurrent_searchers);
        case Field::SearchSpinBackoff:
            return boolean_text(options.search_spin_backoff);
        case Field::TwoFoldDraws:
            return boolean_text(options.search.two_fold_draws);
        case Field::StickyEndgames:
            return boolean_text(options.search.terminal_proof);
        case Field::FpuStrategy:
            return options.search.selection.fpu_absolute ? "absolute" : "reduction";
        case Field::HistoryFill:
            switch (options.search.history_fill) {
                case nn::HistoryFill::No:
                    return "no";
                case nn::HistoryFill::FenOnly:
                    return "fen_only";
                case nn::HistoryFill::Always:
                    return "always";
            }
            throw std::logic_error("Invalid configured history fill");
        case Field::DrawScore:
            return decimal_text(options.search.draw_score);
        case Field::NpsLimit:
            return decimal_text(options.search.nps_limit);
        case Field::Contempt:
            return options.contempt.empty() ? "<empty>" : options.contempt;
        case Field::ContemptMode:
            switch (options.contempt_mode) {
                case ContemptMode::Play:
                    return "play";
                case ContemptMode::WhiteSideAnalysis:
                    return "white_side_analysis";
                case ContemptMode::BlackSideAnalysis:
                    return "black_side_analysis";
                case ContemptMode::Disable:
                    return "disable";
            }
            throw std::logic_error("Invalid configured contempt mode");
        case Field::ContemptMax:
            return decimal_text(options.search.draw_rate.contempt_max);
        case Field::Attenuation:
            return decimal_text(options.search.draw_rate.attenuation);
        case Field::BookExitBias:
            return decimal_text(options.search.draw_rate.book_exit_bias);
        case Field::CalibrationElo:
            return decimal_text(options.search.draw_rate.calibration_elo);
        case Field::EvalObjectivity:
            return decimal_text(options.search.draw_rate.eval_objectivity);
        case Field::Opponent:
            return options.uci_opponent.empty() ? "<empty>" : options.uci_opponent;
        case Field::RatingAdvantage:
            return decimal_text(options.uci_rating_advantage);
        case Field::ScoreType:
            return std::string{score_type_name(options.score_type)};
        case Field::PerPvCounters:
            return boolean_text(options.per_pv_counters);
        case Field::ShowMovesLeft:
            return boolean_text(options.show_moves_left);
        case Field::VerboseMoveStats:
            return boolean_text(options.verbose_move_stats);
        case Field::LogFile:
            return options.log_file.empty() ? "<empty>" : options.log_file;
        case Field::ConfigFile:
            return options.config_file.empty() ? "<empty>" : options.config_file;
        case Field::TimeManager:
            return options.time_manager_text;
        case Field::BackendOptions:
            return options.backend_options.empty() ? "<empty>" : options.backend_options;
        case Field::SyzygyPath:
            return options.syzygy_path.empty() ? "<empty>" : options.syzygy_path;
        case Field::SyzygyFastPlay:
            return boolean_text(options.syzygy_fast_play);
    }
    throw std::logic_error("Invalid internal UCI option field");
}
bool in_choices(std::string_view value, std::string_view choices) {
    while (!choices.empty()) {
        const auto bar = choices.find('|');
        if (choices.substr(0, bar) == value) return true;
        choices = bar == std::string_view::npos ? std::string_view{} : choices.substr(bar + 1);
    }
    return false;
}
// Lc0 BackendOptions, "key=value" pairs joined by commas. Mouse has no options of its own beyond
// the ones below, and any other key is an error, not something silently ignored.
Result<void> apply_backend_options(Options& options, std::string_view text, Reconfigure& impact) {
    while (!text.empty()) {
        const auto comma = text.find(',');
        const auto entry = text.substr(0, comma);
        text = comma == std::string_view::npos ? std::string_view{} : text.substr(comma + 1);
        const auto equals = entry.find('=');
        if (equals == std::string_view::npos)
            return tl::unexpected("BackendOptions entry needs key=value: " + std::string{entry});
        const auto key = entry.substr(0, equals);
        const auto value = entry.substr(equals + 1);
        const auto number = [&]() -> Result<int> {
            int parsed = 0;
            const auto end = std::from_chars(value.data(), value.data() + value.size(), parsed);
            if (value.empty() || end.ec != std::errc{} ||
                end.ptr != value.data() + value.size() || parsed < 0 || parsed > 1023)
                return tl::unexpected("BackendOptions " + std::string{key} +
                                      " requires an integer in 0..1023");
            return parsed;
        };
        if (key == "gpu" || key == "device") {
            auto parsed = number();
            if (!parsed) return tl::unexpected(parsed.error());
            options.backend.device_id = *parsed;
        } else if (key == "threads") {
            auto parsed = number();
            if (!parsed) return tl::unexpected(parsed.error());
            if (*parsed == 0) return tl::unexpected(std::string{"BackendOptions threads requires 1 or more"});
            options.backend.cpu_threads = *parsed;
        } else if (key == "precision") {
            if (value == "fp32")
                options.backend.precision = nn::ComputePrecision::Float32;
            else if (value == "fp16")
                options.backend.precision = nn::ComputePrecision::Float16;
            else
                return tl::unexpected(std::string{"BackendOptions precision requires fp32 or fp16"});
        } else
            return tl::unexpected("Unknown BackendOptions key: " + std::string{key});
        impact = Reconfigure::Backend;
    }
    return {};
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
    const auto definition = std::ranges::find_if(kDefinitions, [&](const Definition& entry) {
        if (same(command.name, entry.name)) return true;
        // Lc0's long flags drop UCI_ and any dashes; only a config file passes those.
        std::string_view name{entry.name};
        if (name.starts_with("UCI_")) name.remove_prefix(4);
        return std::ranges::equal(std::string_view{command.name}, name,
                                  [](char a, char b) { return lower(a) == lower(b); });
    });
    if (definition == kDefinitions.end()) return tl::unexpected("Unknown option: " + command.name);
    if (!command.value) return tl::unexpected("Missing value for " + command.name);
    const std::string_view value{*command.value};
    std::uint64_t numeric = 0;
    std::int64_t signed_numeric = 0;
    double decimal = 0;
    bool boolean = false;
    // "<empty>" is how an empty string option is written in UCI.
    const std::string_view text = value == "<empty>" ? std::string_view{} : value;
    if (definition->kind == Kind::Integer) {
        const auto parsed = integer(value, *definition);
        if (!parsed) return tl::unexpected(parsed.error());
        numeric = *parsed;
    } else if (definition->kind == Kind::SignedInteger) {
        const auto end = std::from_chars(value.data(), value.data() + value.size(), signed_numeric);
        if (value.empty() || end.ec != std::errc{} || end.ptr != value.data() + value.size() ||
            signed_numeric < definition->signed_minimum ||
            signed_numeric > static_cast<std::int64_t>(definition->bounds.maximum))
            return tl::unexpected(std::string{definition->name} + " requires an integer in " +
                                  std::to_string(definition->signed_minimum) + ".." +
                                  std::to_string(definition->bounds.maximum));
    } else if (definition->kind == Kind::Choice) {
        if (!in_choices(value, definition->choices))
            return tl::unexpected(command.name + " requires one of: " + definition->choices);
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
                // Each build offers the providers its ONNX Runtime package contains.
                if (same(value, "cpu")) kind = nn::BackendKind::Cpu;
#if MOUSE_DIRECTML
                else if (same(value, "directml"))
                    kind = nn::BackendKind::DirectMl;
                else
                    return tl::unexpected(std::string{"Backend requires cpu or directml"});
#else
                else if (same(value, "cuda"))
                    kind = nn::BackendKind::Cuda;
                else if (same(value, "tensorrt"))
                    kind = nn::BackendKind::TensorRt;
                else
                    return tl::unexpected(std::string{"Backend requires cpu, cuda or tensorrt"});
#endif
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
            case Field::MovesLeftMaxEffect:
                change(options.search.moves_left.max_effect, decimal, Reconfigure::Search);
                break;
            case Field::MovesLeftThreshold:
                change(options.search.moves_left.threshold, decimal, Reconfigure::Search);
                break;
            case Field::MovesLeftSlope:
                change(options.search.moves_left.slope, decimal, Reconfigure::Search);
                break;
            case Field::MovesLeftConstantFactor:
                change(options.search.moves_left.constant_factor, decimal, Reconfigure::Search);
                break;
            case Field::MovesLeftScaledFactor:
                change(options.search.moves_left.scaled_factor, decimal, Reconfigure::Search);
                break;
            case Field::MovesLeftQuadraticFactor:
                change(options.search.moves_left.quadratic_factor, decimal, Reconfigure::Search);
                break;
            case Field::DrawRateTarget:
                change(options.search.draw_rate.target, decimal, Reconfigure::Search);
                break;
            case Field::DrawRateReference:
                change(options.search.draw_rate.reference, decimal, Reconfigure::Search);
                break;
            case Field::MaxS:
                change(options.search.draw_rate.max_s, decimal, Reconfigure::Search);
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
            case Field::MaxCollisionEvents:
                change(options.search.gather.max_collision_events,
                       static_cast<std::uint32_t>(numeric), Reconfigure::Search);
                break;
            case Field::MaxCollisionVisits:
                change(options.search.gather.max_collision_visits,
                       static_cast<std::uint32_t>(numeric), Reconfigure::Search);
                break;
            case Field::CollisionScalingStart:
                change(options.search.gather.collision_scaling_start,
                       static_cast<std::uint32_t>(numeric), Reconfigure::Search);
                break;
            case Field::CollisionScalingEnd:
                change(options.search.gather.collision_scaling_end,
                       static_cast<std::uint32_t>(numeric), Reconfigure::Search);
                break;
            case Field::CollisionScalingPower:
                change(options.search.gather.collision_scaling_power, decimal, Reconfigure::Search);
                break;
            case Field::OutOfOrderEval:
                change(options.search.gather.out_of_order_eval, boolean, Reconfigure::Search);
                break;
            case Field::OutOfOrderFactor:
                change(options.search.gather.max_out_of_order_factor, decimal, Reconfigure::Search);
                break;
            case Field::MaxPrefetch:
                change(options.search.gather.max_prefetch, static_cast<std::uint32_t>(numeric),
                       Reconfigure::Search);
                break;
            case Field::IdlingMinimumWork:
                change(options.search.gather.idling_minimum_work,
                       static_cast<std::uint32_t>(numeric), Reconfigure::Search);
                break;
            case Field::ThreadIdlingThreshold:
                change(options.search.gather.thread_idling_threshold,
                       static_cast<std::uint32_t>(numeric), Reconfigure::Search);
                break;
            case Field::MinimumProcessingWork:
                change(options.search.gather.minimum_processing_work,
                       static_cast<std::uint32_t>(numeric), Reconfigure::Search);
                break;
            case Field::MinimumPerTaskProcessing:
                change(options.search.gather.minimum_per_task_processing,
                       static_cast<std::uint32_t>(numeric), Reconfigure::Search);
                break;
            case Field::MinimumPickingWork:
                change(options.search.gather.minimum_picking_work,
                       static_cast<std::uint32_t>(numeric), Reconfigure::Search);
                break;
            case Field::MinimumRemainingPickingWork:
                change(options.search.gather.minimum_remaining_picking_work,
                       static_cast<std::uint32_t>(numeric), Reconfigure::Search);
                break;
            case Field::SolidTreeThreshold:
                change(options.search.solid_tree_threshold, static_cast<std::uint32_t>(numeric),
                       Reconfigure::Search);
                break;
            case Field::SearchTemperature:
                change(options.search.temperature.temperature, decimal, Reconfigure::Search);
                break;
            case Field::TempDecayMoves:
                change(options.search.temperature.decay_moves, static_cast<std::uint32_t>(numeric),
                       Reconfigure::Search);
                break;
            case Field::TempDecayDelayMoves:
                change(options.search.temperature.decay_delay_moves,
                       static_cast<std::uint32_t>(numeric), Reconfigure::Search);
                break;
            case Field::TempCutoffMove:
                change(options.search.temperature.cutoff_move, static_cast<std::uint32_t>(numeric),
                       Reconfigure::Search);
                break;
            case Field::TempEndgame:
                change(options.search.temperature.endgame, decimal, Reconfigure::Search);
                break;
            case Field::TempValueCutoff:
                change(options.search.temperature.value_cutoff, decimal, Reconfigure::Search);
                break;
            case Field::TempVisitOffset:
                change(options.search.temperature.visit_offset, decimal, Reconfigure::Search);
                break;
            case Field::NoiseEpsilon:
                change(options.search.noise_epsilon, decimal, Reconfigure::Search);
                break;
            case Field::NoiseAlpha:
                change(options.search.noise_alpha, decimal, Reconfigure::Search);
                break;
            case Field::TaskWorkers:
                change(options.task_workers, signed_numeric, Reconfigure::Search);
                break;
            case Field::MaxConcurrentSearchers:
                change(options.max_concurrent_searchers, numeric, Reconfigure::None);
                break;
            case Field::SearchSpinBackoff:
                change(options.search_spin_backoff, boolean, Reconfigure::None);
                break;
            case Field::TwoFoldDraws:
                change(options.search.two_fold_draws, boolean, Reconfigure::Search);
                break;
            case Field::StickyEndgames:
                change(options.search.terminal_proof, boolean, Reconfigure::Search);
                break;
            case Field::FpuStrategy:
                change(options.search.selection.fpu_absolute, value == "absolute",
                       Reconfigure::Search);
                break;
            case Field::HistoryFill:
                change(options.search.history_fill,
                       value == "no"     ? nn::HistoryFill::No
                       : value == "always" ? nn::HistoryFill::Always
                                           : nn::HistoryFill::FenOnly,
                       Reconfigure::Search);
                break;
            case Field::DrawScore:
                change(options.search.draw_score, decimal, Reconfigure::Search);
                break;
            case Field::NpsLimit:
                change(options.search.nps_limit, decimal, Reconfigure::Search);
                break;
            case Field::Contempt:
                change(options.contempt, std::string{text}, Reconfigure::Search);
                break;
            case Field::ContemptMode:
                change(options.contempt_mode,
                       value == "play"                  ? ContemptMode::Play
                       : value == "white_side_analysis" ? ContemptMode::WhiteSideAnalysis
                       : value == "black_side_analysis" ? ContemptMode::BlackSideAnalysis
                                                        : ContemptMode::Disable,
                       Reconfigure::None);
                break;
            case Field::ContemptMax:
                change(options.search.draw_rate.contempt_max, decimal, Reconfigure::Search);
                break;
            case Field::Attenuation:
                change(options.search.draw_rate.attenuation, decimal, Reconfigure::Search);
                break;
            case Field::BookExitBias:
                change(options.search.draw_rate.book_exit_bias, decimal, Reconfigure::Search);
                break;
            case Field::CalibrationElo:
                change(options.search.draw_rate.calibration_elo, decimal, Reconfigure::Search);
                break;
            case Field::EvalObjectivity:
                change(options.search.draw_rate.eval_objectivity, decimal, Reconfigure::Search);
                break;
            case Field::Opponent:
                change(options.uci_opponent, std::string{text}, Reconfigure::Search);
                break;
            case Field::RatingAdvantage:
                if (decimal < -10000.0 || decimal > 10000.0)
                    return tl::unexpected(std::string{"UCI_RatingAdv requires -10000..10000"});
                change(options.uci_rating_advantage, decimal, Reconfigure::Search);
                break;
            case Field::ScoreType:
                change(options.score_type, *parse_score_type(value), Reconfigure::None);
                break;
            case Field::PerPvCounters:
                change(options.per_pv_counters, boolean, Reconfigure::None);
                break;
            case Field::ShowMovesLeft:
                change(options.show_moves_left, boolean, Reconfigure::None);
                break;
            case Field::VerboseMoveStats:
                change(options.verbose_move_stats, boolean, Reconfigure::None);
                break;
            case Field::LogFile:
                change(options.log_file, std::string{text}, Reconfigure::None);
                break;
            case Field::ConfigFile:
                // Read once at startup, before the first UCI command (lc0 utils/configfile.cc).
                change(options.config_file, std::string{text}, Reconfigure::None);
                break;
            case Field::TimeManager: {
                auto parsed = parse_time_manager(value);
                if (!parsed) return tl::unexpected(parsed.error());
                options.time_manager_text = std::string{value};
                change(options.time_manager, *parsed, Reconfigure::None);
                break;
            }
            case Field::BackendOptions: {
                auto applied = apply_backend_options(options, text, result.impact);
                if (!applied) return tl::unexpected(applied.error());
                options.backend_options = std::string{text};
                break;
            }
            case Field::SyzygyPath:
                change(options.syzygy_path, std::string{text}, Reconfigure::Search);
                break;
            case Field::SyzygyFastPlay:
                change(options.syzygy_fast_play, boolean, Reconfigure::None);
                break;
        }
        const auto contempt = search::resolve_contempt(options.uci_opponent, options.contempt,
                                                       options.uci_rating_advantage);
        if (!contempt) return tl::unexpected(contempt.error());
        options.search.draw_rate.contempt = *contempt;
        const auto valid = search::validate_selection_parameters(options.search.selection);
        if (!valid) return tl::unexpected(valid.error());
        if (auto checked = search::validate_search_tuning(options.search); !checked)
            return tl::unexpected(checked.error());
        if (auto checked = search::validate_moves_left_parameters(options.search.moves_left);
            !checked)
            return tl::unexpected(checked.error());
        if (auto checked = search::validate_draw_rate_parameters(options.search.draw_rate);
            !checked)
            return tl::unexpected(checked.error());
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
            case Kind::SignedInteger:
                line += "spin";
                break;
            case Kind::Boolean:
                line += "check";
                break;
            case Kind::Backend:
            case Kind::Precision:
            case Kind::RootFpuStrategy:
            case Kind::Choice:
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
        else if (definition.kind == Kind::SignedInteger)
            line += " min " + std::to_string(definition.signed_minimum) + " max " +
                    std::to_string(definition.bounds.maximum);
        else if (definition.kind == Kind::Choice) {
            std::string_view rest{definition.choices};
            while (!rest.empty()) {
                const auto bar = rest.find('|');
                line += " var " + std::string{rest.substr(0, bar)};
                rest = bar == std::string_view::npos ? std::string_view{} : rest.substr(bar + 1);
            }
        } else if (definition.kind == Kind::Backend)
#if MOUSE_DIRECTML
            line += " var cpu var directml";
#else
            line += " var cpu var cuda var tensorrt";
#endif
        else if (definition.kind == Kind::Precision)
            line += " var fp32 var fp16";
        else if (definition.kind == Kind::RootFpuStrategy)
            line += " var same var reduction var absolute";
        lines.push_back(std::move(line));
    }
    return lines;
}
Result<Options> apply_config_text(Options options, std::string_view text) {
    // Lc0's long flags whose names differ from the UCI option once dashes and case are ignored.
    constexpr std::array<std::pair<std::string_view, const char*>, 11> kAliases{{
        {"noiseepsilon", "DirichletNoiseEpsilon"},
        {"noisealpha", "DirichletNoiseAlpha"},
        {"syzygypaths", "SyzygyPath"},
        {"weights", "WeightsFile"},
        {"nncache", "NNCacheSize"},
        {"moveoverhead", "MoveOverheadMs"},
        {"backendopts", "BackendOptions"},
        {"logfile", "LogFile"},
        {"config", "ConfigFile"},
        {"policysoftmaxtemp", "PolicyTemperature"},
        {"showwdl", "UCI_ShowWDL"},
    }};
    std::size_t number = 0;
    while (!text.empty()) {
        ++number;
        const auto newline = text.find('\n');
        auto line = text.substr(0, newline);
        text = newline == std::string_view::npos ? std::string_view{} : text.substr(newline + 1);
        constexpr std::string_view blank = " \t\r";
        const auto first = line.find_first_not_of(blank);
        if (first == std::string_view::npos) continue;
        line = line.substr(first, line.find_last_not_of(blank) - first + 1);
        if (line.front() == '#') continue;
        const auto where = "Config line " + std::to_string(number) + ": ";
        if (line.starts_with("--"))
            line.remove_prefix(2);
        else if (line.front() == '-')
            return tl::unexpected(where + "only -- arguments are supported");
        const auto equals = line.find('=');
        if (equals == std::string_view::npos)
            return tl::unexpected(where + "expected name=value");
        std::string name;
        for (const char c : line.substr(0, equals))
            if (c != '-') name.push_back(lower(c));
        std::string option = name;
        for (const auto& [alias, target] : kAliases)
            if (alias == name) option = target;
        SetOptionCommand command{.name = option, .value = std::string{line.substr(equals + 1)}};
        auto updated = apply_option(options, command);
        if (!updated) return tl::unexpected(where + updated.error());
        options = std::move(updated->options);
    }
    return options;
}
}  // namespace mouse::uci
