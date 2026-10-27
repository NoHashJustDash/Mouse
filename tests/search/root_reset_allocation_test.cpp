// SPDX-License-Identifier: GPL-3.0-or-later
// This separate test executable replaces new only to audit the calling thread's
// root transaction. No product allocator hook or shared-test process is changed.
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <new>

#include "engine/chess/attacks.h"
#include "engine/chess/movegen.h"
#include "engine/nn/history.h"
#include "engine/nn/legal_cache.h"
#include "engine/search/search.h"

namespace {
struct AllocationRecord {
    void* memory = nullptr;
    std::size_t bytes = 0;
};
struct AllocationState {
    bool deny = false, track = false, capture = false;
    // Include checked-iterator proxy allocations in Windows Debug builds.
    std::array<AllocationRecord, 16> records{};
    std::size_t recorded_count = 0;
    std::size_t attempts = 0, permitted = 0;
    std::int64_t change = 0, peak = 0;
};
// Replacement global new cannot receive a caller-owned context. This test-only
// state is thread-local and enabled only around the synchronous transaction.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
thread_local AllocationState allocation;
void reject_if_guarded() {
    if (allocation.deny) {
        ++allocation.attempts;
        if (allocation.attempts > allocation.permitted) throw std::bad_alloc{};
    }
}
struct Guard {
    explicit Guard(std::size_t permitted = 0) {
        allocation.attempts = 0;
        allocation.permitted = permitted;
        allocation.deny = true;
    }
    ~Guard() { allocation.deny = false; }
    Guard(const Guard&) = delete;
    Guard& operator=(const Guard&) = delete;
    Guard(Guard&&) = delete;
    Guard& operator=(Guard&&) = delete;
};
struct Provider final : mouse::nn::Backend {
    mouse::nn::ModelMetadata model;
    mouse::nn::BackendInfo identity;
    Provider() { model.input_format = 1; }
    [[nodiscard]] const mouse::nn::ModelMetadata& metadata() const noexcept override {
        return model;
    }
    [[nodiscard]] const mouse::nn::BackendInfo& info() const noexcept override { return identity; }
    mouse::Result<void> evaluate(std::span<const mouse::nn::InputPlanes>,
                                 std::span<mouse::nn::NNOutput> outputs) override {
        for (auto& output : outputs) {
            output = {};
            output.wdl = {0.6F, 0.2F, 0.2F};
        }
        return {};
    }
};
}  // namespace

// A header records requested bytes so the growth audit accounts for frees of
// old vectors too. All accounting is test-only and confined to the input thread.
namespace {
struct Header {
    void* base;
    std::size_t size;
};
// The allocator replacement must own raw C storage and access its private
// header; smart-pointer allocation here would recurse into global new.
// NOLINTBEGIN(cppcoreguidelines-no-malloc,cppcoreguidelines-owning-memory,cppcoreguidelines-pro-bounds-pointer-arithmetic)
void* allocate(std::size_t size, std::size_t alignment) {
    reject_if_guarded();
    size = size == 0 ? 1 : size;
    if (size > std::numeric_limits<std::size_t>::max() - sizeof(Header) - alignment)
        throw std::bad_alloc{};
    auto space = size + alignment;
    auto* base = std::malloc(space + sizeof(Header));
    if (!base) throw std::bad_alloc{};
    void* aligned = static_cast<std::byte*>(base) + sizeof(Header);
    if (!std::align(alignment, size, aligned, space)) {
        std::free(base);
        throw std::bad_alloc{};
    }
    const Header header{.base = base, .size = size};
    std::memcpy(static_cast<std::byte*>(aligned) - sizeof(Header), &header, sizeof(Header));
    if (allocation.track) {
        allocation.change += static_cast<std::int64_t>(size);
        allocation.peak = std::max(allocation.peak, allocation.change);
    }
    if (allocation.capture) {
        const auto index = allocation.recorded_count++;
        if (index < allocation.records.size())
            allocation.records.at(index) = {.memory = aligned, .bytes = size};
    }
    return aligned;
}
void deallocate(void* memory) noexcept {
    if (!memory) return;
    Header header{};
    std::memcpy(&header, static_cast<std::byte*>(memory) - sizeof(Header), sizeof(Header));
    if (allocation.track) allocation.change -= static_cast<std::int64_t>(header.size);
    std::free(header.base);
}
// NOLINTEND(cppcoreguidelines-no-malloc,cppcoreguidelines-owning-memory,cppcoreguidelines-pro-bounds-pointer-arithmetic)
}  // namespace
void* operator new(std::size_t size) { return allocate(size, alignof(std::max_align_t)); }
void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void* memory) noexcept { deallocate(memory); }
void operator delete[](void* memory) noexcept { deallocate(memory); }
void operator delete(void* memory, std::size_t) noexcept { deallocate(memory); }
void operator delete[](void* memory, std::size_t) noexcept { deallocate(memory); }
void* operator new(std::size_t size, std::align_val_t alignment) {
    return allocate(size, static_cast<std::size_t>(alignment));
}
void* operator new[](std::size_t size, std::align_val_t alignment) {
    return ::operator new(size, alignment);
}
void operator delete(void* memory, std::align_val_t) noexcept { deallocate(memory); }
void operator delete[](void* memory, std::align_val_t) noexcept { deallocate(memory); }
void operator delete(void* memory, std::size_t, std::align_val_t) noexcept { deallocate(memory); }
void operator delete[](void* memory, std::size_t, std::align_val_t) noexcept { deallocate(memory); }

int main() {
    try {
        {
            // Observe owned arena allocations without exposing private storage or
            // adding a product test hook. Reroot may initialize scratch for live
            // nodes, but must not write the unused tail of either ID workspace.
            allocation.capture = true;
            auto compact = mouse::search::Tree::create(2U << 20U);
            allocation.capture = false;
            if (!compact || allocation.recorded_count > allocation.records.size()) return 29;
            auto& tree = **compact;
            const auto keep = tree.allocate_node();
            const auto leaf = tree.allocate_node();
            if (!keep || !leaf || !tree.allocate_node()) return 29;
            const std::array seeds{
                mouse::search::EdgeSeed{.move = mouse::chess::Move{0, 1}, .prior_bits = 0x3800}};
            if (!tree.expand(*keep, seeds)) return 29;
            tree.edges(*keep)[0].child = *leaf;
            tree.node(*leaf).visits = 17;
            const auto live = tree.stats().node_count;
            const auto capacity = tree.stats().node_capacity;
            constexpr auto marker = std::byte{0xa5};
            std::array<std::span<std::byte>, 2> tails{};
            std::size_t tail_count = 0;
            const auto workspace_bytes = capacity * sizeof(mouse::search::NodeId);
            for (const auto& record : allocation.records) {
                if (record.bytes < workspace_bytes || record.bytes >= 2 * workspace_bytes) continue;
                if (tail_count == tails.size()) return 29;
                // Some standard libraries overallocate aligned vectors. Their
                // data offset cannot exceed the extra bytes: this intersection
                // lies past every possible live prefix and before every end.
                const auto begin =
                    record.bytes - workspace_bytes + live * sizeof(mouse::search::NodeId);
                if (begin >= workspace_bytes) return 29;
                auto storage = std::span{static_cast<std::byte*>(record.memory), record.bytes};
                tails.at(tail_count) = storage.subspan(begin, workspace_bytes - begin);
                std::ranges::fill(tails.at(tail_count++), marker);
            }
            if (tail_count != 2 || capacity <= live) return 29;
            const auto reused = tree.re_root(*keep);
            if (!reused || reused->kept_nodes != 2 || tree.node(1).visits != 17) return 30;
            std::size_t touched = 0;
            for (const auto tail : tails)
                touched += static_cast<std::size_t>(
                    std::ranges::count_if(tail, [](std::byte value) { return value != marker; }));
            std::cout << "TREE_REUSE_SCRATCH live=" << live << " capacity=" << capacity
                      << " unused_touched=" << touched << '\n';
            if (touched != 0) return 31;
        }
        const mouse::chess::AttackTables attacks;
        Provider provider;
        mouse::search::SearchConfig config;
        config.maximum_batch = 8;
        config.threads = 4;
        config.max_depth = 64;
        config.ram_limit_bytes = 64U << 20U;
        auto root = mouse::chess::Position::from_fen(mouse::chess::kStartFen, attacks);
        auto other = mouse::chess::Position::from_fen("4k3/8/8/8/8/8/P7/R3K3 b - - 0 1", attacks);
        if (!root || !other) return 1;
        mouse::nn::EncodingHistory first, second;
        first.reset(*root);
        second.reset(*other);
        for (int cycle = 0; cycle < 4; ++cycle) {
            for (const auto* text : {"g1f3", "g8f6", "f3g1", "f6g8"}) {
                const auto move = mouse::chess::parse_uci_move(*root, text, attacks);
                if (!move) return 8;
                static_cast<void>(root->make_move(*move, attacks));
                if (!first.append(*root)) return 9;
            }
        }
        // Borrowed input spare capacity is not copied into the managed contexts.
        if (root->history_size() != 17 || !root->reserve_history(1U << 20U)) return 16;
        auto engine = mouse::search::SearchEngine::create(config, attacks, provider);
        if (!engine || !(*engine)->set_position(*root, first)) return 2;
        const auto capacity = (*engine)->snapshot().tree_capacity;
        std::size_t attempts = 0, completed = 0;
        for (std::size_t index = 0; index < 32; ++index) {
            mouse::search::SearchLimits limits;
            limits.nodes = 257;
            const auto searched = (*engine)->run(limits);
            if (!searched || searched->info.tree_nodes < 100) return 3;
            bool success = false;
            try {
                const Guard guard;
                success = static_cast<bool>((*engine)->set_position(
                    index % 2 == 0 ? *other : *root, index % 2 == 0 ? second : first));
            } catch (const std::bad_alloc&) {
                // The pre-fix error-message allocation can itself fail; the
                // guard has unwound before any diagnostic or test bookkeeping.
                success = false;
            }
            attempts += allocation.attempts;
            if (!success) break;
            const auto snapshot = (*engine)->snapshot();
            if (snapshot.nodes != 0 || snapshot.tree_nodes != 1 ||
                snapshot.tree_capacity != capacity ||
                snapshot.charged_bytes > snapshot.budget_bytes || (*engine)->root_moves().size != 0)
                return 4;
            ++completed;
        }
        std::cout << "ROOT_RESET_ALLOCATION completed=" << completed
                  << " expected=32 attempts=" << attempts
                  << " workers=4 populated_nodes_min=100 history_frames=17/1\n";
        if (completed != 32 || attempts != 0) return 5;
        config.ram_limit_bytes = 4U << 20U;
        config.maximum_batch = 1;
        config.max_depth = 16;
        for (int cycle = 0; cycle < 746; ++cycle) {
            for (const auto* text : {"g1f3", "g8f6", "f3g1", "f6g8"}) {
                const auto move = mouse::chess::parse_uci_move(*root, text, attacks);
                if (!move) return 10;
                static_cast<void>(root->make_move(*move, attacks));
                if (!first.append(*root)) return 11;
            }
        }
        auto growing = mouse::search::SearchEngine::create(config, attacks, provider);
        if (!growing) return 12;
        const auto fixed_bytes = (*growing)->snapshot().charged_bytes;
        if (!(*growing)->set_position(*root, first)) return 12;
        const auto before = (*growing)->snapshot();
        const auto move = mouse::chess::parse_uci_move(*root, "g1f3", attacks);
        if (!move) return 13;
        static_cast<void>(root->make_move(*move, attacks));
        if (!first.append(*root)) return 14;
        allocation.change = 0;
        allocation.peak = 0;
        allocation.track = true;
        const auto replaced = (*growing)->set_position(*root, first);
        allocation.track = false;
        const auto requested_peak =
            before.charged_bytes + static_cast<std::size_t>(allocation.peak);
        std::cout << "ROOT_RESET_GROWTH before_charge=" << before.charged_bytes
                  << " peak_requested=" << requested_peak << " budget=" << config.ram_limit_bytes
                  << " history_frames=" << first.frames().size()
                  << " final_charge=" << (*growing)->snapshot().charged_bytes << '\n';
        if (!replaced || requested_peak > config.ram_limit_bytes) return 15;
        // Fill 95% of the remaining history budget, leaving room for the
        // recovered tree but less than one retained old frame buffer. Derive
        // the boundary from actual scratch charge: gather layouts can change
        // without changing the replacement-peak contract tested below.
        const auto context_count = config.threads + 1;
        const auto history_budget = config.ram_limit_bytes - fixed_bytes;
        const auto target_frames =
            (history_budget - history_budget / 20) /
                (context_count * (sizeof(mouse::nn::HistoryFrame) + sizeof(std::uint64_t))) -
            (config.max_depth + 1);
        if (target_frames <= first.frames().size()) return 28;
        while (first.frames().size() + 4 <= target_frames) {
            for (const auto* text : {"g8f6", "f3g1", "f6g8", "g1f3"}) {
                const auto next = mouse::chess::parse_uci_move(*root, text, attacks);
                if (!next) return 17;
                static_cast<void>(root->make_move(*next, attacks));
                if (!first.append(*root)) return 18;
            }
        }
        const auto refused = (*growing)->set_position(*root, first);
        if (refused || refused.error().find("replacement peak") == std::string::npos ||
            (*growing)->snapshot().nodes != 0 || (*growing)->run({}))
            return 19;
        const auto recovered = (*growing)->set_position(*root, first);
        std::cout << "ROOT_RESET_PEAK_FAILURE cleared=" << (!refused ? 1 : 0)
                  << " recovered=" << (recovered ? 1 : 0)
                  << " history_frames=" << first.frames().size()
                  << " final_charge=" << (*growing)->snapshot().charged_bytes
                  << " budget=" << config.ram_limit_bytes << '\n';
        if (!recovered || (*growing)->snapshot().charged_bytes > config.ram_limit_bytes) return 20;
        auto cache = mouse::nn::LegalCache::create(4, "allocation-fixture");
        if (!cache) return 21;
        mouse::nn::PolicyIndices policy;
        policy.size = 1;
        policy.values[0] = 3;
        mouse::nn::LegalOutput output;
        output.wdl = {0.25F, 0.5F, 0.25F};
        // The cache design requires on-demand insertion. The original
        // zero-allocation assertion stays on warmed hits, updates and clear.
        for (std::uint64_t i = 0; i < 32; ++i) {
            mouse::nn::InputPlanes input{};
            input[0].mask = i;
            if (!(*cache)->insert(input, policy, output)) return 22;
            const Guard guard;
            if (!(*cache)->insert(input, policy, output)) return 22;
            auto found = (*cache)->lookup(input, policy);
            if (!found) return 23;
            const auto& value = *found;
            if (!value.has_value() || value.value().wdl != output.wdl) return 23;
            if (i % 8 == 0) (*cache)->clear();
            if (allocation.attempts != 0) return 24;
        }
        std::cout << "LEGAL_CACHE_HOTPATH allocations=" << allocation.attempts
                  << " capacity=" << (*cache)->stats().capacity << '\n';
        mouse::nn::InputPlanes probe{};
        probe[0].mask = 1000;
        std::size_t insertion_allocations = 0;
        {
            const Guard count(std::numeric_limits<std::size_t>::max());
            if (!(*cache)->insert(probe, policy, output)) return 25;
            insertion_allocations = allocation.attempts;
        }
        if (insertion_allocations == 0) return 25;
        for (std::size_t stage = 0; stage < insertion_allocations; ++stage) {
            const auto cache_before = (*cache)->stats();
            probe[0].mask = 2000 + stage;
            bool cache_refused = false;
            {
                const Guard guard(stage);
                const auto inserted = (*cache)->insert(probe, policy, output);
                cache_refused = !inserted && inserted.error().find("OOM") != std::string::npos;
            }
            const auto cache_after = (*cache)->stats();
            if (!cache_refused || cache_after != cache_before) {
                std::cout << "LEGAL_CACHE_OOM_FAILURE stage=" << stage
                          << " refused=" << cache_refused << " attempts=" << allocation.attempts
                          << " charge=" << cache_before.charged_bytes << '/'
                          << cache_after.charged_bytes << " entries=" << cache_before.entries << '/'
                          << cache_after.entries << " collisions=" << cache_before.bucket_collisions
                          << '/' << cache_after.bucket_collisions << '\n';
                return 26;
            }
            const Guard guard;
            for (const std::uint64_t retained : {29U, 30U, 31U, 1000U}) {
                probe[0].mask = retained;
                const auto found = (*cache)->lookup(probe, policy);
                if (!found) return 27;
                const auto& retained_value = *found;
                if (!retained_value.has_value() || retained_value.value().wdl != output.wdl)
                    return 27;
            }
            if (allocation.attempts != 0) return 24;
        }
        std::cout << "LEGAL_CACHE_ON_DEMAND failure_stages=" << insertion_allocations
                  << " preserved_entries=" << (*cache)->stats().entries << '\n';
        return 0;
    } catch (const std::exception& error) {
        // The exception boundary diagnostic must not allocate through iostreams.
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
        std::fprintf(stderr, "Root reset allocation regression: %s\n", error.what());
        return 6;
    } catch (...) {
        std::fputs("Root reset allocation regression: non-standard exception\n", stderr);
        return 7;
    }
}
