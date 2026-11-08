// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/search/tree.h"

#include <algorithm>
#include <bit>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace mouse::search {
namespace {
constexpr std::size_t kAllocationReserve = 3ULL * 64;
constexpr NodeId kReachable = kNoNode - 1;
// Each node the arena could hold also needs a node-map and an edge-order entry.
constexpr std::size_t kWorkspacePerNode = 2 * sizeof(NodeId);
// A DAG tree gives one low node to about this many arena bytes: a position's edges and the child
// nodes it spawns. Table slots are twice the low nodes, so a full table still probes briefly.
constexpr std::size_t kArenaBytesPerLow = 1024;
constexpr std::size_t kTableSlotsPerLow = 2;
static_assert(std::is_nothrow_copy_assignable_v<Node>);
static_assert(std::is_nothrow_copy_assignable_v<Edge>);
// Arena bytes pass between nodes and edges as the tree grows and is compacted.
static_assert(std::is_trivially_copyable_v<Node> && std::is_trivially_destructible_v<Node>);
static_assert(std::is_trivially_copyable_v<Edge> && std::is_trivially_destructible_v<Edge>);
static_assert(alignof(Node) <= __STDCPP_DEFAULT_NEW_ALIGNMENT__ &&
              alignof(Edge) <= __STDCPP_DEFAULT_NEW_ALIGNMENT__);
}  // namespace

Tree::Tree(FactoryToken) {}
Tree::~Tree() = default;

Result<std::unique_ptr<Tree>> Tree::create(std::size_t budget_bytes, bool dag) {
    const auto fixed_bytes = sizeof(Tree) + kAllocationReserve + (dag ? kAllocationReserve : 0);
    if (budget_bytes > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
        return tl::unexpected(std::string{"Tree budget exceeds addressable storage"});
    auto available = budget_bytes < fixed_bytes ? 0 : budget_bytes - fixed_bytes;
    std::size_t low_capacity = 0, table_slots = 0;
    if (dag) {
        const auto per_low = kArenaBytesPerLow + sizeof(LowNode) + kTableSlotsPerLow * sizeof(LowId);
        low_capacity = std::max<std::size_t>(available / per_low, 1);
        table_slots = std::bit_ceil(low_capacity * kTableSlotsPerLow);
        const auto low_bytes = low_capacity * sizeof(LowNode) + table_slots * sizeof(LowId);
        if (low_bytes >= available)
            return tl::unexpected(std::string{"Tree budget cannot fit the low node table"});
        available -= low_bytes;
    }
    const auto per_node_workspace = kWorkspacePerNode + (dag ? sizeof(LowId) : 0);
    const auto node_limit = available / (sizeof(Node) + per_node_workspace);
    const auto arena_bytes = available - node_limit * per_node_workspace;
    const auto edge_slots = arena_bytes / sizeof(Edge);
    if (node_limit == 0 || edge_slots * sizeof(Edge) < sizeof(Node) + sizeof(Edge))
        return tl::unexpected(std::string{"Tree budget cannot fit the root and one edge"});
    if (node_limit > kReachable || edge_slots >= kNoEdge)
        return tl::unexpected(std::string{"Tree capacity exceeds 32-bit arena IDs"});
    try {
        auto result = std::make_unique<Tree>(FactoryToken{});
        // Value-initialised: every page is touched here, not during a search.
        result->arena_ = std::make_unique<std::byte[]>(arena_bytes);
        result->node_map_.resize(node_limit);
        result->edge_order_.resize(node_limit);
        result->edge_slots_ = edge_slots;
        result->stats_.budget_bytes = budget_bytes;
        result->stats_.charged_bytes = fixed_bytes;
        auto charge = [&](std::size_t count, std::size_t width) {
            if (count > (budget_bytes - result->stats_.charged_bytes) / width) return false;
            result->stats_.charged_bytes += count * width;
            return true;
        };
        if (!charge(arena_bytes, 1) || !charge(result->node_map_.capacity(), sizeof(NodeId)) ||
            !charge(result->edge_order_.capacity(), sizeof(NodeId)))
            return tl::unexpected(
                std::string{"Tree actual arena capacity exceeds its byte budget"});
        if (dag) {
            result->low_of_.assign(node_limit, kNoLow);
            result->lows_.resize(low_capacity);
            result->low_table_.assign(table_slots, kNoLow);
            if (!charge(result->low_of_.capacity(), sizeof(LowId)) ||
                !charge(result->lows_.capacity(), sizeof(LowNode)) ||
                !charge(result->low_table_.capacity(), sizeof(LowId)))
                return tl::unexpected(
                    std::string{"Tree actual low node capacity exceeds its byte budget"});
            result->stats_.low_capacity = static_cast<std::uint32_t>(low_capacity);
        }
        result->stats_.node_limit = static_cast<std::uint32_t>(node_limit);
        result->stats_.arena_bytes = edge_slots * sizeof(Edge);
        result->clear_storage();
        return result;
    } catch (const std::bad_alloc&) {
        return tl::unexpected(std::string{"Tree OOM"});
    } catch (const std::length_error&) {
        return tl::unexpected(std::string{"Tree too large"});
    }
}

Node* Tree::node_slot(NodeId id) const noexcept {
    return std::launder(reinterpret_cast<Node*>(arena_.get() + std::size_t{id} * sizeof(Node)));
}
Edge* Tree::edge_slot(EdgeId id) const noexcept {
    return std::launder(reinterpret_cast<Edge*>(arena_.get() + std::size_t{id} * sizeof(Edge)));
}
std::size_t Tree::free_bytes() const noexcept {
    return std::size_t{edge_low_} * sizeof(Edge) -
           std::size_t{node_count_.load(std::memory_order_relaxed)} * sizeof(Node);
}
TreeStats Tree::stats() const noexcept {
    auto result = stats_;
    result.node_count = node_count_.load(std::memory_order_relaxed);
    result.low_count = low_count_;
    const auto free = free_bytes();
    result.node_capacity = static_cast<std::uint32_t>(
        result.node_count +
        std::min<std::size_t>(free / sizeof(Node), stats_.node_limit - result.node_count));
    result.edge_capacity = static_cast<std::uint32_t>(stats_.edge_count + free / sizeof(Edge));
    result.used_bytes = stats_.arena_bytes - free;
    return result;
}

Result<NodeId> Tree::allocate_node() {
    const auto id = node_count_.load(std::memory_order_relaxed);
    if (id == stats_.node_limit || free_bytes() < sizeof(Node))
        return tl::unexpected(std::string{"Tree nodes full"});
    std::construct_at(node_slot(id));
    if (!low_of_.empty()) low_of_[id] = kNoLow;
    node_count_.store(id + 1, std::memory_order_relaxed);
    return id;
}

Result<void> Tree::validate_seeds(std::span<const EdgeSeed> seeds) const {
    if (seeds.empty() || seeds.size() > std::numeric_limits<std::uint16_t>::max())
        return tl::unexpected(std::string{"Tree expansion requires 1..65535 move seeds"});
    if (seeds.size() > free_bytes() / sizeof(Edge))
        return tl::unexpected(std::string{"Tree edges full"});
    for (std::size_t index = 0; index < seeds.size(); ++index) {
        if (seeds[index].prior_bits > 0x3c00)
            return tl::unexpected(std::string{"Tree prior must encode nonnegative IEEE half <=1"});
        for (std::size_t earlier = 0; earlier < index; ++earlier)
            if (seeds[index].move == seeds[earlier].move)
                return tl::unexpected(std::string{"Tree expansion contains duplicate moves"});
    }
    return {};
}
void Tree::place_edges(std::span<const EdgeSeed> seeds) noexcept {
    // A block takes the slots just below the lowest edge block.
    const auto first = static_cast<EdgeId>(edge_low_ - seeds.size());
    for (std::size_t index = 0; index < seeds.size(); ++index)
        std::construct_at(edge_slot(static_cast<EdgeId>(first + index)),
                          Edge{.move = seeds[index].move,
                               .child = kNoNode,
                               .prior_bits = seeds[index].prior_bits});
    edge_low_ = first;
    stats_.edge_count += static_cast<std::uint32_t>(seeds.size());
}
Result<void> Tree::expand(NodeId id, std::span<const EdgeSeed> seeds) {
    if (id >= node_count_.load(std::memory_order_relaxed))
        return tl::unexpected(std::string{"Tree expansion has invalid node ID"});
    auto& current = *node_slot(id);
    if ((current.state != NodeState::Unexpanded && current.state != NodeState::Pending) ||
        current.edge_count != 0 || current.first_edge != kNoEdge)
        return tl::unexpected(std::string{"Tree expansion requires a fresh node"});
    auto valid = validate_seeds(seeds);
    if (!valid) return valid;
    // All fallible validation precedes the first write. Arena capacity is fixed,
    // and these constructions cannot throw; Pending keeps its reservation intact.
    place_edges(seeds);
    current.first_edge = edge_low_;
    current.edge_count = static_cast<std::uint16_t>(seeds.size());
    current.state = NodeState::Expanded;
    return {};
}

std::optional<std::size_t> Tree::solidify(NodeId id, std::span<NodeRelocation> moved,
                                          std::size_t reserved_edges) {
    auto& parent = node(id);
    if (parent.solid || parent.state != NodeState::Expanded || parent.edge_count == 0 ||
        !low_of_.empty() || parent.edge_count > moved.size())
        return std::nullopt;
    const std::size_t count = parent.edge_count;
    const auto first = node_count_.load(std::memory_order_relaxed);
    if (count > std::size_t{stats_.node_limit} - first ||
        free_bytes() < count * sizeof(Node) + reserved_edges * sizeof(Edge))
        return std::nullopt;
    const auto children = edges(id);
    std::size_t relocated = 0;
    for (std::size_t index = 0; index < count; ++index) {
        const auto target = static_cast<NodeId>(first + index);
        auto& slot = *std::construct_at(node_slot(target));
        auto& edge = children[index];
        if (edge.child != kNoNode) {
            slot = *node_slot(edge.child);
            std::construct_at(node_slot(edge.child));
            moved[relocated++] = {.from = edge.child, .to = target};
        }
        edge.child = target;
    }
    node_count_.store(static_cast<std::uint32_t>(first + count), std::memory_order_relaxed);
    parent.solid = true;
    return relocated;
}

LowId Tree::find_low(std::uint64_t key) const noexcept {
    if (low_table_.empty()) return kNoLow;
    const auto mask = low_table_.size() - 1;
    for (auto at = static_cast<std::size_t>(key) & mask;; at = (at + 1) & mask) {
        const auto candidate = low_table_[at];
        if (candidate == kNoLow) return kNoLow;
        if (lows_[candidate].key == key) return candidate;
    }
}
void Tree::insert_low(LowId id) noexcept {
    const auto mask = low_table_.size() - 1;
    auto at = static_cast<std::size_t>(lows_[id].key) & mask;
    while (low_table_[at] != kNoLow) at = (at + 1) & mask;
    low_table_[at] = id;
}
Result<LowId> Tree::create_low(std::uint64_t key, std::span<const EdgeSeed> seeds, bool in_table) {
    if (low_table_.empty()) return tl::unexpected(std::string{"Tree has no low nodes"});
    if (low_count_ == stats_.low_capacity) return tl::unexpected(std::string{"Tree low nodes full"});
    auto valid = validate_seeds(seeds);
    if (!valid) return tl::unexpected(valid.error());
    place_edges(seeds);
    const auto id = low_count_++;
    lows_[id] = LowNode{.key = key,
                        .first_edge = edge_low_,
                        .edge_count = static_cast<std::uint16_t>(seeds.size()),
                        .in_table = in_table};
    if (in_table) insert_low(id);
    return id;
}
Result<void> Tree::attach_low(NodeId id, LowId low_id) {
    auto& current = node(id);
    if (low_of_.empty() || low_id >= low_count_)
        return tl::unexpected(std::string{"Tree attach has invalid low node"});
    if ((current.state != NodeState::Unexpanded && current.state != NodeState::Pending) ||
        current.edge_count != 0 || current.first_edge != kNoEdge)
        return tl::unexpected(std::string{"Tree attach requires a fresh node"});
    auto& shared = lows_[low_id];
    current.first_edge = shared.first_edge;
    current.edge_count = shared.edge_count;
    current.state = NodeState::Expanded;
    low_of_[id] = low_id;
    ++shared.parents;
    if (shared.parents > 1) shared.transposition = true;
    return {};
}

Node& Tree::node(NodeId id) {
    if (id >= node_count_.load(std::memory_order_relaxed)) throw std::out_of_range("Tree node ID");
    return *node_slot(id);
}
const Node& Tree::node(NodeId id) const {
    if (id >= node_count_.load(std::memory_order_relaxed)) throw std::out_of_range("Tree node ID");
    return *node_slot(id);
}
bool Tree::valid_edge_range(const Node& current) const noexcept {
    if (current.edge_count == 0) return current.first_edge == kNoEdge;
    return current.first_edge >= edge_low_ && current.first_edge < edge_slots_ &&
           current.edge_count <= edge_slots_ - current.first_edge;
}
std::span<Edge> Tree::edges(NodeId id) {
    const auto& current = node(id);
    if (!valid_edge_range(current)) throw std::out_of_range("Tree edge range");
    if (current.edge_count == 0) return {};
    return {edge_slot(current.first_edge), current.edge_count};
}
std::span<const Edge> Tree::edges(NodeId id) const {
    const auto& current = node(id);
    if (!valid_edge_range(current)) throw std::out_of_range("Tree edge range");
    if (current.edge_count == 0) return {};
    return {edge_slot(current.first_edge), current.edge_count};
}

bool Tree::quiescent() const noexcept {
    const auto count = node_count_.load(std::memory_order_relaxed);
    for (NodeId id = 0; id < count; ++id)
        if (node_slot(id)->in_flight != 0 || node_slot(id)->state == NodeState::Pending)
            return false;
    return true;
}
void Tree::clear_storage() noexcept {
    std::construct_at(node_slot(0));
    node_count_.store(1, std::memory_order_relaxed);
    stats_.edge_count = 0;
    edge_low_ = static_cast<EdgeId>(edge_slots_);
    if (!low_of_.empty()) {
        low_of_[0] = kNoLow;
        low_count_ = 0;
        std::ranges::fill(low_table_, kNoLow);
    }
}
Result<void> Tree::reset() {
    if (!quiescent())
        return tl::unexpected(std::string{"Tree reset requires drained reservations"});
    if (generation_ == std::numeric_limits<std::uint64_t>::max())
        return tl::unexpected(std::string{"Tree generation exhausted"});
    clear_storage();
    ++generation_;
    return {};
}

Result<ReuseSummary> Tree::prepare_reuse(NodeId id) {
    const auto count = node_count_.load(std::memory_order_relaxed);
    // Reuse only inspects allocated IDs; leave unused arena scratch untouched.
    std::ranges::fill(std::span{node_map_}.first(count), kNoNode);
    for (LowId low_id = 0; low_id < low_count_; ++low_id) lows_[low_id].remap = kNoLow;
    ReuseSummary summary;
    // Breadth first through edge_order_, which ends up holding each kept edge block once. Solid
    // children and shared blocks can have lower IDs than their parents, so IDs give no order.
    std::size_t head = 0, tail = 0;
    const auto keep = [&](NodeId old) -> Result<void> {
        node_map_.at(old) = kReachable;
        ++summary.kept_nodes;
        const auto& current = *node_slot(old);
        if (!valid_edge_range(current))
            return tl::unexpected(std::string{"Tree retained node has invalid edge range"});
        if (current.edge_count == 0) return {};
        if (const auto low_id = low_of(old); low_id != kNoLow) {
            if (low_id >= low_count_)
                return tl::unexpected(std::string{"Tree retained node has invalid low node"});
            // Another node already brought this shared block along.
            if (lows_[low_id].remap != kNoLow) return {};
            lows_[low_id].remap = 0;
        }
        if (current.edge_count > stats_.edge_count - summary.kept_edges)
            return tl::unexpected(std::string{"Tree retained edge blocks overlap"});
        summary.kept_edges += current.edge_count;
        edge_order_.at(tail++) = old;
        return {};
    };
    if (auto kept = keep(id); !kept) return tl::unexpected(kept.error());
    while (head != tail) {
        const auto old = edge_order_.at(head++);
        for (const auto& edge : edges(old)) {
            if (edge.prior_bits > 0x3c00)
                return tl::unexpected(std::string{"Tree retained edge has invalid prior"});
            if (edge.child == kNoNode) continue;
            if (edge.child >= count)
                return tl::unexpected(std::string{"Tree retained child has invalid ID"});
            if (node_map_.at(edge.child) != kNoNode)
                return tl::unexpected(std::string{"Tree retained child has multiple parents"});
            if (auto kept = keep(edge.child); !kept) return tl::unexpected(kept.error());
        }
    }
    reuse_blocks_ = tail;
    auto blocks = std::span{edge_order_}.first(tail);
    std::ranges::sort(blocks, [&](NodeId left, NodeId right) {
        return node_slot(left)->first_edge < node_slot(right)->first_edge;
    });
    EdgeId end = 0;
    for (const auto old : blocks) {
        const auto& current = *node_slot(old);
        if (current.first_edge < end)
            return tl::unexpected(std::string{"Tree retained edge blocks overlap"});
        end = current.first_edge + current.edge_count;
    }
    // The new root takes ID 0, the rest keep their relative order. The old root is never kept
    // below a new root, as nothing points to it, so every other node moves down or stays.
    node_map_.at(id) = 0;
    NodeId next = 1;
    for (NodeId old = 0; old < count; ++old)
        if (old != id && node_map_.at(old) != kNoNode) node_map_.at(old) = next++;
    LowId next_low = 0;
    for (LowId low_id = 0; low_id < low_count_; ++low_id)
        if (lows_[low_id].remap != kNoLow) lows_[low_id].remap = next_low++;
    return summary;
}

Result<ReuseSummary> Tree::re_root(NodeId id) {
    const auto count = node_count_.load(std::memory_order_relaxed);
    if (id >= count) return tl::unexpected(std::string{"Tree reroot has invalid node ID"});
    if (!quiescent())
        return tl::unexpected(std::string{"Tree reroot requires drained reservations"});
    if (generation_ == std::numeric_limits<std::uint64_t>::max())
        return tl::unexpected(std::string{"Tree generation exhausted"});
    auto summary = prepare_reuse(id);
    if (!summary) return tl::unexpected(summary.error());
    {
        // Edge blocks can have been allocated in a different order from nodes.
        // Packing them against the arena's end from the highest old offset down
        // puts each destination at or above its source; copying each block's
        // edges last to first remains safe even for overlapping ranges.
        auto destination = static_cast<EdgeId>(edge_slots_);
        const auto blocks = std::span{edge_order_}.first(reuse_blocks_);
        for (auto block = blocks.rbegin(); block != blocks.rend(); ++block) {
            auto& current = *node_slot(*block);
            const auto source = current.first_edge;
            destination -= current.edge_count;
            current.first_edge = destination;
            if (const auto low_id = low_of(*block); low_id != kNoLow)
                lows_[low_id].first_edge = destination;
            for (std::uint32_t offset = current.edge_count; offset-- > 0;) {
                auto edge = *edge_slot(source + offset);
                if (edge.child != kNoNode) edge.child = node_map_.at(edge.child);
                *edge_slot(destination + offset) = edge;
            }
        }
        // Nodes sharing a block follow the one that carried it.
        const auto settle = [&](Node& copy, LowId low_id) noexcept {
            if (low_id != kNoLow) copy.first_edge = lows_[low_id].first_edge;
        };
        // The new root goes first, as its old slot may be overwritten. Ascending old
        // IDs then copy only toward lower addresses, and no other source is overwritten
        // before it is consumed: slot 0 belonged to the old root, unless nothing moves.
        const auto root_low = low_of(id);
        if (id != 0) *node_slot(0) = *node_slot(id);
        settle(*node_slot(0), root_low);
        for (NodeId old = 1; old < count; ++old) {
            if (old == id || node_map_.at(old) == kNoNode) continue;
            const auto target = node_map_.at(old);
            const auto low_id = low_of(old);
            if (target != old) *node_slot(target) = *node_slot(old);
            settle(*node_slot(target), low_id);
            if (!low_of_.empty()) low_of_[target] = low_id == kNoLow ? kNoLow : lows_[low_id].remap;
        }
        if (!low_of_.empty()) low_of_[0] = root_low == kNoLow ? kNoLow : lows_[root_low].remap;
        node_count_.store(summary->kept_nodes, std::memory_order_relaxed);
        stats_.edge_count = summary->kept_edges;
        edge_low_ = destination;
        if (!low_of_.empty()) {
            // Low nodes hold their ordering too, and the table is rebuilt for the ones left.
            std::ranges::fill(low_table_, kNoLow);
            LowId kept = 0;
            for (LowId low_id = 0; low_id < low_count_; ++low_id) {
                if (lows_[low_id].remap == kNoLow) continue;
                if (kept != low_id) lows_[kept] = lows_[low_id];
                lows_[kept].remap = kNoLow;
                if (lows_[kept].in_table) insert_low(kept);
                ++kept;
            }
            low_count_ = kept;
        }
    }
    ++generation_;
    return summary;
}

Result<std::uint64_t> Tree::revert_twofold(std::span<const NodeId> ancestors, NodeId leaf) {
    auto& terminal = node(leaf);
    if (terminal.state != NodeState::Terminal || !terminal.twofold || !terminal.leaf_moves_left)
        return tl::unexpected(std::string{"Tree twofold revert needs a twofold terminal"});
    if (terminal.in_flight != 0)
        return tl::unexpected(std::string{"Tree twofold revert needs a quiescent terminal"});
    const auto visits = terminal.visits;
    const auto cycle = static_cast<double>(*terminal.leaf_moves_left);
    for (const auto id : ancestors) {
        const auto& current = node(id);
        if (current.visits < visits || current.moves_left_samples < visits)
            return tl::unexpected(std::string{"Tree twofold revert exceeds ancestor visits"});
    }
    for (std::size_t index = 0; index < ancestors.size(); ++index) {
        auto& current = node(ancestors[index]);
        // Each sample was backed up as cycle + its distance from the terminal.
        const auto distance = static_cast<double>(ancestors.size() - index);
        current.visits -= visits;
        current.moves_left_samples -= visits;
        current.moves_left_sum -= static_cast<double>(visits) * (cycle + distance);
        if (current.proof.outcome == ProofOutcome::Unknown) {
            // A draw backs up q = 0 and draw = 1 from every viewpoint.
            current.draw_sum -= static_cast<double>(visits);
            continue;
        }
        // Proven nodes hold their exact value at the real visit count (backup_locked).
        const auto count = static_cast<double>(current.visits);
        current.value_sum = current.proof.outcome == ProofOutcome::Win
                                ? count
                                : (current.proof.outcome == ProofOutcome::Loss ? -count : 0.0);
        current.draw_sum = current.proof.outcome == ProofOutcome::Draw ? count : 0.0;
    }
    terminal = Node{};
    return visits;
}
}  // namespace mouse::search
