// SPDX-License-Identifier: GPL-3.0-or-later
#include "engine/search/tree.h"

#include <algorithm>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace mouse::search {
namespace {
constexpr std::size_t kEdgesPerNode = 8;
constexpr std::size_t kAllocationReserve = 4ULL * 64;
constexpr NodeId kReachable = kNoNode - 1;
static_assert(std::is_nothrow_copy_assignable_v<Node>);
static_assert(std::is_nothrow_copy_assignable_v<Edge>);
}  // namespace

Tree::Tree(FactoryToken) {}
Tree::~Tree() = default;

Result<std::unique_ptr<Tree>> Tree::create(std::size_t budget_bytes) {
    constexpr auto fixed_bytes = sizeof(Tree) + kAllocationReserve;
    constexpr auto bytes_per_slot =
        sizeof(Node) + kEdgesPerNode * sizeof(Edge) + 2 * sizeof(NodeId);
    if (budget_bytes < fixed_bytes || budget_bytes - fixed_bytes < bytes_per_slot)
        return tl::unexpected(std::string{"Tree budget cannot fit one complete arena slot"});
    if (budget_bytes > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
        return tl::unexpected(std::string{"Tree budget exceeds addressable storage"});
    const auto capacity = (budget_bytes - fixed_bytes) / bytes_per_slot;
    if (capacity > static_cast<std::size_t>(kReachable) / kEdgesPerNode)
        return tl::unexpected(std::string{"Tree capacity exceeds 32-bit arena IDs"});
    try {
        auto result = std::make_unique<Tree>(FactoryToken{});
        result->nodes_.resize(capacity);
        result->edges_.resize(capacity * kEdgesPerNode);
        result->node_map_.resize(capacity);
        result->edge_order_.resize(capacity);
        result->stats_.budget_bytes = budget_bytes;
        result->stats_.charged_bytes = fixed_bytes;
        auto charge = [&](std::size_t count, std::size_t width) {
            if (count > (budget_bytes - result->stats_.charged_bytes) / width) return false;
            result->stats_.charged_bytes += count * width;
            return true;
        };
        if (!charge(result->nodes_.capacity(), sizeof(Node)) ||
            !charge(result->edges_.capacity(), sizeof(Edge)) ||
            !charge(result->node_map_.capacity(), sizeof(NodeId)) ||
            !charge(result->edge_order_.capacity(), sizeof(NodeId)))
            return tl::unexpected(
                std::string{"Tree actual arena capacity exceeds its byte budget"});
        result->stats_.node_capacity = static_cast<std::uint32_t>(capacity);
        result->stats_.edge_capacity = static_cast<std::uint32_t>(capacity * kEdgesPerNode);
        result->clear_storage();
        return result;
    } catch (const std::bad_alloc&) {
        return tl::unexpected(std::string{"Tree OOM"});
    } catch (const std::length_error&) {
        return tl::unexpected(std::string{"Tree too large"});
    }
}

Result<NodeId> Tree::allocate_node() {
    if (stats_.node_count == stats_.node_capacity)
        return tl::unexpected(std::string{"Tree nodes full"});
    const auto id = stats_.node_count;
    nodes_.at(id) = Node{};
    ++stats_.node_count;
    return id;
}

Result<void> Tree::expand(NodeId id, std::span<const EdgeSeed> seeds) {
    if (id >= stats_.node_count)
        return tl::unexpected(std::string{"Tree expansion has invalid node ID"});
    auto& current = nodes_.at(id);
    if ((current.state != NodeState::Unexpanded && current.state != NodeState::Pending) ||
        current.edge_count != 0 || current.first_edge != kNoEdge)
        return tl::unexpected(std::string{"Tree expansion requires a fresh node"});
    if (seeds.empty() || seeds.size() > std::numeric_limits<std::uint16_t>::max())
        return tl::unexpected(std::string{"Tree expansion requires 1..65535 move seeds"});
    if (seeds.size() > stats_.edge_capacity - stats_.edge_count)
        return tl::unexpected(std::string{"Tree edges full"});
    for (std::size_t index = 0; index < seeds.size(); ++index) {
        if (seeds[index].prior_bits > 0x3c00)
            return tl::unexpected(std::string{"Tree prior must encode nonnegative IEEE half <=1"});
        for (std::size_t earlier = 0; earlier < index; ++earlier)
            if (seeds[index].move == seeds[earlier].move)
                return tl::unexpected(std::string{"Tree expansion contains duplicate moves"});
    }
    // All fallible validation precedes the first write. Arena capacity is fixed,
    // and these assignments cannot throw; Pending keeps its reservation intact.
    for (std::size_t index = 0; index < seeds.size(); ++index)
        edges_.at(stats_.edge_count + index) = Edge{
            .move = seeds[index].move, .child = kNoNode, .prior_bits = seeds[index].prior_bits};
    current.first_edge = stats_.edge_count;
    current.edge_count = static_cast<std::uint16_t>(seeds.size());
    current.state = NodeState::Expanded;
    stats_.edge_count += current.edge_count;
    return {};
}

Node& Tree::node(NodeId id) {
    if (id >= stats_.node_count) throw std::out_of_range("Tree node ID");
    return nodes_.at(id);
}
const Node& Tree::node(NodeId id) const {
    if (id >= stats_.node_count) throw std::out_of_range("Tree node ID");
    return nodes_.at(id);
}
bool Tree::valid_edge_range(const Node& current) const noexcept {
    if (current.edge_count == 0) return current.first_edge == kNoEdge;
    return current.first_edge < stats_.edge_count &&
           current.edge_count <= stats_.edge_count - current.first_edge;
}
std::span<Edge> Tree::edges(NodeId id) {
    const auto& current = node(id);
    if (!valid_edge_range(current)) throw std::out_of_range("Tree edge range");
    if (current.edge_count == 0) return {};
    return std::span{edges_}.subspan(current.first_edge, current.edge_count);
}
std::span<const Edge> Tree::edges(NodeId id) const {
    const auto& current = node(id);
    if (!valid_edge_range(current)) throw std::out_of_range("Tree edge range");
    if (current.edge_count == 0) return {};
    return std::span{edges_}.subspan(current.first_edge, current.edge_count);
}

bool Tree::quiescent() const noexcept {
    for (NodeId id = 0; id < stats_.node_count; ++id)
        if (nodes_[id].in_flight != 0 || nodes_[id].state == NodeState::Pending) return false;
    return true;
}
void Tree::clear_storage() noexcept {
    nodes_[0] = Node{};
    stats_.node_count = 1;
    stats_.edge_count = 0;
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
    // Reuse only inspects allocated IDs; leave unused arena scratch untouched.
    std::ranges::fill(std::span{node_map_}.first(stats_.node_count), kNoNode);
    node_map_.at(id) = kReachable;
    ReuseSummary summary;
    std::size_t block_count = 0;
    for (NodeId old = id; old < stats_.node_count; ++old) {
        if (node_map_.at(old) == kNoNode) continue;
        const auto& current = nodes_.at(old);
        if (!valid_edge_range(current))
            return tl::unexpected(std::string{"Tree retained node has invalid edge range"});
        node_map_.at(old) = summary.kept_nodes++;
        summary.discarded_context = summary.discarded_context || current.contextual_draw;
        if (current.edge_count == 0) continue;
        if (current.edge_count > stats_.edge_count - summary.kept_edges)
            return tl::unexpected(std::string{"Tree retained edge blocks overlap"});
        summary.kept_edges += current.edge_count;
        edge_order_.at(block_count++) = old;
        for (const auto& edge : edges(old)) {
            if (edge.prior_bits > 0x3c00)
                return tl::unexpected(std::string{"Tree retained edge has invalid prior"});
            if (edge.child == kNoNode) continue;
            if (edge.child <= old || edge.child >= stats_.node_count)
                return tl::unexpected(std::string{"Tree retained child violates append-only IDs"});
            if (node_map_.at(edge.child) != kNoNode)
                return tl::unexpected(std::string{"Tree retained child has multiple parents"});
            node_map_.at(edge.child) = kReachable;
        }
    }
    auto blocks = std::span{edge_order_}.first(block_count);
    std::ranges::sort(blocks, [&](NodeId left, NodeId right) {
        return nodes_.at(left).first_edge < nodes_.at(right).first_edge;
    });
    EdgeId end = 0;
    for (const auto old : blocks) {
        const auto& current = nodes_.at(old);
        if (current.first_edge < end)
            return tl::unexpected(std::string{"Tree retained edge blocks overlap"});
        end = current.first_edge + current.edge_count;
    }
    return summary;
}

Result<ReuseSummary> Tree::re_root(NodeId id) {
    if (id >= stats_.node_count)
        return tl::unexpected(std::string{"Tree reroot has invalid node ID"});
    if (!quiescent())
        return tl::unexpected(std::string{"Tree reroot requires drained reservations"});
    if (generation_ == std::numeric_limits<std::uint64_t>::max())
        return tl::unexpected(std::string{"Tree generation exhausted"});
    auto summary = prepare_reuse(id);
    if (!summary) return tl::unexpected(summary.error());
    if (summary->discarded_context) {
        // Root-relative twofold scores cannot be carried into a new root. This
        // independent conservative policy discards their affected subtree rather
        // than reproducing Lc0 classic/search.cc EnsureNodeTwoFoldCorrectForDepth.
        clear_storage();
        summary->kept_nodes = 1;
        summary->kept_edges = 0;
    } else {
        // Edge blocks can have been allocated in a different order from nodes.
        // Sorting by old offset guarantees each destination precedes its source;
        // forward element copies remain safe even for overlapping ranges.
        EdgeId destination = 0;
        std::size_t block_count = 0;
        for (NodeId old = id; old < stats_.node_count; ++old)
            if (node_map_.at(old) != kNoNode && nodes_.at(old).edge_count != 0) ++block_count;
        for (const auto old : std::span{edge_order_}.first(block_count)) {
            auto& current = nodes_.at(old);
            const auto source = current.first_edge;
            current.first_edge = destination;
            for (std::uint32_t offset = 0; offset < current.edge_count; ++offset) {
                auto edge = edges_.at(source + offset);
                if (edge.child != kNoNode) edge.child = node_map_.at(edge.child);
                edges_.at(destination++) = edge;
            }
        }
        // Ascending old IDs preserve append-only ancestry and copy only toward
        // lower addresses. No source Node is overwritten before it is consumed.
        for (NodeId old = id; old < stats_.node_count; ++old)
            if (node_map_.at(old) != kNoNode) nodes_.at(node_map_.at(old)) = nodes_.at(old);
        stats_.node_count = summary->kept_nodes;
        stats_.edge_count = summary->kept_edges;
    }
    ++generation_;
    return summary;
}
}  // namespace mouse::search
