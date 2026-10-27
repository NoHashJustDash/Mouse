// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "engine/chess/types.h"
#include "engine/search/proof.h"
#include "engine/util/result.h"

namespace mouse::search {
using NodeId = std::uint32_t;
using EdgeId = std::uint32_t;
inline constexpr NodeId kNoNode = std::numeric_limits<NodeId>::max();
inline constexpr EdgeId kNoEdge = std::numeric_limits<EdgeId>::max();
enum class NodeState : std::uint8_t { Unexpanded, Pending, Expanded, Terminal, Frontier };

struct Node {
    double value_sum = 0;
    double draw_sum = 0;
    double moves_left_sum = 0;
    // A mean is usable only when every completed visit supplied known M.
    std::uint64_t moves_left_samples = 0;
    std::uint64_t visits = 0;
    std::uint32_t in_flight = 0;
    EdgeId first_edge = kNoEdge;
    std::uint16_t edge_count = 0;
    NodeState state = NodeState::Unexpanded;
    bool contextual_draw = false;
    float leaf_q = 0;
    float leaf_draw = 0;
    std::optional<float> leaf_moves_left;
    NodeProof proof;
};
struct Edge {
    chess::Move move;
    NodeId child = kNoNode;
    std::uint16_t prior_bits = 0;
};
struct EdgeSeed {
    chess::Move move;
    std::uint16_t prior_bits = 0;
};
struct TreeStats {
    std::size_t budget_bytes = 0;
    std::size_t charged_bytes = 0;
    std::uint32_t node_count = 0;
    std::uint32_t edge_count = 0;
    std::uint32_t node_capacity = 0;
    std::uint32_t edge_capacity = 0;
};
struct ReuseSummary {
    std::uint32_t kept_nodes = 0;
    std::uint32_t kept_edges = 0;
    bool discarded_context = false;
};

// One externally synchronized mutator owns this tree. The coordinator must hold
// its tree mutex around shared access, and drain all reservations before reset
// or re_root. References/spans remain valid until reset/re_root/destruction.
// Child IDs are append-only (child > parent), with exactly one incoming edge.
// The coordinator verifies complete chess/history/model/configuration identity
// before re_root; a board hash alone does not authorize reusing statistics.
class Tree {
    struct FactoryToken {
       private:
        FactoryToken() = default;
        friend class Tree;
    };

   public:
    // Fixed capacity: eight edges per node slot. Charge actual capacities of
    // nodes, edges, node maps and edge-block order, sizeof(Tree), plus 64 bytes
    // per buffer for allocation/alignment bookkeeping. This is managed C++
    // storage, excluding allocator/OS metadata and caller-owned search scratch.
    // Budget zero or a budget unable to fit one complete slot fails explicitly.
    [[nodiscard]] static Result<std::unique_ptr<Tree>> create(std::size_t budget_bytes);
    explicit Tree(FactoryToken);
    ~Tree();
    Tree(const Tree&) = delete;
    Tree& operator=(const Tree&) = delete;
    Tree(Tree&&) = delete;
    Tree& operator=(Tree&&) = delete;

    [[nodiscard]] Result<NodeId> allocate_node();
    // Nonempty unique move seeds, finite nonnegative IEEE half priors <=1.
    // Pending expansion retains its existing reservation. Failure changes no
    // node, edge, count or generation; successful operations never allocate.
    [[nodiscard]] Result<void> expand(NodeId id, std::span<const EdgeSeed> seeds);
    // Invalid IDs/ranges throw out_of_range rather than expose arena slack.
    [[nodiscard]] Node& node(NodeId id);
    [[nodiscard]] const Node& node(NodeId id) const;
    [[nodiscard]] std::span<Edge> edges(NodeId id);
    [[nodiscard]] std::span<const Edge> edges(NodeId id) const;
    [[nodiscard]] NodeId root() const noexcept { return 0; }
    [[nodiscard]] TreeStats stats() const noexcept { return stats_; }
    [[nodiscard]] std::uint64_t generation() const noexcept { return generation_; }
    [[nodiscard]] Result<void> reset();
    // Validates the retained graph before mutation. Contextual tree-twofold
    // statistics anywhere in the retained subtree discard the entire subtree.
    // Otherwise compact edges by their original offsets, then compact nodes.
    // Reclaimed slots are reusable without allocating; new root always has ID0.
    [[nodiscard]] Result<ReuseSummary> re_root(NodeId id);

   private:
    [[nodiscard]] bool quiescent() const noexcept;
    [[nodiscard]] bool valid_edge_range(const Node& current) const noexcept;
    [[nodiscard]] Result<ReuseSummary> prepare_reuse(NodeId id);
    void clear_storage() noexcept;

    std::vector<Node> nodes_;
    std::vector<Edge> edges_;
    std::vector<NodeId> node_map_;
    std::vector<NodeId> edge_order_;
    TreeStats stats_;
    std::uint64_t generation_ = 0;
};
}  // namespace mouse::search
