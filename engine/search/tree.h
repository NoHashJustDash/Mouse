// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <atomic>
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
using LowId = std::uint32_t;
inline constexpr LowId kNoLow = std::numeric_limits<LowId>::max();
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
    // Lc0 Terminal::TwoFold: a draw only because the position repeats an
    // earlier one in the tree path. leaf_moves_left holds the cycle (plies).
    bool twofold : 1 = false;
    // Lc0 Node::MakeSolid: every child sits in one contiguous run of slots.
    bool solid : 1 = false;
    // Lc0 dag_classic Node::IsRepetition: a backup treated this edge as a repetition draw.
    bool repetition : 1 = false;
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
// Lc0 dag_classic LowNode: what every path reaching one position shares, namely the network's
// evaluation, the edges and their children. Values use the position's side-to-move viewpoint.
struct LowNode {
    double value_sum = 0;
    double draw_sum = 0;
    double moves_left_sum = 0;
    std::uint64_t moves_left_samples = 0;
    std::uint64_t visits = 0;
    std::uint64_t key = 0;
    EdgeId first_edge = kNoEdge;
    std::uint32_t parents = 0;
    LowId remap = kNoLow;
    std::uint16_t edge_count = 0;
    // Lc0 LowNode::IsTransposition: more than one node has ever pointed here.
    bool transposition = false;
    // A root copy carrying Dirichlet noise is never found through the table.
    bool in_table = true;
    float leaf_q = 0;
    float leaf_draw = 0;
    std::optional<float> leaf_moves_left;
    NodeProof proof;
};
// An existing child that solidify moved.
struct NodeRelocation {
    NodeId from = kNoNode;
    NodeId to = kNoNode;
};
struct TreeStats {
    std::size_t budget_bytes = 0;
    std::size_t charged_bytes = 0;
    std::uint32_t node_count = 0;
    std::uint32_t edge_count = 0;
    // Nodes and edges share one arena, so these are what fits now: the counts
    // plus what the free arena bytes could still hold of each kind.
    std::uint32_t node_capacity = 0;
    std::uint32_t edge_capacity = 0;
    // Low nodes in use and the most the table holds; both zero outside a DAG tree.
    std::uint32_t low_count = 0;
    std::uint32_t low_capacity = 0;
    // Fixed for the arena's lifetime: the most nodes it can ever hold (the size
    // of the reuse workspaces), its bytes, and the bytes nodes and edges occupy.
    std::uint32_t node_limit = 0;
    std::size_t arena_bytes = 0;
    std::size_t used_bytes = 0;
};
struct ReuseSummary {
    std::uint32_t kept_nodes = 0;
    std::uint32_t kept_edges = 0;
    // The new root was a terminal leaf that is legal to search as a root.
    bool reset_terminal_root = false;
};

// One externally synchronized mutator owns this tree. The coordinator must hold
// its tree mutex around shared access, and drain all reservations before reset
// or re_root. References/spans remain valid until reset/re_root/destruction.
// A child has exactly one incoming edge slot. Its ID is usually above its parent's, but
// solidify and shared edge blocks break that order, so re_root walks the graph.
// The coordinator verifies complete chess/history/model/configuration identity
// before re_root; a board hash alone does not authorize reusing statistics.
class Tree {
    struct FactoryToken {
       private:
        FactoryToken() = default;
        friend class Tree;
    };

   public:
    // Fixed capacity: one arena, allocated and touched here, holds nodes from its
    // start and edge blocks from its end; it is full only when they meet, so the
    // branching factor never strands half the budget (lc0 allocates both on
    // demand). Charge the arena, node maps and edge-block order sized for the
    // most nodes it can hold, sizeof(Tree), plus 64 bytes per buffer for
    // allocation/alignment bookkeeping. This is managed C++ storage, excluding
    // allocator/OS metadata and caller-owned search scratch. Budget zero or a
    // budget unable to fit the root and one edge fails explicitly. A DAG tree also
    // charges its low nodes, their table and one low ID per node from the budget.
    [[nodiscard]] static Result<std::unique_ptr<Tree>> create(std::size_t budget_bytes,
                                                              bool dag = false);
    explicit Tree(FactoryToken);
    ~Tree();
    Tree(const Tree&) = delete;
    Tree& operator=(const Tree&) = delete;
    Tree(Tree&&) = delete;
    Tree& operator=(Tree&&) = delete;

    // Concurrent calls need the caller's own lock; node() and edges() on nodes that are
    // already allocated may run alongside one.
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
    [[nodiscard]] TreeStats stats() const noexcept;
    [[nodiscard]] std::uint64_t generation() const noexcept { return generation_; }
    [[nodiscard]] Result<void> reset();
    // Validates the retained graph before mutation, then compacts edges by
    // their original offsets, then nodes. Twofold terminals are kept; a stale
    // one is reverted when selection reaches it (revert_twofold).
    // Reclaimed slots are reusable without allocating; new root always has ID0.
    [[nodiscard]] Result<ReuseSummary> re_root(NodeId id);
    // Lc0 fd71a2d classic/search.cc EnsureNodeTwoFoldCorrectForDepth with
    // classic/node.cc RevertTerminalVisits: `leaf` is a twofold terminal whose
    // earlier occurrence is now above the root. Remove its visits, draws and
    // moves-left samples (cycle + distance each) from `ancestors` (root..parent)
    // and make it an unexpanded node. Returns the visits removed. Fails without
    // changing anything for a leaf with visits in flight or inconsistent counts.
    [[nodiscard]] Result<std::uint64_t> revert_twofold(std::span<const NodeId> ancestors,
                                                       NodeId leaf);
    // Lc0 fd71a2d classic/node.cc MakeSolid: gives every edge of an expanded node a child in one
    // contiguous run of new slots, moving existing children there and leaving their old slots
    // empty until the next re_root. Unlike lc0 this may run with visits in flight below it, so
    // `moved` receives each moved child and the caller rewrites any path that names one. Returns
    // the number moved, or nothing if the node is not eligible or the arena has no room beyond
    // `reserved_edges`, which it leaves free for expansions the caller has already promised.
    [[nodiscard]] std::optional<std::size_t> solidify(NodeId id, std::span<NodeRelocation> moved,
                                                      std::size_t reserved_edges = 0);
    [[nodiscard]] bool is_dag() const noexcept { return !low_of_.empty(); }
    [[nodiscard]] LowId low_of(NodeId id) const noexcept {
        return low_of_.empty() ? kNoLow : low_of_[id];
    }
    [[nodiscard]] LowNode& low(LowId id) { return lows_.at(id); }
    [[nodiscard]] const LowNode& low(LowId id) const { return lows_.at(id); }
    // Lc0 dag_classic TranspositionTable.
    [[nodiscard]] LowId find_low(std::uint64_t key) const noexcept;
    // Creates a low node with its own edge block, without any node pointing to it yet.
    // `in_table` false keeps it out of find_low. Failure changes nothing.
    [[nodiscard]] Result<LowId> create_low(std::uint64_t key, std::span<const EdgeSeed> seeds,
                                           bool in_table = true);
    // Lc0 dag_classic Node::SetLowNode: the node shares the low node's edges and children.
    [[nodiscard]] Result<void> attach_low(NodeId id, LowId low);

   private:
    [[nodiscard]] bool quiescent() const noexcept;
    [[nodiscard]] bool valid_edge_range(const Node& current) const noexcept;
    [[nodiscard]] Result<ReuseSummary> prepare_reuse(NodeId id);
    [[nodiscard]] Result<void> validate_seeds(std::span<const EdgeSeed> seeds) const;
    void place_edges(std::span<const EdgeSeed> seeds) noexcept;
    void insert_low(LowId id) noexcept;
    void clear_storage() noexcept;
    [[nodiscard]] std::size_t free_bytes() const noexcept;
    // Slot accessors without range checks; node()/edges() check first.
    [[nodiscard]] Node* node_slot(NodeId id) const noexcept;
    [[nodiscard]] Edge* edge_slot(EdgeId id) const noexcept;

    // Nodes occupy slots [0, node_count) from the start; edges occupy slots
    // [edge_low_, edge_slots_) in sizeof(Edge) units counted from the start.
    std::unique_ptr<std::byte[]> arena_;
    std::size_t edge_slots_ = 0;
    EdgeId edge_low_ = 0;
    std::vector<NodeId> node_map_;
    std::vector<NodeId> edge_order_;
    // Blocks prepare_reuse listed in edge_order_.
    std::size_t reuse_blocks_ = 0;
    // Relaxed: picking threads check bounds on nodes they own while another allocates.
    std::atomic<std::uint32_t> node_count_{0};
    // DAG trees only: low node per node, the low nodes, and an open-addressed table of them.
    std::vector<LowId> low_of_;
    std::vector<LowNode> lows_;
    std::vector<LowId> low_table_;
    std::uint32_t low_count_ = 0;
    TreeStats stats_;
    std::uint64_t generation_ = 0;
};
}  // namespace mouse::search
