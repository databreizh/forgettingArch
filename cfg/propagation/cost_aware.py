"""
Cost-aware propagation for collaborative forgetting.

Simplified version:
- Starts from nodes explicitly requested for forgetting.
- Propagates impact along "strong" dependency edges.
- For each invalidated node, a local decision is made between:
    * Deleting it (added to deleted set)
    * Recomputing it (added to recomputed set)
  based on a comparison of deletion vs. recomputation costs.

This variant does not seek global optimality but provides actionable metrics 
(deleted_size vs. recomputed_size) for experimental benchmarks.
"""

from __future__ import annotations

from typing import Set, Tuple

from cfg.core.graph import CFG
from cfg.core.nodes import Node


class CostAwarePropagation:
    """
    Cost-aware forgetting algorithm logic.

    Attributes:
        cfg: The provenance graph (CFG) on which the operation is performed.
        alpha: A weighting parameter for future use to balance structural impact 
               and costs. In this version, decisions remain purely local.
    """

    def __init__(self, cfg: CFG, alpha: float = 0.7) -> None:
        """
        Initialize the propagator with a graph and an optional alpha factor.
        """
        self.cfg = cfg
        self.alpha = alpha

    # ------------------------------------------------------------------
    # Helper: Access a Node object by its ID
    # ------------------------------------------------------------------
    def _get_node(self, node_id: str) -> Node:
        """
        Internal helper to retrieve a Node from the CFG.
        Assumes self.cfg.nodes is a dictionary mapping IDs to Node instances.
        """
        return self.cfg.nodes[node_id]

    # ------------------------------------------------------------------
    # Core Logic: Propagation under strong dependencies
    # ------------------------------------------------------------------
    def _closure_cost_aware(self, initial: Set[str]) -> Tuple[Set[str], Set[str]]:
        """
        Computes the fixed-point closure of the initial set under strong dependencies.
        
        For every node invalidated by the deletion of a parent, the algorithm 
        performs a local cost-benefit analysis to choose the cheapest path 
        between full deletion and data regeneration.

        Returns:
            A tuple (deleted, recomputed) containing sets of node IDs.
        """
        deleted: Set[str] = set(initial)
        recomputed: Set[str] = set()
        changed = True

        while changed:
            changed = False

            # Iterate through all nodes to find those affected by current deletions
            for v in list(self.cfg.node_ids()):
                if v in deleted or v in recomputed:
                    continue

                # Check if any "strong" parent has been marked for deletion
                parents = self.cfg.get_parents(v)
                if not parents:
                    continue

                strong_parents_deleted = any(
                    self.cfg.get_edge_type(u, v) == "strong" and u in deleted
                    for u in parents
                )
                if not strong_parents_deleted:
                    continue

                # Node v is now invalid. Decision logic: delete vs. recompute
                node = self._get_node(v)
                
                # Retrieve costs with reasonable fallback defaults
                deletion_cost = getattr(node, "deletion_cost", 1.0)

                # Access nested recomputability profile attributes
                recomputability_profile = getattr(node, "recomputability", None)
                recomputable = getattr(
                    recomputability_profile,
                    "recomputable",
                    False,
                )
                recompute_cost = getattr(
                    recomputability_profile,
                    "recompute_cost",
                    float("inf"),
                )

                # Local optimization decision
                if recomputable and recompute_cost <= deletion_cost:
                    # Choice: Regenerate the node (recompute)
                    recomputed.add(v)
                else:
                    # Choice: Remove the node and propagate impact to its children
                    deleted.add(v)

                changed = True

        return deleted, recomputed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, request) -> Tuple[Set[str], Set[str]]:
        """
        Execute the cost-aware propagation for a specific forgetting request.

        Args:
            request: A ForgettingRequest-like object containing:
                     - initial_nodes: iterable of node IDs to forget.
                     - mode: string policy (e.g., "strict").

        Returns:
            A tuple (deleted, recomputed) representing the final impact 
            of the forgetting operation.
        """
        initial_nodes = set(request.initial_nodes)
        # mode = getattr(request, "mode", "strict")  # Reserved for future usage

        deleted, recomputed = self._closure_cost_aware(initial_nodes)
        return deleted, recomputed