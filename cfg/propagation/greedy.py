"""
Greedy minimal propagation for collaborative forgetting.

This is a simple, baseline implementation:
- We start from the initial forgetting set S (the seeds).
- We propagate impact along STRONG dependency edges.
- We stop when we reach a fixed point (no more nodes are invalidated).

Recomputability and costs are deliberately ignored here; they are handled
in the cost-aware variant.
"""

from __future__ import annotations

from typing import Set, Iterable, Tuple

from cfg.core.graph import CFG


class GreedyMinimalPropagation:
    """
    Greedy minimal propagation algorithm.

    This class provides a foundational approach to forgetting by identifying 
    all downstream nodes that are structurally invalidated by the removal of 
    their strong dependencies.

    The object is tied to a given CFG instance. To execute the process, 
    call `run(request)`.
    """

    def __init__(self, cfg: CFG) -> None:
        """
        Initialize the propagator with a specific Collaborative Forgetting Graph.
        """
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------
    def _closure_strong_dependencies(self, initial: Set[str]) -> Set[str]:
        """
        Compute the transitive closure of `initial` under strong dependencies.

        Propagation Rule:
        If node 'u' is marked as deleted and a directed edge (u -> v) is of type 
        "strong", then node 'v' must also be marked as deleted. 
        This process repeats until a fixed point is reached (the set of deleted 
        nodes no longer grows).

        Args:
            initial: The set of node IDs that are the source of the forgetting request.

        Returns:
            The complete set of node IDs that must be deleted.
        """
        deleted: Set[str] = set(initial)
        changed = True

        while changed:
            changed = False

            # Scan all nodes in the graph to find newly invalidated candidates
            for v in list(self.cfg.node_ids()):
                if v in deleted:
                    continue

                # If at least one strong parent is deleted, v becomes invalid
                for u in self.cfg.get_parents(v):
                    edge_type = self.cfg.get_edge_type(u, v)
                    if edge_type == "strong" and u in deleted:
                        deleted.add(v)
                        changed = True
                        break  # No need to check other parents for this node

        return deleted

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, request) -> Tuple[Set[str], Set[str]]:
        """
        Run greedy forgetting for a single request.

        This method fulfills the standard API for CFG algorithms, returning 
        both deleted and recomputed sets, although the recomputed set is 
        always empty for this specific variant.

        Parameters
        ----------
        request :
            Any object with at least:
              - request.initial_nodes : Iterable[str]
              - request.mode : str (currently unused, assumed "strict")

        Returns
        -------
        deleted : Set[str]
            Set of node_ids that are deleted or invalidated.
        recomputed : Set[str]
            Set of node_ids that are recomputed. 
            Note: This greedy variant does not support recomputation; 
            the set is returned empty for API consistency.
        """
        # Normalize the initial set from the request
        initial_nodes = set(request.initial_nodes)

        # Mode is currently ignored; "strict" enforcement is assumed.
        # mode = getattr(request, "mode", "strict")

        # Perform greedy propagation along strong edges
        deleted = self._closure_strong_dependencies(initial_nodes)

        # Recomputation is not handled by this baseline algorithm
        recomputed: Set[str] = set()

        return deleted, recomputed