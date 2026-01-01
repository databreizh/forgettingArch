from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Set


@dataclass
class RecomputabilityProfile:
    """
    Profile defining the recomputability logic of a node.

    Attributes:
        recomputable: Boolean indicating if the node can theoretically be recalculated.
        recompute_cost: Abstract cost used by cost-aware algorithms to evaluate 
                        regeneration impact.
    """
    recomputable: bool = False
    recompute_cost: float = 0.0

    def is_recomputable(self, cfg: Any, node_id: str, deleted: Set[str]) -> bool:
        """
        Check if the node can be recomputed based on the current graph state.

        The node is considered recomputable if:
            1. Its 'recomputable' flag is set to True.
            2. It has at least one parent (dependency source) that has not been deleted.

        Args:
            cfg: The Collaborative Forgetting Graph instance.
            node_id: The unique identifier of the node to check.
            deleted: A set of node IDs that have already been marked for deletion.

        Returns:
            bool: True if recomputation is possible, False otherwise.
        """
        if not self.recomputable:
            return False

        parents = cfg.get_parents(node_id)
        if not parents:
            # No ancestors: nothing exists to trigger or feed the recalculation
            return False

        # If at least one parent remains in the graph, the node can (in principle) be recomputed
        return any(p not in deleted for p in parents)

    def effective_recompute_cost(self, cfg: Any, node_id: str, deleted: Set[str]) -> float:
        """
        Calculate the (potentially adjusted) cost of recomputing this node.

        Current Implementation:
            Returns the static recompute_cost value.
        
        Future Improvements:
            This could be adjusted based on graph depth, subgraph size, 
            or recursive costs of parents.
        """
        return self.recompute_cost