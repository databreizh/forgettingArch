# cfg/propagation/naive.py

from __future__ import annotations

from typing import Set, Tuple, Iterable, List
from cfg.forgetting_request import ForgettingRequest
from cfg.core.graph import CollaborativeForgettingGraph


def naive_forgetting(
    cfg: CollaborativeForgettingGraph,
    initial_nodes: Iterable[str],
) -> Set[str]:
    """
    Naive forgetting (Breadth-First Search approach): 
    Deletes the initial nodes and EVERY node reachable from them via outgoing edges,
    regardless of dependency type, recomputability, or costs.

    This serves as the "maximum propagation" baseline used to evaluate the 
    minimality and efficiency of more advanced algorithms.
    """
    to_visit: List[str] = list(initial_nodes)
    deleted: Set[str] = set(initial_nodes)

    while to_visit:
        v = to_visit.pop()
        # successors = all nodes that directly depend on v
        for succ in cfg.successors(v):
            if succ not in deleted:
                deleted.add(succ)
                to_visit.append(succ)

    return deleted


class NaivePropagation:
    """
    Naive Baseline: Simple cascading deletion under strong dependencies.
    
    This implementation follows a basic structural approach:
    - No recomputation decisions are made.
    - No costs are taken into account.
    - No clustering is performed.
    """

    def __init__(self, cfg: CollaborativeForgettingGraph):
        """
        Initialize the naive propagator with a reference graph.
        """
        self.cfg = cfg

    def run(self, request: ForgettingRequest) -> Tuple[Set[str], Set[str]]:
        """
        Execute the naive propagation for a given request.
        
        It computes the transitive closure of the initial set strictly 
        following 'strong' edges.
        
        Returns:
            A tuple (deleted, recomputed). 
            Note: The recomputed set is always empty in this baseline.
        """
        deleted: Set[str] = set(request.initial_nodes)
        recomputed: Set[str] = set()  # Baseline: never performs recomputation

        changed = True
        while changed:
            changed = False

            # Scan all nodes; if a strong parent is deleted, delete the child
            for v in list(self.cfg.nodes.keys()):
                if v in deleted:
                    continue

                for u in self.cfg.get_parents(v):
                    # Propagation rule: strong dependency on a deleted node triggers deletion
                    if self.cfg.get_edge_type(u, v) == "strong" and u in deleted:
                        deleted.add(v)
                        changed = True
                        break

        return deleted, recomputed