"""
Metrics and baselines for collaborative forgetting experiments.

Contains:
- naive_forgetting : "Naive strong propagation" baseline.
- exact_mcf       : Exact Minimal Forgetting Set (MCF) via brute force (for small graphs).
- graph_stats     : Structural statistics for the Collaborative Forgetting Graph (CFG).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Set, Iterable, Optional, Any

from ..dependencies import DependencyType


# -------------------------------------------------------------------
#  1. Naive Baseline: Simple Strong Propagation
# -------------------------------------------------------------------

def naive_forgetting(cfg, initial_nodes: Iterable[str]) -> Set[str]:
    """
    Naive Baseline: Starting from set S, recursively deletes all nodes
    that depend on deleted nodes via STRONG dependencies.

    Assumptions: 
    - No recomputation allowed.
    - Costs are ignored.
    """
    deleted: Set[str] = set(initial_nodes)
    queue = list(initial_nodes)

    while queue:
        v = queue.pop()
        for child in cfg.get_children(v):
            cid = child.id
            if cid in deleted:
                continue
            
            # Check edge type to determine mandatory propagation
            dep_type = cfg.get_dep_type(v, cid)
            if dep_type == DependencyType.STRONG:
                deleted.add(cid)
                queue.append(cid)

    return deleted


# -------------------------------------------------------------------
#  2. Exact MCF via Brute Force (Small-scale graphs only)
# -------------------------------------------------------------------

def _is_consistent_no_recompute(cfg, deleted: Set[str]) -> bool:
    """
    Tests structural consistency in a "no-recompute" model:
    For every STRONG edge u->v, if 'u' is deleted and 'v' is not,
     the configuration is inconsistent.

    This simplified version is used to:
    - Compare against naive_forgetting.
    - Evaluate the approximation quality of heuristics in a 
      strictly structural model.
    """
    for v_id in cfg.nodes.keys():
        if v_id in deleted:
            continue
        
        # v_id is kept; verify all its STRONG parents are also kept
        for parent in cfg.get_parents(v_id):
            pid = parent.id
            if pid in deleted and cfg.get_dep_type(pid, v_id) == DependencyType.STRONG:
                return False
    return True


@dataclass
class ExactMCFResult:
    """
    Container for exact Minimal Forgetting Set (MCF) calculation results.
    """
    solution: Set[str]
    size: int
    explored_subsets: int
    optimal: bool


def exact_mcf(
    cfg,
    initial_nodes: Iterable[str],
    max_nodes: int = 26,
    max_explored: Optional[int] = None,
) -> ExactMCFResult:
    """
    Calculates the exact minimal forgetting set (by cardinality) 
    using a brute-force approach (no-recompute model).

    WARNING:
    - Exponential complexity [O(2^V)] -> Only use for very small graphs.
    - 'max_nodes' limits |V| to prevent system crashes.

    Strategy:
    - Iterate through the universe of nodes U = V.
    - Find the smallest X ⊇ S such that _is_consistent_no_recompute(cfg, X) is True.
    - Exploration is done by increasing size: |S|, |S|+1, ..., |V|.

    Args:
        cfg: The CollaborativeForgettingGraph instance.
        initial_nodes: S (the initial seed set to forget).
        max_nodes: Threshold to reject graphs too large for brute force.
        max_explored: Hard limit on the number of subsets to explore.

    Returns:
        An ExactMCFResult object containing the solution and metadata.
    """
    S: Set[str] = set(initial_nodes)
    all_nodes: Set[str] = set(cfg.nodes.keys())

    if len(all_nodes) > max_nodes:
        raise ValueError(
            f"exact_mcf: graph too large for brute force "
            f"(|V|={len(all_nodes)} > max_nodes={max_nodes})"
        )

    others = sorted(all_nodes - S)
    explored = 0

    # If S is already consistent, it is the minimal solution
    if _is_consistent_no_recompute(cfg, S):
        return ExactMCFResult(solution=S, size=len(S), explored_subsets=1, optimal=True)

    # Search for supersets of S: S ∪ T, where |T|=k
    for k in range(1, len(others) + 1):
        for combo in combinations(others, k):
            explored += 1
            
            # Safety break for long-running explorations
            if max_explored is not None and explored > max_explored:
                return ExactMCFResult(
                    solution=S.union(combo),
                    size=len(S) + k,
                    explored_subsets=explored,
                    optimal=False,
                )

            candidate = S.union(combo)
            if _is_consistent_no_recompute(cfg, candidate):
                # The first solution found at size k is guaranteed to be minimal
                return ExactMCFResult(
                    solution=candidate,
                    size=len(candidate),
                    explored_subsets=explored,
                    optimal=True,
                )

    # Default fallback to full deletion (should technically not be reached)
    return ExactMCFResult(
        solution=all_nodes,
        size=len(all_nodes),
        explored_subsets=explored,
        optimal=False,
    )


# -------------------------------------------------------------------
#  3. Structural Graph Statistics
# -------------------------------------------------------------------

def _compute_depths(cfg) -> Dict[str, int]:
    """
    Calculates the depth (longest path length from a root/input) of 
    every node in a DAG using DFS with memoization.

    Formula:
    depth(v) = 0 if v has no parents
             = 1 + max(depth(parent)) otherwise
    """
    depths: Dict[str, int] = {}

    def dfs(v_id: str) -> int:
        if v_id in depths:
            return depths[v_id]
        
        parents = list(cfg.get_parents(v_id))
        if not parents:
            depths[v_id] = 0
        else:
            depths[v_id] = 1 + max(dfs(p.id) for p in parents)
        return depths[v_id]

    for v_id in cfg.nodes.keys():
        dfs(v_id)

    return depths


def graph_stats(cfg) -> Dict[str, Any]:
    """
    Computes key structural statistics for a CFG.

    Metrics:
      - num_nodes     : |V| (Total nodes)
      - num_edges     : |E| (Total edges)
      - avg_indegree  : Mean of in-degrees
      - avg_outdegree : Mean of out-degrees
      - max_depth     : Length of the longest path in the DAG
      - overlap_ratio : Ratio of nodes with in-degree ≥ 2 (shared dependencies)

    These statistics are vital for characterizing datasets in 
    the evaluation/results section of the paper.
    """
    node_ids = list(cfg.nodes.keys())
    n = len(node_ids)

    indegrees = []
    outdegrees = []
    num_edges = 0

    for v_id in node_ids:
        parents = list(cfg.get_parents(v_id))
        children = list(cfg.get_children(v_id))

        indegrees.append(len(parents))
        outdegrees.append(len(children))
        num_edges += len(children)

    avg_indegree = sum(indegrees) / n if n > 0 else 0.0
    avg_outdegree = sum(outdegrees) / n if n > 0 else 0.0

    # Overlap Ratio tracks how many nodes consume data from multiple sources
    overlap_ratio = (
        sum(1 for d in indegrees if d >= 2) / n if n > 0 else 0.0
    )

    depths = _compute_depths(cfg)
    max_depth = max(depths.values()) if depths else 0

    return {
        "num_nodes": n,
        "num_edges": num_edges,
        "avg_indegree": avg_indegree,
        "avg_outdegree": avg_outdegree,
        "max_depth": max_depth,
        "overlap_ratio": overlap_ratio,
    }