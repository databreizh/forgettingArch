from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from cfg.utils.summary import ExperimentSummary


# ------------------------------------------------------------
# Naive Baseline: Strong dependency closure (no recomputation)
# ------------------------------------------------------------
def naive_forgetting_strong_closure(cfg, initial_nodes: List[str]) -> Tuple[Set[str], Set[str]]:
    """
    Computes the simplest propagation baseline:
    - Deletes initial_nodes.
    - Propagates impact: if a strong parent 'u' is deleted, the child 'v' 
      is also deleted.
    - Completely ignores recomputation options.
    """
    deleted: Set[str] = set(initial_nodes)
    recomputed: Set[str] = set()

    # Using a queue for Breadth-First Search (BFS) propagation
    queue = list(initial_nodes)

    while queue:
        u = queue.pop(0)
        for v in cfg.get_children(u):
            if v in deleted:
                continue
            # Structural propagation rule for the naive baseline
            if cfg.get_edge_type(u, v) == "strong":
                deleted.add(v)
                queue.append(v)

    return deleted, recomputed


# ------------------------------------------------------------
# Raw Cost Calculation: Sum of deletion and recomputation costs
# ------------------------------------------------------------
def _compute_raw_cost(cfg, deleted_ids: List[str], recomputed_ids: List[str]) -> float:
    """
    Calculates the total weighted cost of a forgetting solution.
    
    Formula: Total = sum(deletion_costs) + sum(recompute_costs).
    Includes robust fallbacks if cost attributes are missing from nodes.
    """
    total = 0.0

    # Summing deletion costs for truly removed nodes
    for nid in deleted_ids:
        node = cfg.get_node(nid)
        dc = getattr(node, "deletion_cost", 1.0)
        total += float(dc if dc is not None else 1.0)

    # Summing recomputation costs for regenerated nodes
    for nid in recomputed_ids:
        node = cfg.get_node(nid)
        rp = getattr(node, "recomputability", None)
        rc = getattr(rp, "recompute_cost", None)
        if rc is None:
            # If a node is recomputed but has no cost, assume it's impossible (inf)
            rc = float("inf")
        total += float(rc)

    return total


# ------------------------------------------------------------
# Exact Solver Stub (Optional)
# ------------------------------------------------------------
def _compute_exact_solution_if_needed(
    cfg,
    initial_nodes: List[str],
    run_exact_if_small: bool,
    max_exact_nodes: int,
):
    """
    Placeholder for an optimal solver (e.g., Integer Linear Programming).
    Currently returns None. Can be used to plug in an exact solver 
    for small-scale graph verification.
    """
    _ = (cfg, initial_nodes, run_exact_if_small, max_exact_nodes)
    return None, None, None


# ------------------------------------------------------------
# Experiment Summary Construction
# ------------------------------------------------------------
def compute_experiment_summary(
    experiment: str,
    cfg,
    deleted,
    recomputed,
    runtime_sec: float,
    initial_nodes,
    algorithm_name: str,
    metadata: Dict[str, Any],
    run_exact_if_small: bool = False,
    max_exact_nodes: int = 26,
) -> ExperimentSummary:
    """
    Constructs an ExperimentSummary from raw results.
    
    This function normalizes outputs and computes several key performance 
    indicators (KPIs):
    1. Raw vs. Normalized Costs.
    2. Comparison against the Naive Baseline.
    3. Optimality gap (if an exact solution is provided).
    """

    # Ensure all outputs are JSON-serializable lists
    if not isinstance(deleted, list):
        deleted = list(deleted)
    if not isinstance(recomputed, list):
        recomputed = list(recomputed)
    if not isinstance(initial_nodes, list):
        initial_nodes = list(initial_nodes)

    graph_size = len(cfg.nodes)
    deleted_size = len(deleted)
    recomputed_size = len(recomputed)
    initial_size = len(initial_nodes)

    # Calculate current solution costs
    raw_cost = _compute_raw_cost(cfg, deleted, recomputed)
    normalized_cost = raw_cost / max(1, graph_size)
    weighted_cost = raw_cost / max(1, initial_size)

    # Generate and score the Naive Baseline for comparison
    naive_deleted_set, naive_recomputed_set = naive_forgetting_strong_closure(cfg, initial_nodes)
    naive_deleted = list(naive_deleted_set)
    naive_recomputed = list(naive_recomputed_set)

    naive_deleted_size = len(naive_deleted)
    naive_raw_cost = _compute_raw_cost(cfg, naive_deleted, naive_recomputed)

    # Relative cost comparison (Alg Cost / Naive Cost)
    if naive_raw_cost > 0 and naive_raw_cost != float("inf"):
        relative_cost_naive: Optional[float] = raw_cost / naive_raw_cost
    else:
        relative_cost_naive = None

    # Optional: Compute Exact Solution (if the graph is small enough)
    exact_solution, exact_deleted_size, exact_raw_cost = _compute_exact_solution_if_needed(
        cfg=cfg,
        initial_nodes=initial_nodes,
        run_exact_if_small=run_exact_if_small,
        max_exact_nodes=max_exact_nodes,
    )

    if exact_solution is not None and not isinstance(exact_solution, list):
        exact_solution = list(exact_solution)

    # Optimality gap calculation (Alg Cost / Optimal Cost)
    if exact_raw_cost is not None and exact_raw_cost > 0:
        relative_cost = raw_cost / exact_raw_cost
    else:
        relative_cost = None

    return ExperimentSummary(
        experiment=experiment,
        graph_size=graph_size,
        algorithm_name=algorithm_name,
        deleted=deleted,
        recomputed=recomputed,
        initial_nodes=initial_nodes,
        runtime_sec=runtime_sec,
        deleted_size=deleted_size,
        recomputed_size=recomputed_size,
        initial_size=initial_size,
        raw_cost=raw_cost,
        normalized_cost=normalized_cost,
        weighted_cost=weighted_cost,
        relative_cost=relative_cost,
        exact_deleted_size=exact_deleted_size,
        exact_solution=exact_solution,
        naive_deleted_size=naive_deleted_size,
        naive_raw_cost=naive_raw_cost,
        relative_cost_naive=relative_cost_naive,
        metadata=metadata or {},
    )