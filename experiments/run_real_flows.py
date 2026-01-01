"""
Experimental evaluation suite for Collaborative Forgetting on real-world dataflows.
This script orchestrates the execution of propagation algorithms (Greedy, Cost-Aware, 
Cluster-Based) across semi-realistic pipelines (Crowdsourcing and ML).
"""

import os
import json
import time
import random
from typing import Dict, Any, List, Tuple

# Core framework components
from cfg.utils.io import ensure_dir
from cfg.utils.experiment_summary import compute_experiment_summary
from cfg.forgetting_request import ForgettingRequest

# Algorithm implementations
from cfg.propagation.greedy import GreedyMinimalPropagation
from cfg.propagation.cost_aware import CostAwarePropagation
from cfg.propagation.cluster_based import ClusterBasedPropagation

# Pipeline templates
from cfg.real_flows import build_crowd_flow, build_ml_flow

# Configuration for reproducibility
OUTPUT_ROOT = "experiments_real"
SEEDS = [0, 1, 2]  # Seeds used to generate diverse forgetting requests

# Algorithm suite configuration
ALGORITHMS: List[Tuple[str, callable]] = [
    ("greedy",         lambda cfg: GreedyMinimalPropagation(cfg)),
    ("cost_aware",     lambda cfg: CostAwarePropagation(cfg, alpha=0.5)),
    ("cluster_based",  lambda cfg: ClusterBasedPropagation(cfg)),
]

# Evaluation scenarios
REAL_FLOWS: List[Tuple[str, callable]] = [
    ("crowd", build_crowd_flow),
    ("ml",    build_ml_flow),
]


def build_forgetting_request_real(cfg, seed: int, num_sources: int = 5) -> Dict[str, Any]:
    """
    Generates a targeted forgetting request by sampling from primary data sources.
    
    In multi-party flows, forgetting requests typically originate from actors
    providing 'raw' or 'input' data. This function restricts candidates to 
    these specific node types to simulate realistic GDPR-like scenarios.
    """
    random.seed(seed)

    # Filter nodes to identify eligible source data (annotations, raw inputs)
    candidates = [
        nid
        for nid, node in cfg.nodes.items()
        if node.node_type in {"input", "raw", "annotation"}
    ]
    
    # Fallback to all nodes if no specific source types are defined
    if not candidates:
        candidates = list(cfg.nodes.keys())

    k = min(num_sources, len(candidates))
    initial_nodes = random.sample(candidates, k)

    return {
        "initial_nodes": initial_nodes,
        "mode": "strict",
    }


def run_single_real_experiment(
    flow_name: str,
    cfg,
    seed: int,
    algo_name: str,
    algo_ctor,
    output_root: str,
) -> None:
    """
    Executes a single algorithm instance and persists the metrics.
    
    This function tracks execution time, identifies deleted/recomputed nodes,
    and computes the 'Optimality Gap' by comparing results with an exact
    solver for moderately sized graphs.
    """
    req_dict = build_forgetting_request_real(cfg, seed)
    initial_nodes = req_dict["initial_nodes"]
    mode = req_dict["mode"]

    request = ForgettingRequest(
        initial_nodes=initial_nodes,
        mode=mode,
    )

    # Algorithm initialization
    algo = algo_ctor(cfg)
    experiment_name = f"{flow_name}_seed{seed}_{algo_name}"

    # Results directory structure: <root>/<flow>/<algorithm>/
    out_dir = os.path.join(output_root, flow_name, algo_name)
    ensure_dir(out_dir)

    # Performance measurement phase
    start = time.perf_counter()
    deleted, recomputed = algo.run(request)
    runtime = time.perf_counter() - start

    # Metadata enrichment for post-hoc analysis
    metadata: Dict[str, Any] = {
        "flow_name": flow_name,
        "seed": seed,
        "algo_name": algo_name,
        "graph_size": len(cfg.nodes),
    }
    
    # Capture hyper-parameters if present (e.g., alpha for cost-aware)
    if hasattr(algo, "alpha"):
        metadata["alpha"] = algo.alpha

    # Summarization and benchmarking
    summary = compute_experiment_summary(
        experiment=experiment_name,
        cfg=cfg,
        deleted=deleted,
        recomputed=recomputed,
        runtime_sec=runtime,
        initial_nodes=initial_nodes,
        algorithm_name=algo_name,
        metadata=metadata,
        run_exact_if_small=True,  # Attempt exact resolution for benchmarking
        max_exact_nodes=120,      # Computational threshold for the exact solver
    )

    # Persistence to JSON for later aggregation
    summary_path = os.path.join(out_dir, f"{experiment_name}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary.to_json(), f, indent=2)

    print(f"[PROCESS] Executed {experiment_name} -> Saved to {summary_path}")


def main() -> None:
    """
    Main execution loop. Iterates over all flow templates, algorithm 
    variants, and random seeds to produce a comprehensive evaluation.
    """
    ensure_dir(OUTPUT_ROOT)

    for flow_name, flow_builder in REAL_FLOWS:
        # Construct the graph structure once per flow
        cfg = flow_builder(seed=0)
        print(f"\n=== Evaluating Pipeline: {flow_name} (|V|={len(cfg.nodes)}) ===")
        
        for seed in SEEDS:
            for algo_name, algo_ctor in ALGORITHMS:
                print(f"  --> Seed {seed} | Algorithm: {algo_name}")
                run_single_real_experiment(
                    flow_name=flow_name,
                    cfg=cfg,
                    seed=seed,
                    algo_name=algo_name,
                    algo_ctor=algo_ctor,
                    output_root=OUTPUT_ROOT,
                )


if __name__ == "__main__":
    main()