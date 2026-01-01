# run_real_flows.py

import os
import json
import time
import random
from typing import Dict, Any, List, Tuple

from cfg.utils.io import ensure_dir
from cfg.utils.experiment_summary import compute_experiment_summary
from cfg.forgetting_request import ForgettingRequest

from cfg.propagation.greedy import GreedyMinimalPropagation
from cfg.propagation.cost_aware import CostAwarePropagation
from cfg.propagation.cluster_based import ClusterBasedPropagation

from cfg.real_flows import build_crowd_flow, build_ml_flow


OUTPUT_ROOT = "experiments_real"
SEEDS = [0, 1, 2]   # quelques seeds pour varier les requêtes

ALGORITHMS: List[Tuple[str, callable]] = [
    ("greedy",         lambda cfg: GreedyMinimalPropagation(cfg)),
    ("cost_aware",     lambda cfg: CostAwarePropagation(cfg, alpha=0.5)),
    ("cluster_based",  lambda cfg: ClusterBasedPropagation(cfg)),
]

REAL_FLOWS: List[Tuple[str, callable]] = [
    ("crowd", build_crowd_flow),
    ("ml",    build_ml_flow),
]


def build_forgetting_request_real(cfg, seed: int, num_sources: int = 5) -> Dict[str, Any]:
    """
    Même idée que pour les synthétiques, mais ici on peut
    éventuellement restreindre à certains types de nœuds
    (p.ex. inputs, raw, annotations).
    """
    random.seed(seed)

    # On prend comme candidats les noeuds de type "input" ou "raw" ou "annotation"
    candidates = [
        nid
        for nid, node in cfg.nodes.items()
        if node.node_type in {"input", "raw", "annotation"}
    ]
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
    req_dict = build_forgetting_request_real(cfg, seed)
    initial_nodes = req_dict["initial_nodes"]
    mode = req_dict["mode"]

    request = ForgettingRequest(
        initial_nodes=initial_nodes,
        mode=mode,
    )

    algo = algo_ctor(cfg)

    experiment_name = f"{flow_name}_seed{seed}_{algo_name}"

    out_dir = os.path.join(output_root, flow_name, algo_name)
    ensure_dir(out_dir)

    start = time.perf_counter()
    deleted, recomputed = algo.run(request)
    runtime = time.perf_counter() - start

    metadata: Dict[str, Any] = {
        "flow_name": flow_name,
        "seed": seed,
        "algo_name": algo_name,
        "graph_size": len(cfg.nodes),
    }
    if hasattr(algo, "alpha"):
        metadata["alpha"] = algo.alpha

    summary = compute_experiment_summary(
        experiment=experiment_name,
        cfg=cfg,
        deleted=deleted,
        recomputed=recomputed,
        runtime_sec=runtime,
        initial_nodes=initial_nodes,
        algorithm_name=algo_name,
        metadata=metadata,
        run_exact_if_small=True,   # ces graphes ne sont pas gigantesques
        max_exact_nodes=120,       # à ajuster si besoin
    )

    summary_path = os.path.join(out_dir, f"{experiment_name}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary.to_json(), f, indent=2)

    print(f"[OK] {experiment_name} -> {summary_path}")


def main() -> None:
    ensure_dir(OUTPUT_ROOT)

    for flow_name, flow_builder in REAL_FLOWS:
        cfg = flow_builder(seed=0)   # structure du flow
        print(f"=== Flow {flow_name} (|V|={len(cfg.nodes)}) ===")
        for seed in SEEDS:
            for algo_name, algo_ctor in ALGORITHMS:
                print(f"--> {flow_name} / seed={seed} / algo={algo_name}")
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
