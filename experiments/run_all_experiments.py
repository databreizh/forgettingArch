import os
import json
import time
import random
from typing import Dict, Any, List, Tuple

from cfg.synthetic.generators import generate_random_cfg
from cfg.forgetting_request import ForgettingRequest
from cfg.utils.io import ensure_dir
from cfg.utils.experiment_summary import compute_experiment_summary

from cfg.propagation.greedy import GreedyMinimalPropagation
from cfg.propagation.cost_aware import CostAwarePropagation
from cfg.propagation.cluster_based import ClusterBasedPropagation

from cfg.propagation.naive import NaivePropagation

# -------------------------------------------------------------------
# Configuration globale des expériences
# -------------------------------------------------------------------

# Tailles des graphes à générer
#SIZES: List[int] = [500, 1000]  # tu peux étendre [500, 1_000, 5_000, 10_000]…
# Tailles de graphes testées (à adapter selon le temps que tu es prête à y consacrer)
#SIZES = [500, 800, 1000, 1500, 2000, 3000, 4000, 5000, 10000]

# Graines aléatoires
#SEEDS: List[int] = [0, 1, 2]

# -------------------------------------------------------------------
# Configuration globale des expériences (CAMERA-READY)
# -------------------------------------------------------------------

# Tailles des graphes : celles qui apparaîtront dans le papier
SIZES = [5000, 20000, 50000, 100000, 200000]

# Quelques seeds pour lisser la variance
SEEDS = [0, 1, 2]

# Nouveau répertoire de sortie spécifique à la version camera-ready
OUTPUT_ROOT = "experiments_output_camera"

# Algorithmes évalués dans les figures principales
ALGORITHMS: List[Tuple[str, callable]] = [
    ("naive",             lambda cfg: NaivePropagation(cfg)),
    ("greedy",            lambda cfg: GreedyMinimalPropagation(cfg)),
    ("cost_aware_a02",    lambda cfg: CostAwarePropagation(cfg, alpha=0.2)),
    ("cost_aware_a05",    lambda cfg: CostAwarePropagation(cfg, alpha=0.5)),
    ("cost_aware_a08",    lambda cfg: CostAwarePropagation(cfg, alpha=0.8)),
    ("cluster_based",     lambda cfg: ClusterBasedPropagation(cfg)),
]



# -------------------------------------------------------------------
# Construction d'une requête d'oubli
# -------------------------------------------------------------------

def build_forgetting_request(cfg, seed: int, num_sources: int = 5) -> Dict[str, Any]:
    """
    Construit une requête d'oubli pour un CFG donné.

    On choisit comme candidats les nœuds sans parents (racines de provenance).
    S'il y en a trop peu, on prend l'ensemble des nœuds comme pool.
    La requête retournée est un dict compatible avec ForgettingRequest.
    """
    random.seed(seed)

    # Candidats = nœuds sans parents
    root_nodes = [
        node_id
        for node_id in cfg.nodes.keys()
        if len(cfg.get_parents(node_id)) == 0
    ]

    if len(root_nodes) < num_sources:
        candidates = list(cfg.nodes.keys())
    else:
        candidates = root_nodes

    if not candidates:
        raise ValueError("No candidate nodes available to build a forgetting request.")

    k = min(num_sources, len(candidates))
    initial_nodes = random.sample(candidates, k)

    return {
        "initial_nodes": initial_nodes,
        "mode": "strict",
    }


# -------------------------------------------------------------------
# Nom d'expérience
# -------------------------------------------------------------------

def make_experiment_name(size: int, seed: int, algo_name: str) -> str:
    return f"n{size}_seed{seed}_{algo_name}"


# -------------------------------------------------------------------
# Exécution d'une expérience
# -------------------------------------------------------------------

def run_single_experiment(
    size: int,
    seed: int,
    algo_name: str,
    algo_ctor,
    output_root: str,
) -> None:
    """
    Lance une expérience pour (taille, seed, algorithme).

    Produit :
      - un fichier JSON de résumé : <experiment_name>_summary.json
      (les events.jsonl sont gérés dans les classes d'algo si tu les as implémentés).
    """
    # 1) Générer le CFG synthétique
    cfg = generate_random_cfg(num_nodes=size, seed=seed)

    # 2) Construire la requête d'oubli (dict)
    req_dict = build_forgetting_request(cfg, seed)
    initial_nodes = req_dict["initial_nodes"]
    mode = req_dict.get("mode", "strict")

    # 3) Créer l'objet ForgettingRequest
    request = ForgettingRequest(
        initial_nodes=initial_nodes,
        mode=mode,
    )

    # 4) Instancier l'algo à partir du constructeur fourni
    algo = algo_ctor(cfg)

    experiment_name = make_experiment_name(size, seed, algo_name)

    # 5) Répertoire de sortie spécifique
    out_dir = os.path.join(output_root, f"n{size}", algo_name)
    ensure_dir(out_dir)

    # 6) Lancer l'algorithme avec mesure du temps
    start = time.perf_counter()
    deleted, recomputed = algo.run(request)
    runtime = time.perf_counter() - start

    # 7) Métadonnées
    metadata: Dict[str, Any] = {
        "graph_size": size,
        "seed": seed,
        "algo_name": algo_name,
    }
    if hasattr(algo, "alpha"):
        metadata["alpha"] = algo.alpha

    # 8) Construire le résumé expérimental
    summary = compute_experiment_summary(
        experiment=experiment_name,
        cfg=cfg,
        deleted=deleted,
        recomputed=recomputed,
        runtime_sec=runtime,
        initial_nodes=initial_nodes,
        algorithm_name=algo_name,
        metadata=metadata,
        run_exact_if_small=True,
        max_exact_nodes=26,
    )

    # 9) Sauvegarde du résumé
    summary_path = os.path.join(out_dir, f"{experiment_name}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary.to_json(), f, indent=2)

    print(f"[OK] {experiment_name}  -> {summary_path}")


# -------------------------------------------------------------------
# Point d'entrée
# -------------------------------------------------------------------

def main() -> None:
    ensure_dir(OUTPUT_ROOT)

    for size in SIZES:
        for seed in SEEDS:
            for algo_name, algo_ctor in ALGORITHMS:
                print(f"--> Running {algo_name} on n={size}, seed={seed} ...")
                run_single_experiment(
                    size=size,
                    seed=seed,
                    algo_name=algo_name,
                    algo_ctor=algo_ctor,
                    output_root=OUTPUT_ROOT,
                )


if __name__ == "__main__":
    main()
