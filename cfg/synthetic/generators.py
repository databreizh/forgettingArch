# cfg/synthetic/generators.py

import random
from typing import Optional

from cfg.core.graph import CollaborativeForgettingGraph
from cfg.core.nodes import Node, RecomputabilityProfile


def generate_random_cfg(num_nodes: int, seed: int = 0) -> CollaborativeForgettingGraph:
    """
    Generates a synthetic random Collaborative Forgetting Graph (CFG) for experiments.

    Key characteristics:
    - Nodes are assigned functional types: "input", "transform", "model", "output".
    - Recompute costs are heterogeneous to provide meaningful trade-offs for 
      cost-aware algorithms.
    - The graph is a Directed Acyclic Graph (DAG): edges only point from lower 
      indices to higher indices to prevent cycles.
    """
    random.seed(seed)
    g = CollaborativeForgettingGraph()

    # ---------------------------------------------------------
    # 1) Node Creation Phase
    # ---------------------------------------------------------
    for i in range(num_nodes):
        node_id = f"n{i}"

        # Assign node types: first 10% are usually inputs, others are processing stages
        if i < max(1, num_nodes // 10):
            node_type = "input"
        else:
            node_type = random.choice(["transform", "model", "output"])

        # Symbolic ownership among 3 possible organizations
        owner = {f"Org{random.randint(1, 3)}"}

        # Define Recomputability Profile:
        # ~70% of nodes are recomputable.
        # Within those, ~30% are "heavy" (expensive to recompute).
        if random.random() < 0.7:
            recomputable = True
            if random.random() < 0.3:
                # High-cost node (e.g., complex model training)
                recompute_cost = 10.0
            else:
                # Low-cost node (e.g., simple transformation)
                recompute_cost = 1.0
        else:
            # Non-recomputable node (e.g., unique manual input)
            recomputable = False
            recompute_cost = float("inf")

        recompute_profile = RecomputabilityProfile(
            recomputable=recomputable,
            recompute_cost=recompute_cost,
        )

        # Deletion cost (uniform baseline for this generator)
        deletion_cost = 1.0

        node = Node(
            node_id=node_id,
            node_type=node_type,
            owner=owner,
            recomputability=recompute_profile,
            deletion_cost=deletion_cost,
        )
        g.add_node(node)

    # ---------------------------------------------------------
    # 2) Dependency Creation Phase (DAG Enforcement)
    # ---------------------------------------------------------
    node_ids = list(g.nodes.keys())
    
    for idx, child_id in enumerate(node_ids):
        if idx == 0:
            # The first node is always a root (no parents possible)
            continue

        # Each node is randomly assigned 0 to 3 parents from previously created nodes
        # This ensures the graph remains a DAG.
        max_parents = min(3, idx)
        num_parents = random.randint(0, max_parents)

        if num_parents == 0:
            continue

        # Select unique parents from nodes with smaller indices
        possible_parents = node_ids[:idx]
        parents = random.sample(possible_parents, num_parents)

        for parent_id in parents:
            r = random.random()
            # Distribution of edge impact types
            if r < 0.7:
                dep_type = "strong"      # Critical dependency
            elif r < 0.9:
                dep_type = "weak"        # Non-critical/optional dependency
            else:
                dep_type = "aggregated"  # Multi-source dependency

            g.add_dependency(parent_id, child_id, dep_type=dep_type)

    return g