from __future__ import annotations

from typing import List, Set, Tuple, Dict

from cfg.core.graph import CollaborativeForgettingGraph
from cfg.core.nodes import Node
from cfg.forgetting_request import ForgettingRequest


class ClusterBasedPropagation:
    """
    Implements cluster-based forgetting propagation.
    
    The algorithm follows two main phases:
    1. Identification: Groups nodes into clusters (connected components) based 
       strictly on "strong" dependency edges.
    2. Propagation: Decisions to delete or recompute are made at the cluster 
       level rather than for individual nodes.
    """

    def __init__(self, cfg: CollaborativeForgettingGraph):
        """
        Initialize the propagator and pre-calculate the cluster structure.
        """
        self.cfg = cfg
        self.clusters, self.node_to_cluster = self._build_clusters()

    # ------------------------------------------------------------------
    # Build clusters as connected components on strong deps
    # ------------------------------------------------------------------
    def _build_clusters(self) -> Tuple[List[Set[str]], Dict[str, int]]:
        """
        Partition the graph into clusters using a Depth-First Search (DFS).
        
        A cluster is defined as a set of nodes connected by 'strong' edges, 
        regardless of direction (undirected connectivity within the strong subgraph).
        
        Returns:
            A tuple containing:
            - A list of sets, where each set contains node IDs belonging to a cluster.
            - A dictionary mapping each node_id to its cluster index (ID).
        """
        nodes = list(self.cfg.nodes.keys())
        visited: Set[str] = set()
        clusters: List[Set[str]] = []
        node_to_cluster: Dict[str, int] = {}

        for nid in nodes:
            if nid in visited:
                continue

            # Start a new cluster discovery (DFS)
            stack = [nid]
            comp: Set[str] = set()
            visited.add(nid)

            while stack:
                v = stack.pop()
                comp.add(v)

                # Explore parents via strong edges
                for u in self.cfg.get_parents(v):
                    if self.cfg.get_edge_type(u, v) == "strong" and u not in visited:
                        visited.add(u)
                        stack.append(u)

                # Explore children via strong edges
                for w in self.cfg.get_children(v):
                    if self.cfg.get_edge_type(v, w) == "strong" and w not in visited:
                        visited.add(w)
                        stack.append(w)

            cid = len(clusters)
            clusters.append(comp)
            for v in comp:
                node_to_cluster[v] = cid

        return clusters, node_to_cluster

    # ------------------------------------------------------------------
    # Cluster-level closure
    # ------------------------------------------------------------------
    def _closure_clusters(
        self, initial: Set[str]
    ) -> Tuple[Set[str], Set[str]]:
        """
        Compute the transitive closure of deletions at the cluster level.
        
        This method iteratively finds clusters where at least one node is 
        invalidated by a deleted parent. It then decides whether to delete 
        the entire cluster or recompute it by comparing aggregated costs.
        """
        deleted: Set[str] = set(initial)
        recomputed: Set[str] = set()
        processed_clusters: Set[int] = set()
        changed = True

        while changed:
            changed = False

            for cid, cluster_nodes in enumerate(self.clusters):
                if cid in processed_clusters:
                    continue

                # Skip if the cluster is already fully accounted for
                if cluster_nodes.issubset(deleted) or cluster_nodes.issubset(recomputed):
                    processed_clusters.add(cid)
                    continue

                # Check if any node in the cluster has a 'strong' parent that was deleted
                invalid = False
                for v in cluster_nodes:
                    if v in deleted or v in recomputed:
                        continue
                    parents = self.cfg.get_parents(v)
                    if not parents:
                        continue
                    
                    strong_parents_deleted = any(
                        self.cfg.get_edge_type(u, v) == "strong" and u in deleted
                        for u in parents
                    )
                    if strong_parents_deleted:
                        invalid = True
                        break

                if not invalid:
                    continue

                # Cost-Benefit Analysis: Aggregate costs for the entire cluster
                total_del_cost = 0.0
                total_recomp_cost = 0.0
                all_recomputable = True

                for v in cluster_nodes:
                    node: Node = self.cfg.get_node(v)
                    # Accumulate deletion cost
                    del_cost = getattr(node, "deletion_cost", 1.0)
                    total_del_cost += float(del_cost)

                    # Accumulate recomputability profile
                    rp = getattr(node, "recomputability", None)
                    if rp and getattr(rp, "recomputable", False):
                        rc = getattr(rp, "recompute_cost", None)
                        if rc is None:
                            rc = 0.0
                        total_recomp_cost += float(rc)
                    else:
                        # If one node in a strong cluster is not recomputable,
                        # the whole cluster usually cannot be recomputed.
                        all_recomputable = False
                        total_recomp_cost += float("inf")

                # Decision: Recompute or Delete the whole cluster
                if all_recomputable and total_recomp_cost <= total_del_cost:
                    recomputed.update(cluster_nodes)
                else:
                    deleted.update(cluster_nodes)

                processed_clusters.add(cid)
                changed = True

        return deleted, recomputed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, request: ForgettingRequest) -> Tuple[Set[str], Set[str]]:
        """
        Execute the cluster-based forgetting process for a given request.

        Args:
            request: A ForgettingRequest containing the initial node IDs to forget.

        Returns:
            A tuple (deleted_set, recomputed_set) containing the final 
            state of all affected nodes.
        """
        initial = set(request.initial_nodes)
        deleted, recomputed = self._closure_clusters(initial)
        return deleted, recomputed