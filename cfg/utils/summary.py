from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentSummary:
    """
    Standardized container for collaborative forgetting experiment results.
    
    This class captures the outcome of an algorithm's execution on a specific 
    CFG, including node counts, computational costs, and comparative metrics 
    against baselines (Naive and Exact).
    """
    experiment: str          # Name of the test scenario or dataset
    graph_size: int          # Total number of nodes in the graph (|V|)
    algorithm_name: str      # ID of the algorithm (e.g., 'greedy', 'cost-aware')

    deleted: List[str]       # List of node IDs actually removed
    recomputed: List[str]    # List of node IDs re-generated
    initial_nodes: List[str] # The seed nodes from the forgetting request

    runtime_sec: float       # Execution time in seconds

    # Quantitative metrics (useful for CSV export and plotting)
    deleted_size: int
    recomputed_size: int
    initial_size: int

    # Financial/Resource costs
    raw_cost: float          # Sum(deletion_costs) + Sum(recompute_costs)
    normalized_cost: float   # Cost relative to the total graph size
    weighted_cost: float     # Cost relative to the size of the initial request

    # Benchmarking: Comparison with an Optimal (Exact) Solution
    relative_cost: Optional[float] = None      # Ratio: Algorithm Cost / Exact Cost
    exact_deleted_size: Optional[int] = None   # Number of nodes deleted in the optimal solution
    exact_solution: Optional[List[str]] = None # List of node IDs in the optimal solution

    # Benchmarking: Comparison with the Naive Baseline (Simple Propagation)
    naive_deleted_size: Optional[int] = None   # Impact size if simple propagation was used
    naive_raw_cost: Optional[float] = None     # Total cost of the naive approach
    relative_cost_naive: Optional[float] = None # Ratio: Algorithm Cost / Naive Cost

    # Additional contextual data (e.g., hyperparameters, graph density)
    metadata: Dict[str, Any] = None

    def to_json(self) -> Dict[str, Any]:
        """
        Converts the dataclass instance into a JSON-serializable dictionary.
        
        Returns:
            A dictionary containing all experiment metrics, ready for 
            database storage or web visualization.
        """
        d = asdict(self)
        if d["metadata"] is None:
            d["metadata"] = {}
        return d