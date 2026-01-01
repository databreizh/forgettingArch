# cfg/nodes.py
"""
Node and recomputability definitions for the Collaborative Forgetting Graph.
"""

from dataclasses import dataclass, field
from typing import Set, Optional


@dataclass
class RecomputabilityProfile:
    """
    Defines if and how a node can be recomputed after deletion.
    
    Attributes:
        recomputable: Whether the node data can be regenerated.
        recompute_cost: Numerical cost (time, compute, etc.) to regenerate the node.
        reason: Optional string explaining why a node is not recomputable.
    """
    recomputable: bool = True
    recompute_cost: Optional[float] = 0.0
    reason: str | None = None

    def __post_init__(self) -> None:
        """
        Normalization logic: if a node is not recomputable or the cost is missing,
        ensure the cost is explicitly set to 0.0 to avoid computation errors.
        """
        if (not self.recomputable) or (self.recompute_cost is None):
            self.recompute_cost = 0.0


@dataclass(eq=True)
class Node:
    """
    Basic node type for the CFG.

    Attributes
    ----------
    node_id : str
        Unique identifier for the node (e.g., "n42").
    node_type : str
        Logical category of the node (e.g., "input", "transform", "model", "output").
    owner : Set[str]
        Set of actors or organizations owning the node (e.g., {"Org1"}).
    recomputability : RecomputabilityProfile
        Profile detailing the node's recomputability status and costs.
    deletion_cost : float
        Abstract cost of deletion, used for cost-aware forgetting strategies.
    """
    node_id: str
    node_type: str = "generic"
    owner: Set[str] = field(default_factory=set)
    recomputability: RecomputabilityProfile = field(
        default_factory=RecomputabilityProfile
    )
    deletion_cost: float = 1.0

    def __hash__(self) -> int:
        """
        Hash based on the unique logical identifier.
        Allows Node instances to be used as dictionary keys or in sets.
        """
        return hash(self.node_id)

    @property
    def id(self) -> str:
        """
        Backward compatibility alias for code using node.id instead of node.node_id.
        """
        return self.node_id


@dataclass(eq=True)
class VersionedNode(Node):
    """
    Versioned node representation for tracking explicit iterations (v^(t)).

    Extends the base Node class with an optional version field. 
    Can be treated as a standard Node if the versioning is not required by 
    the specific algorithm or graph traversal.
    """
    version: Optional[int] = None