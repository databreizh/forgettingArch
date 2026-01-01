"""Forgetting request representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Set


@dataclass(frozen=True)
class ForgettingRequest:
    """
    Represents a collaborative forgetting request on a CFG.
    
    This object defines the starting point and the constraints of a forgetting
    operation within the graph.

    Attributes
    ----------
    initial_nodes : Set[str]
        Set of node IDs that are explicitly requested to be forgotten (the seeds).
    mode : str
        The forgetting policy to apply (e.g., "strict", "weak", "scoped").
        "strict" is the default, usually implying recursive deletion of descendants.
    scope : Optional[Set[str]]
        A restrictive set of node IDs. If provided, the forgetting operation 
        is only enforced within this boundary. If None, the request is global.
    actor : Optional[str]
        Identifier for the entity (user, organization, or service) issuing the request.
        Useful for auditing or permission-based forgetting.
    """

    initial_nodes: Set[str] = field(default_factory=set)
    mode: str = "strict"
    scope: Optional[Set[str]] = None
    actor: Optional[str] = None

    @classmethod
    def from_iterable(
        cls,
        node_ids: Iterable[str],
        mode: str = "strict",
        scope: Optional[Iterable[str]] = None,
        actor: Optional[str] = None,
    ) -> "ForgettingRequest":
        """
        Helper factory method to build a request from any iterable collection.

        Args:
            node_ids: An iterable (list, generator, etc.) of strings to forget.
            mode: The forgetting enforcement policy.
            scope: An optional iterable to define the impact boundary.
            actor: The entity responsible for the request.

        Returns:
            A new immutable ForgettingRequest instance.
        """
        initial = set(node_ids)
        scope_set = set(scope) if scope is not None else None
        return cls(initial_nodes=initial, mode=mode, scope=scope_set, actor=actor)