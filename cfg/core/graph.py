"""Graph structure for Collaborative Forgetting Graphs (CFG)."""

from __future__ import annotations
from typing import Dict, Set, Iterable, Optional, Tuple
from dataclasses import dataclass, field

from cfg.core.nodes import Node


@dataclass
class CollaborativeForgettingGraph:
    """
    Basic in-memory representation of a Collaborative Forgetting Graph.

    Attributes:
        nodes: Mapping of node_id (str) to Node objects.
        parents: Mapping of node_id to a set of its direct parent node_ids (incoming edges).
        children: Mapping of node_id to a set of its direct child node_ids (outgoing edges).
        edge_types: Mapping of (source_id, destination_id) tuples to their dependency type 
                    (e.g., "strong", "weak", "aggregated").
    """

    nodes: Dict[str, Node] = field(default_factory=dict)
    parents: Dict[str, Set[str]] = field(default_factory=dict)
    children: Dict[str, Set[str]] = field(default_factory=dict)
    edge_types: Dict[Tuple[str, str], str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def add_node(self, node: Node) -> None:
        """
        Add a node to the graph and initialize its adjacency sets.
        
        If the node_id already exists, the existing node object is updated/overwritten 
        with the new one, but existing edges are preserved.
        """
        nid = node.node_id
        if nid not in self.nodes:
            self.nodes[nid] = node
            self.parents.setdefault(nid, set())
            self.children.setdefault(nid, set())
        else:
            # Optional: Merge or overwrite existing node data
            self.nodes[nid] = node

    def add_edge(self, src, dst, dep_type: str = "strong") -> None:
        """
        Add a directed edge from src to dst with a specific dependency type.

        Arguments 'src' and 'dst' can be either:
            - node_id (string)
            - Node objects (the node_id will be extracted)
            
        Raises:
            ValueError: If either the source or destination node does not exist in the graph.
        """
        # Local import to prevent heavy circular dependencies
        from cfg.core.nodes import Node as CFGNode

        # Normalize inputs to node_id
        if isinstance(src, CFGNode):
            src_id = src.node_id
        else:
            src_id = src

        if isinstance(dst, CFGNode):
            dst_id = dst.node_id
        else:
            dst_id = dst

        # Integrity check: nodes must be added to the graph before creating edges
        if src_id not in self.nodes:
            raise ValueError(f"Unknown source node '{src_id}'")
        if dst_id not in self.nodes:
            raise ValueError(f"Unknown destination node '{dst_id}'")

        self.children.setdefault(src_id, set()).add(dst_id)
        self.parents.setdefault(dst_id, set()).add(src_id)
        self.edge_types[(src_id, dst_id)] = dep_type

    def add_dependency(self, src, dst, dep_type: str = "strong") -> None:
        """
        Alias for add_edge to maintain compatibility with APIs used 
        in generators and algorithms. Accepts both Node objects and node_ids.
        """
        self.add_edge(src, dst, dep_type=dep_type)


    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_node(self, node_id: str) -> Node:
        """Retrieve a Node object by its ID."""
        return self.nodes[node_id]

    def has_node(self, node_id: str) -> bool:
        """Check if a node exists in the graph."""
        return node_id in self.nodes

    def get_children(self, node_id: str) -> Set[str]:
        """Return the set of node IDs that are direct successors of the given node."""
        return self.children.get(node_id, set())

    def get_parents(self, node_id: str) -> Set[str]:
        """Return the set of node IDs that are direct predecessors of the given node."""
        return self.parents.get(node_id, set())

    def get_edge_type(self, src: str, dst: str) -> Optional[str]:
        """Return the type of the edge between src and dst, or None if no edge exists."""
        return self.edge_types.get((src, dst))

    # ------------------------------------------------------------------
    # Iteration / Statistics
    # ------------------------------------------------------------------
    def node_ids(self) -> Iterable[str]:
        """Return an iterable over all node IDs in the graph."""
        return self.nodes.keys()

    def iter_nodes(self) -> Iterable[Node]:
        """Return an iterable over all Node objects in the graph."""
        return self.nodes.values()

    @property
    def num_nodes(self) -> int:
        """Return the total count of nodes in the graph."""
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        """Return the total count of edges in the graph."""
        return len(self.edge_types)

    # Standard Python dunder methods for convenience
    def __len__(self) -> int:
        """Returns the number of nodes (allows len(graph))."""
        return self.num_nodes

    def __contains__(self, node_id: str) -> bool:
        """Allows usage of 'node_id in graph' syntax."""
        return node_id in self.nodes
    
    # Implementation within CollaborativeForgettingGraph

    def successors(self, node_id: str) -> Set[str]:
        """
        Return direct children of node_id. 
        Note: uses the 'children' mapping to provide successor IDs.
        """
        return self.children.get(node_id, set())


# Practical alias for modules expecting a CFG type reference
CFG = CollaborativeForgettingGraph