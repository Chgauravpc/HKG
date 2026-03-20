"""
models.py — Dataclass models and NetworkX-backed knowledge graph.

GraphNode and GraphEdge are plain dataclasses for serialisation.
KnowledgeGraph wraps a networkx.DiGraph and provides JSON export
matching the required schema.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import networkx as nx


# ── Enums ──────────────────────────────────────────────────────────────────────

class NodeType(str, Enum):
    GOAL = "goal"
    FEATURE = "feature"
    DESIGN_DECISION = "design_decision"


class Relationship(str, Enum):
    IMPLEMENTS = "implements"
    GOVERNS = "governs"
    DEPENDS_ON = "depends_on"
    SUB_GOAL_OF = "sub_goal_of"


# ── Nodes ──────────────────────────────────────────────────────────────────────

@dataclass
class GraphNode:
    """A single node in the knowledge graph (Layers 1-3)."""

    id: str
    layer: int                          # 1 = Intent, 2 = Feature, 3 = Design
    type: str                           # goal | feature | design_decision
    label: str
    stale: bool = False
    stale_reason: Optional[str] = None
    last_modified: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    connected_to: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ── Edges ──────────────────────────────────────────────────────────────────────

@dataclass
class GraphEdge:
    """A directed edge in the knowledge graph."""

    from_id: str                        # source node id
    to_id: str                          # target node id
    relationship: str                   # implements | governs | depends_on | sub_goal_of
    layer_crossing: bool = False        # auto-computed at construction

    def to_dict(self) -> dict:
        return {
            "from": self.from_id,
            "to": self.to_id,
            "relationship": self.relationship,
            "layer_crossing": self.layer_crossing,
        }


# ── Graph (NetworkX-backed) ───────────────────────────────────────────────────

class KnowledgeGraph:
    """
    NetworkX-backed knowledge graph.

    Stores nodes and edges in a networkx.DiGraph while exposing
    the same to_dict() / to_json() interface as before.
    """

    def __init__(self):
        self.G: nx.DiGraph = nx.DiGraph()
        self._nodes: dict[str, GraphNode] = {}
        self._edges: list[GraphEdge] = []

    # ── properties for backward compat ────────────────────────────────────

    @property
    def nodes(self) -> list[GraphNode]:
        return list(self._nodes.values())

    @property
    def edges(self) -> list[GraphEdge]:
        return list(self._edges)

    # ── mutators ──────────────────────────────────────────────────────────

    def add_node(self, node: GraphNode) -> None:
        self._nodes[node.id] = node
        self.G.add_node(
            node.id,
            layer=node.layer,
            type=node.type,
            label=node.label,
            stale=node.stale,
            stale_reason=node.stale_reason,
            last_modified=node.last_modified,
        )

    def add_edge(self, edge: GraphEdge) -> None:
        # Duplicate guard — skip if same (from, to, relationship) exists
        for existing in self._edges:
            if (existing.from_id == edge.from_id
                    and existing.to_id == edge.to_id
                    and existing.relationship == edge.relationship):
                return

        # auto-set layer_crossing
        src = self.get_node(edge.from_id)
        dst = self.get_node(edge.to_id)
        if src and dst:
            edge.layer_crossing = src.layer != dst.layer

        self._edges.append(edge)
        self.G.add_edge(
            edge.from_id,
            edge.to_id,
            relationship=edge.relationship,
            layer_crossing=edge.layer_crossing,
        )

        # maintain connected_to lists on the dataclass nodes
        if src and edge.to_id not in src.connected_to:
            src.connected_to.append(edge.to_id)
        if dst and edge.from_id not in dst.connected_to:
            dst.connected_to.append(edge.from_id)

    # ── queries ───────────────────────────────────────────────────────────

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        return self._nodes.get(node_id)

    def neighbors(self, node_id: str) -> list[str]:
        """Return IDs of all nodes connected to node_id (successors + predecessors)."""
        if node_id not in self.G:
            return []
        return list(set(self.G.successors(node_id)) | set(self.G.predecessors(node_id)))

    def shortest_path(self, source: str, target: str) -> list[str]:
        """Return shortest path between two nodes, or empty list if none."""
        try:
            return nx.shortest_path(self.G, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def subgraph_by_layer(self, layer: int) -> nx.DiGraph:
        """Return the subgraph containing only nodes of a given layer."""
        layer_nodes = [nid for nid, n in self._nodes.items() if n.layer == layer]
        return self.G.subgraph(layer_nodes).copy()

    # ── serialisation ─────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    # ── summary ───────────────────────────────────────────────────────────

    def info(self) -> str:
        return (
            f"KnowledgeGraph: {self.G.number_of_nodes()} nodes, "
            f"{self.G.number_of_edges()} edges, "
            f"density={nx.density(self.G):.3f}"
        )
