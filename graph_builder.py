"""
graph_builder.py — Builds the knowledge graph from extracted entities.

Constructs Layers 1-3 using the nodes and edges provided directly
by the LLM extraction engine.
"""

from __future__ import annotations

from models import GraphNode, GraphEdge, KnowledgeGraph
from extractor import ExtractedEntity, ExtractedEdge


def validate_edges(
    edges: list[ExtractedEdge],
    entities: list[ExtractedEntity]
) -> tuple[list[ExtractedEdge], list[str]]:
    """
    Validate edges before adding to graph.
    Returns (valid_edges, list_of_warnings).
    """
    valid = []
    warnings = []
    node_ids = {e.id for e in entities}
    node_layer = {e.id: e.layer for e in entities}
    allowed_relationships = {
        "implements", "governs", "depends_on", "sub_goal_of"
    }

    for edge in edges:
        # 1. Both nodes must exist
        if edge.from_id not in node_ids:
            warnings.append(f"SKIP: unknown source '{edge.from_id}'")
            continue
        if edge.to_id not in node_ids:
            warnings.append(f"SKIP: unknown target '{edge.to_id}'")
            continue

        # 2. Relationship must be in schema
        if edge.relationship not in allowed_relationships:
            warnings.append(
                f"SKIP: invalid relationship '{edge.relationship}' "
                f"on {edge.from_id} → {edge.to_id}"
            )
            continue

        # 3. No self-loops
        if edge.from_id == edge.to_id:
            warnings.append(f"SKIP: self-loop on '{edge.from_id}'")
            continue

        # 4. Layer coherence checks
        from_layer = node_layer[edge.from_id]
        to_layer = node_layer[edge.to_id]

        if edge.relationship == "sub_goal_of" and (from_layer != 1 or to_layer != 1):
            warnings.append(
                f"SKIP: sub_goal_of must be L1→L1, "
                f"got L{from_layer}→L{to_layer} "
                f"({edge.from_id} → {edge.to_id})"
            )
            continue

        if edge.relationship == "governs" and from_layer != 3:
            warnings.append(
                f"SKIP: governs must come from L3, "
                f"got L{from_layer} ({edge.from_id})"
            )
            continue

        if edge.relationship == "implements" and from_layer != 2:
            warnings.append(
                f"SKIP: implements must come from L2, "
                f"got L{from_layer} ({edge.from_id})"
            )
            continue

        valid.append(edge)

    return valid, warnings


def build_graph(
    entities: list[ExtractedEntity],
    edges: list[ExtractedEdge]
) -> KnowledgeGraph:
    """
    Build a KnowledgeGraph directly from extracted entities and edges.
    """
    graph = KnowledgeGraph()

    # ── 1. Create nodes ────────────────────────────────────────────────────
    for ent in entities:
        node = GraphNode(
            id=ent.id,
            layer=ent.layer,
            type=ent.type,
            label=ent.label,
        )
        graph.add_node(node)

    # ── 2. Validate format before adding ───────────────────────────────────
    valid_edges, warnings = validate_edges(edges, entities)

    if warnings:
        print("\n⚠️  Edge validation warnings:")
        for w in warnings:
            print(f"   {w}")

    # ── 3. Create edges ────────────────────────────────────────────────────
    for e in valid_edges:
        edge = GraphEdge(
            from_id=e.from_id,
            to_id=e.to_id,
            relationship=e.relationship,
        )
        # add_edge auto-computes layer_crossing based on node layers
        graph.add_edge(edge)

    return graph


def print_graph_summary(graph: KnowledgeGraph) -> None:
    """Print summary statistics for the graph."""
    nodes_by_layer = {}
    for n in graph.nodes:
        nodes_by_layer.setdefault(n.layer, []).append(n)

    edges_by_rel = {}
    for e in graph.edges:
        edges_by_rel.setdefault(e.relationship, []).append(e)

    cross_layer = [e for e in graph.edges if e.layer_crossing]

    print("\n" + "=" * 60)
    print("📊 GRAPH SUMMARY")
    print("=" * 60)
    print(f"  Total nodes: {len(graph.nodes)}")
    for layer in sorted(nodes_by_layer):
        print(f"    Layer {layer}: {len(nodes_by_layer[layer])} nodes")
    print(f"  Total edges: {len(graph.edges)}")
    for rel in sorted(edges_by_rel):
        print(f"    {rel}: {len(edges_by_rel[rel])} edges")
    print(f"  Cross-layer edges: {len(cross_layer)}")
    print(f"  NetworkX info: {graph.info()}")
    print("=" * 60)
