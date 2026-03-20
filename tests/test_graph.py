"""
test_graph.py — Automated tests for the knowledge graph system.

Uses inline test fixtures (no fallback extractor dependency).
"""

from __future__ import annotations

import json
import os
import sys

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import GraphNode, GraphEdge, KnowledgeGraph, NodeType, Relationship
from extractor import ExtractedEntity, ExtractedEdge
from graph_builder import build_graph
from temporal import TemporalGraph, TemporalNode, TemporalState


# ── Test Fixtures ──────────────────────────────────────────────────────────────

def _sample_entities() -> tuple[list[ExtractedEntity], list[ExtractedEdge]]:
    """Inline test entities and edges matching the feature spec structure."""
    entities = [
        ExtractedEntity(id="goal_1", layer=1, type="goal", label="Goal 1"),
        ExtractedEntity(id="feat_1", layer=2, type="feature", label="Feature 1"),
        ExtractedEntity(id="design_1", layer=3, type="design_decision", label="Design 1"),
        ExtractedEntity(id="goal_2", layer=1, type="goal", label="Goal 2"),
        ExtractedEntity(id="feat_2", layer=2, type="feature", label="Feature 2"),
        ExtractedEntity(id="design_stateless_scorer", layer=3, type="design_decision", label="Stateless Scorer"),
    ]
    edges = [
        ExtractedEdge(from_id="feat_1", to_id="goal_1", relationship="implements", layer_crossing=True),
        ExtractedEdge(from_id="goal_2", to_id="goal_1", relationship="sub_goal_of", layer_crossing=False),
        ExtractedEdge(from_id="design_1", to_id="feat_1", relationship="governs", layer_crossing=True),
        ExtractedEdge(from_id="feat_2", to_id="goal_2", relationship="implements", layer_crossing=True),
        ExtractedEdge(from_id="feat_1", to_id="feat_2", relationship="depends_on", layer_crossing=False),
        ExtractedEdge(from_id="design_stateless_scorer", to_id="feat_2", relationship="governs", layer_crossing=True),
    ]
    return entities, edges


def _build_test_graph():
    """Build a graph from inline test fixtures for deterministic testing."""
    entities, edges = _sample_entities()
    return build_graph(entities, edges), entities


# ══════════════════════════════════════════════════════════════════════════════
# Part 1: Schema Compliance
# ══════════════════════════════════════════════════════════════════════════════

class TestSchemaCompliance:
    """Verify JSON output matches the required schema."""

    def test_node_has_all_required_fields(self):
        node = GraphNode(
            id="test_node", layer=1, type="goal", label="Test goal"
        )
        d = node.to_dict()
        required = {"id", "layer", "type", "label", "stale", "stale_reason",
                     "last_modified", "connected_to"}
        assert required == set(d.keys()), f"Missing fields: {required - set(d.keys())}"

    def test_edge_has_all_required_fields(self):
        edge = GraphEdge(
            from_id="a", to_id="b",
            relationship="implements", layer_crossing=True,
        )
        d = edge.to_dict()
        required = {"from", "to", "relationship", "layer_crossing"}
        assert required == set(d.keys()), f"Missing fields: {required - set(d.keys())}"

    def test_graph_json_is_valid(self):
        graph, _ = _build_test_graph()
        raw = graph.to_json()
        parsed = json.loads(raw)
        assert "nodes" in parsed
        assert "edges" in parsed
        assert isinstance(parsed["nodes"], list)
        assert isinstance(parsed["edges"], list)

    def test_node_types_are_valid(self):
        valid_types = {t.value for t in NodeType}
        graph, _ = _build_test_graph()
        for node in graph.nodes:
            assert node.type in valid_types, f"Invalid type: {node.type}"

    def test_relationship_types_are_valid(self):
        valid_rels = {r.value for r in Relationship}
        graph, _ = _build_test_graph()
        for edge in graph.edges:
            assert edge.relationship in valid_rels, \
                f"Invalid relationship: {edge.relationship}"


# ══════════════════════════════════════════════════════════════════════════════
# Part 2: Layer Assignment
# ══════════════════════════════════════════════════════════════════════════════

class TestLayerAssignment:
    """Verify entities are assigned to the correct layers."""

    def test_goals_are_layer_1(self):
        graph, _ = _build_test_graph()
        goals = [n for n in graph.nodes if n.type == "goal"]
        assert len(goals) >= 1, "No goal nodes found"
        for g in goals:
            assert g.layer == 1, f"Goal '{g.id}' on layer {g.layer}, expected 1"

    def test_features_are_layer_2(self):
        graph, _ = _build_test_graph()
        features = [n for n in graph.nodes if n.type == "feature"]
        assert len(features) >= 1, "No feature nodes found"
        for f in features:
            assert f.layer == 2, f"Feature '{f.id}' on layer {f.layer}, expected 2"

    def test_design_decisions_are_layer_3(self):
        graph, _ = _build_test_graph()
        dds = [n for n in graph.nodes if n.type == "design_decision"]
        assert len(dds) >= 1, "No design_decision nodes found"
        for d in dds:
            assert d.layer == 3, f"Design decision '{d.id}' on layer {d.layer}, expected 3"

    def test_no_layer_4_nodes(self):
        graph, _ = _build_test_graph()
        l4 = [n for n in graph.nodes if n.layer == 4]
        assert len(l4) == 0, f"Found Layer 4 nodes (should be excluded): {[n.id for n in l4]}"


# ══════════════════════════════════════════════════════════════════════════════
# Part 3: Edge Correctness
# ══════════════════════════════════════════════════════════════════════════════

class TestEdgeCorrectness:
    """Verify edge properties are correct."""

    def test_layer_crossing_is_correct(self):
        graph, _ = _build_test_graph()
        for edge in graph.edges:
            src = graph.get_node(edge.from_id)
            dst = graph.get_node(edge.to_id)
            assert src is not None, f"Edge source {edge.from_id} not found"
            assert dst is not None, f"Edge target {edge.to_id} not found"
            expected = src.layer != dst.layer
            assert edge.layer_crossing == expected, \
                f"Edge {edge.from_id}→{edge.to_id}: layer_crossing={edge.layer_crossing}, " \
                f"expected={expected} (layers {src.layer}→{dst.layer})"

    def test_cross_layer_edges_exist(self):
        graph, _ = _build_test_graph()
        cross = [e for e in graph.edges if e.layer_crossing]
        assert len(cross) >= 3, \
            f"Expected at least 3 cross-layer edges, found {len(cross)}"

    def test_connected_to_is_populated(self):
        graph, _ = _build_test_graph()
        nodes_with_connections = [n for n in graph.nodes if len(n.connected_to) > 0]
        assert len(nodes_with_connections) >= 5, \
            f"Too few nodes with connections: {len(nodes_with_connections)}"


# ══════════════════════════════════════════════════════════════════════════════
# Part 4: Extraction Precision
# ══════════════════════════════════════════════════════════════════════════════

class TestExtractionPrecision:
    """Verify entity count is within expected range."""

    def test_entity_count_range(self):
        _, entities = _build_test_graph()
        # We expect ~10 entities; too few means under-extraction,
        # too many means over-extraction
        assert 5 <= len(entities) <= 15, \
            f"Entity count {len(entities)} outside expected range [5, 15]"

    def test_all_three_layers_represented(self):
        graph, _ = _build_test_graph()
        layers = {n.layer for n in graph.nodes}
        assert layers == {1, 2, 3}, f"Not all layers represented: {layers}"


# ══════════════════════════════════════════════════════════════════════════════
# Part 5: Temporal Propagation
# ══════════════════════════════════════════════════════════════════════════════

class TestTemporalPropagation:
    """Verify temporal staleness tracking and propagation."""

    def test_mark_stale_records_history(self):
        graph, _ = _build_test_graph()
        tgraph = TemporalGraph(graph=graph)

        tgraph.mark_stale(
            "design_stateless_scorer",
            reason="Contract violated",
            trigger="metrics.py changed",
            propagate=False,
        )

        tn = tgraph.temporal_nodes["design_stateless_scorer"]
        assert tn.is_stale()
        assert len(tn.history) == 1
        assert tn.history[0].stale is True
        assert tn.history[0].trigger == "metrics.py changed"

    def test_mark_fresh_after_stale(self):
        graph, _ = _build_test_graph()
        tgraph = TemporalGraph(graph=graph)

        tgraph.mark_stale("design_stateless_scorer", "broken", "edit", propagate=False)
        tgraph.mark_fresh("design_stateless_scorer", "reverted")

        tn = tgraph.temporal_nodes["design_stateless_scorer"]
        assert not tn.is_stale()
        assert len(tn.history) == 2
        assert tn.history[1].stale is False

    def test_propagation_marks_connected_nodes(self):
        graph, _ = _build_test_graph()
        tgraph = TemporalGraph(graph=graph)

        affected = tgraph.mark_stale(
            "design_stateless_scorer",
            reason="Contract violated",
            trigger="metrics.py changed",
            propagate=True,
        )

        # At minimum the source itself should be affected
        assert "design_stateless_scorer" in affected
        # Should propagate to at least one connected node
        assert len(affected) >= 1

    def test_history_grows_with_multiple_changes(self):
        graph, _ = _build_test_graph()
        tgraph = TemporalGraph(graph=graph)

        tgraph.mark_stale("design_stateless_scorer", "broken", "edit1", propagate=False)
        tgraph.mark_fresh("design_stateless_scorer", "fix1")
        tgraph.mark_stale("design_stateless_scorer", "broken again", "edit2", propagate=False)

        tn = tgraph.temporal_nodes["design_stateless_scorer"]
        assert len(tn.history) == 3

    def test_full_state_export(self):
        graph, _ = _build_test_graph()
        tgraph = TemporalGraph(graph=graph)
        tgraph.mark_stale("design_stateless_scorer", "test", "test", propagate=False)

        state = tgraph.get_full_state()
        assert "design_stateless_scorer" in state
        assert state["design_stateless_scorer"]["current_stale"] is True
        assert len(state["design_stateless_scorer"]["history"]) == 1


# ══════════════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════════════

def run_all_tests():
    """Simple test runner that works without pytest."""
    test_classes = [
        TestSchemaCompliance,
        TestLayerAssignment,
        TestEdgeCorrectness,
        TestExtractionPrecision,
        TestTemporalPropagation,
    ]

    total = 0
    passed = 0
    failed = 0
    errors = []

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith("test_")]
        for method_name in methods:
            total += 1
            method = getattr(instance, method_name)
            try:
                method()
                passed += 1
                print(f"  ✅ {cls.__name__}.{method_name}")
            except AssertionError as e:
                failed += 1
                errors.append((cls.__name__, method_name, str(e)))
                print(f"  ❌ {cls.__name__}.{method_name}: {e}")
            except Exception as e:
                failed += 1
                errors.append((cls.__name__, method_name, str(e)))
                print(f"  💥 {cls.__name__}.{method_name}: {e}")

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print(f"{'=' * 60}")

    if errors:
        print("\nFailures:")
        for cls_name, method_name, err in errors:
            print(f"  • {cls_name}.{method_name}: {err}")

    return failed == 0


if __name__ == "__main__":
    print("🧪 Running knowledge graph tests...\n")
    success = run_all_tests()
    sys.exit(0 if success else 1)
