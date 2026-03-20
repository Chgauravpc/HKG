"""
main.py — CLI entry point for the hierarchical knowledge graph system.

Runs entity extraction and graph construction.
"""

from __future__ import annotations

import json
import os
import sys

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import KnowledgeGraph
from extractor import extract_entities
from graph_builder import build_graph, print_graph_summary


SPEC_FILE = os.path.join(os.path.dirname(__file__), "feature_spec.txt")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "graph_output.json")


def main() -> None:
    # ── Load spec ──────────────────────────────────────────────────────────
    with open(SPEC_FILE, "r", encoding="utf-8") as f:
        spec = f.read()

    # ══════════════════════════════════════════════════════════════════════
    # PART 1 — Entity Extraction
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("🔍 PART 1 — ENTITY EXTRACTION")
    print("=" * 60)

    entities, edges, reasoning = extract_entities(spec)

    print(f"\n✅ Extracted {len(entities)} entities:\n")
    for ent in entities:
        print(f"  [{ent.layer}] {ent.type:18s}  {ent.id}")
        print(f"      Label: {ent.label}")
        print()

    print(f"\n✅ Extracted {len(edges)} edges:\n")
    for edge in edges:
        print(f"  {edge.from_id} --[{edge.relationship}]--> {edge.to_id}")

    if reasoning.get("included"):
        print(f"\n✅ Included reasoning:\n")
        for inc in reasoning["included"]:
            print(f"  • {inc['label']}\n    Why: {inc['why']}\n")

    if reasoning.get("excluded"):
        print(f"❌ Excluded candidates:\n")
        for exc in reasoning["excluded"]:
            print(f"  • {exc['label']}\n    Why: {exc['why']}\n")

    # ══════════════════════════════════════════════════════════════════════
    # PART 2 — Graph Construction
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("🏗️  PART 2 — GRAPH CONSTRUCTION")
    print("=" * 60)

    graph = build_graph(entities, edges)
    print_graph_summary(graph)

    # Write JSON output
    graph_json = graph.to_json(indent=2)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(graph_json)
    print(f"\n💾 Graph written to: {OUTPUT_FILE}")

    # Print the full JSON
    print("\n📄 Full graph JSON:")
    print(graph_json)

if __name__ == "__main__":
    main()
