"""
extractor.py — LLM-based entity extraction using Groq API.

Sends the feature spec to Groq (llama-3.3-70b-versatile) with a structured
prompt and parses the returned JSON into a list of extracted entities and edges.
Requires GROQ_API_KEY environment variable and the `groq` package.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

# Load .env from project root
load_dotenv(Path(__file__).parent / ".env")


# ── Extraction result ──────────────────────────────────────────────────────────

@dataclass
class ExtractedEntity:
    """An entity extracted from the feature spec."""
    id: str
    layer: int
    type: str
    label: str

@dataclass
class ExtractedEdge:
    """An edge extracted directly by the LLM."""
    from_id: str
    to_id: str
    relationship: str
    layer_crossing: bool


# ── Prompt ─────────────────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """\
You are a knowledge graph architect. Extract a minimal, precise set of entities
from a natural language spec. Quality over quantity.

LAYERS AND TYPES:
  Layer 1 → type "goal"             — why this exists; the intent
  Layer 2 → type "feature"          — what the system does; observable behaviour
  Layer 3 → type "design_decision"  — how it's built; architectural choices

- Layer 1 goal labels must use intent language: "Detect X behavior", "Identify X", 
    "Ensure X". Never use the same label as a Layer 2 feature node.
- If the spec explicitly describes how outputs are combined or composed,
    extract that as a design_decision node, not just a depends_on edge.
- If the spec explicitly describes how outputs are combined or composed,
    extract that as a design_decision node, not just a depends_on edge.        
WHAT TO EXTRACT:
  - Goals: the top-level purpose + meaningful sub-goals
  - Features: the concrete, named capabilities the system exposes — specific enough
    that if one changed independently, it would not affect the others.
    Never use a feature node that just restates the top-level goal in different words.
    If the spec names specific components (linter, scanner, scorer), those are your features.
  - Design decisions: explicit architectural choices that constrain how things are built
  - Sub-goals: if the spec lists named signals or dimensions the system measures,
    each is a sub-goal in L1 AND a feature in L2. The L1 node is the intent
    (detect X), the L2 node is the capability (X signal module).
  - Always extract the system's primary output as a feature node if the spec
    names it explicitly (e.g. "session score", "review report", "notification").
  - Always extract the system's input pipeline as a feature node if the spec
    describes how data enters the system.
  - Use feeds_data_to edges to show data flow between features — which features
    feed into which.
  - Do NOT link features directly to the top-level goal if a sub-goal already
    connects them. Let the hierarchy carry the relationship.  

WHAT TO SKIP:
  - Success criteria / perf numbers → encode in the feature label, not a separate node
  - Implementation details that restate a design decision
  - Anything a developer couldn't independently verify or challenge
  - Outcome states or boolean conditions (e.g. "converging vs diverging") are not
    sub-goals — they are properties of the top-level goal. Skip them.
  - Do not invent feature nodes that aren't explicitly named in the spec.
    If you find yourself naming a feature after the overall system or top-level goal,
    that's a sign you're restating the goal, not extracting a feature. 
  - Named files, modules, or code components (e.g. recorder.py, metrics.py) are
    Layer 4 nodes — do not extract them as Layer 2 features. If the spec names
    a file, extract the capability it provides, not the file itself.   
EDGES:
  sub_goal_of  — sub-goal → parent goal (L1 → L1)
  implements   — feature → sub-goal it directly realises (NOT the top-level goal directly,
                 unless the feature has no sub-goal). The sub_goal_of chain carries
                 the connection upward to the top-level goal automatically.
  governs      — design_decision → every feature it constrains (layer-crossing).
                 Read the spec to determine scope. If the decision applies to
                 multiple features, draw a governs edge to each one individually.
                 Never assume a design decision governs only one feature unless
                 the spec explicitly scopes it that way.
  feeds_data_to — A feeds_data_to B means data flows FROM A INTO B.
                 A is the producer. B is the consumer.
                 Only draw feeds_data_to edges between feature nodes you have
                 already extracted — do not create new nodes to satisfy
                 a feeds_data_to edge.
CRITICAL: Never link a feature directly to the top-level goal if it already
implements a sub-goal. Let the hierarchy do the work.

CRITICAL: feeds_data_to points TO the data consumer. If B receives data from A,
the edge is A feeds_data_to B.
  
CRITICAL: Every sub-goal node must have at least one "implements" edge coming from a feature.
A sub-goal with no incoming "implements" edge is always wrong.

CRITICAL: The feature that represents the system's primary output MUST have
an implements edge to the top-level goal.

Return ONLY valid JSON, no markdown fences, no explanation:
{{
  "reasoning": {{
    "included": [{{"label": "...", "why": "..."}}],
    "excluded": [{{"label": "...", "why": "..."}}]
  }},
  "nodes": [
    {{"id": "...", "layer": 1, "type": "goal", "label": "..."}}
  ],
  "edges": [
    {{"from": "...", "to": "...", "relationship": "...", "layer_crossing": true}}
  ]
}}

Feature specification:
---
{spec}
---
"""

# ── LLM extraction via Groq ───────────────────────────────────────────────────

def _extract_with_llm(spec: str) -> dict:
    """Call Groq API to extract entities and edges from the spec."""
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY environment variable is not set. "
            "Please set it before running: "
            "  export GROQ_API_KEY=your_key_here  (Linux/Mac)\n"
            "  $env:GROQ_API_KEY='your_key_here'  (PowerShell)"
        )

    client = Groq(api_key=api_key)

    print("🔗 Using Groq API (llama-3.3-70b-versatile) for extraction...")

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a precise knowledge-graph extraction engine. "
                           "Return only valid JSON, no markdown fences."
            },
            {
                "role": "user",
                "content": EXTRACTION_PROMPT.format(spec=spec),
            },
        ],
        temperature=0.1,
        max_tokens=2048,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if the model wraps them anyway
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    return json.loads(raw)


# ── Public API ─────────────────────────────────────────────────────────────────

def extract_entities(spec: str) -> tuple[list[ExtractedEntity], list[ExtractedEdge], dict]:
    """
    Extract entities from a feature spec using the Groq LLM API.

    Returns (entities, edges, reasoning) where reasoning is a dict of inclusions/exclusions.
    Raises EnvironmentError if GROQ_API_KEY is not set.
    """
    result = _extract_with_llm(spec)

    entities = []
    for raw in result.get("nodes", []):
        entities.append(ExtractedEntity(
            id=raw["id"],
            layer=raw["layer"],
            type=raw["type"],
            label=raw["label"],
        ))

    edges = []
    for raw in result.get("edges", []):
        from_id = raw["from"]
        to_id = raw["to"]
        rel = raw["relationship"]

        # Parse LLM-friendly 'feeds_data_to' (A -> B) into 
        # schema-compliant 'depends_on' (B -> A) to fix LLM directionality bias
        if rel == "feeds_data_to":
            edges.append(ExtractedEdge(
                from_id=to_id,
                to_id=from_id,
                relationship="depends_on",
                layer_crossing=raw.get("layer_crossing", False)
            ))
        else:
            edges.append(ExtractedEdge(
                from_id=from_id,
                to_id=to_id,
                relationship=rel,
                layer_crossing=raw.get("layer_crossing", False)
            ))

    reasoning = result.get("reasoning", {})
    return entities, edges, reasoning
