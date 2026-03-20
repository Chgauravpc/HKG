# Temporal Extension — How to Make the Knowledge Graph Time-Aware

When a node's `stale` field flips to `true`, the graph must answer not just
"what is stale?" but "when did it become stale, why, and what was its history?"
The right data structure for this is an **append-only event log per node** —
essentially event sourcing at the node level.

Each node maintains a `history` list of `TemporalState` snapshots:
`{stale, stale_reason, timestamp, trigger}`. Every state change appends a new
record; nothing is overwritten. This gives you a full audit trail: you can
reconstruct the node's entire lifecycle by replaying its history.

**Why event sourcing over snapshots?** In a live agent session, events arrive
as a continuous stream — file edits, test runs, git commits. Event sourcing
mirrors this naturally: each trigger maps to exactly one history entry. Full
graph snapshots would be wasteful because most nodes don't change on a given
event; you'd be duplicating thousands of unchanged nodes just to record one
staleness flip.

**Propagation as temporal events.** When a Layer 4 file changes, the staleness
propagates upward through `governs` and `implements` edges. Each propagation
step records its own `TemporalState` with a `trigger` field that traces back
to the originating change. This creates a causal chain: you can trace any
stale node back to the root file change that caused it.

**Trade-offs.** Append-only logs grow unbounded in long sessions. A practical
system would use **windowed retention** — keep the last N entries or entries
from the last T minutes, and compact older history into summary records
(e.g., "was stale 12 times between T1 and T2"). For query performance, you
maintain the current state on the node itself (`stale`, `stale_reason`) as a
materialised view, and only hit the history log for auditing or debugging.
This keeps read-path queries O(1) while preserving full temporal fidelity
where needed.
