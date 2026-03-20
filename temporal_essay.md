# Temporal Extension: Making the Knowledge Graph Time-Aware

When a node's `stale` field flips to `true`, the graph must answer not just "what is stale?" but also:

- when did it become stale,
- why did it become stale, and
- what is its full history?

The right data structure is an append-only event log per node, effectively event sourcing at the node level.

## Per-Node Temporal History

Each node maintains a history list of `TemporalState` snapshots:

```text
{ stale, stale_reason, timestamp, trigger }
```

Every state change appends a new record; nothing is overwritten. This gives a full audit trail, so the node's lifecycle can be reconstructed by replaying history.

## Why Append-Only Ledger vs. Versioned Node Snapshots?

An alternative is copying the entire node on every staleness change, like Git commits with one full snapshot per transition.

That makes point-in-time queries simple, but it:

- bloats storage proportionally to graph size, and
- makes current-state reads expensive because the latest version must be located.

The append-only ledger keeps current-state reads at `O(1)` because `stale` and `stale_reason` on the node always represent the latest state. History queries become `O(n)` over the history array, which is the right tradeoff when:

- live consumers check current state constantly, and
- history is queried mostly for debugging propagation chains.

## Why Event Sourcing vs. Full Graph Snapshots?

In a live agent session, events arrive continuously (file edits, test runs, Git commits). Event sourcing maps naturally to this stream: each trigger maps to one history entry.

Full graph snapshots are wasteful because most nodes do not change for a given event. You would duplicate many unchanged nodes to record a single staleness flip.

## Propagation as Temporal Events

When a Layer 4 file changes, staleness propagates upward through `governs` and `implements` edges.

Each propagation step records its own `TemporalState` with a `trigger` that traces back to the originating change. This creates a causal chain: any stale node can be traced to the root file change.

Each history entry should also include `propagation_depth`:

- `depth = 0`: node went stale directly,
- `depth = 1+`: node became stale because an upstream dependency became stale.

This separates root cause from downstream symptoms. Without this distinction, every node in a cascade appears equally actionable, which adds noise. With it, only depth-0 nodes are surfaced as actionable while deeper nodes are grouped as consequences.

## Snapshot References for Point-in-Time Reconstruction

Each history entry carries a `snapshot_ref`, a pointer to a delta snapshot of graph state at that moment.

Instead of storing the full graph on every event:

- delta snapshots store only modified nodes,
- periodic full checkpoints are added every `N` commits.

This mirrors how Git stores diffs: efficient for high-frequency changes with bounded replay cost. In large, frequently changing production graphs, this keeps storage linear with number of changes rather than graph size.

## Trade-Offs and Practical Retention

Append-only logs grow without bound in long sessions. A practical system should use windowed retention:

- keep the last `N` entries or entries from the last `T` minutes,
- compact older history into summaries (for example, "was stale 12 times between T1 and T2").

For query performance, keep current state (`stale`, `stale_reason`) on the node as a materialized view, and consult history only for auditing/debugging. This preserves `O(1)` read-path performance while retaining temporal fidelity where needed.