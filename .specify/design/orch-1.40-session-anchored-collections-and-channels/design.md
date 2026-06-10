# Design: Session-Anchored Collections + Native Channel Messaging (ORCH-1.40)

> A robust solution to the two ORCH-1.39 deferrals — (a) reliable session-scoped queries
> (`graph_context list`) and (b) a real invoker↔spawned message channel — by attacking their
> shared root cause and building on a powerful primitive that already ships but is unwired.
> Extends ORCH-1.39; composes with KG-2.0 (engine Communication Channels).

## Root cause (shared by both deferrals)

The epistemic-graph Tokio engine is a **pure id-addressed adjacency store with NO secondary,
label, or property indexes** (`epistemic-graph/src/graph.rs`: `node_properties` is an opaque
id→msgpack map; the op set has `HasNode/GetNodeProperties/GetSuccessors/...` but **no
`NodesByProperty`/`MatchNodes`**). Therefore:
- It is **fast + reliable at**: id lookup, and **traversal FROM a known id**
  (`MATCH (s {id:$x})-[:REL]->(t:Label)` → `_exec_rel_match`, routes to fast L1).
- It is **unreliable at**: arbitrary property scans (`MATCH (n:Label {prop:$x})` /
  `WHERE n.prop=$x`) — only servable as a full Python-side scan, and worse, an *unparsed*
  `WHERE` falls through `_legacy_execute` which **returns the ENTIRE graph** when no `id`/`label`
  param is present (`epistemic_graph_backend.py:406-433`) — the "garbage" over-match footgun.

So "find/list by session" must NOT be a property scan. It must be an **id-anchored traversal**.

## The powerful discovery (deferral 2)

The engine ships a native **Communication Channels** subsystem (CONCEPT:KG-2.0,
`epistemic-graph/src/channels.rs`): `create/join/leave/send_message/get_messages/list/close`,
running **inside the shared UDS server** → inherently **cross-process + totally ordered**
(~0.19 ms/op), with P2P channels enforcing exactly 2 members (ideal for invoker↔spawned). The
Python client **and a sync wrapper already expose `.channels`** (`epistemic_graph/client.py`),
but it is **100% unwired in agent-utilities**. The message channel is therefore a *wiring* task,
not an invention.

## Unified design — three pillars

### Pillar 1 — Session-anchored collections (graph-native "index")
Model every session as an id-addressable `Session` node (`id = "session:{sid}"`) with typed edges:
`Session -[:HAS_CONTEXT]-> ContextBlob`, `Session -[:HAS_MESSAGE]-> AgentMessage`,
`Session -[:HAS_RUN]-> RunTrace`. All "list by session" become **id-anchored single-hop
traversals** — the engine's reliable, fast path.
- Write path: on ContextBlob/message/run creation, `MERGE (s:Session {id:$sid})` (supported by
  `_exec_merge_node`) + `engine.add_edge(session_id, child_id, "HAS_*")` (O(1) by id, already
  exists, used by the ORCH-1.39 provenance link).
- Read path: `MATCH (s {id:$sid})-[:HAS_CONTEXT]->(c:ContextBlob) RETURN c ORDER BY c.created_at`
  (`_exec_rel_match`, supports rel-type + label filter + ORDER BY). Replaces the degraded
  `graph_context list` property scan. O(degree), not O(N).
- Backfill: one-time pass building `Session` edges from existing `session_id` props.

### Pillar 2 — Native channel messaging (live, ordered, cross-process)
Wire the engine Channels subsystem for invoker↔spawned dialogue:
- `engine.channels` accessor on `GraphComputeEngine`/`IntelligenceGraphEngine` (reaches
  `self._client.channels`). New `messaging/agent_channel.py` helper.
- Channel key `orch:{session_id}:{run_id}`, `PeerToPeer`, members `invoker:{sid}` + `agent:{run_id}`.
- API: `open_channel(sid, run_id)`, `send(channel_id, sender, payload, durable=False)`,
  `receive(channel_id, since_seq, timeout)` (cursor-based poll loop; needs a `since_seq` cursor —
  small engine add to avoid O(n) history re-reads), `close(channel_id)`.
- Integration: `run_agent` opens the channel + stamps `config["channel_id"]`; new
  `AgentDeps.message_channel_id` field; `graph_orchestrate(execute_agent)` returns `channel_id`;
  a sibling `graph_message` MCP tool (`open/send/receive/close`) mirrors `graph_context`.
- **Bridge**: when the spawned agent needs the *user*, the invoker side forwards channel messages
  to its existing `elicitation_queue`/`ApprovalManager` — clean cross-process → in-process bridge,
  no UI change.

### Pillar 3 — Durable backstop + safety hardening
- **Durable messages**: `send(..., durable=True)` dual-writes each message as a
  `Session -[:HAS_MESSAGE]-> AgentMessage` node (Pillar 1) — so live transport (channels) and
  durable/replayable history (graph, queryable by the anchor traversal) are unified. Channels are
  live/in-RAM (durable imprint only on `close()` today); the graph node is the durable record.
- **Safety hardening (do first, cheap, high-value)**: remove/gate the
  `_legacy_execute` "return the entire graph" branch (`epistemic_graph_backend.py:427-433`) so an
  unparsed `WHERE` can NEVER silently over-match. This is the actual `graph_context list` "garbage"
  bug and a latent correctness hazard for every caller.

### (Deferred sub-option) Engine property index — only if needed
Adding a Rust secondary index + `NodesByProperty` op (Option A) is the only thing that makes
*arbitrary* (non-session) property listing fast at scale. It's a high-risk cross-repo Rust +
protocol + ledger-replay change. The anchor pattern removes the pressure for the session case, so
defer this unless a hot, large-N, non-anchorable property-listing requirement emerges.

## Phased plan
1. **Hardening (smallest, highest safety):** remove the `_legacy_execute` whole-graph fallback;
   add a regression test (unparsed WHERE → empty, never all-nodes). Ship immediately.
2. **Session anchor (Pillar 1):** `Session` node + `HAS_CONTEXT`/`HAS_RUN` edges on the
   ContextBlob/RunTrace write paths; rewrite `graph_context list` as the anchored traversal;
   backfill script. Unit + live round-trip (put N → list-by-session returns exactly N).
3. **Channel wiring (Pillar 2):** `engine.channels` accessor + `agent_channel.py` + `graph_message`
   MCP tool + `run_agent`/`AgentDeps` integration + the `since_seq` cursor engine add. Live two-process
   send/receive test.
4. **Durable + bridge (Pillar 3):** `durable=True` dual-write to `AgentMessage`; elicitation bridge.
5. **(Defer)** engine property index unless a concrete hot path appears.

## Concept & wiring
- **Proposed CONCEPT:ORCH-1.40** — sub-concept of ORCH-1.39; composes with KG-2.0 (channels) and
  ORCH-1.21 (execution bridge). Wire-First: `graph_orchestrate`/`run_agent` → `agent_channel` /
  session-anchor writes → engine channels/edges (≤2 hops).
- Schema additions (`Session`, `AgentMessage`, `HAS_MESSAGE`, `HAS_RUN`) → extend the OWL layer in
  `ontology_orchestration.ttl` (already has `ContextBlob`/`hasContext`).
- Risk: Pillars 1–3 are **additive, Python-side, zero Rust** except the optional `since_seq` cursor
  (small, backward-compatible engine add). The whole-graph-fallback removal is the only
  behavior-change — guard with a regression test + an opt-in escape flag.

## Critical files
- `knowledge_graph/backends/epistemic_graph_backend.py` (`_legacy_execute` hardening; `_exec_rel_match`/`_exec_merge_node`/`_exec_rel_match` are the supported anchor primitives)
- `knowledge_graph/core/engine.py` (`add_edge` provenance pattern; `Session` upsert) + `graph_compute.py` (`.channels` exposure)
- `mcp/kg_server.py` (`graph_context list` → anchored traversal; new `graph_message` tool)
- `orchestration/agent_runner.py` (session-anchor writes; channel open) + `models/agent.py` (`message_channel_id`)
- `messaging/agent_channel.py` (NEW — engine-channels wrapper) ; `knowledge_graph/ontology_orchestration.ttl` (OWL)
- engine (separate repo, only the optional cursor): `epistemic-graph/src/channels.rs`, `epistemic_graph/client.py`
