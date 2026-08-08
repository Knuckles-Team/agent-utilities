# Design Document: OWL-reasoned inferences write back to the durable backend synchronously, not via a silently-broken async queue

CONCEPT:AU-KG.ontology.owl-durable-backfeed

> `agent_utilities/knowledge_graph/core/owl_bridge.py`
> (`_sync_inferred_to_backend`, `_lpg_to_rdf` fast path).

## Decision — persist inferred edges synchronously and idempotently; prefer the engine's native RDF projection over manual triple construction

`_sync_inferred_to_backend` (`owl_bridge.py:1144-1189`) states the bug and the
fix in its own docstring: "Inferred relationships must survive on the durable
tier (not just the in-memory reasoning graph) so other engines and restarts
see them. **The previous implementation queued mutations onto an asyncio queue
+ background task, which silently no-op'd whenever no event loop was
running** — and the daemon tick and pipeline phase both call this
**synchronously**, so inferred triples never reached the backend." The fix
writes them "now via the active engine's `link_nodes` (MERGE-based; carries
edge properties + provenance), with a direct-backend MERGE fallback."

**The rejected alternative — already shipped and already broken — was the
async-queue design.** It looked correct (queue mutations, drain them on a
background task) but silently dropped every inference whenever the calling
context had no running event loop, which was true for both actual call sites
(daemon tick, pipeline phase). The fix is not "add more async plumbing"; it is
the opposite: make the write synchronous and idempotent (`MERGE`, tagged
`inferred: True` / `inferred_from: "owl_reasoner"` for provenance) so a bug
class that depends on "is there an event loop running right now" cannot recur.

A second, related decision lives in the same file's `_lpg_to_rdf` fast path
(`owl_bridge.py:1286-1296`): when the live engine can serve its own canonical
RDF projection (`self.graph.get_rdf()`, an N-Triples round-trip), that is
preferred over manually promoting every node/edge to a triple in Python.
**The rejected alternative is always doing the manual per-node/per-edge
promotion** — the fallback path this file still carries for engines that
cannot serve `get_rdf()`. The manual path is real and needed (offline/
degraded), but produces a document assembled triple-by-triple in Python,
whereas the native path "preserves datatype and language tags exactly; no
property-graph triple-list coercion is involved" — so the native path is tried
first and the manual path is strictly the degrade case, not the primary one.

## Risk Assessment

- **Blast Radius**: `knowledge_graph/core/owl_bridge.py`
  (`_sync_inferred_to_backend`, `_lpg_to_rdf`), every OWL-reasoning caller that
  expects inferred edges to persist across a restart.
- **Backward Compatible**: Yes — same public entrypoints, corrected write
  path.
- **Known weak point**: the fix closes the specific "no event loop" failure
  mode but the MERGE fallback path (`self.backend.execute(...)` with an
  f-string-interpolated relationship type) depends on `_safe_rel_type`
  actually being applied everywhere a predicate reaches this code — an
  unsanitized predicate reaching the fallback branch would be a Cypher
  injection surface if that guard were ever bypassed.
