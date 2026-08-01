# Design Document: Post-conversation KG enrichment — chat becomes durable graph knowledge

CONCEPT:AU-ECO.messaging.post-conversation-enrichment ·
CONCEPT:AU-ECO.messaging.surfaced

> `agent_utilities/messaging/enrichment.py`

## Decision — mine every chat turn for `Concept` nodes AFTER the reply, off the reply path

After a conversation turn, `enrichment.py` mines the chat text for `Concept`
nodes and links them into the shared KG (`MENTIONS`), reusing the SAME
extractor (`extract_text_concepts` + the lite LLM) the IDE-conversation
ingestion path uses (`enrichment.py:1-11`) — rather than a bespoke
chat-specific extractor. This is what turns chat history into durable,
queryable graph knowledge that interweaves with code/docs/research, instead
of a dead transcript nobody queries again.

**The rejected alternative** is not enriching chat at all (treat it as
ephemeral, unlike code/doc ingestion) — the module exists specifically to
reject the asymmetry of "the agent gets smarter from the codebase but not
from talking to the user." It runs in the background (off the reply path)
and is best-effort (`try/except`, logged at debug on failure) so a slow or
failing enrichment never delays the answer the user is waiting for.
Disable with `MESSAGING_ENRICH=0`.

### Pointer — `CONCEPT:AU-ECO.messaging.surfaced`

`enrichment.py:118`, `_surface_intents`. A second, twin behavior of the same
post-conversation pass: beyond linking mentioned concepts, the turn is also
scanned for a concrete GOAL or a feature/spec worth a spec-driven-development
plan, surfaced as `Goal`/`Spec` nodes (`status="surfaced"`,
`origin="chat"`) linked from the chat turn via `HAS_GOAL`/`PROPOSES_SPEC` —
so the loop engine / SDD tooling can pick it up later. It does **not**
auto-execute anything; "surfaced" is deliberately a status short of
"actioned." Idempotent via a stable hashed node id
(`sha256(source_id + desc)`), so re-running enrichment on the same turn
never double-creates the same goal. Opt-out via `MESSAGING_GOALS=0`.

The domain-triage tool's automatic id-shape heuristic flagged this concept
as a "bare generic noun" retire candidate; reading `_surface_intents`
directly shows a real, deliberate design choice (the "surfaced-not-executed"
status), so this document overrides that suggestion rather than retiring the
marker.

## Risk Assessment

- **Blast Radius**: `agent_utilities/messaging/enrichment.py`.
- **Backward Compatible**: Yes.
- **Known weak point**: both mining passes call an LLM per turn
  (`llm_fn(prompt)`); on a high-volume channel this is a recurring inference
  cost with no batching, mitigated only by being off the reply path and
  best-effort.
