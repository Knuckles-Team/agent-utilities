# Design Document: Media a user sends is stored durably as a content-addressed KG blob + node, not discarded after use

CONCEPT:AU-KG.ingest.list-durable-media

> `agent_utilities/knowledge_graph/memory/media_store.py:1-30,287-300,550-565`
> (primary — `MediaStore`), `agent_utilities/messaging/router.py:768-780,826-834,890-895,934-940`
> (the messaging-layer caller and background-persist wiring).

## Decision — persist media bytes in the engine's content-addressed BLOB store and record a KG node referencing it, replacing discard-after-use, with content identity kept separate from occurrence provenance

`media_store.py:1-9` states the prior state and the fix directly: **"Media
(an image a user sent, a voice note, a chart) used to be ephemeral and
absent from the KG: the messaging layer transcribed audio then
`os.unlink`ed it, wrapped images as inline `BinaryContent` then discarded
them, and persisted only TEXT. This module makes media durable and
first-class"** by storing raw bytes in the engine's BLOB store and recording
a KG node that references the blob — "so 'show me the chart they sent
yesterday' becomes a real query."

**The rejected alternative is the prior behavior, named explicitly**:
transcribe/render media for the immediate reply, then delete or discard the
bytes — a design that treats media as ephemeral input to a single turn
rather than as first-class, later-queryable KG content. It is simpler (no
blob storage, no identity/ACL/retention model to maintain) and it loses
because it makes the KG permanently blind to any media the user has sent —
no later query, audit, or agent reasoning can ever reference it again.

**A second, corrective decision is bundled in the same module — content
identity vs. occurrence provenance must NOT collapse onto one node**
(`media_store.py:11-19`, `AU-KG.identity.asset-occurrence`). Earlier versions
derived BOTH the blob id AND the asset node id from the content digest alone.
That is named as a real bug, not a hypothetical: **"it means the SAME bytes
seen in a second message, tenant, or legal context silently collapsed onto
ONE node, overwriting whatever source/tenant/ACL/retention/legal-hold the
first occurrence had recorded — a real provenance loss, not a cache hit."**
The fix separates *what the bytes are* (content identity, the blob digest)
from *how, when, and under which authority they occurred* (an
`:AssetOccurrence` node per occurrence) — the same bytes can now be
referenced by multiple occurrences, each with its own provenance, without
overwriting each other.

**Operational integration decision**: media persistence rides the messaging
background path, off the reply path — `router.py:768-780` persists media
AFTER the reply-relevant work (KG-ingest the message, generate the reply) so
"a mid-batch `store_media` failure here silently skips remaining attachments
in this message, documented as acceptable since the reply already went out."
Every `MediaStore` method is explicitly best-effort-safe for this reason
(`media_store.py:294-297`): a failure logs and returns `None`/`False` rather
than raising, so a media-persistence failure can never block or delay the
user getting an answer — the same "reply is never blocked by a side effect"
discipline the messaging-router doc (`eco-messaging-universal-graph-agent`)
documents for reactions.

## Risk Assessment

- **Blast Radius**: `knowledge_graph/memory/media_store.py`,
  `messaging/router.py` (`_persist_media`, `_resolve_media_store`).
- **Backward Compatible**: Yes — a live engine with a `graph_compute`/`graph`
  client is required; `_resolve_media_store` returns `None` (caller no-ops)
  otherwise, so environments without a bound engine degrade to the old
  discard behavior rather than erroring.
- **Breaking Changes**: None currently shipped — this replaces silent
  discard with durable storage, which is additive.
- **Known weak point**: best-effort persistence means a partial-batch
  failure silently drops remaining attachments in that message with only a
  debug-level log — there is no user-visible or alertable signal that a
  specific piece of media failed to persist durably.
