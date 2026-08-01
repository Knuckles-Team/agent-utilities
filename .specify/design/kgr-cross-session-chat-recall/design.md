# Design Document: Chat-history search is exposed as a retrieval-namespace facade, not reimplemented under it

CONCEPT:AU-KG.retrieval.cross-session-chat-recall

> `agent_utilities/knowledge_graph/retrieval/chat_search.py` (facade),
> `agent_utilities/core/chat_persistence.py` (the underlying implementation,
> re-exported).

## Decision — `knowledge_graph.retrieval.chat_search` is a dedicated entry point that re-exports `core.chat_persistence`'s search, rather than a second implementation

`chat_search.py:4-26` states the shape directly: this module "provides a
dedicated entry point for cross-session chat search functionality," which
"re-exports the keyword-based search implementation from
`agent_utilities.core.chat_persistence` so that downstream consumers
(including the `overview.md` conceptual registry) can import directly from
the `knowledge_graph` namespace." The underlying implementation queries
stored `Thread`/`Message` nodes via the KG Cypher backend, groups results by
session, and computes keyword-hit-density relevance scores — logic that lives
in `core.chat_persistence` and is not duplicated here.

**The rejected alternative**: reimplementing chat-history search inside the
`knowledge_graph.retrieval` package directly, so that the retrieval domain's
conceptual registry has an entry that matches where callers expect to find a
retrieval capability. That would create two implementations of the same
Cypher-backed search that could drift from each other the moment either was
changed without the other following — the facade pattern here means the
retrieval-namespace surface and the `core.chat_persistence` implementation
are, by construction, the exact same code path, not two that happen to agree
today.

## Risk Assessment

- **Blast Radius**: `chat_search.py`, `core/chat_persistence.py`
  (`search_chat_history`, `ChatRecallResult`).
- **Backward Compatible**: Yes — a pure re-export layer; no behavior change
  versus calling `core.chat_persistence` directly.
- **Known weak point**: this is a thin naming/namespace decision, not a
  retrieval-algorithm one — the actual relevance-scoring logic (keyword-hit
  density) lives entirely in `core.chat_persistence` and is out of scope for
  this document; a reader looking here for the ranking algorithm itself needs
  to follow the re-export.
