# Design Document: Bound conversation-history search to the caller's own delegation subtree, not a binary all-vs-conversation toggle

CONCEPT:AU-KG.retrieval.scoped-conversation-search

> `agent_utilities/capabilities/eg_history_source.py:115-254`
> (`_scoped_closure`, `ScopedEgHistorySource.list_runs`,
> `ScopedEgHistorySource.get_run`) and `:484-506`
> (`build_conversation_search_capability`). Renamed from
> `AU-KG.history.scoped-conversation-search` (D-CC-9,
> `agent_utilities/governance/concept_lineage.yaml:169` — `history` was never a
> registered `KG`-pillar domain; `retrieval` is, and already lists
> `search`/`recall` as signals).

## Decision — a caller's own position in the `parent_run_id` delegation tree is the search boundary, not a global on/off switch

Upstream's native `pydantic_ai_harness.conversation_search` module exposes exactly
one scoping knob: `ConversationSearchToolset`'s `scope: 'all' | 'conversation'`
(cited in the module docstring, `eg_history_source.py:15-18`). `scope='all'`
reaches every run the store holds; `scope='conversation'` reaches only the
caller's own turn. Neither setting expresses the shape this system actually
needs: an agent that has delegated to sub-agents should be able to search
across everything it or its descendants did, but never a sibling subtree or an
unrelated caller's runs — a binary toggle cannot express "everything AT OR
BELOW here."

`ScopedEgHistorySource` closes that gap by binding search to one `GraphSession`
(tenant + scope authority) plus one `root_run_id`, then defining "every run
this source enumerates" as the closure of `root_run_id` over `parent_run_id`
edges already stamped on every `RunTrace` node by the canonical trace ontology
(`observability.trace_ontology`, cited at `eg_history_source.py:23`).
`_scoped_closure` (`:115-155`) computes that closure with an in-process BFS
over `(run_id, parent_run_id)` pairs read from the graph, and does it fresh on
every call — `list_runs` (`:207`) and `get_run` (`:251`) both re-derive the
closure and check membership BEFORE reading anything, rather than trusting a
closure computed once at construction time. The docstring at `:39-46` states
why: `ScopedEgHistorySource.__init__` calls `session.require_scope("kg:read")`
and the closure read fails CLOSED (scopes to nothing) if the query fails for
any reason — it never falls back to an unscoped enumeration. `get_run` given an
out-of-scope or unknown `run_id` returns `[]` rather than raising or reaching
outside the closure (`:251-254`).

`build_conversation_search_capability` (`:484-506`) then composes upstream's
own `scope` parameter ON TOP of this, not in place of it: the helper still
takes upstream's `scope='all'|'conversation'` and narrows further within the
already-narrower, rank-scoped corpus `ScopedEgHistorySource` provides — so
upstream's own enforcement is preserved unmodified, and this system's
delegation-tree boundary is the outer bound upstream can never see past.

**The rejected alternative is upstream's own binary toggle, used as-is.**
Wiring `ConversationSearchToolset` directly at `scope='all'` would let any
agent search every run in the store regardless of who delegated it — a
cross-tenant/cross-caller information leak the docstring calls "the
access-control gap upstream's own docs name" (`:14-18`). Wiring it at
`scope='conversation'` instead would UNDER-scope: an orchestrator that fans
out to sub-agents could not search its own delegates' history at all, breaking
the exact multi-agent introspection this capability exists to serve. Neither
setting of the one knob upstream provides is correct; the fix is a second,
hierarchical dimension (the delegation-tree closure) that composes with,
rather than replaces, upstream's own scope parameter.

## Risk Assessment

- **Blast Radius**: `agent_utilities/capabilities/eg_history_source.py` only —
  `ConversationSearchToolset`/`HistorySource` upstream are consumed through
  their public protocol, not modified.
- **Backward Compatible**: Yes — `build_conversation_search_capability` is an
  additive helper; nothing upstream changes shape.
- **Known weak point**: the closure is recomputed by an in-process BFS on every
  `list_runs`/`get_run` call rather than a native recursive graph query
  (`:51`, noted directly in the module docstring as a deliberate simplification
  pending a native traversal primitive) — correct but not the cheapest shape
  once a delegation tree gets deep.
