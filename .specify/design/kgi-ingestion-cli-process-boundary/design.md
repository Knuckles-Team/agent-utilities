# Design Document: A CLI process-boundary entrypoint lets cross-package tools route content into the KG without importing agent-utilities

CONCEPT:AU-KG.ingest.process-boundary-entrypoint

> `agent_utilities/knowledge_graph/ingestion/__main__.py:1-17`.

## Decision — `python -m agent_utilities.knowledge_graph.ingestion` is a thin subprocess entrypoint over `IngestionEngine`, so other packages reach the KG by shelling out instead of importing agent-utilities as a dependency

`__main__.py:2-8` states the purpose directly: **"A thin process-boundary
entry point over the standardized `IngestionEngine` so cross-package tools
(e.g. the universal-skills `web-crawler` / `skill-graph-builder`) can route
content INTO the KG without importing agent-utilities — they shell out
to `python -m agent_utilities.knowledge_graph.ingestion <path-or-url> [...]`."**

**The rejected alternative is a direct Python import** — `universal-skills`
(or any other cross-package caller) importing `agent_utilities` as a library
dependency to call `IngestionEngine` in-process. It is more efficient (no
subprocess overhead, no CLI-argument marshalling) and it loses because it
would couple every cross-package caller's dependency tree to the whole of
`agent-utilities` just to reach one capability — a heavyweight, version-
coupled dependency for what is conceptually a narrow "send this content to
the KG" operation. The process boundary is the actual product here: it
decouples the caller's runtime/dependency graph from agent-utilities' own,
at the cost of a subprocess invocation per ingest call.

**Content-type handling stays consistent with the in-process path**: type is
auto-detected per source (`ContentType.classify`) unless overridden, and
documents flow through the standardized contract (verbatim `Document` +
`IdeaBlock` chunks + `Concept`), "so what lands in the KG is faithfully
re-materialisable (e.g. distilled back into a skill-graph)"
(`__main__.py:12-15`) — the CLI wrapper doesn't create a second, thinner
ingestion contract; it's the same `IngestionEngine`
(`AU-KG.ingest.ingestion-engine`) reached from outside the process.

## Risk Assessment

- **Blast Radius**: `knowledge_graph/ingestion/__main__.py` only; the
  `IngestionEngine` it wraps is shared with every in-process caller.
- **Backward Compatible**: Yes.
- **Breaking Changes**: None.
- **Known weak point**: a subprocess-per-call boundary means cross-package
  callers pay Python interpreter startup + import cost on every invocation;
  there is no long-running-process/batch-mode variant of this entrypoint
  documented here for a caller that needs to ingest many items in sequence.
