# Design Document: Time-series memory has exactly ONE backend — the engine's native tsdb. The SQLite fallback was deleted, not deprecated

CONCEPT:AU-KG.memory.time-series-lives-one

> Realised by `agent_utilities/knowledge_graph/memory/timeseries/__init__.py:22-31`
> (`get_timeseries_backend` — unconditional engine routing plus a hard error)
> and `agent_utilities/knowledge_graph/memory/timeseries/engine_backend.py:1`,
> `:95`, `:126` (`EngineTimeSeriesBackend`). Introduced by commit `5f9cf385`
> ("refactor(kg): time-series memory is engine-only — drop SQLite fallback +
> delete sqlite_backend.py").

## Decision — an unreachable engine is a hard error, never a silent degrade to a second backend

`get_timeseries_backend` routes to the epistemic-graph engine's native tsdb
(`client.timeseries.*`) unconditionally. The `backend_type` parameter is still
accepted for call-compatibility but is **not honoured**. When the engine is
genuinely unreachable the function raises a clear error rather than returning
something that works.

**The rejected alternative existed, shipped, and was deleted — not left behind
a flag.** `sqlite_backend.py` was a working local SQLite implementation of the
same interface. Commit `5f9cf385` removes the file outright, and records the
instruction behind it: *"USER DIRECTIVE: no SQLite even to pass a test."*

The reasoning is about what a fallback *means* for a storage backend. A
fallback that silently accepts writes when the real store is unreachable does
not preserve availability — it forks the data. Some time-series points land in
the engine and some land in a local SQLite file, and nothing reconciles them,
so a query afterwards returns a partial history that looks complete. That
failure is undetectable at the point it matters and unrecoverable afterwards.
Raising instead converts an invisible data-integrity problem into a visible
operational one.

The "even to pass a test" clause is the part that makes this a real commitment
rather than a preference. A fallback retained *only* for tests is the usual
compromise, and it is how a second backend survives: tests exercise the SQLite
path, so the SQLite path keeps working, so it stays plausible to use it in
production under pressure. Deleting it means the test suite must stand up a
real engine — which is what
`CONCEPT:AU-KG.memory.provides-real-ephemeral-one` provides — and the codebase
is then structurally incapable of degrading to a second store.

## Risk Assessment

- **Blast Radius**: everything under
  `agent_utilities/knowledge_graph/memory/timeseries/`, and every caller that
  passed `backend_type`.
- **Backward Compatible**: No. A deployment without a reachable engine, which
  previously worked against SQLite, now fails loudly. That is the intent.
- **Known weak point**: `backend_type` is still accepted and silently ignored.
  A caller passing `backend_type="sqlite"` gets the engine without any
  indication its request was disregarded — the compatibility shim quietly lies
  rather than rejecting the argument, which is a small instance of exactly the
  silent-behaviour-mismatch class this decision otherwise eliminates.
