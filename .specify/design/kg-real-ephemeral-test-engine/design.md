# Design Document: Engine-backed tests run against ONE real, ephemeral engine process per session — never SQLite, never mocks

CONCEPT:AU-KG.memory.provides-real-ephemeral-one

> Realised by `tests/conftest.py:590-691` — the session-scoped autouse
> `_session_engine` fixture plus `tiny_engine` and the per-test `engine_graph`
> tenant fixture. Introduced by commit `f7774cb9`. Cited as the justification
> for hard-failing production code paths in
> `agent_utilities/knowledge_graph/memory/timeseries/__init__.py:11` and
> `engine_backend.py:19`.

## Decision — deploy the actual shipped database once per test session, isolate per test by tenant rather than by process

The session fixture starts one real `epistemic-graph-server` process on an
isolated UDS socket with a temporary persist directory, and tears it down with
a graceful SIGTERM. Individual tests do not get their own engine; they get
their own *tenant* within it (`engine_graph`).

**Two alternatives were rejected, and the first is stated as a directive.** The
fixture docstring records it: *"USER DIRECTIVE: engine-backed tests validate
against the ACTUAL database we ship — NOT SQLite, NOT mocks."*

The argument is about what a passing test entitles you to believe. A mock
asserts that the code calls the interface the way the test author *thought* the
engine behaves; it re-tests the author's model, not the engine. A SQLite
substitute is worse in a specific way — it is real enough to pass, so its
divergences from the shipped engine (transaction semantics, native tsdb and ANN
behaviour, error surfaces) show up only in production. This is the same
commitment that let `CONCEPT:AU-KG.memory.time-series-lives-one` delete
`sqlite_backend.py` outright: a real test engine is the precondition that makes
"no second backend" affordable.

The second rejected alternative is the mechanism this replaced. Commit
`f7774cb9`'s fixture *"fully replac[es] `start_epistemic_graph_server`"* — the
prior lazy/per-test autostart. Per-test engine processes are the obvious way to
get isolation, and they were rejected on cost: standing up and tearing down a
real engine per test makes a suite of any size unusable. Session-scoped deploy
plus per-test tenant isolation gets the same independence between tests at one
process's worth of startup, which is what makes "always use the real engine"
practical rather than aspirational.

Ephemerality is the third element and is what keeps it honest — an isolated
socket and a temp persist dir mean no test can depend on state another test or
a previous run left behind.

## Risk Assessment

- **Blast Radius**: `tests/conftest.py`; the fixture is autouse and session
  scoped, so it affects the whole suite, and it is cited by dozens of test
  modules across the KG suite.
- **Backward Compatible**: N/A — test infrastructure.
- **Known weak point**: isolation is by tenant inside one shared process, so
  it is only as strong as the engine's tenant boundary. Anything genuinely
  process-global — a crash, resource exhaustion, a global config mutation, a
  bug that escapes tenancy — leaks across every test in the session and
  presents as unrelated failures elsewhere. Per-process isolation would not
  have that failure mode; it was traded away for suite runtime.
