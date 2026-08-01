# Design Document: Two small `run_graph` lifecycle decisions — direct-End unwrapping and best-effort service warm-up

CONCEPT:AU-ORCH.execution.node-direct-end ·
CONCEPT:AU-ORCH.execution.service-registry-initialization

> Both sites live inside `agent_utilities/orchestration/engine.py`'s
> `run_graph` function, at different points in the same call. They are two
> distinct, independently small decisions grouped here because they share a
> function, not because they're the same choice — each has its own rejected
> alternative below.

## Decision 1 — a graph node may end the run directly; `pydantic_graph.End` is unwrapped at the top level, not re-routed through a completion node

`CONCEPT:AU-ORCH.execution.node-direct-end`

`engine.py:748-753`. `pydantic-graph` lets any node terminate a run by
returning `End[GraphResponse]` rather than transitioning to another node —
the router's "direct-completion shape." When that happens, `graph.run(...)`
itself returns the `End` wrapper object, not the `GraphResponse` inside it.
`run_graph` explicitly checks `isinstance(result, End)` and unwraps
`result = result.data` before doing anything else with the result.

**The rejected alternative is not unwrapping and letting the fallthrough
`str(result)` handling stringify the wrapper as-is.** The comment states the
concrete failure this produces: the reply becomes the literal text
`"End(data=GraphResponse(…))"` — a nonsense answer surfaced to the end user
instead of the actual response payload. The unwrap is a two-line, low-risk
fix, but it's a real decision because it's easy to omit: any new
node-that-ends-directly code path that doesn't route through this exact spot
in `run_graph` reintroduces the same stringified-wrapper bug.

## Decision 2 — `ServiceRegistry` is warmed up early inside `run_graph`, best-effort, even though it's re-initialized later on the real dispatch path

`CONCEPT:AU-ORCH.execution.service-registry-initialization`

`engine.py:629-639`. Immediately after emitting the `graph_start` event,
`run_graph` calls `ServiceRegistry.instance().initialize()` inside a bare
`try/except` that only logs at `debug` on failure — the code comment is
explicit that this is **redundant**: `svc_registry`/`svc_count` are never
referenced again in this function, and the same lazy singleton is
re-initialized properly later, on the actual dispatch path, in
`agent_runner.py`. This early call exists purely as best-effort warm-up: by
the time the graph's specialist nodes actually need the registry, it has
already paid its one-time initialization cost during the earlier, otherwise-
idle `graph_start` window.

**The rejected alternative is skipping this early call and relying solely on
`agent_runner.py`'s lazy initialization.** That would still be correct — the
registry is a lazy singleton, so the first real access initializes it exactly
once regardless. The tradeoff being made here is latency shape, not
correctness: paying the initialization cost during `run_graph`'s startup
window (before the graph has done any real work) instead of on the critical
path the first time a specialist node needs a registered service. Because it
is pure overlap-hiding and explicitly allowed to fail silently, this is a
best-effort optimization, not a correctness-bearing decision — which is why
it's recorded here as a small, real, but low-stakes choice rather than folded
into a bigger concept.

## Risk Assessment

- **Blast Radius**: `agent_utilities/orchestration/engine.py` only, for both
  decisions.
- **Backward Compatible**: Yes for both — Decision 1 fixes a display bug with
  no behavior change to non-`End`-returning nodes; Decision 2 is a pure
  latency optimization with an already-safe fallback.
- **Known weak point**: Decision 2's `except Exception` swallows any
  registry-init failure at `debug` level with no propagation — a
  systematically broken `ServiceRegistry` would be silently invisible here
  and only surface later, on the dispatch path, as a *different* failure with
  no obvious link back to this early warm-up attempt having also failed.
