# Design Document: One declarative retry/backoff/fallback primitive replaces N hand-rolled retry loops, preserving each site's historical semantics exactly

CONCEPT:AU-ORCH.execution.retry-predicate-raised-treating

> `agent_utilities/orchestration/resilience.py` (`ResiliencePolicy`, primary).
> Adopted at: `agent_utilities/core/http_client.py`,
> `agent_utilities/graph/executor.py`, `agent_utilities/graph/parallel_engine.py`,
> `agent_utilities/knowledge_graph/backends/postgresql_backend.py`,
> `agent_utilities/knowledge_graph/backends/contrib/ladybug_backend.py`,
> `agent_utilities/knowledge_graph/memory/learning_engine.py`,
> `agent_utilities/patterns/prompt_chain.py`.

## The real decision

`resilience.py`'s own docstring names the gap it closes and against what
reference: *"Closes the Reliability & Failure-Management gap (L7) versus the
agentic reference architecture"* (`resilience.py:5`). The platform already had
a circuit breaker and per-server breaker on the specialist-execution path, and
engine-native WorkItem leases/checkpoints — *"but there was no **declarative**
policy describing how an individual unit of work should retry, back off, fall
back, or time out"* (`resilience.py:11-13`). `ResiliencePolicy` is that
missing primitive: one composable object describing retry count, backoff
shape, fallback chain, and timeout for a single callable, with
`compute_backoff` providing deterministic exponential backoff with optional
jitter.

Once the primitive existed, call sites across the codebase migrated their
**pre-existing, bespoke retry logic** onto it — and every migration
deliberately preserved that site's own historical numbers rather than
normalizing them to one shared default:

| Site | Preserved historical behaviour |
|---|---|
| `graph/parallel_engine.py:720-733` | "SWARM-5 backoff": 0.5s, 1s, 2s, ... capped at 8s, no jitter |
| `knowledge_graph/backends/postgresql_backend.py:986-996` | Historical lock backoff `(2**n)*0.1s`, no jitter; healed-schema retries carry a `0.0s` delay hint so they stay immediate |
| `patterns/prompt_chain.py:304-313` | Retry ANY `Exception` up to `max_retries` extra attempts with **no delay** between attempts; exhaustion leaves output `""` rather than raising |
| `graph/executor.py:71-84` (`_specialist_resilience_policy`) | Retries only transient model/tool errors (`TimeoutError`/`ConnectionError` — never `ValueError`/permission errors), short exponential backoff with jitter, `node_timeout` as the per-attempt timeout |

`http_client.py:14-19` documents the intended shape of adoption for new
callers going forward: pass a `ResiliencePolicy` and *"transport-level
failures are retried under it, instead of each call site hand-rolling a retry
loop"* — `http_retry_policy()` builds a policy matching httpx's transport
errors specifically.

## The rejected alternative — the concept's name is the decision

The concept id is drawn directly from the specific code comment that names
the design choice: **when a caller-supplied `retry_on` predicate itself
raises, the failure is treated as non-retryable, not as a crash and not as
retryable.**

```
resilience.py:167-173
    if callable(retry_on) and not isinstance(retry_on, tuple):
        try:
            return bool(retry_on(exc))
        except Exception:  # noqa: BLE001 - a broken predicate must not crash
            logger.warning(
                "[...] retry_on predicate raised; treating as non-retryable"
            )
            return False
```

Two alternatives are rejected here, both worse: letting the predicate's own
exception propagate up through the resilience layer (turning a
misconfigured/buggy `retry_on` into a crash of the very mechanism meant to
make the call *more* robust), or treating a broken predicate as "retryable by
default" (which could retry-loop indefinitely against a caller error that
will never resolve, masking the real bug as transient flakiness). The chosen
fail-closed default — log and treat as non-retryable — surfaces the broken
predicate once via a warning and lets the call fail normally, rather than
either crashing the framework or hiding a bug behind infinite retries.

The broader "hand-roll per call site" alternative is rejected too, but not by
elimination — by **preservation-through-migration**: each of the four
call-site rows above is that site's *own* bespoke retry logic from before
this decision, re-expressed as `ResiliencePolicy` parameters rather than
replaced with one shared default. The decision is "one engine, N declared
policies," not "one engine, one policy" — consolidating the *mechanism*
without erasing each site's judgment about what should retry and how long to
wait.

## Risk Assessment

- **Blast Radius**: `agent_utilities/orchestration/resilience.py`,
  `agent_utilities/core/http_client.py`, `agent_utilities/graph/executor.py`,
  `agent_utilities/graph/parallel_engine.py`,
  `agent_utilities/knowledge_graph/backends/postgresql_backend.py`,
  `agent_utilities/knowledge_graph/backends/contrib/ladybug_backend.py`,
  `agent_utilities/knowledge_graph/memory/learning_engine.py`,
  `agent_utilities/patterns/prompt_chain.py`.
- **Backward Compatible**: Yes — each migration preserved the call site's
  historical retry/backoff numbers exactly, by design.
- **Known weak point**: preserving four different historical backoff shapes
  (jittered exponential, unjittered exponential, no-delay-retry-any-exception,
  immediate-retry-on-heal) inside one shared primitive means the primitive's
  own defaults (`DEFAULT_POLICY`: 3 attempts, 0.5s base, 2.0x factor, 10s cap,
  jitter on) are not actually representative of most real call sites — a
  future caller reading `DEFAULT_POLICY` for guidance would not learn any of
  the four historical behaviours above.
