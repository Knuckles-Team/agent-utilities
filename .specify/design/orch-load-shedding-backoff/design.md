# Design Document: A per-model-endpoint circuit breaker sits outside the adaptive concurrency controller, because shrinking a gate is too slow to stop an endpoint that is already shedding load

CONCEPT:AU-ORCH.routing.load-shedding-backoff

> Realised by `agent_utilities/core/model_circuit_breaker.py:1-100`
> (module docstring, `CircuitState`, `ModelCircuitBreaker`), with the backoff
> call sites at `agent_utilities/core/model_concurrency.py:311` and `:378`.
> Introduced by commit `a6c13c66` ("perf(llm): cap aggregate concurrency at the
> model SERVER's capacity, never the local host").

## Decision — add a fast three-state breaker as a *second*, independent control alongside the existing adaptive concurrency controller

The platform already had a concurrency controller that adapts its target width
from observed latency and error rate. That controller was not removed and is
not being replaced. The decision is that it is the wrong instrument for one
specific failure mode, and that a second instrument is needed next to it.

The module docstring states the reason: *"shrinking the gate width is a slow,
statistical signal — it does not stop the bleeding the instant the endpoint
starts shedding load."* An adaptive controller works by accumulating evidence:
it needs several samples before it moves, and each of those samples is another
request sent at an endpoint that is already failing. When the endpoint is a
shared GPU server, those extra in-flight requests are exactly what turns a
degraded endpoint into a dead host.

`ModelCircuitBreaker` is therefore per-model-endpoint and fast: CLOSED →
OPEN on failure, an exponential-backoff cooldown, then HALF_OPEN admitting a
*single* probe before it will close again. The probe is deliberately one
request, not a fraction of traffic — the point of the state is to learn whether
the endpoint is back without re-applying the load that broke it.

**The rejected alternative is "the adaptive controller alone", and it was
rejected by an incident, not by argument.** The introducing commit records the
root cause from the GB10 kernel logs: *"concurrent embeds + enrichment +
orchestration to the same GB10 vLLM exhausted its 121 GB unified memory ->
NVRM Out of memory -> host death."* Three independent workloads each stayed
within their own locally-reasonable concurrency limits and collectively killed
the machine. The commit title states the corrected principle directly: cap at
the model *server's* capacity, never the local host's — the local view is
structurally incapable of seeing aggregate load on a shared endpoint, so no
amount of tuning the local adaptive gate would have prevented this.

## Risk Assessment

- **Blast Radius**: `agent_utilities/core/model_circuit_breaker.py`,
  `agent_utilities/core/model_concurrency.py`. Every model call that routes
  through the concurrency layer is affected.
- **Backward Compatible**: Yes in the healthy case — a CLOSED breaker adds a
  state check and nothing else. Not compatible in the failing case, by design:
  calls that previously queued against a dying endpoint now fail fast.
- **Known weak point**: the breaker is keyed per model *endpoint*, and its
  state is process-local. Several processes sharing one physical GPU server
  each maintain their own breaker, so the aggregate-load problem the incident
  exposed is mitigated per-process but not solved globally — nothing
  coordinates the breakers, and N processes can each send their single
  HALF_OPEN probe simultaneously.
