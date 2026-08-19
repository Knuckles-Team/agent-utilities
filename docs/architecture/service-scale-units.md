# Service-surface scale units

`CONCEPT:AU-OS.scaling.service-scale-units`

This document defines the common scale contract for every service surface that
can be replicated or drained.  It is a decision boundary, not a controller:
Kubernetes, Docker, a process supervisor, or a native engine actuator may apply
an accepted decision, but none of them invents a surface-specific scaling loop.

## Why one contract

Ingestion, dispatch, gateways, query readers, and connector/media/RLM pools
have different signals and hard dependencies, but they share the same safety
problem. A replica count is safe only when the demand signal, allocatable
capacity, continuity state, and external budgets are fresh and attributable.
`agent_utilities.orchestration.service_scale_units` makes those facts typed and
uses one deterministic `evaluate_scale` function for all surfaces.

The evaluator is fail-closed. Missing or stale evidence yields `blocked`; it
never treats an absent measurement as zero capacity, an unlimited provider, or
an idle continuity state.

## Surface profiles

| Surface | Target signals | Hard capacity / authority | Additional gate |
| --- | --- | --- | --- |
| `ingest_worker` | queue depth, consumer lag, backlog age, p95 latency | CPU, memory, partition slots, lease slots, engine write bytes/s, fsync IOPS | partition + fenced lease |
| `dispatch_worker` | queue depth, consumer lag, active sessions, p95 latency | CPU, memory, partition slots, lease slots | partition + fenced lease |
| `graphos_gateway` | request rate, in-flight, p95 latency, error rate | CPU, memory, session slots, read-admission slots, engine bytes, fsync IOPS | session continuity; engine authority |
| `mcp_gateway` | request rate, in-flight, p95 latency, sessions, continuity load, provider in-flight, errors | CPU, memory, session slots, continuity slots | session continuity + provider-global quota |
| `api_gateway` | request rate, in-flight, p95 latency, error rate | CPU, memory, session slots | stateless or externalized continuity |
| `query_reader` | request rate, in-flight, p95 latency, error rate | CPU, memory, read-admission slots, engine bytes, fsync IOPS | saturated engine blocks reader growth |
| `connector_pool` | queue depth, lag, request rate, provider in-flight, p95 latency, errors | CPU, memory, upstream rate | provider-global quota |
| `media_pool` | queue depth, request rate, GPU utilization, p95 latency | CPU, memory, GPU count, GPU memory | truthful GPU capacity |
| `rlm_pool` | queue depth, request rate, GPU utilization, active sessions | CPU, memory, GPU count, GPU memory | truthful GPU capacity |

Profiles constrain the vocabulary. A contract may add a bounded local signal or
capacity axis only when the profile already permits it; it cannot silently use
a queue signal for an API or omit an engine/read-admission dependency.

## Contract shape

`ScaleUnitContract` is frozen and versioned. It contains:

- a stable unit reference and immutable content identity;
- `ScalePolicy` bounds: minimum/maximum replicas, target signal and target,
  up/down steps, independent cooldowns, drain bound, and explicit scale-to/from
  zero switches;
- one per-replica `CapacityDemand` for each declared hard axis;
- `QuotaBinding` records, including provider-global budgets. Provider quotas are
  shared ceilings, never multiplied into a larger budget by adding MCP or
  connector replicas;
- `PartitionLeaseContract` for partition assignment, TTL, and fencing;
- a `ContinuityContract` describing stateless, session-affine, or externalized
  state and the drain behavior for active sessions/streams;
- an explicit engine-authority axis set for gateway and query surfaces; and
- a required `LoadSafetyObservation` unless the caller explicitly opts out of
  the safety gate in a separately governed contract.

Observations carry bounded source references, a digest, an observation window,
and no credentials or raw payloads. Capacity uses the invariant
`used <= reserved <= allocatable`; quotas use `used <= limit`. A stale or
scope-mismatched observation is not usable evidence.

## Decision flow

```text
typed contract + fresh observations
             |
             v
  validate signal / capacity / quota / continuity / safety
             |
     any missing, stale, unsafe, or mismatched fact?
          /                         \
        yes                          no
        |                             |
     BLOCKED             target = ceil(signal / target), clamped to bounds
                                      |
                    step cap + cooldown + continuity drain checks
                                      |
                      capacity/quota/engine authority projection
                                      |
                               SCALE_UP / SCALE_DOWN / HOLD
```

The result is a signed-by-content `ScaleDecision` with bounded evidence and
reasons. The module does not actuate and does not write high-rate telemetry to
the graph.

### Shared quota rule

For a provider-global quota, the evaluator checks the single observed provider
limit against current usage plus the declared demand of the proposed additional
replicas. The limit is never multiplied by replica count. A missing, stale,
scope-mismatched, exhausted, or explicitly multiplicative provider binding
blocks growth. This is why adding MCP replicas cannot evade an upstream model
or API-provider quota.

### Engine authority rule

`graphos_gateway` and `query_reader` declare the engine axes that they depend on.
If read-admission slots, engine bytes, fsync capacity, or ingest write capacity
is exhausted, the evaluator returns `engine_authority_saturated` rather than
claiming that more front-end/query replicas solve a saturated single engine.
The engine or shard planner must change the authority capacity; this contract
only prevents an unsafe false positive.

### Bounds, floors, and drains

Target tracking computes a clamped desired count and then applies the declared
step cap. A service returns to its minimum under sustained low demand. A
minimum of zero is not enough to permit zero replicas: `allow_scale_to_zero`
must also be true. Likewise, a zero-running unit starts only when
`allow_scale_from_zero` is true. Session-affine and externalized units require
fresh continuity readiness; active sessions/streams block a scale-down when the
contract says they must drain. A drain longer than the policy or continuity
bound produces `blocked` with `drain_required`.

Overload-shedding and noisy-neighbor observations are first-class safety
evidence. They block both growth and shrinkage while the controller cannot
prove that the remaining fleet is safe. Operators can record the bounded
incident reference, but raw traces, tokens, provider credentials, and host
secrets never belong in a scale decision.

## Ownership and integration

The contract is the shared AU authority. A controller reads observations in
batch, calls `evaluate_scale` once per unit, persists the decision through its
own durable/replay-safe path, and then delegates to a typed actuator. It must
not reimplement target tracking for individual surfaces. The same contract can
therefore be used with Kubernetes HPA/KEDA adapters, a Docker/systemd
supervisor, or a native epistemic-graph engine actuator without changing the
scaling safety semantics.
