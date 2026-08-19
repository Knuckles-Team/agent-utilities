# Durable ScaleIntent actuation

**Concept:** `AU-OS.scaling.reactive-replica-autoscaling`

This document defines the AU-side seam between a durable `ScaleIntentRecord`
and a deployment-specific actuator.  It is deliberately independent of
Kubernetes, Swarm, Docker, and any other runtime.  A runtime adapter implements
the typed `ScaleActuator` protocol; the adapter must make the supplied
`execution_key` idempotent.

## Identity and authority

An intent is immutable and its identity digest covers the complete request:

- intent id and revision, expected scale-unit revision, and execution key;
- desired/minimum/maximum replicas and controller mode;
- declared replica-writer id;
- cluster, runtime, workload, target UID, and target resource version;
- controller fence and expiry, retry policy, reason, schema version, and
  creation time.

The ledger must reject both of these cases rather than silently converging:

1. an existing intent revision replayed with a different identity; and
2. an existing execution key replayed with a different identity.

The target binding includes both a stable UID and the observed resource version.
Names, namespaces, or a workload label alone are not an actuation authority.
The controller must also equal the declared `replica_writer_id`.  A lease carries
the intent fence, an attempt number, an instance id, and an expiry.  Results from
an old lease are fenced after a replacement lease is acquired.

## Durable ordering

Every reconciliation follows this ordering, with each transition durable before
the next side effect:

```text
persist intent identity
        ↓
acquire controller lease / fence
        ↓
persist started(attempt, lease)
        ↓
typed actuator.apply(execution_key, target, lease)
        ↓
persist typed result
        ↓
persist exactly one convergence/failure observation
        ↓
return verified
```

The actuator call is the only external side effect.  A crash before the result
is persisted retries the same execution key; a crash after the result is
persisted repairs the missing observation without invoking the actuator again.
A retryable failure may acquire a newer lease and advance the bounded attempt,
but it never mints a new execution key.  A non-retryable failure or exhausted
retry budget is terminal and is observed without another call.
If the budget is exhausted before any typed result becomes durable, the
reconciler fails closed rather than guessing whether an external side effect
occurred; an adapter may then recover the result by its execution key.

The in-memory `MemoryScaleIntentLedger` is a deterministic unit fixture only.
The production graph/database implementation must provide the same atomic
insert-or-replay, lease/fence, result, and observation semantics.  It must not
interpret a process-local cache as durability.

## States and safety rules

`persisted → started → succeeded|failed → verified` is the live path; a
retryable `failed` result loops through a newer fenced `started` attempt while
the execution key remains unchanged.
`persisted → simulated → verified` is the dry-run path.  `simulated` never
claims `scaled`, and a live execution key cannot later be converted into a
simulation.  Callers mint a separate intent/execution key for a separate dry
run.

Policy denial returns `denied` and never invokes an actuator.  Break-glass is a
separate, explicitly authorized input scoped to the exact target and expiry.
Its authorization audit is durably recorded before the bypassed actuator call;
it is not inferred from a policy flag or an operator name.  Break-glass does not
bypass identity, target-version, fence, retry, or observation checks.
The reconciler defaults to a deny policy when no deployment policy is injected;
the allow policy exported by the module is a focused fixture, not a production
authorization decision.

The reconciler returns `verified` only after a durable observation exists.  An
observation is bound to the execution key, intent revision, target, typed
result, and deterministic outcome digest.  Replaying the same observation is
idempotent; attempting to change its outcome is a conflict.

## Adapter contract

The AU module exposes only typed models and protocols:

- `ScaleIntentLedger` — durable intent, lease, result, observation, and
  break-glass operations;
- `ScaleActuator` — `apply(ScaleActuationRequest) -> ScaleActuationResult`;
- `ScalePolicy` — the policy-gated decision before actuation.

Adapters may implement native or delegated control, but must not accept
unbounded dictionaries or caller-provided target names in place of the typed
binding.  Kubernetes/HPA/KEDA wiring, runtime clients, and deployment policy
remain outside this AU contract and are validated by their owning repositories.
