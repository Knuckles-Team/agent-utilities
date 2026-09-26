# Governed Warm-Fork Sandboxes

Agent-utilities exposes one current warm-fork execution path: a governed Firecracker microVM
backend controlled through `forkd`. The shared protocol remains backend-neutral so the engine's
zero-copy KV snapshot/fork primitive and future confined runtimes can use the same lifecycle
without weakening the RLM isolation boundary.

## Runtime contract

`ForkableSandbox` extends the ordinary `Sandbox` contract with three operations:

1. `warm_spec()` returns a content-addressed description of the parent snapshot.
2. `warm(spec)` prepares one confined parent and returns a `ParentHandle`.
3. `run_forked(parent, code, env)` forks one child, executes the snippet, and returns a
   `SandboxResult`.

`ForkableSandbox.execute()` acquires or creates the parent through `WarmParentRegistry`, then
delegates to `run_forked`. The registry is bounded by host CPU and available memory, applies idle
and absolute-age limits, and drains on daemon shutdown. A backend that does not advertise a real
isolation boundary is rejected before it can execute model-generated code.

## Current backend

| Backend | Boundary | Warm primitive | Host callbacks | Availability |
|---|---|---|---|---|
| `firecracker` | dedicated microVM | governed `forkd` snapshot → child | no | reachable controller, KVM, approved snapshot |

The Firecracker client uses the bounded outbound HTTP policy, validates controller paths, and
supports authenticated controller access. It is registered only when its controller and snapshot
are available; otherwise the ordinary RLM router continues over the current confined Monty,
WASM, and Docker backends and warm-fork-specific calls return a structured unavailable result.

## Cross-modal fan-out

`CrossModalForkFanout` retrieves the engine's vector + graph + text candidate set exactly once,
then gives each governed child an isolated view. The result records `retrieval_calls`, so a second
retrieval is a visible correctness failure rather than a hidden N-times recomputation.

When callers provide engine KV page keys, the epistemic-graph KV surface adds zero-copy page
sharing:

```text
retrieve once → KV snapshot(keys) → fork snapshot N times → run N governed branches
```

Snapshot pages remain shared while each branch writes to its own copy-on-write overlay. The
`/kv/fork/stats` counters prove that shared resident bytes stay flat as branch count grows. If the
KV surface is unavailable, branch execution continues without engine page sharing; it never
falls back to an unconfined executor.

## Operator surface

`graph_sandbox` provides lifecycle visibility without exposing arbitrary execution:

- `status` reports current backends, availability, warm-capable rungs, pool size, and reward EMA.
- `warm` prewarms a named current `ForkableSandbox` backend.
- `reap` closes stale warm parents and idle developer workspaces.

`agent-utilities-doctor` reports the same backend and pool state. A warning means warm starts are
unavailable; it does not mean the RLM router will use an unconfined path.

## Security invariants

- Only backends with `SandboxCapabilities.isolated=True` can be routed or prewarmed.
- Missing or unreachable isolation fails closed.
- Controller requests use bounded URLs, timeouts, redirect limits, and configured TLS policy.
- Parent resources have both idle and absolute lifetime caps and are drained on shutdown.
- Branch failures are returned as bounded public errors; secrets and controller responses are not
  copied into model-visible output.
