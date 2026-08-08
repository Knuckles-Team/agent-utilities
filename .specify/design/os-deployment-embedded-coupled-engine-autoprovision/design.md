# Design Document: The embedded/tiny deployment case provisions a lifecycle-COUPLED engine (dies with the process) rather than the resolver's default detached+supervised engine other entrypoints share

CONCEPT:AU-OS.deployment.embedded-auto-provision

> `agent_utilities/knowledge_graph/core/graph_compute.py:1155-1178`
> (`ensure_local_engine`), contrasted with
> `agent_utilities/knowledge_graph/core/engine_resolver.py:1-27`
> (`AU-OS.deployment.engine-resolver-auto-provision`, the general resolver this
> specializes).

## Decision — `ensure_local_engine()` provisions a `GraphComputeEngine(coupled=True)` for a server with no remote engine configured, reusing the SAME lock-guarded scan-or-spawn machinery the general resolver's autostart leg uses, but pinned to the coupled (dies-with-this-process) lifecycle rather than the resolver's default detached+supervised (reference-counted, shared-by-other-entrypoints) lifecycle

`AU-OS.deployment.engine-resolver-auto-provision` documents the general precedence
(remote → shared-local → autostart-shared-supervised) every entrypoint uses to
reach an engine. `ensure_local_engine` is a narrower, explicit special case of that
same machinery for the true single-process embedded deployment: when the resolved
endpoint is local, this provisions the engine as `coupled=True`
(`graph_compute.py:1159-1165`) — the engine's lifetime is tied directly to this
one process, so it terminates when the process does, instead of the resolver's
DEFAULT detached posture (survives the spawning process, reference-counted idle
shutdown, shared by other co-located entrypoints). It reuses the exact same
`engine_lock.engine_spawn_guard`-protected scan-or-spawn path the resolver's
autostart leg already runs — "this adds no new locking" (`graph_compute.py:1167`) —
active only for the packaged-local path (`GRAPH_SERVICE_ENDPOINTS` unset) and a
no-op for every explicit connect-only topology.

## Rejected alternative — always use the resolver's default detached+supervised lifecycle, even for the true single-process embedded case

The resolver's own default behaviour (detached, supervised, reference-counted idle
shutdown, shared across co-located entrypoints) is the CORRECT choice when multiple
processes on a host might want to share one engine — that is precisely what it is
designed for. Applying that same default unconditionally to the embedded/tiny
profile was rejected because a genuinely single-process embedded deployment has no
other entrypoint to share the engine WITH — a detached engine there would outlive
its one and only client for no benefit, consuming resources until its idle-grace
timer expired, and would need a SEPARATE supervisory mechanism to ever be reliably
cleaned up. Pinning `coupled=True` for this specific case makes the engine's
lifetime match its actual usage pattern exactly — one process, one engine, both
start and stop together — without inventing a new locking or spawn mechanism to get
there.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/core/graph_compute.py`
  (`ensure_local_engine`, `GraphComputeEngine.__init__` `coupled=True` path).
- **Backward Compatible**: Yes — active only when `GRAPH_SERVICE_ENDPOINTS` is
  unset (the packaged-local/embedded profile); every explicit-topology deployment
  is unaffected.
- **Known weak point**: the choice between coupled and detached is made once, by
  which code path calls into engine provisioning — a deployment that starts out
  genuinely single-process (correctly gets `coupled=True`) but later adds a second
  co-located entrypoint does not automatically migrate to the shared/detached
  posture; that requires the deployment's own topology decision (setting
  `GRAPH_SERVICE_ENDPOINTS` or otherwise routing through the general resolver) to
  change accordingly.
