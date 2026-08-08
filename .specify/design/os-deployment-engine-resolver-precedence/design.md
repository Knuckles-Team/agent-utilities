# Design Document: ONE engine resolver, ONE fixed precedence (remote → shared-local → autostart-shared-supervised), used by EVERY entrypoint — with the doctor's "engine" check reporting the resolved mode, not just transport reachability

CONCEPT:AU-OS.deployment.engine-resolver-auto-provision (covers the cluster:
`AU-OS.deployment.report-resolved-mode` is the pointer for the doctor-reporting
half of this same resolution)

> `agent_utilities/knowledge_graph/core/engine_resolver.py:1-33`
> (module docstring, `resolve_engine`); consumed by
> `agent_utilities/deployment/doctor.py:1486-1512` (`_check_engine`'s
> resolved-mode reporting branch, `CONCEPT:AU-OS.deployment.report-resolved-mode`).

## Decision — every entrypoint that needs an engine (graph-os MCP, the gateway/host daemon, `IntelligenceGraphEngine`, the facade, `EpistemicGraphBackend`, the tenant engine pool, messaging, agent/serving) funnels through `GraphComputeEngine.__init__` calling `resolve_engine`, which decides by ONE fixed precedence — remote (fail-loud if configured-but-unreachable, never autostarts) → shared-running-local (cheap probe or spawn-lock-holder verification) → autostart-shared-supervised (flock-guarded double-checked spawn) — with NO per-entrypoint code, and `doctor`'s engine check reports which leg actually resolved, not just whether SOME endpoint answered

Before a single resolver, each of the ~7+ entrypoints that need an engine would
otherwise have had to independently decide: is a remote engine configured, is one
already running locally, or do I need to start one. `engine_resolver.py`'s decision
is to make that ONE function, reused by every entrypoint verbatim: **remote** wins
whenever `GRAPH_SERVICE_ENDPOINTS` is configured and never autostarts a local
stand-in even if unreachable ("an unreachable configured remote stays fail-loud" —
`engine_resolver.py:10-12`, so a misconfigured remote never silently degrades into
a divergent local engine); **shared** reuses an already-serving local endpoint
(cheap connect probe, or a verified spawn-lock holder) so co-located entrypoints
share ONE engine; **autostart** spawns a detached, supervised engine under a
first-one-wins flock, reference-counted to self-terminate after its last client
disconnects unless the operator chose persistent. The resolver "invents no new
locking, probing, auth, or topology logic" (`engine_resolver.py:31`) — it composes
existing building blocks (`shard_topology.resolve_endpoints`/`is_local_endpoint`/
`probe_endpoint`). `doctor`'s engine check then reports on THIS resolution
specifically — "report the RESOLVED mode (how this process reaches the engine), not
just transport reachability" (`doctor.py:1486-1487`) — because two deployments can
both show "engine reachable" while one silently autostarted a local engine when
remote was intended, a difference only the resolved mode (not raw reachability)
surfaces.

## Rejected alternative — let each entrypoint decide its own engine-provisioning strategy, and let doctor report only reachability

The rejected shape is named directly: "ONE engine resolver — the single chokepoint
provisions an engine for *every* entrypoint" (`engine_resolver.py:2`) implies the
alternative was N per-entrypoint implementations, each independently probing/
spawning — the same "N places to remember" failure shape rejected elsewhere in this
codebase (`AU-ORCH.execution.delegation-hot-path-authority`): a new entrypoint (or
an existing one nobody thought to update) would silently reinvent — or worse,
subtly diverge from — the remote/shared/autostart precedence, risking two
co-located entrypoints each spawning their OWN engine instead of sharing one.
Separately, reporting only raw transport reachability from `doctor` (rather than the
resolved MODE) was rejected because "resolved mode" and "reachable" are genuinely
different facts: a remote-configured deployment with an unexpectedly-live LOCAL
engine (e.g. leftover from a prior autostart) can show "reachable" on both counts
while pointing at the wrong one — reachability alone cannot distinguish that from
the intended state.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/core/engine_resolver.py`,
  `agent_utilities/knowledge_graph/core/graph_compute.py`
  (`GraphComputeEngine.__init__`), `agent_utilities/deployment/doctor.py`
  (`_check_engine`).
- **Backward Compatible**: Yes — every existing entrypoint already routes through
  `GraphComputeEngine.__init__`; this centralizes decision logic already reachable
  from that one constructor.
- **Known weak point**: the resolver "trusts" (per its own docstring framing of the
  chokepoint property) that every entrypoint really does construct its engine via
  `GraphComputeEngine.__init__` — a future entrypoint that reaches the engine some
  other way would silently bypass the shared precedence, the same class of risk
  `AU-ORCH.execution.delegation-hot-path-authority`'s design doc flags for its own
  convergence-point assumption.
