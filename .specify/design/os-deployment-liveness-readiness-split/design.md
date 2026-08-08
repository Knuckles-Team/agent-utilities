# Design Document: `/health` (liveness) always returns 200 even when unhealthy; only `/health/ready` (readiness) maps health status onto the HTTP status code — one shared health-collector core, two different HTTP-level contracts

CONCEPT:AU-OS.deployment.liveness-vs-readiness-split

> `agent_utilities/mcp/kg_server.py:3811-3821` (the liveness/readiness route
> pair); `agent_utilities/observability/runtime_health.py:1-56` (module
> docstring, the four-state health model, `collect_health`).

## Decision — both `/health` and `/health/ready` (plus `graph_configure(action="health")`) dispatch into the ONE shared `observability.runtime_health.collect_health` core, but the two HTTP routes interpret its output differently: `/health` (liveness) always answers 200 regardless of body content — a dependency-free, status-only liveness signal — while `/health/ready` (readiness) maps the SAME report's overall status onto the actual HTTP status code (200/503)

`collect_health` computes ONE truthful report with a four-state per-check model
(`ok` / `unhealthy` / `not_configured` / `degraded` — `runtime_health.py:32-42`),
used by the KG server's `/health`/`/health/ready`, the REST gateway's own
`/health`/`/health/ready`, and `graph_configure(action="health")` — never a second
implementation that can drift (`kg_server.py:3814-3816`). What differs is only how
each ROUTE consumes that one truthful report: "it is the callers... that decide
what to do with it" (`runtime_health.py:44-45`). Liveness must keep returning 200
because it answers a narrower question than readiness — "is this process itself
alive and answering" — independent of whether some downstream dependency is
currently unreachable. Readiness answers the broader, traffic-routing question by
mapping the SAME report onto 200/503 so a load balancer/kubelet can stop routing
without touching the process itself. The `degraded` state exists specifically so an
optional, non-mandatory dependency's partial impairment (e.g. a read-only KG-mirror
connection) never fails the overall rollup and pulls a healthy engine out of
Service routing over something the engine's own authority does not actually depend
on (`runtime_health.py:37-41`).

## Rejected alternative — one endpoint, one health signal, used for both liveness and readiness (or readiness-shaped liveness)

The rejected alternative is stated as a direct consequence, not a hypothetical: "A
liveness probe must keep returning 200... even when the body says 'unhealthy' —
killing/restarting a process because a *downstream dependency* is unreachable just
crash-loops a perfectly fine pod" (`runtime_health.py:46-48`). Using one endpoint
(or readiness-shaped semantics) for BOTH liveness and readiness means a downstream
outage — a database, an optional connector, anything the process itself has no
control over — gets treated as "this process is broken," triggering a
kill-and-restart cycle that does nothing to fix the actual dependency and instead
adds process churn on top of an already-degraded system. Splitting the two
questions (is the process alive vs. should traffic be routed to it) lets Kubernetes'
own liveness/readiness contract do what it is designed for: readiness correctly
stops routing traffic during an outage, while liveness correctly leaves the
still-functioning process running so it can recover the moment the dependency comes
back, instead of restarting it into the exact same wait.

## Risk Assessment

- **Blast Radius**: `agent_utilities/mcp/kg_server.py` (route pair),
  `agent_utilities/observability/runtime_health.py` (`collect_health`), the REST
  gateway's equivalent `/health`/`/health/ready` pair.
- **Backward Compatible**: Yes — an unauthenticated, additive HTTP surface; no
  existing check semantics change.
- **Known weak point**: correctness of the whole split depends on every deployment
  manifest actually wiring liveness and readiness probes to the RIGHT respective
  route — a Kubernetes manifest that (mis)configures both probes against `/health`
  (liveness) reintroduces exactly the crash-loop-on-dependency-outage failure this
  split exists to prevent, and nothing in this module itself can detect that
  misconfiguration from the server side.
