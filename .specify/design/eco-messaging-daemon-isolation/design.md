# Design Document: Messaging runs as its own isolated daemon process

CONCEPT:AU-ECO.messaging.inbound-messaging-router-runs ·
CONCEPT:AU-ECO.messaging.make-fleet-credentials-present

> `agent_utilities/messaging/daemon.py`

## Decision — the `InboundRouter` runs in its OWN process, not inside the KG host daemon

`CONCEPT:AU-ECO.messaging.inbound-messaging-router-runs`

The messaging daemon (console entry point `agent-utilities-messaging`) connects to the
shared epistemic-graph engine **as a client** (`KG_DAEMON_ROLE=client`,
`daemon.py:247`) rather than hosting it. The KG host daemon owns CPU-bound
maintenance — codebase ingestion, relevance sweeps, enrichment sweeps — on the
same event loop/GIL; if the inbound router shared that process, a maintenance
tick would stall a chat reply mid-flight.

**The rejected alternative** is the obvious one: run the inbound router inside
the same process as the KG host, since it already needs the engine. The code's
own docstring names the cost directly — "an inbound message's reply is never
starved by background work sharing the event loop/GIL" is the property this
buys, not a side effect. Process isolation is the enforcement mechanism; the
OS scheduler, not application-level yielding, guarantees the router gets to
run.

The same `run_forever` body is reused unchanged by `graph-os`'s in-process
co-service supervisor (`agent_utilities.mcp.co_service_supervisor`) when
messaging is configured to run as a co-service instead of standalone — so
there is exactly ONE implementation of "connect the configured backends and
serve until told to stop," and the isolation property is a *deployment*
choice (standalone process vs. co-service) layered on top of it, not a
second code path.

### Pointer — `CONCEPT:AU-ECO.messaging.make-fleet-credentials-present`

`daemon.py:250`, `_validate_fleet_auth()`. Every process that can send
messages — standalone daemon or co-service — consumes the same XDG
`AgentConfig` declaration and validates fleet auth is resolvable *before*
connecting backends, rather than discovering a missing credential on the
first outbound send. Credential material stays behind its runtime reference
(resolved only at the outbound request, never held in the daemon's own
config surface) — the same "credentials stay in one place" principle
`.specify/design/kg-connector-mirroring/design.md` documents for source
connectors, applied here to the daemon's own auth instead of a source's.

## Risk Assessment

- **Blast Radius**: `agent_utilities/messaging/daemon.py`,
  `agent_utilities/mcp/co_service_supervisor.py`.
- **Backward Compatible**: Yes — this describes existing, shipped behavior.
- **Known weak point**: isolation is a process boundary, not a resource
  quota; a runaway backend connection inside the messaging process can still
  starve chat replies within that process. Cross-process starvation (the KG
  host stalling messaging) is what this decision rules out, not
  intra-process starvation.
