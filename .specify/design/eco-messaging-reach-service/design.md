# Design Document: One governed reach service for outbound send + channel routing

CONCEPT:AU-ECO.messaging.messaging-reach-service-governed ·
CONCEPT:AU-ECO.messaging.last-active-channel-routing ·
CONCEPT:AU-ECO.messaging.messaging-ontology-shape-so

> `agent_utilities/messaging/service.py`; the OWL/RDF shape in
> `agent_utilities/knowledge_graph/ontology/interfaces.py`.

## Decision — `MessagingService` is the ONE core every entrypoint dispatches through

`CONCEPT:AU-ECO.messaging.messaging-reach-service-governed`

The MCP tool (`graph_reach`), the REST twin (`/graph/reach`), the inbound
router, the goal-loop, and the pydantic-ai agent toolset all dispatch
outbound sends and channel-routing decisions into ONE `MessagingService`
singleton (`MessagingService.instance()`), not five independent
implementations of "send a message." It owns: the set of connected backends
(auto-detected via `MessagingRegistry`), governed outbound sends (every send
passes the fail-closed `ActionPolicy` gate,
`orchestration/action_policy.py`, `kind="message.send"`), the durable
last-active-channel routing state, and auto-ingestion of every message into
the KG as conversational memory (`service.py:1-16`).

**The rejected alternative** is each of those five call sites (MCP tool,
REST route, router, goal-loop, agent toolset) independently deciding which
backend to use and independently choosing whether to enforce the
ActionPolicy gate — a governance rule enforced in four of five places is not
a governance rule. Centralizing in one singleton means the ActionPolicy gate
cannot be bypassed by adding a sixth caller that forgets to check it; there
is only one send path to gate.

### Pointer — `CONCEPT:AU-ECO.messaging.last-active-channel-routing`

`service.py:37` (`_PREF_NODE_PREFIX = "chanpref:"`) and the docstring's own
citation of it. Like OpenClaw, `reach_user` delivers to whatever channel the
user last interacted on, falling back to a configured default so a fresh
system (no prior interaction recorded) still works. This state is durable
(a KG node, `chanpref:<user>`), not in-memory — a process restart does not
forget which channel a user was last active on.

### Pointer — `CONCEPT:AU-ECO.messaging.messaging-ontology-shape-so`

`knowledge_graph/ontology/interfaces.py:815`. The OWL/RDF-native counterpart
of the routing state above: a `MessagingChannel` is registered as a formal
`Interface` shape — "any platform conversation the system can reach a user
on" — with the durable `UserChannelPreference` node (this same last-active
state) implementing it. This is what lets the KG's reasoner relate users to
their last-active channel as a first-class, queryable OWL/RDF relation,
rather than last-active-channel being an opaque string property only the
Python code understands. The domain-triage tool flagged this id as a
slugified-prose-fragment retire candidate; reading the ontology registration
directly shows it names a real, deliberate SHACL-like shape decision, so
this document documents it rather than retiring the marker.

## Risk Assessment

- **Blast Radius**: `agent_utilities/messaging/service.py`,
  `agent_utilities/orchestration/action_policy.py`,
  `agent_utilities/knowledge_graph/ontology/interfaces.py`.
- **Backward Compatible**: Yes.
- **Known weak point**: the singleton pattern (`MessagingService.instance()`)
  means all five callers share exactly one routing/backend state — correct
  for a single-tenant deployment, but a future multi-tenant deployment would
  need to thread a tenant key through `instance()` rather than relying on
  process-wide global state.
