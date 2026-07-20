# Engine-authoritative placement walkthrough

Agent Utilities does not assign graphs to endpoints. It connects to an Epistemic
Graph coordinator, asks the authenticated `PlacementRoute` authority for a complete
route, validates the group/epoch/fence, and sends the operation to that route.

## Recommended topology

Configure one stable coordinator contact:

```bash
export GRAPH_SERVICE_ENDPOINTS="tls://engine-coordinator.example:9100"
```

Credentials and private trust material belong in the runtime secret provider and
TLS profile, not in repository configuration.

For a deployment that exposes Raft groups on distinct client endpoints, add a
strict group map:

```bash
export GRAPH_SERVICE_ENDPOINTS='["tls://coordinator-a.example:9100","tls://coordinator-b.example:9100"]'
export GRAPH_RAFT_GROUP_ENDPOINTS='{"0":"tls://group-zero.example:9100","1":"tls://group-one.example:9100"}'
```

Group keys must be non-negative integers. Endpoint values require `unix://`,
`tcp://`, or `tls://`. Invalid JSON, missing schemes, and ambiguous multi-contact
topology fail closed.

## Request flow

1. Authentication middleware creates a verified `GraphSession` containing tenant,
   actor, scopes, audience, and policy revision.
2. Tenant naming resolves the logical graph, such as
   `tenant__<opaque-tenant>__<base>`.
3. The routed client asks a configured coordinator for `PlacementRoute`.
4. The client requires `authoritative=true`, a boolean `placed`, numeric `group`,
   `epoch`, and `fencing_token`, with `fencing_token == group`. A placed route with
   epoch zero is invalid.
5. The operation carries that route. A structured stale-route response invalidates
   the short cache and retries exactly once with the same idempotency key.

An unplaced single-node answer is still authoritative: `placed=false`, group zero,
and epoch zero. It is not permission for a caller to hash or choose another target.

## Co-residency and union reads

Graph affinity is placement metadata owned by the engine. The Python pool groups a
multi-graph read by the endpoints returned by the authority and performs bounded
scatter-gather. It never rewrites a graph key to manufacture co-residency.

## Validate

Run the redacted doctor:

```bash
agent-utilities-doctor --only engine
```

The report exposes counts and readiness, not endpoint strings, host paths,
credentials, tenant names, or actor identifiers. A multi-contact deployment without
the required group map is reported as a configuration failure.

For movement, use the governed engine resharding surface. A successful move advances
the catalog epoch and fences the former owner; changing a client contact list never
moves data.
