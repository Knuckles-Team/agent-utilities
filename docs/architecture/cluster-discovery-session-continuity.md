# Engine-authoritative cluster discovery and session continuity

`GRAPH_SERVICE_ENDPOINTS` is a bounded bootstrap contact list. It is not a
per-group placement map. Once an authenticated engine connection is available,
the AU consumer calls the native `ClusterMembers` operation and uses its
verified leader/follower endpoint and certificate metadata for placed groups.

`GRAPH_RAFT_GROUP_ENDPOINTS` remains parseable for configuration migration and
operator audit, but it is not a live endpoint authority. Neither a stale map
nor the additive endpoint hint on a placement response can override the
verified membership snapshot. Remote plaintext `tcp://` members are rejected;
remote members require `tls://` plus a verified server name.

The consumer cache is process-local and keyed by the verified tenant,
principal, agent, and optional cluster pin. A snapshot is usable only while:

- its membership and placement epochs are monotonic;
- its member certificate metadata is currently valid (within the configured
  clock-skew bound); and
- its explicit freshness TTL has not elapsed.

Transport errors may use a still-current last-good snapshot. A malformed,
cross-cluster, cross-tenant, unsigned/rejected, stale-epoch, or certificate
response never falls back to that snapshot. A process restart discards it and
must authenticate and discover again.

Routed `GraphSession` values carry only bounded discovery metadata (cluster id,
epochs, certificate rotation, and a monotonic expiry). It is not durable
credential or session state. The GraphOS/MCP process transport also has an
explicit drain gate: shutdown stops new operations, waits up to
`GRAPH_DRAIN_TIMEOUT_S`, exposes a timed-out drain as a failure, and then closes
the process-owned transport. A replacement pod cannot silently claim continuity
after a timed-out drain or a lost process; it must establish a new verified
session and topology snapshot and recover durable work through its normal
checkpoint/outbox paths.

The four authorities remain separate:

1. the engine signs and verifies cluster membership;
2. AU validates consumer freshness, epoch, certificate, and context bounds;
3. `GraphSession` carries bounded route metadata for one operation; and
4. the MCP/GraphOS lifecycle gate controls admission and drain.

No endpoint, bearer, certificate value, or durable session token is written to
the graph as a continuity claim.
