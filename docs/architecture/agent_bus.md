# Agent Communication Bus

AgentBus provides tenant-isolated presence, direct messaging, topics,
federation, and work delegation through the `graph_bus` MCP/REST surface.

## Current architecture

```text
send
  -> ActionPolicy
  -> native BusOutbox transaction (pending)
  -> bounded partitioned log publish
  -> BusOutbox confirmation (published)

materialize
  -> consume without acknowledgement
  -> native transaction:
       BusInbox + WorkItem + BusDeliveryOutcome + MutationOutbox
  -> acknowledge log receipt

receive
  -> read committed BusInbox rows for the authenticated participant
```

The knowledge graph stores low-churn semantic state (`BusAgent`, `Topic`, and
`BusSubscription`) and durable transaction records. High-volume message bodies
travel through the required partitioned log; there is no `BusMessage` mailbox,
graph fallback, per-recipient queue, per-subscriber consumer, or public manual
acknowledgement path.

## Identity and privacy

Every queue, partition key, envelope, inbox, outbox, subscription, and WorkItem
is tenant-qualified. Mutations require an authenticated, tenant-qualified
`GraphSession`; a supplied tenant that differs from the session is rejected.

Raw tenant, participant, topic, host, session, and actor values are replaced by
stable opaque references before persistence or routing. Payload and metadata use
the shared persistence privacy guard. Production requires an operator-managed
identity HMAC key reference (or the configured service-auth secret). Source code
contains no endpoint, credential, username, or local filesystem profile.

## Bounded delivery plane

`AGENT_BUS_LOG_BACKEND` must be `engine` or `kafka` and defaults to `engine`.
The selected backend is a hard contract and fails closed when unavailable.

- Engine broker: a fixed `AGENT_BUS_PARTITIONS` queues per tenant. Stable
  hashing maps direct and topic events to a partition.
- Kafka: two keyed topics (`agent_bus_direct`, `agent_bus_topic`) and one shared
  materializer consumer group per tenant/topic pair.
- `AGENT_BUS_MAX_CONSUMERS` bounds process-local Kafka consumers.
- `AGENT_BUS_MAX_DEPTH` applies publisher backpressure before accepting more
  work.
- `AGENT_BUS_MAX_TOPIC_SUBSCRIBERS` bounds one event's transactional fan-out;
  subscriptions fail closed at the bound.
- `BUS_LOG_MAX_MESSAGES_PER_RECEIVE` bounds each drain.
- `AGENT_BUS_DELIVERY_LEASE_SECONDS` configures the redeliverable engine
  receipt visibility lease used while inbox transactions commit. Executable
  WorkItems then use renewable fenced leases.
- Poison records go to the backend DLQ using a digest and byte count, never the
  raw body.

Topic subscriptions become effective at `subscribed_at`. Topic events are
expanded against the authoritative subscription registry during materialization.
Subscribers do not receive events created before their subscription. Durable
replay means redelivery of unacknowledged log records and idempotent re-commit;
there is no late-subscriber history/backfill API.

## Commit, deduplication, and recovery

Message groups, outbox records, inboxes, WorkItems, outcomes, and mutation
outboxes have deterministic tenant-scoped identifiers. The broker receipt is
acknowledged only after every target transaction commits. If one target fails,
the event is nacked/requeued; already committed targets are detected as replays.
If acknowledgement fails after commit, redelivery observes the same identifiers
and is safe.

Pending send intents are replayed in bounded batches before receive-side
materialization. A crash after publish but before confirmation can therefore
publish twice, but deterministic inbox/WorkItem identifiers collapse it to one
durable delivery. A failed immediate publish response includes
`durable=true` and `queued_for_replay=true`; it never claims broker publication.

The WorkItem created with each inbox is the sole writable execution authority.
BusInbox and BusDeliveryOutcome are delivery/audit facts, not task-state
projections.

## Presence, topics, and federation

Presence is derived from `last_seen`; no liveness reaper writes status. Topic
subscriptions are first-class records and allocate no broker resources.
Federation relays a sanitized message group through the same outbox/log path,
deduplicates by group, breaks loops with an opaque origin reference, and only
forwards allowed markings.

## Configuration and operations

| Setting | Default | Purpose |
|---|---:|---|
| `AGENT_BUS_LOG_BACKEND` | `engine` | Required delivery log (`engine` or `kafka`) |
| `AGENT_BUS_PARTITIONS` | `6` | Fixed/grow-only delivery partitions |
| `AGENT_BUS_MAX_CONSUMERS` | `32` | Hard Kafka consumer bound |
| `AGENT_BUS_MAX_DEPTH` | `100000` | Publisher backpressure threshold |
| `AGENT_BUS_MAX_TOPIC_SUBSCRIBERS` | `1024` | Maximum subscribers per topic |
| `AGENT_BUS_DELIVERY_LEASE_SECONDS` | `300` | Engine receipt lease duration |
| `BUS_IDENTITY_HMAC_KEY_REF` | unset | Production opaque-identity secret reference |

The bus doctor check reports participant presence, backend, log depth, and
pending send-outbox count. It never reports raw identities, payloads, endpoints,
or local paths.

Implementation:

- `agent_utilities/messaging/bus.py` — registry and orchestration
- `agent_utilities/messaging/bus_log.py` — fixed partition delivery
- `agent_utilities/messaging/bus_inbox.py` — native transactional materialization
- `agent_utilities/messaging/bus_privacy.py` — opaque references and sanitization
- `agent_utilities/messaging/federation.py` — cross-hub relay
