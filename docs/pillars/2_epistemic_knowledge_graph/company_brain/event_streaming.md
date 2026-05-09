# Event Streaming

> **The Question**: How does the Company Brain stay current when work happens across Slack, Jira, GitHub, CRM, and 50 other systems?

---

## The Problem: Batch vs. Real-Time

The existing `ingest_external_batch` API supports batch ingestion — an actor explicitly triggers a data import. This works for periodic synchronization but fails for **operational state**: by the time you batch-import, the state is already stale.

Real-time event streaming means the Company Brain updates **as work happens** — a Slack message posts, a Jira ticket moves, a PR merges, a customer call ends, and the graph reflects it immediately.

---

## Event Sources

The `EventStreamIngester` supports multiple source types:

| Source Type | Description | Example |
|:------------|:------------|:--------|
| `WEBHOOK` | External system pushes events via HTTP | Slack events API, Jira webhooks, GitHub webhooks |
| `KAFKA` | Apache Kafka topic consumer | Enterprise event bus |
| `NATS` | NATS JetStream consumer | Microservice events |
| `REDIS_STREAM` | Redis Streams consumer | Real-time notifications |
| `POLLING` | Active polling of an external API | REST APIs without webhook support |
| `CDC` | Change Data Capture from a database | PostgreSQL logical replication |
| `MCP` | Model Context Protocol server events | Agent ecosystem events |
| `A2A` | Agent-to-Agent protocol messages | Inter-agent communication |

---

## Registering Event Streams

```python
from agent_utilities.knowledge_graph.core.company_brain import CompanyBrain
from agent_utilities.models.company_brain import (
    ActorType, EventSourceType, EventStreamConfig, WebhookEvent
)

brain = CompanyBrain()

# Register a Slack webhook stream
brain.events.register_stream(EventStreamConfig(
    name="Slack Engineering",
    source_type=EventSourceType.WEBHOOK,
    endpoint="https://hooks.slack.com/events/T0001/...",
    tenant_id="engineering",
    actor_id="service:slack-integration",
    actor_type=ActorType.AUTOMATED_SERVICE,
    batch_size=10,
))

# Register a GitHub webhook stream
brain.events.register_stream(EventStreamConfig(
    name="GitHub PRs",
    source_type=EventSourceType.WEBHOOK,
    endpoint="https://api.github.com/webhooks/...",
    tenant_id="engineering",
    actor_id="service:github-integration",
    actor_type=ActorType.AUTOMATED_SERVICE,
))
```

---

## Submitting Events

Events are submitted to the queue and processed in batches:

```python
# A human posts a message in Slack
brain.events.submit_event(WebhookEvent(
    source_type="slack",
    event_type="message.posted",
    payload={
        "channel": "#engineering",
        "user": "jane",
        "text": "Deploying API gateway v2.3 to production",
    },
    actor_id="analyst:jane",
    actor_type=ActorType.HUMAN,
    tenant_id="engineering",
))

# An AI agent creates a Jira ticket
brain.events.submit_event(WebhookEvent(
    source_type="jira",
    event_type="issue.created",
    payload={
        "project": "ENG",
        "summary": "API gateway health check timeout",
        "priority": "high",
    },
    actor_id="agent:monitoring",
    actor_type=ActorType.AI_AGENT,
    tenant_id="engineering",
))

# Process all queued events
result = brain.events.process_batch()
print(f"Ingested {result.events_ingested}/{result.events_received} events")
print(f"Created {result.nodes_created} nodes in {result.duration_ms:.1f}ms")
```

---

## Ingestion Results

Every batch produces an `IngestionResult` with metrics:

| Field | Description |
|:------|:------------|
| `events_received` | Total events in the batch |
| `events_ingested` | Successfully ingested |
| `events_failed` | Failed (with retry) |
| `nodes_created` | New graph nodes |
| `edges_created` | New graph edges |
| `conflicts_detected` | Conflicts found during ingestion |
| `duration_ms` | Processing time |

---

## Continuous OWL Reasoning

When significant state changes arrive via event streams, the Company Brain can trigger `OWLBridge.run_cycle()` to discover new inferred facts. This transforms passive data ingestion into **active knowledge discovery**: a customer call triggers a Slack event, which updates the graph, which triggers OWL reasoning, which discovers that the customer is now in a new risk category.
