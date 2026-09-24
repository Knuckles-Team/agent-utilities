# AU-ECO.toolkit.journey-map-milestones — Native Messaging Backend Abstraction

> **CONCEPT:AU-ECO.messaging.native-backend-abstraction** | Pillar 4: Ecosystem & Peripherals
>
> Provides a pluggable, transport-agnostic messaging framework enabling agents
> to send and receive messages across 17+ platforms with full KG integration.

---

## Overview

The Native Messaging Backend Abstraction extends the agent-utilities ecosystem
with bidirectional, event-driven messaging across 17 platforms. Unlike MCP-based
request-response integrations, this system uses persistent connections with
`AsyncIterator`-based inbound event streaming for real-time messaging.

### Design Philosophy

The architecture follows three proven patterns already in agent-utilities:

1. **`TraceBackend` ABC** (`harness/trace_backend.py`) — abstract methods define
   the contract, concrete methods provide defaults, factory auto-detects backends
2. **`PluginRegistry`** (`core/registry/plugin_adapter.py`) — dynamic discovery via
   `importlib.metadata.entry_points` for zero-config backend registration
3. **`store_memory()`/`recall_memory()`** (`knowledge_graph/core/engine_memory.py`)
   — inbound messages auto-ingest into the KG as tiered episodic memory nodes

---

## Architecture

<div class="admonition architecture" markdown>
<p class="admonition-title">17 backends share one ABC; the router bridges to the planner and KG</p>

`agent_utilities/messaging/` provides `base.py` (`MessagingBackend` ABC),
`models.py` (Pydantic models), `registry.py` (entry-point discovery),
`router.py` (inbound event router), `capabilities.py` (capability
matrix), and `kg_ingest.py` (KG auto-ingest). All 17 platform backends
(discord, slack, telegram, whatsapp, teams, googlechat, googlemeet,
mattermost, matrix, irc, signal, imessage, line, twitch, synology,
voicecall, nextcloud) implement `MessagingBackend`.

`router.py` routes to the Planner Graph Agent
(`AU-ORCH.planning.recursion-nesting-depth`). `kg_ingest.py` calls
`store_memory()` on the Knowledge Graph
(`AU-KG.memory.tiered-memory-caching`), and the Planner calls
`recall_memory()` on the same Knowledge Graph.
</div>

## Inbound Message Flow

<div class="admonition architecture" markdown>
<p class="admonition-title">Inbound message flow: store, recall, route, orchestrate, deliver</p>

A platform (Discord, Slack, …) sends a new message event to its
`MessagingBackend`, which yields an `InboundEvent` to `InboundRouter`.
The router stores episodic memory and recalls context from the Knowledge
Graph, then routes to the Planner Graph Agent with that KG context. The
planner orchestrates specialist agents, which return a response; the
planner calls `send_message(response)` on the backend, which delivers it
back to the platform.
</div>

---

## Installation

```bash
# Single backend
pip install agent-utilities[messaging-discord]

# Multiple backends
pip install agent-utilities[messaging-discord,messaging-slack,messaging-telegram]

# All 17 backends
pip install agent-utilities[messaging]
```

## Quick Start

```python
from agent_utilities.messaging import MessagingRegistry, MessagingBackend

# Discover installed backends
registry = MessagingRegistry()
print(registry.list_backends())  # ['discord', 'slack', ...]

# Create and connect a backend
discord = registry.create_backend("discord")
await discord.connect()

# Send a message
result = await discord.send_message("#general", "Hello from agent!")

# Listen for inbound messages
async for event in discord.listen():
    print(f"[{event.platform}] {event.user_name}: {event.content}")
```

## Configuration (XDG config.json)

All messaging configuration is managed through the unified XDG config file:

```
~/.config/agent-utilities/config.json
```

> **See:** [AU-ECO.toolkit.journey-map-milestones Messaging Configuration Guide](ECO-4.5-Messaging_Configuration_Guide.md)
> for the complete per-platform reference with all keys and env var mappings.

### Config Priority Chain

```
1. Environment variable (MESSAGING_DISCORD_TOKEN=vault://messaging/discord#token) ← highest
2. XDG config.json (~/.config/agent-utilities/config.json)
3. Platform-native process env vars (DISCORD_BOT_TOKEN)       ← lowest
```

### Minimal config.json Example

```json
{
    "messaging_enabled_backends": ["discord", "slack"],
    "messaging_kg_ingest": true,
    "messaging_route_to_planner": true
}
```

Inject concrete platform tokens through the service process environment or a
governed secret resolver. AgentConfig never reads a checkout `.env`, and raw
credential values do not belong in `config.json`.

### How It Works

<div class="admonition architecture" markdown>
<p class="admonition-title">config.json to environment variables to typed fields and backends</p>

`config.json` loads via `_load_xdg_json_config()` into environment
variables, which `AgentConfig` (Pydantic) parses into
`config.messaging_*` fields and which `MessagingRegistry._auto_config()`
reads to create a backend instance. `config.reload()` swaps in a newly
validated proxy snapshot of those fields.
</div>

All `messaging_*` keys in `config.json` are:
1. Loaded at startup by `_load_xdg_json_config()` (uppercased to env vars)
2. Parsed into first-class `AgentConfig` fields (accessible as `config.messaging_discord_token`)
3. Read by `MessagingRegistry._auto_config()` when creating backend instances

### XDG Directory Layout

- `~/.config/agent-utilities/`
    - `config.json` — all messaging config lives here
    - `mcp_config.json`
    - `a2a_config.json`

- `~/.local/share/agent-utilities/`
    - `kg/knowledge_graph.db` — messages stored here as memory nodes
    - `messaging/`
        - `sessions/` — backend-specific auth state
        - `history/` — local message history cache
    - `...`

### WhatsApp Dual Mode

```bash
MESSAGING_WHATSAPP_USE_BUSINESS_API=true \
MESSAGING_WHATSAPP_TOKEN="$(secret-controller read messaging/whatsapp-token)" \
MESSAGING_WHATSAPP_PHONE_NUMBER_ID="$(secret-controller read messaging/whatsapp-phone-id)" \
graph-os
```

Set `messaging_whatsapp_use_business_api` to `false` (default) for the
`neonize` bridge, which connects via QR code with no token required.

---

## Capability Matrix

| Capability | Discord | Slack | Telegram | WhatsApp | Teams | GChat | GMeet | MM | Matrix | IRC | Signal | iMsg | LINE | Twitch | Syno | Voice | NC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Send text | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Media | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ |
| Threads | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Reactions | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Typing | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Inbound | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Voice | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |

---

## Module Reference

| Module | Purpose | CONCEPT |
|---|---|---|
| `messaging/__init__.py` | Public API surface | AU-ECO.toolkit.journey-map-milestones |
| `messaging/base.py` | `MessagingBackend` ABC | AU-ECO.toolkit.journey-map-milestones |
| `messaging/models.py` | Pydantic data models | AU-ECO.toolkit.journey-map-milestones |
| `messaging/registry.py` | Entry-point backend discovery | AU-ECO.toolkit.journey-map-milestones |
| `messaging/capabilities.py` | Platform capability matrix | AU-ECO.toolkit.journey-map-milestones |
| `messaging/router.py` | Inbound → Planner Graph Agent | AU-ECO.toolkit.journey-map-milestones + ORCH-1.1 |
| `messaging/kg_ingest.py` | KG auto-ingest | AU-ECO.toolkit.journey-map-milestones + KG-2.1 |
| `messaging/backends/*.py` | 17 platform implementations | AU-ECO.toolkit.journey-map-milestones |

---

## KG Integration (CONCEPT:AU-KG.memory.tiered-memory-caching)

Inbound messages are automatically ingested as `episodic` memory nodes:

```python
engine.store_memory(
    content="[DISCORD] Message from user in #general: Hello!",
    memory_type="episodic",
    tags=["platform:discord", "channel:general", "user:alice"],
    trust_score=0.7,
)
```

Agent responses are stored as `semantic` memory (longer half-life):

```python
engine.store_memory(
    content="[DISCORD] Agent response in #general: I can help with that.",
    memory_type="semantic",
    trust_score=0.9,
)
```

Both leverage the existing `MemoryDecayConfig` (CONCEPT:AU-KG.memory.auto-similarity-memory-graph) for
Ebbinghaus-curve-based relevance decay over time.

---

## Cross-References

- **CONCEPT:AU-ORCH.planning.recursion-nesting-depth** — Planner Graph Agent receives routed messages
- **CONCEPT:AU-KG.memory.tiered-memory-caching** — Tiered Memory for conversation persistence
- **CONCEPT:AU-KG.memory.auto-similarity-memory-graph** — Memory decay for message relevance scoring
- **CONCEPT:AU-ECO.messaging.native-backend-abstraction** — Plugin registry pattern for backend discovery
- **CONCEPT:AU-OS.safety.doom-loop-detection** — XDG paths for session/config storage
