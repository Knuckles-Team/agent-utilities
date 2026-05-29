# Epistemic Graph Service Layer Architecture

> CONCEPT:KG-2.19 — Tokio-first graph service

## Overview

The epistemic-graph service layer is a long-running Tokio process that holds multiple named graphs in memory and serves requests over Unix Domain Socket (UDS) or TCP. It replaces the previous PyO3 in-process FFI approach as the primary compute backend.

```
Python (agent-utilities API gateway)
  └── GraphComputeEngine (async UDS/TCP client)
        └── epistemic-graph-server (Tokio, long-running)
              ├── GraphRegistry (named graphs)
              ├── ChannelManager (P2P, 1:1, many:many, bus)
              ├── IsolationLayer (ACL enforcement)
              └── Checkpoint/Persistence
```

## Graph Topology

### Graph Types

| Type | Naming Convention | Access | Purpose |
|---|---|---|---|
| **Bus** | `__bus__` | All agents R/W | Global event broadcast, inter-agent messaging |
| **Agent** | `agent:<id>` | Owner: full, Manager: full, Peers: denied | Private agent knowledge, episode memory |
| **Team** | `team:<name>` | Members: read, Manager: R/W | Shared team context, project knowledge |
| **Global** | `global:<name>` | All: read-only | System ontology, tool registry |

### Isolation Rules

1. **Peer isolation**: Agent graphs are invisible to peer agents
2. **Hierarchical access**: Manager agents have full access to subordinate graphs
3. **Bus is public**: `__bus__` readable/writable by all authenticated agents
4. **Team scoping**: Team graphs are read-only for members, read-write for manager
5. **Global read-only**: Global graphs are system-managed, agent-readable

## Dynamic Communication Channels

Agents can create ephemeral channels for P2P or group communication:

- **1:1 channels**: `channel:p2p:<agent_a>:<agent_b>` — direct messaging
- **Many:many channels**: `channel:group:<uuid>` — group created by any agent
- **Lifecycle**: Create → Join → Leave → Close
- **KG Imprint**: On close, the channel creates a permanent KG record with:
  - Vectorized embedding of the conversation summary
  - Participant edges preserved permanently
  - Topic metadata and timestamps

## Configuration

All settings are available in the XDG `config.json`:

| Field | Env Var | Default | Description |
|---|---|---|---|
| `graph_service_socket` | `GRAPH_SERVICE_SOCKET` | `$XDG_RUNTIME_DIR/epistemic-graph.sock` | UDS socket path |
| `graph_service_tcp_addr` | `GRAPH_SERVICE_TCP_ADDR` | `None` | TCP address (e.g., `0.0.0.0:9100`) |
| `graph_service_auth_secret` | `GRAPH_SERVICE_AUTH_SECRET` | `None` | HMAC-SHA256 shared secret |
| `graph_service_checkpoint_secs` | `GRAPH_SERVICE_CHECKPOINT_SECS` | `300` | Auto-checkpoint interval |
| `graph_service_persist_on_shutdown` | `GRAPH_SERVICE_PERSIST_ON_SHUTDOWN` | `true` | Serialize on shutdown |

## Authentication

All connections require HMAC-SHA256 authentication:
- Client computes `HMAC-SHA256(secret, request_id)` and sends it as `auth_token`
- Server verifies the token before processing any request
- For UDS-only deployments, Unix file permissions provide additional isolation
- TCP deployments **require** authentication

## API Gateway Integration

The service lifecycle is tied to the agent-utilities API gateway:
- **Startup**: Gateway sends `Reconcile` to push authoritative state from the backend
- **Shutdown**: Gateway sends `Checkpoint` to persist all graphs
- The service process is managed via the `epistemic-graph-service` CLI

## Migration from PyO3

The `GraphComputeEngine` now connects to the Tokio service by default. To fall back to in-process PyO3:

```bash
export GRAPH_COMPUTE_FALLBACK=embedded
```

This is intended as a temporary escape hatch during migration.
