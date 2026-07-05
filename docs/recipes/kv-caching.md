# Recipe — Enable the shared KV-cache (highly recommended)

> **Turn on the engine's content-addressed KV-cache HTTP surface.** This is the operational
> "how do I switch it on" companion to the
> [KV-Cache Layering guide](../guides/kvcache-vllm-lmcache.md) (what it is) and the
> [KV-Cache-Layering Policy](../architecture/kv-cache-layering-policy.md) (when a call is
> cache-worthy). Concepts: `CONCEPT:AU-KG.backend.kvcache-vllm-connector`
> (`EpistemicGraphKVBackend`, the L2 store) · `CONCEPT:EG-KG.enrichment.content-address-separation`
> (the engine's `SharedKvIndex`).

## TL;DR — two settings

The shared KV-cache is served by the **epistemic-graph engine** over a small HTTP surface
(EG-187), and reached by every client through the `EpistemicGraphKVBackend` connector. It is
**off unless you switch it on**, which is a two-part contract:

| Side | Setting | Effect |
|------|---------|--------|
| **Engine** (serves it) | `EPISTEMIC_GRAPH_KVCACHE_ADDR=<host:port>` (e.g. `0.0.0.0:9130`) | Starts the KV-cache HTTP listener. **Absent ⇒ no listener, and every client silently degrades to a cache miss.** |
| **Client** (graph-os, agents, vLLM/LMCache) | `EPISTEMIC_GRAPH_KVCACHE_URL=http://<host:port>` | Points the connector at the engine's KV surface. Defaults to `http://127.0.0.1:9130`, which is correct only when the engine serves KV on the same host. |

Optional: `EPISTEMIC_GRAPH_KVCACHE_TOKEN` (bearer auth), `EPISTEMIC_GRAPH_KVCACHE_TIMEOUT_S`
(default `2.0`), `EPISTEMIC_GRAPH_KVCACHE_MAX_CONNECTIONS` (default `32`).

> ⚠️ **The connector never raises — it degrades a missing/unreachable surface to a clean miss**
> (`put → stored:false`, `get → None`, `stats → all zeros`). So an unwired KV-cache looks
> *healthy but empty*, not broken. If `graph_kvcache stats` reports `unique_blocks:0` after a
> `put`, the engine isn't serving KV — set `EPISTEMIC_GRAPH_KVCACHE_ADDR` on the engine.

## Enable it

### On the engine

Add the bind address to the engine's launch environment (or pass nothing extra — the flag is
env-driven). The listener starts alongside the RPC transports:

```bash
EPISTEMIC_GRAPH_KVCACHE_ADDR=0.0.0.0:9130 \
epistemic-graph-server --tcp-addr 0.0.0.0:9100 --persist-dir /var/lib/epistemic-graph
# log line to confirm:
#   INFO kvcache-server: serving shared KV-cache HTTP surface on 0.0.0.0:9130
```

In a deployed swarm/compose, add it to the engine service's `environment:` block and expose the
port on the engine's overlay network so graph-os can reach it.

### On the clients (graph-os and agents)

Point the connector at the engine's KV address:

```bash
EPISTEMIC_GRAPH_KVCACHE_URL=http://<engine-host>:9130
```

For graph-os this goes in its MCP `env` block (deployed stack env, or `mcpServers.graph-os.env`
in a local `~/.claude.json`).

## Verify

The exact connector `graph_kvcache` uses, round-tripped end-to-end:

```python
import os
os.environ["EPISTEMIC_GRAPH_KVCACHE_URL"] = "http://127.0.0.1:9130"
from agent_utilities.kvcache import EpistemicGraphKVBackend
b = EpistemicGraphKVBackend.from_env()
assert b.put("probe", b"hello") is True          # stored (not the degraded False)
assert b.get("probe") == b"hello"                 # round-trip
assert b.get("miss") is None                      # clean miss
print(b.stats())                                  # unique_blocks > 0, resident_bytes > 0
```

Or over MCP once graph-os is wired:

```
graph_kvcache action=put key=probe value_b64=aGVsbG8=   # -> {"stored": true}
graph_kvcache action=get key=probe                       # -> {"hit": true, ...}
graph_kvcache action=stats                                # -> unique_blocks > 0
```

## Authentication (paired with the platform's OIDC — not a separate mechanism)

The KV surface authenticates with the **same Keycloak auth as graph-os**. Two modes,
**JWT preferred**; a static token is the documented fallback. Both sides apply the
same precedence: **JWT if configured, else static token, else anonymous** (dev only —
never leave a network-reachable KV surface anonymous in production).

### JWT (recommended) — the platform's OIDC client-credentials

The connector presents the **same Keycloak client-credentials bearer graph-os mints
for the fleet**, and the engine validates it against the realm JWKS (RSA signature +
issuer + audience + expiry). Token **refresh and cold restarts are handled
automatically** by the shared `ClientCredentialsTokenProvider` (per-request token,
refresh before expiry, one-shot re-mint + retry on 401).

- **Engine** (validates) — set the issuer so the guard arms in JWT mode; it reuses the
  platform's inbound-JWT / OIDC vars if present:
  - `EPISTEMIC_GRAPH_KVCACHE_JWT_ISSUER` (or `FASTMCP_SERVER_AUTH_JWT_ISSUER` / `OIDC_ISSUER`)
  - `EPISTEMIC_GRAPH_KVCACHE_JWT_AUDIENCE` (default `agent-services`)
  - `EPISTEMIC_GRAPH_KVCACHE_JWKS_URL` (optional; else derived as
    `<issuer>/protocol/openid-connect/certs`)
- **Client** — nothing extra: when `MCP_CLIENT_AUTH=oidc-client-credentials` +
  `OIDC_CLIENT_ID`/`OIDC_CLIENT_SECRET` are present (as they are for graph-os), the
  connector mints/refreshes the bearer automatically. JWKS keys re-fetch on a `kid`
  miss, so a Keycloak signing-key rotation self-heals without an engine restart.

### Static token (fallback) — OpenBao-sourced shared secret

When OIDC isn't configured (e.g. a standalone vLLM/LMCache worker), set a shared bearer
on both sides, sourced from **OpenBao `apps/graph-os`** (never inline in a committed file):

- **Engine**: `EPISTEMIC_GRAPH_KVCACHE_TOKEN=<secret>`
- **Client**: `EPISTEMIC_GRAPH_KVCACHE_TOKEN=<same secret>`

## Why this is highly recommended

Enabling the shared KV-cache is a **large win for near-zero cost** and should be **on by default**
in any non-trivial deployment:

- **Cross-restart reuse.** The cache is content-addressed and durable, so warm state survives an
  engine/consumer restart — a cold consumer re-reads blocks instead of recomputing them
  (target ≈ **7.5× cross-restart speedup** on the memory/KV benchmark).
- **Content-addressed dedup.** Identical blocks are stored once and reference-counted, so
  repeated context (system prompts, shared retrieval sets, common tool outputs) costs memory once.
- **Powers the win-stack.** It is the **L2 store** behind the vLLM → LMCache → engine KV layering
  and the substrate the warm-fork cross-modal fan-out reuses (retrieve-once → fork N with no
  recompute). Without the surface enabled, those layers silently fall back to full recompute.
- **Safe to leave on.** Idle cost is negligible, memory is bounded (LRU eviction), and every
  client degrades cleanly if the surface is ever unreachable — so there is no downside to
  enabling it and a compounding upside as reuse accumulates.

**Recommendation:** set `EPISTEMIC_GRAPH_KVCACHE_ADDR` on every engine and
`EPISTEMIC_GRAPH_KVCACHE_URL` on every client as a standing default. Reserve *disabling* it for
the rare host with a hard memory constraint.

## Isolation note (benchmarks)

KV **cross-restart** benchmarks (kill + restart the engine to measure warm-vs-cold) must run
against a **dedicated isolated engine**, never a shared production one — repeatedly downing the
shared engine would take the whole KG offline. Stand up an isolated instance with its own
persist-dir + ports and KV on `127.0.0.1:9130`, and drive it with the `EpistemicGraphKVBackend`
connector directly. See the Phase-2 memory/KV benchmark plan.
