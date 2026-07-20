# Recipe — Tiny (all-local, zero-infra)

> Ladder position: this recipe is **rung (a) — Zero-infra dev** of the
> [supported deployment configurations](../guides/deployment-configurations.md#rung-a-zero-infra-dev)
> guide, which lists every default this tier relies on.

For a laptop, a dev box, or an edge node. **No databases, no external services,
no container stack.** The knowledge graph runs entirely on this machine: the
epistemic-graph engine *is* the one database — compute, in-memory cache,
semantic/ontology reasoning, **and** durable persistence in a single engine. It
is packaged with the installation and supervised by GraphOS as an
**out-of-process child** over a private local transport. The reference-counted
engine is shared by local clients and stops after the last client has been idle
for the configured interval. There are **no mirror databases** (Postgres/pg-age, Neo4j,
FalkorDB, Ladybug are optional write-only fan-out targets you do not configure
here). The only thing you need is a model provider configured with a runtime
credential reference or an approved local inference endpoint.

## What runs

| Component | How |
|---|---|
| agent-utilities | GraphOS parent process from the approved pip/uv installation |
| Knowledge graph | the **epistemic-graph engine authority** — one packaged, supervised, out-of-process engine with XDG-discovered durable storage and no mirrors |
| Engine lifecycle | auto-spun-up as a **shared local daemon, reference-counted** (CONCEPT:AU-OS.deployment.engine-resolver-auto-provision): the ONE resolver autostarts a detached engine on first use and it self-stops ~60s after the last client disconnects. Set `engine_lifecycle=persistent` for a **long-living** engine that never auto-stops (warm, like a local service). A remote engine (enterprise) is inherently persistent. |
| **OWL/RDF + reasoning** | **on by default** — local OWL-RL inference (epistemic-graph) over the LPG, no external triplestore |
| **SPARQL** | **local endpoint** at `GET/POST {gateway}/api/sparql` (rdflib materialization + engine `GetRdf` fast path) — zero external deps |
| graph-os MCP | optional, `graph-os` (stdio) |
| External services | **none** |

There is **no 5-container requirement** for tiny — the supervised engine is the
only additional process. OWL/RDF is a **core, always-on** layer here, not an
enterprise add-on: the tiny profile consumes the bundled ontologies, infers new
relationships, and serves SPARQL **locally** with no Fuseki/Stardog.
(Fuseki/Stardog are an *optional* enterprise scale-out, configured only in the
[enterprise recipe](enterprise.md).)

## Steps

```bash
pip install "agent-utilities[serving]"
setup-config generate --profile tiny
```

Keep `GRAPH_SERVICE_ENDPOINTS`, `KG_AUTH_TOKEN_REF`, and `KG_IDENTITY_OAUTH2`
unset. The resulting tiny packaged-local GraphOS stdio boundary creates and
validates a neutral short-lived JWT with an in-memory key as a one-time proof,
then destroys the key and token before returning a process-lifetime session. It
persists no personal identity, host name, endpoint, filesystem path, credential,
or proof material. Verify that boundary before launch:

```bash
agent-utilities-doctor --only graph_identity auth
```

## AgentConfig (reference-only)

```json
{
  "DEPLOYMENT_PROFILE": "tiny",
  "ENGINE_LIFECYCLE": "refcounted",
  "ENGINE_IDLE_SHUTDOWN_SECS": 60,
  "CHAT_MODELS": [
    {
      "id": "chat-model",
      "provider": "openai",
      "base_url": "https://model.example.invalid/v1",
      "api_key_ref": "secret://models/chat-api-key",
      "tools_enabled": true,
      "can_route": true,
      "can_kg": true
    }
  ]
}
```

Absence of `GRAPH_SERVICE_ENDPOINTS` selects the packaged local engine. Any
configured endpoint list is connect-only and never receives a local substitute.
Use `ENGINE_LIFECYCLE=persistent` when this host should keep the engine warm after
the last client disconnects. Resolved credentials, endpoints, certificate paths,
identities, and machine paths must not be copied into source control or reports.
Every network transport, non-tiny profile, explicit engine endpoint, and other
entry point requires exactly one external process identity and its JWT validation
policy. A configured-but-invalid source fails closed and never selects the local
authority.

## Verify

Register `graph-os` natively in Codex:

```bash
setup-config codex
# Equivalent: codex mcp add graph-os -- graph-os --transport stdio
```

The IDE launches exactly the stdio boundary certified above. Standalone library,
daemon, REST, and network MCP processes are outside that exception and require
external process authority. Run `agent-utilities-doctor` after launch; `--live`
additionally performs bounded Langfuse and native optimizer probes when those
capabilities are configured.

## When to graduate

The packaged engine is already durable across restarts of *its own process*. The
moment you want the engine to run independently of any one agent process, or to
share it across containers/hosts, move to
[Single-node prod](single-node-prod.md) — there the same engine runs as its own
container; [enterprise](enterprise.md) points everything at a shared/remote
engine via `GRAPH_SERVICE_ENDPOINTS` and adds optional mirrors. The full
progression (auth, multi-host scale-out, autonomy) is the
[deployment configurations ladder](../guides/deployment-configurations.md).
