# Sovereign / Self-Hosted Epistemic Memory (CONCEPT:AU-OS.deployment.engine-resolver-auto-provision)

**Persona:** self-hoster, homelab operator, or anyone who wants a private
"second brain" knowledge graph that runs on their own hardware, keeps their
data on their own disk, and doesn't phone home to a vendor's vector-DB SaaS.

This guide packages three things that already exist but were scattered
across separate docs: the zero-infra `tiny` deployment profile, the measured
benchmark showing the epistemic engine beats a conventional stitched
vector-DB stack on the same workload, and the concrete steps to run the
whole substrate with the network cut — with an honest note on what is and
isn't a first-class "air-gapped mode" today.

---

## 1. The `tiny` profile — zero external services

`tiny` (rung (a) of the [deployment ladder](deployment-configurations.md#rung-a-zero-infra-dev),
recipe: [`docs/recipes/tiny.md`](../recipes/tiny.md)) is a real, not aspirational,
zero-infrastructure deployment. There is no Docker requirement, no external
database, no container stack. The **epistemic-graph engine is the one
database** — it does compute, in-memory cache, semantic/ontology reasoning,
**and** durable persistence in a single self-contained process:

| Component | Where it runs |
|---|---|
| agent-utilities | in-process, `pip`/`uv` install |
| Knowledge graph | one embedded epistemic-graph engine, durable on local disk (`--persist-dir`), no mirror databases |
| OWL/RDF + reasoning | **on by default** — local OWL-RL inference over the graph, no external triplestore |
| SPARQL | **local endpoint**, `GET/POST {gateway}/api/sparql` — rdflib materialization + engine `GetTriples` fast path, zero external deps |
| Engine lifecycle | auto-spun-up, reference-counted local daemon (`ENGINE_MODE=auto`, `ENGINE_LIFECYCLE=refcounted` by default — self-stops ~60s idle; set `ENGINE_LIFECYCLE=persistent` to keep it warm) |
| External services | **none required** |

The engine binary runs on Raspberry Pi 4+ (`epistemic-graph/README.md`) — a
genuine edge/sovereign target, not a marketing claim about a cloud SKU.

The **only** external dependency in a `tiny` deployment is a model provider —
either a hosted API key, or a local OpenAI-compatible endpoint
(`OPENAI_BASE_URL=http://localhost:8000/v1` pointed at your own vLLM/Ollama).
Everything graph-side — storage, reasoning, retrieval — is local.

```dotenv
# tiny .env — this is genuinely the whole thing
EPISTEMIC_GRAPH_AUTOSTART=1
OPENAI_BASE_URL=http://localhost:8000/v1      # your own vLLM/Ollama, not a cloud key
```

```bash
git clone https://github.com/Knuckles-Team/agent-utilities && cd agent-utilities
./scripts/bootstrap.sh   # venv + `.[all]` + .env + smoke test
```

Verify:

```python
from agent_utilities import create_agent
agent, _ = create_agent(name="assistant", skill_types=["universal", "graphs"])
print(agent.run_sync("Add a node 'hello' of type Greeting, then count nodes.").output)
```

## 2. Why local beats a stitched vector-RAG stack — the measured numbers

This isn't a "trust us, local is fine" claim. The Phase-2 agent-memory
benchmark (`epistemic-graph/docs/benchmarks.md`, "Phase-2: agent-memory +
KV-cache benchmark") measured the unified engine — **one in-transaction
cross-modal plan** (semantic + lexical + graph + OWL + AS-OF + RRF), warm-fork
context reuse, a durable KV cold-tier, incremental indexing — head-to-head
against a **conventional stitched stack** (separate vector store + BM25 +
app-level RRF fusion, no KV cache, no warm-fork, full-rebuild indexing) on
the *same* deterministic agent-memory workload. All numbers below are
**measured**, on a dedicated isolated bench engine (`--features full`,
redb-authoritative); full write-up + reproduction steps in the workspace
report `reports/phase2-memory-kv-benchmark-results.md`.

| Category | epistemic-graph | Stitched vector-RAG baseline | Result |
|---|---|---|---|
| recall@10 | **1.000** (indexed ANN) | 1.000 (exhaustive scan) | tie — same recall, but ours keeps it *with* an index, not a full scan |
| cross-modal retrieval p50 (N=2000) | **7.3 ms** | 26.3 ms | **~3.6× faster** |
| retrieval calls per warm-fork fan-out (N=8/32/128) | **`retrieval_calls == 1`** (100% of branches) | `retrieval_calls == N` | **128× fewer retrievals** at N=128 |
| write → read-fresh | **25.7 ms p50** (incremental, durable) | 19–69 ms (full rebuild) | ours |
| throughput | **799 qps** | N-scan per query | ours |
| KV cross-restart | **100% page survival, ~24 µs/page GET** — over 300× vs recompute | none (full recompute on restart) | ours |

The headline for a self-hoster: you get the retrieval quality of a vector
store (recall@10 = 1.000) with none of the operational surface area of one —
no separate vector DB, no BM25 index to keep in sync, no app-level fusion
code — because it's one engine doing a single in-transaction plan, and it's
**faster**, not just simpler.

Reproduce it yourself:

```bash
# in epistemic-graph
cargo bench -p eg-plan --features "query,owl,text,timeseries" --bench hybrid_queries
python3 scripts/bench_gate.py   # p50 + recall@k gate
```

## 3. Homelab integration

The fleet ships connector agents for common self-hosted services — real MCP
servers, not stubs — that map naturally onto a homelab: `home-assistant-agent`,
`jellyfin-mcp`, `mealie-mcp`, `nextcloud-agent` (all under
`agent-packages/agents/`). The `agent-os-genesis` skill's reference material
covers homelab-specific patterns beyond the app layer — Keycloak realm
consolidation, FreshRSS+SSO, Arr-stack VPN hardening — for operators running
the whole stack, not just the KG.

## 4. Air-gap / offline operating mode

**Honest status first.** There is **no first-class, named "air-gapped mode"
flag** in agent-utilities today — no `AIRGAP=1` switch, and no CI job that
boots the platform with network egress blocked and asserts it still works.
Grepping the codebase for `airgap`/`air-gap`/`offline` finds only
test-fixture and connector-injection use of "offline" (fetch-function
injection for unit tests), not a deployment mode. **This is a real gap, not
a documented feature** — flagged here rather than glossed over, per the
5-intersections assessment (`reports/surpass-6mo/04-five-intersections.md`
§3) that first identified it.

What *is* true, and is why an offline configuration is realistic rather than
aspirational: the `tiny` profile's design makes "nothing requires the
network by default" the natural state, not a special mode you opt into.

### What's local (no network needed once configured)

| Component | Local? |
|---|---|
| Graph storage (durable persist) | **Yes** — local redb store on disk, `--persist-dir` / `GRAPH_SERVICE_PERSIST_DIR` |
| Compute engine | **Yes** — the epistemic-graph binary runs entirely on-host |
| OWL/RDF reasoning | **Yes** — local OWL-RL inference, bundled ontologies |
| SPARQL endpoint | **Yes** — `{gateway}/api/sparql`, served locally (rdflib + engine `GetTriples`) |
| Vector/semantic index | **Yes** — the engine's own ANN index (`eg-ann`), no external vector service |
| Blob/lake tier | **Yes**, when configured — local CAS + Parquet segments, no S3 required unless you opt into the `blob-s3` feature |
| MCP tool surface | **Yes** — `graph-os` runs over stdio or a local HTTP port, no external gateway required |

### What must be provided externally

| Component | Why it's the one dependency |
|---|---|
| A chat/embedding model endpoint | The framework needs *some* LLM to run agent reasoning and to compute embeddings. This can be **fully local** — point `OPENAI_BASE_URL` at your own vLLM/Ollama instance — or a hosted API key. It is the only component in the `tiny` profile that is not local by default; making it local is a configuration choice, not a code change. |

### Concrete offline configuration path (doc-only — not a verified/tested mode)

1. **Stand up a local model endpoint.** vLLM or Ollama on the same host or
   LAN, serving an OpenAI-compatible API.
2. **Point the framework at it, with no hosted-provider key set:**

   ```dotenv
   EPISTEMIC_GRAPH_AUTOSTART=1
   OPENAI_BASE_URL=http://localhost:8000/v1
   # Do not set OPENAI_API_KEY / ANTHROPIC_API_KEY / any hosted provider key —
   # leaving them unset means there is nothing to reach for outbound
   ```

3. **Configure a durable persist dir** so the engine is a real store, not an
   in-memory cache that loses everything on restart:

   ```dotenv
   GRAPH_SERVICE_PERSIST_DIR=/var/lib/agent-utilities/engine
   ```

4. **Do not configure any of the network-reaching optional integrations** —
   leave `LANGFUSE_HOST`, `KAFKA_BOOTSTRAP_SERVERS`,
   `SECRETS_VAULT_URL`, `AUTH_JWT_JWKS_URI`, and the various source
   connectors (LeanIX, GitLab, ServiceNow, …) unset. None of these are
   required by `tiny`; each one you configure is a deliberately added
   external dependency, not an implicit one.
5. **Verify locally** — the same smoke test as the `tiny` recipe, run with
   the host's network interface down (or egress firewalled) to confirm
   nothing in the startup or query path reaches out:

   ```python
   from agent_utilities.knowledge_graph.facade import KnowledgeGraph
   kg = KnowledgeGraph()
   print(kg.query("MATCH (n) RETURN count(n) AS n"))
   ```

### What's genuinely missing (tracked, not shipped)

- **No named air-gap mode/flag.** The above is a *configuration path*, not a
  mode the platform knows about or validates against. Nothing currently
  detects "you're air-gapped" or changes behavior accordingly (e.g. failing
  fast with a clear message if a component unexpectedly tries to reach the
  network).
- **No air-gap CI/smoke gate.** There is no automated job that boots `tiny`
  with egress blocked and asserts full KG/reasoning/SPARQL functionality.
  Until one exists, the path above should be treated as "should work, based
  on reading the code" rather than "verified to work" — validate it in your
  own environment before relying on it for a genuinely offline deployment.

If you need this hardened — a real `AIRGAP=1` startup assertion, or a CI job
that proves it — that is open follow-up work, not something to claim exists
today.

## Where to go next

- [Recipe — Tiny (all-local)](../recipes/tiny.md) — the full rung-(a) recipe
  this guide's §1 summarizes.
- [Supported Deployment Configurations](deployment-configurations.md) — the
  full ladder, including how to graduate from `tiny` to a durable
  single-node deployment without adding a mirror database.
- [Epistemic Audit & Compliance](epistemic-audit-compliance.md) — the sibling
  persona guide, for when your self-hosted deployment also needs to answer
  "prove this wasn't tampered with."
