# Graph Backend Architecture

**The epistemic-graph engine is the ONE database — the authority and system of
record.** A single Rust engine does compute, in-memory cache, semantic/ontology
reasoning, AND durable persistence. All reads are served by the engine; all
writes commit to the engine and then fan out — losslessly and asynchronously
(durable outbox, replay-on-reconnect) — to optional **mirrors** for interop, BI,
external query, and disaster recovery. The mirror backends (PostgreSQL/pg-age,
Neo4j, FalkorDB, LadybugDB) each implement the same unified `GraphBackend`
abstract interface (Cypher query execution, vector search, node/edge CRUD, and
optional SPARQL support), so a mutation runs natively on every store.

The **default** is the engine alone (`GRAPH_BACKEND=epistemic_graph`, also
`memory`/`file` for snapshot modes) — zero external services. Turn on mirroring
with `GRAPH_BACKEND=fanout` and name a mirror set; the engine stays the read
authority and each durable mirror receives the replicated stream. PostgreSQL/
pg-age, Neo4j, and FalkorDB are first-class mirror targets (drivers install as
optional extras under `backends/contrib/`). There is **no tier vocabulary** —
it is the engine authority plus mirrors.

> **Verified parity (KG-2.7).** Node properties (declared / ad-hoc / nested),
> edge existence, **edge properties**, and vector search round-trip on **every**
> backend; the full cross-backend matrix and how to run it live are in
> [backend-parity-and-profile-testing](../guides/backend-parity-and-profile-testing.md).
> SPARQL is served **locally over any backend** via the OWL/RDF layer (see
> [owl_rdf_layer](owl_rdf_layer.md)).

> **PostgreSQL runs Apache AGE (`GRAPH_PG_AGE=1` / `backend_type=age`).** This
> executes **real openCypher** via AGE's `cypher()` function — `count(r)`,
> `RETURN … AS alias`, multi-hop and variable-length traversal all work natively —
> retiring the bounded regex Cypher→SQL transpiler (still the default when AGE is
> off). pgvector continues to back embeddings. Image: `docker/pg-age.compose.yml`.

## Architecture Overview

```mermaid
graph TB
    subgraph "IntelligenceGraphEngine"
        A["KG-2.0: query_cypher()"] --> B["KG-2.0: engine.execute()"]
        C["KG-2.0: add_node()"] --> B
        D["KG-2.0: link_nodes()"] --> B
        E["KG-2.3: search_hybrid()"] --> F["KG-2.3: engine.semantic_search()"]
        G["KG-2.0: load_subgraph()"] --> H["Rust GraphComputeEngine\n(compute + cache)"]
        QR["KG-2.7: QueryRouter"] --> B
        QR --> H
        QR --> F
    end

    subgraph AUTH["KG-2.7: epistemic-graph engine — THE authority (system of record)"]
        EG["Rust-native EpistemicGraph\ncompute + in-mem cache + semantic + durable persistence\n(serves ALL reads, acks ALL writes)"]
    end

    B --> EG
    F --> EG
    H --> EG

    EG -->|"GRAPH_BACKEND=fanout\n(durable outbox · async · lossless)"| OUTBOX["per-mirror outbox\n(replay-on-reconnect)"]

    subgraph MIRRORS["GRAPH_MIRROR_TARGETS — optional mirrors (interop · BI · external query · DR)"]
        M_PG["PostgreSQL / pg-age (AGE)\nopenCypher + pgvector + ParadeDB BM25"]
        M_N4["Neo4j (Bolt)"]
        M_FK["FalkorDB (Redis)"]
        M_LB["LadybugDB (SQLite + HNSW)"]
        M_SD["Stardog (SPARQL · urn:source:*)"]
    end

    OUTBOX --> M_PG
    OUTBOX --> M_N4
    OUTBOX --> M_FK
    OUTBOX --> M_LB
    OUTBOX --> M_SD

    EG -->|"SPARQL via OWL/RDF layer (local - any data)"| SPARQL["W3C SPARQL endpoint\n(core/owl_bridge · pyjena_fuseki)"]
```

## Durability — the engine is redb-authoritative BY DEFAULT (CONCEPT:AU-KG.backend.backend-modes)

**The engine is a durable source of truth out of the box — not a rebuildable
cache.** As of "THE FLIP" (CONCEPT:AU-KG.backend.backend-modes), a stock `epistemic-graph-server`
built with the standard `--features full` includes the **`redb`** store, so its
persist backend (`EPISTEMIC_GRAPH_PERSIST_BACKEND`) defaults to `redb` and runs in
**authoritative mode** whenever a persist dir (`GRAPH_SERVICE_PERSIST_DIR`) is
configured. An acked write survives a `kill -9`. The three guarantees that make
"authoritative" safe (engine-side, CONCEPT:EG-KG.backend.authoritative-dispatch/EG-KG.storage.read-through-seam-exercised):

- **Commit-before-ack** — a durable mutation is fsync-committed to redb *before*
  its response is acked, so an acked write is always on disk.
- **Read-through-safe eviction** — the per-graph node cap stays enforced (memory
  bounded) but a node is dropped from RAM only after redb confirms it on disk;
  evicted nodes are served back from redb on a RAM miss. No data loss.
- **Backpressure, not drop** — the redb writer blocks for capacity rather than
  shedding a write.

> **One-time migration on first authoritative boot.** When the engine first boots
> authoritative and finds legacy snapshot (`.mp`) / WAL files but an empty redb
> store, it runs a **one-time, idempotent, crash-safe `.mp`→redb migration**
> (read-old → write-new) before it binds its socket; the old files are left in
> place as a backstop. On a large KG (thousands of graphs / a multi-hundred-MB
> commons) this can take **minutes** — see the engine's
> [binary-promotion runbook](https://github.com/knuckles-team/epistemic-graph/blob/main/docs/deploy/binary-promotion.md)
> for the deploy-time health-start-period handling.

To opt back into the pre-flip **rebuildable-cache** model — the engine as a fast
in-memory cache over an external durable system-of-record (a `fanout` mirror) —
set `EPISTEMIC_GRAPH_PERSIST_BACKEND=snapshot` on the engine; the local `.mp`
snapshot + WAL then exist only for fast warm restart. (A build without the `redb`
feature is non-authoritative and boots clean with no warning.) This is an
**engine-side** setting (`epistemic-graph-server` env), independent of the
agent-utilities-side `GRAPH_BACKEND` mirror selection below.

## Derived stores route to the engine, NOT a local DB (engine-only, CONCEPT:AU-KG.backend.cache-lives-as–2.248)

Auxiliary stores that used to keep their own local SQLite/JSON file *next to* the
one engine authority now route through the **engine unconditionally** — there is
**no SQLite/JSON/file fallback**. Each resolves the engine-authority backend (the
OS-5.63 resolver auto-starts the pi-tier engine in prod; the AU-KG.memory.provides-real-ephemeral-one test fixture
provides a real ephemeral one) and raises a clear error if the engine is genuinely
unreachable. They share the engine-only helpers in
`knowledge_graph/backends/base.py` — `is_engine_authority_backend` /
`require_engine_authority_backend` — and persist via the engine node API
(`add_node` / `get_node_properties` / `nodes_by_label`, deterministic node ids), so
the reads/writes work against the real engine (the in-process `execute()` Cypher
subset can't do `WHERE … IN $list` / unscoped `MATCH`).

| Store | Engine surface (the only store) | Node id / scan | Concept |
|---|---|---|---|
| LLM card cache (`CardStore`) | `:CardCache` nodes | `cardcache:<ast_hash>` (keyed) | AU-KG.backend.cache-lives-as |
| Registry graph (`RegistryPipeline`) | engine graph nodes/edges via the active backend | `persist_to_ladybug=False` | AU-KG.compute.graph-builder |
| Time-series memory | engine `client.timeseries.*` (eg-tsdb, `series.redb`) | series ids | AU-KG.memory.time-series-lives-one |
| Write-back proposals (`ProposalQueue`) | `:WritebackProposal` nodes | `wbp:<target>:<seq>` + label scan | AU-KG.enrichment.proposals-live-as |
| Code-health baselines | `:CodeHealthBaseline` nodes | `codehealthbaseline:<repo>` (keyed) | AU-KG.maintenance.only-no-file-cache |

> The **ingestion delta manifest** (`DeltaManifest`, `:IngestManifest` /
> `kg_ingest_manifest.db`, CONCEPT:AU-KG.ingest.enterprise-source-extractor) is a SEPARATE, pre-existing store that
> *keeps* its zero-infra SQLite fallback via `is_durable_backend` — it is NOT part
> of this engine-only consolidation.

These enforce the platform's "one authority" discipline: auxiliary state is
queryable beside the graph, never scattered across host-local files, and never a
SQLite straggler. (Intentional non-store local files stay local by design: the
`STATE_DB_URI` state-store default, the mirror outbox, and RAM-only caches.)

## Engine authority vs. mirror comparison

The first column is **the engine** — the one authority that serves reads and acks
writes. The rest are **mirror targets**: durable, full-openCypher (or SPARQL)
stores that receive the replicated write stream for interop/BI/external-query/DR.

| Capability | epistemic-graph (THE authority) | LadybugDB (mirror) | PostgreSQL/pg-age (mirror) | Neo4j (mirror) | FalkorDB (mirror) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Role** | **authority · system of record** | mirror (extra) | mirror (extra) | mirror (extra) | mirror (extra) |
| Cypher Support | subset (id-anchored)¹ | Native (Kuzu) | **Native (AGE)** / transpiled | Native | Native |
| Node props (declared/ad-hoc/nested) | ✅ | ✅ (ad-hoc in `metadata`) | ✅ | ✅ | ✅ |
| **Edge properties** | ✅ | ✅ (JSON `r.properties`) | ✅ | ✅ | ✅ |
| Vector Search | ✅ | ✅ | ✅ pgvector | ✅ (`:Embeddable`) | ⚠️ AVX2 host² |
| SPARQL (via OWL/RDF layer) | ✅ local | ✅ local | ✅ local | ✅ local | ✅ local |
| Graph Traversal (multi-hop) | ✅ native compute¹ | ✅ | ✅ (AGE) | ✅ | ✅ |
| Connection Pooling | UDS client | File Lock | ✅ psycopg_pool | ✅ | — |
| Persistence | **durable (built-in, redb-authoritative)** | File | Server | Server | Redis |
| Zero Config | ✅ | ✅ | — | — | — |

¹ The engine is the authority and serves multi-hop traversal natively from its
own compute/cache over its durable store; `engine.execute` interprets an
operational id-anchored Cypher subset on the write path. ² FalkorDB vector search
is code-correct (Cypher `CREATE VECTOR INDEX` + `db.idx.vector.queryNodes`) but
the `falkordb` image SIGILLs on 768-dim vector ops on non-AVX2 host CPUs.

## PostgreSQL Mirror Deep Dive

PostgreSQL is the richest **mirror** target: it combines three PostgreSQL
extensions into a unified graph + vector + search store, so the engine's
replicated write stream lands in a queryable, BI-friendly, openCypher-capable
durable copy. (It is a mirror, not the authority — the engine remains the read
source of truth.)

### Internal extension architecture

```mermaid
graph LR
    subgraph "Storage (relational)"
        A["KG-2.0: Node Tables\n(Agent, Tool, Memory, ...)"] --- B["KG-2.0: kg_edges Table"]
    end

    subgraph "Graph Index (AGE)"
        C["KG-2.0: AGE traversal\n(graph.traverse)\n(graph.shortest_path)\n(graph.search)"]
    end

    subgraph "Search"
        D["KG-2.3: pgvector HNSW\n(Cosine Similarity)"]
        E["KG-2.3: ParadeDB BM25\n(Lexical Ranking)"]
    end

    A --> C
    B --> C
    A --> D
    A --> E
```

### Cypher Transpilation

The engine speaks Cypher; PostgreSQL speaks SQL. The `transpile()` function in
`backends/cypher_transpiler.py` handles the translation for all patterns the
engine generates:

| Engine Cypher | PostgreSQL SQL |
|---|---|
| `MATCH (n:Agent) WHERE n.id = $id RETURN n` | `SELECT * FROM "Agent" WHERE id = $1` |
| `CREATE (n:Tool {id: $id, name: $name})` | `INSERT INTO "Tool" (id, name) VALUES ($1, $2)` |
| `MATCH (s)-[r:PROVIDES]->(t) MERGE ...` | `INSERT INTO kg_edges ... ON CONFLICT DO UPDATE` |
| `MATCH (n) WHERE toLower(n.name) CONTAINS $q` | `SELECT * FROM ... WHERE LOWER(name) LIKE '%$1%'` |
| Path traversal `(n)-[*1..3]-(t)` | `graph.traverse(seed, max_depth:=3)` |

### Extension Dependencies

**Postgres is the richest mirror target here** (`backend: "age"`), and
to be a first-class graph + vector + search mirror it needs **three** extensions
installed — and, for AGE and pg_search, preloaded:

| Extension | Required | `shared_preload_libraries` | Purpose |
|---|:---:|:---:|---|
| **Apache AGE** (`age`) | **Yes** (for `backend: "age"`) | **yes** | **Native openCypher** — runs the query the engine emits unchanged, so a Postgres connection has `cypher_support = "full"` (parity with Neo4j/FalkorDB). AGE **must** be in `shared_preload_libraries` or it won't load. |
| **pgvector** (`vector`) | **Yes** | no | Embedding storage + HNSW cosine `semantic_search` |
| **ParadeDB** (`pg_search`) | **Yes** | yes | BM25 full-text scoring |
| pg_trgm | Optional | no | Trigram fuzzy text matching |
| pg-age | Optional (legacy) | no | CSR traversal for the regex-transpiler `backend: "postgresql"` path |

> **Why AGE, not just the transpiler.** Without AGE, a Postgres connection falls
> back to the bounded regex transpiler (`backend: "postgresql"`, `cypher_support
> = "subset"`) — fine as a single store, but it cannot serve the full query
> surface a fan-out mirror set shares (CONCEPT:AU-KG.backend.mirror-health-repair). With AGE, Postgres is a
> peer of Neo4j/FalkorDB and the **richest mirror target** (full openCypher).

**The curated image bundles all three.** `services/pg-age/` builds
`registry.arpa/pg-age` **FROM `paradedb/paradedb:latest` (PostgreSQL 18)** — which
already ships `vector` + `pg_search` — and adds **Apache AGE 1.7.0** (the release
that introduced PG18 support). The stack `command` sets
`shared_preload_libraries=pg_search,pg_cron,pg_stat_statements,age` and
`init-extensions.sql` runs `CREATE EXTENSION` for `vector`, `age`, and
`pg_search`. The **stock `paradedb/paradedb` image does NOT include AGE** — using
it directly leaves a Postgres connection on the `subset` transpiler path.

The mirror **gracefully degrades** when extensions are missing — CRUD and basic
search work with plain PostgreSQL; native Cypher requires AGE; vector search
requires pgvector; BM25 requires pg_search.

## Configuration

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GRAPH_BACKEND` | `epistemic_graph` | Engine mode: `epistemic_graph` (default — the engine authority alone, also `memory`/`file` snapshot modes) or `fanout` (engine authority + mirrors). The `tiered`/`GRAPH_BACKEND_L1`/`GRAPH_BACKEND_L2` scheme is **removed**. |
| `GRAPH_AUTHORITY` | `epistemic_graph` | Read source-of-truth under `fanout`; any named durable connection may be named instead, but the engine is the default |
| `GRAPH_MIRROR_TARGETS` | unset | JSON/list of mirror connection names (declared in `KG_CONNECTIONS`) that receive the fanned-out write stream; supersedes the removed `GRAPH_BACKEND_L2` |
| `GRAPH_DB_PATH` | `knowledge_graph.db` | File path for EpistemicGraph (`file` mode) / LadybugDB |
| `GRAPH_DB_URI` | — | Connection URI for Neo4j or PostgreSQL |
| `GRAPH_DB_HOST` | `localhost` | Host for FalkorDB |
| `GRAPH_DB_PORT` | `6379`/`7687` | Port for FalkorDB/Neo4j |
| `GRAPH_DB_USER` | `neo4j` | Username for Neo4j/PostgreSQL |
| `GRAPH_DB_PASSWORD` | `password` | Password for Neo4j/PostgreSQL |
| `GRAPH_DB_NAME` | `agent_graph` | Database/graph name |
| `GRAPH_POOL_MIN` | `2` | PostgreSQL pool minimum connections |
| `GRAPH_POOL_MAX` | `10` | PostgreSQL pool maximum connections |
| `GRAPH_PGGRAPH_SCHEMA` | `public` | Schema for pg-age table registration |
| `GRAPH_FUSEKI_URL` | `http://localhost:3030` | Jena/Apache Fuseki server URL |
| `GRAPH_FUSEKI_DATASET` | `agent_kg` | Fuseki dataset name |
| `GRAPH_FUSEKI_USER` / `GRAPH_FUSEKI_PASSWORD` | — | Optional Fuseki credentials |
| `KG_CONNECTIONS` | — | JSON list of named connections for the multi-connection registry (CONCEPT:AU-KG.backend.multi-connection-registry). See below. |

These are the **agent-utilities-side** (mirror-selection) variables. Engine
durability is configured on the **`epistemic-graph-server`** process itself
(CONCEPT:AU-KG.backend.backend-modes) and is independent of `GRAPH_BACKEND`:

| Variable (engine-side) | Default | Description |
|---|---|---|
| `GRAPH_SERVICE_PERSIST_DIR` | unset | Engine persist dir. Set ⇒ the engine is a durable source of truth; absent ⇒ in-memory only |
| `EPISTEMIC_GRAPH_PERSIST_BACKEND` | `redb` | `redb` = durable authoritative store (default — THE FLIP); `snapshot` = opt-in rebuildable-cache (`.mp` + WAL) |
| `EPISTEMIC_GRAPH_REDB_AUTHORITATIVE` | (auto) | Defaults ON when the redb backend is active; rarely set by hand |

## Multiple Connections at Once (CONCEPT:AU-KG.backend.multi-connection-registry)

The engine has always been vendor-agnostic, but historically only **one** backend
was live per process. The **named multi-connection registry** lets a deployment
keep several live connections side by side and run the *same* graph tools against
any one — or fan out to all — with the backend choice fully abstracted behind a
`target` parameter. No code is forked into a separate server: every existing
`graph_*` MCP tool and its REST twin gains this for free.

### Register connections

Declaratively, via `KG_CONNECTIONS` (each entry is `create_backend` kwargs plus a
`name`):

```bash
export KG_CONNECTIONS='[
  {"name": "prod-neo4j", "backend": "neo4j", "uri": "bolt://neo4j:7687", "user": "neo4j", "password": "..."},
  {"name": "team-falkor", "backend": "falkordb", "host": "falkor", "port": 6379},
  {"name": "pg-main", "backend": "age", "uri": "postgresql://agent:agent@pg:5432/agent_kg"}
]'
```

…or at runtime via `graph_configure`:

```
graph_configure(action="add_connection", config_key="pg-main",
                config_value='{"backend":"age","uri":"postgresql://agent:agent@pg:5432/agent_kg"}')
graph_configure(action="list_connections")          # per-connection health
graph_configure(action="set_default_connection", config_key="pg-main")
graph_configure(action="remove_connection", config_key="pg-main")
```

### Target a connection

Every `graph_query` / `graph_search` / `graph_write` (and the heavier
`graph_ingest`/`graph_analyze`/`graph_orchestrate`) accepts an optional `target`:

| `target` | Behaviour |
|---|---|
| omitted / `""` / `"default"` | the primary engine — **identical to legacy behaviour** (backward compatible) |
| `"pg-main"` | a single named connection; result shape unchanged |
| `"all"` or `"a,b"` or `["a","b"]` | **fan-out** — per-connection labeled results (`{"targets": {...}, "errors": {...}}`), partial success: one backend failing never aborts the others |

**Writes** only fan out on an *explicit* multi-target value (`"all"`/list) — the
default and a single named target stay single-write, so you never accidentally
triple-write.

### Portability: one query, every backend

For the same Cypher to run unchanged everywhere, the backend must speak native
openCypher. Each backend advertises a `cypher_support` tier:

| Backend | `cypher_support` | Notes |
|---|---|---|
| neo4j, falkordb | `full` | native Cypher |
| **Postgres via Apache AGE** (`backend: "age"`) | `full` | native openCypher (`count(r)`, aliases, multi-hop, `-[*1..2]->`, edge props) + pgvector |
| Postgres regex transpiler (`backend: "postgresql"`) | `subset` | only the bounded operational subset the engine emits; fallback when the AGE extension is absent |
| epistemic_graph (in-memory) | `subset` | AU-P0-2: label/property `MATCH` + a real `WHERE` predicate + aggregates/`DISTINCT` route to the engine's OWN native Cypher executor (`GraphComputeEngine.query_cypher`, its `eg-query` parser — still a bounded grammar, e.g. no comma-separated disjoint `MATCH` patterns or arbitrary function calls) instead of a client-side regex scan-and-eval; a rejected/unsupported shape raises (`CypherEngineError`/`NotImplementedError`) rather than silently returning `[]`. Two AU-specific shapes (the virtual `id` node-identity accessor; relationship-type traversal/merge, keyed by `rel_type` not the engine's `relationship`/`type`) stay on typed engine methods because native routing would give silently-wrong results. |

**Register Postgres connections as `age`** (not `postgresql`) when you want one
query to run unchanged across neo4j + falkordb + postgres. `list_connections`
reports each connection's `cypher_support` so fan-out callers can tell which
backends can serve a full query.

## Connection roles + live config (CONCEPT:AU-KG.backend.connection-registry)

Every registered connection carries a **role**, so you can safely attach an existing
third-party graph as a data source — not just for mirroring:

| role | meaning | `target=` writes |
|---|---|---|
| `read` (default) | external **data source** — query/profile/imprint only | rejected |
| `read_write` | full query + write target | allowed |
| `mirror` | receives fan-out replication of *our* KG | rejected (written only via the outbox) |

```jsonc
// KG_CONNECTIONS entry — role + a secret reference (kept out of config.json)
{"name":"prod-neo4j","backend":"neo4j","uri":"bolt://neo4j.arpa:7687",
 "user":"neo4j","password":"vault://agents/kg/neo4j#password","role":"read"}
```

- **Credentials** may be literals *or* `vault://…` / `env://VAR` refs, resolved at
  connect; `list_connections` redacts them either way.
- **The mirror set is derived from `role=mirror`** connections (`GRAPH_MIRROR_TARGETS`
  stays an optional override) — one registry drives both query targets and mirrors.
- **Durable + live:** `graph_configure add_connection/remove_connection` persists the
  list to `config.json` (survives restart). `profile_connection` / `imprint_connection`
  introspect a foreign graph's schema and write a self-describing
  `ExternalGraphReference` catalog node, mapping its labels onto our ontology.
- **Generic live config:** `graph_configure get_config|set_config|list_config`
  read/update/list **any** config option (validated against `config_reference`),
  persisted to config.json and applied live; engine-rebuild settings come back with
  `restart_required: true`. The doctor's `graph_connections` check reports each
  connection's role + flags stalled mirrors. All of this is exposed on **MCP and the
  API gateway** (`POST /api/graph/configure`).

## Mirror every write to N stores at once (CONCEPT:AU-KG.backend.mirror-health-repair)

Where KG-2.63 lets you *target* several connections per call, **fan-out** makes
mirroring the **default** for every write: one configurable **authority** store
serves reads and acks writes, and each mutation is replicated — losslessly and
asynchronously — to any set of durable backends. Turn it on with
`GRAPH_BACKEND=fanout`; the zero-infra default is unchanged (it is only built
when you configure a mirror set).

```bash
# Authority = epistemic-graph engine (fast in-mem reads + durable); mirror to Postgres-AGE,
# Neo4j and FalkorDB. The mirror set names entries declared in KG_CONNECTIONS.
export GRAPH_BACKEND=fanout
export GRAPH_AUTHORITY=epistemic_graph
export GRAPH_MIRROR_TARGETS='["pg-age","prod-neo4j","team-falkor"]'
export KG_CONNECTIONS='[
  {"name":"pg-age","backend":"age","uri":"postgresql://u:p@pg.arpa:5432/agent_kg"},
  {"name":"prod-neo4j","backend":"neo4j","uri":"bolt://neo4j.arpa:7687","user":"neo4j","password":"…"},
  {"name":"team-falkor","backend":"falkordb","host":"falkordb.arpa","port":6379}
]'
```

You may also set the authority to any durable store (e.g. `GRAPH_AUTHORITY=pg-age`)
— whichever connection you name becomes the read source-of-truth, and the rest
are mirrors.

### How it stays lossless

* **Authority commits synchronously; mirrors apply async.** A write returns once
  the authority commits **and** the mutation is durably appended to each mirror's
  outbox. Reads (served from the authority) are always consistent; mirrors are
  eventually-consistent (seconds of lag).
* **Durable per-mirror outbox.** `backends/outbox.py` is a sqlite/WAL append log
  (zero external infra). A per-mirror drainer thread applies entries in order and
  advances a persisted cursor, so a mirror that is **offline or slow keeps its
  unapplied tail and replays from its cursor on reconnect / restart** — a
  transient outage never drops a write. (Unlike an in-memory write-behind queue,
  the outbox survives a crash.)
* **One drainer per mirror = natural single-writer.** A file-locked store
  (LadybugDB/Kuzu) is serialised for free — it is simply the slowest mirror.
* **Reconcile backstop.** `graph_configure(action="reconcile")` runs a full
  authority→mirror re-sync and reports exact remaining drift — the repair for the
  only un-mirrorable window (a crash between the authority commit and the outbox
  append) and for a mirror whose outbox tail was lost.

### Operate it

| Surface | Call | What it does |
|---|---|---|
| MCP | `graph_configure(action="mirror_status")` | Per-mirror replication health: `lag`, `writes`, `failures`, `stalled`, `last_error`, plus `outbox_depth`. Alarm on `lag`/`stalled`. |
| MCP | `graph_configure(action="reconcile", config_key="<mirror>")` | Full drift-repair for one mirror (empty `config_key` = all). |
| REST | `POST /graph/configure {"action":"mirror_status"}` | Same core, REST twin. |

**Portability:** use full-openCypher mirrors (Postgres-AGE / Neo4j / FalkorDB) so
the same mutation runs unchanged on every store — see the `cypher_support` table
above. **LadybugDB** can be a 4th local-write mirror — config only: add a
`kg_connections` entry `{"name":"local-ladybug","backend":"ladybug","db_path":"<data_dir>/mirror_ladybug.db"}`
and list it in `GRAPH_MIRROR_TARGETS`. Its single-writer file lock is serialised
by its one drainer thread; ad-hoc props fold into the `metadata` JSON column and
edge props into the `properties` JSON column (durably stored, conformance-verified).

### Native cross-backend migration (CONCEPT:AU-KG.backend.mirror-health-repair)

`knowledge_graph/migration.py` `copy_graph(source, target)` copies every node +
edge (+ embeddings) from any source (the engine authority, or a full-cypher mirror
read via `MATCH`) into any target backend. The **write** goes through the
engine's proven, dialect-aware MERGE upserts (`IntelligenceGraphEngine._upsert_node`
/ `_upsert_edge`) — so each backend gets a *correct native write*, never one
backend's raw cypher forwarded to all. It is the right layer for interchangeable
storage:

* **MERGE-on-`{id}`** → idempotent + resumable (re-running converges, never dupes);
* per-backend dialect handled once: nested values JSON-encoded for map-rejecting
  drivers (Neo4j/FalkorDB), ad-hoc keys → `metadata` JSON and edge props →
  `properties` JSON for strict-schema Kuzu/Ladybug, transpiler-safe for plain
  Postgres;
* returns exact post-condition drift (`nodes_missing` / `edges_missing`).

This is what **backfills a freshly-added mirror** (`graph_configure(action="reconcile")`
and `reconcile_to_durable` both delegate to `copy_graph`) and what migrates
data between any two backends (e.g. Neo4j → FalkorDB). It replaced the old reconcile
that reconstructed `CREATE (n:Label {`k`: $k})` cypher — fragile on native-cypher
backends (double-escaped reserved keys) and edge-lossy. Parity is proven by
`tests/integration/backends/test_cross_backend_parity.py` (`pytest -m live`): the same
source graph copied into Postgres + Neo4j + FalkorDB + LadybugDB lands as identical
counts.

### Quick Start: PostgreSQL mirror

```bash
# 1. Start the database
docker compose -f docker/pg-age.compose.yml up -d

# 2. Mirror the engine authority to Postgres-AGE
export GRAPH_BACKEND=fanout
export GRAPH_MIRROR_TARGETS='["pg-age"]'
export KG_CONNECTIONS='[{"name":"pg-age","backend":"age","uri":"postgresql://agent:agent@localhost:5433/agent_kg"}]'

# 3. Run the graph-os MCP server — engine serves reads, Postgres mirrors writes
graph-os
```

## Engine stability backstops — surviving the `__commons__` firehose

The engine is one process behind a MessagePack socket holding a multi-tenant
graph (`__commons__` carries 166K+ nodes, each with a 1024-dim embedding). Three
failure modes were hardened after a stress run wedged the engine; each backstop
fixes a specific one.

```mermaid
flowchart TB
    subgraph CLIENT["agent-utilities (engine client)"]
        BR["iter_nodes_by_types() / get_node_data()<br/>core/bounded_read.py — AU-KG.ingest.never-scan-whole-graph"]
        BRK["BreakerClientProxy._guard()<br/>core/engine_breaker.py — AU-KG.compute.single-dropped-connection"]
    end
    subgraph ENGINE["epistemic-graph (Rust)"]
        GUARD["GetNodes handler:<br/>oversize_dump_error(node_count, cap)<br/>handlers/graph_ops.rs — EG-KG.ingest.resets-socket-so-assimilation"]
        CAP["max_response_nodes()<br/>EPISTEMIC_GRAPH_MAX_RESPONSE_NODES<br/>default 50000 (0 = off) — state.rs"]
        LABEL["GetNodesByLabel (bounded, NOT capped)"]
        LOCK["per-graph topology write lock"]
        COAL["write_coalescer.apply_batch()<br/>N writes to 1 core.txn() — EG-KG.sharding.per-graph-write-coalescer"]
    end

    BR -->|"per-label fetch, O(#type) not O(graph)"| LABEL
    BR -. "AVOIDS unbounded nodes(data=True)<br/>166K x 1024-dim, 1GB frame, ConnectionReset" .-> GUARD
    GUARD --> CAP
    GUARD -->|"count > cap"| TOOBIG["RESULT_TOO_LARGE to ResultTooLargeError"]
    BRK -->|"ConnectionReset/BrokenPipe = transient"| RETRY["retry x2, 0.25s backoff<br/>rides client._reconnect, breaker NOT tripped"]
    COAL --> LOCK
```

### The three backstops

- **Bounded reads (AU-KG.ingest.never-scan-whole-graph, `core/bounded_read.py`).** A whole-graph
  `graph.nodes(data=True)` materializes every node into one MessagePack frame — a
  gigabyte payload that overloads and resets the connection. `iter_nodes_by_types`
  iterates by **type** through the engine-side bounded label fetch
  (`get_nodes_by_label`, KG-2.51), and **trusts** an empty bounded result rather
  than falling back to a full scan; `get_node_data` fetches one node by id. A
  type-filtered reader becomes O(#type), not O(graph).
- **Adaptive transient retry (AU-KG.compute.single-dropped-connection, `core/engine_breaker.py`).** A single
  dropped connection mid-op is transient — the client transparently reconnects on
  its next call. `_guard` now retries the op a bounded `_MAX_TRANSIENT_RETRIES`
  (=2) with exponential backoff and records a `"retry"` outcome **without**
  counting it against the circuit breaker, so a brief blip self-heals instead of
  cascading N callers into a tripped breaker.
- **Engine response guard (EG-KG.ingest.resets-socket-so-assimilation, epistemic-graph).** The engine itself caps
  the unbounded full-graph dump: `oversize_dump_error` checks the cheap
  `node_count()` against `max_response_nodes()` (env
  `EPISTEMIC_GRAPH_MAX_RESPONSE_NODES`, default **50000**, `0` disables) **before**
  materializing, and returns `RESULT_TOO_LARGE:` (surfaced client-side as
  `ResultTooLargeError`). Only the `GetNodes` dump is capped — `GetNodesByLabel`
  and per-id reads stay bounded and uncapped.

### The H1 finding — write-lock starvation (EG-011)

A profiling/stress run confirmed **H1**: under sustained ingestion every write
funnels through the **`__commons__` per-graph topology write lock**, so it
serializes and starves reads. Measured: a `semantic_search` that is **~0.02s
idle** climbs to **~14s** under the ingestion firehose (and a single `nodes.add`
to `__commons__` took ~13s), with the engine pinning roughly one core. Two
histograms (CONCEPT:EG-KG.compute.parse-resolve-span, `epistemic_graph/src/metrics.rs`) make the gap
observable:

| metric (labelled `{graph}`) | measures |
|---|---|
| `epistemic_graph_write_lock_wait_seconds` | enqueue to lock acquired (the **starvation** cost) |
| `epistemic_graph_write_lock_hold_seconds` | how long each coalesced batch holds the lock (the **contention** cost) |

```mermaid
flowchart LR
    ING["ingestion firehose<br/>(writes to __commons__)"] --> WLOCK{{"per-graph write lock"}}
    READ["semantic_search<br/>0.02s idle to ~14s under load"] --> WLOCK
    WLOCK -. "wait_seconds (starvation)" .-> HW["EG-011 histograms"]
    WLOCK -. "hold_seconds (contention)" .-> HW
    WLOCK --> FIX1["write_coalescer: N writes to 1 txn/batch<br/>(EG-KG.sharding.per-graph-write-coalescer, ~57x fewer acquisitions)"]
    WLOCK --> FIX2["split control plane onto __control__ graph<br/>(distinct GraphCore = distinct lock)"]
```

The fixes attack both axes: the **write-coalescer (EG-KG.sharding.per-graph-write-coalescer)** turns many
concurrent single-op writes into one lock acquisition per batch (~57× fewer
acquisitions in the engine benchmark), and routing the scheduler/control plane
onto a separate **`__control__`** engine graph gives it its own lock so ingestion
on `__commons__` can never starve it (distinct graphs already isolate — each
`GraphCore` has its own lock).

## Implementing a New Backend

1. Inherit from `GraphBackend` in `backends/base.py`
2. Implement all abstract methods: `execute()`, `execute_batch()`, `create_schema()`,
   `add_embedding()`, `semantic_search()`, `prune()`, `close()`
3. Optionally override `supports_sparql` and `execute_sparql()` for SPARQL support
4. Register in the `create_backend()` factory in `backends/__init__.py`
5. Add optional dependency group to `pyproject.toml`
6. Add integration tests

## Related

- [KG as Bidirectional ETL Hub](kg_etl_hub.md) — how these backends serve as ETL load
  targets (Stardog SPARQL data backend, fan-out mirrors) under the unified `graph_etl`
  interface, plus the one ingestion contract and data lineage.
- [OWL/RDF Layer](owl_rdf_layer.md) — local SPARQL + reasoning over any backend.
