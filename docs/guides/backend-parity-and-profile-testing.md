# Backend parity & deployment-profile testing

Agent-utilities ships the *same* code to a Raspberry Pi 4+ and to an enterprise
cluster. The **epistemic-graph engine is the one database** in every case; the
optional mirrors (Postgres/pg-age, Neo4j, FalkorDB, Ladybug) must each accept the
engine's async write fan-out faithfully. Two test layers keep that promise honest:

1. **Mirror conformance** — one assertion body run against **every** supported
   mirror (`tests/integration/backends/`).
2. **Deployment profiles** — two end-to-end topologies: a zero-external-service
   "tiny" profile (the mandatory full engine only) and the "enterprise" profile,
   the same engine + mirrors
   (`tests/integration/profiles/`).

Both stand real services up in **throwaway containers** via
[`testcontainers`](https://testcontainers-python.readthedocs.io/) — on random
free ports, torn down deterministically — so a run is hermetic and can be
intentionally broken without touching anything you care about.

## Install

```bash
pip install 'agent-utilities[test-backends]'   # drivers + testcontainers
```

Requires a reachable Docker daemon for the live matrix. The zero-infra case
(the packaged, supervised out-of-process epistemic-graph engine, no mirrors)
needs neither Docker nor
`testcontainers`.

## What runs when

| Selection | What runs | Needs Docker? |
|---|---|---|
| `pytest` (default, `-m "not live"`) | tiny-profile (mandatory engine, no external services) e2e; mirror conformance for `ladybug` | no |
| `pytest -m live` | full mirror matrix (pg-age/Neo4j/FalkorDB) + Fuseki SPARQL + enterprise-profile e2e | yes |

The default PR suite therefore continuously enforces the **Pi-4+ engine-only
profile contract** (including a cold-import footprint guard) without requiring Docker, and
the heavyweight cross-backend/cluster checks run under `-m live` and on the
nightly job.

```bash
# Engine-only profile (fast, no Docker)
pytest tests/integration/profiles/test_profile_tiny_zero_dep.py -v

# zero-infra conformance params (no Docker)
pytest tests/integration/backends/test_backend_conformance.py -v

# full live matrix + enterprise profile (Docker required)
pytest tests/integration/backends -m live -v
pytest tests/integration/profiles/test_profile_enterprise_full.py -m live -v
```

The KG/engine assertions skip when no epistemic-graph endpoint is available
(`GRAPH_SERVICE_ENDPOINTS` unset and no packaged engine, e.g. a polyrepo CI without the Rust source); the
footprint guard always runs.

## The two profiles

**Tiny (Raspberry Pi 4+).** The mandatory full engine alone (the one self-contained database,
no mirrors), `OWL_BACKEND=owlready2`, SQLite task
queue, inline dispatch, no `GRAPH_DB_CONNECTION_PROFILE_REF`/`STATE_DB_URI`/Kafka. The test boots the
gateway REST surface in-process and asserts write→query works and the local OWL
reasoner runs — with **zero containers** — plus a subprocess cold-import check
that no mirror/external-service driver
(`aiokafka`/`psycopg`/`neo4j`/`falkordb`/`pystardog`/`confluent_kafka`) leaked
into the footprint.

**Enterprise.** Engine + throwaway pg-age projection + Kafka + Fuseki. Asserts
the three integration seams: writes committed to
the engine fan out and land in the pg-age **mirror**, surviving a reconnect; the
task queue resolves to Kafka and a put→consume→ack round-trips; the ontology
publishes to Fuseki and is queryable over SPARQL.

## Parity status (the 100%-parity program)

A live full-matrix probe (write via the engine, then read each mirror via
`backend.execute`, all mirrors running) drove a phased program that closed the
gaps. The `epistemic_graph` column is the authority; the rest are mirrors.
**Verified current state:**

| Capability | epistemic_graph (authority) | ladybug | pg-age (AGE) | neo4j | falkordb |
|---|---|---|---|---|---|
| node props (declared/ad-hoc/nested) | ✅ | ✅ (ad-hoc in `metadata`) | ✅ | ✅ | ✅ |
| edge existence | ✅ | ✅ | ✅ | ✅ | ✅ |
| edge properties | ✅ | ✅ (JSON `r.properties`) | ✅ | ✅ | ✅ |
| native Cypher (engine-supported grammar) | ✅¹ | ✅ (Kuzu) | ✅ (AGE) | ✅ | ✅ |
| vector search | ✅ | ✅ | ✅ (pgvector) | ✅ (`:Embeddable`) | ⚠️² |
| SPARQL (via OWL/RDF layer) | ✅ local `/sparql` | ✅ | ✅ | ✅ | ✅ |

¹ epistemic_graph is the **authority engine**. Its own `eg-query`
parser/planner serves explicit read and durable write modes; agent-utilities has
no Python-side Cypher interpreter, scan fallback, or mutation compiler.
² FalkorDB vector search is **code-correct** (Cypher DDL `CREATE VECTOR INDEX` +
`db.idx.vector.queryNodes`, verified with small vectors) but the
`falkordb/falkordb` image **crashes (SIGILL) on 768-dim vector ops on non-AVX
host CPUs** — verify on AVX-capable hardware.

What changed:
- **Neo4j/FalkorDB are first-class mirrors** — they crashed on the standard write
  path (`label()`), threw on nested props, and mis-targeted the vector index; all
  fixed. They run in the `-m live` conformance matrix and pass the contract.
- **pg-age runs Apache AGE** (`GRAPH_PG_AGE=1` / `backend_type=age`,
  `docker/pg-age-age.compose.yml`) — real openCypher incl. `count(r)`, multi-hop,
  variable-length, edge props, plus pgvector embeddings.
- **Edge properties** persist on every mirror (Ladybug via a JSON `r.properties`
  column on REL tables).
- **Local SPARQL** is served at `{prefix}/sparql` by the mandatory engine;
  Fuseki/Stardog remain optional scale-out services.

### Remaining gaps (surfaced, not hidden)
The conformance suite `skip`s with a backend-named reason rather than silently
passing where a backend genuinely can't satisfy a check:

- **FalkorDB vector search** needs an AVX-capable host (see ² above).
- **Prune semantics differ** (importance vs `last_accessed`); the suite asserts
  only the shared no-raise contract.
- **Per-mirror ontology object/link/function parity** is exercised through the
  tiny-profile gateway path against the default engine-only backend.

## Adopting this in another agent-package

Any `agents/*` package can copy the testcontainers fixture pattern (it supersedes
the hand-rolled `compose.test.yml` approach in e.g. `vector-mcp`). A minimal
`conftest.py`:

```python
import pytest

@pytest.fixture(scope="session")
def ephemeral_pg():
    pytest.importorskip("testcontainers")
    from testcontainers.postgres import PostgresContainer
    with PostgresContainer("postgres:16") as pg:
        yield pg.get_connection_url()

@pytest.mark.live
def test_against_real_pg(ephemeral_pg):
    ...  # build the client from the URI, run real assertions
```

Conventions to keep parity with this package:
- import `testcontainers` and drivers **lazily inside fixtures** so collection
  works on a minimal install;
- mark container tests `@pytest.mark.live` and keep a no-container path in the default
  suite;
- let `testcontainers` pick the port — never bind the canonical homelab port.

## The REAL ephemeral engine in tests — `tiny_engine` / `engine_graph` (CONCEPT:AU-KG.memory.provides-real-ephemeral-one)

Engine-backed tests validate against the **ACTUAL database we ship** — never
SQLite, never a mock — deployed ephemerally and destroyed afterwards. Two
first-class fixtures in `tests/conftest.py` (backed by `tests/_test_engine.py`)
own this:

- **`tiny_engine`** (session-scoped) — deploys **ONE** real
  `epistemic-graph-server` for the whole session. It resolves an explicitly
  configured binary or the feature-complete binary installed by the hard-base
  `epistemic-graph[full]>=2.23.1,<3.0.0` wheel. Pytest never invokes Cargo or
  builds an alternate artifact. The engine starts on an
  **isolated ephemeral UDS socket** under a unique temp dir, with an isolated
  temp `--persist-dir`, a test `GRAPH_SERVICE_AUTH_SECRET`, and
  `--idle-shutdown-secs 120` (so a crashed suite self-reaps). It exports
  `GRAPH_SERVICE_ENDPOINTS=unix://...` (+ the secret) so the client /
  `EngineResolver` connects to **this** explicitly managed engine (CONCEPT:AU-OS.deployment.engine-resolver-auto-provision — no autostart).
  Teardown is a graceful **SIGTERM** (the engine checkpoints + exits cleanly,
  CONCEPT:EG-KG.backend.tiny-shared), then the temp persist dir + socket are removed — zero
  residue. If the full wheel binary is unavailable, it `skip`s with a clear
  message; an externally provided `GRAPH_SERVICE_ENDPOINTS` contact is reused
  verbatim.

- **`engine_graph`** (function-scoped) — gives each test a **fresh, isolated
  tenant graph** on the session engine: a uniquely-named tenant
  (`GraphComputeEngine(graph_name=…)` auto-creates it) is yielded, then
  **tenant-purged** (CONCEPT:EG-KG.backend.tenant-delete-recreate-same) on teardown so per-test state never leaks.
  This is fast isolation — one engine process, a fresh graph per test — not a new
  process per test.

Opt a test into the real DB by **requesting `engine_graph`** (or marking it
`@pytest.mark.engine`). Example:

```python
import pytest

pytestmark = pytest.mark.engine

def test_node_roundtrip(engine_graph):
    engine_graph.add_node("alpha", {"type": "Agent", "score": 7})
    assert engine_graph.has_node("alpha")
    assert engine_graph._client.nodes.properties("alpha")["score"] == 7
```

**The consolidation seam.** Once this fixture set lands, the consolidation lanes
drop their SQLite fallbacks entirely and route to the engine unconditionally:
they connect through the same `GRAPH_SERVICE_ENDPOINTS` / `GraphComputeEngine` path
the fixtures wire, so an engine-mode consolidation test just requests
`engine_graph` and runs against the real durable engine (an integration
follow-up).
