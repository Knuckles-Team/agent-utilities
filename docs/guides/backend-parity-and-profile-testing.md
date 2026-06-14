# Backend parity & deployment-profile testing

Agent-utilities ships the *same* code to a Raspberry Pi 3 and to an enterprise
cluster, on top of any of several storage backends. Two test layers keep that
promise honest:

1. **Backend conformance** — one assertion body run against **every** supported
   storage backend (`tests/integration/backends/`).
2. **Deployment profiles** — two end-to-end topologies: a zero-dependency "tiny"
   profile and the full "enterprise" profile (`tests/integration/profiles/`).

Both stand real services up in **throwaway containers** via
[`testcontainers`](https://testcontainers-python.readthedocs.io/) — on random
free ports, torn down deterministically — so a run is hermetic and can be
intentionally broken without touching anything you care about.

## Install

```bash
pip install 'agent-utilities[test-backends]'   # drivers + testcontainers
```

Requires a reachable Docker daemon for the live matrix. The two zero-infra cases
(epistemic-graph L1, embedded LadybugDB) need neither Docker nor `testcontainers`.

## What runs when

| Selection | What runs | Needs Docker? |
|---|---|---|
| `pytest` (default, `-m "not live"`) | tiny-profile zero-dep e2e; conformance for `epistemic_graph` + `ladybug` | no |
| `pytest -m live` | full backend matrix (pg-age/Neo4j/FalkorDB) + Fuseki SPARQL + enterprise-profile e2e | yes |

The default PR suite therefore continuously enforces the **Pi-3 zero-dependency
contract** (including a cold-import footprint guard) without requiring Docker, and
the heavyweight cross-backend/cluster checks run under `-m live` and on the
nightly job.

```bash
# Pi-3 guarantee (fast, no Docker)
pytest tests/integration/profiles/test_profile_tiny_zero_dep.py -v

# zero-infra conformance params (no Docker)
pytest tests/integration/backends/test_backend_conformance.py -v

# full live matrix + enterprise profile (Docker required)
pytest tests/integration/backends -m live -v
pytest tests/integration/profiles/test_profile_enterprise_full.py -m live -v
```

The KG/engine assertions skip when the local epistemic-graph engine isn't running
(`GRAPH_SERVICE_SOCKET` unset, e.g. a polyrepo CI without the Rust source); the
footprint guard always runs.

## The two profiles

**Tiny (Raspberry Pi 3).** `GRAPH_BACKEND=tiered` (epistemic-graph L1 + embedded
LadybugDB L2), `OWL_BACKEND=owlready2`, SQLite task queue, inline dispatch, no
`GRAPH_DB_URI`/`STATE_DB_URI`/Kafka. The test boots the gateway REST surface
in-process and asserts write→query works and the local OWL reasoner runs — with
**zero containers** — plus a subprocess cold-import check that no external-service
driver (`aiokafka`/`psycopg`/`neo4j`/`falkordb`/`pystardog`/`confluent_kafka`)
leaked into the footprint.

**Enterprise.** Throwaway pg-age + Kafka + Fuseki. Asserts the three integration
seams: durable graph writes persist in pg-age across a reconnect; the task queue
resolves to Kafka and a put→consume→ack round-trips; the ontology publishes to
Fuseki and is queryable over SPARQL.

## Parity status (the 100%-parity program)

A live full-matrix probe (write via the engine, read via `backend.execute`, all
five backends running) drove a phased program that closed the gaps. **Verified
current state:**

| Capability | epistemic_graph | ladybug | pg-age (AGE) | neo4j | falkordb |
|---|---|---|---|---|---|
| node props (declared/ad-hoc/nested) | ✅ | ✅ (ad-hoc in `metadata`) | ✅ | ✅ | ✅ |
| edge existence | ✅ | ✅ | ✅ | ✅ | ✅ |
| edge properties | ✅ | ✅ (JSON `r.properties`) | ✅ | ✅ | ✅ |
| full Cypher (count/alias/multi-hop) | subset¹ | ✅ (Kuzu) | ✅ (AGE) | ✅ | ✅ |
| vector search | ✅ | ✅ | ✅ (pgvector) | ✅ (`:Embeddable`) | ⚠️² |
| SPARQL (via OWL/RDF layer) | ✅ local `/sparql` | ✅ | ✅ | ✅ | ✅ |

¹ epistemic_graph is the in-memory **L1 working store**; its `backend.execute`
interprets an operational Cypher subset (id-anchored traversals), and multi-hop
traversal is served via the compute layer / tiered L3 — by design, not a gap.
² FalkorDB vector search is **code-correct** (Cypher DDL `CREATE VECTOR INDEX` +
`db.idx.vector.queryNodes`, verified with small vectors) but the
`falkordb/falkordb` image **crashes (SIGILL) on 768-dim vector ops on non-AVX
host CPUs** — verify on AVX-capable hardware.

What changed:
- **Neo4j/FalkorDB are first-class** — they crashed on the standard write path
  (`label()`), threw on nested props, and mis-targeted the vector index; all fixed.
  They run in the `-m live` conformance matrix and pass the contract.
- **pg-age runs Apache AGE** (`GRAPH_PG_AGE=1` / `backend_type=age`,
  `docker/pg-age-age.compose.yml`) — real openCypher incl. `count(r)`, multi-hop,
  variable-length, edge props, plus pgvector embeddings.
- **Edge properties** persist on every backend (Ladybug via a JSON `r.properties`
  column on REL tables).
- **Local SPARQL** is served at `{prefix}/sparql` over the OWL/RDF bridge (rdflib
  materialization) with zero external deps — Fuseki/Stardog are optional scale-out.

### Remaining gaps (surfaced, not hidden)
The conformance suite `skip`s with a backend-named reason rather than silently
passing where a backend genuinely can't satisfy a check:

- **FalkorDB vector search** needs an AVX-capable host (see ² above).
- **Prune semantics differ** (importance vs `last_accessed`); the suite asserts
  only the shared no-raise contract.
- **Per-backend ontology object/link/function parity** is exercised through the
  tiny-profile gateway path against the default tiered backend.

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
        host, port = pg.get_container_host_ip(), pg.get_exposed_port(5432)
        yield f"postgresql://test:test@{host}:{port}/test"

@pytest.mark.live
def test_against_real_pg(ephemeral_pg):
    ...  # build the client from the URI, run real assertions
```

Conventions to keep parity with this package:
- import `testcontainers` and drivers **lazily inside fixtures** so collection
  works on a minimal install;
- mark container tests `@pytest.mark.live` and keep a zero-dep path in the default
  suite;
- let `testcontainers` pick the port — never bind the canonical homelab port.
```
