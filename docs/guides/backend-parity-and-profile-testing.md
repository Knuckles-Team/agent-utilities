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
| `pytest -m live` | full backend matrix (pggraph/Neo4j/FalkorDB) + Fuseki SPARQL + enterprise-profile e2e | yes |

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

**Enterprise.** Throwaway pggraph + Kafka + Fuseki. Asserts the three integration
seams: durable graph writes persist in pggraph across a reconnect; the task queue
resolves to Kafka and a put→consume→ack round-trips; the ontology publishes to
Fuseki and is queryable over SPARQL.

## Known parity gaps (surfaced, not hidden)

The conformance suite `skip`s — with a backend-named reason — rather than silently
passing where a backend genuinely can't satisfy a check. Current gaps:

- **Vector search on generic labels.** Neo4j/FalkorDB index a fixed `:Chunk`
  label for embeddings; semantic search over arbitrary `:Document` nodes returns
  nothing there, so `test_embedding_and_semantic_search_ranking` skips for those
  backends. LadybugDB (label-agnostic cosine) and pggraph (pgvector) pass.
- **Prune semantics differ** (importance vs `last_accessed`); the suite asserts
  only the shared no-raise contract, not per-backend deletion semantics.
- **Ontology object/link/function parity** across exotic backends (Neo4j/Falkor)
  is exercised through the tiny-profile gateway path against the default tiered
  backend; per-backend ontology parity is a documented follow-up.

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
