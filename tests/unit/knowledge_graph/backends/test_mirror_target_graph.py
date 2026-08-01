#!/usr/bin/python
"""Wiring tests for the configurable mirror target graph.

CONCEPT:AU-KG.backend.mirror-target-graph — Mirror Target Graph.
CONCEPT:AU-KG.backend.mirror-nonempty-default-guard — Non-Empty Default Guard.

These are **wiring** tests, not existence tests: each of the four external
mirror backends is driven against a faithful fake of its own client, and we
assert the bytes that reach that client — the Neo4j ``session(database=...)``
option, the FalkorDB ``select_graph`` key, the AGE ``cypher('<graph>', ...)``
SQL, and the Stardog ``GRAPH <...>`` in the emitted SPARQL. For every backend:

* (a) a dedicated target actually routes writes there;
* (b) the instance default is used when explicitly selected;
* (c) the non-empty-default guard actually refuses — nothing is written.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent_utilities.knowledge_graph.backends.mirror_target import (
    DEDICATED_MIRROR_NAME,
    LEVEL_DATABASE,
    LEVEL_GRAPH,
    MODE_DEDICATED,
    MODE_DEFAULT,
    MODE_DEFAULT_OVERWRITE,
    MODE_NAMED,
    MirrorTarget,
    MirrorTargetError,
    MirrorTargetRefused,
    graph_iri_for_target,
    parse_mirror_target,
    preflight_mirror_target,
    resolve_mirror_target,
)

# ── The resolver (the one place a target is decided) ─────────────────────────


def test_omission_on_a_named_connection_is_unchanged():
    """A deployment that already names its database/graph keeps that unit."""
    target = resolve_mirror_target(None, backend_type="neo4j", named_selector="prod_kg")
    assert target == MirrorTarget(mode=MODE_NAMED, name="prod_kg")
    assert not target.guarded  # never redirected, never refused


def test_omission_on_an_unnamed_connection_is_the_guarded_default():
    """The ONE behaviour change: an implicit default is now guarded."""
    target = resolve_mirror_target(None, backend_type="neo4j", named_selector=None)
    assert target.mode == MODE_DEFAULT
    assert target.guarded


def test_dedicated_defaults_to_the_built_in_name():
    target = resolve_mirror_target(
        "dedicated", backend_type="falkordb", named_selector=None
    )
    assert target.mode == MODE_DEDICATED
    assert target.name == DEDICATED_MIRROR_NAME
    assert not target.guarded


def test_dedicated_supersedes_a_named_selector_on_a_single_level_store():
    """AGE/FalkorDB require a graph name, so dedicating must win over it."""
    target = resolve_mirror_target(
        {"mode": "dedicated", "name": "kg_mirror"},
        backend_type="age",
        named_selector="agent_graph",
        default_name="agent_graph",
    )
    assert target.name == "kg_mirror"


def test_default_next_to_a_named_selector_is_refused_not_guessed():
    with pytest.raises(MirrorTargetError, match="disagree"):
        resolve_mirror_target(
            "default", backend_type="age", named_selector="agent_graph"
        )


def test_overwrite_waives_the_guard():
    target = resolve_mirror_target(
        "default-overwrite", backend_type="neo4j", named_selector=None
    )
    assert target.is_instance_default and not target.guarded


@pytest.mark.parametrize(
    "raw",
    [
        "nonsense",
        {"mode": "dedicated", "level": "shard"},
        {"mode": "dedicated", "unknown": 1},
        {"mode": "default", "name": "x"},
        7,
    ],
)
def test_a_malformed_declaration_never_degrades_to_the_default(raw):
    """A typo must raise, never be read as "nothing declared"."""
    with pytest.raises(MirrorTargetError):
        parse_mirror_target(raw)


def test_a_level_the_store_lacks_is_refused():
    with pytest.raises(MirrorTargetError, match="isolation level"):
        resolve_mirror_target(
            {"mode": "dedicated", "level": LEVEL_DATABASE},
            backend_type="falkordb",
            named_selector=None,
        )


# ── The guard itself ─────────────────────────────────────────────────────────


class _Probe:
    """Minimal backend exposing only the mirror-target contract."""

    def __init__(self, target, *, occupied=False, error=None):
        self.mirror_target = target
        self._occupied = occupied
        self._error = error
        self.ensured = 0

    def mirror_target_locator(self) -> str:
        return "the test target"

    def mirror_target_has_data(self) -> bool:
        if self._error is not None:
            raise self._error
        return self._occupied

    def ensure_mirror_target(self) -> None:
        self.ensured += 1


def test_guard_refuses_a_non_empty_instance_default():
    probe = _Probe(MirrorTarget(mode=MODE_DEFAULT), occupied=True)
    with pytest.raises(MirrorTargetRefused) as excinfo:
        preflight_mirror_target("prod-mirror", probe)
    message = str(excinfo.value)
    assert "ALREADY CONTAINS DATA" in message
    assert "nothing has been written" in message
    # Actionable: names BOTH the alternative and the explicit override.
    assert f'"mirror_target": "{MODE_DEDICATED}"' in message
    assert f'"mirror_target": "{MODE_DEFAULT_OVERWRITE}"' in message


def test_guard_allows_an_empty_instance_default():
    preflight_mirror_target("prod-mirror", _Probe(MirrorTarget(mode=MODE_DEFAULT)))


def test_guard_fails_closed_when_emptiness_cannot_be_proven():
    probe = _Probe(MirrorTarget(mode=MODE_DEFAULT), error=OSError("connection reset"))
    with pytest.raises(MirrorTargetRefused) as excinfo:
        preflight_mirror_target("prod-mirror", probe)
    assert "could not be determined" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, OSError)  # cause preserved


def test_guard_never_probes_a_dedicated_or_named_target():
    for target in (
        MirrorTarget(mode=MODE_DEDICATED, name="kg_mirror"),
        MirrorTarget(mode=MODE_NAMED, name="prod_kg"),
        MirrorTarget(mode=MODE_DEFAULT_OVERWRITE),
    ):
        probe = _Probe(target, occupied=True)
        preflight_mirror_target("prod-mirror", probe)  # must not raise
    # ...and a dedicated target is created idempotently on the way through.
    dedicated = _Probe(MirrorTarget(mode=MODE_DEDICATED, name="kg_mirror"))
    preflight_mirror_target("prod-mirror", dedicated)
    assert dedicated.ensured == 1


def test_guard_ignores_a_backend_outside_this_concept():
    preflight_mirror_target("ladybug", object())  # no mirror_target → no-op


# ── Neo4j: the isolation unit is a DATABASE ──────────────────────────────────


class _FakeNeo4jResult(list):
    """Iterable of records, plus the driver's ``consume()``."""

    def consume(self):
        return None


class _FakeNeo4jSession:
    def __init__(self, driver, options):
        self._driver = driver
        self._options = options

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query, params=None):
        self._driver.runs.append((self._options.get("database"), query, params))
        return self._driver.rows_for(query)


class _FakeNeo4jDriver:
    def __init__(self, node_count=0):
        self.runs: list[tuple] = []
        self.node_count = node_count

    def session(self, **options):
        return _FakeNeo4jSession(self, options)

    def rows_for(self, query):
        if "count(n)" in query:
            return _FakeNeo4jResult([{"mirror_target_nodes": self.node_count}])
        return _FakeNeo4jResult()

    def close(self):
        return None


@pytest.fixture()
def fake_neo4j(monkeypatch):
    driver = _FakeNeo4jDriver()
    module = SimpleNamespace(
        GraphDatabase=SimpleNamespace(driver=MagicMock(return_value=driver)),
        TrustCustomCAs=MagicMock(),
        TrustSystemCAs=MagicMock(),
        auth_management=SimpleNamespace(
            ClientCertificate=MagicMock(), ClientCertificateProviders=MagicMock()
        ),
    )
    import agent_utilities.knowledge_graph.backends.contrib.neo4j_backend as mod

    monkeypatch.setattr(mod, "GraphDatabase", module.GraphDatabase)
    return driver


def _neo4j(mirror_target, fake_neo4j):
    from agent_utilities.knowledge_graph.backends.contrib.neo4j_backend import (
        Neo4jBackend,
    )

    return Neo4jBackend(
        uri="bolt://neo:7687",
        user="u",
        password="p",
        mirror_target=mirror_target,
    )


def test_neo4j_dedicated_target_routes_writes_to_that_database(fake_neo4j):
    backend = _neo4j(MirrorTarget(mode=MODE_DEDICATED, name="kg_mirror"), fake_neo4j)
    backend.execute("MERGE (n:Thing {id: $id})", {"id": "a"})
    databases = {db for db, _q, _p in fake_neo4j.runs}
    assert databases == {"kg_mirror"}


def test_neo4j_explicit_default_uses_the_home_database(fake_neo4j):
    backend = _neo4j(MirrorTarget(mode=MODE_DEFAULT_OVERWRITE), fake_neo4j)
    backend.execute("MERGE (n:Thing {id: $id})", {"id": "a"})
    # No ``database`` session option at all → the driver's home database.
    assert [db for db, _q, _p in fake_neo4j.runs] == [None]


def test_neo4j_guard_refuses_a_non_empty_home_database(fake_neo4j):
    fake_neo4j.node_count = 41
    backend = _neo4j(MirrorTarget(mode=MODE_DEFAULT), fake_neo4j)
    with pytest.raises(MirrorTargetRefused, match="ALREADY CONTAINS DATA"):
        preflight_mirror_target("prod-neo4j", backend)
    # The probe is the ONLY thing that touched the instance — no write.
    assert all("count(n)" in q for _db, q, _p in fake_neo4j.runs)


def test_neo4j_guard_passes_on_an_empty_home_database(fake_neo4j):
    preflight_mirror_target(
        "prod-neo4j", _neo4j(MirrorTarget(mode=MODE_DEFAULT), fake_neo4j)
    )


def test_neo4j_dedicated_target_is_created_idempotently(fake_neo4j):
    backend = _neo4j(MirrorTarget(mode=MODE_DEDICATED, name="kg_mirror"), fake_neo4j)
    preflight_mirror_target("prod-neo4j", backend)
    creates = [
        (db, q, p) for db, q, p in fake_neo4j.runs if q.startswith("CREATE DATABASE")
    ]
    assert creates == [
        ("system", "CREATE DATABASE $name IF NOT EXISTS", {"name": "kg_mirror"})
    ]


# ── FalkorDB: the isolation unit is a GRAPH KEY ──────────────────────────────


class _FakeFalkorResult:
    def __init__(self, rows, header):
        self.result_set = rows
        self.header = header


class _FakeFalkorGraph:
    def __init__(self, key, node_count):
        self.key = key
        self.node_count = node_count
        self.queries: list[tuple] = []

    def query(self, query, params=None):
        self.queries.append((query, params))
        if "count(n)" in query:
            return _FakeFalkorResult([[self.node_count]], [(1, "mirror_target_nodes")])
        return _FakeFalkorResult([], [])


class _FakeFalkorClient:
    def __init__(self, node_count=0):
        self.selected: list[str] = []
        self.node_count = node_count
        self.graphs: dict[str, _FakeFalkorGraph] = {}

    def select_graph(self, key):
        self.selected.append(key)
        graph = self.graphs.setdefault(key, _FakeFalkorGraph(key, self.node_count))
        return graph


@pytest.fixture()
def fake_falkordb(monkeypatch):
    client = _FakeFalkorClient()
    import agent_utilities.knowledge_graph.backends.contrib.falkordb_backend as mod

    monkeypatch.setattr(mod, "FalkorDB", MagicMock(return_value=client))
    return client


def _falkor(mirror_target, fake_falkordb):
    from agent_utilities.knowledge_graph.backends.contrib.falkordb_backend import (
        FalkorDBBackend,
    )

    return FalkorDBBackend(host="falkor", port=6379, mirror_target=mirror_target)


def test_falkordb_dedicated_target_routes_writes_to_that_graph_key(fake_falkordb):
    backend = _falkor(
        MirrorTarget(mode=MODE_DEDICATED, name="kg_mirror"), fake_falkordb
    )
    backend.execute("MERGE (n:Thing {id: $id})", {"id": "a"})
    assert fake_falkordb.selected == ["kg_mirror"]
    assert fake_falkordb.graphs["kg_mirror"].queries  # the write landed there
    assert "agent_graph" not in fake_falkordb.graphs


def test_falkordb_explicit_default_uses_the_default_graph_key(fake_falkordb):
    backend = _falkor(
        MirrorTarget(mode=MODE_DEFAULT_OVERWRITE, name="agent_graph"), fake_falkordb
    )
    backend.execute("MERGE (n:Thing {id: $id})", {"id": "a"})
    assert fake_falkordb.selected == ["agent_graph"]


def test_falkordb_guard_refuses_a_non_empty_default_graph_key(fake_falkordb):
    fake_falkordb.node_count = 7
    backend = _falkor(
        MirrorTarget(mode=MODE_DEFAULT, name="agent_graph"), fake_falkordb
    )
    with pytest.raises(MirrorTargetRefused, match="ALREADY CONTAINS DATA"):
        preflight_mirror_target("team-falkor", backend)
    assert all("count(n)" in q for q, _p in fake_falkordb.graphs["agent_graph"].queries)


def test_falkordb_guard_passes_on_an_empty_default_graph_key(fake_falkordb):
    preflight_mirror_target(
        "team-falkor",
        _falkor(MirrorTarget(mode=MODE_DEFAULT, name="agent_graph"), fake_falkordb),
    )


# ── Apache AGE: the isolation unit is an AGE GRAPH ───────────────────────────


class _FakeCursor:
    def __init__(self, conn):
        self._conn = conn
        self.description = None
        self._rows: list[tuple] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self._conn.statements.append((sql, params))
        self._rows = []
        self.description = None
        if "ag_catalog.ag_graph" in sql:
            self._rows = [(1,)] if self._conn.graph_exists else []
            return
        if "cypher(" in sql and "count(n)" in sql:
            self._rows = [(str(self._conn.node_count),)]
            self.description = [SimpleNamespace(name="mirror_target_nodes")]

    def fetchall(self):
        return list(self._rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _FakeConn:
    def __init__(self, owner):
        self._owner = owner

    @property
    def statements(self):
        return self._owner.statements

    @property
    def graph_exists(self):
        return self._owner.graph_exists

    @property
    def node_count(self):
        return self._owner.node_count

    def cursor(self):
        return _FakeCursor(self)

    def commit(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeAgeServer:
    def __init__(self, graph_exists=True, node_count=0):
        self.statements: list[tuple] = []
        self.graph_exists = graph_exists
        self.node_count = node_count


def _age(mirror_target, server, monkeypatch):
    from contextlib import contextmanager

    from agent_utilities.knowledge_graph.backends.age_backend import AGEBackend

    backend = AGEBackend(dsn="postgresql://pg/kg", mirror_target=mirror_target)

    @contextmanager
    def _conn():
        yield _FakeConn(server)

    monkeypatch.setattr(backend, "_conn", _conn)
    monkeypatch.setattr(backend, "_run_resilient", lambda fn, name="": fn())
    return backend


def test_age_dedicated_target_routes_writes_to_that_graph(monkeypatch):
    server = _FakeAgeServer()
    backend = _age(
        MirrorTarget(mode=MODE_DEDICATED, name="kg_mirror"), server, monkeypatch
    )
    backend.execute("MERGE (n:Thing {id: 'a'})")
    cypher_sql = [
        s for s, _p in server.statements if s.startswith("SELECT * FROM cypher(")
    ]
    assert cypher_sql and all("cypher('kg_mirror'" in s for s in cypher_sql)


def test_age_explicit_default_uses_the_default_graph(monkeypatch):
    server = _FakeAgeServer()
    backend = _age(
        MirrorTarget(mode=MODE_DEFAULT_OVERWRITE, name="agent_graph"),
        server,
        monkeypatch,
    )
    backend.execute("MERGE (n:Thing {id: 'a'})")
    cypher_sql = [
        s for s, _p in server.statements if s.startswith("SELECT * FROM cypher(")
    ]
    assert cypher_sql and all("cypher('agent_graph'" in s for s in cypher_sql)


def test_age_guard_refuses_a_non_empty_default_graph(monkeypatch):
    server = _FakeAgeServer(graph_exists=True, node_count=12)
    backend = _age(
        MirrorTarget(mode=MODE_DEFAULT, name="agent_graph"), server, monkeypatch
    )
    with pytest.raises(MirrorTargetRefused, match="ALREADY CONTAINS DATA"):
        preflight_mirror_target("pg-age", backend)
    assert not any("MERGE" in s for s, _p in server.statements)


def test_age_guard_passes_on_an_empty_default_graph(monkeypatch):
    server = _FakeAgeServer(graph_exists=True, node_count=0)
    backend = _age(
        MirrorTarget(mode=MODE_DEFAULT, name="agent_graph"), server, monkeypatch
    )
    preflight_mirror_target("pg-age", backend)


def test_age_guard_treats_an_absent_graph_as_empty(monkeypatch):
    """A graph AGE has never heard of holds nothing — and must not be cypher()'d."""
    server = _FakeAgeServer(graph_exists=False, node_count=99)
    backend = _age(
        MirrorTarget(mode=MODE_DEFAULT, name="agent_graph"), server, monkeypatch
    )
    preflight_mirror_target("pg-age", backend)
    assert not any("cypher(" in s for s, _p in server.statements)


# ── Stardog: TWO levels — a database AND a named graph ───────────────────────


@pytest.fixture()
def fake_stardog(monkeypatch):
    conn = MagicMock(name="connection")
    conn.select.return_value = {"results": {"bindings": []}}

    module = SimpleNamespace()
    module.Connection = MagicMock(return_value=conn)
    admin = MagicMock()
    admin.__enter__ = MagicMock(return_value=admin)
    admin.__exit__ = MagicMock(return_value=False)
    admin.databases.return_value = []
    module.Admin = MagicMock(return_value=admin)
    module.content = SimpleNamespace(Raw=MagicMock())
    monkeypatch.setitem(sys.modules, "stardog", module)
    return SimpleNamespace(module=module, conn=conn, admin=admin)


def _stardog(mirror_target, *, database="agent_kg"):
    from agent_utilities.knowledge_graph.backends.sparql.stardog_backend import (
        StardogSparqlBackend,
    )

    return StardogSparqlBackend(
        endpoint="http://sd:5820",
        database=database,
        username="u",
        password="p",
        mirror_target=mirror_target,
    )


def _updates(fake) -> str:
    return "\n".join(call.args[0] for call in fake.conn.update.call_args_list)


def test_stardog_dedicated_graph_routes_every_write_into_that_graph(fake_stardog):
    """The DEFAULT dedicated level: one named graph, inside the same database."""
    backend = _stardog(
        MirrorTarget(mode=MODE_DEDICATED, name="kg_mirror", level=LEVEL_GRAPH)
    )
    backend.execute(
        "MERGE (n:Application {id: $id}) SET n.`source_system` = $source_system",
        {"id": "app:1", "source_system": "leanix"},
    )
    blob = _updates(fake_stardog)
    assert "GRAPH <urn:mirror:kg_mirror>" in blob
    # Source partitioning must NOT leak the write out of the dedicated graph...
    assert "urn:source:leanix" not in blob
    # ...but the source stays queryable as a property inside it.
    assert "source_system" in blob
    # The database is untouched by a graph-level dedication.
    assert backend._database == "agent_kg"


def test_stardog_dedicated_database_isolates_at_the_other_level(fake_stardog):
    backend = _stardog(
        MirrorTarget(mode=MODE_DEDICATED, name="kg_mirror", level=LEVEL_DATABASE),
        database=None,
    )
    assert backend._database == "kg_mirror"
    # No named-graph override at the database level: source partitioning stands.
    backend.execute(
        "MERGE (n:Application {id: $id}) SET n.`source_system` = $source_system",
        {"id": "app:1", "source_system": "leanix"},
    )
    assert "GRAPH <urn:source:leanix>" in _updates(fake_stardog)


def test_stardog_explicit_default_keeps_todays_routing(fake_stardog):
    backend = _stardog(
        MirrorTarget(mode=MODE_DEFAULT_OVERWRITE, name="agent_kg", level=LEVEL_GRAPH)
    )
    backend.execute("MERGE (n:Claim {id: $id})", {"id": "c1"})
    blob = _updates(fake_stardog)
    assert "GRAPH <" not in blob  # an internal node still lands in the default graph
    assert backend._database == "agent_kg"


def test_stardog_guard_refuses_a_non_empty_default_database(fake_stardog):
    fake_stardog.conn.select.return_value = {"boolean": True}
    backend = _stardog(
        MirrorTarget(mode=MODE_DEFAULT, name="agent_kg", level=LEVEL_GRAPH)
    )
    with pytest.raises(MirrorTargetRefused, match="ALREADY CONTAINS DATA"):
        preflight_mirror_target("stardog", backend)
    assert not fake_stardog.conn.update.called  # nothing written


def test_stardog_guard_passes_on_an_empty_default_database(fake_stardog):
    fake_stardog.conn.select.return_value = {"boolean": False}
    backend = _stardog(
        MirrorTarget(mode=MODE_DEFAULT, name="agent_kg", level=LEVEL_GRAPH)
    )
    preflight_mirror_target("stardog", backend)


def test_stardog_guard_fails_closed_when_the_probe_errors(fake_stardog):
    fake_stardog.conn.select.side_effect = RuntimeError("server unreachable")
    backend = _stardog(
        MirrorTarget(mode=MODE_DEFAULT, name="agent_kg", level=LEVEL_GRAPH)
    )
    with pytest.raises(MirrorTargetRefused, match="could not be determined"):
        preflight_mirror_target("stardog", backend)


def test_stardog_prune_is_scoped_to_the_dedicated_graph(fake_stardog):
    backend = _stardog(
        MirrorTarget(mode=MODE_DEDICATED, name="kg_mirror", level=LEVEL_GRAPH)
    )
    backend.prune({"min_importance": 0.5})
    assert "GRAPH <urn:mirror:kg_mirror>" in _updates(fake_stardog)


def test_stardog_upload_graph_defaults_to_the_dedicated_graph(fake_stardog):
    backend = _stardog(
        MirrorTarget(mode=MODE_DEDICATED, name="kg_mirror", level=LEVEL_GRAPH)
    )
    backend.upload_graph("<a> <b> <c> .")
    assert fake_stardog.conn.add.call_args.kwargs["graph_uri"] == "urn:mirror:kg_mirror"


def test_both_stardog_backends_resolve_the_database_identically(fake_stardog):
    """The OWL reasoning backend must not drift from the SPARQL data backend."""
    from agent_utilities.knowledge_graph.backends.owl.stardog_backend import (
        StardogBackend,
    )

    target = MirrorTarget(mode=MODE_DEDICATED, name="kg_mirror", level=LEVEL_DATABASE)
    data = _stardog(target, database=None)
    owl = StardogBackend(endpoint="http://sd:5820", mirror_target=target)
    assert owl._database == data._database == "kg_mirror"


def test_dedicated_graph_iri_is_predictable():
    target = MirrorTarget(mode=MODE_DEDICATED, name="kg_mirror", level=LEVEL_GRAPH)
    assert graph_iri_for_target(target) == "urn:mirror:kg_mirror"
    assert graph_iri_for_target(MirrorTarget(mode=MODE_DEFAULT)) is None


# ── End-to-end plumbing: kg_connections spec → create_backend → the backend ───


def test_create_backend_threads_a_declared_target_into_the_neo4j_mirror(fake_neo4j):
    """The operator-facing path: ``mirror_target`` in a connection spec must
    survive ``create_backend`` and actually select the session database."""
    from agent_utilities.knowledge_graph import backends as B

    backend = B.create_backend(
        "neo4j",
        uri="bolt://neo:7687",
        user="u",
        password="p",
        mirror_target={"mode": "dedicated", "name": "kg_mirror"},
    )
    assert backend.mirror_target.mode == MODE_DEDICATED
    assert backend.database == "kg_mirror"
    # create_schema() ran during construction — and landed in kg_mirror, not home.
    assert {db for db, _q, _p in fake_neo4j.runs} == {"kg_mirror"}


def test_create_backend_rejects_a_contradictory_declaration(fake_neo4j):
    from agent_utilities.knowledge_graph import backends as B

    with pytest.raises(MirrorTargetError, match="disagree"):
        B.create_backend(
            "neo4j",
            uri="bolt://neo:7687",
            user="u",
            password="p",
            db_name="prod_kg",
            mirror_target="default",
        )


def test_build_mirror_set_records_a_refusal_and_attaches_nothing(monkeypatch):
    """A refused mirror is isolated like any other failed mirror — the authority
    stays up — but it is reported distinctly so a health check says "fix the
    config", not "retry"."""
    from agent_utilities.core.config import config as cfg
    from agent_utilities.knowledge_graph import backends as B

    monkeypatch.delenv("GRAPH_MIRROR_TARGETS", raising=False)
    monkeypatch.setattr(cfg, "graph_mirror_targets", ["prod-neo4j"], raising=False)
    monkeypatch.setattr(
        cfg,
        "kg_connections",
        [{"name": "prod-neo4j", "backend": "neo4j", "mirror_target": "default"}],
        raising=False,
    )
    monkeypatch.setattr(cfg, "continuous_stardog_mirror", False, raising=False)
    monkeypatch.setattr(
        B,
        "_build_member",
        lambda spec: _Probe(MirrorTarget(mode=MODE_DEFAULT), occupied=True),
    )
    B._MIRROR_BUILD_STATUS.clear()
    try:
        mirrors = B._build_mirror_set()
        assert mirrors == {}  # nothing attached → nothing can be written
        status = B.get_mirror_build_status()["prod-neo4j"]
        assert status["ok"] is False
        assert status["refused"] is True
        assert "ALREADY CONTAINS DATA" in status["reason"]
    finally:
        B._MIRROR_BUILD_STATUS.clear()


def test_build_mirror_set_attaches_a_dedicated_mirror(monkeypatch):
    from agent_utilities.core.config import config as cfg
    from agent_utilities.knowledge_graph import backends as B

    monkeypatch.delenv("GRAPH_MIRROR_TARGETS", raising=False)
    monkeypatch.setattr(cfg, "graph_mirror_targets", ["prod-neo4j"], raising=False)
    monkeypatch.setattr(
        cfg,
        "kg_connections",
        [{"name": "prod-neo4j", "backend": "neo4j", "mirror_target": "dedicated"}],
        raising=False,
    )
    monkeypatch.setattr(cfg, "continuous_stardog_mirror", False, raising=False)
    probe = _Probe(MirrorTarget(mode=MODE_DEDICATED, name=DEDICATED_MIRROR_NAME))
    monkeypatch.setattr(B, "_build_member", lambda spec: probe)
    B._MIRROR_BUILD_STATUS.clear()
    try:
        assert B._build_mirror_set() == {"prod-neo4j": probe}
        assert B.get_mirror_build_status()["prod-neo4j"]["ok"] is True
        assert probe.ensured == 1  # created idempotently on the way in
    finally:
        B._MIRROR_BUILD_STATUS.clear()
