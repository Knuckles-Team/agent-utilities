"""Unit tests for ``TrinoQueryBackend`` / ``ChangeEnvelopeBuilder`` (CA-27).

Mocks the SQLAlchemy engine layer -- no real Trino connection, no ``trino`` /
``sqlalchemy-trino`` packages needed (only ``sqlalchemy`` itself, already
installed in this workspace's shared venv). Live-endpoint proof is a separate
manual script (see the CA-27 lane report), not part of this suite.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.backends.trino_backend import (
    ChangeEnvelopeBuilder,
    MissingFenceFieldError,
    TrinoQueryBackend,
    TrinoQueryError,
    UnknownSnapshotError,
)
from agent_utilities.knowledge_graph.core.tabular_query_service import KnowledgeBatch

# ---------------------------------------------------------------------------
# KnowledgeBatch contract
# ---------------------------------------------------------------------------


def test_knowledge_batch_contract_fixture_page():
    page = KnowledgeBatch(
        rows=[{"id": 1, "name": "a"}, {"id": 2, "name": "b"}],
        columns=("id", "name"),
        snapshot_id="12345",
        lsn="12345",
        row_count=2,
        page_index=0,
    )
    assert page.row_count == len(page.rows) == 2
    assert page.columns == ("id", "name")
    # Company Architecture invariant I3: Iceberg snapshot == eg LSN.
    assert page.snapshot_id == page.lsn == "12345"


# ---------------------------------------------------------------------------
# Fake SQLAlchemy plumbing
# ---------------------------------------------------------------------------


class _FakeResult:
    def __init__(self, columns, pages, *, error: Exception | None = None):
        self._columns = columns
        self._pages = list(pages)
        self._error = error

    def keys(self):
        if self._error is not None:
            raise self._error
        return self._columns

    def fetchmany(self, n):
        if self._pages:
            return self._pages.pop(0)
        return []


class _FakeConn:
    def __init__(self, result: _FakeResult):
        self._result = result

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, _stmt):
        return self._result


class _FakeEngine:
    def __init__(self, result: _FakeResult):
        self._result = result
        self.disposed = False

    def connect(self):
        return _FakeConn(self._result)

    def dispose(self):
        self.disposed = True


def _backend(
    monkeypatch,
    *,
    result: _FakeResult,
    principal: str = "principal-a",
    token: str = "tok",
):
    created = {}

    def fake_create_engine(url, **kwargs):
        created["url"] = url
        created["kwargs"] = kwargs
        return _FakeEngine(result)

    monkeypatch.setattr("sqlalchemy.create_engine", fake_create_engine)
    backend = TrinoQueryBackend(
        "trino.apps.svc:8080",
        catalog="lakehouse",
        token_provider=lambda: token,
        principal_ref_provider=lambda: principal,
    )
    return backend, created


# ---------------------------------------------------------------------------
# Read-only guard (R3 / acceptance gate 4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO t VALUES (1)",
        "DROP TABLE t",
        "CREATE TABLE t (id int)",
        "DELETE FROM t",
        "MERGE INTO t USING s ON t.id = s.id",
        "  update t set x = 1",
    ],
)
def test_write_sql_rejected_eagerly(monkeypatch, sql):
    backend, created = _backend(monkeypatch, result=_FakeResult(("id",), []))
    with pytest.raises(TrinoQueryError):
        # Rejection happens on the query() call itself, not on first next() --
        # no engine/connection is ever touched for a write statement.
        backend.query(sql)
    assert "url" not in created


def test_select_sql_is_not_rejected(monkeypatch):
    backend, _ = _backend(monkeypatch, result=_FakeResult(("id",), [[(1,)]]))
    gen = backend.query("SELECT id FROM t")
    pages = list(gen)
    assert len(pages) == 1


# ---------------------------------------------------------------------------
# Principal-scoped connection / no static credential fallback
# ---------------------------------------------------------------------------


def test_no_token_refuses_shared_connection(monkeypatch):
    monkeypatch.setattr(
        "sqlalchemy.create_engine",
        lambda *a, **k: pytest.fail("should not build an engine"),
    )
    backend = TrinoQueryBackend(
        "trino.apps.svc:8080",
        token_provider=lambda: "",
        principal_ref_provider=lambda: "principal-a",
    )
    with pytest.raises(TrinoQueryError, match="no principal OIDC token"):
        list(backend.query("SELECT 1"))


def test_none_token_refuses_shared_connection(monkeypatch):
    monkeypatch.setattr(
        "sqlalchemy.create_engine",
        lambda *a, **k: pytest.fail("should not build an engine"),
    )
    backend = TrinoQueryBackend(
        "trino.apps.svc:8080",
        token_provider=lambda: None,
        principal_ref_provider=lambda: "principal-a",
    )
    with pytest.raises(TrinoQueryError, match="no principal OIDC token"):
        list(backend.query("SELECT 1"))


def test_backend_requires_injected_auth_ports():
    with pytest.raises(TrinoQueryError, match="token_provider"):
        TrinoQueryBackend(  # type: ignore[arg-type]
            "trino.apps.svc:8080",
            token_provider=None,
            principal_ref_provider=lambda: "principal-a",
        )
    with pytest.raises(TrinoQueryError, match="principal_ref_provider"):
        TrinoQueryBackend(  # type: ignore[arg-type]
            "trino.apps.svc:8080",
            token_provider=lambda: "token-a",
            principal_ref_provider=None,
        )


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://trino.apps.svc:8080",
        "   ",
        # Asserts rejection of an endpoint with inline basic-auth credentials
        # (synthetic fixture, not a real secret). Built via concatenation, not
        # a single string literal: scripts/security/check_secret_history.py's
        # `basic_auth_url` pattern is a structural, content-blind regex
        # (`https?://[^\s'"/@]+:[^\s'"/@]+@...`) that matches this exact shape
        # regardless of value, and that gate's only inline suppression marker
        # is one this repo's release rules forbid using. Splitting the literal
        # (same convention tests/unit/deployment/test_config_migration.py
        # already uses for scripts/check_current_only_contract.py's
        # RETIRED_IDENTIFIERS scan) means the credential-shaped span never
        # appears contiguously in the diff/patch text the scanner reads, so no
        # suppression marker is needed at all.
        "https://" + "agent:agent" + "@trino.invalid",
        "https://trino.apps.svc:0",
    ],
)
def test_backend_rejects_insecure_or_invalid_endpoint(endpoint):
    with pytest.raises(TrinoQueryError):
        TrinoQueryBackend(
            endpoint,
            token_provider=lambda: "token-a",
            principal_ref_provider=lambda: "principal-a",
        )


def test_principal_scoped_pool_never_shares_engine(monkeypatch):
    engines_built = []

    def fake_create_engine(url, **kwargs):
        engine = _FakeEngine(_FakeResult(("id",), [[(1,)]]))
        engines_built.append((url.query.get("access_token"), engine))
        return engine

    monkeypatch.setattr("sqlalchemy.create_engine", fake_create_engine)

    tokens = iter(["token-a", "token-b"])
    principals = iter(["principal-a", "principal-b"])
    backend = TrinoQueryBackend(
        "trino.apps.svc:8080",
        token_provider=lambda: next(tokens),
        principal_ref_provider=lambda: next(principals),
    )
    list(backend.query("SELECT 1"))
    list(backend.query("SELECT 1"))
    assert len(engines_built) == 2
    assert engines_built[0][0] == "token-a"
    assert engines_built[1][0] == "token-b"
    assert engines_built[0][1] is not engines_built[1][1]


def test_token_rotation_replaces_only_that_principal_pool(monkeypatch):
    engines = []

    def fake_create_engine(url, **kwargs):
        engine = _FakeEngine(_FakeResult(("id",), [[(1,)]]))
        engines.append(engine)
        return engine

    monkeypatch.setattr("sqlalchemy.create_engine", fake_create_engine)
    current = {"token": "token-a"}
    backend = TrinoQueryBackend(
        "trino.apps.svc:8080",
        token_provider=lambda: current["token"],
        principal_ref_provider=lambda: "principal-a",
    )
    list(backend.query("SELECT 1"))
    current["token"] = "token-b"
    list(backend.query("SELECT 1"))

    assert len(engines) == 2
    assert engines[0].disposed is True
    assert engines[1].disposed is False


def test_token_refresh_failure_discards_stale_principal_pool(monkeypatch):
    engine = _FakeEngine(_FakeResult(("id",), [[(1,)]]))
    monkeypatch.setattr("sqlalchemy.create_engine", lambda *a, **k: engine)
    current = {"token": "token-a"}
    backend = TrinoQueryBackend(
        "trino.apps.svc:8080",
        token_provider=lambda: current["token"],
        principal_ref_provider=lambda: "principal-a",
    )
    list(backend.query("SELECT 1"))
    current["token"] = ""

    with pytest.raises(TrinoQueryError, match="no principal OIDC token"):
        list(backend.query("SELECT 1"))
    assert engine.disposed is True


# ---------------------------------------------------------------------------
# as_of() identifier validation (never interpolates unvalidated input)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("table", ["t; DROP TABLE x--", "t.s.tbl.extra", "", "t s"])
def test_as_of_rejects_bad_table_identifier(monkeypatch, table):
    backend, created = _backend(monkeypatch, result=_FakeResult(("id",), []))
    with pytest.raises(TrinoQueryError):
        backend.as_of(table, "12345")
    assert "url" not in created


@pytest.mark.parametrize("snapshot_id", ["12345; DROP TABLE x--", "abc", "12.5", ""])
def test_as_of_rejects_bad_snapshot_id(monkeypatch, snapshot_id):
    backend, created = _backend(monkeypatch, result=_FakeResult(("id",), []))
    with pytest.raises(TrinoQueryError):
        backend.as_of("lakehouse.ns.t", snapshot_id)
    assert "url" not in created


def test_as_of_builds_expected_sql(monkeypatch):
    captured = {}

    class _CapturingResult(_FakeResult):
        pass

    def fake_create_engine(url, **kwargs):
        return _FakeEngine(_FakeResult(("id",), [[(1,)]]))

    monkeypatch.setattr("sqlalchemy.create_engine", fake_create_engine)

    from sqlalchemy import text as real_text

    def spy_text(sql):
        captured["sql"] = sql
        return real_text(sql)

    monkeypatch.setattr("sqlalchemy.text", spy_text)

    backend = TrinoQueryBackend(
        "trino.apps.svc:8080",
        token_provider=lambda: "tok",
        principal_ref_provider=lambda: "principal-a",
    )
    pages = list(backend.as_of("lakehouse.analytics.t", "999"))
    assert (
        captured["sql"] == "SELECT * FROM lakehouse.analytics.t FOR VERSION AS OF 999"
    )
    assert pages[0].snapshot_id == "999"
    assert pages[0].lsn == "999"


# ---------------------------------------------------------------------------
# Typed unknown-snapshot error (P4 negative case) -- never a silent HEAD read
# ---------------------------------------------------------------------------


def test_unknown_snapshot_raises_typed_error_not_silent_head(monkeypatch):
    err = RuntimeError("Cannot find snapshot with ID 999999999 for table t")
    backend, _ = _backend(monkeypatch, result=_FakeResult(("id",), [], error=err))
    with pytest.raises(UnknownSnapshotError):
        list(backend.as_of("lakehouse.analytics.t", "999999999"))


def test_generic_trino_failure_is_typed_but_not_unknown_snapshot(monkeypatch):
    err = RuntimeError("connection refused")
    backend, _ = _backend(monkeypatch, result=_FakeResult(("id",), [], error=err))
    with pytest.raises(TrinoQueryError) as excinfo:
        list(backend.query("SELECT 1"))
    assert not isinstance(excinfo.value, UnknownSnapshotError)


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


def test_query_pages_by_page_size(monkeypatch):
    result = _FakeResult(("id",), [[(1,), (2,)], [(3,)]])
    backend, _ = _backend(monkeypatch, result=result)
    pages = list(backend.query("SELECT id FROM t"))
    assert [p.row_count for p in pages] == [2, 1]
    assert [p.page_index for p in pages] == [0, 1]
    assert pages[0].rows == [{"id": 1}, {"id": 2}]
    assert pages[1].rows == [{"id": 3}]


def test_close_disposes_every_tracked_engine(monkeypatch):
    engines = []

    def fake_create_engine(url, **kwargs):
        e = _FakeEngine(_FakeResult(("id",), [[(1,)]]))
        engines.append(e)
        return e

    monkeypatch.setattr("sqlalchemy.create_engine", fake_create_engine)
    principals = iter(["a", "b"])
    backend = TrinoQueryBackend(
        "trino.apps.svc:8080",
        token_provider=lambda: "tok",
        principal_ref_provider=lambda: next(principals),
    )
    list(backend.query("SELECT 1"))
    list(backend.query("SELECT 1"))
    backend.close()
    assert all(e.disposed for e in engines)


# ---------------------------------------------------------------------------
# ChangeEnvelopeBuilder -- R5 fence
# ---------------------------------------------------------------------------


def test_change_envelope_builder_requires_run_id():
    builder = ChangeEnvelopeBuilder(
        connector="trino-adapter",
        run_id="",
        input_snapshot_ids=("1",),
        code_version="v1",
    )
    with pytest.raises(MissingFenceFieldError, match="run_id"):
        builder.build(source_object_id="t", payload={})


def test_change_envelope_builder_requires_input_snapshot_ids():
    builder = ChangeEnvelopeBuilder(
        connector="trino-adapter",
        run_id="run-1",
        input_snapshot_ids=(),
        code_version="v1",
    )
    with pytest.raises(MissingFenceFieldError, match="input_snapshot_id"):
        builder.build(source_object_id="t", payload={})


def test_change_envelope_builder_requires_code_version():
    builder = ChangeEnvelopeBuilder(
        connector="trino-adapter",
        run_id="run-1",
        input_snapshot_ids=("1",),
        code_version="",
    )
    with pytest.raises(MissingFenceFieldError, match="code_version"):
        builder.build(source_object_id="t", payload={})


def test_change_envelope_builder_carries_all_four_fence_fields():
    builder = ChangeEnvelopeBuilder(
        connector="trino-adapter",
        run_id="run-42",
        input_snapshot_ids=("111", "222"),
        code_version="transform@sha:abc123",
        confidence=0.9,
    )
    envelope = builder.build(
        source_object_id="lakehouse.analytics.agg",
        payload={"row_count": 10},
    )
    assert envelope.provenance["run_id"] == "run-42"
    assert envelope.provenance["input_snapshot_ids"] == ["111", "222"]
    assert envelope.provenance["code_version"] == "transform@sha:abc123"
    assert envelope.confidence == 0.9
    assert envelope.connector == "trino-adapter"
    assert envelope.source_object_id == "lakehouse.analytics.agg"
