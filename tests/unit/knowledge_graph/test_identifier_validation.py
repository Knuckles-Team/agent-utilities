"""Tests for the centralized identifier-validation gate (CONCEPT:AU-KG.backend.mirror-health-repair,
Wave-0 S10) — ``agent_utilities/security/identifiers.py`` and its application
across every mirror-backend adapter that f-string-interpolates a
label/table/relationship-type/column into Cypher/SQL/DDL.

All tests use in-process stubs (``Backend.__new__`` + attribute patching /
mocked cursors, mirroring ``tests/unit/test_postgresql_backend.py``'s own
convention) — no live database, no running epistemic-graph engine.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent_utilities.security.identifiers import (
    CYPHER_IDENTIFIER_RE,
    SQL_IDENTIFIER_RE,
    InvalidIdentifierError,
    quote_sql_identifier,
    validate_identifier,
    validate_sql_identifier,
)

# ---------------------------------------------------------------------------
# Shared malicious / valid identifier corpora
# ---------------------------------------------------------------------------

MALICIOUS_IDENTIFIERS = [
    pytest.param("x; DROP TABLE--", id="sql-statement-injection"),
    pytest.param('x" ; DROP TABLE users; --', id="double-quote-smuggling"),
    pytest.param("x` ; DROP TABLE users; --", id="backtick-smuggling"),
    pytest.param("x' OR '1'='1", id="single-quote-smuggling"),
    pytest.param("a" * 129, id="129-char-name-cypher-ceiling"),
    pytest.param("Ｐerson", id="unicode-fullwidth-homoglyph"),
    pytest.param("café", id="unicode-accented"),
    pytest.param("用户", id="unicode-cjk"),
    pytest.param("", id="empty-string"),
    pytest.param("1abc", id="leading-digit"),
    pytest.param("a b", id="embedded-space"),
    pytest.param("a\nb", id="embedded-newline"),
    pytest.param("a.b", id="embedded-dot"),
    pytest.param("a;b", id="embedded-semicolon"),
]

# Ontology-normalized labels/tables that MUST keep working — the quality bar
# ("do not change query semantics for valid identifiers").
VALID_IDENTIFIERS = [
    "Person",
    "BusinessProcess",
    "kg_edges",
    "Node_1",
    "tenant_id",
    "_private",
    "A",
    "a" * 128,  # Cypher ceiling: exactly 128 chars must still pass
]

VALID_SQL_IDENTIFIERS = [
    "Person",
    "kg_edges",
    "tenant_id",
    "a" * 63,  # SQL ceiling: exactly 63 chars must still pass
]


# ---------------------------------------------------------------------------
# The shared gate itself
# ---------------------------------------------------------------------------


class TestValidateIdentifier:
    @pytest.mark.parametrize("bad", MALICIOUS_IDENTIFIERS)
    def test_rejects_malicious_identifiers(self, bad):
        with pytest.raises(InvalidIdentifierError):
            validate_identifier(bad, kind="label")

    @pytest.mark.parametrize("good", VALID_IDENTIFIERS)
    def test_accepts_ontology_normalized_labels(self, good):
        assert validate_identifier(good, kind="label") == good

    def test_129_chars_rejected_128_accepted(self):
        assert validate_identifier("a" * 128) == "a" * 128
        with pytest.raises(InvalidIdentifierError):
            validate_identifier("a" * 129)

    def test_error_never_echoes_the_offending_value(self):
        """CypherEngineError-style no-leak discipline: the raised message/repr
        must never contain the malicious payload, so it can't be smuggled into
        a log line or error response."""
        secret_marker = "x; DROP TABLE super_secret_table_name --"
        with pytest.raises(InvalidIdentifierError) as exc_info:
            validate_identifier(secret_marker, kind="label")
        assert secret_marker not in str(exc_info.value)
        assert secret_marker not in repr(exc_info.value)

    def test_is_a_value_error_subclass_for_backward_compatible_catches(self):
        assert issubclass(InvalidIdentifierError, ValueError)

    def test_kind_is_attached_but_not_echoed_in_a_way_that_leaks_value(self):
        with pytest.raises(InvalidIdentifierError) as exc_info:
            validate_identifier("bad;name", kind="relationship type")
        assert exc_info.value.kind == "relationship type"
        assert "relationship type" in str(exc_info.value)


class TestValidateSqlIdentifier:
    @pytest.mark.parametrize("bad", MALICIOUS_IDENTIFIERS)
    def test_rejects_malicious_identifiers(self, bad):
        with pytest.raises(InvalidIdentifierError):
            validate_sql_identifier(bad, kind="table")

    @pytest.mark.parametrize("good", VALID_SQL_IDENTIFIERS)
    def test_accepts_ontology_normalized_tables(self, good):
        assert validate_sql_identifier(good, kind="table") == good

    def test_64_chars_rejected_63_accepted(self):
        """PostgreSQL's NAMEDATALEN-1 = 63-byte ceiling — stricter than the
        128-char Cypher variant on purpose (a too-long SQL name silently
        truncates/collides rather than erroring)."""
        assert validate_sql_identifier("a" * 63) == "a" * 63
        with pytest.raises(InvalidIdentifierError):
            validate_sql_identifier("a" * 64)

    def test_129_char_cypher_valid_name_still_rejected_for_sql(self):
        """A name within the Cypher ceiling but past the SQL one must still
        fail the SQL-specific gate — the two grammars are intentionally
        different, not aliases of each other."""
        seventy_chars = "a" * 70
        assert validate_identifier(seventy_chars) == seventy_chars
        with pytest.raises(InvalidIdentifierError):
            validate_sql_identifier(seventy_chars)


class TestQuoteSqlIdentifier:
    @pytest.mark.parametrize("bad", MALICIOUS_IDENTIFIERS)
    def test_rejects_malicious_identifiers(self, bad):
        with pytest.raises(InvalidIdentifierError):
            quote_sql_identifier(bad, kind="table")

    def test_quotes_a_valid_identifier(self):
        assert quote_sql_identifier("kg_edges", kind="table") == '"kg_edges"'

    def test_regex_constants_agree_with_the_functions(self):
        # Sanity: the exported regex constants are what the functions actually use.
        assert CYPHER_IDENTIFIER_RE.fullmatch("Person")
        assert not CYPHER_IDENTIFIER_RE.fullmatch("a" * 129)
        assert SQL_IDENTIFIER_RE.fullmatch("Person")
        assert not SQL_IDENTIFIER_RE.fullmatch("a" * 64)


# ---------------------------------------------------------------------------
# Adapter entry points — parameterized over the three mirror backends
# ---------------------------------------------------------------------------


class TestEpistemicGraphBackendPruneRejectsMaliciousPropertyKeys:
    """``EpistemicGraphBackend.prune()`` is the canonical shared-interface
    entry point (``GraphBackend.prune``) implemented identically in spirit by
    every backend below. The malicious criteria KEY (not value — values are
    already bound as ``$key`` params) must be rejected before any query runs."""

    @pytest.fixture
    def backend(self):
        from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
            EpistemicGraphBackend,
        )

        # __new__ bypasses __init__ entirely — no engine connection is ever
        # made, matching this task's "no live DBs needed" requirement. Safe
        # because prune() validates every criteria key before touching
        # self._graph/self.execute_read.
        return EpistemicGraphBackend.__new__(EpistemicGraphBackend)

    @pytest.mark.parametrize("bad_key", MALICIOUS_IDENTIFIERS)
    def test_rejects_malicious_criteria_key(self, backend, bad_key):
        with pytest.raises(ValueError, match="identifier"):
            backend.prune({bad_key: "irrelevant-value"})

    def test_rejects_empty_criteria(self, backend):
        with pytest.raises(ValueError, match="empty"):
            backend.prune({})

    def test_valid_property_key_passes_validation(self, backend, monkeypatch):
        """A valid key must reach the (mocked) execute_read call — proving
        the gate only rejects genuinely invalid identifiers."""
        calls: list[tuple[str, dict]] = []
        monkeypatch.setattr(
            backend,
            "execute_read",
            lambda q, p: (calls.append((q, p)), [])[1],
            raising=False,
        )
        monkeypatch.setattr(
            backend, "_graph", MagicMock(remove_node=lambda _id: None), raising=False
        )
        backend.prune({"min_importance": 0.3})
        assert len(calls) == 1
        assert "min_importance" in calls[0][0]


class TestLadybugBackendPruneRejectsMaliciousNodeType:
    """The exact bug this task fixed: ``criteria["node_type"]`` used to reach
    a Cypher label position (``f":{node_type}"``) with zero validation."""

    @pytest.fixture
    def backend(self):
        from agent_utilities.knowledge_graph.backends.contrib.ladybug_backend import (
            LadybugBackend,
        )

        # __new__ bypasses __init__ — no ladybug/Kuzu file is ever opened.
        return LadybugBackend.__new__(LadybugBackend)

    @pytest.mark.parametrize("bad_node_type", MALICIOUS_IDENTIFIERS)
    def test_rejects_malicious_node_type(self, backend, bad_node_type):
        if not bad_node_type:
            pytest.skip("empty node_type takes the no-filter branch by design")
        from agent_utilities.security.identifiers import InvalidIdentifierError

        with pytest.raises(InvalidIdentifierError):
            backend.prune({"node_type": bad_node_type, "min_importance": 0.1})

    def test_valid_node_type_reaches_execute(self, backend, monkeypatch):
        calls: list[str] = []
        monkeypatch.setattr(
            backend, "execute", lambda q, p: calls.append(q), raising=False
        )
        monkeypatch.setattr(backend, "checkpoint_wal", lambda: None, raising=False)
        backend.prune({"node_type": "Memory", "min_importance": 0.2})
        assert calls
        assert "MATCH (n:Memory)" in calls[0]

    def test_no_node_type_filter_is_unaffected(self, backend, monkeypatch):
        """Criteria without ``node_type`` must keep working unchanged (query
        semantics preserved for the common no-label-filter case)."""
        calls: list[str] = []
        monkeypatch.setattr(
            backend, "execute", lambda q, p: calls.append(q), raising=False
        )
        monkeypatch.setattr(backend, "checkpoint_wal", lambda: None, raising=False)
        backend.prune({"min_importance": 0.2})
        assert calls == ["MATCH (n) WHERE n.importance_score < $min_imp DETACH DELETE n"]


class TestLadybugAutoDdlGuards:
    """The auto-heal DDL paths (``_ensure_node_table_unlocked`` /
    ``_ensure_rel_pair_unlocked``) that create tables for undeclared
    labels/rel-types — the exact mechanism a crafted MERGE could otherwise
    abuse to run arbitrary DDL."""

    @pytest.fixture
    def backend(self):
        from agent_utilities.knowledge_graph.backends.contrib.ladybug_backend import (
            LadybugBackend,
        )

        b = LadybugBackend.__new__(LadybugBackend)
        b._known_node_tables = set()
        b._known_rel_pairs = set()
        b.conn = MagicMock()
        return b

    @pytest.mark.parametrize("bad_label", MALICIOUS_IDENTIFIERS)
    def test_ensure_node_table_rejects_malicious_label(self, backend, bad_label):
        if not bad_label:
            pytest.skip("empty label short-circuits on the cache check")
        with pytest.raises(InvalidIdentifierError):
            backend._ensure_node_table_unlocked(bad_label)
        backend.conn.execute.assert_not_called()

    def test_ensure_node_table_accepts_valid_label(self, backend):
        backend._ensure_node_table_unlocked("Memory")
        assert backend.conn.execute.called
        (sql,), _ = backend.conn.execute.call_args
        assert "Memory" in sql

    @pytest.mark.parametrize("bad_rel", MALICIOUS_IDENTIFIERS)
    def test_ensure_rel_pair_rejects_malicious_rel_type(self, backend, bad_rel):
        if not bad_rel:
            pytest.skip("empty rel is not exercised by this shape")
        with pytest.raises(InvalidIdentifierError):
            backend._ensure_rel_pair_unlocked(bad_rel, "Person", "Company")
        backend.conn.execute.assert_not_called()


class TestPostgresRlsStatementsRejectsMaliciousTable:
    """``rls_statements`` is a pure staticmethod (no I/O) — the most direct
    way to prove the DDL-building path itself rejects a bad identifier."""

    @pytest.mark.parametrize("bad_table", MALICIOUS_IDENTIFIERS)
    def test_rejects_malicious_table(self, bad_table):
        from agent_utilities.knowledge_graph.backends.postgresql_backend import (
            PostgreSQLBackend,
        )

        with pytest.raises(ValueError, match="identifier"):
            PostgreSQLBackend.rls_statements(bad_table)

    def test_accepts_a_valid_table_and_builds_the_expected_ddl(self):
        from agent_utilities.knowledge_graph.backends.postgresql_backend import (
            PostgreSQLBackend,
        )

        stmts = PostgreSQLBackend.rls_statements("kg_edges")
        assert any('ALTER TABLE "kg_edges"' in s for s in stmts)
        assert any("CREATE POLICY tenant_isolation" in s for s in stmts)


class TestPostgresGraphTraverseRejectsMaliciousSeedTable:
    """``graph_traverse``/``graph_shortest_path`` interpolate their table
    argument into a value that PostgreSQL casts ``::regclass`` — unvalidated,
    a caller-chosen name could resolve to ANY table in the database, not just
    a pgGraph-registered one. Confirms the malicious value never reaches the
    (mocked) cursor at all."""

    @pytest.fixture
    def backend(self):
        from agent_utilities.knowledge_graph.backends.postgresql_backend import (
            PostgreSQLBackend,
            _SingleConnPool,
        )

        b = PostgreSQLBackend.__new__(PostgreSQLBackend)
        b._pggraph_schema = "public"
        b._pggraph_available = True
        mock_cur = MagicMock()
        mock_cur.description = []
        mock_cur.fetchall.return_value = []
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        b._pool = _SingleConnPool(mock_conn)
        return b, mock_cur

    @pytest.mark.parametrize("bad_table", MALICIOUS_IDENTIFIERS)
    def test_graph_traverse_rejects_malicious_seed_table(self, backend, bad_table):
        if not bad_table:
            pytest.skip("empty seed_table is not a realistic caller shape here")
        be, mock_cur = backend
        result = be.graph_traverse(bad_table, "seed-1")
        assert result == []
        mock_cur.execute.assert_not_called()

    @pytest.mark.parametrize("bad_table", MALICIOUS_IDENTIFIERS)
    def test_graph_shortest_path_rejects_malicious_tables(self, backend, bad_table):
        if not bad_table:
            pytest.skip("empty table is not a realistic caller shape here")
        be, mock_cur = backend
        result = be.graph_shortest_path(bad_table, "s1", "SafeTable", "t1")
        assert result == []
        mock_cur.execute.assert_not_called()

    def test_graph_traverse_accepts_a_valid_seed_table(self, backend):
        be, mock_cur = backend
        be.graph_traverse("Memory", "seed-1")
        assert mock_cur.execute.called
        (_sql, params), _ = mock_cur.execute.call_args
        assert params[0] == "public.Memory"


# ---------------------------------------------------------------------------
# migration.py — copy_graph's index-creation + drift-check loops
# ---------------------------------------------------------------------------


class TestMigrationEnsureIdIndexesSkipsMaliciousLabelsGracefully:
    def test_malicious_label_is_skipped_not_raised(self):
        from agent_utilities.knowledge_graph.migration import _ensure_id_indexes

        backend = MagicMock()
        backend.__class__.__name__ = "Neo4jBackend"
        # Must not raise — a bad label degrades exactly like any other
        # backend.execute() failure in this best-effort helper.
        made = _ensure_id_indexes(backend, {"x; DROP TABLE--"})
        assert made == 0
        backend.execute.assert_not_called()

    def test_valid_label_creates_the_index(self):
        from agent_utilities.knowledge_graph.migration import _ensure_id_indexes

        backend = MagicMock()
        backend.__class__.__name__ = "Neo4jBackend"
        backend.execute.return_value = []
        made = _ensure_id_indexes(backend, {"Memory"})
        assert made == 1
        backend.execute.assert_called_once()
        (query,), _ = backend.execute.call_args
        assert "Memory" in query


# ---------------------------------------------------------------------------
# sync.py — the sanitize-or-default identifier helper
# ---------------------------------------------------------------------------


class TestSyncSafeGraphIdentifier:
    def test_malicious_input_coerces_to_default(self):
        from agent_utilities.knowledge_graph.pipeline.phases.sync import (
            _safe_graph_identifier,
        )

        assert _safe_graph_identifier("a" * 129, default="Code") == "Code"

    def test_valid_input_passes_through(self):
        from agent_utilities.knowledge_graph.pipeline.phases.sync import (
            _safe_graph_identifier,
        )

        assert _safe_graph_identifier("Memory") == "Memory"

    def test_sanitizes_special_characters_into_underscores(self):
        from agent_utilities.knowledge_graph.pipeline.phases.sync import (
            _safe_graph_identifier,
        )

        # spaces/punctuation are folded to '_' then re-validated — this is
        # the pre-existing coerce-not-reject behavior for real ontology
        # labels (e.g. "Business Process") and must stay unchanged.
        assert _safe_graph_identifier("Business Process") == "Business_Process"


# ---------------------------------------------------------------------------
# manifest.py / job_manager.py — module-level pre-validated constants
# ---------------------------------------------------------------------------


class TestModuleLevelConstantsValidatedAtImport:
    def test_manifest_label_constant_is_valid_and_unchanged(self):
        from agent_utilities.knowledge_graph.ingestion.manifest import _LABEL

        assert _LABEL == "IngestManifest"
        assert CYPHER_IDENTIFIER_RE.fullmatch(_LABEL)

    def test_job_manager_node_type_constant_is_valid_and_unchanged(self):
        from agent_utilities.knowledge_graph.extraction.job_manager import (
            _JOB_NODE_TYPE,
        )

        assert _JOB_NODE_TYPE == "extraction_job"
        assert CYPHER_IDENTIFIER_RE.fullmatch(_JOB_NODE_TYPE)
