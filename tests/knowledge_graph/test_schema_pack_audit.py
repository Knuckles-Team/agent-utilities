from __future__ import annotations

"""Tests for the Schema-Pack candidate-type audit.

CONCEPT:AU-KG.ontology.schema-pack-lifecycle-audit — Schema-Pack Lifecycle, Loader & Audit
"""


from types import SimpleNamespace

import pytest

from agent_utilities.models.schema_pack import SchemaPack, SchemaPackMode
from agent_utilities.models.schema_pack_audit import SchemaCandidateAuditor


@pytest.fixture(autouse=True)
def _audit_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAPH_SCHEMA_AUDIT_DIR", str(tmp_path))
    monkeypatch.delenv("GRAPH_SCHEMA_AUDIT_VERBOSE", raising=False)
    SchemaCandidateAuditor.instance().reset()
    yield
    SchemaCandidateAuditor.instance().reset()


def test_record_is_hashed_by_default():
    a = SchemaCandidateAuditor.instance()
    assert a.record("node", "SecretDiseaseType", "core") is True
    rows = a.review()
    assert len(rows) == 1
    assert rows[0]["redacted"] is True
    assert "SecretDiseaseType" not in rows[0]["type"]


def test_verbose_records_raw(monkeypatch):
    monkeypatch.setenv("GRAPH_SCHEMA_AUDIT_VERBOSE", "1")
    a = SchemaCandidateAuditor.instance()
    a.record("edge", "weakens", "core")
    rows = a.review()
    assert rows[-1]["type"] == "weakens"
    assert rows[-1]["redacted"] is False


def test_dedup_within_process():
    a = SchemaCandidateAuditor.instance()
    assert a.record("node", "X", "p") is True
    assert a.record("node", "X", "p") is False
    assert len(a.review()) == 1


def test_review_round_trips_multiple():
    a = SchemaCandidateAuditor.instance()
    a.record("node", "A", "p")
    a.record("edge", "B", "p")
    rows = a.review()
    assert {r["kind"] for r in rows} == {"node", "edge"}


# --- Engine-method integration (observe-only, EXCLUSIVE-mode gated) ---


def _audit(fake_self, kind, name):
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    IntelligenceGraphEngine._audit_candidate_type(fake_self, kind, name)


def test_exclusive_pack_flags_out_of_pack_edge():
    pack = SchemaPack(name="strict", mode=SchemaPackMode.EXCLUSIVE)  # only core types
    _audit(SimpleNamespace(active_schema_pack=pack), "edge", "weakens")
    rows = SchemaCandidateAuditor.instance().review()
    assert len(rows) == 1
    assert rows[0]["kind"] == "edge"


def test_additive_core_pack_flags_nothing():
    from agent_utilities.models.schema_packs import get_schema_pack

    _audit(
        SimpleNamespace(active_schema_pack=get_schema_pack("core")), "edge", "weakens"
    )
    assert SchemaCandidateAuditor.instance().review() == []


def test_no_pack_is_noop():
    _audit(SimpleNamespace(active_schema_pack=None), "node", "Whatever")
    assert SchemaCandidateAuditor.instance().review() == []
