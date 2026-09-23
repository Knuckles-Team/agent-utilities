"""Package-install manifest -> KG auto-extension (CONCEPT:AU-KG.ingest.package-install-autoingest).

Covers the dependency-free parts of the consumer: no-manifest no-op, the
content-hash watermark dedup (via ``DeltaManifest``), ``mode="full"``/``ids``
forcing a re-run past the watermark, and that a failing leg is isolated
(reported, not raised) while the other legs still run. The two reused
ingestion primitives (prompts/skills) are monkeypatched at the
module's own leg-functions so this test never needs a live engine/backend.
"""

from __future__ import annotations

import ast
import inspect
import json
from typing import Any

import pytest

from agent_utilities.knowledge_graph.core import source_sync
from agent_utilities.knowledge_graph.ingestion import package_install_ingest as pii


class _FakeEngine:
    backend = None
    graph_compute = None


def _write_manifest(data_dir, payload: dict[str, Any]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "install-manifest.json").write_text(json.dumps(payload))


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    """Point ``data_dir()`` (and its SQLite ``DeltaManifest`` fallback) at tmp_path."""
    monkeypatch.setenv("AGENT_UTILITIES_DATA_DIR", str(tmp_path))
    yield tmp_path


@pytest.fixture
def _fake_legs(monkeypatch):
    """Stub the two engine-driven ingestion primitives so no live engine is
    needed. The ontologies leg is a pure report (EH-380) and runs for real."""
    calls: list[str] = []

    def _prompts():
        calls.append("prompts")
        return {"status": "ok"}

    def _skills(engine):
        calls.append("skills")
        return {"status": "ok"}

    monkeypatch.setattr(pii, "_ingest_prompts_leg", _prompts)
    monkeypatch.setattr(pii, "_ingest_skills_leg", _skills)
    return calls


def test_no_manifest_is_a_safe_no_op(tmp_path):
    engine = _FakeEngine()
    res = pii.sync_package_install(engine, mode="delta")
    assert res["status"] == "skipped"
    assert "install-manifest.json" in res["reason"]


def test_changed_manifest_runs_every_leg(tmp_path, _fake_legs):
    _write_manifest(
        tmp_path,
        {
            "generated_at": "2026-07-12T00:00:00Z",
            "prompts": {"demo-pkg": 2},
            "ontologies": {},
        },
    )
    engine = _FakeEngine()
    res = pii.sync_package_install(engine, mode="delta")
    assert res["status"] == "ok"
    assert res["skipped_unchanged"] is False
    assert res["manifest_providers"] == ["demo-pkg"]
    assert _fake_legs == ["prompts", "skills"]
    assert res["legs"]["ontologies"] == {"status": "none", "providers": []}
    assert res["failed_legs"] == []


def test_unchanged_manifest_is_deduped_on_the_next_delta_tick(tmp_path, _fake_legs):
    _write_manifest(
        tmp_path,
        {
            "generated_at": "2026-07-12T00:00:00Z",
            "prompts": {"demo-pkg": 2},
            "ontologies": {},
        },
    )
    engine = _FakeEngine()
    first = pii.sync_package_install(engine, mode="delta")
    assert first["skipped_unchanged"] is False
    _fake_legs.clear()

    second = pii.sync_package_install(engine, mode="delta")
    assert second["skipped_unchanged"] is True
    assert _fake_legs == []  # no leg re-run — the watermark short-circuited it


def test_mode_full_bypasses_the_watermark(tmp_path, _fake_legs):
    payload = {
        "generated_at": "2026-07-12T00:00:00Z",
        "prompts": {"demo-pkg": 2},
        "ontologies": {},
    }
    _write_manifest(tmp_path, payload)
    engine = _FakeEngine()
    pii.sync_package_install(engine, mode="delta")
    _fake_legs.clear()

    res = pii.sync_package_install(engine, mode="full")
    assert res["skipped_unchanged"] is False
    assert set(_fake_legs) == {"prompts", "skills"}


def test_ids_forces_a_rerun_and_is_reported(tmp_path, _fake_legs):
    _write_manifest(
        tmp_path,
        {
            "generated_at": "2026-07-12T00:00:00Z",
            "prompts": {"demo-pkg": 2},
            "ontologies": {},
        },
    )
    engine = _FakeEngine()
    pii.sync_package_install(engine, mode="delta")
    _fake_legs.clear()

    res = pii.sync_package_install(engine, mode="delta", ids=["demo-pkg"])
    assert res["skipped_unchanged"] is False
    assert res["requested_providers"] == ["demo-pkg"]
    assert set(_fake_legs) == {"prompts", "skills"}


def test_ontologies_leg_reports_delegation_never_silent_success():
    """EH-380: installed ontologies are listed as delegated to EG
    ConnectorPack import -- never dropped, never reported as ingested."""
    result = pii._ontologies_leg({"ontologies": {"pkg-b": 2, "pkg-a": 1}})
    assert result["status"] == "delegated"
    assert result["providers"] == ["pkg-a", "pkg-b"]
    assert "ConnectorPack" in result["reason"]
    assert pii._ontologies_leg({"ontologies": {}}) == {
        "status": "none",
        "providers": [],
    }


def test_delegated_ontologies_are_not_a_failed_leg(tmp_path, _fake_legs):
    _write_manifest(
        tmp_path,
        {"generated_at": "2026-07-12T00:00:00Z", "ontologies": {"pkg-a": 1}},
    )
    res = pii.sync_package_install(_FakeEngine(), mode="delta")
    assert res["legs"]["ontologies"]["status"] == "delegated"
    assert res["failed_legs"] == []
    assert res["manifest_providers"] == ["pkg-a"]


def test_package_ingest_has_no_upward_ontology_tool_import():
    tree = ast.parse(inspect.getsource(pii))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert "agent_utilities.mcp.tools.ontology_tools" not in imported_modules


def test_skills_leg_drives_both_the_workflow_and_atomic_skill_corpus(monkeypatch):
    """The one gap this whole module names explicitly (see its own docstring's
    "skills" bullet): the automatic, watermarked ``package_install`` schedule
    used to re-drive ONLY ``ingest_skill_workflows`` -- a ``skill_type: skill``
    file got no recurring KG-ingestion path at all. ``_ingest_skills_leg`` must
    now call BOTH siblings on every tick, unmocked at this boundary so a
    regression that silently drops one of the two calls fails here."""
    calls: list[str] = []

    def _wf(engine):
        calls.append("workflows")
        return {"status": "ok", "workflows": 3}

    def _atomic(engine):
        calls.append("atomic_skills")
        return {"status": "ok", "skills": 5}

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.skill_workflow_ingest.ingest_skill_workflows",
        _wf,
    )
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.skill_workflow_ingest.ingest_atomic_skills",
        _atomic,
    )

    result = pii._ingest_skills_leg(_FakeEngine())
    assert calls == ["workflows", "atomic_skills"]
    assert result["status"] == "ok"
    assert result["workflows"]["workflows"] == 3
    assert result["atomic_skills"]["skills"] == 5


def test_skills_leg_isolates_one_failing_sub_leg_from_the_other(monkeypatch):
    """A regression in the workflow leg must not also silence atomic-skill
    ingestion (or vice versa) -- each sub-leg's exception is caught and
    reported independently, matching every other leg in this module."""

    def _wf(engine):
        raise RuntimeError("workflow corpus unreadable")

    def _atomic(engine):
        return {"status": "ok", "skills": 5}

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.skill_workflow_ingest.ingest_skill_workflows",
        _wf,
    )
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.skill_workflow_ingest.ingest_atomic_skills",
        _atomic,
    )

    result = pii._ingest_skills_leg(_FakeEngine())
    assert result["status"] == "partial"
    assert result["workflows"]["status"] == "error"
    assert "workflow corpus unreadable" in result["workflows"]["reason"]
    # the atomic leg still ran and its result is still reported
    assert result["atomic_skills"]["skills"] == 5


def test_a_failing_leg_is_isolated_and_reported(tmp_path, monkeypatch):
    """`sync_package_install` never crashes on one bad leg — each leg function
    already catches its own exceptions, so the dict literal building ``legs``
    always completes; this asserts the aggregate report surfaces the failure
    without blocking the other legs.
    """
    _write_manifest(
        tmp_path,
        {
            "generated_at": "2026-07-12T00:00:00Z",
            "prompts": {"demo-pkg": 1},
            "ontologies": {},
        },
    )

    monkeypatch.setattr(
        pii, "_ingest_prompts_leg", lambda: {"status": "error", "reason": "boom"}
    )
    monkeypatch.setattr(pii, "_ingest_skills_leg", lambda engine: {"status": "ok"})

    engine = _FakeEngine()
    result = pii.sync_package_install(engine, mode="delta")
    assert result["status"] == "ok"
    assert result["legs"]["prompts"]["status"] == "error"
    assert result["failed_legs"] == ["prompts"]
    # the other legs still ran despite the prompts leg failing
    assert result["legs"]["skills"]["status"] == "ok"
    assert result["legs"]["ontologies"]["status"] == "none"


def test_registered_as_a_source_sync_delta_handler():
    """The whole point: reachable via the ONE `source_sync` MCP/REST surface."""
    assert (
        source_sync._DELTA_HANDLERS["package_install"]
        is source_sync._sync_package_install
    )


def test_source_sync_dispatches_package_install(tmp_path, monkeypatch, _fake_legs):
    monkeypatch.setenv("AGENT_UTILITIES_DATA_DIR", str(tmp_path))
    _write_manifest(
        tmp_path,
        {
            "generated_at": "2026-07-12T00:00:00Z",
            "prompts": {"demo-pkg": 1},
            "ontologies": {},
        },
    )
    engine = _FakeEngine()
    res = source_sync.sync_source(engine, "package_install", mode="delta")
    assert res["status"] == "ok"
    assert res["source"] == "package_install"
