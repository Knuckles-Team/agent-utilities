"""Package-install manifest -> KG auto-extension (CONCEPT:AU-KG.ingest.package-install-autoingest).

Covers the dependency-free parts of the consumer: no-manifest no-op, the
content-hash watermark dedup (via ``DeltaManifest``), ``mode="full"``/``ids``
forcing a re-run past the watermark, and that a failing leg is isolated
(reported, not raised) while the other legs still run. The three reused
ingestion primitives (prompts/ontologies/skills) are monkeypatched at the
module's own leg-functions so this test never needs a live engine/backend.
"""

from __future__ import annotations

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
    """Stub the three reused ingestion primitives so no live engine is needed.

    NOT autouse — ``test_ontologies_leg_catches_its_own_exception`` exercises
    the REAL ``_ingest_ontologies_leg`` and would otherwise have it clobbered
    by this stub before its own patch ever ran.
    """
    calls: list[str] = []

    def _prompts():
        calls.append("prompts")
        return {"status": "ok"}

    def _ontologies(engine):
        calls.append("ontologies")
        return {"status": "ok"}

    def _skills(engine):
        calls.append("skills")
        return {"status": "ok"}

    monkeypatch.setattr(pii, "_ingest_prompts_leg", _prompts)
    monkeypatch.setattr(pii, "_ingest_ontologies_leg", _ontologies)
    monkeypatch.setattr(pii, "_ingest_skills_leg", _skills)
    return calls


def test_no_manifest_is_a_safe_no_op(tmp_path):
    engine = _FakeEngine()
    res = pii.sync_package_install(engine, mode="delta")
    assert res["status"] == "skipped"
    assert "install-manifest.json" in res["reason"]


def test_changed_manifest_runs_all_three_legs(tmp_path, _fake_legs):
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
    assert set(_fake_legs) == {"prompts", "ontologies", "skills"}
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
    assert set(_fake_legs) == {"prompts", "ontologies", "skills"}


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
    assert set(_fake_legs) == {"prompts", "ontologies", "skills"}


def test_ontologies_leg_catches_its_own_exception(monkeypatch):
    """Each leg function is the isolation boundary — it must never raise out."""

    class _BoomLifecycle:
        def __init__(self, engine=None):
            raise RuntimeError("ontology backend unavailable")

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ontology.lifecycle.OntologyLifecycle",
        _BoomLifecycle,
    )
    result = pii._ingest_ontologies_leg(_FakeEngine())
    assert result["status"] == "error"
    assert "ontology backend unavailable" in result["reason"]


def test_a_failing_leg_is_isolated_and_reported(tmp_path, monkeypatch):
    """`sync_package_install` never crashes on one bad leg — each leg function

    already catches its own exceptions (see the test above), so the dict
    literal building ``legs`` always completes; this asserts the aggregate
    report surfaces the failure without blocking the other legs.
    """
    _write_manifest(
        tmp_path,
        {
            "generated_at": "2026-07-12T00:00:00Z",
            "prompts": {"demo-pkg": 1},
            "ontologies": {},
        },
    )

    monkeypatch.setattr(pii, "_ingest_prompts_leg", lambda: {"status": "ok"})
    monkeypatch.setattr(
        pii,
        "_ingest_ontologies_leg",
        lambda engine: {"status": "error", "reason": "boom"},
    )
    monkeypatch.setattr(pii, "_ingest_skills_leg", lambda engine: {"status": "ok"})

    engine = _FakeEngine()
    result = pii.sync_package_install(engine, mode="delta")
    assert result["status"] == "ok"
    assert result["legs"]["ontologies"]["status"] == "error"
    assert result["failed_legs"] == ["ontologies"]
    # the other two legs still ran despite the ontology leg failing
    assert result["legs"]["prompts"]["status"] == "ok"
    assert result["legs"]["skills"]["status"] == "ok"


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
