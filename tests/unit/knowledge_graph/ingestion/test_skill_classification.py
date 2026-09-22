"""Skill classification write-back (CONCEPT:AU-KG.ingest.skill-classification-writeback).

Covers ``agent_utilities.knowledge_graph.ingestion.skill_classification``:

* an unclassified skill gets classified and the choice persists to its own
  SKILL.md frontmatter when the source tree IS writable (a local/dev
  checkout);
* when it is NOT writable (every deployed profile -- the ``universal-skills``
  NFS export is read-only), the write is refused, a durable override is
  recorded instead, and the result says exactly that -- never a silent
  success;
* a genuinely failed write (neither the file nor the override could be made
  durable) reports ``persisted: False`` with a reason, never success.

EH-345 (2026-09-22) rewrote ``reclassify_skill`` to call EG's
``FleetCatalogClient.set_override``/``.lookup`` instead of the deleted
``fleet_catalog_tables`` SQL primitives. This file (rewritten from its
predecessor, which reused that module's in-memory SQL-catalog fake) uses a
fake ``fleet_catalog`` client implementing the same synchronous method
surface, applying an override at LOOKUP time -- exactly the "no separate
refresh step, the projection applies it on read" contract AU-CUTOVER.md §2.5
documents.
"""

from __future__ import annotations

import os
import stat
import textwrap
from typing import Any

import pytest

from agent_utilities.knowledge_graph.ingestion.skill_classification import (
    SkillClassificationError,
    reclassify_skill,
)

pytestmark = pytest.mark.concept("AU-KG.ingest.skill-classification-writeback")

_SKILL_MD = textwrap.dedent(
    """\
    ---
    name: mystery-skill
    description: A skill whose classification is not yet a known type.
    skill_type: mystery
    tags: [test]
    ---

    # mystery-skill

    Body instructions.
    """
)

_COMPONENT_ID = "mcp:universal-skills/skill/mystery-skill"


def _write_corpus_file(tmp_path, *, name: str = "mystery-skill") -> str:
    """Lay down one SKILL.md under a fresh corpus root; returns the root path."""
    root = tmp_path / "skills"
    skill_dir = root / "misc" / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(_SKILL_MD, encoding="utf-8")
    return str(root)


class _Row:
    def __init__(self, *, skill_type: str) -> None:
        self.skill_type = skill_type


class _FakeFleetCatalog:
    """A fleet catalog whose ``lookup`` reflects the LATEST ``set_override`` --
    modeling the real projection's "applied at read time" contract."""

    def __init__(self, *, component_id: str, name: str, skill_type: str) -> None:
        self.component_id = component_id
        self.name = name
        self._skill_type = skill_type
        self.override_calls: list[Any] = []
        self.lookup_calls: list[list[str]] = []

    def lookup(self, ids: list[str], grant_digests=()):
        from types import SimpleNamespace

        self.lookup_calls.append(list(ids))
        if self.component_id not in ids:
            return SimpleNamespace(rows=[])
        component = SimpleNamespace(id=self.component_id, name=self.name)
        row = SimpleNamespace(component=component, skill_type=self._skill_type)
        entry = SimpleNamespace(kind="skill", row=row)
        return SimpleNamespace(rows=[entry])

    def set_override(self, request: Any):
        from types import SimpleNamespace

        self.override_calls.append(request)
        assert request.component_id == self.component_id
        self._skill_type = request.value["skill_type"]
        return SimpleNamespace(disposition="written")


def _seed_engine(component_id: str, name: str, skill_type: str) -> Any:
    from types import SimpleNamespace

    fc = _FakeFleetCatalog(component_id=component_id, name=name, skill_type=skill_type)
    engine = SimpleNamespace(
        graph_compute=SimpleNamespace(client=SimpleNamespace(fleet_catalog=fc))
    )
    return engine, fc


# ---------------------------------------------------------------------------
# Caller error: invalid skill_type is rejected before any write is attempted
# ---------------------------------------------------------------------------


def test_invalid_skill_type_raises_before_any_write(tmp_path):
    engine, fc = _seed_engine(_COMPONENT_ID, "mystery-skill", "mystery")
    root = _write_corpus_file(tmp_path)

    with pytest.raises(SkillClassificationError):
        reclassify_skill(
            engine,
            skill_id=_COMPONENT_ID,
            skill_type="not-a-real-type",
            principal="tester",
            root=root,
        )

    # No override was attempted for the rejected request.
    assert fc.override_calls == []


def test_unknown_skill_id_reports_failure_not_success(tmp_path):
    engine, fc = _seed_engine(_COMPONENT_ID, "mystery-skill", "mystery")
    root = _write_corpus_file(tmp_path)

    result = reclassify_skill(
        engine,
        skill_id="mcp:universal-skills/skill/does-not-exist",
        skill_type="skill",
        principal="tester",
        root=root,
    )

    assert result["persisted"] is False
    assert result["reason"]


# ---------------------------------------------------------------------------
# Writable source tree: the real fix lands on disk AND the override is set
# ---------------------------------------------------------------------------


def test_writable_source_file_is_persisted_and_verified(tmp_path):
    # The override write constructs a real FleetOverrideSetRequest -- skip
    # cleanly on a venv whose installed epistemic_graph predates
    # generated/fleet_catalog.py (see registry_api's test file for the same
    # pattern/reason).
    pytest.importorskip("epistemic_graph.generated.fleet_catalog")
    engine, fc = _seed_engine(_COMPONENT_ID, "mystery-skill", "mystery")
    root = _write_corpus_file(tmp_path)

    result = reclassify_skill(
        engine,
        skill_id=_COMPONENT_ID,
        skill_type="workflow",
        principal="tester",
        root=root,
    )

    assert result["persisted"] is True
    assert result["persisted_to_source_file"] is True
    assert result["persisted_as_durable_override"] is True
    assert result["skill_type"] == "workflow"
    assert result["classification"] == "workflow"
    assert result["catalog_refreshed"] is True
    assert result["reason"] is None

    # The file itself was actually rewritten -- not just claimed.
    written = (tmp_path / "skills" / "misc" / "mystery-skill" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    assert "skill_type: workflow" in written
    assert "skill_type: mystery" not in written
    # The rest of the frontmatter/body survived untouched.
    assert "name: mystery-skill" in written
    assert "Body instructions." in written

    # The override was set through the typed EG call.
    assert len(fc.override_calls) == 1
    assert fc.override_calls[0].value == {
        "field": "skill_type",
        "skill_type": "workflow",
    }


# ---------------------------------------------------------------------------
# Read-only source tree (the real deployed shape): override-only persistence,
# never silently reported as a full source-file write.
# ---------------------------------------------------------------------------


def test_read_only_source_tree_falls_back_to_durable_override(tmp_path):
    pytest.importorskip("epistemic_graph.generated.fleet_catalog")
    engine, fc = _seed_engine(_COMPONENT_ID, "mystery-skill", "mystery")
    root = _write_corpus_file(tmp_path)
    skill_dir = tmp_path / "skills" / "misc" / "mystery-skill"

    # Simulate the real deployed shape: the containing directory is not
    # writable by this process (an NFS read-only export), which blocks even
    # creating the atomic-write temp file -- exactly what a real read-only
    # mount does, verified by an ACTUAL write attempt rather than trusted
    # from a permission-bit/mount-flag check (mount flags can lie).
    original_mode = skill_dir.stat().st_mode
    os.chmod(skill_dir, stat.S_IRUSR | stat.S_IXUSR)
    try:
        result = reclassify_skill(
            engine,
            skill_id=_COMPONENT_ID,
            skill_type="skill",
            principal="tester",
            root=root,
        )
    finally:
        os.chmod(skill_dir, original_mode)

    assert result["persisted"] is True
    assert result["persisted_to_source_file"] is False
    assert result["persisted_as_durable_override"] is True
    assert result["reason"] is None  # persisted overall -- no failure to report
    assert result["catalog_refreshed"] is True

    # The source file was NOT modified.
    unwritten = (tmp_path / "skills" / "misc" / "mystery-skill" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    assert "skill_type: mystery" in unwritten

    # But a fresh lookup now reflects the operator's chosen type -- the
    # projection applies the override at read time, no separate refresh call.
    assert fc._skill_type == "skill"


def test_no_matching_skill_file_still_falls_back_to_override(tmp_path):
    """The catalog has a row, but no SKILL.md can be found for it (e.g. an
    mcp-harvested provenance the corpus scan doesn't cover) -- override-only
    persistence still succeeds and is reported accurately."""
    pytest.importorskip("epistemic_graph.generated.fleet_catalog")
    engine, fc = _seed_engine(_COMPONENT_ID, "mystery-skill", "mystery")
    empty_root = str(tmp_path / "empty-corpus")
    os.makedirs(empty_root)

    result = reclassify_skill(
        engine,
        skill_id=_COMPONENT_ID,
        skill_type="graph",
        principal="tester",
        root=empty_root,
    )

    assert result["persisted"] is True
    assert result["persisted_to_source_file"] is False
    assert result["persisted_as_durable_override"] is True
    assert result["reason"] is None


def test_override_write_failure_is_reported_not_silently_succeeded(tmp_path):
    """When EG's fleet-catalog surface isn't wired at all (no client), the
    override write fails silently-nothing-happened, and the overall result
    must reflect that (only ``persisted_to_source_file`` can still save it)."""
    from types import SimpleNamespace

    engine = SimpleNamespace(
        graph_compute=SimpleNamespace(client=SimpleNamespace(fleet_catalog=None))
    )
    root = _write_corpus_file(tmp_path)

    result = reclassify_skill(
        engine,
        skill_id=_COMPONENT_ID,
        skill_type="skill",
        principal="tester",
        root=root,
    )

    # No catalog row was ever found (lookup unavailable), so this reports the
    # same "unknown skill_id" failure as a genuine miss -- never success.
    assert result["persisted"] is False
