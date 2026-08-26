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

Reuses ``test_fleet_catalog_tables``'s in-memory SQL-catalog fake rather than
re-implementing CAS/tenant-scoping emulation -- ``reclassify_skill`` is a
thin orchestrator over exactly the primitives that module's fake already
proves correct (``write_skill_row``, ``write_skill_classification_override``,
``get_skill_row``).
"""

from __future__ import annotations

import os
import stat
import textwrap

import pytest

from agent_utilities.knowledge_graph.core.fleet_catalog_tables import (
    TenantLocalDiscoveryBinding,
    write_skill_row,
)
from agent_utilities.knowledge_graph.core.session import use_session
from agent_utilities.knowledge_graph.ingestion.skill_classification import (
    SkillClassificationError,
    reclassify_skill,
)
from agent_utilities.security.brain_context import use_actor
from tests.unit.knowledge_graph.test_fleet_catalog_tables import (
    _FakeEngine,
    _one_row,
    _session,
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


def _write_corpus_file(tmp_path, *, name: str = "mystery-skill") -> str:
    """Lay down one SKILL.md under a fresh corpus root; returns the root path."""
    root = tmp_path / "skills"
    skill_dir = root / "misc" / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(_SKILL_MD, encoding="utf-8")
    return str(root)


def _seed_catalog_row(eng: _FakeEngine, *, skill_type: str = "mystery") -> str:
    """Seed the skills SQL row an ingester would have written, return its bound id.

    Uses ``TenantLocalDiscoveryBinding`` -- the ONLY binding a locally-sourced
    corpus file (a ``skill``/``workflow``/``graph``/unclassified row; never
    ``mcp_skill``, which is fleet-harvested and excluded from
    ``ALLOWED_SKILL_TYPES``) is ever written under in production
    (``ingest_runnable_skill``/``ingest_agent_skill``). This is what makes
    ``reclassify_skill``'s own re-binding under the same convention land on
    THIS row rather than a fresh one.
    """
    with use_actor(_session("tenant-a").actor), use_session(_session("tenant-a")):
        write_skill_row(
            eng,
            skill_id="skill:mystery-skill",
            name="mystery-skill",
            description="A skill whose classification is not yet a known type.",
            skill_type=skill_type,
            discovery_binding=TenantLocalDiscoveryBinding(tenant_id="tenant-a"),
        )
        return _one_row("skills", "skill:mystery-skill", eng)["id"]


# ---------------------------------------------------------------------------
# Caller error: invalid skill_type is rejected before any write is attempted
# ---------------------------------------------------------------------------


def test_invalid_skill_type_raises_before_any_write(tmp_path):
    eng = _FakeEngine()
    bound_id = _seed_catalog_row(eng)
    root = _write_corpus_file(tmp_path)

    with use_actor(_session("tenant-a").actor), use_session(_session("tenant-a")):
        with pytest.raises(SkillClassificationError):
            reclassify_skill(
                eng,
                skill_id=bound_id,
                skill_type="not-a-real-type",
                principal="tester",
                root=root,
            )

    # No override was written for the rejected request.
    assert not eng.graph_compute.tables.get("skill_classification_overrides")


def test_unknown_skill_id_reports_failure_not_success(tmp_path):
    eng = _FakeEngine()
    _seed_catalog_row(eng)
    root = _write_corpus_file(tmp_path)

    with use_actor(_session("tenant-a").actor), use_session(_session("tenant-a")):
        result = reclassify_skill(
            eng,
            skill_id="skill:does-not-exist__tenant_local",
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
    eng = _FakeEngine()
    bound_id = _seed_catalog_row(eng)
    root = _write_corpus_file(tmp_path)

    with use_actor(_session("tenant-a").actor), use_session(_session("tenant-a")):
        result = reclassify_skill(
            eng,
            skill_id=bound_id,
            skill_type="workflow",
            principal="tester",
            root=root,
        )

    assert result["persisted"] is True
    assert result["persisted_to_source_file"] is True
    assert result["skill_type"] == "workflow"
    assert result["classification"] == "Workflow"
    assert result["catalog_refreshed"] is True
    assert result["reason"] is None

    # The file itself was actually rewritten -- not just claimed.
    written = (
        (tmp_path / "skills" / "misc" / "mystery-skill" / "SKILL.md")
        .read_text(encoding="utf-8")
    )
    assert "skill_type: workflow" in written
    assert "skill_type: mystery" not in written
    # The rest of the frontmatter/body survived untouched.
    assert "name: mystery-skill" in written
    assert "Body instructions." in written

    # The catalog row now reflects the new classification.
    row = _one_row("skills", "skill:mystery-skill", eng)
    assert row["skill_type"] == "workflow"


# ---------------------------------------------------------------------------
# Read-only source tree (the real deployed shape): override-only persistence,
# never silently reported as a full source-file write.
# ---------------------------------------------------------------------------


def test_read_only_source_tree_falls_back_to_durable_override(tmp_path):
    eng = _FakeEngine()
    bound_id = _seed_catalog_row(eng)
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
        with use_actor(_session("tenant-a").actor), use_session(_session("tenant-a")):
            result = reclassify_skill(
                eng,
                skill_id=bound_id,
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
    unwritten = (
        (tmp_path / "skills" / "misc" / "mystery-skill" / "SKILL.md")
        .read_text(encoding="utf-8")
    )
    assert "skill_type: mystery" in unwritten

    # But the catalog row was refreshed to the operator's chosen type...
    row = _one_row("skills", "skill:mystery-skill", eng)
    assert row["skill_type"] == "skill"

    # ...and, critically, the override survives a simulated re-sync that
    # re-derives from the (unwritten, still "mystery") frontmatter.
    with use_actor(_session("tenant-a").actor), use_session(_session("tenant-a")):
        write_skill_row(
            eng,
            skill_id="skill:mystery-skill",
            name="mystery-skill",
            description="A skill whose classification is not yet a known type.",
            skill_type="mystery",
            idempotency_key="simulated-resync",
            discovery_binding=TenantLocalDiscoveryBinding(tenant_id="tenant-a"),
        )
    row = _one_row("skills", "skill:mystery-skill", eng)
    assert row["skill_type"] == "skill"  # override still wins post-resync


def test_no_matching_skill_file_still_falls_back_to_override(tmp_path):
    """The catalog has a row, but no SKILL.md can be found for it (e.g. an
    mcp-harvested provenance the corpus scan doesn't cover) -- override-only
    persistence still succeeds and is reported accurately."""
    eng = _FakeEngine()
    bound_id = _seed_catalog_row(eng)
    empty_root = str(tmp_path / "empty-corpus")
    os.makedirs(empty_root)

    with use_actor(_session("tenant-a").actor), use_session(_session("tenant-a")):
        result = reclassify_skill(
            eng,
            skill_id=bound_id,
            skill_type="graph",
            principal="tester",
            root=empty_root,
        )

    assert result["persisted"] is True
    assert result["persisted_to_source_file"] is False
    assert result["persisted_as_durable_override"] is True
    assert result["reason"] is None
