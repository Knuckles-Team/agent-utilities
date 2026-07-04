"""Unit tests for the concept-ID allocator (CONCEPT:AU-OS.governance.atomic-concept-id-reservation).

Covers the next-id math, the ledger round-trip, reconcile transitions, and the
core proof: two concurrent reservers on the same ledger get distinct ids.
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import pytest
import yaml

from agent_utilities.governance import concept_allocator as ca


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A throwaway repo root with an empty code tree + a small registry."""
    (tmp_path / "agent_utilities").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "concepts.yaml").write_text(
        yaml.safe_dump(
            {
                "concepts": [
                    {"id": "EG-KG.compute.type-scope-resolved-call", "pillar": "EG-KG.compute.backend"},
                    {"id": "EG-KG.compute.model-free-similar-code", "pillar": "EG-KG.compute.backend"},
                    {"id": "KEY-003", "pillar": "KEY"},
                ]
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


# --------------------------------------------------------------------------- #
# next_id math
# --------------------------------------------------------------------------- #
def test_next_id_pillar_dotted():
    taken = {"EG-KG.compute.type-scope-resolved-call", "EG-KG.compute.model-free-similar-code", "EG-KG.domains.forensic-accounting-kernels"}
    assert ca.next_id("EG-KG.compute.backend", taken) == "AU-KG.compute.http-route-graph"


def test_next_id_letter_suffix_does_not_inflate():
    # KG-2.20g must count as sub-index 20, not block .21.
    assert ca.next_id("EG-KG.compute.backend", {"EG-KG.domains.forensic-accounting-kernels"}) == "AU-KG.memory.working-set-eviction"


def test_next_id_package_zero_padded():
    assert ca.next_id("KEY", {"AU-KG.ontology.package-scoped-concept", "KEY-003"}) == "KEY-004"


def test_next_id_fresh_namespace():
    assert ca.next_id("ML-9", set()) == "ML-9.1"
    assert ca.next_id("ZZZ", set()) == "ZZZ-001"


def test_unknown_namespace_rejected():
    with pytest.raises(ValueError):
        ca.next_id("kg-2", set())  # lowercase is not a valid namespace


# --------------------------------------------------------------------------- #
# reserve + ledger round-trip
# --------------------------------------------------------------------------- #
def test_reserve_unions_registry_and_code(repo: Path):
    # Registry already has KG-2.101; a code marker pushes the floor higher.
    (repo / "agent_utilities" / "mod.py").write_text(
        "# CONCEPT:AU-KG.ingest.agent-utilities-checkout something\n", encoding="utf-8"
    )
    rec = ca.reserve_concept_id("EG-KG.compute.backend", session_id="s1", repo_root=repo)
    assert rec["id"] == "AU-KG.compute.gitlab-api-gitlab-atlassian"
    assert rec["status"] == "reserved"
    # Persisted to the ledger, one line per reservation.
    text = ca.ledger_path(repo).read_text(encoding="utf-8")
    assert "AU-KG.compute.gitlab-api-gitlab-atlassian" in text
    body = [ln for ln in text.splitlines() if ln.startswith("- ")]
    assert len(body) == 1


def test_open_reservation_is_counted(repo: Path):
    first = ca.reserve_concept_id("EG-KG.compute.backend", session_id="s1", repo_root=repo)
    second = ca.reserve_concept_id("EG-KG.compute.backend", session_id="s2", repo_root=repo)
    assert first["id"] == "AU-KG.compute.http-route-graph"
    assert second["id"] == "AU-KG.enrichment.read-them-here-so"  # the open reservation was counted


def test_release_frees_the_id(repo: Path):
    rec = ca.reserve_concept_id("KEY", session_id="s1", repo_root=repo)
    assert rec["id"] == "KEY-004"
    assert ca.release_concept_id("KEY-004", repo_root=repo) is True
    # Now the next reservation reuses the freed slot.
    assert (
        ca.reserve_concept_id("KEY", session_id="s2", repo_root=repo)["id"] == "KEY-004"
    )


# --------------------------------------------------------------------------- #
# reconcile transitions
# --------------------------------------------------------------------------- #
def test_reconcile_marks_landed(repo: Path):
    rec = ca.reserve_concept_id("EG-KG.compute.backend", session_id="s1", repo_root=repo)
    # Author lands the marker in code.
    (repo / "agent_utilities" / "feature.py").write_text(
        f"# CONCEPT:{rec['id']} the feature\n", encoding="utf-8"
    )
    out = ca.reconcile(repo_root=repo)
    assert rec["id"] in out["landed"]
    landed = ca.list_reservations(repo_root=repo, status="landed")
    assert [r["id"] for r in landed] == [rec["id"]]


def test_reconcile_expires_stale(repo: Path):
    ca.reserve_concept_id("EG-KG.compute.backend", session_id="s1", repo_root=repo, ttl_seconds=-1)
    out = ca.reconcile(repo_root=repo)
    assert out["expired"] == ["AU-KG.compute.http-route-graph"]
    # An expired reservation no longer holds the slot.
    assert (
        ca.reserve_concept_id("EG-KG.compute.backend", session_id="s2", repo_root=repo)["id"]
        == "AU-KG.compute.http-route-graph"
    )


# --------------------------------------------------------------------------- #
# concurrency proof — the whole point
# --------------------------------------------------------------------------- #
def _reserve_worker(repo_str: str, ns: str, q) -> None:  # pragma: no cover - subprocess
    from agent_utilities.governance import concept_allocator as ca_

    rec = ca_.reserve_concept_id(ns, session_id="race", repo_root=Path(repo_str))
    q.put(rec["id"])


def test_concurrent_reservers_get_distinct_ids(repo: Path):
    """Two processes racing on the same ledger must never mint the same id."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [
        ctx.Process(target=_reserve_worker, args=(str(repo), "EG-KG.compute.backend", q))
        for _ in range(2)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=30)
        assert p.exitcode == 0
    ids = {q.get(timeout=5), q.get(timeout=5)}
    assert ids == {"AU-KG.compute.http-route-graph", "AU-KG.enrichment.read-them-here-so"}  # distinct, contiguous
    # And both survive in the ledger (no overwrite).
    assert len(ca.read_ledger(repo)) == 2
