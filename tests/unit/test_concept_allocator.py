"""Tests for canonical OKF-CIS concept-ID reservation."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import pytest
import yaml

from agent_utilities.governance import concept_allocator as ca


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "agent_utilities").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "concepts.yaml").write_text(
        yaml.safe_dump(
            {
                "concepts": [
                    {"id": "AU-KG.compute.registered-feature", "pillar": "AU-KG"},
                ]
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_reserve_exact_current_id_and_persist_only_references(repo: Path) -> None:
    rec = ca.reserve_concept_id(
        "AU-KG.compute.new-feature",
        session_id="operator-name",
        design_doc="/private/workspace/design.md",
        repo_root=repo,
    )
    assert rec["id"] == "AU-KG.compute.new-feature"
    assert rec["slug"] == "AU"
    assert rec["pillar"] == "KG"
    assert rec["domain"] == "compute"
    assert rec["session_ref"].startswith("pref_concept_session_")
    assert rec["design_ref"].startswith("pref_design_doc_")
    persisted = ca.ledger_path(repo).read_text(encoding="utf-8")
    assert "operator-name" not in persisted
    assert "/private/workspace" not in persisted


@pytest.mark.parametrize(
    "concept_id",
    ["KG-2.101", "AU-KG.compute", "AUX-KG.compute.feature", "AU-KG.unknown.feature"],
)
def test_noncanonical_or_unregistered_domain_is_rejected(
    repo: Path, concept_id: str
) -> None:
    with pytest.raises(ValueError):
        ca.reserve_concept_id(concept_id, session_id="s", repo_root=repo)


def test_registry_and_code_ids_cannot_be_reserved(repo: Path) -> None:
    (repo / "agent_utilities" / "feature.py").write_text(
        "# CONCEPT:AU-KG.compute.code-feature\n", encoding="utf-8"
    )
    for concept_id in (
        "AU-KG.compute.registered-feature",
        "AU-KG.compute.code-feature",
    ):
        with pytest.raises(ValueError, match="already registered or reserved"):
            ca.reserve_concept_id(concept_id, session_id="s", repo_root=repo)


def test_ledger_rejects_raw_identity_and_retired_fields(repo: Path) -> None:
    ca.ledger_path(repo).write_text(
        "- {id: AU-KG.compute.old-feature, slug: AU, pillar: KG, domain: compute, "
        "session: local-operator, reserved_at: '2026-01-01T00:00:00+00:00', "
        "expires_at: '2026-01-02T00:00:00+00:00', status: reserved}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid fields"):
        ca.read_ledger(repo)


def test_open_reservation_blocks_duplicate_and_release_frees_it(repo: Path) -> None:
    concept_id = "AU-KG.compute.claimed-feature"
    ca.reserve_concept_id(concept_id, session_id="s1", repo_root=repo)
    with pytest.raises(ValueError, match="already registered or reserved"):
        ca.reserve_concept_id(concept_id, session_id="s2", repo_root=repo)
    assert ca.release_concept_id(concept_id, repo_root=repo) is True
    assert (
        ca.reserve_concept_id(concept_id, session_id="s3", repo_root=repo)["id"]
        == concept_id
    )


def test_reconcile_marks_landed(repo: Path) -> None:
    concept_id = "AU-KG.compute.landed-feature"
    ca.reserve_concept_id(concept_id, session_id="s", repo_root=repo)
    (repo / "agent_utilities" / "feature.py").write_text(
        f"# CONCEPT:{concept_id}\n", encoding="utf-8"
    )
    assert ca.reconcile(repo_root=repo)["landed"] == [concept_id]
    assert [r["id"] for r in ca.list_reservations(repo_root=repo, status="landed")] == [
        concept_id
    ]


def test_reconcile_expires_stale_claim(repo: Path) -> None:
    concept_id = "AU-KG.compute.expired-feature"
    ca.reserve_concept_id(
        concept_id, session_id="s", repo_root=repo, ttl_seconds=-1
    )
    assert ca.reconcile(repo_root=repo)["expired"] == [concept_id]
    assert (
        ca.reserve_concept_id(concept_id, session_id="s2", repo_root=repo)["id"]
        == concept_id
    )


def _reserve_worker(repo_str: str, concept_id: str, queue: object) -> None:
    from agent_utilities.governance import concept_allocator as allocator

    try:
        allocator.reserve_concept_id(
            concept_id, session_id="race", repo_root=Path(repo_str)
        )
    except ValueError:
        queue.put("denied")  # type: ignore[attr-defined]
    else:
        queue.put("reserved")  # type: ignore[attr-defined]


def test_concurrent_duplicate_claim_has_one_winner(repo: Path) -> None:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    concept_id = "AU-KG.compute.concurrent-feature"
    processes = [
        ctx.Process(target=_reserve_worker, args=(str(repo), concept_id, queue))
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0
    assert sorted((queue.get(timeout=5), queue.get(timeout=5))) == [
        "denied",
        "reserved",
    ]
    assert [r["id"] for r in ca.read_ledger(repo)] == [concept_id]
