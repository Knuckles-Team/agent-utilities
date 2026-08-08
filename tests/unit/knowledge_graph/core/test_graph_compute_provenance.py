"""``GraphComputeEngine.explain_provenance_by_ids`` — the row-path projection that
turns the engine's typed ``EvidenceBundle`` into the epistemic row dict
:class:`~agent_utilities.knowledge_graph.core.epistemic_row.EpistemicRow` consumes
(CONCEPT:EG-KB-CURRENCY, Seam 1).

The proof/contradiction seam (O9): the engine computes ``proof_refs``/
``contradiction_refs`` on each ``EvidenceClaim`` (the SAME ids the Arrow
``KnowledgeBatch`` carries), and this projection is where those wire fields are
renamed to the ``proof_ids``/``contradiction_ids`` keys the row path reads. That
rename was previously untested — a silent break here would leave proof/contradiction
ids stranded on the row path even though both sides "have" the columns.
"""

from __future__ import annotations

from typing import Any

from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine


def _bundle_with(
    proof_refs: list[str], contradiction_refs: list[str]
) -> dict[str, Any]:
    """A wire-shaped ``EvidenceBundle`` dict exactly as
    ``client.query.explain_provenance_by_ids`` returns it (one claim)."""
    return {
        "schema_version": "1",
        "bundle_id": "request:1",
        "resolved": True,
        "answer_ref": None,
        "claims": [
            {
                "claim_ref": "claim1",
                "kind": "Claim",
                "score": 0.9,
                "confidence": 0.6,
                "valid_time": {"start_ms": 5, "end_ms": 50},
                "transaction_time": {"start_ms": 1, "end_ms": None},
                "source_refs": ["evidence1"],
                "evidence_locus_refs": [],
                "contradiction_refs": contradiction_refs,
                "proof_refs": proof_refs,
                "policy_labels": ["epistemic:asserted"],
            }
        ],
        "policy_exclusions": [],
        "next_action_refs": [],
    }


def _engine_returning(bundle: dict[str, Any]) -> GraphComputeEngine:
    engine = GraphComputeEngine.__new__(GraphComputeEngine)  # bypass transport __init__

    class _Query:
        def explain_provenance_by_ids(self, ids: list[str]) -> dict[str, Any]:
            return bundle

    class _Client:
        query = _Query()

    engine._client = _Client()  # type: ignore[attr-defined]
    return engine


def test_projects_proof_and_contradiction_refs_to_ids() -> None:
    engine = _engine_returning(
        _bundle_with(proof_refs=["evidence1", "mid"], contradiction_refs=["counter1"])
    )
    rows = engine.explain_provenance_by_ids(["claim1"])

    assert len(rows) == 1
    row = rows[0]
    # The eg wire fields (proof_refs/contradiction_refs) are renamed to the row-path
    # keys EpistemicRow.from_wire reads (proof_ids/contradiction_ids) — the O9 seam.
    assert row["proof_ids"] == ["evidence1", "mid"]
    assert row["contradiction_ids"] == ["counter1"]
    # ...and the rest of the epistemic envelope projects straight through.
    assert row["id"] == "claim1"
    assert row["confidence"] == 0.6
    assert row["valid_time"] == [5, 50]
    assert row["tx_time"] == [1, None]
    assert row["source_refs"] == ["evidence1"]


def test_empty_proof_and_contradiction_refs_stay_empty_not_fabricated() -> None:
    engine = _engine_returning(_bundle_with(proof_refs=[], contradiction_refs=[]))
    row = engine.explain_provenance_by_ids(["claim1"])[0]
    assert row["proof_ids"] == []
    assert row["contradiction_ids"] == []


def test_no_ids_short_circuits_without_a_round_trip() -> None:
    # ids=[] returns [] without touching the client (no explain call at all).
    engine = GraphComputeEngine.__new__(GraphComputeEngine)
    assert engine.explain_provenance_by_ids([]) == []
