"""Characterization tests for ``ARASeal._l1`` (CX-AU-09).

CCN 15 at time of writing (``agent_utilities/knowledge_graph/research/ara/seal.py``).
``_l1`` is private, so per the CX-AU program's convention (see CX-AU-04's dispatch
brief: a characterization test for an extracted private helper is out of scope and
would itself become a ``test_only_symbols`` finding) it is exercised only through the
public entry point ``ARASeal.review(artifact, level="L1")``, which calls ``_l1``
directly and folds its violations into the returned ``SealReport`` unchanged.

Per the two-commit discipline, this file must be added and pass GREEN against the
UNMODIFIED ``seal.py`` before any refactor commit, and must not change during the
refactor commit that follows.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.research.ara.artifact import (
    Claim,
    CodeSpec,
    Evidence,
    ResearchArtifact,
)
from agent_utilities.knowledge_graph.research.ara.seal import ARASeal


def _base_artifact(**overrides) -> ResearchArtifact:
    defaults = dict(
        article_id="p1",
        title="Paper",
        source_ref="article:p1",
        evidence=[Evidence(id="evidence:p1:0", content="obs")],
        code_specs=[CodeSpec(id="code_spec:p1:0", summary="impl")],
        claims=[
            Claim(
                id="claim:p1:0",
                statement="claim text",
                evidence_ids=["evidence:p1:0"],
                code_spec_ids=["code_spec:p1:0"],
            )
        ],
    )
    defaults.update(overrides)
    return ResearchArtifact(**defaults)


def test_l1_passes_for_fully_grounded_conformant_artifact() -> None:
    report = ARASeal(None).review(_base_artifact(), level="L1")
    assert report.passed is True
    assert report.violations == []


def test_l1_flags_dangling_intra_artifact_evidence_ref() -> None:
    art = _base_artifact(
        claims=[
            Claim(
                id="claim:p1:0",
                statement="x",
                evidence_ids=["evidence:p1:missing"],
            )
        ]
    )
    report = ARASeal(None).review(art, level="L1")
    assert report.passed is False
    codes = [v.code for v in report.violations]
    assert "dangling_evidence_ref" in codes
    viol = next(v for v in report.violations if v.code == "dangling_evidence_ref")
    assert viol.level == "L1"
    assert viol.focus == "claim:p1:0"
    assert "evidence:p1:missing" in viol.message


def test_l1_allows_ecosystem_evidence_ref_outside_the_artifact() -> None:
    # OBSERVED: a claim can reference an evidence id that does NOT start with
    # "evidence:{article_id}" (an ecosystem grounding, e.g. a node id from
    # elsewhere in the graph) without being flagged dangling -- only intra-
    # artifact evidence refs are checked for existence.
    art = _base_artifact(
        claims=[
            Claim(
                id="claim:p1:0",
                statement="x",
                evidence_ids=["some_other_ecosystem_node_id"],
            )
        ]
    )
    report = ARASeal(None).review(art, level="L1")
    codes = [v.code for v in report.violations]
    assert "dangling_evidence_ref" not in codes


def test_l1_flags_dangling_code_ref() -> None:
    art = _base_artifact()
    art.claims[0].code_spec_ids = ["code_spec:p1:missing"]
    report = ARASeal(None).review(art, level="L1")
    assert report.passed is False
    viol = next(v for v in report.violations if v.code == "dangling_code_ref")
    assert viol.level == "L1"
    assert viol.focus == "claim:p1:0"
    assert "code_spec:p1:missing" in viol.message


def test_l1_flags_ungrounded_claim_as_not_conformant() -> None:
    # OBSERVED: conformance is gated on evidence_ids specifically -- a claim
    # with code_spec_ids set but evidence_ids empty is still "ungrounded"
    # (grounded_in is what VerifiableClaim requires, not implemented_by).
    art = _base_artifact(
        claims=[
            Claim(
                id="claim:p1:0",
                statement="ungrounded",
                evidence_ids=[],
                code_spec_ids=["code_spec:p1:0"],
            )
        ]
    )
    report = ARASeal(None).review(art, level="L1")
    assert report.passed is False
    viol = next(v for v in report.violations if v.code == "claim_not_conformant")
    assert viol.level == "L1"
    assert viol.focus == "claim:p1:0"
    assert "ungrounded" in viol.message


def test_l1_flags_artifact_missing_provenance_as_not_conformant() -> None:
    art = _base_artifact(source_ref="")
    report = ARASeal(None).review(art, level="L1")
    assert report.passed is False
    viol = next(v for v in report.violations if v.code == "artifact_not_conformant")
    assert viol.level == "L1"
    assert viol.focus == art.node_id


def test_l1_accumulates_multiple_violations_in_source_order() -> None:
    # OBSERVED order: per-claim reference checks (evidence then code) run
    # before the per-claim conformance check, before the artifact-level check.
    # OBSERVED (subtle): conformance only requires a claim to DECLARE a
    # grounded_in link -- a non-empty evidence_ids list satisfies it even when
    # the referenced evidence id is dangling, so this claim is flagged for the
    # dangling refs but NOT for claim_not_conformant.
    art = _base_artifact(
        source_ref="",
        claims=[
            Claim(
                id="claim:p1:0",
                statement="x",
                evidence_ids=["evidence:p1:missing"],
                code_spec_ids=["code_spec:p1:missing"],
            )
        ],
    )
    report = ARASeal(None).review(art, level="L1")
    codes = [v.code for v in report.violations]
    assert codes == [
        "dangling_evidence_ref",
        "dangling_code_ref",
        "artifact_not_conformant",
    ]
