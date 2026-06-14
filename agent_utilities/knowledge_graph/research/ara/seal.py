#!/usr/bin/python
from __future__ import annotations

"""ARA Seal — OWL/SHACL-grounded review of a research artifact (CONCEPT:KG-2.80).

The paper's ARA-Native Seal reviews an artifact at three escalating levels, withholding
``/evidence`` so reviewers (and agents) cannot fabricate-to-pass. We ground each level
in machinery the ecosystem already has:

- **L1 structural** — cross-layer references resolve (every claim's evidence/code ids
  exist in the artifact) and the artifact + each claim **conform to their ontology
  interfaces** (``ResearchArtifactShape`` / ``VerifiableClaim``, KG-2.38) — i.e. every
  claim is actually grounded. This is the gate the Loop uses before promotion.
- **L2 rigor** — an optional rigor judge (inject ``judge_fn``; the ConceptMatcher LLM
  judge / reliability scorers slot in here) plus a claim-confidence floor.
- **L3 reproducibility** — ``/evidence`` is **withheld via markings** (KG-2.46): the
  certificate records the evidence node ids as restricted; the actual sandbox/regression
  re-run gate is the Loop's existing output gate (AHE-3.14/3.18), wired by the caller.

On pass the Seal emits a signed ``seal_certificate`` node ``--certifies-->`` the
artifact, so the verdict is itself reason-able and queryable.

Concept: ara-seal
"""

import hashlib
import logging
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from ....models.knowledge_graph import RegistryEdgeType
from ...ontology.interfaces import DEFAULT_INTERFACE_REGISTRY, InterfaceRegistry

logger = logging.getLogger(__name__)

LEVELS = ("L1", "L2", "L3")

#: a rigor judge: given (claim_statement, evidence_texts) → confidence in [0,1].
JudgeFn = Callable[[str, list[str]], float]


class SealViolation(BaseModel):
    """One unmet Seal requirement."""

    level: str
    code: str
    focus: str
    message: str


class SealReport(BaseModel):
    """The verdict of one Seal review (CONCEPT:KG-2.80)."""

    article_id: str
    artifact_node: str
    level: str = "L1"
    passed: bool = False
    violations: list[SealViolation] = Field(default_factory=list)
    #: evidence node ids withheld under markings at L3.
    withheld_evidence: list[str] = Field(default_factory=list)
    certificate_id: str = ""
    scores: dict[str, float] = Field(default_factory=dict)


class ARASeal:
    """Review an :class:`ResearchArtifact` at L1/L2/L3 and certify it."""

    def __init__(
        self,
        engine: Any = None,
        *,
        interface_registry: InterfaceRegistry | None = None,
        judge_fn: JudgeFn | None = None,
        confidence_floor: float = 0.0,
    ) -> None:
        self._engine = engine
        self._registry = interface_registry or DEFAULT_INTERFACE_REGISTRY
        self._judge_fn = judge_fn
        self._confidence_floor = confidence_floor

    # -- public ----------------------------------------------------------- #
    def review(
        self, artifact: Any, *, level: str = "L1", materialize: bool = True
    ) -> SealReport:
        """Run the Seal up to ``level`` and (on pass) certify the artifact."""
        if level not in LEVELS:
            level = "L1"
        report = SealReport(
            article_id=artifact.article_id,
            artifact_node=artifact.node_id,
            level=level,
        )
        report.violations.extend(self._l1(artifact))
        if level in ("L2", "L3"):
            report.violations.extend(self._l2(artifact, report))
        if level == "L3":
            report.withheld_evidence = [e.id for e in artifact.evidence]
            report.violations.extend(self._l3(artifact))
        report.passed = not report.violations
        if report.passed and materialize and self._engine is not None:
            report.certificate_id = self._certify(artifact, report)
        return report

    # -- L1 structural + interface conformance ---------------------------- #
    def _l1(self, artifact: Any) -> list[SealViolation]:
        viols: list[SealViolation] = []
        evidence_ids = {e.id for e in artifact.evidence}
        code_ids = {c.id for c in artifact.code_specs}

        # cross-layer reference resolution
        for cl in artifact.claims:
            for ref in cl.evidence_ids:
                # ecosystem groundings (non-artifact ids) are allowed; only dangling
                # intra-artifact evidence refs are violations.
                if ref.startswith(f"evidence:{artifact.article_id}") and (
                    ref not in evidence_ids
                ):
                    viols.append(
                        SealViolation(
                            level="L1",
                            code="dangling_evidence_ref",
                            focus=cl.id,
                            message=f"claim references missing evidence {ref!r}",
                        )
                    )
            for ref in cl.code_spec_ids:
                if ref not in code_ids:
                    viols.append(
                        SealViolation(
                            level="L1",
                            code="dangling_code_ref",
                            focus=cl.id,
                            message=f"claim references missing code spec {ref!r}",
                        )
                    )

        # interface conformance — every claim grounded (VerifiableClaim), artifact
        # well-formed (ResearchArtifactShape). Uses the actual edge set as link view.
        for cl in artifact.claims:
            link_types = []
            if cl.evidence_ids:
                link_types.append(RegistryEdgeType.GROUNDED_IN.value)
            if cl.code_spec_ids:
                link_types.append(RegistryEdgeType.IMPLEMENTED_BY.value)
            obj = {"statement": cl.statement, "link_types": link_types}
            if not self._registry.conforms(obj, "VerifiableClaim"):
                viols.append(
                    SealViolation(
                        level="L1",
                        code="claim_not_conformant",
                        focus=cl.id,
                        message="claim does not conform to VerifiableClaim "
                        "(ungrounded — no grounded_in link)",
                    )
                )
        art_links = [RegistryEdgeType.CONTAINS.value]
        if artifact.source_ref:
            art_links.append(RegistryEdgeType.WAS_DERIVED_FROM.value)
        art_obj = {"timestamp": artifact.timestamp, "link_types": art_links}
        if not self._registry.conforms(art_obj, "ResearchArtifactShape"):
            viols.append(
                SealViolation(
                    level="L1",
                    code="artifact_not_conformant",
                    focus=artifact.node_id,
                    message="artifact does not conform to ResearchArtifactShape "
                    "(missing provenance/contains)",
                )
            )
        return viols

    # -- L2 rigor --------------------------------------------------------- #
    def _l2(self, artifact: Any, report: SealReport) -> list[SealViolation]:
        viols: list[SealViolation] = []
        confidences: list[float] = []
        for cl in artifact.claims:
            conf = float(cl.confidence)
            if self._judge_fn is not None:
                ev_texts = [
                    e.content for e in artifact.evidence if e.id in cl.evidence_ids
                ]
                try:
                    conf = float(self._judge_fn(cl.statement, ev_texts))
                except Exception as e:  # noqa: BLE001
                    logger.debug("rigor judge failed for %s: %s", cl.id, e)
            confidences.append(conf)
            if conf < self._confidence_floor:
                viols.append(
                    SealViolation(
                        level="L2",
                        code="below_confidence_floor",
                        focus=cl.id,
                        message=f"claim confidence {conf:.2f} < floor "
                        f"{self._confidence_floor:.2f}",
                    )
                )
        report.scores["mean_confidence"] = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )
        return viols

    # -- L3 reproducibility (evidence withheld) --------------------------- #
    def _l3(self, artifact: Any) -> list[SealViolation]:
        # The anti-fabrication guarantee: evidence is recorded as restricted on the
        # certificate (markings, KG-2.46). The executable re-run gate is the Loop's
        # existing regression/sandbox gate, invoked by the caller; absence of any
        # evidence at all is the one thing L3 can structurally flag here.
        if not artifact.evidence:
            return [
                SealViolation(
                    level="L3",
                    code="no_evidence_to_reproduce",
                    focus=artifact.node_id,
                    message="L3 requires at least one evidence unit to reproduce",
                )
            ]
        return []

    # -- certificate ------------------------------------------------------ #
    def _certify(self, artifact: Any, report: SealReport) -> str:
        payload = f"{artifact.node_id}|{report.level}|{len(artifact.claims)}"
        digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
        cert_id = f"seal_certificate:{artifact.article_id}:{report.level}"
        props = {
            "name": f"Seal {report.level} for {artifact.article_id}",
            "level": report.level,
            "signature": digest,
            "passed": True,
            "withheld_evidence": list(report.withheld_evidence),
            "markings": ["restricted"] if report.withheld_evidence else [],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        try:
            self._engine.add_node(cert_id, "seal_certificate", properties=props)
            self._engine.add_edge(cert_id, artifact.node_id, "certifies")
        except Exception as e:  # noqa: BLE001 — best-effort persist
            logger.debug("seal certificate persist failed: %s", e)
            return ""
        return cert_id


__all__ = ["ARASeal", "SealReport", "SealViolation", "JudgeFn", "LEVELS"]
