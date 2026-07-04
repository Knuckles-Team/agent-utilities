#!/usr/bin/python
from __future__ import annotations

"""ARA — the Agent-Native Research Artifact as an OWL-native ontology object (KG-2.80).

arXiv:2604.24658 recasts a paper from narrative prose into a 4-layer, agent-executable
artifact: ``/logic`` (claims), ``/src`` (code specs), ``/trace`` (the exploration DAG
of decisions/dead-ends/pivots) and ``/evidence`` (raw outputs). We make those layers
**first-class nodes and typed object-properties in the one ecosystem ontology** rather
than opaque files, so OWL/RDF reasoning extrapolates over them — chaining a claim to the
ecosystem code/services that substantiate it (``claim -grounded_in-> evidence``,
``claim -implemented_by-> code_spec``), and surfacing the evidence→claim ``supports``
inverse and transitive grounding automatically (see :mod:`...core.owl_bridge`).

This module is the *runtime artifact*: in-memory dataclasses plus
:meth:`ResearchArtifact.to_graph_payload` / :meth:`ResearchArtifact.materialize`, which
emit the nodes + forensic edges the :class:`OntologyReasoningDriver` then reasons over.
The matching ontology **interfaces** (``ResearchArtifactShape`` / ``VerifiableClaim``),
**typed links** (``artifact_contains_claim`` / ``grounds`` / ``implements_claim``) and
**promotable node/edge types** are registered natively in
:mod:`...ontology.interfaces`, :mod:`...ontology.links` and ``owl_bridge`` — always on,
no facade. Populate from :class:`...extraction.fact_extractor` output or the legacy
:class:`...adaptation.research_artifacts.ResearchArtifact`.

Concept: research-artifact
"""

import logging
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from ....models.knowledge_graph import RegistryEdgeType, RegistryNodeType

logger = logging.getLogger(__name__)

ExplorationKind = Literal[
    "question", "decision", "experiment", "dead_end", "pivot", "result"
]


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _slug(text: str, *, limit: int = 80) -> str:
    out = []
    for ch in (text or "").lower():
        out.append(ch if ch.isalnum() else "-")
    s = "".join(out).strip("-")
    while "--" in s:
        s = s.replace("--", "-")
    return (s[:limit] or "x").rstrip("-")


# ── /evidence, /src, /logic, /trace layer units ─────────────────────────────


class Evidence(BaseModel):
    """A raw-output evidence unit (ARA ``/evidence`` layer)."""

    id: str
    content: str = ""
    #: provenance pointer to the raw artifact this evidence was derived from.
    source_ref: str = ""
    kind: str = "observation"


class CodeSpec(BaseModel):
    """An executable code specification (ARA ``/src`` layer)."""

    id: str
    summary: str = ""
    language: str = ""
    symbol: str = ""
    path: str = ""


class Claim(BaseModel):
    """A research claim (ARA ``/logic`` layer) — the unit reasoning verifies."""

    id: str
    statement: str
    claim_type: str = "contribution"
    confidence: float = 0.0
    #: evidence units that ground this claim (→ ``grounded_in`` edges).
    evidence_ids: list[str] = Field(default_factory=list)
    #: code specs that implement this claim (→ ``implemented_by`` edges).
    code_spec_ids: list[str] = Field(default_factory=list)


class ExplorationNode(BaseModel):
    """A node in the research exploration DAG (ARA ``/trace`` layer).

    ``kind`` distinguishes question/decision/experiment/result from the two the
    paper highlights as the high-signal forensic markers: ``dead_end`` and
    ``pivot``. The dedicated producer (A2) populates these from failure clustering
    and ConceptMatcher rejections; A1 carries the type + edges so they reason.
    """

    id: str
    kind: ExplorationKind = "decision"
    text: str = ""
    #: the prior exploration node this one branched from, if any.
    parent_id: str = ""


# ── the artifact envelope ────────────────────────────────────────────────────


class ResearchArtifact(BaseModel):
    """The OWL-native 4-layer Agent-Native Research Artifact (CONCEPT:AU-KG.ontology.verified-by-implemented-by)."""

    article_id: str
    title: str
    summary: str = ""
    claims: list[Claim] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    code_specs: list[CodeSpec] = Field(default_factory=list)
    exploration: list[ExplorationNode] = Field(default_factory=list)
    authors: list[str] = Field(default_factory=list)
    source_url: str = ""
    #: provenance node id (raw paper/repo) for the HasProvenance shape.
    source_ref: str = ""
    timestamp: str = Field(default_factory=_now_iso)

    # -- identity ---------------------------------------------------------- #
    @property
    def node_id(self) -> str:
        return f"research_artifact:{self.article_id}"

    # -- graph projection -------------------------------------------------- #
    def to_graph_payload(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Project the artifact into ``(nodes, edges)`` for promotion/reasoning.

        Nodes carry the ontology node types (``research_artifact``/``claim``/
        ``evidence``/``code_spec``/``exploration_node``); edges are the forensic
        bindings the reasoner chains: ``contains`` (envelope→claims), ``grounded_in``
        (claim→evidence, transitive + ``supports`` inverse), ``implemented_by``
        (claim→code), ``was_derived_from`` (provenance) and the ``/trace`` DAG
        (``pivoted_from`` / ``reached_dead_end`` / ``contains``).
        """
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        aid = self.node_id

        nodes.append(
            {
                "id": aid,
                "type": RegistryNodeType.RESEARCH_ARTIFACT.value,
                "properties": {
                    "name": self.title,
                    "title": self.title,
                    "summary": self.summary,
                    "authors": list(self.authors),
                    "source_url": self.source_url,
                    "timestamp": self.timestamp,
                },
            }
        )
        # provenance: artifact -was_derived_from-> raw source (HasProvenance shape)
        if self.source_ref:
            edges.append(
                self._edge(aid, self.source_ref, RegistryEdgeType.WAS_DERIVED_FROM)
            )

        for ev in self.evidence:
            nodes.append(
                {
                    "id": ev.id,
                    "type": RegistryNodeType.EVIDENCE.value,
                    "properties": {
                        "name": (ev.content[:80] or ev.id),
                        "content": ev.content,
                        "kind": ev.kind,
                        "timestamp": self.timestamp,
                    },
                }
            )
            if ev.source_ref:
                edges.append(
                    self._edge(ev.id, ev.source_ref, RegistryEdgeType.WAS_DERIVED_FROM)
                )

        for cs in self.code_specs:
            nodes.append(
                {
                    "id": cs.id,
                    "type": RegistryNodeType.CODE_SPEC.value,
                    "properties": {
                        "name": cs.symbol or cs.id,
                        "summary": cs.summary,
                        "language": cs.language,
                        "symbol": cs.symbol,
                        "path": cs.path,
                    },
                }
            )

        for cl in self.claims:
            nodes.append(
                {
                    "id": cl.id,
                    "type": RegistryNodeType.CLAIM.value,
                    "properties": {
                        "name": (cl.statement[:80] or cl.id),
                        "statement": cl.statement,
                        "claim_type": cl.claim_type,
                        "confidence": float(cl.confidence),
                        "timestamp": self.timestamp,
                    },
                }
            )
            edges.append(self._edge(aid, cl.id, RegistryEdgeType.CONTAINS))
            for ev_id in cl.evidence_ids:
                edges.append(self._edge(cl.id, ev_id, RegistryEdgeType.GROUNDED_IN))
            for cs_id in cl.code_spec_ids:
                edges.append(self._edge(cl.id, cs_id, RegistryEdgeType.IMPLEMENTED_BY))

        for ex in self.exploration:
            nodes.append(
                {
                    "id": ex.id,
                    "type": RegistryNodeType.EXPLORATION_NODE.value,
                    "properties": {
                        "name": (ex.text[:80] or ex.id),
                        "exploration_kind": ex.kind,
                        "text": ex.text,
                        "timestamp": self.timestamp,
                    },
                }
            )
            edges.append(self._edge(aid, ex.id, RegistryEdgeType.CONTAINS))
            if ex.parent_id:
                rel = {
                    "pivot": RegistryEdgeType.PIVOTED_FROM,
                    "dead_end": RegistryEdgeType.REACHED_DEAD_END,
                }.get(ex.kind, RegistryEdgeType.WAS_DERIVED_FROM)
                edges.append(self._edge(ex.id, ex.parent_id, rel))

        return nodes, edges

    @staticmethod
    def _edge(src: str, dst: str, rel: RegistryEdgeType) -> dict[str, Any]:
        return {"source": src, "target": dst, "type": rel.value}

    def materialize(self, engine: Any) -> dict[str, Any]:
        """Write the artifact's nodes + forensic edges into the graph (best-effort).

        Idempotent (the engine upserts by id). Returns counts; never raises into a
        Loop — a partial write degrades, it does not abort the cycle.
        """
        nodes, edges = self.to_graph_payload()
        n_ok = e_ok = 0
        for node in nodes:
            try:
                engine.add_node(node["id"], node["type"], properties=node["properties"])
                n_ok += 1
            except Exception as exc:  # noqa: BLE001 — best-effort persist
                logger.debug("ARA node persist failed (%s): %s", node["id"], exc)
        for edge in edges:
            try:
                engine.add_edge(edge["source"], edge["target"], edge["type"])
                e_ok += 1
            except Exception as exc:  # noqa: BLE001
                logger.debug("ARA edge persist failed: %s", exc)
        return {"nodes": n_ok, "edges": e_ok, "artifact": self.node_id}

    # -- builders ---------------------------------------------------------- #
    @classmethod
    def from_extracted(
        cls,
        article_id: str,
        title: str,
        *,
        claims: list[str] | None = None,
        evidence: list[str] | None = None,
        code_specs: list[str] | None = None,
        summary: str = "",
        authors: list[str] | None = None,
        source_url: str = "",
        source_ref: str = "",
    ) -> ResearchArtifact:
        """Build an ARA from flat extractor output, wiring every claim to all
        evidence/code (the conservative full-binding the Seal later prunes)."""
        ev_units = [
            Evidence(id=f"evidence:{article_id}:{i}", content=text)
            for i, text in enumerate(evidence or [])
        ]
        code_units = [
            CodeSpec(id=f"code_spec:{article_id}:{i}", summary=text)
            for i, text in enumerate(code_specs or [])
        ]
        ev_ids = [e.id for e in ev_units]
        code_ids = [c.id for c in code_units]
        claim_units = [
            Claim(
                id=f"claim:{article_id}:{_slug(text, limit=40)}:{i}",
                statement=text,
                evidence_ids=list(ev_ids),
                code_spec_ids=list(code_ids),
            )
            for i, text in enumerate(claims or [])
        ]
        return cls(
            article_id=article_id,
            title=title,
            summary=summary,
            claims=claim_units,
            evidence=ev_units,
            code_specs=code_units,
            authors=list(authors or []),
            source_url=source_url,
            source_ref=source_ref or (f"article:{article_id}"),
        )

    @classmethod
    def from_research_artifact(cls, legacy: Any) -> ResearchArtifact:
        """Lift the legacy :class:`adaptation.research_artifacts.ResearchArtifact`
        (key_contributions / methods / suggested_experiments) into an OWL-native ARA.

        Contributions become claims, methods become code specs, suggested
        experiments seed exploration nodes — so existing extracted papers gain a
        reason-able 4-layer shape without re-extraction.
        """
        aid = str(
            getattr(legacy, "article_id", "") or _slug(getattr(legacy, "title", ""))
        )
        contributions = list(getattr(legacy, "key_contributions", []) or [])
        methods = list(getattr(legacy, "methods", []) or [])
        experiments = list(getattr(legacy, "suggested_experiments", []) or [])
        art = cls.from_extracted(
            aid,
            str(getattr(legacy, "title", aid)),
            claims=contributions,
            code_specs=methods,
            summary=str(getattr(legacy, "summary", "")),
            authors=list(getattr(legacy, "authors", []) or []),
            source_url=str(getattr(legacy, "source_url", "")),
        )
        art.exploration = [
            ExplorationNode(
                id=f"exploration_node:{aid}:{i}", kind="experiment", text=text
            )
            for i, text in enumerate(experiments)
        ]
        return art


__all__ = [
    "Claim",
    "CodeSpec",
    "Evidence",
    "ExplorationKind",
    "ExplorationNode",
    "ResearchArtifact",
]
