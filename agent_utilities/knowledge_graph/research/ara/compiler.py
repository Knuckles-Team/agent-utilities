#!/usr/bin/python
from __future__ import annotations

"""ARA Compiler — legacy paper/repo → OWL-native ARA, ecosystem-grounded (KG-2.80).

The paper's ARA Compiler turns narrative research into the 4-layer artifact via
semantic-deconstruct → cognitive-map → physical-ground → exploration-extract. We
reuse the existing extraction (``ResearchArtifactGenerator``) for deconstruction and
add the move the paper defers to "critical mass": **physical-ground against the one
ecosystem ontology** — every claim is grounded to the ecosystem Concepts/code/services
it touches via :class:`ConceptMatcher`, materialized as ``grounded_in`` edges. Because
``grounded_in`` is transitive with a ``supports`` inverse (``owl_bridge``), reasoning
then extrapolates cross-domain links from the very first compiled artifact — surpassing
the paper, which can only infer collectively once a corpus accumulates.

Grounding is injectable (``ground_fn``) so the compiler is unit-testable without a live
embedder/LLM; the default best-effort path queries the engine for candidate Concepts.

Concept: ara-compiler
"""

import logging
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from .artifact import Evidence, ResearchArtifact

logger = logging.getLogger(__name__)

#: a grounding function: given a claim statement, return ecosystem node ids it grounds in.
GroundFn = Callable[[str], list[str]]


class CompileReport(BaseModel):
    """What one compile pass produced (CONCEPT:KG-2.80)."""

    article_id: str
    artifact_node: str
    n_claims: int = 0
    n_grounded: int = 0
    n_nodes: int = 0
    n_edges: int = 0
    #: claim id → ecosystem node ids it was grounded to (the cross-domain links).
    groundings: dict[str, list[str]] = Field(default_factory=dict)
    error: str = ""


class ARACompiler:
    """Compile an ingested paper into an ecosystem-grounded OWL-native ARA."""

    def __init__(
        self,
        engine: Any,
        *,
        ground_fn: GroundFn | None = None,
        generator: Any = None,
        ground_to_ecosystem: bool = True,
        max_groundings: int = 5,
    ) -> None:
        self._engine = engine
        self._ground_fn = ground_fn
        self._generator = generator
        self._ground_to_ecosystem = ground_to_ecosystem
        self._max_groundings = max_groundings

    # -- public ----------------------------------------------------------- #
    def compile(
        self,
        article_id: str,
        *,
        target_codebase: str | None = None,
        materialize: bool = True,
    ) -> tuple[ResearchArtifact, CompileReport]:
        """Deconstruct → lift → ecosystem-ground → (materialize) the artifact."""
        artifact = self._deconstruct(article_id, target_codebase)
        groundings = self._ground(artifact) if self._ground_to_ecosystem else {}
        report = CompileReport(
            article_id=article_id,
            artifact_node=artifact.node_id,
            n_claims=len(artifact.claims),
            n_grounded=sum(1 for v in groundings.values() if v),
            groundings=groundings,
        )
        if materialize:
            stats = artifact.materialize(self._engine)
            report.n_nodes = int(stats.get("nodes", 0))
            report.n_edges = int(stats.get("edges", 0))
            self._materialize_groundings(groundings, report)
        return artifact, report

    # -- stages ----------------------------------------------------------- #
    def _deconstruct(
        self, article_id: str, target_codebase: str | None
    ) -> ResearchArtifact:
        """Reuse the existing extractor, then lift to the OWL-native ARA shape."""
        gen = self._generator or self._default_generator()
        if gen is None:
            return ResearchArtifact(article_id=article_id, title=article_id)
        try:
            legacy = gen.generate_paper_artifact(article_id, target_codebase)
        except Exception as e:  # noqa: BLE001 — extraction is best-effort
            logger.debug("ARA deconstruct failed for %s: %s", article_id, e)
            return ResearchArtifact(article_id=article_id, title=article_id)
        return ResearchArtifact.from_research_artifact(legacy)

    def _ground(self, artifact: ResearchArtifact) -> dict[str, list[str]]:
        """Ground each claim to the ecosystem Concepts/code it touches.

        Adds the matched ecosystem node ids to the claim's ``evidence_ids`` so the
        materialized ``grounded_in`` edges connect /logic claims to the real
        ecosystem estate — the cross-domain links reasoning then extrapolates over.
        """
        ground = self._ground_fn or self._default_ground_fn()
        out: dict[str, list[str]] = {}
        if ground is None:
            return out
        for claim in artifact.claims:
            try:
                hits = list(ground(claim.statement))[: self._max_groundings]
            except Exception as e:  # noqa: BLE001
                logger.debug("grounding failed for claim %s: %s", claim.id, e)
                hits = []
            if hits:
                # de-dup while preserving the claim's own evidence units
                for nid in hits:
                    if nid and nid not in claim.evidence_ids:
                        claim.evidence_ids.append(nid)
                out[claim.id] = hits
        return out

    def _materialize_groundings(
        self, groundings: dict[str, list[str]], report: CompileReport
    ) -> None:
        """Write claim -grounded_in-> ecosystem-node edges (best-effort)."""
        for claim_id, node_ids in groundings.items():
            for nid in node_ids:
                try:
                    self._engine.add_edge(claim_id, nid, "grounded_in")
                    report.n_edges += 1
                except Exception as e:  # noqa: BLE001
                    logger.debug("grounding edge persist failed: %s", e)

    # -- defaults (best-effort, never required for the unit path) --------- #
    def _default_generator(self) -> Any:
        try:
            from ...adaptation.research_artifacts import ResearchArtifactGenerator

            return ResearchArtifactGenerator(self._engine)
        except Exception as e:  # noqa: BLE001
            logger.debug("no default ARA generator: %s", e)
            return None

    def _default_ground_fn(self) -> GroundFn | None:
        """A SUPPORTED-query grounding: nearest existing Concepts by id token match.

        Intentionally simple (no embeddings) so the always-on path never blocks; the
        rich :class:`ConceptMatcher` path is wired by callers that have an embedder.
        """
        engine = self._engine
        if engine is None or not hasattr(engine, "query_cypher"):
            return None

        def _ground(statement: str) -> list[str]:
            try:
                rows = engine.query_cypher(
                    "MATCH (c:Concept) RETURN c.id AS id, c.name AS name LIMIT 200"
                )
            except Exception:  # noqa: BLE001
                return []
            toks = {t for t in statement.lower().split() if len(t) > 4}
            hits: list[str] = []
            for r in rows or []:
                if not isinstance(r, dict) or not r.get("id"):
                    continue
                name = str(r.get("name") or "").lower()
                if name and any(t in name for t in toks):
                    hits.append(r["id"])
            return hits

        return _ground


def evidence_from_groundings(article_id: str, node_ids: list[str]) -> list[Evidence]:
    """Wrap grounded ecosystem nodes as Evidence units (helper for callers)."""
    return [
        Evidence(
            id=nid, content=f"ecosystem grounding for {article_id}", kind="ecosystem"
        )
        for nid in node_ids
    ]


__all__ = ["ARACompiler", "CompileReport", "GroundFn", "evidence_from_groundings"]
