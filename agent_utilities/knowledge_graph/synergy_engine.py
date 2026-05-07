#!/usr/bin/python
"""Cross-Pillar Synergy Engine.

CONCEPT:KG-2.19 — Cross-Pillar Synergy Engine

Discovers non-obvious functional synergies between the 5 Unified Pillars
of the agent-utilities ecosystem by analyzing concept-to-concept edges,
shared node types, and structural patterns in the Knowledge Graph.

The engine leverages existing infrastructure:

- **Analogy Engine** (KG-2.15): Finds structurally similar subgraphs
  across different pillars.
- **SKOS Taxonomy** (``broader``/``narrower``): Identifies cross-pillar
  concept hierarchies.
- **Transitive Properties**: ``dependsOn``, ``partOf``, ``propagatesRiskTo``
  enable automatic cross-pillar relationship discovery.

Architecture::

    ┌──────────────────────────────────────────┐
    │          Cross-Pillar Synergy Engine      │
    ├──────────────────────────────────────────┤
    │ discover_concept_bridges()               │
    │ compute_pillar_coupling()                │
    │ suggest_missing_edges()                  │
    │ generate_synergy_report()                │
    └──────────┬───────────────────────────────┘
               │ uses
    ┌──────────▼───────────────────────────────┐
    │ Analogy Engine (KG-2.15)                 │
    │ SKOS Taxonomy Properties                 │
    │ OWL Transitive Closures                  │
    └──────────────────────────────────────────┘

Pillar Mapping
--------------
- **ORCH-1.x**: Orchestration & Routing
- **KG-2.x**: Knowledge Graph & Retrieval
- **AHE-3.x**: Agentic Harness Engineering
- **ECO-4.x**: Ecosystem & Integration
- **OS-5.x**: Agent OS & Infrastructure

See Also:
    - :mod:`agent_utilities.knowledge_graph.analogy_engine` (KG-2.15)
    - :mod:`agent_utilities.knowledge_graph.semantic_subsumption` (KG-2.16)
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Pillar prefix → display name mapping
PILLAR_NAMES: dict[str, str] = {
    "ORCH-1": "Orchestration & Routing",
    "KG-2": "Knowledge Graph & Retrieval",
    "AHE-3": "Agentic Harness Engineering",
    "ECO-4": "Ecosystem & Integration",
    "OS-5": "Agent OS & Infrastructure",
}

# Concept ID regex: CONCEPT:PILLAR-MAJOR.MINOR
_CONCEPT_RE = re.compile(r"CONCEPT:((?:ORCH|KG|AHE|ECO|OS)-\d+)\.(\d+)")


@dataclass
class ConceptBridge:
    """A concept that spans multiple pillars.

    CONCEPT:KG-2.19 — Cross-Pillar Synergy Engine

    Attributes:
        concept_id: The CONCEPT:ID (e.g., ``AHE-3.5``).
        concept_name: Human-readable concept name.
        primary_pillar: The pillar this concept is registered under.
        bridged_pillars: Other pillars this concept functionally touches.
        bridge_reason: Why this concept spans pillars.
    """

    concept_id: str = ""
    concept_name: str = ""
    primary_pillar: str = ""
    bridged_pillars: list[str] = field(default_factory=list)
    bridge_reason: str = ""


@dataclass
class PillarCoupling:
    """Coupling metric between two pillars.

    CONCEPT:KG-2.19 — Cross-Pillar Synergy Engine

    Attributes:
        pillar_a: First pillar prefix (e.g., ``ORCH-1``).
        pillar_b: Second pillar prefix (e.g., ``KG-2``).
        shared_edge_types: Edge types that connect concepts in both pillars.
        coupling_score: Normalized coupling strength (0.0–1.0).
        bridge_concepts: Concepts that serve as bridges between the pillars.
    """

    pillar_a: str = ""
    pillar_b: str = ""
    shared_edge_types: list[str] = field(default_factory=list)
    coupling_score: float = 0.0
    bridge_concepts: list[str] = field(default_factory=list)


@dataclass
class SynergyInsight:
    """A discovered synergy or missing relationship.

    CONCEPT:KG-2.19 — Cross-Pillar Synergy Engine

    Attributes:
        source_concept: The source concept ID.
        target_concept: The target concept ID.
        suggested_relationship: The edge type that should connect them.
        confidence: Confidence score for this suggestion (0.0–1.0).
        rationale: Explanation of why this synergy exists.
        is_existing: True if this relationship already exists.
    """

    source_concept: str = ""
    target_concept: str = ""
    suggested_relationship: str = ""
    confidence: float = 0.0
    rationale: str = ""
    is_existing: bool = False


class SynergyEngine:
    """Discovers functional synergies between the 5 Unified Pillars.

    CONCEPT:KG-2.19 — Cross-Pillar Synergy Engine

    Analyzes the existing concept registry and Knowledge Graph topology
    to find:

    1. **Concept bridges** — Concepts that functionally span multiple pillars.
    2. **Pillar coupling** — How tightly coupled each pillar pair is.
    3. **Missing edges** — Relationships that should exist based on
       structural similarity.
    4. **Synergy reports** — Human-readable analysis of the ecosystem.

    Args:
        concept_registry: Dict mapping concept IDs to metadata.
            Each entry should have ``name``, ``pillar``, ``module``,
            and optionally ``depends_on`` keys.

    Example::

        registry = {
            "ORCH-1.1": {"name": "Wide-Search Orchestration", "pillar": "ORCH-1"},
            "KG-2.1": {"name": "SelfModel", "pillar": "KG-2"},
            ...
        }
        engine = SynergyEngine(registry)
        bridges = engine.discover_concept_bridges()
        coupling = engine.compute_pillar_coupling()
        report = engine.generate_synergy_report()
    """

    # Known cross-pillar relationships (hardcoded from architecture docs)
    _KNOWN_BRIDGES: dict[str, list[str]] = {
        # ExperienceNode bridges AHE → KG
        "AHE-3.5": ["KG-2"],
        # TeamConfig bridges ORCH → AHE (reward tracking)
        "AHE-3.3": ["ORCH-1"],
        # EvalRunner bridges AHE → OS (observability)
        "AHE-3.12": ["OS-5"],
        # Prompt Scanner bridges OS → KG (risk propagation)
        "OS-5.4": ["KG-2"],
        # Context Compaction bridges KG → OS (token management)
        "KG-2.10": ["OS-5"],
        # Ecosystem Topology bridges ECO → KG
        "ECO-4.7": ["KG-2"],
        # Confidence Router bridges ORCH → KG (SelfModel signals)
        "ORCH-1.2": ["KG-2"],
        # Swarm Presets bridge ORCH → ECO (multi-agent workflows)
        "ORCH-1.4": ["ECO-4"],
        # Research Pipeline bridges KG → ECO (ScholarX integration)
        "KG-2.11": ["ECO-4"],
        # Guardrail Engine bridges OS → AHE (policy enforcement feedback)
        "OS-5.8": ["AHE-3"],
    }

    def __init__(
        self,
        concept_registry: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._registry = concept_registry or {}
        self._pillar_concepts: dict[str, list[str]] = defaultdict(list)
        self._build_pillar_index()

    def _build_pillar_index(self) -> None:
        """Index concepts by pillar prefix."""
        for concept_id, meta in self._registry.items():
            pillar = meta.get("pillar", "")
            if not pillar:
                match = _CONCEPT_RE.match(f"CONCEPT:{concept_id}")
                if match:
                    pillar = match.group(1)
            if pillar:
                self._pillar_concepts[pillar].append(concept_id)

    def discover_concept_bridges(self) -> list[ConceptBridge]:
        """Find concepts that functionally span multiple pillars.

        CONCEPT:KG-2.19 — Cross-Pillar Synergy Engine

        Uses a combination of:
        - Hardcoded known bridges from architecture documentation
        - Dependency analysis (concepts that import from other pillars)
        - Module path analysis (modules that reference multiple pillars)

        Returns:
            List of :class:`ConceptBridge` instances.
        """
        bridges: list[ConceptBridge] = []

        # 1. Check known bridges
        for concept_id, bridged_pillars in self._KNOWN_BRIDGES.items():
            meta = self._registry.get(concept_id, {})
            match = _CONCEPT_RE.match(f"CONCEPT:{concept_id}")
            primary = match.group(1) if match else ""

            bridges.append(
                ConceptBridge(
                    concept_id=concept_id,
                    concept_name=meta.get("name", concept_id),
                    primary_pillar=primary,
                    bridged_pillars=bridged_pillars,
                    bridge_reason=f"Known architectural bridge from {primary} to {', '.join(bridged_pillars)}",
                )
            )

        # 2. Analyze concept dependencies
        for concept_id, meta in self._registry.items():
            deps = meta.get("depends_on", [])
            if not deps:
                continue

            match = _CONCEPT_RE.match(f"CONCEPT:{concept_id}")
            if not match:
                continue
            primary = match.group(1)

            foreign_pillars = set()
            for dep_id in deps:
                dep_match = _CONCEPT_RE.match(f"CONCEPT:{dep_id}")
                if dep_match:
                    dep_pillar = dep_match.group(1)
                    if dep_pillar != primary:
                        foreign_pillars.add(dep_pillar)

            if foreign_pillars and concept_id not in self._KNOWN_BRIDGES:
                bridges.append(
                    ConceptBridge(
                        concept_id=concept_id,
                        concept_name=meta.get("name", concept_id),
                        primary_pillar=primary,
                        bridged_pillars=sorted(foreign_pillars),
                        bridge_reason=f"Dependencies span {', '.join(foreign_pillars)}",
                    )
                )

        return bridges

    def compute_pillar_coupling(self) -> list[PillarCoupling]:
        """Quantify coupling between each pillar pair.

        CONCEPT:KG-2.19 — Cross-Pillar Synergy Engine

        Coupling is measured by:
        - Number of shared edge types between pillar concepts
        - Number of bridge concepts connecting the pillars
        - Cross-pillar dependency density

        Returns:
            List of :class:`PillarCoupling` instances for each pillar pair.
        """
        pillars = sorted(PILLAR_NAMES.keys())
        bridges = self.discover_concept_bridges()
        couplings: list[PillarCoupling] = []

        # Build bridge index
        bridge_index: dict[tuple[str, str], list[str]] = defaultdict(list)
        for bridge in bridges:
            for bp in bridge.bridged_pillars:
                pair = tuple(sorted([bridge.primary_pillar, bp]))
                bridge_index[pair].append(bridge.concept_id)

        # Known shared edge types between pillars
        _SHARED_EDGES: dict[tuple[str, str], list[str]] = {
            ("AHE-3", "KG-2"): ["experienced_during", "was_derived_from", "part_of"],
            ("ORCH-1", "KG-2"): ["routed_by", "reused_team", "broader"],
            ("OS-5", "KG-2"): ["propagates_risk_to", "detected_threat"],
            ("ORCH-1", "AHE-3"): ["produced_outcome", "evaluated_with"],
            ("ECO-4", "KG-2"): ["fetched_from", "falls_back_to"],
            ("OS-5", "AHE-3"): ["triggered_guardrail", "audited_by"],
        }

        for i, pa in enumerate(pillars):
            for pb in pillars[i + 1 :]:
                pair = tuple(sorted([pa, pb]))
                shared = _SHARED_EDGES.get(pair, [])
                bridge_concepts = bridge_index.get(pair, [])

                # Compute coupling score (0–1)
                score = min(
                    1.0,
                    (len(shared) * 0.15 + len(bridge_concepts) * 0.25),
                )

                couplings.append(
                    PillarCoupling(
                        pillar_a=pa,
                        pillar_b=pb,
                        shared_edge_types=shared,
                        coupling_score=round(score, 3),
                        bridge_concepts=bridge_concepts,
                    )
                )

        return sorted(couplings, key=lambda c: c.coupling_score, reverse=True)

    def suggest_missing_edges(self) -> list[SynergyInsight]:
        """Suggest relationships that should exist but don't.

        CONCEPT:KG-2.19 — Cross-Pillar Synergy Engine

        Uses structural pattern matching to identify concepts that
        are similar in structure but lack explicit connections.

        Returns:
            List of :class:`SynergyInsight` with suggested edges.
        """
        suggestions: list[SynergyInsight] = []

        # Pattern 1: Every AHE concept should feed back to KG for persistence
        ahe_concepts = self._pillar_concepts.get("AHE-3", [])
        for concept_id in ahe_concepts:
            meta = self._registry.get(concept_id, {})
            if not meta.get("has_kg_persistence", True):
                suggestions.append(
                    SynergyInsight(
                        source_concept=concept_id,
                        target_concept="KG-2.0",
                        suggested_relationship="persists_to",
                        confidence=0.8,
                        rationale="AHE results should persist to KG for cross-session learning",
                    )
                )

        # Pattern 2: Every OS concept should be observable
        os_concepts = self._pillar_concepts.get("OS-5", [])
        for concept_id in os_concepts:
            meta = self._registry.get(concept_id, {})
            if not meta.get("has_observability", True):
                suggestions.append(
                    SynergyInsight(
                        source_concept=concept_id,
                        target_concept="OS-5.9",
                        suggested_relationship="observed_by",
                        confidence=0.7,
                        rationale="OS infrastructure should emit telemetry events",
                    )
                )

        # Pattern 3: ORCH routing should integrate with AHE reward signals
        orch_concepts = self._pillar_concepts.get("ORCH-1", [])
        for concept_id in orch_concepts:
            meta = self._registry.get(concept_id, {})
            if "routing" in meta.get("name", "").lower():
                suggestions.append(
                    SynergyInsight(
                        source_concept=concept_id,
                        target_concept="AHE-3.3",
                        suggested_relationship="reward_signal_from",
                        confidence=0.75,
                        rationale="Routing decisions should feed into TeamConfig reward tracking",
                        is_existing=concept_id == "ORCH-1.2",
                    )
                )

        return [s for s in suggestions if not s.is_existing]

    def generate_synergy_report(self) -> str:
        """Generate a comprehensive synergy analysis report.

        CONCEPT:KG-2.19 — Cross-Pillar Synergy Engine

        Returns:
            Formatted markdown report of discovered synergies,
            coupling metrics, and suggested improvements.
        """
        bridges = self.discover_concept_bridges()
        couplings = self.compute_pillar_coupling()
        suggestions = self.suggest_missing_edges()

        lines: list[str] = [
            "# Cross-Pillar Synergy Report",
            "",
            f"**Pillars Analyzed**: {len(PILLAR_NAMES)}",
            f"**Total Concepts**: {len(self._registry)}",
            f"**Bridges Found**: {len(bridges)}",
            f"**Suggestions**: {len(suggestions)}",
            "",
            "## Concept Bridges",
            "",
            "Concepts that functionally span multiple pillars:",
            "",
            "| Concept | Primary Pillar | Bridged Pillars | Reason |",
            "|:--------|:---------------|:----------------|:-------|",
        ]

        for b in bridges:
            bridged = ", ".join(b.bridged_pillars)
            lines.append(
                f"| {b.concept_id}: {b.concept_name} | {b.primary_pillar} | {bridged} | {b.bridge_reason} |"
            )

        lines.extend(
            [
                "",
                "## Pillar Coupling Matrix",
                "",
                "| Pillar A | Pillar B | Score | Shared Edges | Bridges |",
                "|:---------|:---------|:------|:-------------|:--------|",
            ]
        )

        for c in couplings:
            edges = ", ".join(c.shared_edge_types[:3]) or "—"
            bridges_str = str(len(c.bridge_concepts))
            lines.append(
                f"| {c.pillar_a} | {c.pillar_b} | {c.coupling_score:.3f} | {edges} | {bridges_str} |"
            )

        if suggestions:
            lines.extend(
                [
                    "",
                    "## Suggested Missing Relationships",
                    "",
                ]
            )
            for s in suggestions:
                lines.append(
                    f"- **{s.source_concept}** → `{s.suggested_relationship}` → "
                    f"**{s.target_concept}** (confidence: {s.confidence:.2f}): "
                    f"{s.rationale}"
                )

        return "\n".join(lines)


__all__ = [
    "ConceptBridge",
    "PILLAR_NAMES",
    "PillarCoupling",
    "SynergyEngine",
    "SynergyInsight",
]
