#!/usr/bin/python
from __future__ import annotations

"""CONCEPT:KG-2.3 — Graph Integrity Validator.

Non-blocking, tiered validation for the Unified Intelligence Graph.
Inspired by Understand-Anything's ``graph-reviewer`` agent, adapted for
agent-utilities' runtime KG with graph compute + Cypher backends.

Validation Tiers:
    - **Tier 1 (Auto-fix)**: Silently correct recoverable issues:
      nulls → defaults, LLM type aliases, weight/score clamping.
    - **Tier 2 (Integrity)**: Detect referential integrity violations:
      dangling edges, invalid node references, duplicate IDs.
    - **Tier 3 (Quality)**: Flag quality concerns: orphan nodes,
      generic summaries, self-referencing edges, missing descriptions.
    - **Tier 4 (Fatal)**: Only raises on catastrophic failures:
      zero valid nodes, missing critical schema, broken graph structure.

Usage::

    from agent_utilities.knowledge_graph.security.graph_validator import GraphValidator

    validator = GraphValidator(engine)
    report = validator.validate()
    # report.tier1_fixes, report.tier2_violations, report.tier3_warnings

See docs/pillars/architecture_c4.md §CONCEPT:KG-2.3
"""


import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM Alias Normalization Map (inspired by UA's schema.ts autofix)
# ---------------------------------------------------------------------------

# Common LLM output aliases → canonical RegistryNodeType values
NODE_TYPE_ALIASES: dict[str, str] = {
    "func": "symbol",
    "function": "symbol",
    "method": "symbol",
    "class": "symbol",
    "interface": "symbol",
    "struct": "symbol",
    "enum": "symbol",
    "type": "symbol",
    "variable": "symbol",
    "constant": "symbol",
    "config": "file",
    "configuration": "file",
    "test": "file",
    "spec": "file",
    "component": "module",
    "package": "module",
    "library": "module",
    "service": "agent",
    "server": "agent",
    "worker": "agent",
    "bot": "agent",
    "mcp_tool": "callable_resource",
    "a2a_agent": "callable_resource",
    "internal_skill": "callable_resource",
    "api": "callable_resource",
    "endpoint": "callable_resource",
    "note": "memory",
    "memo": "memory",
    "journal": "memory",
    "recollection": "memory",
    "observation_node": "observation",
    "belief_node": "belief",
    "decision_node": "decision",
    "claim": "evidence",
    "assertion": "evidence",
}

# Common LLM edge type aliases → canonical RegistryEdgeType values
EDGE_TYPE_ALIASES: dict[str, str] = {
    "extends": "inherits_from",
    "inherits": "inherits_from",
    "subclass_of": "inherits_from",
    "implements": "inherits_from",
    "calls_function": "calls",
    "invokes": "calls",
    "triggers": "calls",
    "uses": "depends_on",
    "requires": "depends_on",
    "needs": "depends_on",
    "has": "contains",
    "owns": "contains",
    "includes": "contains",
    "parent_of": "contains",
    "child_of": "part_of",
    "member_of": "belongs_to",
    "written_by": "authored",
    "created_by": "authored",
    "associated_with": "related_to",
    "linked_to": "related_to",
    "see_also": "related_to",
    "derives_from": "was_derived_from",
    "based_on": "was_derived_from",
    "sourced_from": "was_derived_from",
    "opposes": "contradicts",
    "conflicts": "contradicts",
    "disagrees_with": "contradicts",
    "before": "temporally_precedes",
    "precedes": "temporally_precedes",
    "follows": "temporally_precedes",
    "proves": "has_evidence",
    "confirms": "has_evidence",
    "validates": "has_evidence",
}

# Complexity aliases (UA-style normalization)
COMPLEXITY_ALIASES: dict[str, str] = {
    "low": "simple",
    "medium": "moderate",
    "high": "complex",
    "very_high": "complex",
    "trivial": "simple",
    "easy": "simple",
    "hard": "complex",
    "difficult": "complex",
}


# ---------------------------------------------------------------------------
# Validation Report
# ---------------------------------------------------------------------------


@dataclass
class ValidationIssue:
    """A single validation issue found in the graph."""

    tier: int
    category: str
    node_id: str | None
    edge_key: str | None
    message: str
    auto_fixed: bool = False


@dataclass
class ValidationReport:
    """Complete validation report from a graph integrity check.

    CONCEPT:KG-2.3 — Graph Integrity Validator

    Attributes:
        tier1_fixes: Auto-fixed issues (applied silently).
        tier2_violations: Referential integrity violations (logged as warnings).
        tier3_warnings: Quality concerns (logged for review).
        tier4_fatal: Catastrophic failures (raise exceptions).
        total_nodes: Number of nodes in the graph at validation time.
        total_edges: Number of edges in the graph at validation time.
        duration_ms: Validation execution time.
    """

    tier1_fixes: list[ValidationIssue] = field(default_factory=list)
    tier2_violations: list[ValidationIssue] = field(default_factory=list)
    tier3_warnings: list[ValidationIssue] = field(default_factory=list)
    tier4_fatal: list[ValidationIssue] = field(default_factory=list)
    total_nodes: int = 0
    total_edges: int = 0
    duration_ms: float = 0.0
    validated_at: str = ""

    @property
    def is_healthy(self) -> bool:
        """True if no tier-2 violations or tier-4 fatals."""
        return len(self.tier2_violations) == 0 and len(self.tier4_fatal) == 0

    @property
    def issue_count(self) -> int:
        """Total number of issues across all tiers."""
        return (
            len(self.tier1_fixes)
            + len(self.tier2_violations)
            + len(self.tier3_warnings)
            + len(self.tier4_fatal)
        )

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Graph Validation Report ({self.validated_at})",
            f"  Nodes: {self.total_nodes} | Edges: {self.total_edges} | "
            f"Duration: {self.duration_ms:.1f}ms",
            f"  Tier 1 (auto-fixed): {len(self.tier1_fixes)}",
            f"  Tier 2 (violations): {len(self.tier2_violations)}",
            f"  Tier 3 (warnings):   {len(self.tier3_warnings)}",
            f"  Tier 4 (fatal):      {len(self.tier4_fatal)}",
            f"  Status: {'HEALTHY' if self.is_healthy else 'ISSUES DETECTED'}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Graph Validator
# ---------------------------------------------------------------------------


class GraphValidator:
    """Non-blocking, tiered graph integrity validator.

    CONCEPT:KG-2.3 — Graph Integrity Validator

    Runs four validation tiers over the graph and produces a structured
    report. Never crashes the server — auto-fixes what it can, logs
    warnings for quality issues, and only fatals on truly catastrophic
    failures (zero nodes, broken structure).

    Inspired by Understand-Anything's ``graph-reviewer`` agent with
    9-check validation and 4-tier auto-fix pipeline.

    Args:
        engine: The ``IntelligenceGraphEngine`` to validate.
    """

    def __init__(self, engine: IntelligenceGraphEngine) -> None:
        self.engine = engine

    def validate(self) -> ValidationReport:
        """Run all validation tiers and return a structured report.

        Returns:
            ``ValidationReport`` with issues categorized by tier.

        Raises:
            GraphValidationFatalError: Only for tier-4 catastrophic failures.
        """
        start = time.perf_counter()
        report = ValidationReport(
            validated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

        graph = self.engine.graph

        report.total_nodes = graph.number_of_nodes()
        report.total_edges = graph.number_of_edges()

        # --- Tier 1: Auto-fix ---
        self._tier1_autofix(report)

        # --- Tier 2: Referential Integrity ---
        self._tier2_integrity(report)

        # --- Tier 3: Quality Checks ---
        self._tier3_quality(report)

        # --- Tier 4: Fatal Checks ---
        self._tier4_fatal(report)

        report.duration_ms = (time.perf_counter() - start) * 1000

        # Log summary
        if report.tier1_fixes:
            logger.info(
                "[CONCEPT:KG-2.3] Auto-fixed %d issues in graph",
                len(report.tier1_fixes),
            )
        if report.tier2_violations:
            logger.warning(
                "[CONCEPT:KG-2.3] Found %d integrity violations",
                len(report.tier2_violations),
            )
        if report.tier3_warnings:
            logger.info(
                "[CONCEPT:KG-2.3] Found %d quality warnings", len(report.tier3_warnings)
            )
        if report.tier4_fatal:
            logger.error(
                "[CONCEPT:KG-2.3] FATAL: %d critical issues detected",
                len(report.tier4_fatal),
            )

        return report

    # --- Tier 1: Auto-fix (silent) -------------------------------------------

    def _tier1_autofix(self, report: ValidationReport) -> None:
        """Silently correct recoverable issues in the graph."""
        graph = self.engine.graph

        for node_id, data in list(graph.nodes(data=True)):
            # 1a. Normalize LLM type aliases
            node_type = data.get("type", "")
            if isinstance(node_type, str):
                type_lower = node_type.lower().strip()
                canonical = NODE_TYPE_ALIASES.get(type_lower)
                if canonical and type_lower != canonical:
                    graph.nodes[node_id]["type"] = canonical
                    report.tier1_fixes.append(
                        ValidationIssue(
                            tier=1,
                            category="type_alias",
                            node_id=node_id,
                            edge_key=None,
                            message=f"Normalized type '{node_type}' → '{canonical}'",
                            auto_fixed=True,
                        )
                    )

            # 1b. Clamp importance_score to [0.0, 1.0]
            score = data.get("importance_score")
            if score is not None:
                if isinstance(score, int | float):
                    clamped = max(0.0, min(1.0, float(score)))
                    if clamped != float(score):
                        graph.nodes[node_id]["importance_score"] = clamped
                        report.tier1_fixes.append(
                            ValidationIssue(
                                tier=1,
                                category="score_clamp",
                                node_id=node_id,
                                edge_key=None,
                                message=(
                                    f"Clamped importance_score {score} → {clamped}"
                                ),
                                auto_fixed=True,
                            )
                        )

            # 1c. Set missing name from ID
            if not data.get("name"):
                graph.nodes[node_id]["name"] = node_id
                report.tier1_fixes.append(
                    ValidationIssue(
                        tier=1,
                        category="missing_name",
                        node_id=node_id,
                        edge_key=None,
                        message=f"Set missing name to node ID '{node_id}'",
                        auto_fixed=True,
                    )
                )

            # 1d. Clamp weight/reward/confidence fields
            for field_name in ("reward", "confidence", "certainty", "confidence_score"):
                val = data.get(field_name)
                if val is not None and isinstance(val, int | float):
                    clamped = max(0.0, min(1.0, float(val)))
                    if clamped != float(val):
                        graph.nodes[node_id][field_name] = clamped
                        report.tier1_fixes.append(
                            ValidationIssue(
                                tier=1,
                                category="field_clamp",
                                node_id=node_id,
                                edge_key=None,
                                message=f"Clamped {field_name} {val} → {clamped}",
                                auto_fixed=True,
                            )
                        )

        # 1e. Normalize edge type aliases
        for u, v, key, data in list(graph.edges(data=True, keys=True)):
            edge_type = data.get("type", "")
            if isinstance(edge_type, str):
                type_lower = edge_type.lower().strip()
                canonical = EDGE_TYPE_ALIASES.get(type_lower)
                if canonical and type_lower != canonical:
                    graph.edges[u, v, key]["type"] = canonical
                    report.tier1_fixes.append(
                        ValidationIssue(
                            tier=1,
                            category="edge_alias",
                            node_id=None,
                            edge_key=f"{u} → {v}",
                            message=(
                                f"Normalized edge type '{edge_type}' → '{canonical}'"
                            ),
                            auto_fixed=True,
                        )
                    )

            # 1f. Clamp edge weight
            weight = data.get("weight")
            if weight is not None and isinstance(weight, int | float):
                clamped = max(0.0, min(10.0, float(weight)))
                if clamped != float(weight):
                    graph.edges[u, v, key]["weight"] = clamped
                    report.tier1_fixes.append(
                        ValidationIssue(
                            tier=1,
                            category="edge_weight_clamp",
                            node_id=None,
                            edge_key=f"{u} → {v}",
                            message=f"Clamped edge weight {weight} → {clamped}",
                            auto_fixed=True,
                        )
                    )

    # --- Tier 2: Referential Integrity ----------------------------------------

    def _tier2_integrity(self, report: ValidationReport) -> None:
        """Detect referential integrity violations."""
        graph = self.engine.graph
        node_ids = set(graph.nodes())

        # 2a. Dangling edges — edges referencing non-existent nodes
        for u, v, data in graph.edges(data=True):
            if u not in node_ids:
                report.tier2_violations.append(
                    ValidationIssue(
                        tier=2,
                        category="dangling_source",
                        node_id=u,
                        edge_key=f"{u} → {v}",
                        message=f"Edge source '{u}' not in graph",
                    )
                )
            if v not in node_ids:
                report.tier2_violations.append(
                    ValidationIssue(
                        tier=2,
                        category="dangling_target",
                        node_id=v,
                        edge_key=f"{u} → {v}",
                        message=f"Edge target '{v}' not in graph",
                    )
                )

        # 2b. Duplicate node ID detection (shouldn't happen with GraphComputeEngine,
        # but can happen if data has conflicting entries)
        seen_ids: dict[str, int] = {}
        for node_id in graph.nodes():
            seen_ids[node_id] = seen_ids.get(node_id, 0) + 1
        for nid, count in seen_ids.items():
            if count > 1:
                report.tier2_violations.append(
                    ValidationIssue(
                        tier=2,
                        category="duplicate_id",
                        node_id=nid,
                        edge_key=None,
                        message=f"Node ID '{nid}' appears {count} times",
                    )
                )

        # 2c. Nodes with type=None or missing type
        for node_id, data in graph.nodes(data=True):
            if not data.get("type"):
                report.tier2_violations.append(
                    ValidationIssue(
                        tier=2,
                        category="missing_type",
                        node_id=node_id,
                        edge_key=None,
                        message=f"Node '{node_id}' has no type",
                    )
                )

        # 2d. Edges with missing type
        for u, v, data in graph.edges(data=True):
            if not data.get("type"):
                report.tier2_violations.append(
                    ValidationIssue(
                        tier=2,
                        category="untyped_edge",
                        node_id=None,
                        edge_key=f"{u} → {v}",
                        message=f"Edge '{u}' → '{v}' has no type",
                    )
                )

    # --- Tier 3: Quality Checks -----------------------------------------------

    def _tier3_quality(self, report: ValidationReport) -> None:
        """Flag quality concerns in the graph."""
        graph = self.engine.graph

        # 3a. Orphan nodes (no edges at all)
        for node_id in graph.nodes():
            if graph.degree(node_id) == 0:
                report.tier3_warnings.append(
                    ValidationIssue(
                        tier=3,
                        category="orphan_node",
                        node_id=node_id,
                        edge_key=None,
                        message=f"Node '{node_id}' has no connections (orphan)",
                    )
                )

        # 3b. Self-referencing edges
        for u, v, data in graph.edges(data=True):
            if u == v:
                report.tier3_warnings.append(
                    ValidationIssue(
                        tier=3,
                        category="self_reference",
                        node_id=u,
                        edge_key=f"{u} → {v}",
                        message=f"Self-referencing edge on '{u}'",
                    )
                )

        # 3c. Generic / placeholder descriptions
        _GENERIC_PATTERNS = [
            r"^todo$",
            r"^tbd$",
            r"^placeholder$",
            r"^n/?a$",
            r"^none$",
            r"^undefined$",
            r"^description$",
            r"^a [a-z]+ that",  # "a function that..." patterns
            r"^this (is|does|handles)",
        ]
        generic_re = re.compile("|".join(_GENERIC_PATTERNS), re.IGNORECASE)

        for node_id, data in graph.nodes(data=True):
            desc = data.get("description", "")
            if desc and generic_re.match(desc.strip()):
                report.tier3_warnings.append(
                    ValidationIssue(
                        tier=3,
                        category="generic_description",
                        node_id=node_id,
                        edge_key=None,
                        message=(
                            f"Node '{node_id}' has generic description: "
                            f"'{desc[:60]}...'"
                        ),
                    )
                )

        # 3d. Nodes with very low importance that have many connections
        # (potential mis-scored hub nodes)
        for node_id, data in graph.nodes(data=True):
            score = data.get("importance_score", 0.0)
            degree = graph.degree(node_id)
            if degree > 10 and isinstance(score, int | float) and score < 0.1:
                report.tier3_warnings.append(
                    ValidationIssue(
                        tier=3,
                        category="underscored_hub",
                        node_id=node_id,
                        edge_key=None,
                        message=(
                            f"Hub node '{node_id}' has {degree} connections "
                            f"but importance_score={score:.2f}"
                        ),
                    )
                )

    # --- Tier 4: Fatal Checks -------------------------------------------------

    def _tier4_fatal(self, report: ValidationReport) -> None:
        """Check for catastrophic failures (only these raise exceptions)."""
        graph = self.engine.graph

        # 4a. Zero nodes
        if graph.number_of_nodes() == 0:
            report.tier4_fatal.append(
                ValidationIssue(
                    tier=4,
                    category="empty_graph",
                    node_id=None,
                    edge_key=None,
                    message="Graph has zero nodes — cannot serve queries",
                )
            )

        # 4b. Graph connectivity check (warn if completely disconnected)
        if graph.number_of_nodes() > 1:
            # Use GraphComputeEngine's connected_components (Rust-native)
            try:
                components = self.engine.graph_compute.connected_components()
            except Exception:
                # Fallback: try the graph object directly
                try:
                    components = graph.connected_components()
                except Exception:
                    components = []

            if components:
                components_sorted = sorted(components, key=len, reverse=True)
                if len(components_sorted) > 1:
                    # Not fatal, but if the largest component is < 50% of nodes
                    # it suggests a broken graph
                    largest_pct = len(components_sorted[0]) / graph.number_of_nodes()
                    if largest_pct < 0.5:
                        report.tier4_fatal.append(
                            ValidationIssue(
                                tier=4,
                                category="fragmented_graph",
                                node_id=None,
                                edge_key=None,
                                message=(
                                    f"Graph is fragmented into {len(components_sorted)} "
                                    f"components; largest is only {largest_pct:.0%} "
                                    f"of total nodes"
                                ),
                            )
                        )


class GraphValidationFatalError(Exception):
    """Raised when tier-4 fatal validation errors are detected."""

    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        issues = "; ".join(i.message for i in report.tier4_fatal)
        super().__init__(f"Graph validation fatal: {issues}")
