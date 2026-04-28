#!/usr/bin/python
"""Cognitive Consolidation Engine.

Implements the *systems-consolidation* analogue (hippocampus → neocortex,
McClelland, McNaughton & O'Reilly 1995) for the Unified Intelligence Graph.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

from pydantic import BaseModel, Field

if TYPE_CHECKING:  # pragma: no cover - type-only imports
    from .engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rule data models
# ---------------------------------------------------------------------------


ProposedNodeType = Literal[
    "PreferenceNode",
    "PrincipleNode",
    "ConceptEdge",
    "SystemNode",
    "BeliefNode",
]


class ConsolidationProposal(BaseModel):
    """A proposed new V2 node emerging from a consolidation rule.

    Proposals land as ``ProposedSkillNode``-style review items (§4.4) and
    are promoted to real nodes only after explicit approval.
    """

    proposal_id: str
    rule_name: str
    proposed_node_type: ProposedNodeType
    proposed_payload: dict[str, Any]
    evidence_node_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    created_at: str
    status: Literal["pending", "approved", "rejected", "deferred"] = "pending"
    signature: str = Field(
        default="",
        description=(
            "Hash of sorted evidence_node_ids + rule_name; used for "
            "idempotent re-proposal suppression (§4.5)."
        ),
    )

    def compute_signature(self) -> str:
        """Compute a stable hash of (rule, sorted evidence)."""
        payload = "|".join([self.rule_name, *sorted(self.evidence_node_ids)]).encode(
            "utf-8"
        )
        return hashlib.sha256(payload).hexdigest()[:16]


class ConsolidationRule(Protocol):
    """A rule scans the graph and yields zero or more proposals."""

    name: str
    min_evidence_count: int
    min_confidence: float

    def detect(self, engine: IntelligenceGraphEngine) -> list[ConsolidationProposal]:
        """Return zero or more proposals derived from the current graph."""
        ...


# ---------------------------------------------------------------------------
# Rule 1 — Episode-to-Preference (example / skeleton)
# ---------------------------------------------------------------------------


@dataclass
class EpisodeToPreferenceRule:
    """Rule 1 (§4.3) — Episodic → Preference abstraction.

    **Heuristic:** N ≥ ``min_evidence_count`` ``EpisodeNode`` instances that
    all share a single tool / agent used with high outcome reward (≥
    ``reward_threshold``) → propose a ``PreferenceNode`` saying "agent
    prefers tool X for this kind of work".

    This is the *minimum-viable* implementation of the full design doc §4.3
    rule set. It does **not** try to detect shared phase/goal context — a
    follow-up (``rule2_decisions_to_principle``) covers the
    ``PrincipleNode`` proposal path.

    The detector operates off the in-memory NetworkX graph so it does not
    require a live backend connection; that makes it safe to run in tests
    with a pure ``nx.MultiDiGraph`` engine.
    """

    name: str = "episode_to_preference"
    min_evidence_count: int = 5
    min_confidence: float = 0.6
    reward_threshold: float = 0.8

    def detect(self, engine: IntelligenceGraphEngine) -> list[ConsolidationProposal]:
        proposals: list[ConsolidationProposal] = []
        graph = engine.graph

        # Count per-tool co-occurrence of successful episodes.
        tool_to_episode_ids: dict[str, list[str]] = {}

        for episode_id, attrs in graph.nodes(data=True):
            if attrs.get("type") != "episode":
                continue

            # Outgoing edges: EPISODE -[:PRODUCED_OUTCOME]-> OutcomeEvaluation
            # and EPISODE -[:USED_TOOL / USED_RESOURCE]-> ToolCall/Resource.
            outcome_reward: float | None = None
            tool_names: set[str] = set()

            for _src, tgt, edge_attrs in graph.out_edges(episode_id, data=True):
                edge_type = edge_attrs.get("type", "")
                if edge_type == "produced_outcome":
                    outcome_attrs = graph.nodes.get(tgt, {})
                    reward = outcome_attrs.get("reward")
                    if reward is not None:
                        outcome_reward = float(reward)
                elif edge_type in {"used_tool", "used_resource"}:
                    tgt_attrs = graph.nodes.get(tgt, {})
                    # ToolCallNode.tool_name, or fall-back on node name
                    tool_name = tgt_attrs.get("tool_name") or tgt_attrs.get("name")
                    if tool_name:
                        tool_names.add(tool_name)

            if outcome_reward is None or outcome_reward < self.reward_threshold:
                continue

            for tool_name in tool_names:
                tool_to_episode_ids.setdefault(tool_name, []).append(episode_id)

        # Emit one proposal per tool with enough evidence.
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for tool_name, ep_ids in tool_to_episode_ids.items():
            if len(ep_ids) < self.min_evidence_count:
                continue
            # Simple confidence model: more evidence → higher confidence.
            confidence = min(1.0, self.min_confidence + 0.05 * len(ep_ids))
            payload = {
                "category": "tool",
                "value": tool_name,
                "statement": (
                    f"Agent repeatedly succeeded using '{tool_name}' "
                    f"(across {len(ep_ids)} successful episodes)."
                ),
            }
            proposal = ConsolidationProposal(
                proposal_id=hashlib.sha256(
                    f"{self.name}:{tool_name}".encode()
                ).hexdigest()[:8],
                rule_name=self.name,
                proposed_node_type="PreferenceNode",
                proposed_payload=payload,
                evidence_node_ids=sorted(ep_ids),
                confidence=confidence,
                created_at=now,
                status="pending",
            )
            proposal.signature = proposal.compute_signature()
            proposals.append(proposal)

        return proposals


# ---------------------------------------------------------------------------
# Consolidation engine
# ---------------------------------------------------------------------------


@dataclass
class ConsolidationEngine:
    """Runs all registered rules and collects proposals.

    ``dry_run=True`` returns proposals without persisting them, which is the
    recommended mode for initial rollout (§4.5 — idempotence policy).
    Persistence hooks into ``engine.add_consolidation_proposal`` are a
    follow-up (the engine method does not yet exist — persistence is a
    no-op for now, matching the "minimum viable v2" scope).
    """

    engine: IntelligenceGraphEngine
    rules: list[ConsolidationRule] = field(default_factory=list)

    def register(self, rule: ConsolidationRule) -> None:
        """Register a rule to be run on the next ``run()`` call."""
        self.rules.append(rule)

    def run(self, dry_run: bool = True) -> list[ConsolidationProposal]:
        """Run every registered rule and return all proposals.

        Per-rule isolation: a broken rule logs a warning and the run
        continues (§4.2 of the design doc).
        """
        all_proposals: list[ConsolidationProposal] = []
        for rule in self.rules:
            try:
                proposals = rule.detect(self.engine)
                all_proposals.extend(proposals)
            except Exception as exc:  # noqa: BLE001 — per-rule isolation
                logger.warning("Consolidation rule %r failed: %s", rule.name, exc)
        if not dry_run:
            self._persist_proposals(all_proposals)
        return all_proposals

    def _persist_proposals(self, proposals: list[ConsolidationProposal]) -> None:
        """Persist proposals as ProposedSkillNode review items.

        The actual engine integration (``engine.add_consolidation_proposal``)
        is the follow-up deliverable; for now we log each proposal so
        test-time observability is preserved.
        """
        for p in proposals:
            logger.info(
                "Consolidation proposal %s (rule=%s, type=%s, "
                "confidence=%.2f, evidence=%d)",
                p.proposal_id,
                p.rule_name,
                p.proposed_node_type,
                p.confidence,
                len(p.evidence_node_ids),
            )

    # Convenience ----------------------------------------------------------

    def dedup_by_signature(
        self, proposals: list[ConsolidationProposal]
    ) -> list[ConsolidationProposal]:
        """Return proposals de-duplicated by their ``signature`` field."""
        seen: set[str] = set()
        out: list[ConsolidationProposal] = []
        for p in proposals:
            sig = p.signature or p.compute_signature()
            if sig in seen:
                continue
            seen.add(sig)
            out.append(p)
        return out
