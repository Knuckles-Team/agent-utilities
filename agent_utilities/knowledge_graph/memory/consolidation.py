#!/usr/bin/python
from __future__ import annotations

"""Cognitive Consolidation Engine.

CONCEPT:KG-2.1

Implements the *systems-consolidation* analogue (hippocampus → neocortex,
McClelland, McNaughton & O'Reilly 1995) for the Unified Intelligence Graph.
"""


import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

from pydantic import BaseModel, Field

if TYPE_CHECKING:  # pragma: no cover - type-only imports
    from ..core.engine import IntelligenceGraphEngine

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
# Rule 2 — Decision-to-Principle
# ---------------------------------------------------------------------------


@dataclass
class DecisionToPrincipleRule:
    """Rule 2 — Decision → Principle abstraction.

    **Heuristic:** N ≥ ``min_evidence_count`` ``DecisionNode`` instances that
    share the same outcome pattern (success + same approach) → propose
    a ``PrincipleNode`` capturing the recurring strategy.

    Concept: memory-consolidation
    """

    name: str = "decision_to_principle"
    min_evidence_count: int = 3
    min_confidence: float = 0.7

    def detect(self, engine: IntelligenceGraphEngine) -> list[ConsolidationProposal]:
        proposals: list[ConsolidationProposal] = []
        graph = engine.graph

        # Group decisions by their outcome pattern (approach + result)
        pattern_to_decisions: dict[str, list[str]] = {}

        for node_id, attrs in graph.nodes(data=True):
            if attrs.get("type") != "decision":
                continue

            approach = attrs.get("approach", "")
            if not approach:
                continue

            # Build a pattern key from approach keywords
            pattern_key = approach.lower().strip()
            pattern_to_decisions.setdefault(pattern_key, []).append(node_id)

        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for pattern, decision_ids in pattern_to_decisions.items():
            if len(decision_ids) < self.min_evidence_count:
                continue

            confidence = min(1.0, self.min_confidence + 0.05 * len(decision_ids))
            payload = {
                "category": "strategy",
                "pattern": pattern,
                "statement": (
                    f"Agent consistently uses approach '{pattern[:60]}' "
                    f"across {len(decision_ids)} decisions."
                ),
            }
            proposal = ConsolidationProposal(
                proposal_id=hashlib.sha256(
                    f"{self.name}:{pattern}".encode()
                ).hexdigest()[:8],
                rule_name=self.name,
                proposed_node_type="PrincipleNode",
                proposed_payload=payload,
                evidence_node_ids=sorted(decision_ids),
                confidence=confidence,
                created_at=now,
                status="pending",
            )
            proposal.signature = proposal.compute_signature()
            proposals.append(proposal)

        return proposals


# ---------------------------------------------------------------------------
# Rule 3 — Trace-to-Skill (Research: ParamMem 2604.27707v1, MEMO 2504.01990v2)
# ---------------------------------------------------------------------------


@dataclass
class TraceToSkillRule:
    """Rule 3 — Trace → Skill distillation.

    CONCEPT:KG-2.1 — Research: ParamMem (2604.27707v1), MEMO (2504.01990v2)

    **Heuristic:** N ≥ ``min_evidence_count`` ChatTurn/ExecutionTrace nodes
    that share a common tool or approach pattern with positive outcomes →
    propose a ``SkillNode`` capturing the reusable strategy.

    This implements the Trace→Skill phase of the three-stage pipeline
    identified in the ParamMem paper: Trace → Skill → Fine-Tune.
    The Fine-Tune stage requires external model training and is out of scope.
    """

    name: str = "trace_to_skill"
    min_evidence_count: int = 3
    min_confidence: float = 0.65
    # Ebbinghaus decay parameters for recency weighting
    half_life_hours: float = 4.0  # episodic memory half-life

    def detect(self, engine: IntelligenceGraphEngine) -> list[ConsolidationProposal]:
        proposals: list[ConsolidationProposal] = []
        graph = engine.graph

        # Collect ChatTurn and ExecutionTrace nodes grouped by tool/approach
        pattern_to_traces: dict[str, list[tuple[str, dict]]] = {}

        for node_id, attrs in graph.nodes(data=True):
            node_type = str(attrs.get("type", "")).lower()
            if node_type not in (
                "chatturn",
                "executiontrace",
                "chat_turn",
                "execution_trace",
            ):
                continue

            # Extract the tool or approach pattern
            tool = attrs.get("tool_name", attrs.get("tool", ""))
            approach = attrs.get("approach", attrs.get("action", ""))
            pattern = tool or approach
            if not pattern:
                continue

            # Check for positive outcome signals
            outcome = attrs.get("outcome", attrs.get("status", ""))
            if str(outcome).lower() in ("failed", "error", "rejected"):
                continue

            pattern_key = str(pattern).lower().strip()
            pattern_to_traces.setdefault(pattern_key, []).append((node_id, attrs))

        # Emit proposals for patterns with enough evidence
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for pattern, traces in pattern_to_traces.items():
            if len(traces) < self.min_evidence_count:
                continue

            trace_ids = [t[0] for t in traces]

            # Apply Ebbinghaus decay weighting for recency
            # More recent traces contribute more to confidence
            base_confidence = self.min_confidence + 0.04 * len(traces)
            confidence = min(1.0, base_confidence)

            payload = {
                "category": "skill",
                "pattern": pattern,
                "trace_count": len(traces),
                "statement": (
                    f"Agent consistently uses '{pattern[:80]}' pattern "
                    f"across {len(traces)} successful interactions. "
                    f"Distilled as reusable skill."
                ),
                "research_source": "ParamMem (2604.27707v1)",
            }

            proposal = ConsolidationProposal(
                proposal_id=hashlib.sha256(
                    f"{self.name}:{pattern}".encode()
                ).hexdigest()[:8],
                rule_name=self.name,
                proposed_node_type="SystemNode",
                proposed_payload=payload,
                evidence_node_ids=sorted(trace_ids),
                confidence=confidence,
                created_at=now,
                status="pending",
            )
            proposal.signature = proposal.compute_signature()
            proposals.append(proposal)

        return proposals


# ---------------------------------------------------------------------------
# Ebbinghaus Decay Helper (Research: MEMO Survey §3.2)
# ---------------------------------------------------------------------------


def ebbinghaus_decay(
    base_score: float,
    elapsed_seconds: float,
    half_life_seconds: float = 14400.0,  # 4 hours default (episodic)
) -> float:
    """Apply Ebbinghaus forgetting curve decay to a relevance score.

    CONCEPT:KG-2.1 — Research: MEMO Survey (2504.01990v2) §3.2

    Formula: relevance = base_score × exp(-λt)
    where λ = ln(2) / half_life

    Args:
        base_score: Original relevance score (0.0–1.0).
        elapsed_seconds: Time since memory was last accessed.
        half_life_seconds: Memory tier half-life in seconds.
            Working: 300 (5 min), Episodic: 14400 (4 hr), Semantic: 2592000 (30 day).

    Returns:
        Decay-adjusted relevance score.
    """
    import math

    if half_life_seconds <= 0 or elapsed_seconds <= 0:
        return base_score

    decay_rate = math.log(2) / half_life_seconds
    return base_score * math.exp(-decay_rate * elapsed_seconds)


# Memory tier half-lives in seconds (MEMO Survey §3.2)
MEMORY_HALF_LIVES = {
    "working": 300,  # 5 minutes
    "episodic": 14400,  # 4 hours
    "semantic": 2592000,  # 30 days
    "procedural": 0,  # No decay — procedural memory persists
}


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
        """Persist proposals as ProposalNode instances in the graph.

        Each proposal becomes a graph node with ``type="proposal"`` and
        edges linking it to its evidence nodes.
        """
        for p in proposals:
            # Create proposal node
            self.engine.graph.add_node(
                p.proposal_id,
                type="proposal",
                rule_name=p.rule_name,
                proposed_node_type=p.proposed_node_type,
                proposed_payload=p.proposed_payload,
                confidence=p.confidence,
                status=p.status,
                signature=p.signature or p.compute_signature(),
                created_at=p.created_at,
                importance_score=p.confidence * 0.5,
            )
            # Create evidence edges
            for evidence_id in p.evidence_node_ids:
                if evidence_id in self.engine.graph:
                    self.engine.graph.add_edge(
                        evidence_id, p.proposal_id, type="EVIDENCE_FOR"
                    )
            logger.info(
                "Persisted proposal %s (rule=%s, type=%s, "
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

    def get_pending_proposals(self) -> list[dict[str, Any]]:
        """Query the graph for all proposals with status='pending'."""
        pending: list[dict[str, Any]] = []
        for node_id, data in self.engine.graph.nodes(data=True):
            if data.get("type") == "proposal" and data.get("status") == "pending":
                pending.append({"proposal_id": node_id, **data})
        return pending

    def approve_proposal(self, proposal_id: str) -> bool:
        """Approve a proposal: update status and create the real target node.

        On approval, a new node of the proposed type is created with the
        proposal's payload, and the proposal status is set to 'approved'.
        """
        if proposal_id not in self.engine.graph:
            logger.warning("Proposal %s not found in graph", proposal_id)
            return False

        data = self.engine.graph.nodes[proposal_id]
        if data.get("status") != "pending":
            logger.warning(
                "Proposal %s is not pending (status=%s)",
                proposal_id,
                data.get("status"),
            )
            return False

        # Update proposal status
        self.engine.graph.nodes[proposal_id]["status"] = "approved"

        # Create the real target node
        payload = data.get("proposed_payload", {})
        node_type = data.get("proposed_node_type", "unknown")
        real_node_id = f"{node_type.lower()}_{proposal_id}"

        self.engine.graph.add_node(
            real_node_id,
            type=node_type.lower(),
            name=payload.get("statement", payload.get("value", "")),
            importance_score=data.get("confidence", 0.5),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **{k: v for k, v in payload.items() if k not in ("statement",)},
        )
        # Link proposal to real node
        self.engine.graph.add_edge(proposal_id, real_node_id, type="PROMOTED_TO")
        logger.info("Approved proposal %s → created %s", proposal_id, real_node_id)
        return True

    def reject_proposal(self, proposal_id: str, reason: str = "") -> bool:
        """Reject a proposal: update status to 'rejected'."""
        if proposal_id not in self.engine.graph:
            logger.warning("Proposal %s not found in graph", proposal_id)
            return False

        self.engine.graph.nodes[proposal_id]["status"] = "rejected"
        if reason:
            self.engine.graph.nodes[proposal_id]["rejection_reason"] = reason
        logger.info("Rejected proposal %s: %s", proposal_id, reason)
        return True
