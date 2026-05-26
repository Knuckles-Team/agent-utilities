#!/usr/bin/python
from __future__ import annotations

"""Governance Workflow — Unified Policy Enforcement Pipeline.

CONCEPT:ECO-4.22 — Governance Workflow Pipeline

Orchestrates the full governance lifecycle across the ecosystem:

1. **Policy Loading**: Loads permission policies, AGENTS.md rules, and
   constitution policies from disk and KG into a unified rule set.
2. **Change Proposal Gate**: Intercepts ecosystem mutations (hook installs,
   plugin additions, AGENTS.md edits) and routes them through approval.
3. **Audit Trail**: Persists every governance decision as a KG node for
   post-hoc compliance review.
4. **Periodic Review**: Coordinates staleness audits and reflector proposals
   into a unified review cycle.

Integrates with:
    - ``permission_policy.py`` (ECO-4.5) — File/tool deny/allow rules
    - ``permissions_kernel.py`` (OS-5.1) — Identity-based RBAC
    - ``approval_manager.py`` (OS-5.1) — Async human-in-the-loop
    - ``policy_ingestor.py`` (KG-2.2) — Constitution → KG
    - ``agents_md_reflector.py`` (ECO-4.2) — Self-improving stop hook
    - ``config_staleness_auditor.py`` (ECO-4.6) — Periodic cleanup
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..knowledge_graph.core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)

__all__ = [
    "GovernanceWorkflow",
    "GovernanceDecision",
    "ChangeProposal",
    "ChangeType",
    "GovernanceReport",
]


class ChangeType(StrEnum):
    """Types of ecosystem mutations that require governance review."""

    AGENTS_MD_EDIT = "agents_md_edit"
    HOOK_INSTALL = "hook_install"
    HOOK_UNINSTALL = "hook_uninstall"
    PLUGIN_INSTALL = "plugin_install"
    PLUGIN_UNINSTALL = "plugin_uninstall"
    PERMISSION_CHANGE = "permission_change"
    POLICY_UPDATE = "policy_update"
    CONSTITUTION_AMEND = "constitution_amend"
    SKILL_INSTALL = "skill_install"
    TOOL_REGISTRATION = "tool_registration"


class ApprovalStatus(StrEnum):
    """Status of a governance decision."""

    PENDING = "pending"
    AUTO_APPROVED = "auto_approved"
    HUMAN_APPROVED = "human_approved"
    HUMAN_REJECTED = "human_rejected"
    POLICY_DENIED = "policy_denied"


@dataclass
class ChangeProposal:
    """A proposed ecosystem mutation requiring governance review.

    Every mutation to the agent ecosystem (new hooks, AGENTS.md edits,
    plugin installs, permission changes) must be wrapped in a
    ChangeProposal and submitted to the GovernanceWorkflow.
    """

    id: str = field(default_factory=lambda: f"chg-{uuid.uuid4().hex[:12]}")
    change_type: ChangeType = ChangeType.AGENTS_MD_EDIT
    title: str = ""
    description: str = ""
    proposed_by: str = ""  # agent_id or "human"
    target_path: str = ""  # File or resource affected
    diff: str = ""  # Textual diff of the change
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    risk_score: float = 0.0  # 0.0 = safe, 1.0 = high risk
    confidence: float = 0.5


@dataclass
class GovernanceDecision:
    """The result of governance review on a ChangeProposal."""

    proposal_id: str = ""
    status: ApprovalStatus = ApprovalStatus.PENDING
    decided_by: str = ""  # "policy_engine", "human:<user>", "auto"
    reason: str = ""
    conditions: list[str] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "status": self.status,
            "decided_by": self.decided_by,
            "reason": self.reason,
            "conditions": self.conditions,
            "timestamp": self.timestamp,
        }


@dataclass
class GovernanceReport:
    """Summary report of governance activity over a period."""

    period_start: str = ""
    period_end: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    total_proposals: int = 0
    auto_approved: int = 0
    human_approved: int = 0
    human_rejected: int = 0
    policy_denied: int = 0
    pending: int = 0
    decisions: list[dict[str, Any]] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            f"# Governance Report — {self.period_end[:10]}",
            "",
            f"**Period**: {self.period_start[:10] or 'inception'} → {self.period_end[:10]}",
            "",
            "| Metric | Count |",
            "|---|---|",
            f"| Total Proposals | {self.total_proposals} |",
            f"| Auto-Approved | {self.auto_approved} |",
            f"| Human Approved | {self.human_approved} |",
            f"| Human Rejected | {self.human_rejected} |",
            f"| Policy Denied | {self.policy_denied} |",
            f"| Pending | {self.pending} |",
            "",
        ]
        if self.decisions:
            lines.append("## Recent Decisions\n")
            for d in self.decisions[-20:]:
                status = d.get("status", "?")
                emoji = {
                    "auto_approved": "✅",
                    "human_approved": "✅",
                    "human_rejected": "❌",
                    "policy_denied": "🚫",
                }.get(status, "⏳")
                lines.append(
                    f"- {emoji} **{d.get('proposal_id', '?')}** — "
                    f"{d.get('reason', 'No reason')} ({status})"
                )
        return "\n".join(lines)


# ── Risk scoring rules ─────────────────────────────────────────────

_RISK_WEIGHTS: dict[ChangeType, float] = {
    ChangeType.CONSTITUTION_AMEND: 0.9,
    ChangeType.PERMISSION_CHANGE: 0.8,
    ChangeType.POLICY_UPDATE: 0.7,
    ChangeType.HOOK_INSTALL: 0.5,
    ChangeType.HOOK_UNINSTALL: 0.6,
    ChangeType.PLUGIN_INSTALL: 0.4,
    ChangeType.PLUGIN_UNINSTALL: 0.5,
    ChangeType.AGENTS_MD_EDIT: 0.3,
    ChangeType.SKILL_INSTALL: 0.3,
    ChangeType.TOOL_REGISTRATION: 0.2,
}

# Auto-approve threshold: proposals below this risk score are auto-approved
_AUTO_APPROVE_THRESHOLD = 0.4


class GovernanceWorkflow:
    """Unified governance pipeline for ecosystem mutations.

    CONCEPT:ECO-4.22 — Governance Workflow Pipeline

    Usage::

        gov = GovernanceWorkflow(engine, workspace="/my/project")

        # Submit a change
        proposal = ChangeProposal(
            change_type=ChangeType.HOOK_INSTALL,
            title="Install lint hook",
            proposed_by="claude",
        )
        decision = gov.submit(proposal)

        # Review pending proposals
        pending = gov.list_pending()

        # Human approves
        gov.approve(proposal.id, reviewer="admin")

        # Generate report
        report = gov.generate_report()
    """

    def __init__(
        self,
        engine: IntelligenceGraphEngine | None = None,
        workspace: str | Path = ".",
        auto_approve_threshold: float = _AUTO_APPROVE_THRESHOLD,
    ) -> None:
        self.engine = engine
        self.workspace = Path(workspace).resolve()
        self.auto_approve_threshold = auto_approve_threshold
        self._proposals: dict[str, ChangeProposal] = {}
        self._decisions: dict[str, GovernanceDecision] = {}

    # ── Submission ─────────────────────────────────────────────────

    def submit(self, proposal: ChangeProposal) -> GovernanceDecision:
        """Submit a change proposal for governance review.

        Calculates risk score, checks policy rules, and either
        auto-approves (low risk) or queues for human review (high risk).

        Args:
            proposal: The change proposal to review.

        Returns:
            A ``GovernanceDecision`` with the initial ruling.
        """
        # Calculate risk score
        proposal.risk_score = self._calculate_risk(proposal)
        self._proposals[proposal.id] = proposal

        # Check policy-level denials first
        denial_reason = self._check_policy_denial(proposal)
        if denial_reason:
            decision = GovernanceDecision(
                proposal_id=proposal.id,
                status=ApprovalStatus.POLICY_DENIED,
                decided_by="policy_engine",
                reason=denial_reason,
            )
            self._decisions[proposal.id] = decision
            self._persist_decision(proposal, decision)
            logger.info("[ECO-4.9] DENIED %s: %s", proposal.id, denial_reason)
            return decision

        # Auto-approve low-risk changes
        if proposal.risk_score < self.auto_approve_threshold:
            decision = GovernanceDecision(
                proposal_id=proposal.id,
                status=ApprovalStatus.AUTO_APPROVED,
                decided_by="governance_workflow",
                reason=f"Low risk ({proposal.risk_score:.2f} < {self.auto_approve_threshold})",
            )
            self._decisions[proposal.id] = decision
            self._persist_decision(proposal, decision)
            logger.info(
                "[ECO-4.9] AUTO-APPROVED %s (risk=%.2f)",
                proposal.id,
                proposal.risk_score,
            )
            return decision

        # Queue for human review
        decision = GovernanceDecision(
            proposal_id=proposal.id,
            status=ApprovalStatus.PENDING,
            decided_by="",
            reason=f"Risk score {proposal.risk_score:.2f} exceeds auto-approve threshold",
        )
        self._decisions[proposal.id] = decision
        self._persist_decision(proposal, decision)
        logger.info(
            "[ECO-4.9] PENDING %s (risk=%.2f)", proposal.id, proposal.risk_score
        )
        return decision

    # ── Human Review ───────────────────────────────────────────────

    def approve(
        self,
        proposal_id: str,
        reviewer: str = "human",
        conditions: list[str] | None = None,
    ) -> GovernanceDecision:
        """Approve a pending proposal.

        Args:
            proposal_id: ID of the proposal to approve.
            reviewer: Identity of the reviewer.
            conditions: Optional conditions attached to approval.

        Returns:
            Updated ``GovernanceDecision``.

        Raises:
            KeyError: If proposal_id is not found.
        """
        if proposal_id not in self._decisions:
            raise KeyError(f"Proposal {proposal_id} not found")

        decision = self._decisions[proposal_id]
        decision.status = ApprovalStatus.HUMAN_APPROVED
        decision.decided_by = f"human:{reviewer}"
        decision.reason = "Approved by human reviewer"
        decision.conditions = conditions or []
        decision.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        proposal = self._proposals.get(proposal_id)
        if proposal:
            self._persist_decision(proposal, decision)
        logger.info("[ECO-4.9] APPROVED %s by %s", proposal_id, reviewer)
        return decision

    def reject(
        self,
        proposal_id: str,
        reviewer: str = "human",
        reason: str = "Rejected by reviewer",
    ) -> GovernanceDecision:
        """Reject a pending proposal.

        Args:
            proposal_id: ID of the proposal to reject.
            reviewer: Identity of the reviewer.
            reason: Rejection reason.

        Returns:
            Updated ``GovernanceDecision``.
        """
        if proposal_id not in self._decisions:
            raise KeyError(f"Proposal {proposal_id} not found")

        decision = self._decisions[proposal_id]
        decision.status = ApprovalStatus.HUMAN_REJECTED
        decision.decided_by = f"human:{reviewer}"
        decision.reason = reason
        decision.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        proposal = self._proposals.get(proposal_id)
        if proposal:
            self._persist_decision(proposal, decision)
        logger.info("[ECO-4.9] REJECTED %s by %s: %s", proposal_id, reviewer, reason)
        return decision

    # ── Queries ─────────────────────────────────────────────────────

    def list_pending(self) -> list[ChangeProposal]:
        """Return all proposals awaiting human review."""
        return [
            self._proposals[pid]
            for pid, d in self._decisions.items()
            if d.status == ApprovalStatus.PENDING and pid in self._proposals
        ]

    def get_decision(self, proposal_id: str) -> GovernanceDecision | None:
        """Look up the decision for a proposal."""
        return self._decisions.get(proposal_id)

    def is_approved(self, proposal_id: str) -> bool:
        """Check if a proposal has been approved (auto or human)."""
        d = self._decisions.get(proposal_id)
        if not d:
            return False
        return d.status in (ApprovalStatus.AUTO_APPROVED, ApprovalStatus.HUMAN_APPROVED)

    # ── Reporting ──────────────────────────────────────────────────

    def generate_report(self) -> GovernanceReport:
        """Generate a governance activity report."""
        report = GovernanceReport(total_proposals=len(self._decisions))

        for d in self._decisions.values():
            if d.status == ApprovalStatus.AUTO_APPROVED:
                report.auto_approved += 1
            elif d.status == ApprovalStatus.HUMAN_APPROVED:
                report.human_approved += 1
            elif d.status == ApprovalStatus.HUMAN_REJECTED:
                report.human_rejected += 1
            elif d.status == ApprovalStatus.POLICY_DENIED:
                report.policy_denied += 1
            elif d.status == ApprovalStatus.PENDING:
                report.pending += 1
            report.decisions.append(d.to_dict())

        # Write to disk
        report_dir = self.workspace / ".agents" / "governance"
        report_dir.mkdir(parents=True, exist_ok=True)
        date = time.strftime("%Y-%m-%d", time.gmtime())
        fp = report_dir / f"governance_{date}.md"
        fp.write_text(report.to_markdown(), encoding="utf-8")

        return report

    # ── Integrated Audit Cycle ─────────────────────────────────────

    def run_audit_cycle(self) -> dict[str, Any]:
        """Run a full governance audit cycle.

        Coordinates:
        1. Config staleness audit (ECO-4.6)
        2. AGENTS.md reflector proposals (ECO-4.2)
        3. Permission policy review
        4. Generate combined governance report

        Returns:
            Combined audit results.
        """
        results: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }

        # 1. Staleness audit
        try:
            from .config_staleness_auditor import ConfigStalenessAuditor

            auditor = ConfigStalenessAuditor(
                engine=self.engine, workspace=self.workspace
            )
            if auditor.should_run():
                staleness = auditor.run_audit()
                results["staleness"] = staleness.to_dict()

                # Submit removal proposals for stale items
                for item in staleness.remove_items:
                    proposal = ChangeProposal(
                        change_type=ChangeType.AGENTS_MD_EDIT,
                        title=f"Remove stale {item.item_type}: {item.name}",
                        description=item.reason,
                        proposed_by="staleness_auditor",
                        confidence=item.confidence,
                    )
                    self.submit(proposal)
            else:
                results["staleness"] = "skipped (not due)"
        except Exception as e:
            logger.debug("[ECO-4.9] Staleness audit failed: %s", e)
            results["staleness"] = f"error: {e}"

        # 2. Check for pending reflector proposals in KG
        try:
            if self.engine:
                pending = self.engine.query_cypher(
                    "MATCH (p) WHERE p.node_type = 'agents_md_proposal' "
                    "AND p.applied = false RETURN p.node_id as id, "
                    "p.section as section, p.proposed_content as content "
                    "LIMIT 10"
                )
                results["reflector_proposals"] = len(pending)
                for row in pending:
                    proposal = ChangeProposal(
                        change_type=ChangeType.AGENTS_MD_EDIT,
                        title=f"Reflector: update {row.get('section', '?')}",
                        description=str(row.get("content", ""))[:200],
                        proposed_by="agents_md_reflector",
                    )
                    self.submit(proposal)
        except Exception as e:
            logger.debug("[ECO-4.9] Reflector check failed: %s", e)
            results["reflector_proposals"] = 0

        # 3. Generate combined report
        report = self.generate_report()
        results["report"] = report.to_markdown()

        logger.info(
            "[ECO-4.9] Audit cycle complete: %d proposals processed",
            report.total_proposals,
        )
        return results

    # ── Private ────────────────────────────────────────────────────

    def _calculate_risk(self, proposal: ChangeProposal) -> float:
        """Calculate risk score from change type and content analysis."""
        base = _RISK_WEIGHTS.get(proposal.change_type, 0.5)

        # Modifiers
        if proposal.proposed_by == "human":
            base *= 0.7  # Human-initiated changes are lower risk
        if "delete" in proposal.title.lower() or "remove" in proposal.title.lower():
            base = min(base + 0.2, 1.0)
        if "constitution" in proposal.target_path.lower():
            base = min(base + 0.3, 1.0)
        if proposal.confidence > 0.8:
            base *= 0.8  # High confidence reduces risk

        return round(min(max(base, 0.0), 1.0), 3)

    def _check_policy_denial(self, proposal: ChangeProposal) -> str:
        """Check if any loaded policy explicitly denies this change."""
        # Constitution amendments always require human review (never auto-denied)
        # but we check for structural violations
        if proposal.change_type == ChangeType.CONSTITUTION_AMEND:
            if not proposal.proposed_by or proposal.proposed_by == "unknown":
                return "Constitution amendments require identified proposer"

        # Permission changes from sandbox/guest agents are denied
        if proposal.change_type == ChangeType.PERMISSION_CHANGE:
            if self.engine:
                try:
                    results = self.engine.query_cypher(
                        "MATCH (i) WHERE i.node_type = 'AgentIdentityNode' "
                        "AND i.name CONTAINS $agent "
                        "RETURN i.role as role",
                        {"agent": proposal.proposed_by},
                    )
                    if results and results[0].get("role") in ("sandbox", "guest"):
                        return (
                            f"Agent '{proposal.proposed_by}' with role "
                            f"'{results[0]['role']}' cannot modify permissions"
                        )
                except Exception:
                    pass

        return ""  # No denial

    def _persist_decision(
        self, proposal: ChangeProposal, decision: GovernanceDecision
    ) -> None:
        """Persist a governance decision to the KG."""
        if not self.engine:
            return
        try:
            node_id = f"gov_decision:{proposal.id}"
            self.engine.add_node(
                node_id,
                "governance_decision",
                {
                    "name": f"Governance: {proposal.title[:60]}",
                    "description": decision.reason,
                    "proposal_id": proposal.id,
                    "change_type": str(proposal.change_type),
                    "status": str(decision.status),
                    "decided_by": decision.decided_by,
                    "risk_score": proposal.risk_score,
                    "proposed_by": proposal.proposed_by,
                    "timestamp": decision.timestamp,
                },
            )
        except Exception as e:
            logger.debug("[ECO-4.9] KG persist failed: %s", e)
