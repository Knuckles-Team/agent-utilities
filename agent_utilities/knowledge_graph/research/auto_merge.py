#!/usr/bin/python
from __future__ import annotations

"""Governed golden-loop auto-merge (CONCEPT:AU-AHE.assimilation.research-auto-merge).

The loop engine is *propose-only* by design (``loop_controller.py``): it synthesizes
``TeamSpec`` / ``AgentSpec`` / ``PromptSpec`` proposals and persists them as KG
nodes, but never promotes them to *active* skills/prompts — promotion is left to
a human. This module adds the missing GOVERNED auto-merge path: a proposal that
passes a quality + governance gate is promoted ``proposal → active`` automatically
and audited; one that fails stays proposal-only (human-gated).

It is deliberately conservative and OFF by default:

  - the default quality threshold is high (``0.85``),
  - SHACL/governance validity is required (when a validator is reachable),
  - regression is checked (a promoted artifact must not lower a tracked metric),
  - the whole step is opt-in via ``LoopController(auto_merge=...)`` /
    ``KG_GOLDEN_AUTO_MERGE=1`` — the existing propose-only safety is the default
    unless explicitly enabled,
  - the merger's own promotion decision consults the operational
    :class:`~agent_utilities.orchestration.action_policy.ActionPolicy`
    (CONCEPT:AU-OS.deployment.fleet-lifecycle-control) under the reserved ``merge_promotion`` kind before the
    lifecycle flip (the AHE-3.20 adoption that was previously a noted
    follow-up): ``deny`` blocks promotion outright (recorded + audited),
    ``queue_approval`` files/reuses the SAME ``ActionApproval`` the AHE-3.21
    publication step consumes (deduped per kind+target) while the KG-internal
    lifecycle flip proceeds — the real-world materialization stays
    human-gated — and ``allow``/``allow_notify`` proceed (the policy itself
    emits the notification).

Every promotion (and every rejection) emits an
:class:`~agent_utilities.observability.audit_logger.AuditLogger` entry, so the
auto-merge trail is queryable.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from agent_utilities.observability.audit_logger import AuditLogger

logger = logging.getLogger(__name__)

AUDIT_AUTO_MERGE = "loop_engine.auto_merge"

#: Conservative default — only very high-quality proposals auto-merge.
DEFAULT_QUALITY_THRESHOLD = 0.85


def _probe_production_validator(engine: Any, policy: MergePolicy) -> Any:
    """Build the default production governance validator (CONCEPT:AU-AHE.harness.promotion-governance-validator).

    Best-effort: returns ``None`` when the validator cannot be constructed, in
    which case the merger falls back to its hold-when-required behaviour.
    """
    try:
        from .promotion_governance import PromotionGovernanceValidator

        return PromotionGovernanceValidator(engine, policy=policy)
    except Exception as exc:  # noqa: BLE001
        logger.debug("production governance validator unavailable: %s", exc)
        return None


@dataclass
class _FailClosedDecision:
    """Deny-shaped stand-in when the OS-5.24 gate itself cannot be consulted."""

    reason: str
    decision: str = "deny"
    approval_id: str | None = None


@dataclass
class MergePolicy:
    """The governance policy for auto-merging a proposal. CONCEPT:AU-AHE.assimilation.research-auto-merge.

    Attributes:
        enabled: Master switch. When ``False`` (default) nothing is promoted —
            the loop stays propose-only.
        quality_threshold: Minimum quality score [0,1] to qualify for merge.
        require_governance_valid: Require SHACL/governance validity (when a
            validator is reachable) before promoting.
        require_no_regression: Reject a merge that would regress a tracked metric.
    """

    enabled: bool = False
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD
    require_governance_valid: bool = True
    require_no_regression: bool = True

    @classmethod
    def from_env(cls, enabled: bool | None = None) -> MergePolicy:
        """Build a policy from config, defaulting to conservative propose-only.

        ``enabled`` (explicit) wins; otherwise ``config.kg_golden_auto_merge``.
        The threshold is ``config.kg_golden_merge_threshold`` (falls back to the
        default when unset).
        """
        # Fresh config (reads env at construction) so this reflects the current
        # environment, as the name implies — routed through the typed
        # AgentConfig fields rather than bare os.environ.
        from agent_utilities.core.config import AgentConfig

        cfg = AgentConfig()
        if enabled is None:
            enabled = cfg.kg_golden_auto_merge
        threshold = cfg.kg_golden_merge_threshold
        if threshold is None:
            threshold = DEFAULT_QUALITY_THRESHOLD
        return cls(enabled=bool(enabled), quality_threshold=float(threshold))


@dataclass
class MergeEvaluation:
    """The result of evaluating one proposal for auto-merge. CONCEPT:AU-AHE.assimilation.research-auto-merge."""

    proposal_id: str
    quality_score: float
    governance_valid: bool
    no_regression: bool
    merged: bool = False
    reason: str = ""
    audit_ref: str = ""
    failures: list[str] = field(default_factory=list)
    #: Evolution→branch bridge outcome (CONCEPT:AU-AHE.harness.evolution-branch-bridge): the governed_publish
    #: report for a merged proposal — ``status`` is ``approval_queued`` under
    #: the shipped ActionPolicy, ``published`` once allowed (branch + sha +
    #: gate verdict inside), ``None`` when the proposal did not merge.
    publication: dict[str, Any] | None = None
    #: Operational ActionPolicy verdict on the promotion itself (CONCEPT:AU-OS.deployment.fleet-lifecycle-control,
    #: the AHE-3.20 adoption): ``{"decision", "reason", "approval_id"}`` when
    #: the ``merge_promotion`` gate was consulted, ``None`` when promotion was
    #: never attempted (disabled policy / ineligible proposal).
    action_decision: dict[str, Any] | None = None

    @property
    def eligible(self) -> bool:
        """True when all gates passed (independent of whether merge was enabled)."""
        return not self.failures


class GovernedAutoMerger:
    """Evaluates + (when governed) promotes golden-loop proposals. CONCEPT:AU-AHE.assimilation.research-auto-merge.

    Parameters
    ----------
    engine:
        The KG engine whose ``backend`` persists artifacts (proposal + active).
    policy:
        The :class:`MergePolicy` (defaults to env-resolved, conservative).
    audit:
        Audit logger for the promotion/rejection trail.
    governance_validator:
        Optional ``(spec) -> bool`` validity check (SHACL/governance). When
        ``None`` and an ``engine`` is available, the production
        :class:`~agent_utilities.knowledge_graph.research.promotion_governance.PromotionGovernanceValidator`
        is constructed as the default (CONCEPT:AU-AHE.harness.promotion-governance-validator): SHACL shapes,
        recorded regression-gate verdicts, MergePolicy thresholds, and
        constitution rules. If no validator can be built, governance validity
        defaults to ``True`` only when the policy does not *require* it
        (otherwise the proposal is held).
    regression_check:
        Optional ``(spec) -> bool`` returning ``True`` when promoting does NOT
        regress a tracked metric. Defaults to ``True`` (no regression known).
    promoter:
        Optional ``(spec) -> bool`` that performs the actual proposal→active
        promotion and returns success. Defaults to :meth:`_default_promote`.
    publisher:
        Optional :class:`~agent_utilities.knowledge_graph.research.change_publisher.ChangePublisher`
        for the evolution→branch bridge (CONCEPT:AU-AHE.harness.evolution-branch-bridge). ``None`` resolves
        the registered/default publisher at publication time.
    action_policy:
        Optional :class:`~agent_utilities.orchestration.action_policy.ActionPolicy`
        override (CONCEPT:AU-OS.deployment.fleet-lifecycle-control). ``None`` resolves the engine-bound policy
        at decision time. The merger consults it with
        ``kind="merge_promotion"`` BEFORE the lifecycle flip: ``deny`` blocks
        the promotion (recorded), ``queue_approval`` proceeds with the
        KG-internal flip while the publication step stays approval-gated
        (same deduped ``ActionApproval``), ``allow``/``allow_notify`` proceed.
    """

    def __init__(
        self,
        engine: Any = None,
        *,
        policy: MergePolicy | None = None,
        audit: AuditLogger | None = None,
        governance_validator: Any = None,
        regression_check: Any = None,
        promoter: Any = None,
        publisher: Any = None,
        action_policy: Any = None,
    ) -> None:
        self.engine = engine
        self.policy = policy or MergePolicy.from_env()
        self.audit = audit or AuditLogger()
        if governance_validator is None and engine is not None:
            # Default to the PRODUCTION validator (CONCEPT:AU-AHE.harness.promotion-governance-validator) — the
            # daemon/golden-loop path previously had no validator at all, so a
            # governance-required policy held everything and a non-required one
            # validated nothing.
            governance_validator = _probe_production_validator(engine, self.policy)
        self._governance_validator = governance_validator
        self._regression_check = regression_check
        self._promoter = promoter
        self._publisher = publisher
        self._action_policy = action_policy

    # ── scoring ────────────────────────────────────────────────────────
    @staticmethod
    def score_proposal(spec: Any) -> float:
        """Compute a quality score [0,1] for a proposal. CONCEPT:AU-AHE.assimilation.research-auto-merge.

        An explicit ``quality_score`` on the spec wins. Otherwise a structural
        heuristic rewards completeness: a team with a named lead + members, a
        described goal, and members carrying sub-structure scores higher. This is
        intentionally a *floor*: a richer evaluator can be injected, but a bare
        skeleton proposal never clears the conservative default threshold by
        accident.
        """
        explicit = getattr(spec, "quality_score", None)
        if explicit is None and isinstance(spec, dict):
            explicit = spec.get("quality_score")
        if explicit is not None:
            try:
                return max(0.0, min(1.0, float(explicit)))
            except (TypeError, ValueError):
                pass

        score = 0.0
        name = getattr(spec, "name", "") or (
            spec.get("name", "") if isinstance(spec, dict) else ""
        )
        goal = getattr(spec, "goal", "") or (
            spec.get("goal", "") if isinstance(spec, dict) else ""
        )
        if name:
            score += 0.25
        if goal:
            score += 0.25
        members = getattr(spec, "members", None)
        if members is None and isinstance(spec, dict):
            members = spec.get("members")
        if members:
            score += 0.25
            if len(members) >= 2:
                score += 0.15
        lead = getattr(spec, "lead", "") or (
            spec.get("lead", "") if isinstance(spec, dict) else ""
        )
        if lead:
            score += 0.10
        return max(0.0, min(1.0, score))

    # ── governance + regression gates ──────────────────────────────────
    def _check_governance(self, spec: Any) -> bool:
        if self._governance_validator is not None:
            try:
                return bool(self._governance_validator(spec))
            except Exception as exc:  # noqa: BLE001
                logger.debug("governance validator failed: %s", exc)
                return False
        # No validator wired: only pass when the policy does not require one.
        return not self.policy.require_governance_valid

    def _check_regression(self, spec: Any) -> bool:
        if self._regression_check is not None:
            try:
                return bool(self._regression_check(spec))
            except Exception as exc:  # noqa: BLE001
                logger.debug("regression check failed: %s", exc)
                return False
        return True  # no tracked regression → safe

    # ── evaluation + promotion ─────────────────────────────────────────
    def evaluate(self, spec: Any) -> MergeEvaluation:
        """Evaluate a proposal against the policy WITHOUT promoting it."""
        proposal_id = self._spec_id(spec)
        quality = self.score_proposal(spec)
        governance = self._check_governance(spec)
        no_regression = self._check_regression(spec)

        failures: list[str] = []
        if quality < self.policy.quality_threshold:
            failures.append(
                f"quality {quality:.2f} < threshold {self.policy.quality_threshold:.2f}"
            )
        if self.policy.require_governance_valid and not governance:
            failures.append("governance/SHACL invalid")
        if self.policy.require_no_regression and not no_regression:
            failures.append("regression detected")

        return MergeEvaluation(
            proposal_id=proposal_id,
            quality_score=quality,
            governance_valid=governance,
            no_regression=no_regression,
            failures=failures,
        )

    def consider(self, spec: Any) -> MergeEvaluation:
        """Evaluate and — when enabled + eligible — promote proposal→active.

        Always audits. When the policy is disabled the proposal stays
        proposal-only (the conservative default) even if it would be eligible.

        An enabled + eligible promotion additionally consults the operational
        OS-5.24 :class:`ActionPolicy` under the reserved ``merge_promotion``
        kind (the AHE-3.20 adoption): ``deny`` blocks the lifecycle flip
        (recorded on the evaluation + audit trail), ``queue_approval`` keeps
        the AHE-3.21 semantics — the KG-internal flip proceeds and the
        real-world publication queues the (deduped) ``ActionApproval`` —
        and ``allow``/``allow_notify`` proceed (the policy notifies).
        """
        evaluation = self.evaluate(spec)
        promote = self.policy.enabled and evaluation.eligible
        denied_reason = ""
        if promote:
            decision = self._consult_action_policy(spec)
            if decision is not None:
                evaluation.action_decision = {
                    "decision": getattr(decision, "decision", "deny"),
                    "reason": getattr(decision, "reason", ""),
                    "approval_id": getattr(decision, "approval_id", None),
                }
                if evaluation.action_decision["decision"] == "deny":
                    promote = False
                    denied_reason = (
                        "blocked by action policy (merge_promotion): "
                        f"{evaluation.action_decision['reason']}"
                    )
        if promote:
            try:
                evaluation.merged = self._promote(spec)
                evaluation.reason = (
                    "auto-merged" if evaluation.merged else "promotion failed"
                )
            except Exception as exc:  # noqa: BLE001 — never crash the loop
                logger.warning(
                    "auto-merge promotion error for %s: %s", evaluation.proposal_id, exc
                )
                evaluation.reason = f"promotion error: {exc}"
            if evaluation.merged:
                # Evolution→branch bridge (CONCEPT:AU-AHE.harness.evolution-branch-bridge): a merged proposal
                # becomes a reviewable git branch — gated by the OS-5.24
                # ActionPolicy's reserved ``merge_promotion`` kind (the shipped
                # default queues a human approval; publication then proceeds
                # via the one-shot ``publish_proposal`` action).
                evaluation.publication = self._publish(spec)
        elif denied_reason:
            evaluation.reason = denied_reason
        else:
            evaluation.reason = (
                "proposal-only (auto-merge disabled)"
                if not self.policy.enabled
                else "proposal-only: " + "; ".join(evaluation.failures)
            )
        self._audit(evaluation)
        return evaluation

    def _consult_action_policy(self, spec: Any) -> Any:
        """Decide ``merge_promotion`` for this proposal via the OS-5.24 gate.

        The AHE-3.20 → ActionPolicy adoption: the merger's own promotion
        decision routes through the same operational decision point the
        AHE-3.21 publication path consults — same reserved kind, same target
        (the proposal id, so a queued/granted ``ActionApproval`` is SHARED
        with the publication step via the policy's per-kind+target dedup).
        A gate failure fails CLOSED (deny), mirroring ``governed_publish``.
        """
        target = self._spec_id(spec)
        try:
            from agent_utilities.orchestration.action_policy import (
                ActionRequest,
                get_action_policy,
            )

            policy = self._action_policy or get_action_policy(self.engine)
            return policy.decide(
                ActionRequest(
                    kind="merge_promotion",
                    target=target,
                    params={
                        "stage": "promotion",
                        "name": str(
                            getattr(spec, "name", None)
                            or (spec.get("name") if isinstance(spec, dict) else "")
                            or ""
                        ),
                    },
                    source="loop_engine",
                    reason="promote evolution proposal proposal→active",
                )
            )
        except Exception as exc:  # noqa: BLE001 — gate failure ⇒ fail closed
            logger.warning(
                "auto-merge action-policy consult failed for %s: %s", target, exc
            )
            return _FailClosedDecision(
                reason=f"action policy unavailable (fail closed): {exc}"
            )

    def _promote(self, spec: Any) -> bool:
        """Promote a proposal to an active artifact (proposal → active)."""
        if self._promoter is not None:
            return bool(self._promoter(spec))
        return self._default_promote(spec)

    def _publish(self, spec: Any) -> dict[str, Any] | None:
        """Bridge a merged proposal to a reviewable branch (CONCEPT:AU-AHE.harness.evolution-branch-bridge).

        Best-effort: a publication failure never undoes the lifecycle merge or
        crashes the loop — the report (or the error) is carried on the
        evaluation and audited.
        """
        try:
            from .change_publisher import governed_publish

            return governed_publish(
                self.engine,
                spec,
                publisher=self._publisher,
                regression_check=self._regression_check,
                source="loop_engine",
            )
        except Exception as exc:  # noqa: BLE001 — never crash the loop
            logger.warning(
                "auto-merge publication error for %s: %s", self._spec_id(spec), exc
            )
            return {"status": "error", "detail": str(exc)}

    def _default_promote(self, spec: Any) -> bool:
        """Default promotion: flip the artifact's lifecycle to ``active`` in the KG.

        Persists the spec via the existing orchestration converters with a
        ``lifecycle="active"`` stamp (proposals carry ``lifecycle="proposal"``),
        through the same ``GraphBackend`` the propose-only path writes to — so
        promotion is a real graph mutation, not a flag in memory.
        """
        backend = getattr(self.engine, "backend", None)
        if backend is None:
            logger.debug("auto-merge: no backend; promotion is a no-op")
            return False
        try:
            from agent_utilities.knowledge_graph.enrichment.synthesize import (
                persist_synthesis,
            )

            # Stamp lifecycle=active on the spec copy before persisting.
            active_spec = self._with_active_lifecycle(spec)
            nodes, _edges = persist_synthesis(backend, active_spec)
            return nodes > 0
        except Exception as exc:  # noqa: BLE001
            logger.debug("auto-merge default promotion failed: %s", exc)
            return False

    @staticmethod
    def _with_active_lifecycle(spec: Any) -> Any:
        """Return a spec copy stamped active where the model supports it."""
        if hasattr(spec, "model_copy"):
            data = spec.model_dump() if hasattr(spec, "model_dump") else {}
            # description carries an [active] marker for backends without a
            # dedicated lifecycle field; pydantic specs ignore unknown kwargs.
            desc = data.get("description", "")
            marker = "[lifecycle:active]"
            if marker not in desc:
                try:
                    return spec.model_copy(
                        update={"description": f"{desc} {marker}".strip()}
                    )
                except Exception:  # noqa: BLE001
                    return spec
        return spec

    # ── helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _spec_id(spec: Any) -> str:
        sid = getattr(spec, "id", None)
        if not sid and isinstance(spec, dict):
            sid = spec.get("id")
        if sid:
            return str(sid)
        name = getattr(spec, "name", None) or (
            spec.get("name") if isinstance(spec, dict) else None
        )
        return f"proposal:{name or 'unknown'}"

    def _audit(self, ev: MergeEvaluation) -> None:
        record = self.audit.log(
            actor="loop_engine",
            action=AUDIT_AUTO_MERGE,
            resource_type="proposal",
            resource_id=ev.proposal_id,
            details={
                "quality_score": round(ev.quality_score, 4),
                "threshold": self.policy.quality_threshold,
                "governance_valid": ev.governance_valid,
                "no_regression": ev.no_regression,
                "merged": ev.merged,
                "reason": ev.reason,
                "failures": ev.failures,
                "enabled": self.policy.enabled,
                "action_decision": (ev.action_decision or {}).get("decision", ""),
                "action_approval_id": (ev.action_decision or {}).get("approval_id")
                or "",
                "publication_status": (ev.publication or {}).get("status", ""),
                "publication_branch": ((ev.publication or {}).get("publish") or {}).get(
                    "branch", ""
                ),
            },
        )
        if record is not None:
            ev.audit_ref = record.id
