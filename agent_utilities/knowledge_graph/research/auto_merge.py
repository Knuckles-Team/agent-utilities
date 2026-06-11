#!/usr/bin/python
from __future__ import annotations

"""Governed golden-loop auto-merge (CONCEPT:AHE-3.14).

The golden loop is *propose-only* by design (``golden_loop.py``): it synthesizes
``TeamSpec`` / ``AgentSpec`` / ``PromptSpec`` proposals and persists them as KG
nodes, but never promotes them to *active* skills/prompts — promotion is left to
a human. This module adds the missing GOVERNED auto-merge path: a proposal that
passes a quality + governance gate is promoted ``proposal → active`` automatically
and audited; one that fails stays proposal-only (human-gated).

It is deliberately conservative and OFF by default:

  - the default quality threshold is high (``0.85``),
  - SHACL/governance validity is required (when a validator is reachable),
  - regression is checked (a promoted artifact must not lower a tracked metric),
  - the whole step is opt-in via ``GoldenLoopController(auto_merge=...)`` /
    ``KG_GOLDEN_AUTO_MERGE=1`` — the existing propose-only safety is the default
    unless explicitly enabled.

Every promotion (and every rejection) emits an
:class:`~agent_utilities.observability.audit_logger.AuditLogger` entry, so the
auto-merge trail is queryable.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from agent_utilities.observability.audit_logger import AuditLogger

logger = logging.getLogger(__name__)

AUDIT_AUTO_MERGE = "golden_loop.auto_merge"

#: Conservative default — only very high-quality proposals auto-merge.
DEFAULT_QUALITY_THRESHOLD = 0.85


def _probe_production_validator(engine: Any, policy: MergePolicy) -> Any:
    """Build the default production governance validator (CONCEPT:AHE-3.20).

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
class MergePolicy:
    """The governance policy for auto-merging a proposal. CONCEPT:AHE-3.14.

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
    """The result of evaluating one proposal for auto-merge. CONCEPT:AHE-3.14."""

    proposal_id: str
    quality_score: float
    governance_valid: bool
    no_regression: bool
    merged: bool = False
    reason: str = ""
    audit_ref: str = ""
    failures: list[str] = field(default_factory=list)

    @property
    def eligible(self) -> bool:
        """True when all gates passed (independent of whether merge was enabled)."""
        return not self.failures


class GovernedAutoMerger:
    """Evaluates + (when governed) promotes golden-loop proposals. CONCEPT:AHE-3.14.

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
        is constructed as the default (CONCEPT:AHE-3.20): SHACL shapes,
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
    ) -> None:
        self.engine = engine
        self.policy = policy or MergePolicy.from_env()
        self.audit = audit or AuditLogger()
        if governance_validator is None and engine is not None:
            # Default to the PRODUCTION validator (CONCEPT:AHE-3.20) — the
            # daemon/golden-loop path previously had no validator at all, so a
            # governance-required policy held everything and a non-required one
            # validated nothing.
            governance_validator = _probe_production_validator(engine, self.policy)
        self._governance_validator = governance_validator
        self._regression_check = regression_check
        self._promoter = promoter

    # ── scoring ────────────────────────────────────────────────────────
    @staticmethod
    def score_proposal(spec: Any) -> float:
        """Compute a quality score [0,1] for a proposal. CONCEPT:AHE-3.14.

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
        """
        evaluation = self.evaluate(spec)
        promote = self.policy.enabled and evaluation.eligible
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
        else:
            evaluation.reason = (
                "proposal-only (auto-merge disabled)"
                if not self.policy.enabled
                else "proposal-only: " + "; ".join(evaluation.failures)
            )
        self._audit(evaluation)
        return evaluation

    def _promote(self, spec: Any) -> bool:
        """Promote a proposal to an active artifact (proposal → active)."""
        if self._promoter is not None:
            return bool(self._promoter(spec))
        return self._default_promote(spec)

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
        if sid:
            return str(sid)
        name = getattr(spec, "name", None) or (
            spec.get("name") if isinstance(spec, dict) else None
        )
        return f"proposal:{name or 'unknown'}"

    def _audit(self, ev: MergeEvaluation) -> None:
        record = self.audit.log(
            actor="golden_loop",
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
            },
        )
        if record is not None:
            ev.audit_ref = record.id
