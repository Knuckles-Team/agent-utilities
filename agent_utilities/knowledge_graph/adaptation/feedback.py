#!/usr/bin/python
from __future__ import annotations

"""Human-correction → rule/outcome/eval feedback loop (CONCEPT:KG-2.8).

The compounding layer the "Company Brain" was missing: a single entry point where
a human says "this was wrong, here's the fix" and the correction becomes
persistent future behaviour. Three correction types:

* ``outcome`` — adjust the reward EMA of a designated entity so routing/retrieval
  prefers/avoids it next time (reuses :meth:`CapabilityIndex.record_outcome`).
* ``rule`` — persist a ``Correction`` node (+ ``CORRECTS`` edge) and a durable,
  active governance/voice/source rule. Because a *human* asserted it, the rule is
  authoritative immediately — no synthesis threshold needed. It is consumed at
  retrieval time by :func:`apply_governance_rules`.
* ``eval`` — append a regression case to the eval corpus so the mistake is caught
  automatically from then on.

Dependencies are injected so the service is unit-testable without a live engine;
:meth:`from_engine` wires it from a running :class:`IntelligenceGraphEngine`.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_VALID = {"outcome", "rule", "eval"}

# Map a free-form rule scope to the persisted node type consumed by
# governance_rules.load_active_rules.
_RULE_TYPE = {
    "voice": "voice_rule",
    "source": "source_rule",
    "governance": "governance_rule",
    "preference": "preference",
}


@dataclass
class CorrectionResult:
    correction_type: str
    target_id: str
    applied: bool
    detail: str
    created_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "correction_type": self.correction_type,
            "target_id": self.target_id,
            "applied": self.applied,
            "detail": self.detail,
            "created_ids": self.created_ids,
        }


class FeedbackService:
    """Turn human corrections into durable behaviour change.

    Args:
        backend: anything exposing ``add_node``/``add_edge`` (graph writer).
        capability_index: optional :class:`CapabilityIndex` for outcome rewards.
        eval_corpus: optional :class:`EvalCorpus` for eval corrections.
    """

    def __init__(
        self,
        backend: Any = None,
        capability_index: Any = None,
        eval_corpus: Any = None,
    ) -> None:
        self.backend = backend
        self.capability_index = capability_index
        self.eval_corpus = eval_corpus

    @classmethod
    def from_engine(cls, engine: Any) -> FeedbackService:
        backend = getattr(engine, "backend", None) or getattr(engine, "store", None)
        kg = getattr(engine, "knowledge_graph", None) or getattr(engine, "kg", None)
        cap = getattr(kg, "retrieval", None) if kg is not None else None
        corpus = None
        try:
            from ...harness.eval_corpus import EvalCorpus

            corpus = EvalCorpus(backend)
        except Exception:  # pragma: no cover - corpus optional
            corpus = None
        return cls(backend=backend, capability_index=cap, eval_corpus=corpus)

    # ------------------------------------------------------------------
    def record_correction(
        self,
        correction_type: str,
        target_id: str,
        corrected_value: Any = None,
        reason: str = "",
        *,
        actor_id: str = "human",
        rule_scope: str = "governance",
        rule_kind: str = "forbid",
        reward: float | None = None,
    ) -> CorrectionResult:
        """Record a human correction and apply it durably."""
        ctype = correction_type.strip().lower()
        if ctype not in _VALID:
            return CorrectionResult(
                ctype, target_id, False, f"unknown correction_type {correction_type!r}"
            )
        if ctype == "outcome":
            return self._apply_outcome(target_id, reward, corrected_value, reason)
        if ctype == "rule":
            return self._apply_rule(
                target_id, corrected_value, reason, actor_id, rule_scope, rule_kind
            )
        return self._apply_eval(target_id, corrected_value, reason)

    # ------------------------------------------------------------------
    def export_preference_pairs(self, *, min_margin: float = 0.1) -> list[Any]:
        """Consolidate eval corpus + distilled episodes + corrections into a
        reliability-filtered, DPO-ready preference-pair corpus (CONCEPT:AHE-3.17).

        This is the read-side of the feedback loop: every correction/eval recorded
        through this service flows back out as clean (chosen ≻ rejected) pairs, with
        RAPPO ambiguous-pair filtering applied. Layers TI-DPO token weights / InSPO
        reflection are opt-in on the returned pairs.
        """
        from agent_utilities.harness.preference_pairs import (
            PreferencePairExporter,
            reliability_filter,
        )

        exporter = PreferencePairExporter(backend=self.backend)
        kept, dropped = reliability_filter(exporter.export(), min_margin=min_margin)
        if dropped:
            logger.info(
                "[AHE-3.17] preference export: kept=%d dropped=%d (ambiguous/degenerate)",
                len(kept),
                dropped,
            )
        return kept

    # ------------------------------------------------------------------
    def _apply_outcome(self, target_id, reward, corrected_value, reason):
        if self.capability_index is None or not hasattr(
            self.capability_index, "record_outcome"
        ):
            return CorrectionResult(
                "outcome", target_id, False, "no capability_index available"
            )
        r = reward
        if r is None and corrected_value is not None:
            try:
                r = float(corrected_value)
            except (TypeError, ValueError):
                r = None
        if r is None:
            return CorrectionResult(
                "outcome",
                target_id,
                False,
                "outcome correction needs reward/corrected_value",
            )
        new = self.capability_index.record_outcome(target_id, reward=r)
        return CorrectionResult(
            "outcome", target_id, True, f"reward updated to {new:.3f} ({reason})"
        )

    def _apply_rule(
        self, target_id, corrected_value, reason, actor_id, rule_scope, rule_kind
    ):
        if self.backend is None or not hasattr(self.backend, "add_node"):
            return CorrectionResult(
                "rule", target_id, False, "no backend to persist rule"
            )
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        corr_id = f"correction:{uuid.uuid4().hex[:12]}"
        self.backend.add_node(
            corr_id,
            type="correction",
            target=target_id,
            reason=reason,
            corrected_value=str(corrected_value or ""),
            actor_id=actor_id,
            assertion_type="human_judgment",
            timestamp=ts,
        )
        created = [corr_id]
        if target_id and hasattr(self.backend, "add_edge"):
            try:
                self.backend.add_edge(corr_id, target_id, rel_type="corrects")
            except Exception as exc:  # pragma: no cover
                logger.debug("corrects edge failed: %s", exc)
        rule_id = f"rule:{uuid.uuid4().hex[:12]}"
        rule_type = _RULE_TYPE.get(rule_scope, "governance_rule")
        self.backend.add_node(
            rule_id,
            type=rule_type,
            kind=rule_kind,
            target=target_id,
            weight=0.5,
            reason=reason,
            active=True,
            assertion_type="human_judgment",
            source_correction=corr_id,
            timestamp=ts,
        )
        created.append(rule_id)
        return CorrectionResult(
            "rule",
            target_id,
            True,
            f"persisted {rule_type} ({rule_kind}) for {target_id}",
            created,
        )

    def _apply_eval(self, target_id, corrected_value, reason):
        if self.eval_corpus is None or not hasattr(self.eval_corpus, "add_case"):
            return CorrectionResult(
                "eval", target_id, False, "no eval corpus available"
            )
        case_id = self.eval_corpus.add_case(
            query=target_id,
            expected_output=str(corrected_value or ""),
            tags=["from_correction"],
            reason=reason,
        )
        return CorrectionResult(
            "eval", target_id, True, f"added eval case {case_id}", [case_id]
        )
