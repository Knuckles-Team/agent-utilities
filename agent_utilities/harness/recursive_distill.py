#!/usr/bin/python
from __future__ import annotations

"""Recursive distillation: train a better prior from the loop's own winners.

CONCEPT:AHE-3.31 — a recursive-distillation loop that fine-tunes a new prior from the search-distilled corpus, gates the candidate model behind a monotone capability ratchet, and promotes it only on a non-regressing improvement so test-time search converts into a stronger next prior

The paper (§5.3, memetic RSI) calls converting test-time compute into a better prior —
distilling search-improved outputs back into the model that drives the next search,
AlphaZero-style — likely the most important recursive-improvement mechanism. AU now has
both halves but no loop: OS-5.36 harvests verified winners into a corpus, and the
capability ratchet (AHE-3.24) can gate a change. This closes the loop: when the corpus has
grown enough, fine-tune a candidate prior, evaluate its capability vector, and promote it
as the new prior **only** if every tracked capability stays at-or-above the baseline
(monotone ratchet) — otherwise discard. The fine-tune and the model evaluation are the
*external-compute* dependencies (a GPU trainer + an eval run), injected here; the loop,
gating and promotion are pure and in-repo, so the whole cycle is testable end-to-end.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DistillReport:
    """Outcome of one recursive-distillation attempt."""

    status: str  # skipped | bootstrap | promoted | rejected | error
    corpus_size: int = 0
    scores: dict[str, float] = field(default_factory=dict)
    baseline: dict[str, float] = field(default_factory=dict)
    regressions: list[str] = field(default_factory=list)
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "corpus_size": self.corpus_size,
            "scores": self.scores,
            "regressions": self.regressions,
            "detail": self.detail,
        }


class RecursiveDistiller:
    """The AlphaZero-style loop: corpus → fine-tune → capability-gate → promote.

    Args:
        engine: optional KG engine for baseline + result persistence.
        corpus_source: ``() -> list`` of distilled rows (e.g. the OS-5.36 harvester's
            ``corpus``). The loop runs once it holds at least ``min_rows`` rows.
        trainer: ``(corpus) -> model`` — the GPU fine-tune (external; injected).
            Returns a model artifact, or ``None`` when unavailable.
        evaluate_model: ``(model) -> {capability: score}`` — the candidate's capability
            vector from an eval run (external; injected).
        promote: optional ``(model) -> None`` — install the model as the new prior.
        tolerance: per-capability slack below baseline still counted as non-regressing.
    """

    def __init__(
        self,
        engine: Any = None,
        *,
        corpus_source: Callable[[], list[Any]],
        trainer: Callable[[list[Any]], Any],
        evaluate_model: Callable[[Any], dict[str, float]],
        promote: Callable[[Any], None] | None = None,
        min_rows: int = 50,
        tolerance: float = 0.0,
    ) -> None:
        self.engine = engine
        self._corpus_source = corpus_source
        self._trainer = trainer
        self._evaluate_model = evaluate_model
        self._promote = promote
        self.min_rows = int(min_rows)
        self.tolerance = float(tolerance)

    def maybe_distill(self) -> DistillReport:
        """Run one cycle if the corpus is large enough; gate + promote the result."""
        corpus = list(self._corpus_source() or [])
        if len(corpus) < self.min_rows:
            return DistillReport(
                "skipped",
                len(corpus),
                detail=f"corpus {len(corpus)} < min {self.min_rows}",
            )
        try:
            model = self._trainer(corpus)
        except Exception as exc:  # noqa: BLE001 — a trainer failure is non-fatal
            return DistillReport("error", len(corpus), detail=f"trainer error: {exc}")
        if model is None:
            return DistillReport("skipped", len(corpus), detail="trainer unavailable")

        try:
            scores = {str(k): float(v) for k, v in self._evaluate_model(model).items()}
        except Exception as exc:  # noqa: BLE001
            return DistillReport("error", len(corpus), detail=f"eval error: {exc}")

        baseline = self._load_baseline()
        if baseline is None:
            self._store_baseline(scores)
            if self._promote is not None:
                self._promote(model)
            report = DistillReport(
                "bootstrap",
                len(corpus),
                scores=scores,
                detail="established model baseline",
            )
            self._record(report)
            return report

        regressions = [
            cap
            for cap, s in scores.items()
            if s < baseline.get(cap, 0.0) - self.tolerance
        ]
        if regressions:
            report = DistillReport(
                "rejected",
                len(corpus),
                scores=scores,
                baseline=baseline,
                regressions=regressions,
                detail="candidate regressed; discarded",
            )
            self._record(report)
            return report

        # monotone ratchet: promote and advance the baseline to the new (≥) vector.
        merged = {
            c: max(scores.get(c, 0.0), baseline.get(c, 0.0))
            for c in set(scores) | set(baseline)
        }
        self._store_baseline(merged)
        if self._promote is not None:
            self._promote(model)
        report = DistillReport(
            "promoted",
            len(corpus),
            scores=scores,
            baseline=baseline,
            detail="new prior promoted",
        )
        self._record(report)
        return report

    # ── persistence (best-effort) ────────────────────────────────────
    def _load_baseline(self) -> dict[str, float] | None:
        if self.engine is None:
            return None
        try:
            rows = self.engine.query_cypher(
                "MATCH (n:DistilledModelBaseline) "
                "RETURN n.scores_json AS scores, n.recorded_at AS ts"
            )
        except Exception:  # noqa: BLE001
            return None
        rows = [r for r in (rows or []) if isinstance(r, dict) and r.get("scores")]
        if not rows:
            return None
        import json

        latest = max(rows, key=lambda r: str(r.get("ts") or ""))
        try:
            return {
                str(k): float(v) for k, v in dict(json.loads(latest["scores"])).items()
            }
        except (TypeError, ValueError):
            return None

    def _store_baseline(self, scores: dict[str, float]) -> None:
        self._add_node("DistilledModelBaseline", {"scores_json": _dumps(scores)})

    def _record(self, report: DistillReport) -> None:
        self._add_node(
            "RecursiveDistillationResult",
            {"status": report.status, "metrics_json": _dumps(report.to_dict())},
        )

    def _add_node(self, label: str, props: dict[str, Any]) -> None:
        if self.engine is None:
            return
        import time
        import uuid

        try:
            self.engine.add_node(
                f"{label.lower()}:{uuid.uuid4().hex[:12]}",
                label,
                properties={
                    **props,
                    "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("[AHE-3.31] could not persist %s: %s", label, exc)


def _dumps(obj: Any) -> str:
    import json

    return json.dumps(obj)
