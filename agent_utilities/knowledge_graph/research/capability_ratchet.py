#!/usr/bin/python
from __future__ import annotations

"""Capability ratchet + verified apply→verify→rollback (CONCEPT:AHE-3.24, AHE-3.23).

Before this, the deployed evolution loop's only regression gate was an occurrence-
count *spike* monitor over Langfuse (``failure_analyzer.make_regression_check``) —
no before/after re-measurement of capability, and a merged change could pass its
own targeted tests while silently regressing untested capabilities. These two
concepts close that gap on the live publish path:

* **AHE-3.24 — capability ratchet.** Run a standing *capability suite* in the
  freshly-published worktree, producing a per-capability score vector; compare it
  against a persisted ``CapabilityScoreVector`` baseline node and require every
  tracked capability to stay ≥ baseline (monotone ratchet). The first run with no
  baseline *establishes* it (bootstrap) without blocking.
* **AHE-3.23 — verified verdict.** The keep/abandon decision is the authoritative
  one from the existing :class:`~agent_utilities.harness.verifier.ManifestVerifier`
  (its ``recommendation``: ``confirm`` / ``partial_revert`` / ``full_revert`` from
  the measured benchmark delta), fed the ratchet's before/after scores. A
  ``*_revert`` recommendation — or any per-capability regression — abandons the
  branch instead of merging it.

The ratchet *measures what it can*: if the capability probe targets do not exist in
the worktree (e.g. a synthetic test repo), it measures nothing and does not block —
honest "can't ratchet what you can't measure", logged rather than silently passing.
"""

import json
import logging
import os
import re
import subprocess  # nosec B404 — fixed interpreter, repo-relative targets
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

#: Standing capability probes run in the published worktree. These are the
#: harness's own deterministic reliability/eval tests (AHE-3.1/3.12) — a change
#: that regresses the harness fails them. Tunable; a probe absent from the
#: worktree is skipped (not a failure).
DEFAULT_CAPABILITY_TARGETS: tuple[str, ...] = (
    "tests/harness/test_reliability_scorers.py",
    "tests/test_ahe_harness.py",
)

_PASS_RE = re.compile(r"(\d+)\s+passed")
_FAIL_RE = re.compile(r"(\d+)\s+(?:failed|error[s]?)")


@dataclass
class RatchetVerdict:
    """Outcome of a capability-ratchet evaluation."""

    passed: bool
    recommendation: str  # confirm | partial_revert | full_revert | not_measured | bootstrap
    scores: dict[str, float] = field(default_factory=dict)
    baseline: dict[str, float] = field(default_factory=dict)
    regressions: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "recommendation": self.recommendation,
            "scores": self.scores,
            "regressions": self.regressions,
            "reason": self.reason,
        }


def _mean(values: Any) -> float:
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else 0.0


def _run_sync(coro: Any) -> Any:
    """Run a coroutine to completion from sync code, even under a live loop."""
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(lambda: asyncio.run(coro)).result()


class CapabilityRatchet:
    """Measure → verify → ratchet a published worktree against a capability baseline."""

    def __init__(
        self,
        engine: Any = None,
        *,
        suite_runner: Callable[[str], dict[str, float]] | None = None,
        targets: tuple[str, ...] | None = None,
        tolerance: float = 0.0,
        timeout: float = 600.0,
    ) -> None:
        self.engine = engine
        self._runner = suite_runner
        self.targets = tuple(targets or DEFAULT_CAPABILITY_TARGETS)
        self.tolerance = float(tolerance)
        self.timeout = float(timeout)

    # ── measurement ──────────────────────────────────────────────────
    def measure(self, worktree_path: str) -> dict[str, float]:
        """A per-capability ``{target: pass_rate}`` vector for the worktree."""
        if self._runner is not None:
            return dict(self._runner(worktree_path) or {})
        present = [
            t for t in self.targets if os.path.isfile(os.path.join(worktree_path, t))
        ]
        out: dict[str, float] = {}
        for target in present:
            rate = self._score_target(worktree_path, target)
            if rate is not None:
                out[target] = rate
        return out

    def _score_target(self, worktree_path: str, target: str) -> float | None:
        argv = [
            sys.executable, "-m", "pytest", "-q", "--no-header",
            "-p", "no:cacheprovider", target,
        ]
        try:
            proc = subprocess.run(  # nosec B603 — fixed interpreter, repo-relative target
                argv, cwd=worktree_path, capture_output=True, text=True, timeout=self.timeout
            )
        except Exception as exc:  # noqa: BLE001 — an unrunnable probe is not measured
            logger.warning("[AHE-3.24] capability probe %s did not run: %s", target, exc)
            return None
        text = (proc.stdout or "") + (proc.stderr or "")
        passed = int(m.group(1)) if (m := _PASS_RE.search(text)) else 0
        failed = int(m.group(1)) if (m := _FAIL_RE.search(text)) else 0
        total = passed + failed
        return passed / total if total else None

    # ── baseline + audit nodes ───────────────────────────────────────
    def _load_baseline(self) -> dict[str, float] | None:
        if self.engine is None:
            return None
        try:
            rows = self.engine.query_cypher(
                "MATCH (n:CapabilityScoreVector) "
                "RETURN n.scores_json AS scores, n.recorded_at AS ts"
            )
        except Exception:  # noqa: BLE001 — no engine/query support ⇒ no baseline
            return None
        rows = [r for r in (rows or []) if isinstance(r, dict) and r.get("scores")]
        if not rows:
            return None
        latest = max(rows, key=lambda r: str(r.get("ts") or ""))
        try:
            data = json.loads(latest["scores"])
            return {str(k): float(v) for k, v in dict(data).items()}
        except (TypeError, ValueError):
            return None

    def _store_baseline(self, scores: dict[str, float]) -> None:
        if self.engine is None:
            return
        import uuid

        try:
            self.engine.add_node(
                f"capability_baseline:{uuid.uuid4().hex[:12]}",
                "CapabilityScoreVector",
                properties={
                    "scores_json": json.dumps(scores),
                    "recorded_at": _now_iso(),
                },
            )
        except Exception as exc:  # noqa: BLE001 — persistence is best-effort
            logger.debug("[AHE-3.24] could not persist capability baseline: %s", exc)

    def _record(self, proposal_id: str, verdict: RatchetVerdict) -> None:
        if self.engine is None:
            return
        import uuid

        try:
            self.engine.add_node(
                f"capability_ratchet:{uuid.uuid4().hex[:12]}",
                "CapabilityRatchetResult",
                properties={
                    "proposal_id": proposal_id,
                    "result": "pass" if verdict.passed else "hold",
                    "recommendation": verdict.recommendation,
                    "regressions_json": json.dumps(verdict.regressions),
                    "recorded_at": _now_iso(),
                },
            )
        except Exception as exc:  # noqa: BLE001 — audit is best-effort
            logger.debug("[AHE-3.24] could not record capability ratchet result: %s", exc)

    # ── verified verdict (AHE-3.23) ──────────────────────────────────
    def _recommendation(
        self,
        worktree_path: str,
        change_set: Any,
        baseline: dict[str, float],
        scores: dict[str, float],
    ) -> str:
        """The authoritative confirm/revert recommendation from ManifestVerifier."""
        from agent_utilities.harness.component_registry import HarnessComponentRegistry
        from agent_utilities.harness.evidence_corpus import EvidenceCorpus
        from agent_utilities.harness.manifest import (
            ChangeManifest,
            ComponentEdit,
            ComponentType,
        )
        from agent_utilities.harness.verifier import ManifestVerifier

        edits = [
            ComponentEdit(
                component_type=ComponentType.TOOL_IMPLEMENTATION,
                file_path=f.path,
                edit_summary="evolution change",
            )
            for f in getattr(change_set, "files", []) or []
        ]
        manifest = ChangeManifest(edits=edits)
        verifier = ManifestVerifier(HarnessComponentRegistry(worktree_path))
        result = _run_sync(
            verifier.verify(
                manifest,
                EvidenceCorpus(benchmark_score=_mean(baseline.values())),
                EvidenceCorpus(benchmark_score=_mean(scores.values())),
            )
        )
        return str(result.recommendation)

    # ── evaluation ───────────────────────────────────────────────────
    def evaluate(
        self,
        worktree_path: str,
        *,
        change_set: Any = None,
        proposal_id: str = "",
    ) -> RatchetVerdict:
        """Measure the worktree, verify against the baseline, ratchet on pass."""
        scores = self.measure(worktree_path)
        if not scores:
            return RatchetVerdict(
                passed=True,
                recommendation="not_measured",
                reason="no capability probes present in the worktree to measure",
            )

        baseline = self._load_baseline()
        if baseline is None:
            self._store_baseline(scores)
            verdict = RatchetVerdict(
                passed=True,
                recommendation="bootstrap",
                scores=scores,
                reason="established the capability baseline (nothing to ratchet yet)",
            )
            self._record(proposal_id, verdict)
            return verdict

        regressions = [
            cap
            for cap, score in scores.items()
            if score < baseline.get(cap, 0.0) - self.tolerance
        ]
        recommendation = self._recommendation(
            worktree_path, change_set, baseline, scores
        )
        passed = not regressions and recommendation not in {"full_revert", "partial_revert"}

        if passed:
            merged = {
                cap: max(scores.get(cap, 0.0), baseline.get(cap, 0.0))
                for cap in set(scores) | set(baseline)
            }
            self._store_baseline(merged)
            reason = "capability confirmed; baseline advanced"
        else:
            reason = (
                f"capability regression ({recommendation}); "
                f"blocked: {', '.join(regressions) or 'aggregate benchmark delta < 0'}"
            )
        verdict = RatchetVerdict(
            passed=passed,
            recommendation=recommendation,
            scores=scores,
            baseline=baseline,
            regressions=regressions,
            reason=reason,
        )
        self._record(proposal_id, verdict)
        return verdict


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def latest_ratchet_result(engine: Any, proposal_id: str) -> str | None:
    """The most recent recorded ratchet result for a proposal (``pass``/``hold``).

    Used by the promotion governance gate (AHE-3.20) to consult a recorded
    capability verdict, mirroring its recorded-regression-gate predicate.
    """
    if engine is None or not proposal_id:
        return None
    try:
        rows = engine.query_cypher(
            "MATCH (r:CapabilityRatchetResult) WHERE r.proposal_id = $pid "
            "RETURN r.result AS result, r.recorded_at AS ts",
            {"pid": proposal_id},
        )
    except Exception:  # noqa: BLE001
        return None
    rows = [r for r in (rows or []) if isinstance(r, dict) and r.get("result")]
    if not rows:
        return None
    return str(max(rows, key=lambda r: str(r.get("ts") or ""))["result"]).lower()
