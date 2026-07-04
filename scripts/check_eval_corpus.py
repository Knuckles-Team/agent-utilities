#!/usr/bin/env python3
"""Eval-corpus gate (CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort / KG-2.8).

Proves the feedback→eval pipeline is functional and has teeth: builds a small
synthetic corpus (the shape produced by human ``eval`` corrections), runs it
through the real :class:`EvalRunner`, and FAILS if the pass-rate on cases whose
answer is correct drops below a floor — catching a broken scorer before it ships.

This mirrors the other synthetic-fixture gates (``check_retrieval_quality.py``,
``check_designation_eval.py``): no network, no live KG, deterministic.

**Hermetic scoring.** The synthetic cases have exact expected outputs, so the
gate scores them with the deterministic normalized/lexical EXACT_MATCH strategy
(:class:`_LexicalEvalRunner`) — it never reaches for an embedding model or an
LLM judge. The default :class:`EvalRunner` strategy is ``COMPOSITE``, which
calls ``_semantic_similarity_eval`` (an ``OpenAIEmbedding``) and the LLM judge;
those resolve to the homelab vLLM backend locally but have no backend in CI,
making the gate pass-local / fail-CI. Forcing EXACT_MATCH makes CI == local.

Usage::

    python3 scripts/check_eval_corpus.py [--degrade]

``--degrade`` returns wrong answers so the gate trips (used by the meta-test).
Exit 0 = pass-rate at/above floor. 1 = regression. 2 = build error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_utilities.harness.continuous_evaluation_engine import (
        EvalResult,
        EvalRunner,
        TestCase,
    )

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

FLOOR = 0.9

# (query, expected_output) cases — the shape FeedbackService("eval", ...) writes.
_CASES = [
    ("what is our refund window?", "30-day refund window"),
    ("primary support channel?", "email support"),
    ("default deployment target?", "docker swarm"),
    ("which model tier for drafts?", "haiku"),
]


def _lexical_runner() -> EvalRunner:
    """A deterministic, offline EvalRunner that scores by EXACT_MATCH only.

    The default :class:`EvalRunner` defers to each ``TestCase.strategy`` (which
    is ``COMPOSITE``), pulling in the embedding + LLM-judge scorers. The
    synthetic cases here have exact expected outputs, so forcing the normalized
    EXACT_MATCH strategy gives the correct verdict with zero network/embedding
    calls — keeping the gate hermetic (CI == local).
    """
    from agent_utilities.harness.continuous_evaluation_engine import (
        EvalRunner,
        EvalStrategy,
    )

    class _LexicalEvalRunner(EvalRunner):
        def run_eval(
            self,
            test_case: TestCase,
            actual_output: str,
            strategy: EvalStrategy | None = None,
        ) -> EvalResult:
            return super().run_eval(
                test_case, actual_output, strategy=EvalStrategy.EXACT_MATCH
            )

    return _LexicalEvalRunner()


def _run(degrade: bool) -> float:
    from agent_utilities.harness.eval_corpus import EvalCorpus

    corpus = EvalCorpus()
    for q, a in _CASES:
        corpus.add_case(q, a, tags=["from_correction"])

    answers = {q: a for q, a in _CASES}

    def actual(case):
        if degrade:
            return "completely unrelated wrong answer"
        return answers.get(case.query, "")

    results = corpus.run_corpus(actual_output_fn=actual, runner=_lexical_runner())
    if not results:
        return 0.0
    passed = sum(1 for r in results if r.passed)
    return passed / len(results)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--degrade", action="store_true")
    args = ap.parse_args()
    try:
        rate = _run(args.degrade)
    except Exception as exc:  # pragma: no cover - build error
        print(f"ERROR: eval corpus build failed: {exc}", file=sys.stderr)
        return 2

    print(f"Eval corpus pass-rate: {rate:.2f} (floor {FLOOR:.2f})")
    if rate < FLOOR:
        print("FAIL: eval corpus pass-rate below floor.", file=sys.stderr)
        return 1
    print("OK: eval corpus at/above floor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
