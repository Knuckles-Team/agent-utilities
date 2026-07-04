from __future__ import annotations

"""DSPy optimization of classifier/policy decisions from existing supervision.

CONCEPT:AU-AHE.optimization.concept-matching-routing-policy — Concept-matching and routing-policy optimization.

Two decisions in the system are *classifiers* whose ground truth the graph already holds,
so DSPy can optimize them with free supervision:

* **Concept matching** — "does this source address this research topic?" The
  ``ADDRESSES`` edges the loop has already created are positive labels; non-edges are
  negatives. :func:`optimize_concept_matcher` optimizes the matcher against
  classification accuracy on those labels.
* **Routing policy** — "which primitive/tier for this task?" Each historical
  ``ExecutionTrace`` records the chosen route *and whether it succeeded*, so a routing
  Signature can be optimized against realized success — complementing the
  ``TraceLearnedPolicy`` that already learns classically.

The metrics (:func:`classification_accuracy`, :func:`routing_success_rate`) are
dependency-free and offline-testable; the DSPy passes are best-effort and no-op without
DSPy/an LLM.
"""

import logging
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Testable metrics
# --------------------------------------------------------------------------- #
def classification_accuracy(
    predictions: Sequence[bool], labels: Sequence[bool]
) -> float:
    """Accuracy of boolean predictions against labels (CONCEPT:AU-AHE.optimization.concept-matching-routing-policy)."""
    pairs = list(zip(predictions, labels, strict=False))
    if not pairs:
        return 0.0
    correct = sum(1 for p, y in pairs if bool(p) == bool(y))
    return correct / len(pairs)


def routing_success_rate(decisions: Sequence[Any]) -> float:
    """Mean realized success of a set of routing decisions (CONCEPT:AU-AHE.optimization.concept-matching-routing-policy).

    Each item is a trace exposing ``success: bool`` (and optionally ``quality_score``);
    returns the mean success in [0, 1]. The objective a routing optimizer maximizes.
    """
    items = list(decisions)
    if not items:
        return 0.0
    total = 0.0
    for d in items:
        ok = d.get("success") if isinstance(d, dict) else getattr(d, "success", False)
        total += 1.0 if ok else 0.0
    return total / len(items)


# --------------------------------------------------------------------------- #
# Best-effort DSPy passes (LLM-gated)
# --------------------------------------------------------------------------- #
def optimize_concept_matcher(
    labeled_pairs: Sequence[tuple[str, str, bool]],
    *,
    optimizer_name: str = "BootstrapFewShot",
) -> dict[str, Any] | None:
    """DSPy-optimize the concept→source relevance classifier (CONCEPT:AU-AHE.optimization.concept-matching-routing-policy).

    ``labeled_pairs`` are ``(article_text, concept_text, is_relevant)`` drawn from the
    KG's ``ADDRESSES`` edges (positives) and sampled non-edges (negatives). Optimizes a
    ``(article, concept) → relevant`` Signature against classification accuracy. Returns
    ``None`` when DSPy/an LLM is unavailable or no pairs are given. Never raises.
    """
    if not labeled_pairs:
        return None
    try:
        import dspy

        from agent_utilities.harness.dspy_optimization import build_optimizer

        class MatchSignature(dspy.Signature):
            """Decide whether the source article addresses the research concept."""

            article: str = dspy.InputField(desc="Candidate source text.")
            concept: str = dspy.InputField(desc="The research concept/topic.")
            relevant: str = dspy.OutputField(
                desc="'yes' if it addresses it, else 'no'."
            )

        class MatchModule(dspy.Module):
            def __init__(self) -> None:
                super().__init__()
                self.predict = dspy.Predict(MatchSignature)

            def forward(self, article: str, concept: str) -> Any:
                return self.predict(article=article, concept=concept)

        def metric(example: Any, pred: Any, trace: Any = None) -> bool:
            want = str(getattr(example, "relevant", "")).strip().lower().startswith("y")
            got = str(getattr(pred, "relevant", "")).strip().lower().startswith("y")
            return want == got

        trainset = [
            dspy.Example(
                article=a, concept=c, relevant=("yes" if rel else "no")
            ).with_inputs("article", "concept")
            for a, c, rel in labeled_pairs
        ]
        optimizer = build_optimizer(optimizer_name, metric)
        compiled = optimizer.compile(MatchModule(), trainset=trainset)
        return {
            "compiled_state": compiled.dump_state(),
            "optimizer": optimizer_name,
            "labels": len(labeled_pairs),
        }
    except Exception as e:  # noqa: BLE001 - best-effort, LLM-gated
        logger.warning("optimize_concept_matcher failed: %s", e)
        return None


def optimize_routing_policy(
    traces: Sequence[Any],
    *,
    optimizer_name: str = "BootstrapFewShot",
) -> dict[str, Any] | None:
    """DSPy-optimize the routing policy from historical execution outcomes (CONCEPT:AU-AHE.optimization.concept-matching-routing-policy).

    Each trace exposes ``task_text``, the chosen ``primitive_used``/``model_used``, and
    ``success``. Optimizes a ``task → primitive`` Signature using only the *successful*
    traces as demonstrations (so the policy imitates what worked). Returns ``None`` when
    DSPy/an LLM is unavailable or there are no successful traces. Never raises.
    """
    if not traces:
        return None
    try:
        import dspy

        from agent_utilities.harness.dspy_optimization import build_optimizer

        class RouteSignature(dspy.Signature):
            """Choose the execution primitive best suited to the task."""

            task: str = dspy.InputField(desc="The task description.")
            primitive: str = dspy.OutputField(
                desc="One of: direct, delegate, decompose."
            )

        class RouteModule(dspy.Module):
            def __init__(self) -> None:
                super().__init__()
                self.predict = dspy.Predict(RouteSignature)

            def forward(self, task: str) -> Any:
                return self.predict(task=task)

        def _get(t: Any, k: str, d: Any = "") -> Any:
            return t.get(k, d) if isinstance(t, dict) else getattr(t, k, d)

        demos = [
            dspy.Example(
                task=str(_get(t, "task_text")),
                primitive=str(_get(t, "primitive_used", "direct")),
            ).with_inputs("task")
            for t in traces
            if _get(t, "success", False)
        ]
        if not demos:
            return None

        def metric(example: Any, pred: Any, trace: Any = None) -> bool:
            return (
                str(getattr(example, "primitive", "")).strip().lower()
                == str(getattr(pred, "primitive", "")).strip().lower()
            )

        optimizer = build_optimizer(optimizer_name, metric)
        compiled = optimizer.compile(RouteModule(), trainset=demos)
        return {
            "compiled_state": compiled.dump_state(),
            "optimizer": optimizer_name,
            "demonstrations": len(demos),
        }
    except Exception as e:  # noqa: BLE001 - best-effort, LLM-gated
        logger.warning("optimize_routing_policy failed: %s", e)
        return None
