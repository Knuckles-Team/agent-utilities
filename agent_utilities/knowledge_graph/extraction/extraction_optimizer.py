from __future__ import annotations

"""Self-supervised optimization of the fact-extraction prompt.

CONCEPT:AU-AHE.optimization.dspy-optimization-kg-extraction — DSPy optimization of the KG extraction prompt against a
*self-supervised* quality metric — no human labels required.

The extraction prompt (``FACT_EXTRACTION_PROMPT``) is large and hand-tuned. Its quality
is measurable without ground truth: a good extraction produces facts that (1) do not
duplicate each other (the embedding deduper already measures this) and (2) refer to each
entity by a *consistent canonical name* rather than fragmenting it across surface forms.
That gives a metric DSPy can optimize directly.

The metric (:func:`extraction_quality`) is dependency-free and offline-testable (inject a
deterministic ``embed_fn``); the DSPy optimization pass (:func:`optimize_extraction_prompt`)
is best-effort and a no-op when DSPy/an LLM is unavailable.
"""

import logging
import re
from collections.abc import Callable, Sequence
from typing import Any

logger = logging.getLogger(__name__)

_WS = re.compile(r"\s+")


def _norm_entity(name: str) -> str:
    """Canonical-comparison form of an entity surface string."""
    return _WS.sub(" ", str(name).strip().lower())


def canonical_consistency(facts: Sequence[Any]) -> float:
    """Fraction of entity mentions that use a *consistent* surface form (CONCEPT:AU-AHE.optimization.dspy-optimization-kg-extraction).

    Entities that differ only by casing/whitespace (``"Acme Corp"`` vs ``"acme  corp"``)
    are fragmentation — the same real entity under multiple surface forms. Returns
    ``1.0`` when every normalized entity has exactly one surface form, decreasing as
    fragmentation grows. Deterministic; no embedder needed.
    """
    forms: dict[str, set[str]] = {}
    for f in facts:
        for key in ("subject", "object"):
            raw = f.get(key) if isinstance(f, dict) else getattr(f, key, "")
            if not raw:
                continue
            forms.setdefault(_norm_entity(raw), set()).add(str(raw))
    if not forms:
        return 1.0
    fragmented = sum(1 for v in forms.values() if len(v) > 1)
    return 1.0 - fragmented / len(forms)


def extraction_quality(
    facts: Sequence[Any],
    *,
    embed_fn: Callable[[str], list[float]] | None = None,
    dedup_threshold: float = 0.90,
    dedup_weight: float = 0.6,
) -> dict[str, float]:
    """Self-supervised quality of an extracted fact set (CONCEPT:AU-AHE.optimization.dspy-optimization-kg-extraction).

    Combines two label-free signals:

    * **non-duplicate rate** — ``1 - (duplicates / total)`` via the embedding
      :class:`~agent_utilities.knowledge_graph.extraction.fact_extractor.FactDeduper`
      (inject ``embed_fn`` for an offline/deterministic run);
    * **canonical consistency** — :func:`canonical_consistency`.

    Returns ``{"score", "non_duplicate_rate", "canonical_consistency", "n_facts"}`` with
    ``score`` the ``dedup_weight`` blend. An empty fact set scores ``0`` (an extraction
    that finds nothing is not "perfect").
    """
    from agent_utilities.knowledge_graph.extraction.fact_extractor import (
        ExtractedFact,
        FactDeduper,
    )

    fact_objs: list[ExtractedFact] = []
    for f in facts:
        if isinstance(f, ExtractedFact):
            fact_objs.append(f)
        elif isinstance(f, dict):
            subject = str(f.get("subject") or "")
            obj = str(f.get("object") or "")
            if not subject and not obj:
                continue
            fact_objs.append(
                ExtractedFact(
                    subject=subject,
                    predicate=str(f.get("predicate") or ""),
                    object=obj,
                    title=str(f.get("title") or ""),
                    description=str(f.get("description") or ""),
                )
            )
    n = len(fact_objs)
    if n == 0:
        return {
            "score": 0.0,
            "non_duplicate_rate": 0.0,
            "canonical_consistency": 0.0,
            "n_facts": 0,
        }

    deduper = FactDeduper(embed_fn=embed_fn, threshold=dedup_threshold)
    duplicates = 0
    for fact in fact_objs:
        is_dup, _sim = deduper.check(fact)
        if is_dup:
            duplicates += 1
    non_dup = 1.0 - duplicates / n
    consistency = canonical_consistency(fact_objs)
    score = dedup_weight * non_dup + (1.0 - dedup_weight) * consistency
    return {
        "score": score,
        "non_duplicate_rate": non_dup,
        "canonical_consistency": consistency,
        "n_facts": float(n),
    }


def optimize_extraction_prompt(
    documents: Sequence[str],
    *,
    optimizer_name: str = "BootstrapFewShot",
    embed_fn: Callable[[str], list[float]] | None = None,
) -> dict[str, Any] | None:
    """DSPy-optimize the extraction prompt against :func:`extraction_quality`.

    CONCEPT:AU-AHE.optimization.dspy-optimization-kg-extraction. Wraps extraction as a DSPy module (``document → facts_json``) whose
    metric parses the predicted facts and scores them self-supervised. Best-effort: returns
    ``None`` when DSPy/an LLM is unavailable or ``documents`` is empty (so it is safe to
    call from a daemon tick or the optimize-component surface). Never raises.
    """
    if not documents:
        return None
    try:
        import json

        import dspy

        from agent_utilities.harness.dspy_optimization import build_optimizer
        from agent_utilities.knowledge_graph.extraction.fact_extractor import (
            FACT_EXTRACTION_PROMPT,
        )

        class ExtractSignature(dspy.Signature):
            document: str = dspy.InputField(desc="Source document text.")
            facts_json: str = dspy.OutputField(
                desc="JSON object with a 'facts' array of {subject,predicate,object}."
            )

        ExtractSignature.__doc__ = FACT_EXTRACTION_PROMPT

        class ExtractModule(dspy.Module):
            def __init__(self) -> None:
                super().__init__()
                self.predict = dspy.Predict(ExtractSignature)

            def forward(self, document: str) -> Any:
                return self.predict(document=document)

        def metric(example: Any, pred: Any, trace: Any = None) -> float:
            try:
                payload = json.loads(getattr(pred, "facts_json", "") or "{}")
                facts = payload.get(
                    "facts", payload if isinstance(payload, list) else []
                )
            except Exception:  # noqa: BLE001
                return 0.0
            return extraction_quality(facts, embed_fn=embed_fn)["score"]

        trainset = [
            dspy.Example(document=d, facts_json="").with_inputs("document")
            for d in documents
        ]
        optimizer = build_optimizer(optimizer_name, metric)
        compiled = optimizer.compile(ExtractModule(), trainset=trainset)
        return {
            "compiled_state": compiled.dump_state(),
            "optimizer": optimizer_name,
            "documents": len(documents),
        }
    except Exception as e:  # noqa: BLE001 - best-effort, LLM-gated
        logger.warning("optimize_extraction_prompt failed: %s", e)
        return None
