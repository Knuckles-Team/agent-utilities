#!/usr/bin/python
from __future__ import annotations

"""Neural cross-encoder relevance scorer for ScoreGate / reranking.

CONCEPT:KG-2.85 — Adaptive Chunk Selection (neural cross-encoder half)

This module supplies the *real* cross-encoder relevance signal that
:mod:`agent_utilities.knowledge_graph.retrieval.score_gate` fuses against the
fast bi-encoder (vector) score, and that
:mod:`agent_utilities.knowledge_graph.retrieval.reasoning_reranker` reranks with.
Where :class:`LexicalRelevanceScorer` is a deterministic *proxy* (token overlap,
no model), :class:`NeuralCrossEncoderReranker` scores a ``(query, passage)`` pair
*jointly* through a trained cross-encoder — strictly more faithful — and satisfies
the same :class:`RerankScorer` protocol so it is a drop-in replacement.

Native-by-default / opt-in-by-auto-detection
--------------------------------------------
The heavy model dependency (``sentence-transformers`` / ``torch``) is OPTIONAL.
Importing this module must never import torch or sentence-transformers; the model
library is touched only lazily, on first scoring or via :meth:`is_available`.
Wiring code calls :func:`build_rerank_scorer`, which auto-detects: it returns the
neural cross-encoder when a model is injected or the library is importable, and
otherwise gracefully falls back to the always-available lexical proxy. No
environment is read — the model name and batch size come from arguments/constants.

Determinism
-----------
A cross-encoder in eval/inference mode is a deterministic function of its weights
and inputs, so for a fixed model the scores are reproducible. Raw model logits are
squashed to ``[0, 1]`` with a numerically-stable logistic sigmoid (stdlib math).
"""

import math
from typing import Any

__all__ = [
    "DEFAULT_CROSS_ENCODER_MODEL",
    "NeuralCrossEncoderReranker",
    "build_rerank_scorer",
    "sigmoid",
]

# Default distilled MS-MARCO cross-encoder; small and CPU-friendly. A constant,
# not an environment read — callers override via the ``model_name`` argument.
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def sigmoid(value: float) -> float:
    """Numerically-stable logistic squash of a raw logit into ``(0, 1)``.

    Cross-encoders emit unbounded relevance logits; rerankers and ScoreGate
    expect calibrated ``[0, 1]`` scores. The two-branch form avoids ``exp``
    overflow on large-magnitude logits.
    """
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _compose_query(query: str, instruction: str) -> str:
    """Fold an optional task instruction onto the query side of the pair.

    The cross-encoder scores a single ``(query, passage)`` pair, so an
    instruction (task/intent) is prepended to the query text rather than passed
    as a third field — keeping the joint-encoding interface intact while making
    scoring instruction-aware.
    """
    instruction = (instruction or "").strip()
    if instruction:
        return f"{instruction}\n{query}"
    return query


class NeuralCrossEncoderReranker:
    """Real cross-encoder relevance scorer satisfying the RerankScorer protocol (CONCEPT:KG-2.85).

    Wraps a sentence-transformers CrossEncoder (or a compatible injected model) that scores a
    (query, passage) PAIR jointly — strictly more faithful than the lexical proxy. The model dep
    is optional; absent it, callers fall back to LexicalRelevanceScorer.
    """

    name: str = "neural_cross_encoder"

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
        *,
        model: Any = None,
        batch_size: int = 32,
    ) -> None:
        """Construct the scorer without importing any ML library.

        Args:
            model_name: Cross-encoder identifier to lazily load when no model is
                injected. Used by :class:`CrossEncoder` on first scoring.
            model: An already-constructed, compatible model exposing
                ``predict(list[tuple[str, str]]) -> list[float]`` (raw logits).
                When supplied it is used as-is and no library is ever imported.
            batch_size: Forwarded to the model's ``predict`` for batched scoring.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        # Lazily-loaded model handle; an injected model short-circuits loading.
        self._model: Any = model

    def _ensure_model(self) -> Any:
        """Return the model, loading a CrossEncoder on first use if none injected.

        Raises:
            ImportError: If sentence-transformers is not installed and no model
                was injected — callers should fall back to the lexical proxy.
        """
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - exercised only with deps
            raise ImportError(
                "sentence-transformers is required for NeuralCrossEncoderReranker "
                "when no model is injected; install it or use build_rerank_scorer() "
                "to fall back to the lexical proxy."
            ) from exc
        self._model = CrossEncoder(self.model_name)
        return self._model

    def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Run the model over ``(query, passage)`` pairs, returning raw logits."""
        model = self._ensure_model()
        raw = model.predict(pairs, batch_size=self.batch_size)
        # Accept numpy arrays, lists, or scalars uniformly.
        try:
            return [float(x) for x in raw]
        except TypeError:
            return [float(raw)]

    def score(self, query: str, text: str, instruction: str = "") -> float:
        """Return a normalized ``[0, 1]`` relevance for the pair.

        The optional ``instruction`` is folded onto the query side; the joint
        cross-encoder logit is sigmoid-squashed into ``[0, 1]``.
        """
        composed = _compose_query(query, instruction)
        logits = self._predict([(composed, text)])
        return sigmoid(logits[0]) if logits else 0.0

    def score_batch(
        self, query: str, texts: list[str], instruction: str = ""
    ) -> list[float]:
        """Score one query against many passages, returning ``[0, 1]`` relevances.

        A single batched ``predict`` call over all pairs, each logit squashed.
        """
        if not texts:
            return []
        composed = _compose_query(query, instruction)
        pairs = [(composed, text) for text in texts]
        logits = self._predict(pairs)
        return [sigmoid(value) for value in logits]

    @staticmethod
    def is_available(model_name: str = DEFAULT_CROSS_ENCODER_MODEL) -> bool:
        """Return True iff the sentence-transformers library is importable.

        Does not download or load weights — it only probes that the cross-encoder
        machinery can be constructed, so auto-detection is cheap and crash-free
        when the dependency is absent. ``model_name`` is accepted for API symmetry
        and forward extension (e.g. local-path probing) without env reads.
        """
        _ = model_name
        try:
            import importlib.util

            return importlib.util.find_spec("sentence_transformers") is not None
        except (ImportError, ValueError):  # pragma: no cover - defensive
            return False


def build_rerank_scorer(
    *,
    prefer_neural: bool = True,
    model: Any = None,
    model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
) -> Any:
    """Auto-detection factory returning the best available RerankScorer.

    Returns a :class:`NeuralCrossEncoderReranker` when ``prefer_neural`` is set
    and a neural scorer is usable (a ``model`` was injected, or
    :meth:`NeuralCrossEncoderReranker.is_available` reports the library present);
    otherwise it returns a :class:`LexicalRelevanceScorer` (imported lazily from
    :mod:`.reasoning_reranker`). This is what retrieval wiring calls to obtain the
    strongest scorer the deployment can actually run.

    Args:
        prefer_neural: When False, always return the lexical proxy.
        model: Optional pre-built cross-encoder to inject (forces neural).
        model_name: Cross-encoder identifier used for lazy loading / availability.

    Returns:
        An object satisfying the ``RerankScorer`` protocol.
    """
    if prefer_neural and (
        model is not None or NeuralCrossEncoderReranker.is_available(model_name)
    ):
        return NeuralCrossEncoderReranker(model_name, model=model)

    # Lazy import keeps this module's import graph free of even the lexical proxy
    # until a fallback is actually needed.
    from agent_utilities.knowledge_graph.retrieval.reasoning_reranker import (
        LexicalRelevanceScorer,
    )

    return LexicalRelevanceScorer()
