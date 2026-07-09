#!/usr/bin/python
from __future__ import annotations

"""Implicit-reasoning generative recommendation over semantic IDs (CONCEPT:AU-KG.retrieval.pauserec-implicit-reasoning-generative).

Assimilates the *mechanism* of "PauseRec: Implicit Reasoning for LLM-based
Generative Recommendation" (He et al., arXiv:2606.14142) for an agentic
framework rather than for backbone training.

PauseRec observes that LLM-based generative recommenders (GR) represent items as
**Semantic IDs (SIDs)** -- short tuples of discrete codes outside the natural
-language vocabulary -- and that *explicit* Chain-of-Thought over those SIDs is
brittle for three reasons: (1) world knowledge becomes hard to *verbalize* after
reasoning fine-tuning, (2) natural-language and SID token embeddings drift apart
(text<->SID **misalignment**), and (3) recommendation quality is fragile w.r.t.
the exact rationale text. PauseRec's remedy is *implicit* reasoning: it inserts a
short run of trainable ``<pause>`` tokens before SID generation, giving the model
latent computation steps -- optimized only by the next-item objective, with no
decoded rationale -- that bridge the language and SID spaces.

Adaptation framing (important). We are an agentic framework; we do **not** train
an LLM backbone here. We therefore adopt PauseRec's mechanism at *inference /
agentic* time over the SIDs already produced by
:class:`~agent_utilities.knowledge_graph.retrieval.temporal_semantic_id.TemporalSemanticIdEncoder`
(CONCEPT:AU-KG.query.chronoid-fits-residual-quantization):

* **Latent-reasoning budget** -- a configurable number of ``pause_steps``
  deliberate refinement steps performed before a recommendation is emitted. This
  is the inference analogue of PauseRec's literal ``<pause>`` tokens: a dedicated
  latent computation window, not a trained embedding. Each step nudges a working
  representation in SID space toward the catalog items and the user's history it
  is closest to -- collaborative, knowledge-shaped refinement with **no explicit
  rationale string** (matching PauseRec's finding that explicit CoT is brittle
  for SID-based GR). As in the paper, the useful budget saturates: a couple of
  steps already sharpen the ranking.
* **Text<->SID bridge** (:class:`TextSidBridge`) -- a projection that aligns a
  natural-language *query* embedding into the *same* SID space the items occupy,
  by routing the query through the encoder's shared codebooks. This directly
  addresses the paper's text<->SID misalignment: the query's world knowledge can
  shape SID selection because query and items now live in one code space.

Everything is deterministic and dependency-injected: the encoder is passed in,
the working space is the encoder's continuous residual reconstruction, and no
LLM, training, or network call is involved. stdlib + numpy only.

Layer contract: a pure L2 retrieval helper built strictly on top of
``TemporalSemanticIdEncoder``; no I/O, no upward dependencies.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agent_utilities.numeric import NDArray
from agent_utilities.numeric import xp as np

if TYPE_CHECKING:
    from agent_utilities.knowledge_graph.retrieval.temporal_semantic_id import (
        TemporalSemanticIdEncoder,
    )

__all__ = ["Recommendation", "TextSidBridge", "ImplicitReasoningRecommender"]


@dataclass(frozen=True)
class Recommendation:
    """A single ranked recommendation.

    Attributes:
        item_id: The catalog item's identifier.
        semantic_id: The item's content semantic ID (the codebook code tuple).
        score: Ranking score in ``[0, 1]``; higher means a closer match to the
            refined latent target. Returned lists are sorted best-first.
    """

    item_id: str
    semantic_id: tuple[int, ...]
    score: float


def _reconstruct(
    encoder: TemporalSemanticIdEncoder, codes: Sequence[int]
) -> NDArray:
    """Map a content code tuple back to its continuous residual reconstruction.

    The reconstruction is the sum of the chosen centroid at each residual level
    -- the encoder's own continuous stand-in for the discrete SID. This is the
    space in which latent refinement happens, so refinement stays faithful to the
    geometry the codebooks were trained on.
    """
    # ``_codebooks`` is the encoder's per-level centroid stack; codes index it.
    vec = np.zeros(encoder._dim or 0, dtype=np.float64)  # noqa: SLF001
    for level, code in enumerate(codes):
        vec = vec + encoder._codebooks[level][code]  # noqa: SLF001
    return vec


class TextSidBridge:
    """Aligns NL query embeddings into the semantic-ID space (PauseRec text<->SID bridge).

    Routing a natural-language query embedding through the *same* codebooks the
    items were encoded with places the query and the catalog in one shared SID
    space, so the world knowledge carried by the query embedding can shape which
    SIDs are selected -- the inference-time answer to the paper's text<->SID
    embedding misalignment.
    """

    def __init__(self, encoder: TemporalSemanticIdEncoder) -> None:
        """Bind the bridge to a (fitted-or-later-fitted) shared SID encoder."""
        self._encoder = encoder

    def project(
        self, query_embedding: Sequence[float], *, now_epoch: float | None = None
    ) -> tuple[int, ...]:
        """Project an NL query embedding into the catalog's content-SID space.

        A query has no item event time, so -- unlike items -- it carries no
        temporal token; we use :meth:`TemporalSemanticIdEncoder.encode_content`
        so the query lands among the items on content alone. ``now_epoch`` is
        accepted for call-site symmetry with the item-encoding path and is
        deliberately unused (queries are not time-bucketed).

        Returns:
            The content code tuple (length ``encoder.n_codebooks``); every code
            lies in ``[0, encoder.codebook_size)``.
        """
        del now_epoch  # queries are not time-bucketed; kept for call-site symmetry
        return self._encoder.encode_content(query_embedding)


class ImplicitReasoningRecommender:
    """PauseRec-style implicit-reasoning generative recommender over SIDs (CONCEPT:AU-KG.retrieval.pauserec-implicit-reasoning-generative).

    Recommendation is framed as generation in SID space: a query is projected
    into the shared code space, refined for a latent-reasoning budget of
    ``pause_steps`` deliberate steps (no decoded rationale), then catalog items
    are ranked by proximity to the refined target. World knowledge (carried by
    the query embedding) and collaborative signal (the user's history SIDs)
    jointly shape the target during refinement.
    """

    def __init__(
        self, encoder: TemporalSemanticIdEncoder, *, pause_steps: int = 2
    ) -> None:
        """Bind the recommender to a shared SID encoder and a latent budget.

        Args:
            encoder: The shared semantic-ID encoder. Items and queries are both
                placed in this encoder's code space (dependency injection makes
                the recommender unit-testable with a synthetic encoder).
            pause_steps: The latent-reasoning budget -- number of deliberate
                refinement steps before emitting a recommendation. ``0`` means
                rank directly off the projected query (no implicit reasoning);
                positive values sharpen the target toward nearby catalog items
                and history. Must be ``>= 0``.
        """
        if pause_steps < 0:
            raise ValueError(f"pause_steps must be >= 0, got {pause_steps}")
        self._encoder = encoder
        self._pause_steps = pause_steps
        self._bridge = TextSidBridge(encoder)
        # Catalog state, populated by fit_catalog().
        self._item_ids: list[str] = []
        self._item_sids: list[tuple[int, ...]] = []  # content codes only
        self._item_vectors: NDArray = np.empty((0, 0), dtype=np.float64)

    @property
    def pause_steps(self) -> int:
        """The configured latent-reasoning budget (number of refinement steps)."""
        return self._pause_steps

    @property
    def catalog_size(self) -> int:
        """Number of items currently indexed by :meth:`fit_catalog`."""
        return len(self._item_ids)

    # ------------------------------------------------------------------
    # Catalog fitting
    # ------------------------------------------------------------------
    def fit_catalog(
        self,
        items: list[tuple[str, Sequence[float]]],
        *,
        event_times: dict[str, float] | None = None,
        now_epoch: float | None = None,
    ) -> None:
        """Fit the encoder on item embeddings and assign each item its semantic ID.

        The encoder's codebooks are trained on the item embeddings, then every
        item is encoded. When ``event_times`` is supplied the full *temporal*
        semantic ID is computed (CONCEPT:AU-KG.query.chronoid-fits-residual-quantization) -- exercising the recency token
        -- but ranking and refinement always operate on the content codes, which
        is the space shared with projected queries.

        Args:
            items: ``(item_id, embedding)`` pairs; embeddings must share a length.
            event_times: Optional ``item_id -> event-time epoch`` for temporal
                SIDs. Requires ``now_epoch``.
            now_epoch: Reference "now" epoch for temporal bucketing; required iff
                ``event_times`` is given.

        Raises:
            ValueError: On an empty catalog, or ``event_times`` without
                ``now_epoch``.
        """
        if not items:
            raise ValueError("fit_catalog() requires a non-empty item list")
        if event_times is not None and now_epoch is None:
            raise ValueError("now_epoch is required when event_times is provided")

        ids = [item_id for item_id, _ in items]
        embeddings = [list(vec) for _, vec in items]
        self._encoder.fit(embeddings)

        content_sids: list[tuple[int, ...]] = []
        for item_id, vec in zip(ids, embeddings, strict=False):
            content = self._encoder.encode_content(vec)
            if event_times is not None:
                # Compute the full temporal SID so the recency token is exercised;
                # the leading time bucket is dropped for the content-space index.
                assert now_epoch is not None  # guarded above
                self._encoder.encode(vec, event_times.get(item_id), now_epoch=now_epoch)
            content_sids.append(content)

        self._item_ids = ids
        self._item_sids = content_sids
        self._item_vectors = np.vstack(
            [_reconstruct(self._encoder, sid) for sid in content_sids]
        )

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------
    def _latent_refine(
        self, target: NDArray, history_vectors: NDArray
    ) -> NDArray:
        """Run ``pause_steps`` deterministic latent refinement steps.

        Each step is the inference analogue of one PauseRec ``<pause>`` token: it
        blends the working target toward (a) the centroid of the catalog items it
        is currently closest to -- world-knowledge / content pull -- and (b) the
        centroid of the user's history -- collaborative pull -- then renormalizes.
        No rationale is produced; the computation is purely latent.
        """
        if self._pause_steps == 0 or self._item_vectors.shape[0] == 0:
            return target
        # How many nearest catalog items inform each step (a small salient subset,
        # mirroring the paper's late-pause focus on a few relevant SIDs).
        n_focus = max(1, min(3, self._item_vectors.shape[0]))
        blend = 0.5  # fraction of the move taken toward the pulled centroid per step
        work = target.astype(np.float64).copy()
        for _ in range(self._pause_steps):
            sims = self._item_vectors @ work
            top = np.argsort(-sims)[:n_focus]
            content_centroid = self._item_vectors[top].mean(axis=0)
            if history_vectors.shape[0] > 0:
                history_centroid = history_vectors.mean(axis=0)
                pull = 0.5 * content_centroid + 0.5 * history_centroid
            else:
                pull = content_centroid
            work = (1.0 - blend) * work + blend * pull
            norm = float(np.linalg.norm(work))
            if norm > 0.0:
                work = work / norm
        return work

    def recommend(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int = 5,
        history_sids: Sequence[tuple[int, ...]] | None = None,
        now_epoch: float | None = None,
    ) -> list[Recommendation]:
        """Recommend the ``top_k`` catalog items for a query, best-first.

        Pipeline (PauseRec, adapted to inference):

        1. **Bridge** the NL query into SID space via :class:`TextSidBridge`.
        2. **Latent reasoning**: run ``pause_steps`` deliberate refinement steps
           (:meth:`_latent_refine`) that nudge the working SID-space target
           toward the closest catalog items *and* the user's history SIDs, so
           query world-knowledge and collaborative history jointly shape the
           target -- with no explicit rationale.
        3. **Rank** catalog items by codebook-overlap blended with cosine
           proximity to the refined target; return the top ``top_k``.

        Args:
            query_embedding: NL query embedding of the fitted dimensionality.
            top_k: Maximum number of recommendations to return.
            history_sids: Optional content SIDs of the user's interaction
                history; biases the target toward history-similar items.
            now_epoch: Accepted for call-site symmetry; queries are not
                time-bucketed, so it is unused.

        Returns:
            Up to ``top_k`` :class:`Recommendation` objects sorted by descending
            score.
        """
        if not self._item_ids:
            raise RuntimeError("catalog is empty; call fit_catalog() first")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        query_sid = self._bridge.project(query_embedding, now_epoch=now_epoch)
        target = _l2(_reconstruct(self._encoder, query_sid))

        history_vectors = self._history_vectors(history_sids)
        refined = _l2(self._latent_refine(target, history_vectors))
        refined_sid = self._encoder.encode_content(_reconstruct_back(refined))

        scores = self._rank(refined, refined_sid)
        order = np.argsort(-scores)[:top_k]
        return [
            Recommendation(
                item_id=self._item_ids[i],
                semantic_id=self._item_sids[i],
                score=float(scores[i]),
            )
            for i in order
        ]

    def _history_vectors(
        self, history_sids: Sequence[tuple[int, ...]] | None
    ) -> NDArray:
        """Reconstruct continuous vectors for the user's history SIDs."""
        if not history_sids:
            return np.empty((0, self._item_vectors.shape[1]), dtype=np.float64)
        return np.vstack(
            [_l2(_reconstruct(self._encoder, sid)) for sid in history_sids]
        )

    def _rank(self, refined: NDArray, refined_sid: tuple[int, ...]) -> NDArray:
        """Score every catalog item against the refined latent target.

        The score fuses two signals over the shared SID space: per-level
        codebook-code overlap with the refined target SID (the discrete,
        generative view) and cosine proximity to the refined continuous target
        (the smooth view). Both lie in ``[0, 1]``; the average keeps scores in
        ``[0, 1]``.
        """
        n_levels = self._encoder.n_codebooks
        refined_unit = _l2(refined)
        cos = (self._item_vectors @ refined_unit + 1.0) / 2.0
        overlap = np.array(
            [
                sum(1 for a, b in zip(sid, refined_sid, strict=False) if a == b)
                / n_levels
                for sid in self._item_sids
            ],
            dtype=np.float64,
        )
        return 0.5 * overlap + 0.5 * cos

    def explain_budget(self) -> dict[str, Any]:
        """Surface the latent-reasoning budget and that reasoning is implicit.

        Returns a small, JSON-friendly record describing the configured budget
        and -- per PauseRec -- that the reasoning is *implicit*: there is no
        decoded Chain-of-Thought rationale string, only a latent computation
        window before the SID is emitted.
        """
        return {
            "pause_steps": self._pause_steps,
            "implicit": True,
            "rationale": None,
            "decodes_rationale": False,
            "mechanism": "latent-reasoning budget (PauseRec <pause> analogue)",
            "paper": "PauseRec (arXiv:2606.14142)",
            "concept": "AU-KG.retrieval.pauserec-implicit-reasoning-generative",
        }


def _l2(vec: NDArray) -> NDArray:
    """Return ``vec`` at unit L2 norm (zero vectors pass through unchanged)."""
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return vec / norm


def _reconstruct_back(refined: NDArray) -> NDArray:
    """Identity passthrough naming the refined continuous target for re-encoding.

    The refined target already lives in the encoder's continuous space, so it can
    be re-encoded directly by ``encode_content``; this thin alias documents the
    "latent target -> SID" step at the call site.
    """
    return refined
