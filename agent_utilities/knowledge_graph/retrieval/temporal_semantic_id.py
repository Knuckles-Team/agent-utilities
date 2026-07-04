#!/usr/bin/python
from __future__ import annotations

"""Temporal semantic IDs for generative retrieval (CONCEPT:AU-KG.query.chronoid-fits-residual-quantization).

Implements ChronoID-style **temporal semantic IDs** following the method of
"ChronoID: Infusing Explicit Temporal Signals into Semantic IDs for Generative
Recommendation" (Meta, arXiv 2606.x).

A *semantic ID* is a short tuple of discrete codes that stands in for a dense
content embedding, so a generative retrieval / recommendation model can emit
items autoregressively as code sequences instead of scoring a full catalog.
ChronoID's contribution is to make recency a first-class, *explicit* token of
that ID: each semantic ID is prefixed by a discrete **time-bucket token**
derived from the item's ``event_time``, so the generator can condition on (and
the index can pivot on) how recent an item is rather than relying on the content
codes alone.

This module provides :class:`TemporalSemanticIdEncoder`, which builds the
content codes via *residual quantization* (an RQ-VAE-lite): a stack of
``n_codebooks`` codebooks where level 0 quantizes the (L2-normalized) vector and
each subsequent level quantizes the residual left by the prior levels. Codebooks
are learned with a small, seeded, dependency-light numpy k-means so the whole
encoder is deterministic given ``seed``.

Layer contract: this is a pure, self-contained L2 retrieval helper. It depends
only on numpy, performs no I/O and no network calls, and shares the recency /
``event_time`` framing of
:mod:`agent_utilities.knowledge_graph.core.bitemporal` (age = ``now`` minus
``event_time``; ``None`` event-time means "unknown") and the defensive numpy
L2-normalization used by
:mod:`agent_utilities.knowledge_graph.retrieval.capability_index`.
"""

from collections.abc import Sequence

from agent_utilities.numeric import xp as np

__all__ = ["TemporalSemanticIdEncoder"]

_SECONDS_PER_DAY = 86400.0


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Return ``vec`` scaled to unit L2 norm, with NaN/inf scrubbed to 0.

    Zero (or all-non-finite) vectors are returned unchanged after scrubbing, so
    quantization never sees NaN. Mirrors ``capability_index._l2_normalize`` plus
    a defensive ``nan_to_num``.
    """
    clean = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    norm = float(np.linalg.norm(clean))
    if norm == 0.0:
        return clean
    return clean / norm


class TemporalSemanticIdEncoder:
    """Encode content embeddings into recency-prefixed semantic IDs.

    Args:
        n_codebooks: Number of residual quantization levels (the length of the
            content-code portion of every semantic ID).
        codebook_size: Number of centroids per codebook; codes fall in
            ``[0, codebook_size)``.
        n_time_buckets: Number of discrete recency buckets, including the
            dedicated trailing "unknown" bucket (see :meth:`time_bucket`).
        time_span_days: Age horizon, in days, mapped across the recency buckets;
            ages at or beyond it clamp to the oldest known bucket.
        seed: RNG seed making codebook training fully deterministic.
    """

    def __init__(
        self,
        *,
        n_codebooks: int = 2,
        codebook_size: int = 64,
        n_time_buckets: int = 16,
        time_span_days: float = 365.0,
        seed: int = 0,
    ) -> None:
        if n_codebooks < 1:
            raise ValueError(f"n_codebooks must be >= 1, got {n_codebooks}")
        if codebook_size < 1:
            raise ValueError(f"codebook_size must be >= 1, got {codebook_size}")
        if n_time_buckets < 2:
            raise ValueError(
                f"n_time_buckets must be >= 2 (one known + one unknown), "
                f"got {n_time_buckets}"
            )
        if time_span_days <= 0.0:
            raise ValueError(f"time_span_days must be > 0, got {time_span_days}")
        self._n_codebooks = n_codebooks
        self._codebook_size = codebook_size
        self._n_time_buckets = n_time_buckets
        self._time_span_days = float(time_span_days)
        self._seed = seed
        # One (k, dim) centroid matrix per residual level; empty until fit().
        self._codebooks: list[np.ndarray] = []
        self._dim: int | None = None

    @property
    def is_fitted(self) -> bool:
        """True once :meth:`fit` has built the residual codebooks."""
        return bool(self._codebooks)

    @property
    def n_codebooks(self) -> int:
        """Number of residual quantization levels (content-code length)."""
        return self._n_codebooks

    @property
    def codebook_size(self) -> int:
        """Centroids per codebook; codes are in ``[0, codebook_size)``."""
        return self._codebook_size

    @property
    def n_time_buckets(self) -> int:
        """Number of recency buckets, including the trailing unknown bucket."""
        return self._n_time_buckets

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _kmeans(
        self, data: np.ndarray, rng: np.random.Generator, iters: int = 10
    ) -> np.ndarray:
        """Seeded k-means returning a ``(k, dim)`` centroid matrix.

        ``k`` is capped at the number of samples. Centroids are initialized by
        sampling distinct rows; empty clusters are reseeded to a random point
        each iteration so no centroid is left undefined.
        """
        n_samples = data.shape[0]
        k = min(self._codebook_size, n_samples)
        init_idx = rng.choice(n_samples, size=k, replace=False)
        centroids = data[init_idx].copy()
        for _ in range(iters):
            # Assign each point to its nearest centroid (squared L2).
            dists = (
                np.sum(data**2, axis=1)[:, None]
                - 2.0 * data @ centroids.T
                + np.sum(centroids**2, axis=1)[None, :]
            )
            labels = np.argmin(dists, axis=1)
            for c in range(k):
                members = data[labels == c]
                if members.shape[0] == 0:
                    # Reseed an empty cluster to a random point (determinism via rng).
                    centroids[c] = data[rng.integers(0, n_samples)]
                else:
                    centroids[c] = members.mean(axis=0)
        return centroids

    def fit(self, embeddings: Sequence[Sequence[float]]) -> TemporalSemanticIdEncoder:
        """Build the residual codebooks from a corpus of embeddings.

        Level 0 is trained on the L2-normalized vectors; each subsequent level
        is trained on the residual left after subtracting the prior levels'
        nearest centroids. Deterministic given ``seed``.

        Args:
            embeddings: A non-empty sequence of equal-length content vectors.

        Returns:
            ``self`` (fitted), for chaining.
        """
        arr = np.asarray(embeddings, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] == 0:
            raise ValueError("fit() requires a non-empty 2-D sequence of vectors")
        data = np.vstack([_l2_normalize(row) for row in arr])
        self._dim = int(data.shape[1])

        rng = np.random.default_rng(self._seed)
        self._codebooks = []
        residual = data
        for _ in range(self._n_codebooks):
            centroids = self._kmeans(residual, rng)
            self._codebooks.append(centroids)
            # Subtract each point's assigned centroid to form the next residual.
            dists = (
                np.sum(residual**2, axis=1)[:, None]
                - 2.0 * residual @ centroids.T
                + np.sum(centroids**2, axis=1)[None, :]
            )
            labels = np.argmin(dists, axis=1)
            residual = residual - centroids[labels]
        return self

    # ------------------------------------------------------------------
    # Temporal token
    # ------------------------------------------------------------------
    def time_bucket(self, event_time_epoch: float | None, *, now_epoch: float) -> int:
        """Bucket an event's recency into ``[0, n_time_buckets)``.

        The age ``now_epoch - event_time_epoch`` is mapped across
        ``[0, time_span_days]`` into the first ``n_time_buckets - 1`` *known*
        buckets, so the most recent items get bucket ``0`` and items at or
        beyond the horizon clamp to bucket ``n_time_buckets - 2``. Future
        timestamps (negative age) also map to bucket ``0``.

        Convention: a ``None`` event time is "unknown" and returns the dedicated
        last bucket ``n_time_buckets - 1``, kept distinct from any known-age
        bucket so the generator can tell "very old" from "no timestamp".
        """
        if event_time_epoch is None:
            return self._n_time_buckets - 1
        known = self._n_time_buckets - 1  # buckets reserved for known ages
        age_days = (now_epoch - float(event_time_epoch)) / _SECONDS_PER_DAY
        if age_days <= 0.0:
            return 0
        frac = age_days / self._time_span_days
        bucket = int(frac * known)
        if bucket >= known:
            bucket = known - 1
        return bucket

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def encode_content(self, embedding: Sequence[float]) -> tuple[int, ...]:
        """Return the residual semantic codes for ``embedding`` (no time token).

        Args:
            embedding: A content vector of the fitted dimensionality.

        Returns:
            A tuple of ``n_codebooks`` codes, each in ``[0, codebook_size)``.
        """
        if not self.is_fitted or self._dim is None:
            raise RuntimeError("encoder is not fitted; call fit() first")
        vec = np.asarray(embedding, dtype=np.float64).reshape(-1)
        if vec.size != self._dim:
            raise ValueError(
                f"Embedding dim mismatch: expected {self._dim}, got {vec.size}"
            )
        residual = _l2_normalize(vec)
        codes: list[int] = []
        for centroids in self._codebooks:
            dists = np.sum((centroids - residual) ** 2, axis=1)
            code = int(np.argmin(dists))
            codes.append(code)
            residual = residual - centroids[code]
        return tuple(codes)

    def encode(
        self,
        embedding: Sequence[float],
        event_time_epoch: float | None,
        *,
        now_epoch: float,
    ) -> tuple[int, ...]:
        """Return the full temporal semantic ID: ``(time_bucket, *content_codes)``.

        The explicit recency token leads (per ChronoID) so generative retrieval
        conditions on time before content. Deterministic for a fixed encoder and
        identical ``(embedding, event_time_epoch, now_epoch)``.
        """
        bucket = self.time_bucket(event_time_epoch, now_epoch=now_epoch)
        return (bucket, *self.encode_content(embedding))
