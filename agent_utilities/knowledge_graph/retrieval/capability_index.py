#!/usr/bin/python
from __future__ import annotations

"""Capability-aware designation index (Plan 04 — L2 retrieval).

This module provides :class:`CapabilityIndex`, the structure that turns the
multi-layer knowledge graph's capability ontology
(``providesCapability`` / ``requiresCapability`` / ``swappableWith``) into a
fast *designation* primitive used by the execution plane.

Designation answers the question: *given a task (as a prompt embedding) and an
optional set of required capabilities, which entities (agents/tools/skills)
should handle it?* It does so in two stages:

1. **Capability filtering** — restrict candidates to entities that
   ``providesCapability`` for every ``required_cap`` via O(1) set lookups in a
   ``capability -> set[id]`` inverted index. This replaces the O(n) per-tool
   scan in ``graph/routing.py`` with a set-intersection.
2. **Similarity ranking** — rank the restricted candidate set by cosine
   similarity between the prompt embedding and each entity's embedding, using
   an approximate-nearest-neighbour (ANN) index when available.

The ANN backend is chosen automatically and transparently:

* ``hnswlib`` (``backend == "hnsw"``) when importable — an O(log N) HNSW index.
* a numpy cosine-similarity brute-force ranker (``backend == "numpy"``)
  otherwise.

The public API never requires a network call: ``designate()`` accepts a raw
prompt-embedding vector, so callers (and tests) may pass synthetic vectors.

Layer contract: this is an L2 component. It is consumed by the
:class:`~agent_utilities.knowledge_graph.facade.KnowledgeGraph` facade and,
through it, by the ``graph/*`` execution plane. It has no upward dependencies.
"""

import logging
import pickle  # nosec B403 — only loads index snapshots written by this class's save()
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Optional ANN backend — never a hard import.
try:  # pragma: no cover - import guard
    import hnswlib  # type: ignore

    _HNSW_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    hnswlib = None  # type: ignore
    _HNSW_AVAILABLE = False


__all__ = ["Designation", "CapabilityIndex"]


@dataclass
class Designation:
    """A ranked designation produced by :meth:`CapabilityIndex.designate`.

    Attributes:
        id: The designated entity identifier (agent/tool/skill id).
        score: Cosine similarity between the prompt embedding and the entity
            embedding, in ``[-1.0, 1.0]`` (typically ``[0, 1]`` for the
            non-negative embeddings used in practice).
        capabilities: The set of capabilities the entity provides.
        provenance: How the designation was produced — backend used, the
            required capabilities that gated it, and any alternatives.
    """

    id: str
    score: float
    capabilities: set[str] = field(default_factory=set)
    provenance: dict[str, Any] = field(default_factory=dict)


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Return ``vec`` scaled to unit L2 norm (zero vectors returned as-is)."""
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return vec / norm


class CapabilityIndex:
    """Capability-filtered ANN index over entity embeddings.

    Maintains two structures:

    * a ``capability -> set[id]`` inverted index for O(1) capability filtering;
    * an ANN vector index (``id -> embedding``) for similarity ranking.

    The vector backend is selected automatically: HNSW via ``hnswlib`` if it is
    importable, else a numpy brute-force cosine ranker. The active backend is
    exposed via :attr:`backend` for introspection and tests.

    Args:
        dim: Dimensionality of the embeddings. May be ``None`` and inferred
            from the first :meth:`add`.
        space: Distance space — only ``"cosine"`` is supported (vectors are
            L2-normalized so inner product equals cosine similarity).
        prefer_backend: Force a backend (``"hnsw"`` or ``"numpy"``) instead of
            auto-selection. Used by tests; falls back to numpy if hnsw is
            unavailable.
        max_elements: Initial capacity hint for the HNSW backend (grows
            automatically as needed).
    """

    def __init__(
        self,
        dim: int | None = None,
        *,
        space: str = "cosine",
        prefer_backend: str | None = None,
        max_elements: int = 1024,
    ) -> None:
        if space != "cosine":
            raise ValueError(f"Only 'cosine' space is supported, got {space!r}")
        self._dim = dim
        self._space = space
        self._max_elements = max(1, max_elements)

        # Choose backend.
        if prefer_backend == "numpy":
            self._backend = "numpy"
        elif prefer_backend == "hnsw":
            if not _HNSW_AVAILABLE:
                logger.warning(
                    "hnswlib requested but unavailable; using numpy fallback."
                )
                self._backend = "numpy"
            else:
                self._backend = "hnsw"
        else:
            self._backend = "hnsw" if _HNSW_AVAILABLE else "numpy"

        # capability -> set[id]
        self._cap_to_ids: dict[str, set[str]] = {}
        # id -> set[capability]
        self._id_to_caps: dict[str, set[str]] = {}
        # id -> normalized embedding (always kept; the source of truth for
        # ranking and for rebuilding the HNSW index on resize/load).
        self._id_to_vec: dict[str, np.ndarray] = {}
        # swappableWith adjacency (symmetric)
        self._swappable: dict[str, set[str]] = {}
        # id -> reward EMA in [0, 1] (0.5 = neutral/unproven). Updated by
        # record_outcome() to close the learning loop (Plan 08 Synergy 5).
        self._reward: dict[str, float] = {}

        # HNSW-specific state
        self._hnsw: Any = None
        self._label_to_id: dict[int, str] = {}
        self._id_to_label: dict[str, int] = {}
        self._next_label = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def backend(self) -> str:
        """The active ANN backend: ``"hnsw"`` or ``"numpy"``."""
        return self._backend

    @property
    def dim(self) -> int | None:
        """Embedding dimensionality, or ``None`` if no vectors added yet."""
        return self._dim

    def __len__(self) -> int:
        return len(self._id_to_vec)

    # ------------------------------------------------------------------
    # HNSW helpers
    # ------------------------------------------------------------------
    def _ensure_hnsw(self, capacity_hint: int) -> None:
        """Initialize the HNSW index lazily once ``dim`` is known."""
        if self._backend != "hnsw" or self._dim is None:
            return
        if self._hnsw is None:
            self._hnsw = hnswlib.Index(space="cosine", dim=self._dim)
            init_cap = max(self._max_elements, capacity_hint, 1)
            self._hnsw.init_index(max_elements=init_cap, ef_construction=200, M=16)
            self._hnsw.set_ef(max(50, init_cap))
            self._max_elements = init_cap

    def _hnsw_resize_if_needed(self, additional: int) -> None:
        if self._backend != "hnsw" or self._hnsw is None:
            return
        needed = self._next_label + additional
        if needed > self._max_elements:
            new_cap = max(needed, self._max_elements * 2)
            self._hnsw.resize_index(new_cap)
            self._max_elements = new_cap

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def add(
        self,
        id: str,
        embedding: Any,
        capabilities: Any,
        *,
        swappable_with: Any = None,
    ) -> None:
        """Add (or replace) an entity in the index.

        Args:
            id: Unique entity identifier.
            embedding: A vector (list/tuple/ndarray) of length ``dim``.
            capabilities: Iterable of capability identifiers the entity
                ``providesCapability``.
            swappable_with: Optional iterable of ids this entity is
                ``swappableWith`` (edges are made symmetric).
        """
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vec.size == 0:
            raise ValueError(f"Empty embedding for id {id!r}")
        if self._dim is None:
            self._dim = int(vec.size)
        elif vec.size != self._dim:
            raise ValueError(
                f"Embedding dim mismatch for {id!r}: expected {self._dim}, "
                f"got {vec.size}"
            )

        norm_vec = _l2_normalize(vec)
        is_update = id in self._id_to_vec
        self._id_to_vec[id] = norm_vec

        # Capability maps — clear old assignments on update.
        if is_update:
            for cap in self._id_to_caps.get(id, set()):
                self._cap_to_ids.get(cap, set()).discard(id)
        caps = {str(c) for c in (capabilities or [])}
        self._id_to_caps[id] = caps
        for cap in caps:
            self._cap_to_ids.setdefault(cap, set()).add(id)

        # swappableWith — symmetric.
        if swappable_with:
            partners = {str(p) for p in swappable_with if str(p) != id}
            self._swappable.setdefault(id, set()).update(partners)
            for p in partners:
                self._swappable.setdefault(p, set()).add(id)

        # ANN index maintenance.
        if self._backend == "hnsw":
            self._ensure_hnsw(capacity_hint=len(self._id_to_vec))
            if id in self._id_to_label:
                label = self._id_to_label[id]
            else:
                label = self._next_label
                self._next_label += 1
                self._id_to_label[id] = label
                self._label_to_id[label] = id
            self._hnsw_resize_if_needed(additional=1)
            self._hnsw.add_items(norm_vec.reshape(1, -1), np.array([label]))

    def build_from_edges(self, nodes: Any) -> None:
        """Bulk-load the index from an iterable of node descriptors.

        Each node may be a mapping or an object exposing ``id``, ``embedding``,
        ``capabilities`` (or ``provides``/``providesCapability``), and an
        optional ``swappable_with`` (or ``swappableWith``).

        Args:
            nodes: Iterable of node descriptors.
        """
        for node in nodes:
            if isinstance(node, dict):
                getter = node.get
            else:

                def getter(key: str, default: Any = None, _n: Any = node) -> Any:
                    return getattr(_n, key, default)

            nid = getter("id")
            if nid is None:
                continue
            emb = getter("embedding")
            if emb is None:
                continue
            caps = (
                getter("capabilities")
                or getter("provides")
                or getter("providesCapability")
                or []
            )
            swap = getter("swappable_with") or getter("swappableWith") or None
            self.add(str(nid), emb, caps, swappable_with=swap)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def _candidate_ids(self, required_caps: set[str] | None) -> set[str] | None:
        """Return the candidate id set after capability filtering.

        Returns ``None`` to mean "no restriction" (all ids), or a concrete set
        (possibly empty) when ``required_caps`` is provided.
        """
        if not required_caps:
            return None
        # Intersection of provider sets — O(sum of set sizes).
        candidate: set[str] | None = None
        for cap in required_caps:
            providers = self._cap_to_ids.get(cap, set())
            if candidate is None:
                candidate = set(providers)
            else:
                candidate &= providers
            if not candidate:
                return set()
        return candidate or set()

    def designate(
        self,
        prompt_embedding: Any,
        required_caps: Any = None,
        k: int = 5,
        reward_weight: float = 0.15,
    ) -> list[Designation]:
        """Designate the top-``k`` entities for a task.

        Args:
            prompt_embedding: The task/query embedding vector.
            required_caps: Optional iterable of capabilities the entity must
                provide. Candidates are restricted to the set-intersection of
                providers before ranking.
            k: Maximum number of designations to return.

        Returns:
            Up to ``k`` :class:`Designation` objects sorted by descending
            similarity score.
        """
        if k <= 0 or not self._id_to_vec:
            return []

        query = np.asarray(prompt_embedding, dtype=np.float32).reshape(-1)
        if self._dim is not None and query.size != self._dim:
            raise ValueError(
                f"Prompt embedding dim mismatch: expected {self._dim}, "
                f"got {query.size}"
            )
        query = _l2_normalize(query)

        req = {str(c) for c in required_caps} if required_caps else None
        candidates = self._candidate_ids(req)
        if candidates is not None and not candidates:
            return []

        # Oversample by pure similarity (keeps the ANN fast path intact), then
        # blend in the learned reward EMA so proven designations rise to the top
        # (Plan 08 Synergy 5). reward_weight=0 disables the boost.
        if reward_weight and self._reward:
            pool_size = (
                len(candidates) if candidates is not None else len(self._id_to_vec)
            )
            oversample = min(pool_size, max(k, k * 4))
            ranked = self._rank(query, candidates, oversample)
            ranked = sorted(
                ranked,
                key=lambda t: t[1]
                + reward_weight * (self._reward.get(t[0], 0.5) - 0.5),
                reverse=True,
            )
        else:
            ranked = self._rank(query, candidates, k)

        results: list[Designation] = []
        for nid, score in ranked[:k]:
            caps = set(self._id_to_caps.get(nid, set()))
            provenance: dict[str, Any] = {
                "backend": self._backend,
                "required_caps": sorted(req) if req else [],
                "capability_filtered": candidates is not None,
                "candidate_pool_size": (
                    len(candidates) if candidates is not None else len(self._id_to_vec)
                ),
            }
            provenance["reward"] = round(self._reward.get(nid, 0.5), 4)
            alts = self.alternatives(nid)
            if alts:
                provenance["alternatives"] = alts
            results.append(
                Designation(
                    id=nid,
                    score=float(score),
                    capabilities=caps,
                    provenance=provenance,
                )
            )
        return results

    def _rank(
        self, query: np.ndarray, candidates: set[str] | None, k: int
    ) -> list[tuple[str, float]]:
        """Rank ids by cosine similarity to ``query``.

        Uses HNSW when the backend is hnsw *and* there is no capability
        restriction (HNSW lacks native id pre-filtering); otherwise ranks the
        restricted set with a numpy brute-force cosine computation.
        """
        if self._backend == "hnsw" and candidates is None and self._hnsw is not None:
            n = len(self._id_to_vec)
            top = min(k, n)
            labels, distances = self._hnsw.knn_query(query.reshape(1, -1), k=top)
            out: list[tuple[str, float]] = []
            for label, dist in zip(labels[0], distances[0], strict=False):
                nid = self._label_to_id.get(int(label))
                if nid is None:
                    continue
                # hnswlib cosine distance == 1 - cosine_similarity.
                out.append((nid, 1.0 - float(dist)))
            return out

        # numpy brute-force over the (optionally restricted) candidate set.
        ids = (
            [i for i in candidates if i in self._id_to_vec]
            if candidates is not None
            else list(self._id_to_vec.keys())
        )
        if not ids:
            return []
        matrix = np.stack([self._id_to_vec[i] for i in ids])  # already normalized
        sims = matrix @ query  # cosine sim since both sides L2-normalized
        order = np.argsort(-sims)[:k]
        return [(ids[int(j)], float(sims[int(j)])) for j in order]

    def alternatives(self, id: str) -> list[str]:
        """Return ids ``swappableWith`` the given entity (sorted, stable)."""
        return sorted(self._swappable.get(id, set()))

    # ------------------------------------------------------------------
    # Reward write-back (Plan 08 Synergy 5 — closes the learning loop)
    # ------------------------------------------------------------------
    def record_outcome(
        self,
        id: str,
        success: bool | None = None,
        reward: float | None = None,
        alpha: float = 0.3,
    ) -> float:
        """Record an execution outcome for a designated entity.

        Updates an exponential-moving-average reward in ``[0, 1]`` (0.5 =
        neutral). Pass either ``success`` (mapped to 1.0/0.0) or an explicit
        ``reward`` in ``[0, 1]``. The EMA weight ``alpha`` bounds how fast a
        single outcome moves the score, preventing reward-hacking spikes.
        Returns the updated reward.
        """
        if reward is None:
            if success is None:
                raise ValueError("record_outcome requires success or reward")
            reward = 1.0 if success else 0.0
        reward = min(1.0, max(0.0, float(reward)))
        prev = self._reward.get(id, 0.5)
        updated = (1.0 - alpha) * prev + alpha * reward
        self._reward[id] = updated
        return updated

    def reward_of(self, id: str) -> float:
        """Current reward EMA for ``id`` (0.5 if no outcomes recorded yet)."""
        return self._reward.get(id, 0.5)

    def decay_rewards(self, factor: float = 0.99) -> None:
        """Decay all rewards toward the neutral 0.5 prior (call periodically).

        Keeps proven-but-stale designations from dominating forever and lets
        the cold-path exploration fraction resurface newer entities.
        """
        factor = min(1.0, max(0.0, factor))
        for id, r in list(self._reward.items()):
            self._reward[id] = 0.5 + (r - 0.5) * factor

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Persist the index to ``path`` (a directory).

        The capability maps, embeddings, and swappable adjacency are pickled;
        the HNSW index (when active) is additionally saved via its native
        serializer so reloads stay O(log N).

        Args:
            path: Destination directory (created if absent).
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        meta = {
            "backend": self._backend,
            "dim": self._dim,
            "space": self._space,
            "max_elements": self._max_elements,
            "cap_to_ids": {c: sorted(ids) for c, ids in self._cap_to_ids.items()},
            "id_to_caps": {i: sorted(c) for i, c in self._id_to_caps.items()},
            "swappable": {i: sorted(s) for i, s in self._swappable.items()},
            "reward": dict(self._reward),
            "id_to_label": self._id_to_label,
            "next_label": self._next_label,
            "ids": list(self._id_to_vec.keys()),
        }
        with open(path / "capability_index.pkl", "wb") as fh:
            pickle.dump(meta, fh)

        # Embeddings stored as a stacked array + ordered id list for fast load.
        ids = list(self._id_to_vec.keys())
        if ids:
            arr = np.stack([self._id_to_vec[i] for i in ids])
        else:
            arr = np.zeros((0, self._dim or 0), dtype=np.float32)
        np.save(path / "embeddings.npy", arr)

        if self._backend == "hnsw" and self._hnsw is not None:
            self._hnsw.save_index(str(path / "hnsw.bin"))

    @classmethod
    def load(cls, path: str | Path) -> CapabilityIndex:
        """Reload an index previously written by :meth:`save`.

        Args:
            path: Directory passed to :meth:`save`.

        Returns:
            A reconstructed :class:`CapabilityIndex` with identical ranking
            behaviour.
        """
        path = Path(path)
        # nosec B301 — deserializes only a snapshot produced by this class's save(),
        # a trusted local artifact, not untrusted external input.
        with open(path / "capability_index.pkl", "rb") as fh:
            meta = pickle.load(fh)  # nosec B301

        idx = cls(
            dim=meta["dim"],
            space=meta["space"],
            prefer_backend=meta["backend"],
            max_elements=meta.get("max_elements", 1024),
        )
        idx._cap_to_ids = {c: set(ids) for c, ids in meta["cap_to_ids"].items()}
        idx._id_to_caps = {i: set(c) for i, c in meta["id_to_caps"].items()}
        idx._swappable = {i: set(s) for i, s in meta["swappable"].items()}
        idx._reward = dict(meta.get("reward", {}))
        idx._id_to_label = dict(meta["id_to_label"])
        idx._label_to_id = {v: k for k, v in idx._id_to_label.items()}
        idx._next_label = meta["next_label"]

        ids = meta["ids"]
        arr = np.load(path / "embeddings.npy")
        for i, nid in enumerate(ids):
            idx._id_to_vec[nid] = np.asarray(arr[i], dtype=np.float32)

        if idx._backend == "hnsw":
            hnsw_path = path / "hnsw.bin"
            if hnsw_path.exists() and idx._dim is not None:
                idx._hnsw = hnswlib.Index(space="cosine", dim=idx._dim)
                idx._hnsw.load_index(str(hnsw_path), max_elements=idx._max_elements)
                idx._hnsw.set_ef(max(50, idx._max_elements))
            elif idx._dim is not None and ids:
                # Native file missing — rebuild from embeddings to preserve
                # the hnsw backend contract.
                idx._hnsw = None
                idx._next_label = 0
                idx._id_to_label = {}
                idx._label_to_id = {}
                vecs = idx._id_to_vec
                idx._id_to_vec = {}
                for nid, vec in vecs.items():
                    caps = idx._id_to_caps.get(nid, set())
                    # re-add restores both the vector map and hnsw index
                    idx.add(nid, vec, caps)
        return idx
