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

**AU-P1-3 — engine-native capability index.** This class is now, by design, a
*bounded, non-authoritative cache* — not the source of truth. The authority is
the engine's own native filtered ANN (see
:mod:`agent_utilities.knowledge_graph.retrieval.engine_capability_search`),
queried directly with capability/tenant/policy filters composed into ONE
``query_unified`` plan (``Scan``/``Filter``/``Rank``/``Limit``). This structure
now exists only as: (a) a fallback when the engine ANN is unreachable (dev, or
a lean engine build), and (b) a fast in-process ranking surface kept fresh by
CDC deltas (:mod:`agent_utilities.graph.reactive.engine_subscription`) rather
than a periodic full rebuild — see
``graph/routing/enrichers/capability_designation.py``. Pass
``bounded_cache_size`` to cap it with LRU eviction (:meth:`remove` is called on
the evicted id so no backend — HNSW or numpy — grows without bound); omitting
it keeps the historical unbounded behaviour for the other in-process reward-EMA
consumers (``OutcomeRouter``, ``ReasonerRouter``) that reuse this class as a
generic learner and are not part of the capability-designation cache.
"""

import logging
import pickle  # nosec B403 — only loads index snapshots written by this class's save()
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_utilities.numeric import NDArray
from agent_utilities.numeric import xp as np

logger = logging.getLogger(__name__)

# Optional ANN backend — never a hard import.
try:  # pragma: no cover - import guard
    import hnswlib  # type: ignore

    _HNSW_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    hnswlib = None  # type: ignore
    _HNSW_AVAILABLE = False


# CONCEPT:AU-KG.memory.generation-scoped-selective-reward — generation-scoped selective reward erasure for memory maintenance.
# When an entity is re-ingested with a *materially* different
# embedding, the learned reward EMA was scored under a now-superseded
# representation and is no longer valid evidence about the new content. A
# re-add whose new vector sits at a cosine *distance* greater than this from the
# stored one counts as a new "generation" and triggers selective erasure of that
# id's reward (reset to the neutral prior). Content-stable re-adds keep their
# reward. This is the Red Queen Gödel Machine's epoch-boundary *selective
# erasure* (arXiv:2606.26294) applied to the retrieval router's utility records:
# erase only the evidence tied to the displaced generation, preserve everything
# unrelated, so the router re-climbs under the new regime instead of carrying
# stale (possibly reward-hacked) utility forward forever. One correct value, not
# a flag — bge-m3 minor-edit re-embeds stay well above 0.75 similarity; a
# material rewrite drops below it.
_REWARD_REGEN_DISTANCE = 0.25


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


def _l2_normalize(vec: NDArray) -> NDArray:
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
        bounded_cache_size: When set, caps the number of resident ids —
            ``add()`` evicts the least-recently-touched id (via :meth:`remove`)
            once this many are resident (AU-P1-3: AU keeps only a bounded
            cache; the engine is the authority). ``None`` (default) preserves
            the historical unbounded behaviour.
    """

    def __init__(
        self,
        dim: int | None = None,
        *,
        space: str = "cosine",
        prefer_backend: str | None = None,
        max_elements: int = 1024,
        bounded_cache_size: int | None = None,
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
        self._id_to_vec: dict[str, NDArray] = {}
        # swappableWith adjacency (symmetric)
        self._swappable: dict[str, set[str]] = {}
        # id -> reward EMA in [0, 1] (0.5 = neutral/unproven). Updated by
        # record_outcome() to close the learning loop (Plan 08 Synergy 5).
        self._reward: dict[str, float] = {}
        # Running count of reward records selectively erased (CONCEPT:AU-KG.memory.generation-scoped-selective-reward),
        # for observability/doctor. Transient — not persisted across save/load.
        self._reward_erasures: int = 0
        # id -> ontology type/class (CONCEPT:AU-KG.ontology.optional-populated-from). Optional; populated from the
        # live node's ``type`` when the funnel/bulk loader knows it. Lets
        # ``designate`` re-project the flat cosine neighbourhood through the ontology
        # class structure (the structured-prior analogue of arXiv:2606.09828's
        # depth-guided back-projection) instead of ranking on cosine alone.
        self._id_to_type: dict[str, str] = {}
        # id -> tenant (CONCEPT:AU-P1-3 — policy/tenant filters). ``None``/absent
        # means the entity is tenant-agnostic (visible to every tenant).
        self._id_to_tenant: dict[str, str] = {}
        # id -> set[policy tag] (CONCEPT:AU-P1-3). A candidate is eligible under a
        # ``required_policy_tags`` filter only if it carries EVERY required tag —
        # an untagged entity fails any non-empty policy requirement (fail-closed).
        self._id_to_policy_tags: dict[str, set[str]] = {}

        # HNSW-specific state
        self._hnsw: Any = None
        self._label_to_id: dict[int, str] = {}
        self._id_to_label: dict[str, int] = {}
        self._next_label = 0

        # Bounded-cache LRU (CONCEPT:AU-P1-3): ``None`` disables eviction outright,
        # preserving the historical unbounded behaviour for non-designation callers
        # (OutcomeRouter, ReasonerRouter, ...) that reuse this class as a plain
        # reward-EMA learner over a small, naturally-bounded id space.
        self._bounded_cache_size = (
            int(bounded_cache_size) if bounded_cache_size else None
        )
        self._lru: OrderedDict[str, None] = OrderedDict()

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

    def __contains__(self, id: str) -> bool:
        return id in self._id_to_vec

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
        node_type: str | None = None,
        tenant: str | None = None,
        policy_tags: Any = None,
        reward: float | None = None,
    ) -> None:
        """Add (or replace) an entity in the index.

        Args:
            id: Unique entity identifier.
            embedding: A vector (list/tuple/ndarray) of length ``dim``.
            capabilities: Iterable of capability identifiers the entity
                ``providesCapability``.
            swappable_with: Optional iterable of ids this entity is
                ``swappableWith`` (edges are made symmetric).
            node_type: Optional ontology type/class of the entity (CONCEPT:AU-KG.ontology.optional-populated-from).
                When supplied, ``designate`` uses it to bias ranking toward the
                ontology-coherent neighbourhood; omitting it leaves ranking on pure
                cosine (+ reward), exactly as before.
            tenant: Optional tenant this entity is scoped to (CONCEPT:AU-P1-3). Omitted
                means the entity is visible to every tenant.
            policy_tags: Optional iterable of policy tags this entity satisfies
                (CONCEPT:AU-P1-3) — used by ``designate(required_policy_tags=...)``.
            reward: Optional durable reward EMA to seed on first sight (CONCEPT:AU-P1-3 —
                durable contextual-bandit outcomes). Only applied when ``id`` has no
                in-process reward yet, so it never clobbers a value this process has
                already learned from live outcomes; pass the value hydrated from the
                engine's durable ``capability_reward`` node property (see
                :mod:`.durable_outcome_store`) to survive a process restart.
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
        # CONCEPT:AU-KG.memory.generation-scoped-selective-reward — selective reward erasure on the ingestion upsert path.
        # A re-add whose representation has materially diverged from the stored one is a
        # new generation: the reward EMA accrued under the old content is stale
        # evidence, so erase only that id's record (RQGM selective erasure). The
        # cosine distance is a near-free dot product on two already-normalized
        # vectors, so this runs natively on every ingestion upsert with no flag.
        if is_update and id in self._reward:
            prev_vec = self._id_to_vec[id]
            distance = 1.0 - float(np.dot(prev_vec, norm_vec))
            if distance > _REWARD_REGEN_DISTANCE:
                self._reward.pop(id, None)
                self._reward_erasures += 1
                logger.debug(
                    "[KG-2.276] selective reward erasure for %r "
                    "(embedding drift %.3f > %.2f)",
                    id,
                    distance,
                    _REWARD_REGEN_DISTANCE,
                )
        self._id_to_vec[id] = norm_vec

        # Ontology type for structured-prior ranking (KG-2.44b). Only overwrite when
        # a type is supplied so a typeless update never erases a known type.
        if node_type:
            self._id_to_type[id] = str(node_type)

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

        # Tenant/policy scoping (CONCEPT:AU-P1-3). Only overwrite when a value is
        # supplied so an unscoped update never erases a known tenant/policy set.
        if tenant is not None:
            self._id_to_tenant[id] = str(tenant)
        if policy_tags is not None:
            self._id_to_policy_tags[id] = {str(p) for p in policy_tags}

        # Durable reward hydration (CONCEPT:AU-P1-3): seed from the engine's durably
        # persisted value ONLY when this process has not already learned a reward for
        # ``id`` — a live in-process outcome always wins over a colder durable read.
        if reward is not None and id not in self._reward:
            self._reward[id] = min(1.0, max(0.0, float(reward)))

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

        # Bounded-cache LRU eviction (CONCEPT:AU-P1-3) — touch ``id`` as most-recently
        # used, then evict the oldest entries beyond the cap via :meth:`remove` so
        # every backing structure (HNSW label, capability/tenant/policy/reward maps)
        # stays in lockstep. A no-op unless ``bounded_cache_size`` was set.
        if self._bounded_cache_size:
            self._lru[id] = None
            self._lru.move_to_end(id)
            while len(self._lru) > self._bounded_cache_size:
                oldest, _ = self._lru.popitem(last=False)
                if oldest != id:
                    self.remove(oldest)

    def remove(self, id: str) -> bool:
        """Evict ``id`` from every backing structure (CONCEPT:AU-P1-3 — bounded cache).

        Removes the vector, capability/tenant/policy/type/reward/swappable state, and
        (for the HNSW backend) marks the label deleted so the ANN index never grows
        without bound. Returns ``False`` (no-op) when ``id`` was not resident.
        """
        if id not in self._id_to_vec:
            return False
        del self._id_to_vec[id]
        self._lru.pop(id, None)
        for cap in self._id_to_caps.pop(id, set()):
            self._cap_to_ids.get(cap, set()).discard(id)
        self._id_to_type.pop(id, None)
        self._id_to_tenant.pop(id, None)
        self._id_to_policy_tags.pop(id, None)
        self._reward.pop(id, None)
        partners = self._swappable.pop(id, set())
        for p in partners:
            self._swappable.get(p, set()).discard(id)
        if self._backend == "hnsw":
            label = self._id_to_label.pop(id, None)
            if label is not None:
                self._label_to_id.pop(label, None)
                if self._hnsw is not None:
                    try:
                        self._hnsw.mark_deleted(label)
                    except Exception as e:  # noqa: BLE001 — best-effort; stale label is harmless
                        logger.debug("hnsw mark_deleted failed for %r: %s", id, e)
        return True

    def build_from_edges(self, nodes: Any) -> None:
        """Bulk-load the index from an iterable of node descriptors.

        Each node may be a mapping or an object exposing ``id``, ``embedding``,
        ``capabilities`` (or ``provides``/``providesCapability``), an optional
        ``swappable_with`` (or ``swappableWith``), and optional ``tenant`` /
        ``policy_tags`` (CONCEPT:AU-P1-3).

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
            node_type = (
                getter("type") or getter("node_type") or getter("nodeType") or None
            )
            tenant = getter("tenant")
            policy_tags = getter("policy_tags") or getter("policyTags")
            reward = getter("capability_reward") or getter("reward")
            self.add(
                str(nid),
                emb,
                caps,
                swappable_with=swap,
                node_type=node_type,
                tenant=tenant,
                policy_tags=policy_tags,
                reward=reward,
            )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def _candidate_ids(
        self,
        required_caps: set[str] | None,
        *,
        tenant: str | None = None,
        required_policy_tags: set[str] | None = None,
    ) -> set[str] | None:
        """Return the candidate id set after capability/tenant/policy filtering.

        Returns ``None`` to mean "no restriction" (all ids), or a concrete set
        (possibly empty) when any filter is provided (CONCEPT:AU-P1-3 — policy/tenant
        filters). Every filter is an O(1)-per-id set operation — never a full scan.
        """
        candidate: set[str] | None = None
        if required_caps:
            # Intersection of provider sets — O(sum of set sizes).
            for cap in required_caps:
                providers = self._cap_to_ids.get(cap, set())
                candidate = set(providers) if candidate is None else candidate & providers
                if not candidate:
                    return set()
        if tenant is not None:
            # Fail-open on tenant: an entity with no recorded tenant is global (visible
            # to every tenant); one with a DIFFERENT tenant is excluded.
            tenant_ids = {
                i
                for i in self._id_to_vec
                if self._id_to_tenant.get(i) in (None, tenant)
            }
            candidate = tenant_ids if candidate is None else candidate & tenant_ids
            if not candidate:
                return set()
        if required_policy_tags:
            # Fail-closed on policy: an entity must explicitly carry EVERY required
            # tag — an untagged entity does not satisfy a non-empty requirement.
            policy_ids = {
                i
                for i in self._id_to_vec
                if required_policy_tags <= self._id_to_policy_tags.get(i, set())
            }
            candidate = policy_ids if candidate is None else candidate & policy_ids
            if not candidate:
                return set()
        return candidate

    def designate(
        self,
        prompt_embedding: Any,
        required_caps: Any = None,
        k: int = 5,
        reward_weight: float = 0.15,
        *,
        ontology_prior: Any = None,
        prior_weight: float = 0.15,
        tenant: str | None = None,
        required_policy_tags: Any = None,
    ) -> list[Designation]:
        """Designate the top-``k`` entities for a task.

        Args:
            prompt_embedding: The task/query embedding vector.
            required_caps: Optional iterable of capabilities the entity must
                provide. Candidates are restricted to the set-intersection of
                providers before ranking.
            k: Maximum number of designations to return.
            reward_weight: Blend weight for the learned reward EMA.
            ontology_prior: Optional structured prior (CONCEPT:AU-KG.ontology.optional-populated-from) — a mapping
                ``id -> alignment∈[0,1]`` or a callable ``id -> alignment`` (0.5 =
                neutral). When omitted, an *exact-type-coherence* prior is derived
                automatically from the stored node types (the dominant ontology type
                among the strongest cosine candidates is boosted), so ontology-grounded
                ranking is on by default; pass a richer (e.g. subsumption-aware) prior
                to override it, or ``prior_weight=0`` to fall back to pure cosine.
            prior_weight: Blend weight for the ontology prior; ``0`` disables it.
            tenant: Optional tenant filter (CONCEPT:AU-P1-3 — policy/tenant filters).
                Restricts candidates to entities scoped to this tenant or unscoped
                (tenant-agnostic) entities.
            required_policy_tags: Optional iterable of policy tags every candidate
                must carry (CONCEPT:AU-P1-3). An entity with no policy tags fails any
                non-empty requirement (fail-closed).

        Returns:
            Up to ``k`` :class:`Designation` objects sorted by descending
            similarity score.
        """
        if k <= 0 or not self._id_to_vec:
            return []

        query = np.asarray(prompt_embedding, dtype=np.float32).reshape(-1)
        if self._dim is not None and query.size != self._dim:
            raise ValueError(
                f"Prompt embedding dim mismatch: expected {self._dim}, got {query.size}"
            )
        query = _l2_normalize(query)

        req = {str(c) for c in required_caps} if required_caps else None
        req_policy = (
            {str(p) for p in required_policy_tags} if required_policy_tags else None
        )
        candidates = self._candidate_ids(
            req, tenant=tenant, required_policy_tags=req_policy
        )
        if candidates is not None and not candidates:
            return []

        # Oversample by pure similarity (keeps the ANN fast path intact), then blend
        # in two structured boosts so proven / ontology-coherent designations rise to
        # the top: the learned reward EMA (Plan 08 Synergy 5) and the ontology-type
        # prior (KG-2.44b). Either weight at 0 (or no signal present) disables its
        # term, and with both off the ranking is pure cosine — exact prior parity.
        use_reward = bool(reward_weight) and bool(self._reward)
        oversample_size = (
            len(candidates) if candidates is not None else len(self._id_to_vec)
        )
        oversample = min(oversample_size, max(k, k * 4))
        prior_fn = (
            self._resolve_ontology_prior(ontology_prior, query, candidates, oversample)
            if prior_weight
            else None
        )
        if use_reward or prior_fn is not None:
            ranked = self._rank(query, candidates, oversample)
            ranked = sorted(
                ranked,
                key=lambda t: (
                    t[1]
                    + (
                        reward_weight * (self._reward.get(t[0], 0.5) - 0.5)
                        if use_reward
                        else 0.0
                    )
                    + (
                        prior_weight * (prior_fn(t[0]) - 0.5)
                        if prior_fn is not None
                        else 0.0
                    )
                ),
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
            if prior_fn is not None:
                provenance["ontology_type"] = self._id_to_type.get(nid)
                provenance["ontology_prior"] = round(prior_fn(nid), 4)
            alts = self.alternatives(nid)
            if alts:
                provenance["alternatives"] = alts
            # CONCEPT:AU-P1-3 — explainable routing: why this candidate was eligible.
            if req or tenant is not None or req_policy:
                provenance["eligibility"] = self.explain(
                    nid, required_caps=req, tenant=tenant, required_policy_tags=req_policy
                )
            results.append(
                Designation(
                    id=nid,
                    score=float(score),
                    capabilities=caps,
                    provenance=provenance,
                )
            )
        return results

    def _resolve_ontology_prior(
        self,
        ontology_prior: Any,
        query: NDArray,
        candidates: set[str] | None,
        oversample: int,
    ) -> Any:
        """Resolve the ontology prior into a callable ``id -> alignment∈[0,1]``.

        CONCEPT:AU-KG.ontology.optional-populated-from. An explicit ``ontology_prior`` (mapping or callable) is
        honoured as-is — that is how a caller injects a richer, subsumption-aware
        prior. When none is given, derive the default *exact-type-coherence* prior:
        rank the candidate pool by cosine, take the dominant ontology type among the
        strongest hits, and boost every candidate sharing it. Returns ``None`` when no
        structured signal exists (no stored types and no explicit prior) so ranking
        stays pure cosine — exact parity with the pre-KG-2.44b behaviour.
        """
        if ontology_prior is not None:
            if callable(ontology_prior):
                return lambda nid: float(ontology_prior(nid))
            return lambda nid: float(ontology_prior.get(nid, 0.5))
        if not self._id_to_type:
            return None
        # Dominant type among the strongest cosine candidates = the coherent
        # neighbourhood the flat ranking is re-projected toward.
        ranked = self._rank(query, candidates, oversample)
        types = [self._id_to_type.get(nid) for nid, _ in ranked]
        typed = [t for t in types if t]
        if not typed:
            return None
        from collections import Counter

        modal_type = Counter(typed).most_common(1)[0][0]
        return lambda nid: 1.0 if self._id_to_type.get(nid) == modal_type else 0.5

    def _rank(
        self, query: NDArray, candidates: set[str] | None, k: int
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
        # Candidates are a hash-ordered set → sort them and use a STABLE
        # argsort so exact-tie scores rank deterministically (lexicographic),
        # identically before and after a save/load round-trip.
        ids = (
            sorted(i for i in candidates if i in self._id_to_vec)
            if candidates is not None
            else list(self._id_to_vec.keys())
        )
        if not ids:
            return []
        matrix = np.stack([self._id_to_vec[i] for i in ids])  # already normalized
        sims = matrix @ query  # cosine sim since both sides L2-normalized
        order = np.argsort(-sims, kind="stable")[:k]
        return [(ids[int(j)], float(sims[int(j)])) for j in order]

    def alternatives(self, id: str) -> list[str]:
        """Return ids ``swappableWith`` the given entity (sorted, stable)."""
        return sorted(self._swappable.get(id, set()))

    # ------------------------------------------------------------------
    # Explainable routing (CONCEPT:AU-P1-3 — explainable-features output)
    # ------------------------------------------------------------------
    def explain(
        self,
        id: str,
        required_caps: Any = None,
        *,
        tenant: str | None = None,
        required_policy_tags: Any = None,
    ) -> dict[str, Any]:
        """Return WHY ``id`` was (or would be) eligible for a filtered designation.

        A pure lookup — never ranks or embeds — so a caller/test can ask "why was
        this candidate eligible" independent of a ``designate()`` call. Covers every
        candidate-selection gate this class enforces: capability coverage, tenant
        scoping, and policy-tag coverage. ``eligible`` is the AND of every gate that
        was actually asked for (an omitted filter never disqualifies).
        """
        caps = set(self._id_to_caps.get(id, set()))
        req = {str(c) for c in required_caps} if required_caps else set()
        missing_caps = sorted(req - caps)

        tenant_of = self._id_to_tenant.get(id)
        tenant_match = None if tenant is None else tenant_of in (None, tenant)

        policy_of = set(self._id_to_policy_tags.get(id, set()))
        req_policy = (
            {str(p) for p in required_policy_tags} if required_policy_tags else set()
        )
        missing_policy = sorted(req_policy - policy_of)

        eligible = (
            not missing_caps
            and tenant_match is not False
            and not missing_policy
        )
        return {
            "id": id,
            "capabilities": sorted(caps),
            "required_caps": sorted(req),
            "missing_caps": missing_caps,
            "capabilities_matched": not missing_caps,
            "tenant": tenant_of,
            "required_tenant": tenant,
            "tenant_match": tenant_match,
            "policy_tags": sorted(policy_of),
            "required_policy_tags": sorted(req_policy),
            "missing_policy_tags": missing_policy,
            "policy_matched": not missing_policy,
            "reward": round(self.reward_of(id), 4),
            "ontology_type": self._id_to_type.get(id),
            "eligible": eligible,
        }

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

    def selective_erase_rewards(self, ids: Any) -> int:
        """Selectively erase the reward EMA for exactly ``ids`` (CONCEPT:AU-KG.memory.generation-scoped-selective-reward).

        The explicit, provenance-scoped form of the Red Queen Gödel Machine's
        *selective erasure* (arXiv:2606.26294): when the source/evaluator/impl
        that produced a set of designations is superseded (a capability is
        redeployed, a model regime changes, a document version is retracted), the
        utility records scored under it are no longer valid. This erases only
        those records — every reward not in ``ids`` is preserved — so the router
        re-learns the affected entities from the neutral prior instead of being
        anchored by stale evidence. Order-independent: erasing ``{a, b}`` then
        ``{c}`` is identical to erasing ``{a, b, c}`` at once. Unlike
        :meth:`decay_rewards` (uniform time decay toward neutral), this is
        targeted by *provenance*, not by *age*.

        Returns the number of records actually erased.
        """
        erased = 0
        for raw in ids or ():
            id = str(raw)
            if self._reward.pop(id, None) is not None:
                erased += 1
        self._reward_erasures += erased
        return erased

    @property
    def reward_erasures(self) -> int:
        """Total reward records selectively erased this process (KG-2.276)."""
        return self._reward_erasures

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
            "id_to_type": dict(self._id_to_type),
            "id_to_tenant": dict(self._id_to_tenant),
            "id_to_policy_tags": {
                i: sorted(p) for i, p in self._id_to_policy_tags.items()
            },
            "bounded_cache_size": self._bounded_cache_size,
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
            bounded_cache_size=meta.get("bounded_cache_size"),
        )
        idx._cap_to_ids = {c: set(ids) for c, ids in meta["cap_to_ids"].items()}
        idx._id_to_caps = {i: set(c) for i, c in meta["id_to_caps"].items()}
        idx._swappable = {i: set(s) for i, s in meta["swappable"].items()}
        idx._reward = dict(meta.get("reward", {}))
        idx._id_to_type = dict(meta.get("id_to_type", {}))
        idx._id_to_tenant = dict(meta.get("id_to_tenant", {}))
        idx._id_to_policy_tags = {
            i: set(p) for i, p in meta.get("id_to_policy_tags", {}).items()
        }
        idx._id_to_label = dict(meta["id_to_label"])
        idx._label_to_id = {v: k for k, v in idx._id_to_label.items()}
        idx._next_label = meta["next_label"]

        ids = meta["ids"]
        arr = np.load(path / "embeddings.npy")
        for i, nid in enumerate(ids):
            idx._id_to_vec[nid] = np.asarray(arr[i], dtype=np.float32)
            if idx._bounded_cache_size:
                idx._lru[nid] = None

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
