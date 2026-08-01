"""Semantic response caching — similarity-scoped, freshness-bounded, opt-in only.

CONCEPT:AU-KG.memory.semantic-response-cache — genuinely new capability (the audit that motivated
this module found no ``SemanticCache``/``ResponseCache``/``QueryCache`` class anywhere in this
repo). Unlike the exact prompt cache (``agent_utilities.caching.prompt_cache``, which adopts
provider-native byte-prefix caching and is default-on because it can never change WHAT a call
returns), this module decides whether to skip a live model call entirely and return a PRIOR
response for a semantically-similar-but-not-identical request. That is a correctness trade, not
just a performance one, so every default here is the conservative one:

* **Default-OFF.** :attr:`SemanticCachePolicy.enabled` defaults ``False`` AND the process-wide
  ``AU_SEMANTIC_CACHE`` env switch (also default ``False``) must independently agree — a caller
  opting in locally still can't serve cached answers on a deployment that hasn't turned the
  capability on. Both must be true.
* **Side-effect-free only.** :attr:`SemanticCachePolicy.side_effect_free` must be explicitly
  ``True`` — there is no way to "forget" to declare it, because the default is the refusal.
* **Freshness-bounded.** :attr:`SemanticCachePolicy.freshness_tolerance_seconds` must be an
  explicit non-negative int. ``None`` (the default) means "the caller never said how stale an
  answer may be", which is treated identically to "never" — refuse to serve from cache.
* **Fails to a live call, never to a stale answer.** Every refusal path in :meth:`SemanticCache.lookup`
  returns ``hit=False`` with a specific ``reason`` — it never raises and never invents a response;
  the caller always falls through to its own live call on anything but an explicit hit.

Cache-key discipline (the security boundary)
----------------------------------------------
:class:`SemanticCacheKey` carries every component the constitution requires: tenant, principal,
policy_version, model_identity, prompt_version, tool_schema_version, retrieval_snapshot,
ontology_version, safety_posture. :attr:`SemanticCacheKey.fingerprint` hashes all nine — mirrors
:class:`~agent_utilities.kvcache.checkpoint.KVCheckpointKey`'s discipline exactly (see that
module's docstring for why this is a security boundary, not a cache-key nicety). A bucket is the
fingerprint; **similarity search only ever happens WITHIN one bucket** — two requests that differ
in even one key component land in different buckets and can never see each other's entries, by
construction, independent of any similarity score. :meth:`SemanticCache.lookup`/:meth:`store`
never search or write across buckets — there is no code path that could.

No heavy dependency
--------------------
Embedding is fully pluggable via the ``embed_fn`` constructor argument. When omitted, one is
resolved LAZILY (only when a caller actually enables the cache) from the embedding stack this repo
already ships as an optional extra (``agent_utilities.core.embedding_utilities.create_embedding_model``,
already used by retrieval/enrichment) — never imported at module load, never a new dependency.
Similarity is plain-Python cosine (no numpy at runtime — see ``pyproject.toml``'s
"numpy is a DEV/TEST-ONLY dependency" note; this module holds to that).
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
import time
from collections import OrderedDict
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_utilities.core.config import setting

logger = logging.getLogger(__name__)

__all__ = [
    "SemanticCache",
    "SemanticCacheKey",
    "SemanticCacheLookup",
    "SemanticCachePolicy",
    "get_semantic_cache",
    "resolve_semantic_cache_key",
]

_KEY_SEPARATOR = "\x1f"
_DEFAULT_MAX_BUCKETS = 512
_DEFAULT_MAX_ENTRIES_PER_BUCKET = 16
_DEFAULT_SIMILARITY_THRESHOLD = 0.95


class SemanticCacheKey(BaseModel):
    """Every component that scopes a semantic-cache bucket (CONCEPT:AU-KG.memory.semantic-response-cache).

    A SECURITY boundary exactly like :class:`~agent_utilities.kvcache.checkpoint.KVCheckpointKey`:
    :attr:`fingerprint` hashes all nine fields, so changing ANY one of them — a grant, a policy
    revision, an ontology/catalog-epoch bump, a model swap, a tool-contract revision — mints a
    different bucket. There is no cross-bucket reuse path.
    """

    tenant: str = Field(
        min_length=1, description="Owning tenant id — mandatory, never defaulted."
    )
    principal: str = Field(default="", description="Requesting principal/actor id.")
    policy_version: str = Field(
        default="", description="Authorization/policy revision."
    )
    model_identity: str = Field(
        default="", description="provider:model_id this response came from."
    )
    prompt_version: str = Field(
        default="", description="Prompt/template content-version fingerprint."
    )
    tool_schema_version: str = Field(
        default="", description="Bound tool/schema contract version."
    )
    retrieval_snapshot: str = Field(
        default="",
        description="Retrieval/evidence-set fingerprint (e.g. ContextBundle.cache_key).",
    )
    ontology_version: str = Field(
        default="",
        description="KG schema/catalog fingerprint (GraphSession.catalog_epoch).",
    )
    safety_posture: str = Field(
        default="", description="Active tool-guard mode (TOOL_GUARD_MODE)."
    )

    model_config = ConfigDict(frozen=True, extra="forbid")

    @property
    def fingerprint(self) -> str:
        """Deterministic sha256 over every component — the bucket id."""
        canonical = _KEY_SEPARATOR.join(
            (
                self.tenant,
                self.principal,
                self.policy_version,
                self.model_identity,
                self.prompt_version,
                self.tool_schema_version,
                self.retrieval_snapshot,
                self.ontology_version,
                self.safety_posture,
            )
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class SemanticCachePolicy(BaseModel):
    """The explicit, per-request/per-policy opt-in (CONCEPT:AU-KG.memory.semantic-response-cache).

    Every field defaults to the REFUSAL value. A caller must positively set ``enabled``,
    ``side_effect_free``, AND a non-negative ``freshness_tolerance_seconds`` for a lookup to ever
    be eligible for a hit.
    """

    enabled: bool = False
    side_effect_free: bool = False
    freshness_tolerance_seconds: int | None = None
    similarity_threshold: float = Field(
        default=_DEFAULT_SIMILARITY_THRESHOLD, ge=0.0, le=1.0
    )

    model_config = ConfigDict(extra="forbid")


class SemanticCacheLookup(BaseModel):
    """Outcome of one :meth:`SemanticCache.lookup` — ALWAYS returned, hit or miss.

    Never indistinguishable from a fresh call: ``hit=True`` always carries ``age_seconds`` and
    ``similarity`` alongside the response, and every outcome (hit or refusal) carries the full
    :class:`SemanticCacheKey` it was scoped by, for :meth:`~agent_utilities.harness.trace_backend.
    KGTraceBackend.record_cache_decision` to attach to the trace.
    """

    key: SemanticCacheKey
    outcome: str  # hit | miss | disabled | refused_side_effect | refused_freshness | stale | below_threshold | embed_unavailable
    hit: bool = False
    response_text: str | None = None
    similarity: float | None = None
    age_seconds: float | None = None

    model_config = ConfigDict(extra="forbid")


def _safety_posture() -> str:
    try:
        from agent_utilities.core.config import TOOL_GUARD_MODE

        return str(TOOL_GUARD_MODE or "")
    except Exception:  # noqa: BLE001 — best-effort context, never break the caller
        return ""


def _ambient_context() -> tuple[str, str, str, str]:
    """Best-effort ``(tenant, principal, policy_version, ontology_version)`` from the ambient
    :class:`~agent_utilities.knowledge_graph.core.session.GraphSession`. Empty strings (never
    raises) when no verified session is bound."""
    try:
        from agent_utilities.knowledge_graph.core.session import GraphSession

        session = GraphSession.from_ambient()
    except Exception:  # noqa: BLE001 — no ambient session is the common case
        return "", "", "", ""
    principal = str(getattr(session.actor, "actor_id", "") or "")
    ontology_version = (
        str(session.catalog_epoch) if session.catalog_epoch is not None else ""
    )
    return (
        str(session.tenant or ""),
        principal,
        str(session.policy_version or ""),
        ontology_version,
    )


def resolve_semantic_cache_key(
    *,
    tenant: str | None = None,
    principal: str | None = None,
    policy_version: str | int | None = None,
    model_identity: str = "",
    prompt_version: str = "",
    tool_schema_version: str = "",
    retrieval_snapshot: str = "",
    ontology_version: str | None = None,
    safety_posture: str | None = None,
) -> SemanticCacheKey:
    """Build a :class:`SemanticCacheKey`, filling unset scoping fields from the ambient
    :class:`~agent_utilities.knowledge_graph.core.session.GraphSession` when one is bound.

    Raises ``ValueError`` when no tenant can be resolved (explicit or ambient) — ``tenant`` is
    mandatory on :class:`SemanticCacheKey` (a security boundary, never defaulted; see its
    docstring). Callers that want caching to simply be skipped when no tenant is resolvable
    (e.g. a best-effort integration point) should catch this and fall through to a live call.
    """
    amb_tenant, amb_principal, amb_policy, amb_ontology = _ambient_context()
    resolved_tenant = tenant if tenant is not None else amb_tenant
    if not resolved_tenant:
        raise ValueError(
            "resolve_semantic_cache_key: no tenant resolvable (explicit or ambient) — "
            "SemanticCacheKey.tenant is mandatory"
        )
    return SemanticCacheKey(
        tenant=resolved_tenant,
        principal=principal if principal is not None else amb_principal,
        policy_version=str(policy_version)
        if policy_version is not None
        else amb_policy,
        model_identity=model_identity,
        prompt_version=prompt_version,
        tool_schema_version=tool_schema_version,
        retrieval_snapshot=retrieval_snapshot,
        ontology_version=ontology_version
        if ontology_version is not None
        else amb_ontology,
        safety_posture=safety_posture
        if safety_posture is not None
        else _safety_posture(),
    )


class _Entry:
    __slots__ = (
        "embedding",
        "response_text",
        "created_at",
        "tolerance_seconds",
        "priority_rank",
    )

    def __init__(
        self,
        *,
        embedding: list[float],
        response_text: str,
        created_at: float,
        tolerance_seconds: int,
        priority_rank: int,
    ) -> None:
        self.embedding = embedding
        self.response_text = response_text
        self.created_at = created_at
        self.tolerance_seconds = tolerance_seconds
        self.priority_rank = priority_rank


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Plain-Python cosine similarity — no numpy at runtime (see module docstring)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _record_op(op: str, outcome: str) -> None:
    try:
        from agent_utilities.observability.gateway_metrics import SEMANTIC_CACHE_OPS

        SEMANTIC_CACHE_OPS.labels(op=op, outcome=outcome).inc()
    except Exception as exc:  # noqa: BLE001 — metrics must never break the caller
        logger.debug("semantic-cache telemetry recording failed: %s", exc)


def _record_trace(*, lookup: SemanticCacheLookup) -> None:
    """Best-effort trace attachment (CONCEPT:AU-KG.memory.semantic-response-cache — "every cache hit
    is visible in the trace with its key components and age"). Mirrors
    ``core/model_factory.py``'s ``_record_routing_decision`` best-effort-sink pattern."""
    try:
        from agent_utilities.harness.tracing import get_kg_trace_sink, get_trace_id

        sink = get_kg_trace_sink()
        trace_id = get_trace_id()
        if sink is not None and trace_id and hasattr(sink, "record_cache_decision"):
            sink.record_cache_decision(trace_id=trace_id, decision=lookup)
    except Exception as exc:  # noqa: BLE001 — trace attachment is best-effort
        logger.debug("semantic-cache trace attachment failed: %s", exc)


def _default_embed_fn(text: str) -> list[float] | None:
    """Lazily resolve an embedder from the existing (optional) embedding stack.

    Returns ``None`` (never raises) when the optional embedding dependency/endpoint is
    unavailable — the caller treats that as ``embed_unavailable`` and falls back to a live call.
    """
    try:
        from agent_utilities.core.embedding_utilities import create_embedding_model

        model = create_embedding_model()
        vector = model.get_text_embedding(text)
        return [float(v) for v in vector]
    except Exception as exc:  # noqa: BLE001 — embedding is a soft dependency here
        logger.debug("semantic-cache default embedder unavailable: %s", exc)
        return None


class SemanticCache:
    """Similarity-scoped, freshness-bounded, opt-in-only response cache
    (CONCEPT:AU-KG.memory.semantic-response-cache). See module docstring for the full safety
    contract. Thread-safe (a single lock guards the bucket store)."""

    def __init__(
        self,
        *,
        embed_fn: Any = None,
        max_buckets: int = _DEFAULT_MAX_BUCKETS,
        max_entries_per_bucket: int = _DEFAULT_MAX_ENTRIES_PER_BUCKET,
    ) -> None:
        self._embed_fn = embed_fn or _default_embed_fn
        self._max_buckets = max_buckets
        self._max_entries_per_bucket = max_entries_per_bucket
        # Keyed by fingerprint; each bucket keeps the ORIGINAL SemanticCacheKey alongside its
        # entries (not just the hash) so component-filtered invalidate() can actually match —
        # the fingerprint alone is one-way and can't be un-hashed back into tenant/policy/etc.
        self._buckets: OrderedDict[str, tuple[SemanticCacheKey, list[_Entry]]] = (
            OrderedDict()
        )
        self._lock = threading.Lock()

    @staticmethod
    def _master_switch_enabled() -> bool:
        return bool(setting("AU_SEMANTIC_CACHE", False))

    def _embed(self, text: str) -> list[float] | None:
        try:
            vector = self._embed_fn(text)
        except Exception as exc:  # noqa: BLE001 — embedding must never break a lookup/store
            logger.debug("semantic-cache embed_fn failed: %s", exc)
            return None
        if not vector:
            return None
        return list(vector)

    def lookup(
        self, key: SemanticCacheKey, query_text: str, *, policy: SemanticCachePolicy
    ) -> SemanticCacheLookup:
        """Look up a semantically-similar prior response, or refuse — NEVER raises.

        Fail-closed on every axis (see module docstring): disabled policy/master-switch, a
        request that isn't declared side-effect-free, no freshness tolerance, an unavailable
        embedder, a stale best match, or a best match below ``policy.similarity_threshold`` all
        return ``hit=False`` with a distinct ``reason`` — the caller always has a clean signal to
        fall through to its own live call.
        """
        outcome = self._lookup_inner(key, query_text, policy=policy)
        _record_op("lookup", outcome.outcome)
        if outcome.outcome == "hit":
            try:
                from agent_utilities.observability.gateway_metrics import (
                    SEMANTIC_CACHE_STALENESS_SECONDS,
                )

                SEMANTIC_CACHE_STALENESS_SECONDS.observe(outcome.age_seconds or 0.0)
            except Exception as exc:  # noqa: BLE001 — metrics must never break the caller
                logger.debug("semantic-cache staleness metric failed: %s", exc)
        _record_trace(lookup=outcome)
        return outcome

    def _lookup_inner(
        self, key: SemanticCacheKey, query_text: str, *, policy: SemanticCachePolicy
    ) -> SemanticCacheLookup:
        if not policy.enabled or not self._master_switch_enabled():
            return SemanticCacheLookup(key=key, outcome="disabled")
        if not policy.side_effect_free:
            return SemanticCacheLookup(key=key, outcome="refused_side_effect")
        if (
            policy.freshness_tolerance_seconds is None
            or policy.freshness_tolerance_seconds < 0
        ):
            return SemanticCacheLookup(key=key, outcome="refused_freshness")

        embedding = self._embed(query_text)
        if embedding is None:
            return SemanticCacheLookup(key=key, outcome="embed_unavailable")

        now = time.time()
        with self._lock:
            existing = self._buckets.get(key.fingerprint)
            bucket = existing[1] if existing is not None else []

        best: _Entry | None = None
        best_similarity = 0.0
        best_stale = False
        for entry in bucket:
            age = now - entry.created_at
            tolerance = min(policy.freshness_tolerance_seconds, entry.tolerance_seconds)
            if age > tolerance:
                best_stale = True
                continue
            similarity = _cosine_similarity(embedding, entry.embedding)
            if similarity > best_similarity:
                best, best_similarity = entry, similarity

        if best is None:
            return SemanticCacheLookup(
                key=key, outcome="stale" if best_stale else "miss"
            )
        if best_similarity < policy.similarity_threshold:
            return SemanticCacheLookup(
                key=key, outcome="below_threshold", similarity=best_similarity
            )

        return SemanticCacheLookup(
            key=key,
            outcome="hit",
            hit=True,
            response_text=best.response_text,
            similarity=best_similarity,
            age_seconds=now - best.created_at,
        )

    def store(
        self,
        key: SemanticCacheKey,
        query_text: str,
        response_text: str,
        *,
        policy: SemanticCachePolicy,
    ) -> bool:
        """Store ``response_text`` for ``query_text`` under ``key``'s bucket. Refuses (returns
        ``False``, never raises) under the exact same conditions as :meth:`lookup`."""
        if not policy.enabled or not self._master_switch_enabled():
            _record_op("store", "disabled")
            return False
        if not policy.side_effect_free:
            _record_op("store", "refused_side_effect")
            return False
        if (
            policy.freshness_tolerance_seconds is None
            or policy.freshness_tolerance_seconds < 0
        ):
            _record_op("store", "refused_freshness")
            return False

        embedding = self._embed(query_text)
        if embedding is None:
            _record_op("store", "embed_unavailable")
            return False

        from agent_utilities.core.resource_priority import (
            PriorityClass,
            current_priority,
        )

        priority = current_priority() or PriorityClass.ORCHESTRATION
        entry = _Entry(
            embedding=embedding,
            response_text=response_text,
            created_at=time.time(),
            tolerance_seconds=policy.freshness_tolerance_seconds,
            priority_rank=priority.rank,
        )
        with self._lock:
            if key.fingerprint not in self._buckets:
                self._evict_bucket_if_needed_locked()
                self._buckets[key.fingerprint] = (key, [])
            self._buckets.move_to_end(key.fingerprint)
            bucket = self._buckets[key.fingerprint][1]
            bucket.append(entry)
            if len(bucket) > self._max_entries_per_bucket:
                self._evict_entry_locked(bucket)
        _record_op("store", "stored")
        return True

    def _evict_bucket_if_needed_locked(self) -> None:
        while len(self._buckets) >= self._max_buckets and self._buckets:
            evicted_fp, _ = self._buckets.popitem(last=False)
            logger.debug(
                "semantic-cache evicted bucket %s (FIFO bound)", evicted_fp[:12]
            )
            _record_op("evict", "bucket_fifo")

    def _evict_entry_locked(self, bucket: list[Any]) -> None:
        # Lowest-priority (highest rank, mirrors kvcache.policy's "background never wins
        # contention") first, then oldest — same precedence the KV-cache-layering policy already
        # applies for a shared resource under PriorityClass.
        victim_idx = max(
            range(len(bucket)),
            key=lambda i: (bucket[i].priority_rank, -bucket[i].created_at),
        )
        bucket.pop(victim_idx)
        _record_op("evict", "entry_bound")

    def invalidate(
        self,
        *,
        tenant: str | None = None,
        principal: str | None = None,
        policy_version: str | None = None,
        model_identity: str | None = None,
        ontology_version: str | None = None,
        tool_schema_version: str | None = None,
    ) -> int:
        """Explicit invalidation (CONCEPT:AU-KG.memory.semantic-response-cache — "explicit
        invalidation on grant, policy, ontology, model, or tool-contract revision").

        A change to any key component ALREADY mints a different bucket fingerprint (implicit
        invalidation — see the class docstring), so this is the operator-facing escape hatch for
        purging entries a caller may still hold a stale fingerprint reference to (e.g. bulk-clear
        after a security incident). Filters by exact-match on every given non-``None`` argument
        against each bucket's ORIGINAL :class:`SemanticCacheKey` (kept alongside its entries for
        exactly this); an omitted argument matches anything. No filters at all clears the whole
        cache. Returns the number of buckets dropped.
        """
        filters = {
            "tenant": tenant,
            "principal": principal,
            "policy_version": policy_version,
            "model_identity": model_identity,
            "ontology_version": ontology_version,
            "tool_schema_version": tool_schema_version,
        }
        active_filters = {k: v for k, v in filters.items() if v is not None}
        dropped = 0
        with self._lock:
            for fp in list(self._buckets.keys()):
                bucket_key, _ = self._buckets[fp]
                if all(
                    getattr(bucket_key, field) == value
                    for field, value in active_filters.items()
                ):
                    del self._buckets[fp]
                    dropped += 1
        if dropped:
            _record_op("invalidate", "cleared")
        return dropped

    def invalidate_key(self, key: SemanticCacheKey) -> bool:
        """Drop exactly the bucket for ``key`` (its deterministic fingerprint). Returns whether a
        bucket existed."""
        with self._lock:
            existed = self._buckets.pop(key.fingerprint, None) is not None
        if existed:
            _record_op("invalidate", "cleared")
        return existed

    def stats(self) -> dict[str, int]:
        """Bounded introspection for tests/telemetry: bucket + entry counts."""
        with self._lock:
            return {
                "buckets": len(self._buckets),
                "entries": sum(len(entries) for _, entries in self._buckets.values()),
            }


_default_cache: SemanticCache | None = None
_default_cache_lock = threading.Lock()


def get_semantic_cache() -> SemanticCache:
    """The process-wide default :class:`SemanticCache` singleton (lazy, thread-safe)."""
    global _default_cache
    if _default_cache is None:
        with _default_cache_lock:
            if _default_cache is None:
                _default_cache = SemanticCache()
    return _default_cache
