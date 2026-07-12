#!/usr/bin/python
from __future__ import annotations

"""``ContextCompiler`` — policy-aware LLM context assembly (CONCEPT:AU-KG.retrieval.context-compiler, Codex X-7).

**The gap this closes.** Every caller that needs to hand an LLM "the relevant
context" for a query today does the same ad-hoc thing: run a retrieval call,
then flatten the hits into a flat text block sorted by raw similarity score
(see e.g. ``mcp/tools/query_tools.py``'s ``graph_search`` formatter, or
``knowledge_graph/core/context_builder.py``'s single-node string concatenation).
That path optimizes for exactly one axis — relevance — and drops everything
else on the floor: near-duplicate hits crowd out coverage, low-confidence or
stale claims ride alongside well-evidenced ones with no signal, nothing checks
whether the requesting actor may even see a hit, nothing caps the result to a
token budget, and nothing tells the caller *why* an item was included or
dropped.

``ContextCompiler`` replaces that pattern with one selection/assembly layer
that optimizes six axes and returns a fully-cited, provenance-bearing bundle:

1. **Relevance** — the engine's own retrieval score (native ANN / hybrid
   retriever — CONCEPT:AU-KG.compute.kg-2), min-max normalized across the
   candidate pool so it composes fairly with the other axes.
2. **Diversity** — greedy MMR (Maximal Marginal Relevance) over embedding
   cosine similarity (falling back to lexical Jaccard overlap when no
   embedding is present) so near-duplicate hits don't crowd out coverage.
3. **Evidence quality** — reads the KnowledgeBatch-shaped epistemic columns
   (CONCEPT:EPI-P3-1: ``confidence``, ``source_refs``, ``evidence_refs``,
   ``proof_ids``, ``contradiction_ids``, ``policy_labels``) when a result
   carries them, rewarding well-sourced/proven claims and penalizing
   contested ones (the ``"epistemic:contested"`` policy label EPI-P3-1's
   ``KnowledgeBatch`` documents), neutral (the same Bayesian prior
   ``retrieval_quality.TRUST_PRIOR`` uses) when absent.
4. **Freshness** — bi-temporal recency decay against ``event_time`` /
   ``valid_from`` / ``timestamp`` / ``created_at`` (same half-life-decay shape
   ``HybridRetriever._recency_boost`` uses), neutral for undated results.
5. **Token cost** — the existing ``RetrievalBudgetManager`` (CONCEPT:AU-KG.memory.tiered-memory-caching)
   greedily fits the MMR-ranked list to the caller's token budget; nothing is
   silently truncated — every drop is logged.
6. **Policy** — every candidate is passed through the SAME fine-grained
   permissioning gate the live read path uses,
   :func:`~agent_utilities.knowledge_graph.ontology.permissioning.enforce`
   (row-level drop for markings/ACLs the actor lacks, column-level redaction
   otherwise) — no new permission system, no bypass.

The result is a :class:`ContextBundle`: the selected :class:`ContextItem`\\ s
(each carrying its per-axis scores), a flat ``citations`` list (evidence
refs/spans a generation step can cite), a ``proof_graph`` of
supports/contradicts/alternative-to edges (from the candidates' own
``proof_ids``/``contradiction_ids``/``alternative_ids`` columns, augmented —
when the engine exposes it — by
:meth:`~agent_utilities.knowledge_graph.orchestration.engine_query.QueryMixin.retrieve_epistemic_view`'s
belief/support/contradiction traversal), and a ``decisions`` log recording
every selection/rejection with its scores — the observable, benchmarkable
half of the contract: same candidates + same session ⇒ same bundle, and the
``decisions`` log is exactly what a benchmark harness or an audit needs to
diff two runs.

This is a SELECTION/ASSEMBLY/OPTIMIZATION layer. It does not retrieve (it
calls the engine's ``search_hybrid`` / a ``HybridRetriever`` — CONCEPT:AU-KG.compute.kg-2)
and it does not gate permissions itself (it calls ``permissioning.enforce`` —
CONCEPT:AU-KG.ontology.redact-object-materialize-restricted). It composes both, plus the epistemic columns, into
one benchmarkable pass.

**Seam 6 (CONCEPT:AU-KG.retrieval.context-compiler-kv-seam)** — the compiled bundle can
optionally be routed through the SAME shared, content-addressed KV-cache layer the
engine's ``/kv`` HTTP surface exposes for LMCache/vLLM token-block reuse
(``agent_utilities.kvcache.EpistemicGraphKVBackend`` — CONCEPT:AU-KG.backend.kvcache-vllm-connector,
also driven by the ``graph_kvcache`` MCP tool). Passing ``kv_backend=`` to
:meth:`ContextCompiler.compile` (any duck-typed object exposing
``get(key) -> bytes | None`` / ``put(key, bytes) -> bool``, exactly the shape
``EpistemicGraphKVBackend`` implements) makes :meth:`compile` compute a stable
cache key from the bundle's *evidence identity* — the sorted, policy-filtered
candidate ids ("evidence ids") the bundle would be assembled from, plus the
session's ``policy_version`` and the caller's ``token_budget`` (folding in the
remaining assembly parameters — ``top_k``, ``diversity_lambda``, ``weights``,
``freshness_half_life_days``, ``as_of``, ``mask_redactions`` — so no two
inputs that could produce a *different* bundle ever collide) — and, on a hit,
returns the previously-assembled bundle instead of re-running MMR diversity
scoring, evidence/freshness scoring, budget-fitting, and proof-graph
construction. On a miss it assembles as normal and stores the result under
that key for the next caller. This is opt-in (``kv_backend=None`` — the
default — leaves :meth:`compile` byte-for-byte unchanged) and reuses the
EXISTING KV surface as a generic content-addressed store; it does not
reimplement KV caching, and it is a distinct namespace from the raw
token-block bytes LMCache stores under the same backend (see the module
docstring note below on how the two relate).
"""

import hashlib
import json
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..core.engine import cosine_similarity
from ..core.session import GraphSession
from ..ontology.permissioning import enforce
from .budget import RetrievalBudgetManager
from .hybrid_retriever import _parse_instant

logger = logging.getLogger(__name__)

__all__ = [
    "Citation",
    "ProofEdge",
    "ContextItem",
    "ContextBundle",
    "ContextCompiler",
    "compute_bundle_cache_key",
    "DEFAULT_BUNDLE_SYSTEM_PREAMBLE",
]

# Neutral evidence-quality prior — matches retrieval_quality.TRUST_PRIOR so an
# unscored claim is treated exactly like an unscored memory node elsewhere.
_NEUTRAL_CONFIDENCE = 0.5
# Neutral freshness for an undated result — no penalty, no bonus.
_NEUTRAL_FRESHNESS = 0.5
# The epistemic policy label EPI-P3-1's KnowledgeBatch column layout documents
# for a contested claim (see eg-plan/src/knowledge_batch.rs module docs).
_CONTESTED_LABEL = "epistemic:contested"

# Seam 6 serving-layer wire (CONCEPT:AU-KG.retrieval.context-compiler-kv-seam) — the fixed literal
# that opens every rendered prompt, BEFORE the bundle's own text. It is a constant
# string (never templated with per-call values), so it contributes the same
# leading tokens on every call and never itself breaks prefix-cache reuse; see
# :meth:`ContextBundle.as_prompt_messages`.
DEFAULT_BUNDLE_SYSTEM_PREAMBLE = (
    "You are a careful assistant. Use ONLY the context below — with its "
    "citations and proof graph — to answer the user's next message. If the "
    "context does not support an answer, say so explicitly.\n\nContext:"
)


@dataclass(frozen=True)
class Citation:
    """One evidence reference a generation step can cite alongside its answer."""

    node_id: str
    kind: str
    evidence_kind: str | None = None
    source_refs: tuple[str, ...] = ()
    span: str = ""
    confidence: float = _NEUTRAL_CONFIDENCE

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "evidence_kind": self.evidence_kind,
            "source_refs": list(self.source_refs),
            "span": self.span,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Citation:
        return cls(
            node_id=str(data.get("node_id", "")),
            kind=str(data.get("kind", "")),
            evidence_kind=data.get("evidence_kind"),
            source_refs=tuple(data.get("source_refs") or ()),
            span=str(data.get("span", "")),
            confidence=float(data.get("confidence", _NEUTRAL_CONFIDENCE)),
        )


@dataclass(frozen=True)
class ProofEdge:
    """One edge in the proof graph: which claim supports/contradicts which."""

    src: str
    dst: str
    relation: str  # "supports" | "contradicts" | "alternative_to"

    def to_dict(self) -> dict[str, Any]:
        return {"src": self.src, "dst": self.dst, "relation": self.relation}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProofEdge:
        return cls(
            src=str(data.get("src", "")),
            dst=str(data.get("dst", "")),
            relation=str(data.get("relation", "")),
        )


@dataclass
class ContextItem:
    """One selected context unit with its full per-axis score breakdown."""

    id: str
    kind: str
    text: str
    tokens: int
    relevance: float
    evidence_quality: float
    freshness: float
    diversity_penalty: float
    composite_score: float
    citation: Citation

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "text": self.text,
            "tokens": self.tokens,
            "relevance": round(self.relevance, 4),
            "evidence_quality": round(self.evidence_quality, 4),
            "freshness": round(self.freshness, 4),
            "diversity_penalty": round(self.diversity_penalty, 4),
            "composite_score": round(self.composite_score, 4),
            "citation": self.citation.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ContextItem:
        return cls(
            id=str(data.get("id", "")),
            kind=str(data.get("kind", "")),
            text=str(data.get("text", "")),
            tokens=int(data.get("tokens", 0)),
            relevance=float(data.get("relevance", 0.0)),
            evidence_quality=float(data.get("evidence_quality", 0.0)),
            freshness=float(data.get("freshness", 0.0)),
            diversity_penalty=float(data.get("diversity_penalty", 0.0)),
            composite_score=float(data.get("composite_score", 0.0)),
            citation=Citation.from_dict(data.get("citation") or {}),
        )


@dataclass
class ContextBundle:
    """The compiled, citable, policy-filtered, budget-fit LLM context."""

    query: str
    items: list[ContextItem] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    proof_graph: list[ProofEdge] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    token_budget: int = 0
    tokens_used: int = 0
    dropped_policy: int = 0
    dropped_redundant: int = 0
    dropped_budget: int = 0
    session_tenant: str = ""
    session_actor: str = ""
    policy_version: str | int = ""
    # Seam 6 (CONCEPT:AU-KG.retrieval.context-compiler-kv-seam) — populated only when
    # ``compile(..., kv_backend=...)`` is used. ``cache_key`` is the stable identity
    # this bundle was stored/looked-up under; ``kv_cache_hit`` is True iff this
    # bundle was reconstructed from the KV layer rather than freshly assembled.
    cache_key: str = ""
    kv_cache_hit: bool = False

    def as_text(self) -> str:
        """Render the bundle to a citation-annotated text block for an LLM prompt.

        Deterministic given the same items (selection order is preserved) —
        the benchmarkable rendering half of the contract.
        """
        if not self.items:
            return f"No context found for: {self.query!r}"
        blocks = []
        for i, item in enumerate(self.items, start=1):
            header = (
                f"[{i}] ({item.kind}) id={item.id} "
                f"score={item.composite_score:.3f} "
                f"relevance={item.relevance:.2f} evidence={item.evidence_quality:.2f} "
                f"freshness={item.freshness:.2f}"
            )
            blocks.append(f"{header}\n{item.text}")
        body = "\n---\n".join(blocks)
        cites = "\n".join(
            f"[{i}] {c.node_id}"
            + (f" (sources: {', '.join(c.source_refs)})" if c.source_refs else "")
            for i, c in enumerate(self.citations, start=1)
        )
        proof = "\n".join(
            f"{e.src} --{e.relation}--> {e.dst}" for e in self.proof_graph
        )
        parts = [body]
        if cites:
            parts.append(f"Citations:\n{cites}")
        if proof:
            parts.append(f"Proof graph:\n{proof}")
        return "\n\n".join(parts)

    def as_prompt_messages(
        self,
        turn_text: str,
        *,
        system_preamble: str = DEFAULT_BUNDLE_SYSTEM_PREAMBLE,
    ) -> list[dict[str, str]]:
        """Render an OpenAI-style ``messages`` list with this bundle as the STABLE PREFIX.

        CONCEPT:AU-KG.retrieval.context-compiler-kv-seam (serving-layer half). The
        prior half of Seam 6 (:meth:`ContextCompiler.compile`'s ``kv_backend=``)
        caches the *assembled bundle object* app-side; this method is the wire from
        that bundle into the actual request vLLM sees, so its own automatic prefix
        cache can reuse the KV blocks of a repeated bundle turn-to-turn.

        The returned messages are ``[system, user]`` where the ``system`` message's
        content is ``system_preamble + as_text()`` — BYTE-IDENTICAL for two calls
        with the same bundle and preamble, regardless of ``turn_text`` — and the
        ``user`` message is ``turn_text``, the only part that varies call-to-call.
        Because it is both the first message and content-identical across calls,
        an OpenAI-compatible server that applies a deterministic chat template
        (vLLM's default) tokenizes an identical leading token run for every call
        sharing this bundle, which is exactly what vLLM's automatic prefix-cache
        (on by default) and LMCache's token-hash cache key off of — no template or
        server change required, just calling this instead of hand-rolling
        ``messages=`` around ``as_text()``.

        A DIFFERENT bundle (different evidence ids, different policy_version, a
        different ``as_of``/budget — anything that changes :meth:`as_text`'s
        output) renders a different ``system`` content and therefore a different
        token prefix, so it never falsely reuses another bundle's cached KV.

        Args:
            turn_text: The turn-specific suffix (the caller's actual question/
                instruction for this call) — appended as the final ``user``
                message, after the stable bundle prefix.
            system_preamble: Fixed literal prepended before :meth:`as_text`'s
                output. Keep this a CONSTANT across calls (the default is) — a
                templated/varying preamble would itself break the stable prefix
                this method exists to produce.

        Returns:
            ``[{"role": "system", "content": <preamble><as_text()>}, {"role":
            "user", "content": <turn_text>}]`` — hand this straight to
            ``client.chat.completions.create(messages=...)``.
        """
        return [
            {"role": "system", "content": f"{system_preamble}{self.as_text()}"},
            {"role": "user", "content": turn_text},
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "items": [it.to_dict() for it in self.items],
            "citations": [c.to_dict() for c in self.citations],
            "proof_graph": [e.to_dict() for e in self.proof_graph],
            "decisions": self.decisions,
            "token_budget": self.token_budget,
            "tokens_used": self.tokens_used,
            "dropped_policy": self.dropped_policy,
            "dropped_redundant": self.dropped_redundant,
            "dropped_budget": self.dropped_budget,
            "session_tenant": self.session_tenant,
            "session_actor": self.session_actor,
            "policy_version": self.policy_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ContextBundle:
        """Reconstruct a bundle from :meth:`to_dict` (the KV-cache round-trip shape).

        ``cache_key``/``kv_cache_hit`` are NOT part of the serialized payload — the
        caller (:meth:`ContextCompiler.compile`) sets them on the returned instance
        after deserializing, since they describe THIS lookup, not the stored bundle.
        """
        return cls(
            query=str(data.get("query", "")),
            items=[ContextItem.from_dict(it) for it in data.get("items") or []],
            citations=[Citation.from_dict(c) for c in data.get("citations") or []],
            proof_graph=[ProofEdge.from_dict(e) for e in data.get("proof_graph") or []],
            decisions=list(data.get("decisions") or []),
            token_budget=int(data.get("token_budget", 0)),
            tokens_used=int(data.get("tokens_used", 0)),
            dropped_policy=int(data.get("dropped_policy", 0)),
            dropped_redundant=int(data.get("dropped_redundant", 0)),
            dropped_budget=int(data.get("dropped_budget", 0)),
            session_tenant=str(data.get("session_tenant", "")),
            session_actor=str(data.get("session_actor", "")),
            policy_version=data.get("policy_version", ""),
        )


def compute_bundle_cache_key(
    evidence_ids: Iterable[str],
    *,
    policy_version: str | int,
    token_budget: int,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """Stable cache key for a :class:`ContextBundle` — CONCEPT:AU-KG.retrieval.context-compiler-kv-seam.

    The identity is the SORTED (order-independent — retrieval doesn't guarantee a
    stable candidate order, only a stable *set* for an unchanged corpus) tuple of
    evidence-node ids the bundle would be assembled from, plus the session's
    ``policy_version`` and the caller's ``token_budget`` — the three axes Seam 6
    calls out: a different evidence set, a different policy version, or a
    different token budget must each mint a different key (no false reuse).
    ``extra`` folds in the remaining assembly parameters (``top_k``,
    ``diversity_lambda``, ``weights``, ``freshness_half_life_days``, ``as_of``,
    ``mask_redactions``) that also change the assembled result, so those can't
    collide either — the three named axes are the documented contract, ``extra``
    is the correctness belt-and-suspenders around it.

    Pure function of its inputs — no I/O, no engine/session object — so it is
    trivially unit-testable and reusable outside :meth:`ContextCompiler.compile`.
    """
    payload = {
        "evidence_ids": sorted(str(e) for e in evidence_ids),
        "policy_version": policy_version,
        "token_budget": int(token_budget),
        "extra": extra or {},
    }
    canonical = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"ctxbundle:{digest}"


def _record_kv_cache_outcome(outcome: str) -> None:
    """Bump the Seam-6 bundle-cache hit/miss counter (WS-4, additive, best-effort).

    CONCEPT:AU-KG.retrieval.context-compiler-kv-seam. Mirrors
    :func:`~agent_utilities.observability.TelemetryEngine.annotate_epistemic`'s
    posture: a metrics-recording failure (extra absent, registry quirk) must
    never break ``compile()`` — this is purely additive observability over an
    already-decided branch, not a new decision point.
    """
    try:
        from agent_utilities.observability.gateway_metrics import (
            CONTEXT_COMPILER_KV_CACHE,
        )

        CONTEXT_COMPILER_KV_CACHE.labels(outcome=outcome).inc()
    except Exception as exc:  # noqa: BLE001 — metrics must never break compile()
        logger.debug("context-compiler kv-cache metric recording failed: %s", exc)


def _record_compile_metrics(bundle: ContextBundle, tokens_in: int) -> None:
    """Emit the WS-4 ContextCompiler efficiency metrics for one ``compile()`` call.

    CONCEPT:AU-KG.retrieval.context-compiler. Records the Prometheus counters/
    histogram (``observability.gateway_metrics.CONTEXT_COMPILER_*`` — no-op
    wherever the ``metrics`` extra is absent) and widens the current OTel span
    via :meth:`~agent_utilities.observability.TelemetryEngine.annotate_context_compiler`
    (no-op wherever no tracing pipeline is active). Best-effort: any failure here
    is logged and swallowed, never raised — this is pure observability over an
    already-assembled bundle, never part of the selection/assembly contract.
    """
    try:
        from agent_utilities.observability.gateway_metrics import (
            CONTEXT_COMPILER_ITEMS,
            CONTEXT_COMPILER_TOKENS,
        )

        CONTEXT_COMPILER_ITEMS.labels(outcome="selected").inc(len(bundle.items))
        CONTEXT_COMPILER_ITEMS.labels(outcome="dropped_policy").inc(
            bundle.dropped_policy
        )
        CONTEXT_COMPILER_ITEMS.labels(outcome="dropped_redundant").inc(
            bundle.dropped_redundant
        )
        CONTEXT_COMPILER_ITEMS.labels(outcome="dropped_budget").inc(
            bundle.dropped_budget
        )
        CONTEXT_COMPILER_TOKENS.labels(kind="in").observe(tokens_in)
        CONTEXT_COMPILER_TOKENS.labels(kind="selected").observe(bundle.tokens_used)
    except Exception as exc:  # noqa: BLE001 — metrics must never break compile()
        logger.debug("context-compiler metric recording failed: %s", exc)

    try:
        from agent_utilities.observability import get_telemetry_engine

        get_telemetry_engine().annotate_context_compiler(
            items_selected=len(bundle.items),
            tokens_in=tokens_in,
            tokens_selected=bundle.tokens_used,
            token_budget=bundle.token_budget,
            dropped_policy=bundle.dropped_policy,
            dropped_redundant=bundle.dropped_redundant,
            dropped_budget=bundle.dropped_budget,
            kv_cache_hit=bundle.kv_cache_hit if bundle.cache_key else None,
        )
    except Exception as exc:  # noqa: BLE001 — tracing must never break compile()
        logger.debug("context-compiler span annotation failed: %s", exc)


def _node_id(node: dict[str, Any]) -> str:
    for key in ("id", "node_id", "_id"):
        val = node.get(key)
        if isinstance(val, str) and val:
            return val
    return ""


def _jaccard(a: str, b: str) -> float:
    """Token-overlap similarity fallback when neither node carries an embedding."""
    ta = {t for t in a.lower().split() if t}
    tb = {t for t in b.lower().split() if t}
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


class ContextCompiler:
    """Assemble a policy-aware, budgeted, cited LLM context bundle (CONCEPT:AU-KG.retrieval.context-compiler, Codex X-7).

    Args:
        engine: An ``IntelligenceGraphEngine`` (preferred — its
            ``search_hybrid`` already composes the native ANN, quality gate,
            and score normalization), a bare ``HybridRetriever``, or any
            duck-typed object exposing ``retrieve_hybrid``/``search_hybrid``.
        hybrid_retriever: Optional explicit retriever, overriding ``engine``'s
            own retrieval path (mainly for tests / composition).
    """

    def __init__(self, engine: Any, *, hybrid_retriever: Any | None = None) -> None:
        self.engine = engine
        self._retriever = hybrid_retriever

    # ------------------------------------------------------------------
    # Retrieval (reused, never reimplemented)
    # ------------------------------------------------------------------
    def _candidate_source(self) -> Any | None:
        if self._retriever is not None:
            return self._retriever
        if hasattr(self.engine, "retrieve_hybrid"):
            return self.engine
        return getattr(self.engine, "hybrid_retriever", None)

    def _retrieve(
        self, query: str, top_k: int, *, as_of: str | None
    ) -> list[dict[str, Any]]:
        search_hybrid = getattr(self.engine, "search_hybrid", None)
        if self._retriever is None and callable(search_hybrid):
            return list(search_hybrid(query, top_k=top_k, as_of=as_of or None) or [])
        source = self._candidate_source()
        if source is None:
            raise TypeError(
                "ContextCompiler needs an engine with `search_hybrid`/"
                "`retrieve_hybrid`, or an explicit `hybrid_retriever=`."
            )
        return list(
            source.retrieve_hybrid(
                query, context_window=top_k, as_of=as_of, skip_quality_gate=True
            )
            or []
        )

    # ------------------------------------------------------------------
    # Per-axis scoring
    # ------------------------------------------------------------------
    # Priority order for the text whose tokens count against the budget and
    # whose overlap drives the MMR diversity fallback. Broader than
    # ``HybridRetriever._node_text`` (which omits ``description``, the field
    # most KG node bodies actually use — see ``graph_search``'s formatter and
    # ``context_builder.build_contextual_description``) since a compiler that
    # can't see the node's own description would assemble near-empty context.
    _TEXT_FIELDS = ("content", "text", "description", "summary", "name")

    @staticmethod
    def _text_of(node: dict[str, Any]) -> str:
        parts = [str(node.get(k, "")) for k in ContextCompiler._TEXT_FIELDS]
        body = " ".join(p for p in parts if p)
        return body or str(node.get("id") or node)

    @staticmethod
    def _raw_relevance(node: dict[str, Any]) -> float:
        val = node.get("score")
        if val is None:
            val = node.get("_score")
        try:
            return float(val) if val is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _evidence_quality(node: dict[str, Any]) -> float:
        """Score a candidate's evidence quality from its epistemic columns (CONCEPT:EPI-P3-1).

        Reads the ``KnowledgeBatch``-shaped fields when present — ``confidence``,
        ``source_refs``, ``evidence_refs``, ``proof_ids``, ``contradiction_ids``,
        ``policy_labels`` — and degrades gracefully to a neutral prior for a
        plain node carrying none of them.
        """
        conf = node.get("confidence")
        if conf is None:
            conf = node.get("trust_score")
        try:
            conf = float(conf) if conf is not None else _NEUTRAL_CONFIDENCE
        except (TypeError, ValueError):
            conf = _NEUTRAL_CONFIDENCE
        conf = max(0.0, min(1.0, conf))

        bonus = 0.0
        if node.get("source_refs") or node.get("sources"):
            bonus += 0.1
        if node.get("evidence_refs") or node.get("evidence"):
            bonus += 0.1
        if node.get("proof_ids"):
            bonus += 0.05

        penalty = 0.0
        policy_labels = node.get("policy_labels") or []
        if node.get("contradiction_ids") or _CONTESTED_LABEL in policy_labels:
            penalty += 0.2

        return max(0.0, min(1.0, conf + bonus - penalty))

    @staticmethod
    def _freshness(
        node: dict[str, Any], *, as_of: str | None, half_life_days: float
    ) -> float:
        """Bi-temporal recency score in ``[0, 1]`` (1.0 = now, 0.5 at one half-life)."""
        raw = (
            node.get("event_time")
            or node.get("valid_from")
            or node.get("timestamp")
            or node.get("created_at")
            or node.get("updated_at")
        )
        ts = _parse_instant(raw)
        if ts is None:
            return _NEUTRAL_FRESHNESS
        ref = _parse_instant(as_of) or datetime.now(UTC)
        age_days = max(0.0, (ref - ts).total_seconds() / 86400.0)
        return 0.5 ** (age_days / max(1e-6, half_life_days))

    def _max_similarity(
        self, node: dict[str, Any], selected: list[dict[str, Any]]
    ) -> float:
        if not selected:
            return 0.0
        emb_a = node.get("embedding")
        text_a = self._text_of(node)
        best = 0.0
        for other in selected:
            emb_b = other.get("embedding")
            if emb_a and emb_b:
                sim = cosine_similarity(emb_a, emb_b)
            else:
                sim = _jaccard(text_a, self._text_of(other))
            best = max(best, sim)
        return best

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------
    def compile(
        self,
        query: str,
        session: GraphSession | None = None,
        *,
        top_k: int = 8,
        candidate_pool: int = 40,
        token_budget: int = 2000,
        diversity_lambda: float = 0.5,
        freshness_half_life_days: float = 30.0,
        weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
        as_of: str | None = None,
        mask_redactions: bool = True,
        kv_backend: Any | None = None,
    ) -> ContextBundle:
        """Assemble a policy-aware, budgeted, cited context bundle for ``query``.

        Args:
            query: The natural-language query to retrieve and assemble context for.
            session: The requesting :class:`GraphSession` (actor/tenant/policy);
                defaults to :meth:`GraphSession.from_ambient`.
            top_k: Maximum items in the final bundle.
            candidate_pool: How many candidates to over-fetch before scoring/MMR
                (must be >= ``top_k`` to leave MMR/budget room to trade off).
            token_budget: Token budget the bundle must fit inside.
            diversity_lambda: MMR trade-off — 1.0 is pure relevance/composite,
                0.0 is pure novelty. 0.5 balances both.
            freshness_half_life_days: Recency decay half-life for the freshness axis.
            weights: ``(relevance, evidence_quality, freshness)`` composite weights;
                renormalized to sum to 1.
            as_of: Optional ISO-8601 reference instant for both retrieval and the
                freshness axis (knowledge-state-as-of-date-D).
            mask_redactions: Redact policy-denied columns (``MASK_TOKEN``) rather
                than dropping them outright — passed straight to
                :func:`~agent_utilities.knowledge_graph.ontology.permissioning.enforce`.
            kv_backend: OPT-IN (CONCEPT:AU-KG.retrieval.context-compiler-kv-seam) — a duck-typed
                object exposing ``get(key) -> bytes | None`` / ``put(key, bytes) -> bool``
                (the exact shape ``agent_utilities.kvcache.EpistemicGraphKVBackend``
                implements). When supplied, ``compile`` computes a stable cache key
                from the post-policy evidence-id set + ``session.policy_version`` +
                ``token_budget`` (see :func:`compute_bundle_cache_key`); a hit
                returns the previously-assembled bundle WITHOUT re-running
                relevance/evidence/freshness scoring, MMR diversity selection,
                budget-fitting, or proof-graph construction, and a miss assembles
                normally then stores the result for the next caller. ``None``
                (the default) leaves this method's behavior byte-for-byte
                unchanged from before Seam 6.

        Returns:
            A :class:`ContextBundle` — items with per-axis scores, citations,
            proof graph, and a full ``decisions`` log. When ``kv_backend`` is
            used, also ``cache_key`` (the identity it was stored/looked-up under)
            and ``kv_cache_hit`` (True iff served from the KV layer).
        """
        session = session or GraphSession.from_ambient()
        pool = max(candidate_pool, top_k)
        candidates = self._retrieve(query, pool, as_of=as_of)
        decisions: list[dict[str, Any]] = []

        # ---- 6. POLICY — the SAME fine-grained gate the live read path uses.
        allowed = enforce(candidates, session.actor, mask=mask_redactions)
        allowed_ids = {_node_id(n) for n in allowed}
        dropped_policy = 0
        for cand in candidates:
            nid = _node_id(cand)
            if nid not in allowed_ids:
                dropped_policy += 1
                decisions.append(
                    {
                        "id": nid,
                        "stage": "policy",
                        "included": False,
                        "reason": "policy_denied",
                    }
                )

        # ---- Seam 6: KV-cache lookup, keyed on the post-policy evidence-id set
        # (the pool this bundle would be assembled from) + policy_version +
        # token_budget — see compute_bundle_cache_key. Computed here (after
        # retrieval+policy, BEFORE the expensive scoring/MMR/budget/proof-graph
        # work below) so a hit can skip straight past all of it.
        cache_key = ""
        if kv_backend is not None:
            cache_key = compute_bundle_cache_key(
                allowed_ids,
                policy_version=session.policy_version,
                token_budget=token_budget,
                extra={
                    "top_k": top_k,
                    "diversity_lambda": diversity_lambda,
                    "freshness_half_life_days": freshness_half_life_days,
                    "weights": list(weights),
                    "as_of": as_of,
                    "mask_redactions": mask_redactions,
                },
            )
            cached_bytes = kv_backend.get(cache_key)
            if cached_bytes is not None:
                try:
                    cached_bundle = ContextBundle.from_dict(
                        json.loads(cached_bytes.decode("utf-8"))
                    )
                except (
                    json.JSONDecodeError,
                    UnicodeDecodeError,
                    TypeError,
                    ValueError,
                    KeyError,
                ) as exc:
                    logger.debug(
                        "[CONCEPT:AU-KG.retrieval.context-compiler-kv-seam] cache hit for "
                        "key=%s failed to deserialize, recomputing: %s",
                        cache_key,
                        exc,
                    )
                else:
                    cached_bundle.cache_key = cache_key
                    cached_bundle.kv_cache_hit = True
                    logger.info(
                        "[CONCEPT:AU-KG.retrieval.context-compiler-kv-seam] kv-cache hit "
                        "key=%s items=%d (assembly skipped)",
                        cache_key,
                        len(cached_bundle.items),
                    )
                    _record_kv_cache_outcome("hit")
                    return cached_bundle
            # A miss (no cached bytes, or a cached blob that failed to
            # deserialize) falls through to the normal assembly path below.
            _record_kv_cache_outcome("miss")

        # ---- 1/3/4. RELEVANCE (normalized) + EVIDENCE QUALITY + FRESHNESS.
        raw_scores = [self._raw_relevance(n) for n in allowed]
        lo = min(raw_scores) if raw_scores else 0.0
        hi = max(raw_scores) if raw_scores else 0.0
        spread = hi - lo

        w_sum = sum(weights) or 1.0
        w_rel, w_ev, w_fresh = (w / w_sum for w in weights)

        records: list[dict[str, Any]] = []
        for node, raw in zip(allowed, raw_scores, strict=False):
            relevance = (raw - lo) / spread if spread > 0 else 1.0
            evidence_quality = self._evidence_quality(node)
            freshness = self._freshness(
                node, as_of=as_of, half_life_days=freshness_half_life_days
            )
            composite = (
                w_rel * relevance + w_ev * evidence_quality + w_fresh * freshness
            )
            records.append(
                {
                    "node": node,
                    "nid": _node_id(node),
                    "relevance": relevance,
                    "evidence_quality": evidence_quality,
                    "freshness": freshness,
                    "composite": composite,
                }
            )
        # Deterministic base order: composite desc, id asc tie-break.
        records.sort(key=lambda r: (-r["composite"], r["nid"]))

        # ---- 2. DIVERSITY — greedy MMR selection.
        selected: list[dict[str, Any]] = []
        selected_nodes: list[dict[str, Any]] = []
        remaining = list(records)
        while remaining and len(selected) < top_k:
            best_rec: dict[str, Any] | None = None
            best_mmr = float("-inf")
            best_div = 0.0
            for rec in remaining:
                sim = self._max_similarity(rec["node"], selected_nodes)
                mmr_score = (
                    diversity_lambda * rec["composite"] - (1 - diversity_lambda) * sim
                )
                if (
                    best_rec is None
                    or mmr_score > best_mmr + 1e-12
                    or (
                        abs(mmr_score - best_mmr) <= 1e-12
                        and rec["nid"] < best_rec["nid"]
                    )
                ):
                    best_rec = rec
                    best_mmr = mmr_score
                    best_div = sim
            assert best_rec is not None
            best_rec = dict(best_rec)
            best_rec["diversity_penalty"] = best_div
            selected.append(best_rec)
            selected_nodes.append(best_rec["node"])
            remaining.remove(next(r for r in remaining if r["nid"] == best_rec["nid"]))

        dropped_redundant = len(records) - len(selected)
        selected_ids = {r["nid"] for r in selected}
        for rec in records:
            if rec["nid"] not in selected_ids:
                decisions.append(
                    {
                        "id": rec["nid"],
                        "stage": "select",
                        "included": False,
                        "reason": "mmr_not_selected",
                        "composite_score": round(rec["composite"], 4),
                    }
                )

        # WS-4: "tokens-in" for the efficiency metrics below — the token cost of
        # the MMR-ranked pool as handed to the budget fit, i.e. before any
        # budget-driven truncation (the selection-efficiency signal is
        # tokens_selected / tokens_in).
        tokens_in = sum(
            _estimate_item_tokens(self._text_of(r["node"])) for r in selected
        )

        # ---- 5. TOKEN COST — fit the MMR-ranked selection within budget.
        mgr = RetrievalBudgetManager(token_budget)
        budget_result = mgr.fit(selected, text_of=lambda r: self._text_of(r["node"]))
        kept_ids = {r["nid"] for r in budget_result.kept}
        dropped_budget = budget_result.dropped
        for rec in selected:
            if rec["nid"] not in kept_ids:
                decisions.append(
                    {
                        "id": rec["nid"],
                        "stage": "budget",
                        "included": False,
                        "reason": "token_budget",
                        "composite_score": round(rec["composite"], 4),
                    }
                )

        items: list[ContextItem] = []
        records_by_id: dict[str, dict[str, Any]] = {}
        for rec in budget_result.kept:
            node = rec["node"]
            text = self._text_of(node)
            citation = Citation(
                node_id=rec["nid"],
                kind=str(node.get("type") or node.get("label") or "Unknown"),
                evidence_kind=node.get("evidence_kind"),
                source_refs=tuple(node.get("source_refs") or node.get("sources") or ()),
                span=text[:240],
                confidence=float(node.get("confidence", _NEUTRAL_CONFIDENCE) or 0.0),
            )
            item = ContextItem(
                id=rec["nid"],
                kind=citation.kind,
                text=text,
                tokens=_estimate_item_tokens(text),
                relevance=rec["relevance"],
                evidence_quality=rec["evidence_quality"],
                freshness=rec["freshness"],
                diversity_penalty=rec["diversity_penalty"],
                composite_score=rec["composite"],
                citation=citation,
            )
            items.append(item)
            records_by_id[rec["nid"]] = node
            decisions.append(
                {
                    "id": rec["nid"],
                    "stage": "select",
                    "included": True,
                    "relevance": round(rec["relevance"], 4),
                    "evidence_quality": round(rec["evidence_quality"], 4),
                    "freshness": round(rec["freshness"], 4),
                    "diversity_penalty": round(rec["diversity_penalty"], 4),
                    "composite_score": round(rec["composite"], 4),
                }
            )

        proof_graph = self._proof_graph(records_by_id, query)

        bundle = ContextBundle(
            query=query,
            items=items,
            citations=[it.citation for it in items],
            proof_graph=proof_graph,
            decisions=decisions,
            token_budget=token_budget,
            tokens_used=budget_result.tokens_used,
            dropped_policy=dropped_policy,
            dropped_redundant=max(0, dropped_redundant),
            dropped_budget=dropped_budget,
            session_tenant=session.tenant,
            session_actor=session.actor.actor_id if session.actor else "",
            policy_version=session.policy_version,
            cache_key=cache_key,
        )
        logger.info(
            "[CONCEPT:AU-KG.retrieval.context-compiler] context compiled: query=%r items=%d tokens=%d/%d "
            "dropped(policy=%d redundant=%d budget=%d)",
            query,
            len(items),
            bundle.tokens_used,
            token_budget,
            dropped_policy,
            bundle.dropped_redundant,
            dropped_budget,
        )
        _record_compile_metrics(bundle, tokens_in)

        # ---- Seam 6: register the freshly-assembled bundle with the KV-cache
        # layer under the SAME key just computed, so the next caller with an
        # identical evidence set/policy_version/token_budget gets the reuse
        # path above. Best-effort — a failed store never fails compilation.
        if kv_backend is not None and cache_key:
            try:
                stored = kv_backend.put(
                    cache_key,
                    json.dumps(bundle.to_dict(), default=str).encode("utf-8"),
                )
            except Exception as exc:  # noqa: BLE001 — store is best-effort
                logger.debug(
                    "[CONCEPT:AU-KG.retrieval.context-compiler-kv-seam] kv-cache store "
                    "for key=%s failed, continuing without caching: %s",
                    cache_key,
                    exc,
                )
            else:
                logger.debug(
                    "[CONCEPT:AU-KG.retrieval.context-compiler-kv-seam] kv-cache store "
                    "key=%s stored=%s",
                    cache_key,
                    stored,
                )
        return bundle

    # ------------------------------------------------------------------
    # Proof graph
    # ------------------------------------------------------------------
    def _proof_graph(
        self, records_by_id: dict[str, dict[str, Any]], query: str
    ) -> list[ProofEdge]:
        """Build the supports/contradicts/alternative-to graph for the selection.

        Reads the epistemic columns directly on the selected candidates
        (``proof_ids``/``contradiction_ids``/``alternative_ids`` — CONCEPT:EPI-P3-1),
        then — when the engine exposes it — augments with
        ``retrieve_epistemic_view``'s belief/support/contradiction traversal
        rather than re-deriving it.
        """
        edges: list[ProofEdge] = []
        seen: set[tuple[str, str, str]] = set()

        def _add(src: str, dst: str, relation: str) -> None:
            if not src or not dst:
                return
            key = (src, dst, relation)
            if key in seen:
                return
            seen.add(key)
            edges.append(ProofEdge(src=src, dst=dst, relation=relation))

        for nid, node in records_by_id.items():
            for pid in node.get("proof_ids") or []:
                _add(str(pid), nid, "supports")
            for cid in node.get("contradiction_ids") or []:
                _add(nid, str(cid), "contradicts")
            for aid in node.get("alternative_ids") or []:
                _add(nid, str(aid), "alternative_to")

        epistemic_view = getattr(self.engine, "retrieve_epistemic_view", None)
        if callable(epistemic_view):
            try:
                view = epistemic_view(query, top_k=max(5, len(records_by_id))) or {}
                for s in view.get("supporting") or []:
                    target = s.get("_target_claim")
                    src = s.get("id")
                    if src and target:
                        _add(str(src), str(target), "supports")
                for c in view.get("contradicting") or []:
                    target = c.get("_target_claim")
                    src = c.get("id")
                    if src and target:
                        _add(str(target), str(src), "contradicts")
            except Exception as e:  # noqa: BLE001 — augmentation is best-effort
                logger.debug("epistemic view proof-graph augmentation skipped: %s", e)

        return edges


def _estimate_item_tokens(text: str) -> int:
    """Per-item token estimate, matching the shared ``estimate_tokens`` heuristic."""
    from ..memory.agent_context import estimate_tokens

    return estimate_tokens(text)
