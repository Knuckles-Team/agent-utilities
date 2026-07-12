"""The graph-os **intent surface** — Seam 8, Phases 2-5 (CONCEPT:AU-ECO.mcp.intent-surface-condensed-collapse).

graph-os's condensed action-routed surface is already ~95 tools — past the point
where more tools *lowers* LLM tool-selection accuracy (worst for small/cheap
models). This module adds a SMALL, additional set of **intent verbs**
(``ask``/``find``/``write``/``act``/``manage``/``why``) that collapse the whole
granular surface behind six tiny schemas: the caller states an intent (natural
language, optionally with structured ``hints``), a resolver ranks the matching
granular capability, dispatches it through the **same** ``_execute_tool`` core
every condensed tool uses, and returns the result **with the routing
justification attached** — proof-carrying capability dispatch, not an opaque
choice.

**Nothing is removed.** Every granular tool + its REST twin still exists exactly
as before; the intent verbs are an ADDITIONAL surface, registered only under
``MCP_TOOL_MODE=intent`` (see ``mcp/verbose_tools.py``), where the granular
tools are also tagged :data:`~agent_utilities.mcp.verbose_tools.GATED_TAG` and
held back from the default session view — ``load_tools`` (or pinning
``hints={"tool": "..."}``) always reaches the exact tool.

**Resolver — CPD-backed, with a graceful pre-CPD lexical fallback.** The
Capability Power Descriptor (CPD — the per-capability ``does``/``examples``/
``intent_verbs``/``eligibility`` record generated to
``docs/capabilities-power.json``, see ``capability_power_descriptor.py``) has
landed (Seam 8 Phase 1, ``feat/au-cpd``). This resolver now ranks each
capability against its OWN CPD text (``one_line`` + ``examples`` + ``does[]``
action names) when a CPD exists for that tool — a richer, drift-gated signal
than a bare docstring — via the SAME dependency-free lexical scorer (still no
embeddings, no new heavy dependency). A tool with no CPD entry (a brand-new
tool ahead of the next ``gen_capability_power.py --write``, or the CPD JSON
being entirely absent — e.g. a lean/headless install that doesn't ship
``docs/``) falls back per-capability to the original local candidate table
(:data:`TOOL_VERBS` + live ``REGISTERED_TOOLS`` docstrings +
``_graphos_action_manifest``) — never an error, never a gap. The
``CapabilityCandidate``/``resolve_intent`` contract is unchanged, exactly as
this module's original design intended.

**Calibrated-outcomes learning loop.** Every :func:`dispatch_intent` call
feeds its success/failure back into the SAME durable-bandit reward-EMA
mechanism the rest of the platform already uses for outcome-learned routing
(:class:`~agent_utilities.orchestration.outcome_router.OutcomeRouter`, itself a
thin wrapper over :class:`~agent_utilities.knowledge_graph.retrieval.
capability_index.CapabilityIndex`'s ``record_outcome``/``reward_of`` — no
second learner). :func:`resolve_intent` blends each candidate's learned
reward EMA (keyed ``verb:tool``) into its lexical score, so a capability that
keeps failing under a verb quietly sinks in the ranking and one that keeps
succeeding rises — real calibrated-outcomes routing, not merely the static
(always-empty at generation time) ``calibrated_outcomes`` field baked into the
checked-in CPD JSON.

**Resolution caching.** Ranked (non-pinned) resolutions are cached in a small
bounded in-process LRU keyed by ``(verb, normalized intent, hints, top_k)``
PLUS two monotonic generation counters — the candidate-table generation (bumps
whenever the CPD/tool surface is rebuilt) and the reward epoch (bumps on every
outcome recorded) — so a repeated intent is served straight from the cache
until the routing policy it was ranked under actually changes, and a learned
outcome invalidates exactly the entries whose ranking it could have altered.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Any

from pydantic import Field

from agent_utilities.knowledge_graph.retrieval.capability_context import load_cpds
from agent_utilities.mcp import kg_server

logger = logging.getLogger(__name__)

__all__ = [
    "INTENT_VERBS",
    "TOOL_VERBS",
    "CapabilityCandidate",
    "resolve_intent",
    "dispatch_intent",
    "register_intent_tools",
]

#: The six intent verbs (CONCEPT:AU-ECO.mcp.intent-surface-condensed-collapse). ``find`` is unfiltered (ranks
#: across every verb — it is capability DISCOVERY, not one category of work).
INTENT_VERBS: tuple[str, ...] = ("ask", "find", "write", "act", "manage", "why")

#: Every granular tool's home verb(s) — first entry is primary. Built by hand
#: from the design doc's verb table (program-design-2026-07-11) plus the live
#: ``graph-os`` tool inventory (95 ``REGISTERED_TOOLS`` as of Seam 8 kickoff);
#: a tool absent here still resolves (falls back to ``("ask",)``) so a NEW
#: granular tool is never silently unroutable — only unranked-by-name until
#: this table (or the future CPD) is extended.
TOOL_VERBS: dict[str, tuple[str, ...]] = {
    # ── reads / NL / search / analysis ──
    "ask_data": ("ask",),
    "nl_query": ("ask",),
    "graph_query": ("ask",),
    "graph_ask": ("ask",),
    "graph_search": ("ask",),
    "graph_search_synthesis": ("ask",),
    "graph_analyze": ("ask", "why"),
    "graph_context": ("ask",),
    "graph_document_tree": ("ask", "write"),
    "graph_table": ("ask", "write"),
    "graph_promql": ("ask",),
    "graph_federated_search": ("ask",),
    "graph_code": ("ask",),
    "graph_code_nav": ("ask",),
    "graph_reach": ("act",),
    "graph_gis": ("ask",),
    "usage_query": ("ask",),
    "concept_registry": ("find", "ask"),
    "object_index": ("ask", "find"),
    "object_set": ("ask", "write"),
    "research_artifact": ("ask", "write"),
    "quant": ("ask", "act"),
    "engine_query": ("ask",),
    "engine_analytics": ("ask",),
    "engine_datascience": ("ask",),
    "engine_mining": ("ask",),
    "engine_graph": ("ask", "write"),
    "graph_mine": ("ask",),
    "graph_mine_deep": ("act", "ask"),
    "graph_learn": ("act", "ask"),
    "graph_ops_causal": ("why", "ask"),
    "graph_traces": ("ask", "why"),
    # ── writes / ingest / persist ──
    "graph_write": ("write",),
    "graph_ingest": ("write",),
    "graph_writeback": ("write",),
    "graph_etl": ("write",),
    "source_sync": ("write",),
    "source_connector": ("manage", "write"),
    "source_drain": ("write",),
    "ingest_sessions": ("write",),
    "object_edits": ("write",),
    "ontology_derive": ("write",),
    "ontology_link_materialize": ("write",),
    "ontology_leanix_sync": ("write",),
    "document_process": ("write",),
    "spec_ticket": ("write", "ask"),
    "engine_nodes": ("write", "ask"),
    "engine_edges": ("write",),
    "engine_blob": ("write",),
    "engine_rdf": ("write", "ask"),
    "engine_timeseries": ("write", "ask"),
    "graph_share": ("write", "manage"),
    "graph_feedback": ("why", "write"),
    "ontology_function": ("act", "write"),
    # ── act / orchestrate / execute / schedule ──
    "graph_orchestrate": ("act",),
    "graph_loops": ("act",),
    "graph_goals": ("act",),
    "graph_sandbox": ("act",),
    "graph_runvcs": ("act",),
    "graph_fork": ("act",),
    "graph_bus": ("act",),
    "graph_broker": ("act",),
    "graph_message": ("act",),
    "graph_feeds": ("act", "ask"),
    "graph_research": ("ask", "act"),
    "engine_txn": ("act",),
    "engine_consensus": ("act",),
    "engine_channels": ("act",),
    "engine_streaming": ("act",),
    "engine_ledger": ("write", "act"),
    # ── manage / configure / admin ──
    "graph_configure": ("manage",),
    "graph_secret": ("manage",),
    "graph_sessions": ("manage", "ask"),
    "graph_kvcache": ("manage",),
    "graph_hydrate": ("manage", "write"),
    "graph_schedules": ("manage", "act"),
    "graph_ontology": ("write", "ask", "manage"),
    "ontology_property_types": ("manage", "ask"),
    "ontology_value_types": ("manage", "ask"),
    "ontology_interface": ("manage", "ask"),
    "ontology_sampling_profile": ("manage", "ask"),
    "object_permissioning": ("manage",),
    "engine_tenants": ("manage",),
    "engine_lifecycle": ("manage", "act"),
    "engine_resharding": ("manage",),
    "engine_rbac": ("manage",),
    "engine_admin": ("manage",),
    # ── why / explain / evaluate / observe ──
    "graph_explain": ("why",),
    "graph_evaluate": ("why",),
    "graph_observe": ("why", "ask"),
    "graph_memory": ("write", "ask"),
    "engine_reasoning": ("why",),
}

#: Tools whose primary argument is a single free-text NL string — the resolver
#: seeds it from the caller's raw ``intent`` when the caller supplied no
#: structured ``hints`` for it, so ``ask("<plain English>")`` works with ZERO
#: hints (the specific UX program-design-2026-07-11 calls out). Every other
#: tool still resolves and dispatches — it just needs the caller's ``hints`` to
#: carry its real parameters, exactly as calling it directly would.
_PRIMARY_TEXT_PARAM: dict[str, str] = {
    "nl_query": "text",
    "ask_data": "question",
    "graph_ask": "question",
    "graph_search": "query",
    "graph_promql": "query",
    "graph_federated_search": "query",
    "graph_analyze": "query",
    "graph_explain": "query",
    "graph_evaluate": "query",
    "graph_observe": "query",
}

#: Safe universal fallback for ``ask`` when the top-ranked candidate needs
#: structured params the caller didn't supply — the engine's own NL planner
#: (CONCEPT:AU-KG.query.ask-gateway-rest-twin) always accepts raw text.
_ASK_FALLBACK_TOOL = "nl_query"

_STOPWORDS = frozenset(
    "a an the of for to in on with and or is are was were be do does what how "
    "why when where which who this that it its into over about via my me i "
    "you your please can could should would like".split()
)

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> Counter:
    words = [
        w for w in _WORD_RE.findall(str(text or "").lower()) if w not in _STOPWORDS
    ]
    return Counter(words)


@dataclass
class CapabilityCandidate:
    """One rankable graph-os capability (a tool, or a tool+action pair)."""

    tool: str
    action: str | None
    verbs: tuple[str, ...]
    doc: str
    score: float = 0.0
    matched_terms: list[str] = field(default_factory=list)

    @property
    def capability_id(self) -> str:
        return f"{self.tool}:{self.action}" if self.action else self.tool


_CANDIDATES_CACHE: list[CapabilityCandidate] | None = None
_ACTIONS_BY_TOOL_CACHE: dict[str, list[str]] | None = None

#: Bumped every time :func:`_build_candidates` actually rebuilds (CONCEPT:AU-ECO.mcp.intent-surface-resolution-cache)
#: — the "CPD/policy version" half of the resolution-cache key below. Tests
#: force a rebuild by resetting ``_CANDIDATES_CACHE`` to ``None``.
_CANDIDATES_GENERATION: int = 0

#: Bumped every time an outcome is recorded (CONCEPT:AU-ECO.mcp.intent-surface-outcome-learning) — the
#: "policy just changed" half of the resolution-cache key: a fresh outcome
#: invalidates cached rankings that could have used it, without a full flush.
_REWARD_EPOCH: int = 0

#: Bounded LRU of ranked (non-pinned) resolutions (CONCEPT:AU-ECO.mcp.intent-surface-resolution-cache) — see
#: :func:`_cache_key`. A pinned (``hints={"tool": ...}``) resolution is O(1)
#: already and is never cached.
_RESOLUTION_CACHE: OrderedDict[
    tuple[Any, ...], list[CapabilityCandidate]
] = OrderedDict()
_RESOLUTION_CACHE_MAX = 256

#: Soft weight of the learned reward-EMA blend into the lexical score (mirrors
#: ``CapabilityIndex.designate``'s own ``reward_weight`` default) — early on
#: (every candidate at the neutral 0.5 prior) the lexical ranking is
#: untouched; as outcomes accumulate a candidate's blended score rises or
#: sinks with its real success rate under that verb.
_LEARNED_REWARD_WEIGHT = 0.2

#: Lazily-constructed shared learner (CONCEPT:AU-ECO.mcp.intent-surface-outcome-learning) — ``None`` until first
#: touched so importing this module never pays ``OutcomeRouter``'s (numeric
#: package) import cost; degrades to a no-op neutral fallback if that optional
#: dependency is unavailable (lean/headless install), never breaking routing.
_OUTCOME_ROUTER: Any = None


class _NullOutcomeRouter:
    """No-op stand-in when the durable-bandit learner is unavailable.

    Every candidate reads back the neutral 0.5 prior and ``record`` is a
    no-op — the learning loop degrades to "off", never to a crash.
    """

    def reward_of(self, _task_class: str, _choice: str) -> float:
        return 0.5

    def record(self, _task_class: str, _choice: str, _reward: float) -> None:
        return None


def _outcome_router() -> Any:
    """The shared :class:`OutcomeRouter` for intent-surface dispatch outcomes.

    CONCEPT:AU-ECO.mcp.intent-surface-outcome-learning. ONE learner, reused — the same
    durable-bandit mechanism (``CapabilityIndex.record_outcome``/``reward_of``)
    every other outcome-learned choice in this codebase already shares
    (``ReasonerRouter``, ``variant_pool.evolve_profile``), keyed
    ``intent_surface:<verb>:<capability_id>`` so it never collides with
    another router's namespace.
    """
    global _OUTCOME_ROUTER
    if _OUTCOME_ROUTER is None:
        try:
            from agent_utilities.orchestration.outcome_router import OutcomeRouter

            _OUTCOME_ROUTER = OutcomeRouter(namespace="intent_surface")
        except Exception as e:  # noqa: BLE001 — learning is best-effort, never load-bearing
            logger.debug(
                "[Seam 8] outcome router unavailable, learning disabled: %s", e
            )
            _OUTCOME_ROUTER = _NullOutcomeRouter()
    return _OUTCOME_ROUTER


def _record_dispatch_outcome(verb: str, tool: str, *, success: bool) -> None:
    """Feed one dispatch's success/failure back into the shared reward-EMA.

    CONCEPT:AU-ECO.mcp.intent-surface-outcome-learning. Bumps :data:`_REWARD_EPOCH` so any
    cached resolution that could have used this outcome is invalidated on its
    next lookup rather than served stale.
    """
    global _REWARD_EPOCH
    try:
        _outcome_router().record(verb, tool, 1.0 if success else 0.0)
    finally:
        _REWARD_EPOCH += 1


def _normalize_intent(intent: str) -> str:
    """Whitespace/case-fold an intent string for resolution-cache keying."""
    return " ".join(str(intent or "").strip().lower().split())


def _cache_key(
    verb: str | None, intent: str, hints: dict[str, Any], top_k: int
) -> tuple[Any, ...]:
    """The resolution-cache key (CONCEPT:AU-ECO.mcp.intent-surface-resolution-cache).

    Normalized intent + verb + a stable ``hints`` projection + ``top_k``, PLUS
    the two monotonic generation counters — so the SAME intent under the SAME
    routing policy hits the cache, and either the tool surface/CPD changing or
    a fresh outcome being recorded naturally busts exactly the entries that
    could be affected (their key simply no longer matches).
    """
    hints_key = tuple(sorted((str(k), str(v)) for k, v in (hints or {}).items()))
    return (
        verb,
        _normalize_intent(intent),
        hints_key,
        int(top_k),
        _CANDIDATES_GENERATION,
        _REWARD_EPOCH,
    )


def _load_cpds_safe() -> dict[str, dict[str, Any]]:
    """``{tool: cpd_dict}`` from ``docs/capabilities-power.json``, or ``{}``.

    :func:`~agent_utilities.knowledge_graph.retrieval.capability_context.load_cpds`
    already degrades to ``{}`` when the file is absent (lean/headless install,
    or ahead of the next ``--write``); this wraps it in a defensive
    ``try/except`` too so an unexpected read/parse error can never take down
    the resolver — it just falls back to the pre-CPD lexical candidate table.
    """
    try:
        return load_cpds()
    except Exception as e:  # noqa: BLE001 — CPD is an enrichment, never load-bearing
        logger.debug("[Seam 8] CPD load failed, using lexical fallback: %s", e)
        return {}


def _actions_by_tool() -> dict[str, list[str]]:
    global _ACTIONS_BY_TOOL_CACHE
    if _ACTIONS_BY_TOOL_CACHE is not None:
        return _ACTIONS_BY_TOOL_CACHE
    out: dict[str, list[str]] = {}
    try:
        from agent_utilities.mcp._graphos_action_manifest import GRAPHOS_ACTIONS

        for op in GRAPHOS_ACTIONS:
            action = op.get("action")
            if action is None:
                continue
            out.setdefault(op["tool"], []).append(action)
    except Exception:  # noqa: BLE001 — manifest is a generated convenience, never load-bearing
        pass
    _ACTIONS_BY_TOOL_CACHE = out
    return out


def _tool_doc(tool: str) -> str:
    fn = kg_server.REGISTERED_TOOLS.get(tool)
    doc = getattr(fn, "__doc__", None) or ""
    return " ".join(doc.split())[:400]


def _build_candidates(*, force: bool = False) -> list[CapabilityCandidate]:
    """One :class:`CapabilityCandidate` per live ``REGISTERED_TOOLS`` entry.

    CONCEPT:AU-ECO.mcp.intent-surface-cpd-ranking. For a tool with a generated CPD entry
    (``docs/capabilities-power.json``), the candidate's ``doc`` text is built
    from the CPD's OWN ``one_line`` + ``examples`` + ``does[]`` action names
    (richer than a bare docstring) and its ``verbs`` are the UNION of the
    hand-curated :data:`TOOL_VERBS` entry (if any) with the CPD's own
    ``intent_verbs`` — union, never narrower, so switching to the CPD can only
    ADD routing surface, never silently drop a verb a real skill already
    documents. A tool absent from the CPD set (new tool, or the CPD JSON
    missing entirely) falls back per-capability to the original lexical
    candidate (:data:`TOOL_VERBS` + live docstring) — never an error.

    Cached process-wide (tool registration happens once at server build); pass
    ``force=True`` in tests after monkeypatching ``REGISTERED_TOOLS``. Bumps
    :data:`_CANDIDATES_GENERATION` on every actual rebuild (CONCEPT:AU-ECO.mcp.intent-surface-resolution-cache).
    """
    global _CANDIDATES_CACHE, _CANDIDATES_GENERATION
    if _CANDIDATES_CACHE is not None and not force:
        return _CANDIDATES_CACHE
    kg_server.ensure_tools_registered()
    actions_by_tool = _actions_by_tool()
    cpds = _load_cpds_safe()
    out: list[CapabilityCandidate] = []
    for tool in sorted(kg_server.REGISTERED_TOOLS):
        hand_verbs = TOOL_VERBS.get(tool)
        cpd = cpds.get(tool)
        if cpd is not None:
            cpd_verbs = tuple(str(v) for v in (cpd.get("intent_verbs") or ()))
            verbs = tuple(sorted(set(hand_verbs or ()) | set(cpd_verbs))) or ("ask",)
            examples_text = " ".join(str(e) for e in (cpd.get("examples") or ()))
            does_text = " ".join(
                str(d.get("action", "")) for d in (cpd.get("does") or ())
            )
            doc = (
                f"{tool} {' '.join(actions_by_tool.get(tool, []))} "
                f"{cpd.get('one_line', '')} {examples_text} {does_text}"
            )
        else:
            verbs = hand_verbs or ("ask",)
            doc = f"{tool} {' '.join(actions_by_tool.get(tool, []))} {_tool_doc(tool)}"
        out.append(CapabilityCandidate(tool=tool, action=None, verbs=verbs, doc=doc))
    _CANDIDATES_CACHE = out
    _CANDIDATES_GENERATION += 1
    return out


def _score(
    intent_tokens: Counter, candidate: CapabilityCandidate
) -> tuple[float, list[str]]:
    """Dependency-free lexical overlap score (see module docstring — pre-CPD fallback).

    Weighted count-overlap normalized by intent length, with a name-token bonus
    (a match on the tool's OWN name/action words counts double — those are the
    strongest routing signal, e.g. intent "search the graph" hitting
    ``graph_search``'s own name) plus a small name-COVERAGE tie-breaker: when
    two tools tie on overlap, the one whose ENTIRE name is accounted for by the
    matched terms ranks first (``graph_search`` over ``graph_search_synthesis``
    for intent "search the graph" — the extra unmatched ``synthesis`` token
    makes it the less precise name match).
    """
    if not intent_tokens:
        return 0.0, []
    name_tokens = set(_WORD_RE.findall(candidate.tool.lower()))
    doc_tokens = _tokenize(candidate.doc)
    matched: list[str] = []
    score = 0.0
    name_hits = 0
    for term, weight in intent_tokens.items():
        in_name = term in name_tokens
        in_doc = doc_tokens.get(term, 0) > 0
        if not (in_name or in_doc):
            continue
        matched.append(term)
        if in_name:
            name_hits += 1
        score += weight * (2.0 if in_name else 1.0)
    total_weight = sum(intent_tokens.values()) or 1
    base = score / total_weight
    coverage_bonus = (name_hits / len(name_tokens)) * 0.01 if name_tokens else 0.0
    return base + coverage_bonus, matched


def resolve_intent(
    verb: str | None,
    intent: str,
    *,
    hints: dict[str, Any] | None = None,
    top_k: int = 5,
) -> list[CapabilityCandidate]:
    """Rank candidate capabilities for ``intent`` under ``verb`` (``None`` = all verbs).

    An explicit ``hints["tool"]`` (or ``hints["_tool"]``) pins the resolution to
    that exact tool (score ``1.0``, ``matched_terms=["explicit tool hint"]``) —
    the structured escape hatch alongside ``load_tools`` (never cached — it's
    already O(1) and the hint is caller-specific).

    A ranked (non-pinned) resolution is served from the bounded resolution
    cache (CONCEPT:AU-ECO.mcp.intent-surface-resolution-cache) when the SAME
    ``(verb, normalized intent, hints, top_k)`` was already resolved under the
    CURRENT routing policy (candidate-table generation + reward epoch
    unchanged); otherwise it re-ranks, blending each candidate's learned
    outcome reward-EMA (CONCEPT:AU-ECO.mcp.intent-surface-outcome-learning) into its lexical score
    before caching the result.
    """
    hints = hints or {}
    pinned = hints.get("tool") or hints.get("_tool")
    candidates = _build_candidates()
    if pinned:
        for c in candidates:
            if c.tool == pinned:
                return [
                    CapabilityCandidate(
                        tool=c.tool,
                        action=hints.get("action") or c.action,
                        verbs=c.verbs,
                        doc=c.doc,
                        score=1.0,
                        matched_terms=["explicit tool hint"],
                    )
                ]
        return []

    cache_key = _cache_key(verb, intent, hints, top_k)
    cached = _RESOLUTION_CACHE.get(cache_key)
    if cached is not None:
        _RESOLUTION_CACHE.move_to_end(cache_key)
        return list(cached)

    intent_tokens = _tokenize(intent)
    pool = candidates if verb is None else [c for c in candidates if verb in c.verbs]
    router = _outcome_router()
    ranked: list[CapabilityCandidate] = []
    for c in pool:
        score, matched = _score(intent_tokens, c)
        task_class = verb if verb is not None else (c.verbs[0] if c.verbs else "ask")
        reward = router.reward_of(task_class, c.capability_id)
        if reward != 0.5:
            score += _LEARNED_REWARD_WEIGHT * (reward - 0.5)
        ranked.append(
            CapabilityCandidate(
                tool=c.tool,
                action=c.action,
                verbs=c.verbs,
                doc=c.doc,
                score=score,
                matched_terms=matched,
            )
        )
    ranked.sort(key=lambda c: (c.score, c.tool), reverse=True)
    result = ranked[:top_k]

    _RESOLUTION_CACHE[cache_key] = result
    _RESOLUTION_CACHE.move_to_end(cache_key)
    while len(_RESOLUTION_CACHE) > _RESOLUTION_CACHE_MAX:
        _RESOLUTION_CACHE.popitem(last=False)
    return list(result)


def _pick_action(tool: str, intent: str) -> str | None:
    """Best-matching action for a multi-action tool, when the caller pinned none."""
    actions = _actions_by_tool().get(tool)
    if not actions:
        return None
    intent_tokens = _tokenize(intent)
    best, best_score = None, -1.0
    for action in actions:
        score, _ = _score(
            intent_tokens,
            CapabilityCandidate(
                tool=tool, action=action, verbs=(), doc=action.replace("_", " ")
            ),
        )
        if score > best_score:
            best, best_score = action, score
    return best


async def dispatch_intent(
    verb: str,
    intent: str,
    *,
    hints: dict[str, Any] | None = None,
    execute: bool = True,
    top_k: int = 5,
) -> dict[str, Any]:
    """Resolve ``intent`` under ``verb`` and dispatch the winner via ``_execute_tool``.

    Returns ``{"result", "routing", "executed"}`` (or ``{"error", "routing"}`` on
    a dispatch failure) — the SAME shape whichever of the six verbs called it, so
    a caller never needs verb-specific parsing. ``routing`` carries the "why"
    (a CPD-backed justification when the winning tool has a generated CPD
    entry, else the pre-CPD lexical one) PLUS ``calibrated_outcome_reward``
    (CONCEPT:AU-ECO.mcp.intent-surface-outcome-learning) — the learned reward-EMA for THIS
    ``verb:tool`` pairing at the moment of ranking. Every execution attempt
    (success or failure) feeds its outcome back into that same EMA so a later
    resolution under the same verb reflects real performance.
    """
    raw_hints = dict(hints or {})
    explicit_action = raw_hints.get("action")
    call_kwargs = {
        k: v for k, v in raw_hints.items() if k not in ("tool", "_tool", "action")
    }

    candidates = resolve_intent(verb, intent, hints=raw_hints, top_k=top_k)
    if not candidates:
        return {
            "error": f"No graph-os capability matched verb={verb!r}.",
            "routing": {"verb": verb, "intent": intent, "candidates": []},
        }

    top = candidates[0]
    chosen_tool = top.tool
    chosen_action = explicit_action or top.action or _pick_action(chosen_tool, intent)

    text_param = _PRIMARY_TEXT_PARAM.get(chosen_tool)
    fell_back = False
    if text_param and text_param not in call_kwargs:
        call_kwargs[text_param] = intent
    elif not call_kwargs and chosen_tool not in _PRIMARY_TEXT_PARAM and verb == "ask":
        # No structured hints AND the winning tool has no known free-text param:
        # fall back to the NL planner rather than dispatch a call we know is
        # missing required args (CONCEPT:AU-KG.query.ask-gateway-rest-twin).
        fell_back = True
        chosen_tool = _ASK_FALLBACK_TOOL
        chosen_action = None
        call_kwargs = {_PRIMARY_TEXT_PARAM[_ASK_FALLBACK_TOOL]: intent}

    if chosen_action is not None:
        call_kwargs.setdefault("action", chosen_action)

    routing: dict[str, Any] = {
        "verb": verb,
        "intent": intent,
        "chosen_tool": chosen_tool,
        "action": chosen_action,
        "score": round(top.score, 4),
        "matched_terms": top.matched_terms,
        "fell_back_to_nl_planner": fell_back,
        "why": (
            f"'{top.tool}' best matched the {verb!r} intent on terms {top.matched_terms!r}"
            if top.matched_terms
            else f"'{top.tool}' is the only/first candidate registered for verb {verb!r}"
        )
        + (
            f"; dispatched via '{_ASK_FALLBACK_TOOL}' (NL planner) — no structured hints for '{top.tool}'."
            if fell_back
            else "."
        ),
        "alternatives": [
            {"tool": c.tool, "action": c.action, "score": round(c.score, 4)}
            for c in candidates[1:]
        ],
        "capability_source": (
            "graph-os Capability Power Descriptor (docs/capabilities-power.json — "
            "does/examples/intent_verbs ranking, CONCEPT:AU-ECO.mcp.intent-surface-cpd-ranking)"
            if chosen_tool in _load_cpds_safe()
            else "graph-os local capability index (pre-CPD lexical fallback for "
            f"{chosen_tool!r} — no generated CPD entry yet)"
        ),
        "calibrated_outcome_reward": round(
            _outcome_router().reward_of(verb, chosen_tool), 4
        ),
    }

    if not execute:
        return {"routing": routing, "executed": False}

    try:
        result = await kg_server._execute_tool(chosen_tool, **call_kwargs)
    except Exception as e:  # noqa: BLE001 — surface as a structured routing failure, not a 500
        _record_dispatch_outcome(verb, chosen_tool, success=False)
        return {"routing": routing, "executed": False, "error": str(e)}
    _record_dispatch_outcome(verb, chosen_tool, success=True)
    return {"result": result, "routing": routing, "executed": True}


async def _find_capability(mcp: Any, intent: str, top_k: int = 8) -> dict[str, Any]:
    """``find`` — capability discovery across every verb, plus a best-effort fleet-wide search.

    ``mcp`` is the live FastMCP server (closure-captured by the caller,
    :func:`register_intent_tools`) — the fleet multiplexer, when attached
    (:func:`agent_utilities.mcp.multiplexer.attach_fleet_loader`), is stashed on
    it as ``mcp._fleet_mux`` so this can widen the search fleet-wide without a
    second multiplexer instance. Absent (embedded/headless builds) it degrades
    to local-only results — never an error.
    """
    local = resolve_intent(None, intent, top_k=top_k)
    payload: dict[str, Any] = {
        "query": intent,
        "count": len(local),
        "results": [
            {
                "tool": c.tool,
                "action": c.action,
                "verbs": list(c.verbs),
                "score": round(c.score, 4),
                "matched_terms": c.matched_terms,
                "how_to_call": (
                    f"load_tools(tools=['{c.tool}']) then call it directly, or "
                    f"call the '{c.verbs[0]}' intent verb with this same wording "
                    "(hints={'tool': '" + c.tool + "'} to pin it)."
                ),
            }
            for c in local
        ],
    }
    try:
        from agent_utilities.mcp import multiplexer as _mux_mod

        mux = getattr(mcp, "_fleet_mux", None)
        if mux is not None:
            loaded = mux.session_loaded(_mux_mod._session_key())
            discovery = await mux.discover_tools(intent, top_k=top_k, loaded=loaded)
            payload["fleet_results"] = discovery.get("results", [])
            payload["fleet_unavailable"] = discovery.get("unavailable", {})
    except Exception:  # noqa: BLE001 — fleet search is best-effort; local results always stand
        pass
    return payload


#: ``manage`` hints ``{"action": ...}`` values that route to the tool-lifecycle
#: (load/unload) core instead of the granular-tool resolver (CONCEPT:AU-ECO.mcp.intent-surface-tool-lifecycle) —
#: responsible tool usage is a MANAGE concern (the same verb that owns
#: configure/tenants/lifecycle), not a new seventh verb.
_RECLAIM_ACTIONS = frozenset({"unload", "reclaim", "load"})


async def _manage_lifecycle(mcp: Any, hints: dict[str, Any]) -> dict[str, Any] | None:
    """Handle a ``manage`` lifecycle action (load/unload/reclaim) directly.

    Returns ``None`` when ``hints`` carries no lifecycle action — the caller
    then falls through to the normal capability resolver. This is how
    "responsible tool usage" (load -> use -> unload) is reachable from the
    intent surface without a seventh verb: ``manage`` already owns
    configure/tenants/lifecycle, and reclaiming context IS a lifecycle op.
    """
    action = str(hints.get("action") or "").strip().lower()
    if action not in _RECLAIM_ACTIONS:
        return None
    mux = getattr(mcp, "_fleet_mux", None)
    if mux is None:
        return {
            "error": "No fleet multiplexer attached (embedded/headless build) — "
            "load/unload lifecycle needs a directly-served graph-os process."
        }
    from agent_utilities.mcp.multiplexer import load_session_tools, unload_session_tools

    tools = hints.get("tools")
    servers = hints.get("servers")
    if action == "load":
        return await load_session_tools(
            mcp,
            mux,
            tools=tools,
            servers=servers,
            auto_unload=bool(hints.get("auto_unload", False)),
        )
    toolsets = hints.get("toolsets")
    return await unload_session_tools(
        mcp, mux, tools=tools, servers=servers, toolsets=toolsets
    )


def register_intent_tools(mcp: Any) -> list[str]:
    """Register the six intent-verb tools (CONCEPT:AU-ECO.mcp.intent-surface-condensed-collapse).

    Called by ``kg_server._build_server`` only when ``MCP_TOOL_MODE=="intent"``
    (the intent verbs are an ADDITIONAL surface, opt-in via that profile — the
    default ``condensed`` mode is completely unaffected). Each tool is a thin
    wrapper over :func:`dispatch_intent`/:func:`_find_capability`; every one of
    them also gets a REST twin (see ``kg_server.ACTION_TOOL_ROUTES``) and a
    ``REGISTERED_TOOLS`` entry so ``_execute_tool`` — the same core the
    granular tools use — dispatches it identically from MCP or REST.
    """
    registered: list[str] = []

    def _intent_tool(verb: str):
        async def _tool(
            intent: str = Field(description=f"Natural-language {verb} intent."),
            hints_json: str = Field(
                default="{}",
                description=(
                    "Optional JSON object of structured args forwarded to the "
                    'resolved tool (e.g. {"node_id": "..."} for a write, or '
                    '{"tool": "graph_write"} to pin the exact tool).'
                ),
            ),
            execute: bool = Field(
                default=True,
                description="When false, return only the routing decision (preview/dry-run).",
            ),
        ) -> str:
            hints = json.loads(hints_json) if hints_json else {}
            if verb == "manage":
                lifecycle = await _manage_lifecycle(mcp, hints)
                if lifecycle is not None:
                    return json.dumps({"lifecycle": lifecycle}, default=str)
            result = await dispatch_intent(verb, intent, hints=hints, execute=execute)
            return json.dumps(result, default=str)

        return _tool

    verb_descriptions = {
        "ask": (
            "Ask the Knowledge Graph a natural-language READ question. Resolves to the best "
            "granular read tool (query/search/analyze/explain/nl_query/ask_data/code_context/"
            "reach/table/promql/federated_search/...) and returns its result PLUS the routing "
            'justification (which tool, why, alternatives). Pass hints_json={"tool": "..."} '
            "to pin an exact tool instead of letting the resolver choose."
        ),
        "find": (
            "Discover the graph-os capability (or fleet-wide MCP tool) that matches a "
            "natural-language description of a task — the generalized 'what can do X?' search "
            "across ALL verbs (not just reads). Returns ranked candidates with how to call each "
            "one (load_tools, or the matching intent verb pinned to that tool)."
        ),
        "write": (
            "Perform a natural-language WRITE/ingest intent. Resolves to add/write/ingest/"
            "writeback/etl/source_sync/... Structured fields (node ids, props, connector "
            "config, ...) go in hints_json — the resolver picks WHICH tool, hints_json supplies "
            "what that tool needs, exactly as calling it directly would."
        ),
        "act": (
            "Perform a natural-language ACT/execute intent. Resolves to orchestrate/loops/"
            "goals/execute_agent/workflow/schedule/... Put the agent/workflow name, args, etc. "
            "in hints_json."
        ),
        "manage": (
            "Perform a natural-language MANAGE/configure intent. Resolves to configure/"
            "tenants/lifecycle/resharding/secret/schedules/permissions/... Put the config key, "
            "scope, etc. in hints_json. RESPONSIBLE TOOL USAGE (CONCEPT:AU-ECO.mcp.intent-surface-tool-lifecycle): "
            'pass hints_json={"action": "unload", "tools": [...]} (or "servers"/'
            '"toolsets") to reclaim context from previously loaded granular tools, or '
            '{"action": "load", "tools": [...], "auto_unload": true} to pull one in '
            "for a single one-shot use — it auto-retracts after its next call so long "
            "sessions don't accumulate tool schemas. Nothing is lost; load again anytime."
        ),
        "why": (
            "Ask WHY — explain a belief/decision/change. Resolves to explain/evaluate/observe/"
            "causal-analysis/... and returns the explanation PLUS the routing justification "
            "for this dispatch itself (proof-carrying capability dispatch)."
        ),
    }

    for verb in ("ask", "write", "act", "manage", "why"):
        fn = _intent_tool(verb)
        fn.__name__ = verb
        fn.__doc__ = verb_descriptions[verb]
        mcp.tool(name=verb, tags={"intent"}, description=verb_descriptions[verb])(fn)
        kg_server.REGISTERED_TOOLS[verb] = fn
        # REST twin (CONCEPT:AU-ECO.mcp.two-surfaces-mcp-rest) — the generic
        # ACTION_TOOL_ROUTES loop in _mount_rest_routes wires this automatically,
        # exactly like nl_query/ask_data (query_tools.py) do.
        kg_server.ACTION_TOOL_ROUTES[verb] = f"/intent/{verb}"
        registered.append(verb)

    async def _find_tool(
        intent: str = Field(
            description="Natural-language description of the capability/task."
        ),
        top_k: int = Field(default=8, description="Max ranked candidates to return."),
    ) -> str:
        payload = await _find_capability(mcp, intent, top_k=top_k)
        return json.dumps(payload, default=str)

    _find_tool.__name__ = "find"
    _find_tool.__doc__ = verb_descriptions["find"]
    mcp.tool(name="find", tags={"intent"}, description=verb_descriptions["find"])(
        _find_tool
    )
    kg_server.REGISTERED_TOOLS["find"] = _find_tool
    kg_server.ACTION_TOOL_ROUTES["find"] = "/intent/find"
    registered.append("find")

    return registered
