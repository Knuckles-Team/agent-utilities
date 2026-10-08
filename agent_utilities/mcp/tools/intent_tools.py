"""The graph-os intent surface (CONCEPT:AU-ECO.mcp.intent-surface-condensed-collapse).

Six intent verbs (``ask``/``find``/``write``/``act``/``manage``/``why``) are
the default model-facing GraphOS API. The packaged Capability Power Descriptor
(CPD) set is the required routing and operation-safety authority. Missing CPDs
fail closed so a newly registered capability cannot silently bypass verb,
scope, effect, or approval classification.

Dispatch is proof-carrying. Every response includes the ranked evidence,
effective operation, policy classification, preview plan, and result
provenance. ``ask`` is read-only; a tool pin cannot change that policy.
Non-read verbs preview by default, ambiguous mutations do not execute, and
an approval-required (destructive, or a non-``auto`` ``approval_class``)
operation only executes after the caller's session approved that exact
operation through the separately governed ``manage(action="approve")``
(BUG-040 — see :func:`_operation_approved`); a tool pin never bypasses it.

The six verbs are graph-os's whole MCP surface: each takes the ecosystem's
condensed contract (:mod:`agent_utilities.mcp.intent_contract`) — ``action``
(an operation id of the generated manifest, or ``describe``), ``params``,
optional natural-language ``intent`` and ``execute``. ``find`` also reaches the
fleet catalog and ``act`` also calls fleet tools (``fleet.call``) and the
host's native operations, so no separate fleet meta-tool exists.

Outcome learning accepts only the observed result of an unpinned,
unambiguous, policy-authorized execution. Caller-supplied feedback is rejected.
Both reward keys and resolution-cache keys are partitioned by an opaque digest
of the verified tenant and policy revision, preventing cross-tenant or stale-
policy influence. The shared :class:`OutcomeRouter` remains the sole reward-EMA
mechanism; no second learner is introduced.
"""

from __future__ import annotations

import inspect
import json
import logging
import re
import time
from collections import Counter, OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from agent_utilities.knowledge_graph.retrieval.capability_context import load_cpds
from agent_utilities.mcp import kg_server
from agent_utilities.mcp.intent_contract import (
    DESCRIBE_ACTION,
    READ_ACTION_PREFIXES,
    operation_id,
)
from agent_utilities.mcp.optional_tool_features import OPTIONAL_TOOL_FEATURES
from agent_utilities.mcp.tool_specs import INTENT_VERBS, READ_ONLY_ACTIONS, TOOL_VERBS
from agent_utilities.security.error_surface import public_error_payload
from agent_utilities.security.persistence_privacy import persistence_reference
from agent_utilities.security.threat_defense_engine import PromptInjectionScanner

logger = logging.getLogger(__name__)

__all__ = [
    "INTENT_VERBS",
    "TOOL_VERBS",
    "CapabilityCandidate",
    "resolve_intent",
    "dispatch_intent",
    "register_intent_tools",
]

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
    "graph_code": "query",
    "graph_explain": "query",
    "graph_evaluate": "query",
    "graph_observe": "query",
    "graph_orchestrate": "task",
}

#: Safe universal fallback for ``ask`` when the top-ranked candidate needs
#: structured params the caller didn't supply — the engine's own NL planner
#: (CONCEPT:AU-KG.query.ask-gateway-rest-twin) always accepts raw text.
_ASK_FALLBACK_TOOL = "nl_query"

_DISPATCH_VERBS = frozenset({"ask", "write", "act", "manage", "why"})
_READ_ONLY_VERBS = frozenset({"ask", "why"})
_NON_READ_VERBS = frozenset({"write", "act", "manage"})
_AMBIGUITY_MARGIN = 0.05
_CALLER_OUTCOME_FIELDS = frozenset(
    {
        "calibrated_outcome_reward",
        "dispatch_outcome",
        "outcome_reward",
        "routing_feedback",
        "routing_reward",
    }
)
_CONTROL_HINT_FIELDS = frozenset({"tool", "_tool", "action", "plan_ref"})
#: Kept at the intent boundary rather than adding duplicate public parameters to
#: the underlying tool.  A delegation target is an ``agent_name`` regardless of
#: whether a caller describes it as an agent or its backing MCP server.
_DOCUMENTED_HINT_ALIASES: dict[str, dict[str, str]] = {
    "graph_orchestrate": {"agent": "agent_name", "server": "agent_name"},
}
_DESTRUCTIVE_TERMS = frozenset(
    {
        "clear",
        "deactivate",
        "delete",
        "destroy",
        "drop",
        "evict",
        "invalidate",
        "prune",
        "purge",
        "remove",
        "revoke",
        "terminate",
        "uninstall",
        "unref",
        "wipe",
    }
)

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
    #: Structural signature of a capability-DELEGATION façade (CONCEPT:AU-ECO.mcp.
    #: intent-surface-delegation-shape): its CPD declares a ``skill_name`` input
    #: parameter. Derived from the packaged CPD's ``typed_io.input_params``, not
    #: from the tool's literal name — any future delegation façade that accepts
    #: an exact ingested-skill name qualifies automatically, with zero new
    #: per-tool special-casing.
    accepts_skill_name: bool = False

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
_RESOLUTION_CACHE: OrderedDict[tuple[Any, ...], list[CapabilityCandidate]] = (
    OrderedDict()
)
_RESOLUTION_CACHE_MAX = 256

#: Short-lived, process-local previews keyed by their opaque plan reference.
#: The cache makes the documented ``preview -> resubmit plan_ref`` contract
#: executable without forcing clients to replay every original hint. Entries
#: remain bound to the verb, intent, tenant/policy partition, and exact
#: argument digest; a restart or expiry intentionally requires a new preview.
_PREVIEW_PLAN_CACHE: OrderedDict[str, _PreviewPlan] = OrderedDict()
_PREVIEW_PLAN_CACHE_MAX = 256
_PREVIEW_PLAN_TTL_SECONDS = 600.0

#: Soft weight of the learned reward-EMA blend into the lexical score (mirrors
#: ``CapabilityIndex.designate``'s own ``reward_weight`` default) — early on
#: (every candidate at the neutral 0.5 prior) the lexical ranking is
#: untouched; as outcomes accumulate a candidate's blended score rises or
#: sinks with its real success rate under that verb.
_LEARNED_REWARD_WEIGHT = 0.2

#: Lazily constructed shared learner (CONCEPT:AU-ECO.mcp.intent-surface-outcome-learning).
_OUTCOME_ROUTER: Any = None


@dataclass(frozen=True)
class _PreviewPlan:
    """The minimum private state required to replay a reviewed intent plan."""

    created_at: float
    verb: str
    intent_ref: str
    outcome_scope_ref: str | None
    hints: dict[str, Any]


def _outcome_router() -> Any:
    """The shared :class:`OutcomeRouter` for intent-surface dispatch outcomes.

    CONCEPT:AU-ECO.mcp.intent-surface-outcome-learning. ONE learner, reused — the same
    durable-bandit mechanism (``CapabilityIndex.record_outcome``/``reward_of``)
    every other outcome-learned choice in this codebase already shares
    (``ReasonerRouter``, ``variant_pool.evolve_profile``). The task class is
    an opaque tenant+policy reference followed by the verb; raw identity and
    policy values never enter the learner.
    """
    global _OUTCOME_ROUTER
    if _OUTCOME_ROUTER is None:
        from agent_utilities.orchestration.outcome_router import OutcomeRouter

        _OUTCOME_ROUTER = OutcomeRouter(namespace="intent_surface")
    return _OUTCOME_ROUTER


def _outcome_scope_ref() -> str | None:
    """Return an opaque verified authority partition for routing state.

    An absent ambient :class:`GraphSession` means there is no verified
    authority from which to derive a learning partition. Resolution remains
    available at the neutral prior, but no outcome may be learned.
    """

    from agent_utilities.knowledge_graph.core.session import current_session

    session = current_session()
    if session is None:
        return None
    tenant_ref = persistence_reference("tenant", session.tenant)
    authority_policy = json.dumps(
        {
            "policy_version": str(session.policy_version),
            "audience": str(session.audience),
            "scopes": sorted(str(scope) for scope in session.scopes),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return persistence_reference(
        "intent_policy", authority_policy, namespace=tenant_ref
    )


def _reward_task_class(verb: str, scope_ref: str) -> str:
    return f"{scope_ref}:{verb}"


def _record_dispatch_outcome(
    scope_ref: str, verb: str, tool: str, *, success: bool
) -> None:
    """Record one trusted execution result in its tenant/policy partition.

    CONCEPT:AU-ECO.mcp.intent-surface-outcome-learning. Bumps :data:`_REWARD_EPOCH` so any
    cached resolution that could have used this outcome is invalidated on its
    next lookup rather than served stale.
    """
    global _REWARD_EPOCH
    try:
        _outcome_router().record(
            _reward_task_class(verb, scope_ref), tool, 1.0 if success else 0.0
        )
    finally:
        _REWARD_EPOCH += 1


def _normalize_intent(intent: str) -> str:
    """Whitespace/case-fold an intent string for resolution-cache keying."""
    return " ".join(str(intent or "").strip().lower().split())


def _cache_key(
    verb: str | None,
    intent: str,
    hints: dict[str, Any],
    top_k: int,
    outcome_scope_ref: str | None,
) -> tuple[Any, ...]:
    """The resolution-cache key (CONCEPT:AU-ECO.mcp.intent-surface-resolution-cache).

    Intent and hints are represented by non-reversible references so the LRU
    never retains prompt text, credentials, personal data, endpoints, or local
    paths. The opaque tenant+policy partition prevents a cached ranking from
    crossing an authorization-policy boundary.
    """
    normalized = _normalize_intent(intent)
    hints_json = json.dumps(
        hints or {}, sort_keys=True, default=str, separators=(",", ":")
    )
    return (
        verb,
        persistence_reference("intent", normalized),
        persistence_reference("intent_hints", hints_json),
        outcome_scope_ref or "unverified",
        int(top_k),
        _CANDIDATES_GENERATION,
        _REWARD_EPOCH,
    )


def _load_cpds_required() -> dict[str, dict[str, Any]]:
    """Return the packaged current CPD authority or fail closed."""

    cpds = load_cpds()
    if not cpds:
        raise RuntimeError("GraphOS capability descriptors are unavailable")
    return cpds


def _actions_by_tool() -> dict[str, list[str]]:
    global _ACTIONS_BY_TOOL_CACHE
    if _ACTIONS_BY_TOOL_CACHE is not None:
        return _ACTIONS_BY_TOOL_CACHE
    out: dict[str, list[str]] = {}
    from agent_utilities.mcp._graphos_action_manifest import GRAPHOS_ACTIONS

    for op in GRAPHOS_ACTIONS:
        action = op.get("action")
        if action is None:
            continue
        out.setdefault(op["tool"], []).append(action)
    _ACTIONS_BY_TOOL_CACHE = out
    return out


def _build_candidates(*, force: bool = False) -> list[CapabilityCandidate]:
    """Build one CPD-backed candidate per live granular tool.

    CONCEPT:AU-ECO.mcp.intent-surface-cpd-ranking. The packaged CPD set is
    mandatory: a registered tool without a descriptor fails closed instead of
    inheriting a guessed verb or effect policy. Candidate text comes from its
    current ``one_line``/``examples``/``does`` descriptor fields.

    Cached process-wide (tool registration happens once at server build); pass
    ``force=True`` in tests after monkeypatching ``REGISTERED_TOOLS``. Bumps
    :data:`_CANDIDATES_GENERATION` on every actual rebuild (CONCEPT:AU-ECO.mcp.intent-surface-resolution-cache).
    """
    global _CANDIDATES_CACHE, _CANDIDATES_GENERATION
    if _CANDIDATES_CACHE is not None and not force:
        return _CANDIDATES_CACHE
    kg_server.ensure_tools_registered()
    actions_by_tool = _actions_by_tool()
    cpds = _load_cpds_required()
    live_tools = sorted(set(kg_server.REGISTERED_TOOLS) - set(INTENT_VERBS))
    _require_cpd_coverage(live_tools, cpds)
    _require_verb_authority_coverage(live_tools)

    # Intent verbs have CPDs because they are first-class MCP/REST entry points,
    # but they can never be resolver targets: selecting ``ask`` from inside
    # ``ask`` would recursively dispatch intent routing instead of a capability.
    out = [
        _build_candidate_for_tool(tool, cpds[tool], actions_by_tool)
        for tool in live_tools
    ]
    _CANDIDATES_CACHE = out
    _CANDIDATES_GENERATION += 1
    return out


def _require_cpd_coverage(
    live_tools: list[str], cpds: dict[str, dict[str, Any]]
) -> None:
    """Helper for `_build_candidates`: fail closed on a registered tool without a CPD."""
    missing_cpds = sorted(set(live_tools) - set(cpds))
    if missing_cpds:
        raise RuntimeError(
            "GraphOS capability descriptors are missing for registered tools: "
            + ", ".join(missing_cpds)
        )


def _require_verb_authority_coverage(live_tools: list[str]) -> None:
    """Helper for `_build_candidates`: fail closed on a tool without verb authority."""
    missing_authority = sorted(set(live_tools) - set(TOOL_VERBS))
    if missing_authority:
        raise RuntimeError(
            "GraphOS intent-verb authority is missing registered tools: "
            + ", ".join(missing_authority)
        )


def _validate_tool_authority_verbs(tool: str, authority_verbs: tuple[str, ...]) -> None:
    """Helper for `_build_candidates`: authority_verbs must be a non-empty, unique subset of INTENT_VERBS."""
    if (
        not authority_verbs
        or len(set(authority_verbs)) != len(authority_verbs)
        or not set(authority_verbs) <= set(INTENT_VERBS)
    ):
        raise RuntimeError(
            f"GraphOS intent-verb authority is invalid for {tool}: {authority_verbs!r}"
        )


def _validate_packaged_verbs(
    tool: str, cpd: dict[str, Any], authority_verbs: tuple[str, ...]
) -> None:
    """Helper for `_build_candidates`: packaged CPD intent_verbs must match TOOL_VERBS exactly."""
    packaged_verbs = cpd.get("intent_verbs")
    if not isinstance(packaged_verbs, list) or not all(
        isinstance(verb, str) for verb in packaged_verbs
    ):
        raise RuntimeError(
            "GraphOS capability descriptor has an invalid intent_verbs "
            f"field for {tool}"
        )
    if tuple(packaged_verbs) != authority_verbs:
        raise RuntimeError(
            "GraphOS capability descriptor intent-verb drift for "
            f"{tool}: expected {list(authority_verbs)!r}, packaged {packaged_verbs!r}"
        )


def _build_candidate_for_tool(
    tool: str,
    cpd: dict[str, Any],
    actions_by_tool: dict[str, list[str]],
) -> CapabilityCandidate:
    """Helper for `_build_candidates`: validate one tool's CPD and build its candidate."""
    authority_verbs = TOOL_VERBS[tool]
    _validate_tool_authority_verbs(tool, authority_verbs)
    _validate_packaged_verbs(tool, cpd, authority_verbs)

    examples_text = " ".join(str(e) for e in (cpd.get("examples") or ()))
    does_text = " ".join(str(d.get("action", "")) for d in (cpd.get("does") or ()))
    doc = (
        f"{tool} {' '.join(actions_by_tool.get(tool, []))} "
        f"{cpd.get('one_line', '')} {examples_text} {does_text}"
    )
    input_params = (cpd.get("typed_io") or {}).get("input_params") or ()
    accepts_skill_name = any(
        isinstance(p, dict) and p.get("name") == "skill_name" for p in input_params
    )
    return CapabilityCandidate(
        tool=tool,
        action=None,
        verbs=authority_verbs,
        doc=doc,
        accepts_skill_name=accepts_skill_name,
    )


def _score(
    intent_tokens: Counter, candidate: CapabilityCandidate
) -> tuple[float, list[str]]:
    """Dependency-free lexical overlap score over current CPD evidence.

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


#: One hyphenated slug (the KG's own ingested-skill naming convention, e.g.
#: ``servicenow-incident-management``) immediately adjacent to the literal
#: word "skill" — in either order, optionally introduced by "named"/"called"
#: or a colon. This is a STRUCTURAL fingerprint of "the caller is naming a
#: specific skill to delegate to", independent of what words make up the
#: slug (CONCEPT:AU-ECO.mcp.intent-surface-delegation-shape / D-INT-4). It
#: deliberately does NOT special-case any domain vocabulary ("incident",
#: "servicenow", ...) — a slug about ANY domain matches the same way, which
#: is exactly why naming the skill you want must stop working against you.
_SKILL_SLUG = r"[a-z][a-z0-9]*(?:-[a-z0-9]+){1,6}"
_SKILL_DELEGATION_RE = re.compile(
    rf"\b{_SKILL_SLUG}\s+skill\b"
    rf"|\bskill(?:\s+(?:named|called))?\s*[:\-]?\s+{_SKILL_SLUG}\b",
    re.IGNORECASE,
)

#: Flat score bonus applied to a delegation-façade candidate (one whose CPD
#: declares ``skill_name``, see :data:`CapabilityCandidate.accepts_skill_name`)
#: when :data:`_SKILL_DELEGATION_RE` matches the intent. Large enough to
#: dominate any lexical overlap the named skill's own words happen to create
#: with an unrelated tool (measured worst case pre-fix: 0.3383 for an
#: incident-analysis tool against a ServiceNow-incident SKILL delegation), a
#: score no ordinary multi-term lexical match realistically reaches.
_SKILL_DELEGATION_BONUS = 0.5


def _skill_delegation_bonus(intent: str, candidate: CapabilityCandidate) -> float:
    """Structural delegation-shape bonus — see :data:`_SKILL_DELEGATION_RE`."""

    if not candidate.accepts_skill_name:
        return 0.0
    if not _SKILL_DELEGATION_RE.search(str(intent or "")):
        return 0.0
    return _SKILL_DELEGATION_BONUS


def _declared_action_phrase_bonus(intent: str, tool: str) -> float:
    """Prefer the capability that declares an exact multi-word action phrase."""

    words = " ".join(_WORD_RE.findall(str(intent or "").lower()))
    if not words:
        return 0.0
    longest = 0
    for action in _actions_by_tool().get(tool, ()):
        action_words = _WORD_RE.findall(action.lower())
        if len(action_words) < 2:
            continue
        if f" {' '.join(action_words)} " in f" {words} ":
            longest = max(longest, len(action_words))
    return min(0.05 * longest, 0.15)


def resolve_intent(
    verb: str | None,
    intent: str,
    *,
    hints: dict[str, Any] | None = None,
    top_k: int = 5,
) -> list[CapabilityCandidate]:
    """Rank candidate capabilities for ``intent`` under ``verb`` (``None`` = all verbs).

    An explicit ``hints["tool"]`` (or ``hints["_tool"]``) pins resolution only
    when that tool is authorized for the requested verb. A pin can remove
    ranking ambiguity; it cannot elevate ``ask`` into a mutation policy.

    A ranked (non-pinned) resolution is served from the bounded resolution
    cache (CONCEPT:AU-ECO.mcp.intent-surface-resolution-cache) when the SAME
    ``(verb, normalized intent, hints, top_k)`` was already resolved under the
    CURRENT routing policy (candidate-table generation + reward epoch
    unchanged); otherwise it re-ranks, blending each candidate's learned
    outcome reward-EMA (CONCEPT:AU-ECO.mcp.intent-surface-outcome-learning) into its lexical score
    before caching the result.
    """
    if verb is not None and verb not in INTENT_VERBS:
        return []
    top_k = max(1, min(int(top_k), 20))
    hints = hints or {}
    pinned = hints.get("tool") or hints.get("_tool")
    candidates = _build_candidates()
    if pinned:
        return _pinned_resolution(candidates, verb, pinned, hints)

    outcome_scope_ref = _outcome_scope_ref()
    cache_key = _cache_key(verb, intent, hints, top_k, outcome_scope_ref)
    cached = _RESOLUTION_CACHE.get(cache_key)
    if cached is not None:
        _RESOLUTION_CACHE.move_to_end(cache_key)
        return list(cached)

    intent_tokens = _tokenize(intent)
    explicit_action = hints.get("action")
    pool = _resolution_pool(candidates, verb, explicit_action)
    router = _outcome_router() if outcome_scope_ref is not None else None
    cpds = _load_cpds_required() if explicit_action is not None else {}
    ctx = _RankingContext(
        intent=intent,
        intent_tokens=intent_tokens,
        verb=verb,
        explicit_action=explicit_action,
        cpds=cpds,
        router=router,
        outcome_scope_ref=outcome_scope_ref,
    )
    ranked = [_rank_candidate(c, ctx) for c in pool]
    ranked.sort(key=lambda c: (c.score, c.tool), reverse=True)
    result = ranked[:top_k]

    _cache_resolution(cache_key, result)
    return list(result)


def _cache_resolution(
    cache_key: tuple[Any, ...], result: list[CapabilityCandidate]
) -> None:
    """Helper for `resolve_intent`: store `result`, evicting the LRU entry over capacity."""
    _RESOLUTION_CACHE[cache_key] = result
    _RESOLUTION_CACHE.move_to_end(cache_key)
    while len(_RESOLUTION_CACHE) > _RESOLUTION_CACHE_MAX:
        _RESOLUTION_CACHE.popitem(last=False)


def _pinned_resolution(
    candidates: list[CapabilityCandidate],
    verb: str | None,
    pinned: str,
    hints: dict[str, Any],
) -> list[CapabilityCandidate]:
    """Helper for `resolve_intent`: resolve a hints["tool"]-pinned request.

    Empty list if `pinned` names no candidate authorized for `verb`. ``act``
    is authorized for an operation none of the tool's declared verbs can
    classify (:func:`_act_fallback`); it previews and executes it as a mutation.
    """
    action = hints.get("action")
    for c in candidates:
        authorized = (
            verb is None
            or verb in c.verbs
            or (verb == "act" and _act_fallback(c.tool, action or c.action))
        )
        if c.tool == pinned and authorized:
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


def _resolution_pool(
    candidates: list[CapabilityCandidate],
    verb: str | None,
    explicit_action: Any,
) -> list[CapabilityCandidate]:
    """Helper for `resolve_intent`: filter candidates by verb, then by explicit action."""
    pool = candidates if verb is None else [c for c in candidates if verb in c.verbs]
    if explicit_action is not None:
        actions_by_tool = _actions_by_tool()
        pool = [
            candidate
            for candidate in pool
            if explicit_action in actions_by_tool.get(candidate.tool, ())
        ]
    return pool


@dataclass
class _RankingContext:
    """Helper for `resolve_intent`/`_rank_candidate`: bundles the per-call scoring

    context so `_rank_candidate` stays within the 7-parameter cap.
    """

    intent: str
    intent_tokens: Counter
    verb: str | None
    explicit_action: Any
    cpds: dict[str, dict[str, Any]]
    router: Any
    outcome_scope_ref: str | None


def _rank_candidate(
    c: CapabilityCandidate, ctx: _RankingContext
) -> CapabilityCandidate:
    """Helper for `resolve_intent`: score + build the ranked candidate for one pool entry."""
    scoring_candidate = c
    if ctx.explicit_action is not None:
        cpd = ctx.cpds[c.tool]
        scoring_candidate = CapabilityCandidate(
            tool=c.tool,
            action=str(ctx.explicit_action),
            verbs=c.verbs,
            doc=f"{c.tool} {ctx.explicit_action} {cpd.get('one_line', '')}",
        )
    score, matched = _score(ctx.intent_tokens, scoring_candidate)
    if ctx.explicit_action is None:
        score += _declared_action_phrase_bonus(ctx.intent, c.tool)
    score += _skill_delegation_bonus(ctx.intent, c)
    task_verb = ctx.verb if ctx.verb is not None else c.verbs[0]
    reward = (
        ctx.router.reward_of(
            _reward_task_class(task_verb, ctx.outcome_scope_ref), c.capability_id
        )
        if ctx.router is not None and ctx.outcome_scope_ref is not None
        else 0.5
    )
    if reward != 0.5:
        score += _LEARNED_REWARD_WEIGHT * (reward - 0.5)
    return CapabilityCandidate(
        tool=c.tool,
        action=(
            str(ctx.explicit_action) if ctx.explicit_action is not None else c.action
        ),
        verbs=c.verbs,
        doc=c.doc,
        score=score,
        matched_terms=matched,
        accepts_skill_name=c.accepts_skill_name,
    )


def _rank_actions(tool: str, intent: str) -> list[tuple[str, float]]:
    """Return actions ranked by current lexical intent evidence."""

    actions = _actions_by_tool().get(tool)
    if not actions:
        return []
    intent_tokens = _tokenize(intent)
    ranked: list[tuple[str, float]] = []
    for action in actions:
        score, _ = _score(
            intent_tokens,
            CapabilityCandidate(
                tool=tool, action=action, verbs=(), doc=action.replace("_", " ")
            ),
        )
        ranked.append((action, score))
    return sorted(ranked, key=lambda item: (item[1], item[0]), reverse=True)


def _literal_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    normalized = str(value or "").strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return None


def _operation_record(cpd: dict[str, Any], action: str | None) -> dict[str, Any] | None:
    operations = [op for op in cpd.get("does", ()) if isinstance(op, dict)]
    if action is not None:
        return next((op for op in operations if op.get("action") == action), None)
    if len(operations) == 1:
        return operations[0]
    return None


def _operation_is_destructive(tool: str, action: str | None) -> bool:
    words = set(_WORD_RE.findall(f"{tool} {action or ''}".lower()))
    return not words.isdisjoint(_DESTRUCTIVE_TERMS)


def _tool_accepts_argument(tool: str, argument: str) -> bool:
    function = kg_server.REGISTERED_TOOLS.get(tool)
    if function is None:
        return False
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return False
    return argument in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def _normalize_documented_hint_aliases(hints: dict[str, Any]) -> dict[str, Any]:
    """Normalize only documented aliases for an explicitly pinned tool.

    Conflicting names are denied rather than silently choosing a target.  The
    normalized form is what preview storage and plan references bind, so a
    plan-ref-only replay remains exact while callers may use either spelling.
    """

    tool = hints.get("tool") or hints.get("_tool")
    aliases = _DOCUMENTED_HINT_ALIASES.get(str(tool or ""), {})
    normalized = dict(hints)
    for alias, canonical in aliases.items():
        if alias not in normalized:
            continue
        alias_value = normalized.pop(alias)
        if canonical in normalized and normalized[canonical] != alias_value:
            raise ValueError(
                f"Conflicting intent hints {alias!r} and {canonical!r}; "
                f"use only {canonical!r}."
            )
        normalized[canonical] = alias_value
    return normalized


def _unsupported_hint_arguments(tool: str, call_kwargs: dict[str, Any]) -> list[str]:
    """Return unsupported direct-call arguments for a selected tool.

    A ``**kwargs`` tool explicitly owns its open-ended schema.  All other
    tools are closed at this façade, avoiding a raw Python ``TypeError`` after
    a reviewed plan has been produced.
    """

    function = kg_server.REGISTERED_TOOLS.get(tool)
    if function is None:
        return sorted(call_kwargs)
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        # An opaque callable has no inspectable public input contract.
        return sorted(call_kwargs)
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return []
    accepted = {
        parameter.name
        for parameter in parameters
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return sorted(set(call_kwargs) - accepted)


def _hint_argument_error(tool: str, unsupported: list[str]) -> str:
    """Give callers a stable correction path without exposing raw exceptions."""

    function = kg_server.REGISTERED_TOOLS.get(tool)
    try:
        parameters = inspect.signature(function).parameters.values() if function else ()
        accepted = sorted(
            parameter.name
            for parameter in parameters
            if parameter.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        )
    except (TypeError, ValueError):
        accepted = []
    names = ", ".join(repr(name) for name in unsupported)
    supported = ", ".join(accepted) or "the tool's documented schema"
    return (
        f"Unsupported intent hint argument(s) for {tool!r}: {names}. "
        f"Use documented parameters: {supported}."
    )


#: Action-name prefixes treated as read-only when nothing more authoritative
#: (a declared ``mutates``, the destructive-terms check, or the reviewed
#: READ_ONLY_ACTIONS allowlist) has already classified the operation. Used
#: only by :func:`_resolve_mutates`.
_READ_ACTION_PREFIXES = READ_ACTION_PREFIXES


def _mutates_from_declaration(
    declared_mutation: bool | None, destructive: bool
) -> bool | None:
    """Helper for `_resolve_mutates`: declared/destructive authority (None = inconclusive)."""
    if declared_mutation is not None:
        return declared_mutation
    if destructive:
        return True
    return None


def _mutates_from_action_policy(
    tool: str, action: str | None, action_is_declared_read: bool
) -> bool | None:
    """Helper for `_resolve_mutates`: the READ_ONLY_ACTIONS allowlist policy."""
    if action_is_declared_read:
        return False
    if action is not None and tool in READ_ONLY_ACTIONS:
        # The reviewed allowlist is the action policy for a mixed surface:
        # anything not explicitly declared read-only is non-read.
        return True
    return None


def _mutates_from_verb_and_prefix(
    verb: str, action: str | None, action_prefix: str
) -> bool | None:
    """Helper for `_resolve_mutates`: name/verb-based inference fallback."""
    if action_prefix in _READ_ACTION_PREFIXES:
        return False
    if action is None and verb in _READ_ONLY_VERBS:
        return False
    if verb in _NON_READ_VERBS:
        return True
    return None


def _resolve_mutates(
    verb: str,
    tool: str,
    action: str | None,
    declared_mutation: bool | None,
    destructive: bool,
) -> bool | None:
    """Helper for `_operation_plan`: resolve the operation's mutation classification.

    Tries, in priority order: declared/destructive authority, the
    READ_ONLY_ACTIONS allowlist policy, then name/verb-based inference.
    """
    result = _mutates_from_declaration(declared_mutation, destructive)
    if result is not None:
        return result
    action_is_declared_read = action in READ_ONLY_ACTIONS.get(tool, frozenset())
    result = _mutates_from_action_policy(tool, action, action_is_declared_read)
    if result is not None:
        return result
    action_prefix = str(action or "").split("_", 1)[0]
    return _mutates_from_verb_and_prefix(verb, action, action_prefix)


def _execution_class_and_impact(
    destructive: bool, mutates: bool | None
) -> tuple[str, str]:
    """Helper for `_operation_plan`: (execution_class, impact_summary)."""
    if destructive:
        return "destructive", "May remove or irreversibly invalidate governed state."
    if mutates is True:
        return (
            "mutation",
            "Changes governed state within the selected capability scope.",
        )
    if mutates is False:
        return "read_only", "Reads or computes without a declared state mutation."
    return "unclassified", "Effect metadata is insufficient; execution fails closed."


def _approval_info(cpd: dict[str, Any], destructive: bool) -> dict[str, Any]:
    """Helper for `_operation_plan`: the ``approval`` block."""
    raw_policy = cpd.get("policy")
    policy = raw_policy if isinstance(raw_policy, dict) else {}
    approval_class = str(policy.get("approval_class") or "unclassified")
    approval_required = destructive or approval_class != "auto"
    return {
        "class": approval_class,
        "required": approval_required,
        "route": "exact_tool" if approval_required else "intent_policy",
    }


def _declared_flag(operation: dict[str, Any] | None, key: str) -> bool | None:
    """Helper for `_operation_plan`: a declared boolean CPD operation flag, or None."""
    return _literal_bool(operation.get(key) if operation is not None else None)


def _operation_cost_latency(cpd: dict[str, Any]) -> dict[str, Any]:
    """Helper for `_operation_metadata`: cost/latency, with fallbacks applied."""
    cost = cpd.get("cost") if isinstance(cpd.get("cost"), dict) else {}
    latency = cpd.get("latency") if isinstance(cpd.get("latency"), dict) else {}
    return {
        "cost": cost or {"estimate": "not_available"},
        "latency": latency or {"estimate": "not_available"},
    }


def _operation_impact_fields(
    cpd: dict[str, Any], operation: dict[str, Any] | None
) -> dict[str, Any]:
    """Helper for `_operation_metadata`: scopes/durability/transaction, with fallbacks applied."""
    scopes = sorted(str(scope) for scope in (cpd.get("scopes") or ()))
    durability = str(operation.get("durability") or "") if operation is not None else ""
    transaction = (
        str(operation.get("txn_participation") or "") if operation is not None else ""
    )
    return {
        "scopes": scopes,
        "durability": durability or "unclassified",
        "transaction": transaction or "unclassified",
    }


def _operation_metadata(
    cpd: dict[str, Any], operation: dict[str, Any] | None
) -> dict[str, Any]:
    """Helper for `_operation_plan`: cost/latency/scopes/durability/transaction, fallbacks applied."""
    return {
        **_operation_cost_latency(cpd),
        **_operation_impact_fields(cpd, operation),
    }


def _operation_plan(
    verb: str,
    tool: str,
    action: str | None,
    call_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Build a value-free preview from the current CPD operation authority."""

    cpd = _load_cpds_required()[tool]
    operation = _operation_record(cpd, action)
    declared_mutation = _declared_flag(operation, "mutates")
    destructive = _operation_is_destructive(tool, action)
    mutates = _resolve_mutates(verb, tool, action, declared_mutation, destructive)

    declared_idempotency = _declared_flag(operation, "idempotent")
    if declared_idempotency is None and mutates is False:
        declared_idempotency = True

    metadata = _operation_metadata(cpd, operation)
    execution_class, impact_summary = _execution_class_and_impact(destructive, mutates)

    return {
        "tool": tool,
        "action": action,
        "execution_class": execution_class,
        "mutates": mutates,
        "destructive": destructive,
        "idempotent": declared_idempotency,
        "preview_required": verb in _NON_READ_VERBS,
        "approval": _approval_info(cpd, destructive),
        "impact": {
            "summary": impact_summary,
            "scopes": metadata["scopes"],
            "durability": metadata["durability"],
            "transaction": metadata["transaction"],
        },
        "cost": metadata["cost"],
        "latency": metadata["latency"],
        "forwarded_fields": sorted(call_kwargs),
    }


def _plan_ref(
    verb: str,
    tool: str,
    action: str | None,
    call_kwargs: dict[str, Any],
    outcome_scope_ref: str | None,
) -> str:
    """Bind a non-read preview to its policy scope and exact operation."""

    payload = json.dumps(
        {
            "verb": verb,
            "tool": tool,
            "action": action,
            "arguments": call_kwargs,
            "outcome_scope_ref": outcome_scope_ref,
        },
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    return persistence_reference("intent_plan", payload)


def _expire_preview_plans(now: float) -> None:
    """Drop expired previews before a cache read or write."""

    expired = [
        ref
        for ref, preview in _PREVIEW_PLAN_CACHE.items()
        if now - preview.created_at > _PREVIEW_PLAN_TTL_SECONDS
    ]
    for ref in expired:
        _PREVIEW_PLAN_CACHE.pop(ref, None)


def _remember_preview_plan(
    plan_ref: str,
    *,
    verb: str,
    intent_ref: str,
    outcome_scope_ref: str | None,
    hints: dict[str, Any],
) -> None:
    """Retain one bounded plan so ``plan_ref`` alone can replay its hints."""

    now = time.monotonic()
    _expire_preview_plans(now)
    _PREVIEW_PLAN_CACHE[plan_ref] = _PreviewPlan(
        created_at=now,
        verb=verb,
        intent_ref=intent_ref,
        outcome_scope_ref=outcome_scope_ref,
        hints=deepcopy({k: v for k, v in hints.items() if k != "plan_ref"}),
    )
    _PREVIEW_PLAN_CACHE.move_to_end(plan_ref)
    while len(_PREVIEW_PLAN_CACHE) > _PREVIEW_PLAN_CACHE_MAX:
        _PREVIEW_PLAN_CACHE.popitem(last=False)


def _restore_preview_hints(
    plan_ref: str,
    *,
    verb: str,
    intent_ref: str,
    outcome_scope_ref: str | None,
) -> dict[str, Any] | None:
    """Return reviewed hints only when the caller and policy context match."""

    _expire_preview_plans(time.monotonic())
    preview = _PREVIEW_PLAN_CACHE.get(plan_ref)
    if preview is None:
        return None
    if (
        preview.verb != verb
        or preview.intent_ref != intent_ref
        or preview.outcome_scope_ref != outcome_scope_ref
    ):
        return None
    _PREVIEW_PLAN_CACHE.move_to_end(plan_ref)
    return deepcopy(preview.hints)


def _margin_ambiguity_fields(
    top_score: float,
    second_score: float | None,
    *,
    explicit: bool,
    extra_ambiguous_gate: bool = True,
) -> dict[str, Any]:
    """Shared score-margin ambiguity computation.

    Used by both `_ambiguity_evidence` and `_action_ambiguity_evidence`;
    `extra_ambiguous_gate` carries the latter's extra ``len(ranked_actions) >
    1`` condition (always True — a no-op AND — for the former).
    """
    margin = top_score - second_score if second_score is not None else None
    ambiguous = (
        not explicit
        and extra_ambiguous_gate
        and (
            top_score <= 0.0
            or (
                second_score is not None
                and second_score > 0.0
                and margin is not None
                and margin < _AMBIGUITY_MARGIN
            )
        )
    )
    return {
        "ambiguous": ambiguous,
        "explicit": explicit,
        "top_score": round(top_score, 4),
        "runner_up_score": (
            round(second_score, 4) if second_score is not None else None
        ),
        "margin": round(margin, 4) if margin is not None else None,
        "required_margin": _AMBIGUITY_MARGIN,
    }


def _ambiguity_evidence(
    candidates: list[CapabilityCandidate], *, explicit: bool
) -> dict[str, Any]:
    top_score = candidates[0].score if candidates else 0.0
    second_score = candidates[1].score if len(candidates) > 1 else None
    return _margin_ambiguity_fields(top_score, second_score, explicit=explicit)


def _action_ambiguity_evidence(
    ranked_actions: list[tuple[str, float]], *, explicit: bool
) -> dict[str, Any]:
    if not ranked_actions:
        return {
            "ambiguous": False,
            "explicit": explicit,
            "candidates": [],
        }
    top_score = ranked_actions[0][1]
    second_score = ranked_actions[1][1] if len(ranked_actions) > 1 else None
    fields = _margin_ambiguity_fields(
        top_score,
        second_score,
        explicit=explicit,
        extra_ambiguous_gate=len(ranked_actions) > 1,
    )
    fields["candidates"] = [
        {"action": action, "score": round(score, 4)}
        for action, score in ranked_actions[:5]
    ]
    return fields


def _intent_security_failure(
    intent: str, hints: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """Reject prompt/command injection without echoing untrusted content."""

    serialized_hints = json.dumps(
        hints or {}, sort_keys=True, default=str, separators=(",", ":")
    )
    scan = PromptInjectionScanner().scan_text(f"{intent}\n{serialized_hints}")
    if not scan.is_malicious:
        return None
    return {
        "error": "Intent rejected by the prompt-injection policy.",
        "executed": False,
        "security": {
            "decision": "deny",
            "confidence": round(scan.confidence, 4),
            "finding_ref": scan.finding_id,
            "patterns": sorted(
                {
                    str(match.get("pattern_name") or "")
                    for match in scan.matches
                    if match.get("pattern_name")
                }
            ),
        },
    }


def _execution_succeeded(result: Any) -> bool:
    """Classify only the observed tool result; caller feedback is never read.

    Many graph-os tools (``graph_query``, ``graph_ask``, ``nl_query``, the
    ``analyze``/``analysis`` surfaces, …) return a typed :class:`EvidenceBundle`
    rather than a plain dict/str. Before the ``BaseModel`` branch below existed,
    such a result matched none of the dict/str cases and fell through to
    ``result is not None`` — which a real, non-empty ``EvidenceBundle`` always
    satisfies, so a FAILED operation (an ``EvidenceBundle`` whose dedicated
    ``error`` field is populated) was reported as succeeded. Dumping the model
    first routes it through the same dict-shaped checks below, so
    ``EvidenceBundle.error`` (truthy exactly when the source payload carried a
    real operation failure — never fabricated, see
    ``agent_utilities.models.evidence_bundle``) is honored as the single source
    of truth instead of being invisible to this classifier.
    """

    if isinstance(result, BaseModel):
        return _execution_succeeded(result.model_dump())
    if isinstance(result, dict):
        return _execution_succeeded_dict(result)
    if isinstance(result, str):
        return _execution_succeeded_str(result)
    return result is not None


def _execution_succeeded_dict(result: dict[str, Any]) -> bool:
    """Helper for `_execution_succeeded`: classify a dict-shaped result."""
    if result.get("error") or result.get("ok") is False:
        return False
    if result.get("success") is False or result.get("executed") is False:
        return False
    if str(result.get("status") or "").strip().lower() in {
        "cancelled",
        "denied",
        "error",
        "failed",
        "forbidden",
    }:
        return False
    return True


def _execution_succeeded_str(result: str) -> bool:
    """Helper for `_execution_succeeded`: classify a str-shaped result."""
    stripped = result.strip()
    if not stripped:
        return False
    try:
        decoded = json.loads(stripped)
    except (TypeError, ValueError):
        lowered = stripped.lower()
        return not lowered.startswith(("error", "failed", "forbidden", "denied"))
    return _execution_succeeded(decoded)


def _require_candidate(candidate: CapabilityCandidate | None) -> CapabilityCandidate:
    """The resolved top candidate, refusing rather than returning ``None``.

    Reaching here with ``None`` is a contract violation: the resolver returns
    an outcome dict whenever it finds no candidate. Stating that as a raise
    keeps the invariant in ONE place instead of eight attribute reads.
    """
    if candidate is None:
        raise RuntimeError("intent routing resolved no capability candidate")
    return candidate


#: Session-scoped approvals recorded by ``manage(action="approve")``:
#: ``{opaque approval key: created_at}``. Bounded and expiring like the preview
#: cache; a restart or expiry requires a fresh approval.
_APPROVALS: OrderedDict[str, float] = OrderedDict()
_APPROVALS_MAX = 256
_APPROVAL_TTL_SECONDS = 600.0


def _approval_key(op_id: str) -> str:
    """Opaque key binding an approval to session, authority partition and op."""
    from agent_utilities.mcp.multiplexer import _session_key

    payload = json.dumps(
        [_session_key(), _outcome_scope_ref(), op_id], separators=(",", ":")
    )
    return persistence_reference("intent_approval", payload)


def _expire_approvals(now: float) -> None:
    expired = [
        key
        for key, created in _APPROVALS.items()
        if now - created > _APPROVAL_TTL_SECONDS
    ]
    for key in expired:
        _APPROVALS.pop(key, None)


def _record_approval(op_id: str) -> None:
    """Record this session's approval of exactly ``op_id``."""
    now = time.monotonic()
    _expire_approvals(now)
    key = _approval_key(op_id)
    _APPROVALS[key] = now
    _APPROVALS.move_to_end(key)
    while len(_APPROVALS) > _APPROVALS_MAX:
        _APPROVALS.popitem(last=False)


def _operation_approved(op_id: str) -> bool:
    """BUG-040: has THIS caller's session approved exactly ``op_id``?

    An approval-required (destructive, or non-``auto`` ``approval_class``)
    plan executes only after the session approved that exact operation with
    ``manage(action="approve", params={"action": "<tool>.<op>"})`` — itself a
    governed preview → ``plan_ref`` → execute round trip — and then
    resubmitted the original plan's ``plan_ref``. Naming or pinning the
    operation alone never satisfies it (a pin selects WHICH candidate; it
    never bypasses approval).

    ``approval_class`` is a static CPD declaration, not a per-call check inside
    the target tool; the real authorization boundary remains the actor/session
    identity verification in ``kg_server.verified_tool_session_scope`` /
    ``_execute_tool``. The approval step replaces the retired session
    ``load_tools`` acknowledgement with one every client can perform (no
    tool-list refresh involved — the gap BUG-040/BUG-050 found).
    """
    _expire_approvals(time.monotonic())
    try:
        return _approval_key(op_id) in _APPROVALS
    except Exception:  # noqa: BLE001 — a broken session probe must fail closed
        return False


def _plan_ref_hints_match(
    raw_hints: dict[str, Any], restored_hints: dict[str, Any]
) -> bool:
    """Helper for `dispatch_intent`'s `_restore_from_plan_ref` closure.

    D-GIS-1: this equality check exists so a caller resubmitting the SAME
    hints they previewed (plus plan_ref) is provably replaying what was
    reviewed. But `_remember_preview_plan` stores `raw_hints` from AFTER the
    resolver's own `chosen_tool` merge (see `_select_tool_and_action`'s "if
    chosen_tool in _DOCUMENTED_HINT_ALIASES"), which injects a "tool"/"_tool"
    key the caller never supplied and has no way to know in advance (it is
    the resolver's inferred routing target, e.g. an unpinned
    skill-delegation intent auto-resolving to `graph_orchestrate`).
    Comparing against that resolver-injected key made every unpinned
    non-read `act` reject its own documented "resubmit the same hints plus
    plan_ref" flow unconditionally -- the caller's hints could never
    contain a key they were never told to supply. Excluding it here does
    not weaken the check: everything the CALLER actually controls must
    still match exactly, and the resolver will independently re-derive the
    identical `tool` value from the (matched) remaining hints during
    dispatch, so nothing forged by a caller can smuggle a different tool
    through this path.
    """
    resolver_injected_hint_fields = ("tool", "_tool")
    replayed_hints = {
        k: v
        for k, v in raw_hints.items()
        if k != "plan_ref" and k not in resolver_injected_hint_fields
    }
    comparable_restored_hints = {
        k: v
        for k, v in restored_hints.items()
        if k not in resolver_injected_hint_fields
    }
    return not replayed_hints or replayed_hints == comparable_restored_hints


def _normalize_chosen_tool_alias(
    raw_hints: dict[str, Any], chosen_tool: str
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Helper for `dispatch_intent`'s `_select_tool_and_action` closure.

    A non-pinned resolver result can still be the orchestration façade. It is
    safe to normalize its two documented target aliases only after the
    resolver has selected that tool. Returns (raw_hints, error_response);
    error_response is None on success.
    """
    if chosen_tool not in _DOCUMENTED_HINT_ALIASES:
        return raw_hints, None
    try:
        return (
            _normalize_documented_hint_aliases({**raw_hints, "tool": chosen_tool}),
            None,
        )
    except ValueError as exc:
        # ``exc.args[0]`` (not ``str(exc)``/``repr(exc)``): this ValueError's
        # message is a developer-authored, non-sensitive validation
        # explanation (see _normalize_documented_hint_aliases) that callers
        # need to self-correct their hints -- test_orchestration_target_
        # alias_conflict_fails_closed and test_intent_rejects_unknown_hint_
        # before_creating_preview assert on the exact text. Collapsing to
        # the exception TYPE name here would satisfy the served-boundary
        # exception-surface gate's letter while losing the caller-facing
        # detail those tests require; .args[0] satisfies both.
        return raw_hints, {
            "error": exc.args[0] if exc.args else type(exc).__name__,
            "executed": False,
        }


def _validate_explicit_action(
    verb: str,
    chosen_tool: str,
    explicit_action: Any,
    available_actions: list[str],
) -> dict[str, Any] | None:
    """Helper for `dispatch_intent`'s `_select_tool_and_action` closure.

    Rejects an explicit action not declared for the selected capability.
    """
    if explicit_action is not None and explicit_action not in available_actions:
        return {
            "error": "Requested action is not declared for the selected capability.",
            "executed": False,
            "routing": {
                "verb": verb,
                "chosen_tool": chosen_tool,
                "declared_actions": sorted(available_actions),
            },
        }
    return None


def _restrict_to_read_actions(
    verb: str,
    chosen_tool: str,
    explicit_action: Any,
    ranked_actions: list[tuple[str, float]],
) -> tuple[list[tuple[str, float]], dict[str, Any] | None]:
    """Helper for `dispatch_intent`'s `_select_tool_and_action` closure.

    For a read-only verb with a declared read-action allowlist: validates the
    explicit action (if any), filters `ranked_actions` to just the read-only
    ones, and checks the result for ambiguity. Returns (ranked_actions,
    error_response); `ranked_actions` is returned unchanged, and
    error_response is None, when the verb/tool has no read-only allowlist.
    """
    read_actions = READ_ONLY_ACTIONS.get(chosen_tool)
    if not (verb in _READ_ONLY_VERBS and read_actions is not None):
        return ranked_actions, None
    if explicit_action is not None and explicit_action not in read_actions:
        return ranked_actions, {
            "error": "Read-only intent action is not declared read-only.",
            "executed": False,
            "routing": {
                "verb": verb,
                "chosen_tool": chosen_tool,
                "declared_read_actions": sorted(read_actions),
            },
        }
    filtered = [
        (action, score) for action, score in ranked_actions if action in read_actions
    ]
    read_action_ambiguity = _action_ambiguity_evidence(
        filtered, explicit=explicit_action is not None
    )
    if explicit_action is None and read_action_ambiguity["ambiguous"]:
        return filtered, {
            "error": "Ambiguous read-only action requires an explicit action.",
            "executed": False,
            "routing": {
                "verb": verb,
                "chosen_tool": chosen_tool,
                "declared_read_actions": sorted(read_actions),
                "ambiguity": {"action": read_action_ambiguity},
            },
        }
    return filtered, None


def _should_fall_back_to_nl_planner(
    verb: str,
    chosen_tool: str,
    call_kwargs: dict[str, Any],
    raw_hints: dict[str, Any],
    explicit_action: Any,
) -> bool:
    """Helper for `dispatch_intent`'s `_finalize_call_kwargs` closure.

    True when there are no structured hints AND the winning tool has no
    known free-text param: falling back to the NL planner is safer than
    dispatching a call known to be missing required args
    (CONCEPT:AU-KG.query.ask-gateway-rest-twin). An explicit pin/action
    never falls back — that would silently replace the caller's declared
    route.
    """
    return (
        not call_kwargs
        and chosen_tool not in _PRIMARY_TEXT_PARAM
        and verb == "ask"
        and not raw_hints.get("tool")
        and not raw_hints.get("_tool")
        and explicit_action is None
    )


def _policy_gate_preview_response(
    verb: str,
    should_execute: bool,
    plan_ref: str,
    intent_ref: str,
    outcome_scope_ref: str | None,
    raw_hints: dict[str, Any],
    routing: dict[str, Any],
) -> dict[str, Any] | None:
    """Helper for `dispatch_intent`'s `_policy_gate` closure: the preview (not-execute) branch."""
    if should_execute:
        return None
    if verb in _NON_READ_VERBS:
        _remember_preview_plan(
            plan_ref,
            verb=verb,
            intent_ref=intent_ref,
            outcome_scope_ref=outcome_scope_ref,
            hints=raw_hints,
        )
    return {"routing": routing, "executed": False}


def _policy_gate_execution_checks(
    verb: str,
    plan: dict[str, Any],
    candidate_ambiguity: dict[str, Any],
    action_ambiguity: dict[str, Any],
    supplied_plan_ref: str,
    plan_ref: str,
    routing: dict[str, Any],
) -> dict[str, Any] | None:
    """Helper for `dispatch_intent`'s `_policy_gate` closure: execution-path policy checks

    (read-only violation, unclassified effect, ambiguity, stale plan_ref) —
    everything except the final approval/session-load check.
    """
    if verb in _READ_ONLY_VERBS and plan["mutates"] is not False:
        return {
            "error": "Read-only intent policy rejected a mutating or unclassified route.",
            "routing": routing,
            "executed": False,
        }
    if plan["execution_class"] == "unclassified":
        return {
            "error": "Operation effect metadata is unclassified; execution denied.",
            "routing": routing,
            "executed": False,
        }
    if verb in _NON_READ_VERBS and (
        candidate_ambiguity["ambiguous"] or action_ambiguity["ambiguous"]
    ):
        return {
            "error": "Ambiguous non-read intent requires an explicit tool and action.",
            "routing": routing,
            "executed": False,
        }
    if verb in _NON_READ_VERBS and supplied_plan_ref != plan_ref:
        return {
            "error": (
                "Preview required: call with execute=false, review the plan, then "
                "resubmit its plan_ref with execute=true."
            ),
            "routing": routing,
            "executed": False,
        }
    return None


def _policy_gate_approval_check(
    plan: dict[str, Any],
    routing: dict[str, Any],
) -> dict[str, Any] | None:
    """Helper for `dispatch_intent`'s `_policy_gate` closure: the approval check."""
    required = operation_id(plan["tool"], plan["action"])
    if not plan["approval"]["required"] or _operation_approved(required):
        return None
    return {
        "error": (
            "Approval-required operation: approve it with manage(action='approve', "
            f"params={{'action': {required!r}}}) (preview, then execute with its "
            "plan_ref), then resubmit this plan_ref with execute=true."
        ),
        "routing": routing,
        "executed": False,
        "approval_required": True,
        "required_approval": required,
    }


def _dispatch_hint_derived_fields(
    raw_hints: dict[str, Any],
) -> tuple[str, bool, dict[str, Any]]:
    """Helper for `dispatch_intent`: pinned_name, explicit_tool, and call_kwargs from raw_hints."""
    pinned_name = str(raw_hints.get("tool") or raw_hints.get("_tool") or "")
    explicit_tool = bool(pinned_name)
    call_kwargs = {k: v for k, v in raw_hints.items() if k not in _CONTROL_HINT_FIELDS}
    return pinned_name, explicit_tool, call_kwargs


def _dispatch_reward_and_learning_eligibility(
    verb: str,
    chosen_tool: str,
    outcome_scope_ref: str | None,
    explicit_tool: bool,
    candidate_ambiguity: dict[str, Any],
    action_ambiguity: dict[str, Any],
) -> tuple[float, bool]:
    """Helper for `dispatch_intent`: the calibrated outcome reward + learning eligibility."""
    reward = (
        _outcome_router().reward_of(
            _reward_task_class(verb, outcome_scope_ref), chosen_tool
        )
        if outcome_scope_ref is not None
        else 0.5
    )
    learning_eligible = bool(
        outcome_scope_ref
        and not explicit_tool
        and not candidate_ambiguity["ambiguous"]
        and not action_ambiguity["ambiguous"]
    )
    return reward, learning_eligible


async def _dispatch_execute_and_record(
    verb: str,
    chosen_tool: str,
    call_kwargs: dict[str, Any],
    outcome_scope_ref: str | None,
    learning_eligible: bool,
    routing: dict[str, Any],
) -> dict[str, Any]:
    """Helper for `dispatch_intent`: execute the chosen tool, record the outcome,

    and build the final response dict.
    """
    try:
        result = await kg_server._execute_tool(chosen_tool, **call_kwargs)
    except Exception as e:  # noqa: BLE001 — surface as a structured routing failure, not a 500
        if learning_eligible and outcome_scope_ref is not None:
            _record_dispatch_outcome(
                outcome_scope_ref, verb, chosen_tool, success=False
            )
        routing["decision_trace"]["result_provenance"]["status"] = "failed"
        routing["learning"]["recorded"] = learning_eligible
        return {
            "routing": routing,
            "executed": False,
            **public_error_payload(e, logger=logger),
        }
    observed_success = _execution_succeeded(result)
    if learning_eligible and outcome_scope_ref is not None:
        _record_dispatch_outcome(
            outcome_scope_ref, verb, chosen_tool, success=observed_success
        )
    routing["decision_trace"]["result_provenance"]["status"] = (
        "succeeded" if observed_success else "failed"
    )
    routing["learning"]["recorded"] = learning_eligible
    return {"result": result, "routing": routing, "executed": True}


def _question_routing(intent_ref: str, route: str, why: str) -> dict[str, Any]:
    return {
        "verb": "ask",
        "intent_ref": intent_ref,
        "chosen_tool": route,
        "action": None,
        "why": why,
        "capability_source": "graphos_question_route",
        "decision_trace": {
            "policy": {"verb_class": "read_only", "read_only_enforced": True},
            "route": {"tool": route, "action": None, "fallback": False},
            "result_provenance": {"execution_core": route, "status": "succeeded"},
        },
    }


async def _question_route(
    verb: str, intent: str, raw_hints: dict[str, Any], intent_ref: str
) -> dict[str, Any] | None:
    """Route an unpinned ``ask`` that is a planning or cross-source question.

    "How can I ...", "what steps ..." and "which agents should ..." go to the
    task planner (AU-CONTROL-R028). A question whose ontology concepts span
    two or more installed virtual sources goes to the cross-source report
    (AU-CONTROL-R029). Both answers are read-only and carry provenance. Any
    hint (a pinned tool, an action or structured arguments) keeps the
    caller's declared route.
    """
    if verb != "ask" or raw_hints:
        return None
    from agent_utilities.decide.consumers.task_planner import (
        is_planning_question,
        process_task_planner,
    )
    from agent_utilities.knowledge_graph.virtual_graph.federation import (
        answer_cross_source,
    )

    if is_planning_question(intent):
        plan = await process_task_planner().plan(intent)
        why = "Planning question: answered by the task planner."
        routing = _question_routing(intent_ref, "task_planner", why)
        return {"result": plan.to_dict(), "routing": routing, "executed": True}
    report = await answer_cross_source(intent)
    if report is None:
        return None
    why = "Ontology concepts span installed virtual sources: federated report."
    routing = _question_routing(intent_ref, "cross_source_report", why)
    return {"result": report.to_dict(), "routing": routing, "executed": True}


async def dispatch_intent(
    verb: str,
    intent: str,
    *,
    hints: dict[str, Any] | None = None,
    execute: bool | None = None,
    top_k: int = 5,
    mcp: Any = None,
) -> dict[str, Any]:
    """Resolve, preview, policy-check, and optionally dispatch one intent.

    ``ask``/``why`` execute by default and remain read-only. Non-read verbs
    preview by default; execution requires the opaque ``plan_ref`` returned by
    that exact preview. An approval-required (destructive or non-``auto``
    ``approval_class``) plan only executes once the caller's session approved
    that exact operation (BUG-040 — see :func:`_operation_approved`); otherwise
    it fails closed with an actionable ``required_approval`` hint. Outcome
    learning consumes only the verified tool result of an unpinned,
    unambiguous execution in the current tenant/policy partition.
    """
    raw_hints: dict[str, Any] = {}

    def _intake() -> dict[str, Any] | None:
        nonlocal raw_hints
        try:
            raw_hints = _normalize_documented_hint_aliases(dict(hints or {}))
        except ValueError as exc:
            # ``exc.args[0]`` (not ``str(exc)``/``repr(exc)``): see the identical
            # comment on the second _normalize_documented_hint_aliases call site
            # below.
            return {
                "error": exc.args[0] if exc.args else type(exc).__name__,
                "executed": False,
            }
        if verb not in _DISPATCH_VERBS:
            return {
                "error": "Unsupported GraphOS intent verb.",
                "executed": False,
                "routing": {"verb": verb, "candidates": []},
            }
        security_failure = _intent_security_failure(intent, raw_hints)
        if security_failure is not None:
            return security_failure
        supplied_outcomes = sorted(set(raw_hints) & _CALLER_OUTCOME_FIELDS)
        if supplied_outcomes:
            return {
                "error": "Caller-supplied routing outcomes are forbidden.",
                "executed": False,
                "security": {
                    "decision": "deny",
                    "fields": supplied_outcomes,
                    "reason": "Only verified execution results update routing rewards.",
                },
            }
        return None

    _intake_outcome = _intake()
    if _intake_outcome is not None:
        return _intake_outcome

    should_execute = verb in _READ_ONLY_VERBS if execute is None else bool(execute)
    intent_ref = persistence_reference("intent", intent)
    outcome_scope_ref = _outcome_scope_ref()
    supplied_plan_ref = str(raw_hints.get("plan_ref") or "")

    async def _restore_from_plan_ref() -> dict[str, Any] | None:
        nonlocal raw_hints
        if not (verb in _NON_READ_VERBS and should_execute and supplied_plan_ref):
            # No reviewed plan to restore: a bare planning or cross-source
            # ``ask`` is answered here, before capability ranking.
            return await _question_route(verb, intent, raw_hints, intent_ref)
        restored_hints = _restore_preview_hints(
            supplied_plan_ref,
            verb=verb,
            intent_ref=intent_ref,
            outcome_scope_ref=outcome_scope_ref,
        )
        if restored_hints is None:
            return {
                "error": (
                    "Unknown, expired, or context-mismatched plan_ref; request a "
                    "new preview before execution."
                ),
                "executed": False,
                "routing": {"verb": verb, "intent_ref": intent_ref},
            }
        if not _plan_ref_hints_match(raw_hints, restored_hints):
            return {
                "error": "Supplied hints do not match the reviewed preview plan.",
                "executed": False,
                "routing": {"verb": verb, "intent_ref": intent_ref},
            }
        raw_hints = {**restored_hints, "plan_ref": supplied_plan_ref}
        return None

    _restore_outcome = await _restore_from_plan_ref()
    if _restore_outcome is not None:
        return _restore_outcome

    pinned_name, explicit_tool, call_kwargs = _dispatch_hint_derived_fields(raw_hints)
    explicit_action = raw_hints.get("action")

    candidates: list[CapabilityCandidate] = []
    top: CapabilityCandidate | None = None
    chosen_tool: str = ""

    def _resolve_candidates() -> dict[str, Any] | None:
        nonlocal candidates, top, chosen_tool
        candidates = resolve_intent(
            verb, intent, hints=raw_hints, top_k=max(2, int(top_k))
        )
        if not candidates:
            if explicit_tool:
                # Two DIFFERENT failure reasons collapse to the same empty
                # `candidates` from `resolve_intent` — distinguish them so the
                # error is actionable instead of implying a verb-specific policy
                # restriction that doesn't exist. A tool the router has never
                # heard of (most commonly a FLEET tool — it carries no
                # Capability Power Descriptor and was never a candidate at all)
                # is not "disallowed for this verb"; it is reached through
                # act(action="fleet.call") instead.
                known_to_intent_surface = any(
                    candidate.tool == pinned_name for candidate in _build_candidates()
                )
                error = (
                    "Pinned capability is not allowed for this intent verb."
                    if known_to_intent_surface
                    else (
                        f"'{pinned_name}' is not a GraphOS intent-routable capability "
                        "(no Capability Power Descriptor is registered for it). "
                        "Call a fleet tool with act(action='fleet.call', "
                        f"params={{'tool': '{pinned_name}', 'arguments': {{...}}}})."
                    )
                )
            else:
                error = "No GraphOS capability matched the requested intent verb."
            return {
                "error": error,
                "executed": False,
                "routing": {
                    "verb": verb,
                    "intent_ref": intent_ref,
                    "candidates": [],
                },
            }

        top = candidates[0]
        chosen_tool = top.tool
        return None

    _candidates_outcome = _resolve_candidates()
    if _candidates_outcome is not None:
        return _candidates_outcome

    # `_resolve_candidates` returns an outcome dict whenever it finds no
    # candidate, so `top` is bound past this point by contract. Binding a
    # non-optional name for it says that once, HERE, instead of leaving every
    # later `top.<attr>` to be read as a possible attribute-on-None — which is
    # what the closures below made it, since a closure reads the DECLARED type
    # of a free variable, not a narrowed one.
    chosen_candidate = _require_candidate(top)

    chosen_action: str | None = None
    ranked_actions: list[tuple[str, float]] = []

    def _select_tool_and_action() -> dict[str, Any] | None:
        nonlocal raw_hints, chosen_tool, chosen_action, ranked_actions
        # A non-pinned resolver result can still be the orchestration façade.  It
        # is safe to normalize its two documented target aliases only after the
        # resolver has selected that tool.
        raw_hints, alias_error = _normalize_chosen_tool_alias(raw_hints, chosen_tool)
        if alias_error is not None:
            return alias_error

        available_actions = _actions_by_tool().get(chosen_tool, [])
        action_error = _validate_explicit_action(
            verb, chosen_tool, explicit_action, available_actions
        )
        if action_error is not None:
            return action_error

        ranked_actions = _rank_actions(chosen_tool, intent)
        ranked_actions, read_action_error = _restrict_to_read_actions(
            verb, chosen_tool, explicit_action, ranked_actions
        )
        if read_action_error is not None:
            return read_action_error

        chosen_action = (
            explicit_action
            or chosen_candidate.action
            or (ranked_actions[0][0] if ranked_actions else None)
        )
        return None

    _select_outcome = _select_tool_and_action()
    if _select_outcome is not None:
        return _select_outcome

    fell_back = False

    def _finalize_call_kwargs() -> dict[str, Any] | None:
        nonlocal chosen_tool, chosen_action, call_kwargs, ranked_actions, fell_back
        text_param = _PRIMARY_TEXT_PARAM.get(chosen_tool)
        fell_back = False
        if text_param and text_param not in call_kwargs:
            call_kwargs[text_param] = intent
        elif _should_fall_back_to_nl_planner(
            verb, chosen_tool, call_kwargs, raw_hints, explicit_action
        ):
            # No structured hints AND the winning tool has no known free-text param:
            # fall back to the NL planner rather than dispatch a call we know is
            # missing required args (CONCEPT:AU-KG.query.ask-gateway-rest-twin).
            # An explicit pin/action never falls back because that would silently
            # replace the caller's declared route.
            fell_back = True
            chosen_tool = _ASK_FALLBACK_TOOL
            chosen_action = None
            call_kwargs = {_PRIMARY_TEXT_PARAM[_ASK_FALLBACK_TOOL]: intent}
            ranked_actions = []

        if chosen_action is not None and _tool_accepts_argument(chosen_tool, "action"):
            call_kwargs.setdefault("action", chosen_action)

        unsupported_hints = _unsupported_hint_arguments(chosen_tool, call_kwargs)
        if unsupported_hints:
            return {
                "error": _hint_argument_error(chosen_tool, unsupported_hints),
                "executed": False,
                "routing": {
                    "verb": verb,
                    "intent_ref": intent_ref,
                    "chosen_tool": chosen_tool,
                    "unsupported_hint_arguments": unsupported_hints,
                },
            }
        return None

    _kwargs_outcome = _finalize_call_kwargs()
    if _kwargs_outcome is not None:
        return _kwargs_outcome

    candidate_ambiguity = _ambiguity_evidence(candidates, explicit=explicit_tool)
    action_ambiguity = _action_ambiguity_evidence(
        ranked_actions, explicit=explicit_action is not None
    )
    plan = _operation_plan(verb, chosen_tool, chosen_action, call_kwargs)
    plan_ref = _plan_ref(
        verb, chosen_tool, chosen_action, call_kwargs, outcome_scope_ref
    )
    plan["plan_ref"] = plan_ref

    reward, learning_eligible = _dispatch_reward_and_learning_eligibility(
        verb,
        chosen_tool,
        outcome_scope_ref,
        explicit_tool,
        candidate_ambiguity,
        action_ambiguity,
    )

    def _build_routing() -> dict[str, Any]:
        routing: dict[str, Any] = {
            "verb": verb,
            "intent_ref": intent_ref,
            "chosen_tool": chosen_tool,
            "action": chosen_action,
            "score": round(chosen_candidate.score, 4),
            "matched_terms": chosen_candidate.matched_terms,
            "fell_back_to_nl_planner": fell_back,
            "why": (
                f"'{chosen_candidate.tool}' best matched the {verb!r} intent on descriptor terms "
                f"{chosen_candidate.matched_terms!r}"
                if chosen_candidate.matched_terms
                else f"'{chosen_candidate.tool}' is the highest-ranked capability for verb {verb!r}"
            )
            + (
                f"; routed through '{_ASK_FALLBACK_TOOL}' because the selected "
                f"capability requires structured arguments."
                if fell_back
                else "."
            ),
            "alternatives": [
                {"tool": c.tool, "action": c.action, "score": round(c.score, 4)}
                for c in candidates[1:]
            ],
            "capability_source": "packaged_graphos_cpd",
            "calibrated_outcome_reward": round(reward, 4),
            "ambiguity": {
                "capability": candidate_ambiguity,
                "action": action_ambiguity,
            },
            "plan": plan,
            "decision_trace": {
                "evidence": {
                    "matched_terms": chosen_candidate.matched_terms,
                    "candidate_count": len(candidates),
                    "capability_source": "packaged_graphos_cpd",
                },
                "policy": {
                    "verb_class": (
                        "read_only" if verb in _READ_ONLY_VERBS else "non_read"
                    ),
                    "read_only_enforced": verb in _READ_ONLY_VERBS,
                    "preview_required": plan["preview_required"],
                    "approval": plan["approval"],
                },
                "route": {
                    "tool": chosen_tool,
                    "action": chosen_action,
                    "fallback": fell_back,
                },
                "result_provenance": {
                    "execution_core": "graphos_verified_execute_tool",
                    "status": "preview",
                },
            },
            "learning": {
                "eligible": learning_eligible,
                "partition_ref": outcome_scope_ref or "unverified",
                "source": "verified_execution_result_only",
            },
        }
        return routing

    routing = _build_routing()

    def _policy_gate() -> dict[str, Any] | None:
        preview_response = _policy_gate_preview_response(
            verb,
            should_execute,
            plan_ref,
            intent_ref,
            outcome_scope_ref,
            raw_hints,
            routing,
        )
        if preview_response is not None:
            return preview_response
        execution_error = _policy_gate_execution_checks(
            verb,
            plan,
            candidate_ambiguity,
            action_ambiguity,
            supplied_plan_ref,
            plan_ref,
            routing,
        )
        if execution_error is not None:
            return execution_error
        return _policy_gate_approval_check(plan, routing)

    _policy_outcome = _policy_gate()
    if _policy_outcome is not None:
        return _policy_outcome

    return await _dispatch_execute_and_record(
        verb, chosen_tool, call_kwargs, outcome_scope_ref, learning_eligible, routing
    )


#: ``act`` operation that calls one tool of another fleet MCP server.
FLEET_CALL_ACTION = "fleet.call"
#: ``manage`` operations that mount fleet servers/tools ahead of use and
#: release them again (``fleet.call`` mounts on demand either way).
FLEET_LOAD_ACTION = "fleet.load"
FLEET_UNLOAD_ACTION = "fleet.unload"
#: ``manage`` operation that approves one approval-required operation for the
#: caller's session (see :func:`_operation_approved`).
APPROVE_ACTION = "approve"
#: ``manage``'s read-only lakehouse maintenance status operation.
LAKEHOUSE_STATUS_ACTION = "lakehouse_status"
#: ``find``'s fixed action set: capability search (default), fleet tool search,
#: the fleet catalog, fleet health, and operation descriptions.
FIND_ACTIONS = ("capabilities", "tools", "catalog", "status", DESCRIBE_ACTION)
_FIND_TOP_K = 8
#: Request bounds on fleet operations (the retired load_tools/unload_tools
#: meta-tools' safety boundary).
_FLEET_LIST_LIMITS = {"tools": 128, "servers": 32}

#: Operations a verb serves beside the generated manifest.
_HOST_ACTIONS: dict[str, tuple[str, ...]] = {
    "find": FIND_ACTIONS,
    "act": (FLEET_CALL_ACTION,),
    "manage": (
        APPROVE_ACTION,
        FLEET_LOAD_ACTION,
        FLEET_UNLOAD_ACTION,
        LAKEHOUSE_STATUS_ACTION,
    ),
}

_VERB_SUMMARIES = {
    "ask": "Read or answer from the Knowledge Graph.",
    "find": "Discover capabilities, operations, fleet tools and the fleet catalog.",
    "write": "Create or change graph data; previews first, then execute with plan_ref.",
    "act": "Run work, an operation or a fleet tool; previews first, then execute with plan_ref.",
    "manage": "Configure and govern graph-os and fleet loading; previews first, then execute with plan_ref.",
    "why": "Explain beliefs, decisions, provenance and changes.",
}

_OPERATION_VERBS_CACHE: dict[str, tuple[str, ...]] | None = None


def _verb_description(verb: str) -> str:
    if verb == "find":
        return (
            f"{_VERB_SUMMARIES[verb]} Default: rank capabilities for `intent`; "
            "'describe' explains an operation."
        )
    return (
        f"{_VERB_SUMMARIES[verb]} action='<tool>.<op>' (see action='describe'), "
        "or leave it empty to route `intent`."
    )


def _policy_verbs(tool: str, action: str | None) -> tuple[str, ...]:
    """The dispatch verbs whose declared policy classifies this operation."""
    verbs: list[str] = []
    for verb in TOOL_VERBS.get(tool, ()):
        if verb not in _DISPATCH_VERBS:
            continue
        plan = _operation_plan(verb, tool, action, {})
        if plan["execution_class"] == "unclassified":
            continue
        if verb in _READ_ONLY_VERBS and plan["mutates"] is not False:
            continue
        verbs.append(verb)
    return tuple(verbs)


def _accepting_verbs(tool: str, action: str | None) -> tuple[str, ...]:
    """The intent verbs whose router policy accepts this exact operation.

    An operation none of its declared verbs can classify (no declared effect,
    not on a reviewed read-only allowlist) is reachable through ``act`` alone,
    which treats it as a mutation: preview, ``plan_ref`` and the CPD approval
    policy all apply (see :func:`_act_fallback`).
    """
    return _policy_verbs(tool, action) or ("act",)


def _act_fallback(tool: str, action: str | None) -> bool:
    """Whether ``act`` serves ``tool``'s ``action`` as an unclassified operation."""
    return "act" not in TOOL_VERBS.get(tool, ()) and not _policy_verbs(tool, action)


def operation_verbs() -> dict[str, tuple[str, ...]]:
    """``{operation id: verbs that accept it}`` for every manifest operation."""
    global _OPERATION_VERBS_CACHE
    if _OPERATION_VERBS_CACHE is None:
        from agent_utilities.mcp.graphos_surface import manifest_operations

        _OPERATION_VERBS_CACHE = {
            op_id: _accepting_verbs(tool, action)
            for op_id, (tool, action) in manifest_operations().items()
        }
    return _OPERATION_VERBS_CACHE


def _verb_catalog(mcp: Any, verb: str, group: str | None) -> dict[str, Any]:
    """Operation ids ``verb`` accepts, grouped by routing group."""
    from agent_utilities.mcp.graphos_surface import (
        group_for_tool,
        host_operations,
        manifest_operations,
    )

    table = manifest_operations()
    grouped: dict[str, list[str]] = {}
    for op_id, verbs in operation_verbs().items():
        name = group_for_tool(table[op_id][0]) or ""
        if verb in verbs and (group is None or name == group):
            grouped.setdefault(name, []).append(op_id)
    extra = list(_HOST_ACTIONS.get(verb, ()))
    if verb == "act":
        extra += sorted(host_operations(mcp))
    if extra and group is None:
        grouped["host"] = extra
    return {"verb": verb, "operations": grouped}


def _describe_host_operation(mcp: Any, op_id: str) -> dict[str, Any] | None:
    """``describe`` for a host-native backing operation, or ``None``."""
    from agent_utilities.mcp.graphos_surface import backing_tools, host_operations
    from agent_utilities.mcp.intent_contract import schema_without_action

    resolved = host_operations(mcp).get(op_id)
    if resolved is None:
        return None
    tool = backing_tools(mcp)[resolved[0]]
    return {
        "action": op_id,
        "verbs": ["act"],
        "description": getattr(tool, "description", None) or "",
        "params_schema": schema_without_action(tool.parameters or {}),
    }


def _describe_operation(mcp: Any, op_id: str) -> dict[str, Any]:
    """One operation: backing tool, accepting verbs, description, argument schema."""
    from agent_utilities.mcp.graphos_surface import (
        backing_tools,
        group_for_tool,
        resolve_operation,
    )
    from agent_utilities.mcp.intent_contract import schema_without_action

    resolved = resolve_operation(op_id)
    if resolved is None:
        host = _describe_host_operation(mcp, op_id)
        if host is not None:
            return host
        return {"error": f"Unknown operation {op_id!r}; action='describe' lists them."}
    tool_name, action = resolved
    tool = backing_tools(mcp).get(tool_name)
    cpd = _load_cpds_required().get(tool_name) or {}
    record = _operation_record(cpd, action) or {}
    return {
        "action": op_id,
        "group": group_for_tool(tool_name),
        "verbs": list(operation_verbs().get(op_id, ())),
        "description": str(record.get("description") or cpd.get("one_line") or ""),
        "params_schema": schema_without_action(getattr(tool, "parameters", None) or {}),
        "available": tool is not None,
    }


def _describe(mcp: Any, verb: str, params: dict[str, Any]) -> dict[str, Any]:
    target = params.get("action")
    if target:
        return _describe_operation(mcp, str(target))
    group = params.get("group")
    return _verb_catalog(mcp, verb, str(group) if group else None)


def _operation_hints(
    tool: str, action: str | None, params: dict[str, Any]
) -> dict[str, Any]:
    """Router hints for an explicit operation: shaped arguments + the pin."""
    from agent_utilities.mcp.intent_contract import shape_arguments

    controls = {key: params.pop(key) for key in ("plan_ref",) if key in params}
    function = kg_server.REGISTERED_TOOLS.get(tool)
    accepted: set[str] = set()
    var_keyword = False
    if function is not None:
        parameters = inspect.signature(function).parameters
        accepted = set(parameters)
        var_keyword = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
        )
    hints = shape_arguments(accepted, params, action, accepts_var_keyword=var_keyword)
    hints.update(controls)
    hints["tool"] = tool
    return hints


def _fleet_mux(mcp: Any) -> Any:
    mux = getattr(mcp, "_fleet_mux", None)
    if mux is None:
        raise ValueError(
            "No fleet multiplexer is attached to this server (embedded/headless build)."
        )
    return mux


def _host_plan_ref(verb: str, op_id: str, arguments: dict[str, Any]) -> str:
    """Bind a host/fleet operation preview to its verb, operation and arguments."""
    return _plan_ref(verb, "host", op_id, arguments, _outcome_scope_ref())


async def _governed(
    verb: str,
    op_id: str,
    arguments: dict[str, Any],
    params: dict[str, Any],
    execute: bool,
    run: Any,
) -> dict[str, Any]:
    """Preview → ``plan_ref`` → execute for an operation outside the manifest.

    The same contract :func:`dispatch_intent` enforces for non-read verbs: a
    call without ``execute`` returns the plan and its ``plan_ref``; execution
    requires resubmitting that ``plan_ref`` for the identical operation and
    arguments.
    """
    plan_ref = _host_plan_ref(verb, op_id, arguments)
    plan = {
        "action": op_id,
        "arguments": sorted(arguments),
        "preview_required": True,
        "plan_ref": plan_ref,
    }
    if not execute:
        return {"executed": False, "plan": plan}
    if str(params.get("plan_ref") or "") != plan_ref:
        return {
            "executed": False,
            "plan": plan,
            "error": (
                "Preview required: call with execute=false, review the plan, then "
                "resubmit its plan_ref with execute=true."
            ),
        }
    return {"executed": True, "action": op_id, "result": await run()}


def _bounded_names(params: dict[str, Any], key: str) -> list[str]:
    names = params.get(key) or []
    if not isinstance(names, list) or len(names) > _FLEET_LIST_LIMITS[key]:
        raise ValueError(
            f"params.{key} must be a list of at most {_FLEET_LIST_LIMITS[key]} names"
        )
    return [str(name) for name in names]


async def _find(mcp: Any, action: str, params: dict[str, Any], intent: str) -> Any:
    """``find``: capability ranking, fleet tool search, fleet catalog, fleet health."""
    selected = action or "capabilities"
    if selected not in FIND_ACTIONS:
        return {
            "error": f"Unknown find action {action!r}.",
            "actions": list(FIND_ACTIONS),
        }
    top_k = max(1, min(int(params.get("top_k") or _FIND_TOP_K), 100))
    if selected == "capabilities":
        if not intent:
            return {"error": "find needs an `intent` to rank capabilities."}
        return await _find_capability(mcp, intent, top_k=top_k)
    mux = _fleet_mux(mcp)
    mux.require_capability("discover")
    if selected == "tools":
        query = intent or str(params.get("query") or "")
        return await mux.discover_tools(query, top_k=top_k, loaded=set())
    if selected == "catalog":
        return await mux.list_catalog(
            server=str(params.get("server") or "")[:128],
            include_tools=bool(params.get("include_tools", True)),
        )
    return mux.status_snapshot()


async def _fleet_call(mcp: Any, params: dict[str, Any], execute: bool) -> Any:
    """``act(action="fleet.call")``: call one fleet tool by its catalog name."""
    tool = str(params.get("tool") or "")
    arguments = params.get("arguments") or {}
    if not tool or not isinstance(arguments, dict):
        return {
            "error": "fleet.call needs params {'tool': '<fleet tool>', 'arguments': {...}}."
        }

    async def _run() -> Any:
        mux = _fleet_mux(mcp)
        mux.require_capability("delegate")
        await mux.resolve_and_mount(tools=[tool], servers=None)
        if mux._authority_scope is None:
            result = await mux.call_proxied_tool(tool, arguments)
        else:
            with mux._authority_scope():
                result = await mux.call_proxied_tool(tool, arguments)
        return result.model_dump(mode="json", by_alias=True, exclude_none=True)

    return await _governed(
        "act", FLEET_CALL_ACTION, {"tool": tool, **arguments}, params, execute, _run
    )


async def _fleet_load(mcp: Any, params: dict[str, Any], execute: bool) -> Any:
    """``manage(action="fleet.load")``: mount fleet servers/tools ahead of use."""
    tools = _bounded_names(params, "tools")
    servers = _bounded_names(params, "servers")

    async def _run() -> Any:
        mux = _fleet_mux(mcp)
        mux.require_capability("delegate")
        mounted, available, failed = await mux.resolve_and_mount(
            tools=tools or None, servers=servers or None
        )
        return {"mounted": mounted, "available": available, "failed": failed}

    arguments = {"tools": tools, "servers": servers}
    return await _governed(
        "manage", FLEET_LOAD_ACTION, arguments, params, execute, _run
    )


async def _fleet_unload(mcp: Any, params: dict[str, Any], execute: bool) -> Any:
    """``manage(action="fleet.unload")``: release mounted fleet tools."""
    tools = _bounded_names(params, "tools")

    async def _run() -> Any:
        mux = _fleet_mux(mcp)
        mux.require_capability("delegate")
        released = [name for name in tools if mux.forget_tool(name) is not None]
        return {"released": released}

    return await _governed(
        "manage", FLEET_UNLOAD_ACTION, {"tools": tools}, params, execute, _run
    )


async def _approve(mcp: Any, params: dict[str, Any], execute: bool) -> Any:
    """``manage(action="approve")``: approve one operation for this session."""
    from agent_utilities.mcp.graphos_surface import resolve_operation

    op_id = str(params.get("action") or "")
    if resolve_operation(op_id) is None:
        raise ValueError(
            "approve needs params {'action': '<tool>.<op>'} naming a graph-os operation."
        )

    async def _run() -> Any:
        _record_approval(op_id)
        return {"approved": op_id, "ttl_seconds": int(_APPROVAL_TTL_SECONDS)}

    return await _governed(
        "manage", APPROVE_ACTION, {"action": op_id}, params, execute, _run
    )


async def _host_operation(
    mcp: Any, op_id: str, params: dict[str, Any], execute: bool
) -> Any:
    """``act`` on a host-native backing operation (e.g. graph-os browser control)."""
    from agent_utilities.mcp.graphos_surface import (
        backing_server,
        backing_tools,
        host_operations,
    )
    from agent_utilities.mcp.intent_contract import shape_arguments

    tool_name, action = host_operations(mcp)[op_id]
    call_params = {key: value for key, value in params.items() if key != "plan_ref"}
    schema = backing_tools(mcp)[tool_name].parameters or {}
    arguments = shape_arguments(
        frozenset((schema.get("properties") or {}).keys()),
        call_params,
        action,
        accepts_var_keyword=bool(schema.get("additionalProperties")),
    )

    async def _run() -> Any:
        result = await backing_server(mcp).call_tool(tool_name, arguments)
        return {
            "content": [
                item.model_dump(mode="json", by_alias=True, exclude_none=True)
                for item in result.content
            ],
            "structured_content": result.structured_content,
        }

    return await _governed("act", op_id, arguments, params, execute, _run)


def _find_result(candidate: CapabilityCandidate) -> dict[str, Any]:
    """One ``find`` result: the capability plus how to call it."""
    from agent_utilities.mcp.graphos_surface import group_for_tool

    actions = _actions_by_tool().get(candidate.tool) or []
    example = operation_id(candidate.tool, actions[0]) if actions else candidate.tool
    verbs = [verb for verb in candidate.verbs if verb in _DISPATCH_VERBS] or ["act"]
    return {
        "tool": candidate.tool,
        "action": candidate.action,
        "group": group_for_tool(candidate.tool),
        "verbs": list(candidate.verbs),
        "score": round(candidate.score, 4),
        "matched_terms": candidate.matched_terms,
        "how_to_call": (
            f"{verbs[0]}(intent=<same wording>) to let the router choose, "
            f"or {verbs[0]}(action='{example}', params={{...}}); "
            f"find(action='describe', params={{'action': '{example}'}}) gives its arguments."
        ),
    }


async def _find_capability(mcp: Any, intent: str, top_k: int = 8) -> dict[str, Any]:
    """Capability discovery across every verb, plus a best-effort fleet-wide search.

    ``mcp`` is the live FastMCP server — the fleet multiplexer, when attached,
    is stashed on it as ``mcp._fleet_mux`` so this can widen the search
    fleet-wide without a second multiplexer instance. Absent (embedded/headless
    builds) it degrades to local-only results — never an error.
    """
    security_failure = _intent_security_failure(intent)
    if security_failure is not None:
        return security_failure
    local = resolve_intent(None, intent, top_k=top_k)
    payload: dict[str, Any] = {
        "query_ref": persistence_reference("intent", intent),
        "count": len(local),
        "results": [_find_result(c) for c in local],
    }
    try:
        mux = getattr(mcp, "_fleet_mux", None)
        if mux is not None:
            mux.require_capability("discover")
            discovery = await mux.discover_tools(intent, top_k=top_k, loaded=set())
            payload["fleet_results"] = discovery.get("results", [])
            payload["fleet_unavailable"] = discovery.get("unavailable", {})
    except Exception:  # noqa: BLE001 — remote discovery health is reported elsewhere
        pass
    return payload


async def _lakehouse_status(hints: dict[str, Any]) -> dict[str, Any]:
    """CA-28's ``manage(action="lakehouse_status")`` surface (design: policy-
    bundle epoch = what CA-26 last applied, index-rebuild trigger = CA-24's
    indexer, CDC-lag = CA-21's consumer) — read-only status plus pointing at
    the single trigger action, NOT a new subsystem (this program's non-goal
    for CA-28).

    Each sub-surface degrades to a typed ``"unavailable"`` entry instead of
    raising when its owning lane's real module has not landed yet — a status
    read must never fail closed just because a dependency lane is mid-flight.
    """
    out: dict[str, Any] = {}
    # Policy-bundle epoch — CA-26 extends permission_sync.py with the outbound
    # policy-bundle apply (FO-CA-011); not yet landed as of this lane.
    # ``getattr`` (not ``from ... import name``) deliberately: the function does
    # not exist yet, so a static import would be an unconditional mypy
    # attr-defined error rather than the runtime-optional lookup this status
    # surface needs to degrade gracefully once CA-26 lands.
    try:
        import agent_utilities.protocols.source_connectors.permission_sync as _perm_sync

        epoch_fn = getattr(_perm_sync, "current_policy_bundle_epoch", None)
        if epoch_fn is None:
            out["policy_bundle_epoch"] = {"status": "unavailable", "owner": "CA-26"}
        else:
            out["policy_bundle_epoch"] = epoch_fn()
    except ImportError:
        out["policy_bundle_epoch"] = {"status": "unavailable", "owner": "CA-26"}
    except Exception as exc:  # noqa: BLE001 — status read is best-effort
        out["policy_bundle_epoch"] = {
            "status": "error",
            "detail": type(exc).__name__,
        }
    # Index-rebuild trigger — CA-24 already landed a REAL, working rebuild
    # (`graph_ingest` `action="opensearch_reindex"`, DEC-CA-09): this surfaces
    # it as discoverable status/how-to-call under `manage`, not a second
    # implementation of the rebuild itself.
    out["index_rebuild"] = {
        "status": "available",
        "owner": "CA-24",
        "how_to_call": (
            "graph_ingest(action='opensearch_reindex', "
            "corpus_name=<eg graph/tenant id>)"
        ),
    }
    # CDC-lag — CA-21's Debezium consumer (kafka_adapter.py) has no lag-
    # measurement function yet. Same ``getattr`` rationale as above.
    try:
        import agent_utilities.knowledge_graph.streams.kafka_adapter as _kafka_adapter

        lag_fn = getattr(_kafka_adapter, "consumer_lag_status", None)
        if lag_fn is None:
            out["cdc_lag"] = {"status": "unavailable", "owner": "CA-21"}
        else:
            out["cdc_lag"] = lag_fn()
    except ImportError:
        out["cdc_lag"] = {"status": "unavailable", "owner": "CA-21"}
    except Exception as exc:  # noqa: BLE001 — status read is best-effort
        out["cdc_lag"] = {"status": "error", "detail": type(exc).__name__}
    return out


async def _host_dispatch(
    mcp: Any, verb: str, action: str, params: dict[str, Any], execute: bool
) -> Any:
    """Serve a verb's operation outside the manifest, or ``None`` if not one."""
    from agent_utilities.mcp.graphos_surface import host_operations

    handlers = {
        ("act", FLEET_CALL_ACTION): _fleet_call,
        ("manage", APPROVE_ACTION): _approve,
        ("manage", FLEET_LOAD_ACTION): _fleet_load,
        ("manage", FLEET_UNLOAD_ACTION): _fleet_unload,
    }
    handler = handlers.get((verb, action))
    if handler is not None:
        return await handler(mcp, params, execute)
    if verb == "manage" and action == LAKEHOUSE_STATUS_ACTION:
        status = await _lakehouse_status(params)
        return {"executed": True, "action": action, "status": status}
    if verb == "act" and action in host_operations(mcp):
        return await _host_operation(mcp, action, params, execute)
    return None


def _explicit_operation_hints(
    verb: str, action: str, params: dict[str, Any]
) -> dict[str, Any]:
    """Router hints for ``verb(action=<manifest operation id>)``.

    Raises ``ValueError`` for an unknown operation or one ``verb`` does not
    accept.
    """
    from agent_utilities.mcp.graphos_surface import resolve_operation

    resolved = resolve_operation(action)
    if resolved is None:
        raise ValueError(
            f"Unknown operation {action!r}; {verb}(action='describe') lists them."
        )
    feature = OPTIONAL_TOOL_FEATURES.get(resolved[0])
    if feature and resolved[0] not in kg_server.REGISTERED_TOOLS:
        raise ValueError(
            f"{action!r} needs the optional '{feature}' feature, which this "
            "deployment does not install."
        )
    verbs = operation_verbs().get(action, ())
    if verb not in verbs:
        raise ValueError(
            f"{action!r} is not a {verb} operation; use one of {list(verbs)}."
        )
    return _operation_hints(resolved[0], resolved[1], params)


async def _dispatch_verb(
    mcp: Any,
    verb: str,
    action: str,
    params: dict[str, Any],
    intent: str,
    execute: bool,
) -> Any:
    """The intent contract over the governed router (see module docstring)."""
    if action == DESCRIBE_ACTION:
        return _describe(mcp, verb, params)
    if verb == "find":
        return await _find(mcp, action, params, intent)
    security_failure = _intent_security_failure(intent, params)
    if security_failure is not None:
        return security_failure
    try:
        hosted = (
            await _host_dispatch(mcp, verb, action, params, execute) if action else None
        )
        if hosted is not None:
            return hosted
        hints = _explicit_operation_hints(verb, action, params) if action else params
    except ValueError as exc:
        return {
            "error": exc.args[0] if exc.args else "invalid params",
            "executed": False,
        }
    return await dispatch_intent(verb, intent, hints=hints, execute=execute, mcp=mcp)


def register_intent_tools(mcp: Any) -> list[str]:
    """Register graph-os's six intent tools (CONCEPT:AU-ECO.mcp.intent-surface-condensed-collapse).

    Each takes the ecosystem's condensed contract (``action`` + ``params`` +
    optional ``intent`` + ``execute``; :mod:`agent_utilities.mcp.intent_contract`)
    and runs through :func:`dispatch_intent` — the same governed router and the
    same ``_execute_tool`` core the backing operations use. Every one also gets a
    ``REGISTERED_TOOLS`` entry and a REST twin (``kg_server.ACTION_TOOL_ROUTES``)
    so MCP and REST dispatch it identically.
    """
    from agent_utilities.mcp.intent_contract import make_intent_tool

    registered: list[str] = []
    for verb in INTENT_VERBS:

        def _handler(action, params, intent, execute, _verb=verb):
            return _dispatch_verb(mcp, _verb, action, params, intent, execute)

        async def _run(action, params, intent, execute, _handle=_handler) -> str:
            return json.dumps(
                await _handle(action, params, intent, execute), default=str
            )

        fn = make_intent_tool(
            verb,
            _run,
            actions=FIND_ACTIONS if verb == "find" else None,
            execute_default=verb in _READ_ONLY_VERBS or verb == "find",
        )
        fn.__doc__ = _verb_description(verb)
        mcp.tool(name=verb, tags={"intent"}, description=_verb_description(verb))(fn)
        kg_server.REGISTERED_TOOLS[verb] = fn
        # REST twin (CONCEPT:AU-ECO.mcp.two-surfaces-mcp-rest) — the generic
        # ACTION_TOOL_ROUTES loop in _mount_rest_routes wires this automatically.
        kg_server.ACTION_TOOL_ROUTES[verb] = f"/intent/{verb}"
        registered.append(verb)
    return registered
