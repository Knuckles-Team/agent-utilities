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
destructive operations must be called through their exact dynamically loaded
tool so its client approval policy remains in force.

Outcome learning accepts only the observed result of an unpinned,
unambiguous, policy-authorized execution. Caller-supplied feedback is rejected.
Both reward keys and resolution-cache keys are partitioned by an opaque digest
of the verified tenant and policy revision, preventing cross-tenant or stale-
policy influence. The shared :class:`OutcomeRouter` remains the sole reward-EMA
mechanism; no second learner is introduced.
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
from agent_utilities.mcp.tool_specs import INTENT_VERBS, TOOL_VERBS
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
    "graph_explain": "query",
    "graph_evaluate": "query",
    "graph_observe": "query",
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

#: Soft weight of the learned reward-EMA blend into the lexical score (mirrors
#: ``CapabilityIndex.designate``'s own ``reward_weight`` default) — early on
#: (every candidate at the neutral 0.5 prior) the lexical ranking is
#: untouched; as outcomes accumulate a candidate's blended score rises or
#: sinks with its real success rate under that verb.
_LEARNED_REWARD_WEIGHT = 0.2

#: Lazily constructed shared learner (CONCEPT:AU-ECO.mcp.intent-surface-outcome-learning).
_OUTCOME_ROUTER: Any = None


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
    hints_json = json.dumps(hints or {}, sort_keys=True, default=str, separators=(",", ":"))
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
    missing_cpds = sorted(set(live_tools) - set(cpds))
    if missing_cpds:
        raise RuntimeError(
            "GraphOS capability descriptors are missing for registered tools: "
            + ", ".join(missing_cpds)
        )
    missing_authority = sorted(set(live_tools) - set(TOOL_VERBS))
    if missing_authority:
        raise RuntimeError(
            "GraphOS intent-verb authority is missing registered tools: "
            + ", ".join(missing_authority)
        )
    out: list[CapabilityCandidate] = []
    # Intent verbs have CPDs because they are first-class MCP/REST entry points,
    # but they can never be resolver targets: selecting ``ask`` from inside
    # ``ask`` would recursively dispatch intent routing instead of a capability.
    for tool in live_tools:
        authority_verbs = TOOL_VERBS[tool]
        if (
            not authority_verbs
            or len(set(authority_verbs)) != len(authority_verbs)
            or not set(authority_verbs) <= set(INTENT_VERBS)
        ):
            raise RuntimeError(
                f"GraphOS intent-verb authority is invalid for {tool}: "
                f"{authority_verbs!r}"
            )
        cpd = cpds[tool]
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
                f"{tool}: expected {list(authority_verbs)!r}, "
                f"packaged {packaged_verbs!r}"
            )
        examples_text = " ".join(str(e) for e in (cpd.get("examples") or ()))
        does_text = " ".join(
            str(d.get("action", "")) for d in (cpd.get("does") or ())
        )
        doc = (
            f"{tool} {' '.join(actions_by_tool.get(tool, []))} "
            f"{cpd.get('one_line', '')} {examples_text} {does_text}"
        )
        out.append(
            CapabilityCandidate(
                tool=tool,
                action=None,
                verbs=authority_verbs,
                doc=doc,
            )
        )
    _CANDIDATES_CACHE = out
    _CANDIDATES_GENERATION += 1
    return out


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
        for c in candidates:
            if c.tool == pinned and (verb is None or verb in c.verbs):
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

    outcome_scope_ref = _outcome_scope_ref()
    cache_key = _cache_key(verb, intent, hints, top_k, outcome_scope_ref)
    cached = _RESOLUTION_CACHE.get(cache_key)
    if cached is not None:
        _RESOLUTION_CACHE.move_to_end(cache_key)
        return list(cached)

    intent_tokens = _tokenize(intent)
    pool = candidates if verb is None else [c for c in candidates if verb in c.verbs]
    router = _outcome_router() if outcome_scope_ref is not None else None
    ranked: list[CapabilityCandidate] = []
    for c in pool:
        score, matched = _score(intent_tokens, c)
        task_verb = verb if verb is not None else c.verbs[0]
        reward = (
            router.reward_of(
                _reward_task_class(task_verb, outcome_scope_ref), c.capability_id
            )
            if router is not None and outcome_scope_ref is not None
            else 0.5
        )
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


def _operation_plan(
    verb: str,
    tool: str,
    action: str | None,
    call_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Build a value-free preview from the current CPD operation authority."""

    cpd = _load_cpds_required()[tool]
    operation = _operation_record(cpd, action)
    declared_mutation = _literal_bool(
        operation.get("mutates") if operation is not None else None
    )
    destructive = _operation_is_destructive(tool, action)

    read_prefixes = frozenset(
        {
            "check",
            "count",
            "describe",
            "discover",
            "doctor",
            "explain",
            "export",
            "fetch",
            "find",
            "get",
            "has",
            "history",
            "inspect",
            "list",
            "lookup",
            "metrics",
            "preflight",
            "profile",
            "query",
            "read",
            "recall",
            "report",
            "search",
            "show",
            "status",
            "validate",
            "view",
        }
    )
    action_prefix = str(action or "").split("_", 1)[0]
    tool_has_non_read_policy = bool(set(TOOL_VERBS.get(tool, ())) & _NON_READ_VERBS)
    if declared_mutation is not None:
        mutates: bool | None = declared_mutation
    elif destructive:
        mutates = True
    elif action_prefix in read_prefixes:
        mutates = False
    elif verb in _READ_ONLY_VERBS and not tool_has_non_read_policy:
        mutates = False
    elif verb in _NON_READ_VERBS:
        mutates = True
    else:
        mutates = None

    declared_idempotency = _literal_bool(
        operation.get("idempotent") if operation is not None else None
    )
    if declared_idempotency is None and mutates is False:
        declared_idempotency = True

    policy = cpd.get("policy") if isinstance(cpd.get("policy"), dict) else {}
    approval_class = str(policy.get("approval_class") or "unclassified")
    approval_required = destructive or approval_class != "auto"
    cost = cpd.get("cost") if isinstance(cpd.get("cost"), dict) else {}
    latency = cpd.get("latency") if isinstance(cpd.get("latency"), dict) else {}
    scopes = sorted(str(scope) for scope in (cpd.get("scopes") or ()))
    durability = (
        str(operation.get("durability") or "") if operation is not None else ""
    )
    transaction = (
        str(operation.get("txn_participation") or "")
        if operation is not None
        else ""
    )

    if destructive:
        execution_class = "destructive"
        impact_summary = "May remove or irreversibly invalidate governed state."
    elif mutates is True:
        execution_class = "mutation"
        impact_summary = "Changes governed state within the selected capability scope."
    elif mutates is False:
        execution_class = "read_only"
        impact_summary = "Reads or computes without a declared state mutation."
    else:
        execution_class = "unclassified"
        impact_summary = "Effect metadata is insufficient; execution fails closed."

    return {
        "tool": tool,
        "action": action,
        "execution_class": execution_class,
        "mutates": mutates,
        "destructive": destructive,
        "idempotent": declared_idempotency,
        "preview_required": verb in _NON_READ_VERBS,
        "approval": {
            "class": approval_class,
            "required": approval_required,
            "route": "exact_tool" if approval_required else "intent_policy",
        },
        "impact": {
            "summary": impact_summary,
            "scopes": scopes,
            "durability": durability or "unclassified",
            "transaction": transaction or "unclassified",
        },
        "cost": cost or {"estimate": "not_available"},
        "latency": latency or {"estimate": "not_available"},
        "forwarded_fields": sorted(call_kwargs),
    }


def _plan_ref(
    verb: str, tool: str, action: str | None, call_kwargs: dict[str, Any]
) -> str:
    """Bind a non-read preview to its exact operation and value digest."""

    payload = json.dumps(
        {
            "verb": verb,
            "tool": tool,
            "action": action,
            "arguments": call_kwargs,
        },
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    return persistence_reference("intent_plan", payload)


def _ambiguity_evidence(
    candidates: list[CapabilityCandidate], *, explicit: bool
) -> dict[str, Any]:
    top_score = candidates[0].score if candidates else 0.0
    second_score = candidates[1].score if len(candidates) > 1 else None
    margin = top_score - second_score if second_score is not None else None
    ambiguous = not explicit and (
        top_score <= 0.0
        or (
            second_score is not None
            and second_score > 0.0
            and margin is not None
            and margin < _AMBIGUITY_MARGIN
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
    margin = top_score - second_score if second_score is not None else None
    ambiguous = not explicit and len(ranked_actions) > 1 and (
        top_score <= 0.0
        or (
            second_score is not None
            and second_score > 0.0
            and margin is not None
            and margin < _AMBIGUITY_MARGIN
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
        "candidates": [
            {"action": action, "score": round(score, 4)}
            for action, score in ranked_actions[:5]
        ],
    }


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
    """Classify only the observed tool result; caller feedback is never read."""

    if isinstance(result, dict):
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
    if isinstance(result, str):
        stripped = result.strip()
        if not stripped:
            return False
        try:
            decoded = json.loads(stripped)
        except (TypeError, ValueError):
            lowered = stripped.lower()
            return not lowered.startswith(("error", "failed", "forbidden", "denied"))
        return _execution_succeeded(decoded)
    return result is not None


async def dispatch_intent(
    verb: str,
    intent: str,
    *,
    hints: dict[str, Any] | None = None,
    execute: bool | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Resolve, preview, policy-check, and optionally dispatch one intent.

    ``ask``/``why`` execute by default and remain read-only. Non-read verbs
    preview by default; execution requires the opaque ``plan_ref`` returned by
    that exact preview. Destructive plans are never nested-dispatched because
    doing so would bypass the dynamically loaded exact tool's approval policy.
    Outcome learning consumes only the verified tool result of an unpinned,
    unambiguous execution in the current tenant/policy partition.
    """
    raw_hints = dict(hints or {})
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

    should_execute = verb in _READ_ONLY_VERBS if execute is None else bool(execute)
    explicit_tool = bool(raw_hints.get("tool") or raw_hints.get("_tool"))
    explicit_action = raw_hints.get("action")
    supplied_plan_ref = str(raw_hints.get("plan_ref") or "")
    call_kwargs = {
        k: v for k, v in raw_hints.items() if k not in _CONTROL_HINT_FIELDS
    }

    candidates = resolve_intent(
        verb, intent, hints=raw_hints, top_k=max(2, int(top_k))
    )
    if not candidates:
        return {
            "error": (
                "Pinned capability is not allowed for this intent verb."
                if explicit_tool
                else "No GraphOS capability matched the requested intent verb."
            ),
            "executed": False,
            "routing": {
                "verb": verb,
                "intent_ref": persistence_reference("intent", intent),
                "candidates": [],
            },
        }

    top = candidates[0]
    chosen_tool = top.tool
    available_actions = _actions_by_tool().get(chosen_tool, [])
    if (
        explicit_action is not None
        and available_actions
        and explicit_action not in available_actions
    ):
        return {
            "error": "Requested action is not declared for the pinned capability.",
            "executed": False,
            "routing": {
                "verb": verb,
                "chosen_tool": chosen_tool,
                "declared_actions": sorted(available_actions),
            },
        }
    ranked_actions = _rank_actions(chosen_tool, intent)
    chosen_action = (
        explicit_action
        or top.action
        or (ranked_actions[0][0] if ranked_actions else None)
    )

    text_param = _PRIMARY_TEXT_PARAM.get(chosen_tool)
    fell_back = False
    if text_param and text_param not in call_kwargs:
        call_kwargs[text_param] = intent
    elif (
        not call_kwargs
        and chosen_tool not in _PRIMARY_TEXT_PARAM
        and verb == "ask"
        and not raw_hints.get("tool")
        and not raw_hints.get("_tool")
        and explicit_action is None
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

    if chosen_action is not None:
        call_kwargs.setdefault("action", chosen_action)

    candidate_ambiguity = _ambiguity_evidence(candidates, explicit=explicit_tool)
    action_ambiguity = _action_ambiguity_evidence(
        ranked_actions, explicit=explicit_action is not None
    )
    plan = _operation_plan(verb, chosen_tool, chosen_action, call_kwargs)
    plan_ref = _plan_ref(verb, chosen_tool, chosen_action, call_kwargs)
    plan["plan_ref"] = plan_ref

    outcome_scope_ref = _outcome_scope_ref()
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
    routing: dict[str, Any] = {
        "verb": verb,
        "intent_ref": persistence_reference("intent", intent),
        "chosen_tool": chosen_tool,
        "action": chosen_action,
        "score": round(top.score, 4),
        "matched_terms": top.matched_terms,
        "fell_back_to_nl_planner": fell_back,
        "why": (
            f"'{top.tool}' best matched the {verb!r} intent on descriptor terms "
            f"{top.matched_terms!r}"
            if top.matched_terms
            else f"'{top.tool}' is the highest-ranked capability for verb {verb!r}"
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
                "matched_terms": top.matched_terms,
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

    read_policy_violation = verb in _READ_ONLY_VERBS and plan["mutates"] is not False
    if not should_execute:
        return {"routing": routing, "executed": False}

    if read_policy_violation:
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
    if plan["approval"]["required"]:
        return {
            "error": (
                "Operation requires the exact dynamically loaded tool and its "
                "approval policy."
            ),
            "routing": routing,
            "executed": False,
            "approval_required": True,
        }

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


async def _find_capability(mcp: Any, intent: str, top_k: int = 8) -> dict[str, Any]:
    """``find`` — capability discovery across every verb, plus a best-effort fleet-wide search.

    ``mcp`` is the live FastMCP server (closure-captured by the caller,
    :func:`register_intent_tools`) — the fleet multiplexer, when attached
    (:func:`agent_utilities.mcp.multiplexer.attach_fleet_loader`), is stashed on
    it as ``mcp._fleet_mux`` so this can widen the search fleet-wide without a
    second multiplexer instance. Absent (embedded/headless builds) it degrades
    to local-only results — never an error.
    """
    security_failure = _intent_security_failure(intent)
    if security_failure is not None:
        return security_failure
    local = resolve_intent(None, intent, top_k=top_k)
    payload: dict[str, Any] = {
        "query_ref": persistence_reference("intent", intent),
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
    except Exception:  # noqa: BLE001 — remote discovery health is reported elsewhere
        pass
    return payload


#: ``manage`` hints ``{"action": ...}`` values that route to the tool-lifecycle
#: (load/unload) core instead of the granular-tool resolver (CONCEPT:AU-ECO.mcp.intent-surface-tool-lifecycle) —
#: responsible tool usage is a MANAGE concern (the same verb that owns
#: configure/tenants/lifecycle), not a new seventh verb.
_RECLAIM_ACTIONS = frozenset({"unload", "reclaim", "load"})


async def _manage_lifecycle(
    mcp: Any, hints: dict[str, Any], *, execute: bool = False
) -> dict[str, Any] | None:
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
    plan_hints = {k: v for k, v in hints.items() if k != "plan_ref"}
    plan_ref = persistence_reference(
        "intent_plan",
        json.dumps(
            {"verb": "manage", "lifecycle": plan_hints},
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        ),
    )
    plan = {
        "action": action,
        "execution_class": "session_visibility",
        "mutates": False,
        "destructive": False,
        "idempotent": True,
        "preview_required": True,
        "plan_ref": plan_ref,
        "approval": {"required": False, "route": "dynamic_tool_policy"},
    }
    if not execute:
        return {"executed": False, "plan": plan}
    if str(hints.get("plan_ref") or "") != plan_ref:
        return {
            "executed": False,
            "error": (
                "Preview required: review the lifecycle plan and resubmit its "
                "plan_ref with execute=true."
            ),
            "plan": plan,
        }
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

    Called by ``kg_server._build_server`` when the default
    ``MCP_TOOL_MODE=="intent"`` profile is active. Each tool is a thin
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
                    '{"tool": "graph_write"} to pin the exact tool). For a '
                    "non-read execution, resubmit the preview's plan_ref here."
                ),
            ),
            execute: bool = Field(
                default=verb in _READ_ONLY_VERBS,
                description=(
                    "Execute a read-only plan immediately. Non-read verbs default "
                    "to preview and require the returned plan_ref before execution."
                ),
            ),
        ) -> str:
            hints = json.loads(hints_json) if hints_json else {}
            if verb == "manage":
                security_failure = _intent_security_failure(intent, hints)
                if security_failure is not None:
                    return json.dumps(security_failure, default=str)
                lifecycle = await _manage_lifecycle(mcp, hints, execute=execute)
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
            "Preview or perform a natural-language WRITE/ingest intent. Structured fields "
            "go in hints_json. Preview returns effect, cost/impact, idempotency, ambiguity, "
            "approval classification, and a plan_ref required for execution."
        ),
        "act": (
            "Preview or perform a natural-language ACT/execute intent. The resolver returns "
            "a governed plan first; review it and resubmit its plan_ref to execute. Ambiguous "
            "or unclassified operations fail closed."
        ),
        "manage": (
            "Preview or perform a natural-language MANAGE/configure intent. Review and "
            "resubmit the returned plan_ref before execution. RESPONSIBLE TOOL USAGE "
            "(CONCEPT:AU-ECO.mcp.intent-surface-tool-lifecycle): "
            'pass hints_json={"action": "unload", "tools": [...]} (or "servers"/'
            '"toolsets") to reclaim context from previously loaded granular tools, or '
            '{"action": "load", "tools": [...], "auto_unload": true} to pull one in '
            "for a single one-shot use — it auto-retracts after its next call so long "
            "sessions don't accumulate tool schemas. Dynamically loaded exact tools retain "
            "their own authority and approval policy."
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
