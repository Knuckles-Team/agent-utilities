"""Per-entrypoint execution profiles (CONCEPT:ORCH-1.62).

The universal orchestration path (``Orchestrator.execute_agent`` → ``run_agent`` →
``_build_execution_config`` → ``_execute_graph``) is the ONE path every entrypoint
feeds. But a single chat turn and a multi-step task want very different *altitudes*:
a chat turn must answer inside a human-scale budget (tens of seconds), while a task
may legitimately run several specialist LLM rounds each bounded by the long default
node timeout.

Historically both used ``DEFAULT_GRAPH_ROUTER_TIMEOUT``/``DEFAULT_GRAPH_VERIFIER_TIMEOUT``
(300 s each). On a degraded backend, the first router round of a chat turn alone could
stall for the full 300 s — far above the messaging reply budget, which then killed the
run and made *another* slow LLM call via the plain-chat fallback (the measured >90 s).

An ``ExecutionProfile`` selects the node-timeout budget (and a couple of altitude flags)
for a given entrypoint. It is built once here and threaded — *not* re-implemented per
surface — so every entrypoint inherits the same contract (Universal capability):

* ``task`` (default) — the existing long node timeouts; unchanged behaviour for the
  full multi-agent orchestration of a real task.
* ``chat`` — node timeouts bounded to the chat budget (router/verifier ≈ 12 s, total
  ≤ the messaging reply timeout), fast-path eligible, and a *cheap* fallback so a
  degraded backend yields ONE fast bounded attempt + a graceful message, never a
  second full LLM call.

The messaging reply path passes ``"chat"``; ``graph_orchestrate``/CLI/A2A callers keep
the default ``"task"``.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)

# The messaging reply path caps the whole turn at MESSAGING_REPLY_TIMEOUT (default 45 s).
# The chat-profile node budget MUST be far below that so a turn resolves *inside* the
# graph instead of being killed and retried via the plain-chat fallback. We read the
# reply timeout via ``setting`` at resolve time so a deployment that raises/lowers it
# keeps the node budget aligned.
_DEFAULT_MESSAGING_REPLY_TIMEOUT = 45.0

# Chat-profile per-node budget. Each sequential LLM round (router, verifier, …) is bounded
# to this so even a few rounds stay inside the reply budget; on a degraded backend a single
# stalled round fails fast at this bound instead of the 300 s default.
CHAT_NODE_TIMEOUT_S = 12.0


@dataclass(frozen=True)
class ExecutionProfile:
    """A bundle of altitude/timeout settings for one entrypoint class.

    Attributes:
        name: ``"chat"`` or ``"task"``.
        router_timeout: Per-node timeout for the router LLM round (seconds), or ``None``
            to use the long default. The dispatcher/expert/verifier/synthesizer node
            timeouts derive from this + ``verifier_timeout``.
        verifier_timeout: Per-node timeout for the verifier (+repair) LLM round (seconds),
            or ``None`` for the long default.
        fast_path_eligible: When True the router may short-circuit a simple turn onto the
            single-round fast path (CONCEPT:ORCH-1.63) instead of the full graph.
        cheap_fallback: When True a graph timeout/error must NOT trigger a second *full*
            LLM call to the same (possibly degraded) endpoint; the fallback is a single
            short bounded attempt and a graceful message (CONCEPT:ORCH-1.62, removes the
            double-LLM tax).

    CONCEPT:ORCH-1.67 — the profile is no longer a fixed per-entrypoint *preset*; it is the
    **dynamically-constructed execution shape** for ONE job. ``plan_execution_shape`` builds
    it per task from cheap signals (escalating to the KG / an LLM planner only when the job
    is uncertain), so a trivial turn gets a lean shape (``direct_complete`` on a local model,
    no usage-guard LLM round, no discovery/verifier, no KG agent resolution) while a real task
    gets the full graph — same path, dynamically shaped. The shape fields below let each graph
    node decide whether to run its work or pass through for this job (CONCEPT:ORCH-1.68).

    Shape attributes:
        direct_complete: Answer the turn with ONE lite/local-model completion and skip the
            router/dispatcher/verifier graph entirely.
        skip_usage_guard: Skip the usage-guard policy-LLM round for this job.
        run_discovery: Run the router's pre-LLM KG discovery bundle (specialist/tool/policy
            lookup) for this job.
        run_verifier: Run the verifier (+repair) round for this job.
        resolve_agent: Resolve the agent name against the KG (a semantic search) before the
            run; ``False`` skips it for a job that doesn't target a specific specialist.
        enable_reasoning: Run the model with extended reasoning ("thinking" / RLM-style
            recursive reasoning) ENABLED for this job. A trivial turn turns it off so the
            local reasoning model answers in ~0.4 s instead of emitting a multi-second
            chain-of-thought trace; a job that benefits from deliberation turns it on. This is
            a per-job *capability* toggle on the model — the first of the dynamic model/agent
            capabilities the planner selects (CONCEPT:ORCH-1.68).
        model_id: Per-job model override; ``None`` lets ``create_model`` pick the local
            default. Never a hard-coded remote model.
        origin: Which planner stage produced this shape (``preset`` / ``heuristic`` /
            ``designate`` / ``llm`` / ``cache:<id>``) — provenance for the learning loop.
        confidence: The planner's confidence in this shape, in ``[0, 1]``; low confidence is
            what triggers escalation to the next planning stage.
    """

    name: str
    router_timeout: float | None
    verifier_timeout: float | None
    fast_path_eligible: bool
    cheap_fallback: bool
    # CONCEPT:ORCH-1.67 — dynamic per-job shape (all default to the prior full-graph behaviour
    # so existing constructions are unchanged; the planner opts a job into the lean shape).
    direct_complete: bool = False
    skip_usage_guard: bool = False
    run_discovery: bool = True
    run_verifier: bool = True
    resolve_agent: bool = True
    enable_reasoning: bool = True
    model_id: str | None = None
    origin: str = "preset"
    confidence: float = 1.0

    @property
    def is_chat(self) -> bool:
        return self.name == "chat"


def _messaging_reply_timeout() -> float:
    """The configured messaging reply budget (live), default 45 s."""
    from agent_utilities.core.config import setting

    try:
        return float(
            setting("MESSAGING_REPLY_TIMEOUT", str(_DEFAULT_MESSAGING_REPLY_TIMEOUT))
        )
    except Exception:  # noqa: BLE001 — bad value → safe default
        return _DEFAULT_MESSAGING_REPLY_TIMEOUT


def resolve_execution_profile(
    profile: str | ExecutionProfile | None,
) -> ExecutionProfile:
    """Resolve a profile name (or instance) into a concrete :class:`ExecutionProfile`.

    ``None`` / unknown / ``"task"`` → the default long-timeout task profile (unchanged
    behaviour). ``"chat"`` → the chat-budget profile whose node timeouts are bounded to
    the chat budget (≤ ``CHAT_NODE_TIMEOUT_S`` per node, total under the reply timeout).
    """
    if isinstance(profile, ExecutionProfile):
        return profile

    if profile == "chat":
        # Keep each node well under the reply budget; if the reply budget is set lower than
        # our nominal node budget, shrink the node budget so a single round still fits.
        reply_budget = _messaging_reply_timeout()
        node_timeout = min(CHAT_NODE_TIMEOUT_S, max(reply_budget - 3.0, 4.0))
        return ExecutionProfile(
            name="chat",
            router_timeout=node_timeout,
            verifier_timeout=node_timeout,
            fast_path_eligible=True,
            cheap_fallback=True,
        )

    # Default: the long-timeout task profile (None timeouts → callers use the long defaults).
    return ExecutionProfile(
        name="task",
        router_timeout=None,
        verifier_timeout=None,
        fast_path_eligible=True,
        cheap_fallback=False,
    )


# ── Recipe cache (CONCEPT:ORCH-1.70) ─────────────────────────────────────────────────
# The expensive part of planning a shape is the KG/LLM resolution (stages 2/3); an identical
# job should REUSE the proven recipe, not recompute it. Bounded in-process LRU keyed by a
# normalized job signature. The durable cross-process layer (persisting recipes as
# AgentTemplates + reuse on success) is the learning loop (CONCEPT:ORCH-1.70).
_RECIPE_CACHE: OrderedDict[str, ExecutionProfile] = OrderedDict()
_RECIPE_CACHE_MAX = 512

# Lean (trivial/conversational) vs full (real multi-step/specialist) field overrides.
_LEAN_FIELDS: dict[str, Any] = dict(
    direct_complete=True,
    skip_usage_guard=True,
    run_discovery=False,
    run_verifier=False,
    resolve_agent=False,
    enable_reasoning=False,
)
_FULL_FIELDS: dict[str, Any] = dict(
    direct_complete=False,
    skip_usage_guard=False,
    run_discovery=True,
    run_verifier=True,
    resolve_agent=True,
    enable_reasoning=True,
)


def _job_signature(task: str, profile_hint: str | ExecutionProfile | None) -> str:
    """A stable cache key for a job: its normalized word-set + the entrypoint altitude."""
    words = sorted(set(re.findall(r"[a-z0-9]+", (task or "").lower())))
    digest = hashlib.sha1(" ".join(words).encode("utf-8")).hexdigest()[:16]
    hint = (
        profile_hint.name
        if isinstance(profile_hint, ExecutionProfile)
        else str(profile_hint or "")
    )
    return f"{hint}|{digest}"


def _recipe_cache_get(sig: str) -> ExecutionProfile | None:
    shape = _RECIPE_CACHE.get(sig)
    if shape is not None:
        _RECIPE_CACHE.move_to_end(sig)
    return shape


def _recipe_cache_put(sig: str, shape: ExecutionProfile) -> None:
    _RECIPE_CACHE[sig] = shape
    _RECIPE_CACHE.move_to_end(sig)
    while len(_RECIPE_CACHE) > _RECIPE_CACHE_MAX:
        _RECIPE_CACHE.popitem(last=False)


def reset_recipe_cache() -> None:
    """Clear the in-process recipe cache (tests; a deployment wanting a cold planner)."""
    _RECIPE_CACHE.clear()


def record_shape_outcome(
    task: str, profile_hint: str | ExecutionProfile | None, *, success: bool
) -> None:
    """Close the learning loop on a planned recipe (CONCEPT:ORCH-1.70).

    The planner caches a shape at plan-time; this folds the RUN RESULT back in. A successful
    run leaves the recipe cached (it is reused for the next identical job — reinforcement); a
    failed run EVICTS it, so the next identical job re-plans from scratch instead of blindly
    repeating a shape that did not work. Cheap, in-process, and best-effort — never raises into
    the caller's result path.
    """
    if success:
        return  # the cached entry stays (reused next time)
    try:
        _RECIPE_CACHE.pop(_job_signature(task, profile_hint), None)
    except Exception as e:  # noqa: BLE001
        logger.debug("[ORCH-1.70] recipe outcome record skipped: %s", e)


def _resolve_job_capabilities(
    engine: IntelligenceGraphEngine, task: str, *, top_k: int = 8
) -> list[dict[str, Any]] | None:
    """Stage-2 job→capability search, routed through the Rust engine (CONCEPT:ORCH-1.69).

    Uses the engine's ``search_hybrid`` (``HybridRetriever`` → ``graph_compute`` → the Rust
    tokio/MessagePack engine over UDS) instead of a per-process Python HNSW cold-build — ~15×
    faster live (~4.5 s incl. the query embedding vs >70 s for the Python ``CapabilityIndex``).
    Returns the hit list (possibly empty), or ``None`` when search is unavailable (so the caller
    keeps the safe full-graph default rather than mistaking "unavailable" for "no match").
    """
    try:
        hits = engine.search_hybrid(task, top_k=top_k)
        return list(hits or [])
    except Exception as e:  # noqa: BLE001 — search must never break planning
        logger.debug("[ORCH-1.69] stage-2 capability search unavailable: %s", e)
        return None


def _refine_with_kg(
    engine: IntelligenceGraphEngine, task: str, base: ExecutionProfile
) -> ExecutionProfile:
    """Stage 2 — disambiguate the *ambiguous middle* with a cheap, Rust-routed capability search."""
    hits = _resolve_job_capabilities(engine, task)
    if hits is None:
        # Search unavailable → keep the full graph (safe default for an action-shaped turn).
        return replace(base, **_FULL_FIELDS, origin="heuristic", confidence=0.6)
    if hits:
        # The job maps to real fleet capabilities → it IS a tool task; keep the full graph.
        return replace(base, **_FULL_FIELDS, origin="designate", confidence=0.8)
    # Search succeeded but found NOTHING relevant → the borderline turn is conversational after
    # all; downgrade to the lean shape so it gets the fast path instead of the full graph.
    return replace(base, **_LEAN_FIELDS, origin="designate-empty", confidence=0.7)


def plan_execution_shape(
    task: str,
    *,
    profile_hint: str | ExecutionProfile | None = None,
    engine: IntelligenceGraphEngine | None = None,
) -> ExecutionProfile:
    """Construct the execution shape for ONE job (CONCEPT:ORCH-1.67/1.69/1.70).

    The single, dynamic entry the orchestrator uses to decide *how much graph* a job needs —
    replacing the static ``"chat"``/``"task"`` preset. It runs an **escalating planner** (a
    "classifier for the classifier"): each stage costs more than the last and is reached only
    when the cheaper stage is not confident, so a trivial turn pays only the free structural
    check while a genuinely complex job earns the KG search it needs.

      * **Stage 0 — reuse a cached recipe** (CONCEPT:ORCH-1.70): an identical job returns its
        cached shape, skipping all resolution. (Durable cross-process reuse = the learning loop.)
      * **Stage 1 — free structural signals**: the graded ``orchestration_signal_strength``
        (single source of truth in ``fast_path``). Strength 0 → confident lean; ≥2 → confident
        full; **1 → the ambiguous middle**, escalated to stage 2.
      * **Stage 2 — cheap, Rust-routed KG search** (CONCEPT:ORCH-1.69): only the ambiguous
        middle pays this — ``search_hybrid`` disambiguates tool-task vs. conversational.
      * **Stage 3 — LLM planning** (CONCEPT:ORCH-1.69, planned): genuinely complex/uncertain
        jobs earn an HTN decomposition.

    ``profile_hint`` (the entrypoint altitude, e.g. messaging passes ``"chat"``) seeds the
    timeout budget; the planner refines the shape from the job itself.
    """
    base = resolve_execution_profile(profile_hint)
    sig = _job_signature(task, profile_hint)

    # Stage 0 — reuse a cached recipe.
    cached = _recipe_cache_get(sig)
    if cached is not None:
        return replace(cached, origin=f"cache:{cached.origin}")

    # Stage 1 — free, deterministic structural classifier (single source of truth in
    # ``fast_path``). Imported lazily to keep this module dependency-light.
    from agent_utilities.graph.routing.strategies.fast_path import (
        orchestration_signal_strength,
    )

    strength = orchestration_signal_strength(task or "")
    if strength == 0:
        # Clearly lean — a simple conversational/Q&A turn answered directly on a local model.
        shape = replace(base, **_LEAN_FIELDS, origin="heuristic", confidence=0.9)
    elif strength >= 2 or engine is None:
        # Clearly a real task (or no engine to disambiguate): the full multi-agent graph.
        shape = replace(
            base,
            **_FULL_FIELDS,
            origin="heuristic",
            confidence=0.9 if strength >= 2 else 0.6,
        )
    else:
        # Stage 2 — the ambiguous middle (strength == 1): cheap, Rust-routed disambiguation.
        shape = _refine_with_kg(engine, task, base)

    _recipe_cache_put(sig, shape)
    return shape
