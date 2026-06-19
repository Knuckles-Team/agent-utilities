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

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

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


def plan_execution_shape(
    task: str,
    *,
    profile_hint: str | ExecutionProfile | None = None,
    engine: IntelligenceGraphEngine | None = None,
) -> ExecutionProfile:
    """Construct the execution shape for ONE job (CONCEPT:ORCH-1.67).

    This is the single, dynamic entry the orchestrator uses to decide *how much graph* a job
    needs — replacing the static ``"chat"``/``"task"`` preset choice. It runs an **escalating
    planner** (a "classifier for the classifier"): each stage costs more than the last and is
    only reached when the cheaper stage is not confident, so a trivial turn pays only the free
    structural check while a genuinely complex job earns the KG / LLM planning it needs.

      * **Stage 0 — reuse a proven shape** (CONCEPT:ORCH-1.70): if a previously-constructed
        shape for a similar job was persisted (as a skill-workflow) and reused successfully,
        return it. Wired in ORCH-1.70; until shapes are persisted this finds nothing.
      * **Stage 1 — free structural signals** (here): the rules-first classifier
        (``is_trivial_query`` — the single source of truth) shapes a lean direct-completion
        turn vs. the full graph, with no I/O and no LLM.
      * **Stage 2 — cheap KG signals** (CONCEPT:ORCH-1.69): when stage 1 is uncertain, a
        capability designation / policy lookup refines *which* nodes/specialists the job needs.
      * **Stage 3 — LLM planning** (CONCEPT:ORCH-1.69): only for genuinely complex/uncertain
        jobs, an HTN decomposition produces the shape.

    ``profile_hint`` (the entrypoint's altitude, e.g. messaging passes ``"chat"``) seeds the
    timeout budget; the planner then refines the shape from the job itself.
    """
    base = resolve_execution_profile(profile_hint)

    # Stage 1 — free, deterministic structural classifier (single source of truth in
    # ``fast_path``). Imported lazily to keep this module dependency-light.
    from agent_utilities.graph.routing.strategies.fast_path import is_trivial_query

    if is_trivial_query(task or ""):
        # A simple conversational/Q&A turn: answer it directly on a local model and skip the
        # whole heavy apparatus (usage-guard LLM round, KG agent resolution, discovery,
        # verifier). This is the lean shape that lands a chat reply in a human-scale budget.
        return replace(
            base,
            direct_complete=True,
            skip_usage_guard=True,
            run_discovery=False,
            run_verifier=False,
            resolve_agent=False,
            origin="heuristic",
            confidence=0.9,
        )

    # A real task: keep the full multi-agent graph (discovery + verifier on), targeting a
    # specialist (resolve_agent). Stages 2/3 (ORCH-1.69) refine this per job; for now the
    # structural signal carries it.
    return replace(
        base,
        direct_complete=False,
        skip_usage_guard=False,
        run_discovery=True,
        run_verifier=True,
        resolve_agent=True,
        origin="heuristic",
        confidence=0.7,
    )
