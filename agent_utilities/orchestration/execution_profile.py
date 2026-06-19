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

from dataclasses import dataclass

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
    """

    name: str
    router_timeout: float | None
    verifier_timeout: float | None
    fast_path_eligible: bool
    cheap_fallback: bool

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
