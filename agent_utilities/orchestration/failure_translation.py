"""Error-translation registry for messaging-orchestration transparency.

CONCEPT:AU-ORCH.execution.messaging-orchestration-transparency — every non-``ok`` terminal
``run_summary`` (:mod:`agent_utilities.orchestration.agent_runner`'s ``run_agent``) carries a
``failure`` detail built here: ``{raw, translated, category, hint}``. This module is the SINGLE
place that maps an internal error signature (an exception message, a tool-error string, a
log-line fragment) to a short, chat-appropriate explanation plus an actionable hint — so a chat
user (or an operator reading the same text in a log/trace) gets a real troubleshooting entry
point instead of a bare "something failed".

Design:
  * **Data-driven.** :data:`_SIGNATURES` is an ordered tuple of :class:`_Signature` — extend the
    registry by adding an entry, no branching logic to touch. Matching is a case-insensitive
    SUBSTRING test against the raw error text (defensive: exact exception wording drifts across
    pydantic-ai/httpx/anyio versions; a substring survives that). Entries are checked in order —
    more specific markers are placed before more generic ones so a narrow signature is never
    shadowed.
  * **Never a bare "failure".** :func:`translate_failure` always returns a non-empty, specific
    ``translated`` string. An unmatched error falls through to :func:`_generic_fallback`, which
    still surfaces the (sanitized) raw error tail — the one hard requirement of this module: a
    chat reply must never be LESS informative than the raw text already sitting in the log.
  * **Privacy-safe.** :func:`build_failure_detail` sanitizes the raw text (via
    ``core.log_privacy.sanitize_log_text``, the same redactor the process-wide log boundary
    uses) before it is stored on the run_summary — this failure detail can ride all the way out
    to an external chat surface (Telegram et al.), a strictly MORE exposed boundary than the
    internal logs that sanitizer was built for, so the same endpoint/path/email redaction
    applies here too.

Cross-reference: the fleet HTTPS gate (``agent_utilities/orchestration/agent_runner.py``
``_fleet_server_url``), the multiplexer's child-tool-error re-raise
(``agent_utilities/mcp/multiplexer.py``), and the retrieval quality gate
(``agent_utilities/knowledge_graph/retrieval/retrieval_quality.py``) are the concrete sources
several of these signatures were reverse-engineered from — see each ``_Signature.markers`` tuple
for the literal string it targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "FailureTranslation",
    "build_failure_detail",
    "translate_failure",
]


@dataclass(frozen=True)
class FailureTranslation:
    """One resolved translation: what to say, its class, and what to do about it."""

    translated: str
    category: str
    hint: str


@dataclass(frozen=True)
class _Signature:
    category: str
    markers: tuple[str, ...]  # case-insensitive substrings; any() match wins
    translated: str
    hint: str


# Ordered MOST-specific -> least-specific: the first matching signature wins, so a narrow
# marker (e.g. "wall-clock budget") is placed before any broader one that could also appear
# in its own message.
_SIGNATURES: tuple[_Signature, ...] = (
    _Signature(
        category="fleet_https_gate",
        markers=("requires https outside loopback",),
        translated=(
            "The fleet gateway rejected a plain-HTTP connection outside loopback "
            "(TLS is required for any non-local endpoint)."
        ),
        hint="Ask an operator to put that server behind HTTPS, or run it on loopback.",
    ),
    _Signature(
        category="fleet_config_missing",
        markers=(
            "fleet_mcp_url_template must contain",
            "requires fleet_mcp_url_template",
        ),
        translated="The fleet gateway has no URL configured for that server.",
        hint="An operator needs to set FLEET_MCP_URL_TEMPLATE.",
    ),
    _Signature(
        category="fleet_endpoint_invalid",
        markers=("fleet mcp endpoint is invalid",),
        translated="That fleet server's configured endpoint URL is malformed.",
        hint="An operator needs to fix its URL in mcp_config.json.",
    ),
    _Signature(
        category="fleet_tool_error",
        markers=("delegated_child_tool_failed",),
        translated="A fleet tool call failed inside its own MCP server.",
        hint="Check that server's own logs/health — the gateway only sees a generic failure.",
    ),
    _Signature(
        category="tool_wall_clock_timeout",
        markers=("wall-clock budget",),
        translated="A bound tool call blocked and exceeded its execution time limit.",
        hint="That tool/service likely hung — check its health/logs, then retry.",
    ),
    _Signature(
        category="toolset_bind_failed",
        markers=(
            "could not be bound",
            "no bound toolset to invoke",
            "does not support tool filtering",
        ),
        translated="The delegated agent's tools failed to bind before it could run.",
        hint="Configuration/registration gap, not a bad query — check its declared toolsets.",
    ),
    _Signature(
        category="retrieval_quality",
        markers=(
            "low_relevance_topk",
            "composite=0.00",
            "retrieval quality gate failed",
        ),
        translated="No relevant knowledge has been ingested yet for this query.",
        hint="Ingest/sync the relevant source, or rephrase — nothing cleared the relevance bar.",
    ),
    _Signature(
        category="reply_budget_timeout",
        markers=("reply budget", "reply-budget"),
        translated="The assistant backend did not finish inside the reply time budget.",
        hint="Often a slow/overloaded backend or a blocked tool call — try again shortly.",
    ),
    _Signature(
        category="engine_unreachable",
        markers=(
            "engineunreachableerror",
            "engine unreachable",
            "no kg backend active",
            "no backend active",
            "connection refused",
            "connectionrefusederror",
            "failed to connect",
        ),
        translated="The knowledge-graph engine backing this request is unreachable.",
        hint="Infrastructure issue (the engine process/connection), not a bad query.",
    ),
    _Signature(
        category="permission_bootstrap",
        markers=(
            "permissionbootstraperror",
            "permission signing key",
            "permission context bootstrap",
            "permission context verification",
            "governed execution",
        ),
        translated=(
            "Governed tool execution could not establish a signed permission "
            "identity (the permission signing key could not be resolved or "
            "provisioned)."
        ),
        hint=(
            "The signing key self-provisions into the engine's durable secret store "
            "by default; if this persists, confirm the engine secret store is "
            "reachable, or set PERMISSIONS_SIGNING_KEY_REF to an external key."
        ),
    ),
    _Signature(
        category="access_denied",
        markers=(
            "access_denied",
            "access denied",
            "permission denied",
            "permissionerror",
            "not authorized",
            "unauthorized",
            "forbidden by policy",
        ),
        translated="This action was denied by an access-control policy.",
        hint="Not transient — check the caller/service identity's RBAC/ActionPolicy grant.",
    ),
    _Signature(
        category="security_blocked",
        markers=("security: ",),
        translated="This request was blocked by the prompt-injection/security guard.",
        hint="If this looks like a false positive, rephrase; repeated blocks need a policy review.",
    ),
)


def _coerce_text(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, BaseException):
        return f"{type(raw).__name__}: {raw}".strip()
    return str(raw)


def translate_failure(raw: Any) -> FailureTranslation:
    """Map an internal error signature to a chat-appropriate translation.

    ``raw`` may be a string, ``None``, or a :class:`BaseException` (rendered as
    ``"<Type>: <message>"``, mirroring ``agent_runner._flatten_exception_group``'s own
    convention). Matches against :data:`_SIGNATURES` by case-insensitive substring, first
    match wins. An unmatched (or empty) error never returns a bare "failure" — see
    :func:`_generic_fallback`.
    """
    text = _coerce_text(raw)
    low = text.lower()
    for sig in _SIGNATURES:
        if any(marker in low for marker in sig.markers):
            return FailureTranslation(
                translated=sig.translated, category=sig.category, hint=sig.hint
            )
    return _generic_fallback(text)


def _generic_fallback(text: str) -> FailureTranslation:
    from agent_utilities.core.log_privacy import sanitize_log_text

    tail = sanitize_log_text(text).strip()
    if len(tail) > 240:
        tail = tail[-240:]
    translated = (
        f"An internal step failed ({tail})"
        if tail
        else "An internal step failed, with no error detail captured."
    )
    return FailureTranslation(
        translated=translated,
        category="unknown",
        hint="No known pattern matched this error — see the trace for full context.",
    )


def build_failure_detail(raw: Any) -> dict[str, str]:
    """The ready-to-store ``run_summary.failure`` dict: ``{raw, translated, category, hint}``.

    ``raw`` is privacy-sanitized (``core.log_privacy.sanitize_log_text``) and length-capped
    before storage — this detail can ride all the way out to an external chat surface, a more
    exposed boundary than the internal logs that sanitizer targets.
    """
    from agent_utilities.core.log_privacy import sanitize_log_text

    text = _coerce_text(raw)
    translation = translate_failure(text)
    return {
        "raw": sanitize_log_text(text)[:2000],
        "translated": translation.translated,
        "category": translation.category,
        "hint": translation.hint,
    }
