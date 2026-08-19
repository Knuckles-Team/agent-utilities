"""Provider-native exact prompt caching — adopt, don't reimplement.

CONCEPT:AU-ORCH.optimization.provider-prompt-cache — an exhaustive audit of this repo found no
``PromptCache``/``ResponseCache`` class and no Anthropic ``cache_control`` wiring anywhere. This
module closes that gap the way the constitution requires: it does NOT reimplement byte-exact
prefix caching (Anthropic/OpenAI already run that server-side, correctly, for free) — it turns it
ON by default and makes the savings measurable.

What "adopt" means concretely
------------------------------
pydantic-ai 2.29.0 (the supported exact release) already ships first-class settings for both providers:

* Anthropic — ``AnthropicModelSettings.anthropic_cache`` / ``anthropic_cache_instructions`` /
  ``anthropic_cache_tool_definitions`` translate directly into ``cache_control: {"type":
  "ephemeral"}`` breakpoints on the wire (see ``anthropic.types.cache_control_ephemeral_param``).
* OpenAI — automatic prefix caching applies server-side above ~1024 tokens with NO client action
  required; ``OpenAIChatModelSettings.openai_prompt_cache_key`` is an optional routing hint that
  improves hit-rate CONSISTENCY for repeat traffic from the same tenant/policy (verified against
  the installed ``openai`` 2.46.0 SDK's ``ChatCompletionCreateParams.prompt_cache_key`` field —
  it does not change WHAT gets cached, only how OpenAI's infra clusters cache-eligible requests).

Neither provider needs a locally-stored cache: the "cache" IS the provider's own KV/prefix reuse.
This module's job is (1) turn the settings on, tenant/policy-scoped, and (2) surface the resulting
``cache_read_tokens``/``cache_write_tokens`` (already normalized across providers by pydantic-ai's
``RequestUsage``) into agent-utilities' own trace/telemetry so the savings are measurable — see
:func:`fold_prompt_cache_hint` (the live per-call seam, wired into
``agent_utilities.agent.sampling_profile.attach_profile_resolver`` exactly like the sibling
``kvcache.policy.fold_kv_hint``) and :func:`record_prompt_cache_usage` (wired into
``agent_utilities.harness.tracing``'s always-on per-call trace wrapper).

Cache-key discipline (why folding is tenant/policy-scoped, not just "on")
---------------------------------------------------------------------------
:class:`PromptCacheKey` mirrors :class:`~agent_utilities.kvcache.checkpoint.KVCheckpointKey`'s
discipline: tenant, principal, policy_version, model_identity, prompt_fingerprint (a content hash
of the stable prefix — analogous to ``StructuredPrompt.version_hash()``), and safety_posture
(``TOOL_GUARD_MODE``) all feed :attr:`PromptCacheKey.fingerprint`. Unlike ``KVCheckpointKey`` this
is NOT a security boundary — the "cache" being scoped lives entirely on the provider's own infra
and never returns provider-cached CONTENT across a changed prefix (a byte-different prompt simply
misses; there is no data path through which one tenant could read another tenant's cached
response). The scoping exists so the ``openai_prompt_cache_key`` routing hint stays a stable,
low-cardinality string per (tenant, policy, model, prompt-shape) rather than one global constant,
which is the documented best practice for maximizing OpenAI's own cache hit-rate.

Default-on, with one opt-out
-----------------------------
Cheap and safe (Anthropic cache writes cost 1.25x tokens once, reads cost 0.1x forever after;
OpenAI's own caching is free either way), so this is folded on EVERY agent call by default — the
single env var ``AU_PROMPT_CACHE`` (default ``True``) is the one opt-out for a real disable case
(e.g. a provider/gateway that rejects unknown request fields).
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_utilities.core.config import setting

logger = logging.getLogger(__name__)

__all__ = [
    "PromptCacheKey",
    "fold_prompt_cache_hint",
    "prompt_cache_create_kwargs",
    "record_prompt_cache_usage",
    "resolve_prompt_cache_key",
    "usage_cache_tokens",
]

#: Field separator for :attr:`PromptCacheKey.fingerprint` — mirrors
#: ``kvcache.checkpoint._KEY_SEPARATOR`` (a non-printable byte no legitimate identifier contains).
_KEY_SEPARATOR = "\x1f"

#: OpenAI's ``prompt_cache_key`` request field has a practical length ceiling; a 32-hex-char
#: prefix of a sha256 digest is already collision-safe for this use (a low-cardinality ROUTING
#: hint, not a security token) and comfortably under it.
_OPENAI_CACHE_KEY_LEN = 32


class PromptCacheKey(BaseModel):
    """Every component that scopes a provider prompt-cache hint (CONCEPT:AU-ORCH.optimization.provider-prompt-cache).

    Not a security boundary (see module docstring) — a byte-different prompt simply misses the
    provider's own cache regardless of what this key says. It exists so the ``openai_prompt_cache_key``
    routing hint (and, for symmetry, trace/telemetry grouping) stays stable and tenant/policy-scoped
    rather than a single global constant.
    """

    tenant: str = Field(
        default="", description="Owning tenant id, best-effort from GraphSession."
    )
    principal: str = Field(
        default="", description="Requesting principal/actor id, best-effort."
    )
    policy_version: str = Field(
        default="",
        description="Authorization/policy revision (mirrors GraphSession.policy_version).",
    )
    model_identity: str = Field(
        default="", description="provider:model_id, e.g. 'anthropic:claude-...'."
    )
    prompt_fingerprint: str = Field(
        default="", description="Content hash of the stable system/instructions prefix."
    )
    safety_posture: str = Field(
        default="",
        description="Active tool-guard mode (TOOL_GUARD_MODE), e.g. 'strict'.",
    )

    model_config = ConfigDict(frozen=True, extra="forbid")

    @property
    def fingerprint(self) -> str:
        """Deterministic sha256 over every component — any change mints a different fingerprint."""
        canonical = _KEY_SEPARATOR.join(
            (
                self.tenant,
                self.principal,
                self.policy_version,
                self.model_identity,
                self.prompt_fingerprint,
                self.safety_posture,
            )
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _prompt_fingerprint(system_prompt: str | None) -> str:
    if not system_prompt:
        return ""
    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:16]


def _safety_posture() -> str:
    try:
        from agent_utilities.core.config import TOOL_GUARD_MODE

        return str(TOOL_GUARD_MODE or "")
    except Exception:  # noqa: BLE001 — best-effort context, never break the caller
        return ""


def _ambient_tenant_principal_policy() -> tuple[str, str, str]:
    """Best-effort ``(tenant, principal, policy_version)`` from the ambient GraphSession.

    Returns empty strings (never raises) when no verified session is bound — most direct model
    calls (unit tests, bare utility calls) have none, and an empty-scoped key is still safe (see
    module docstring: this is a routing hint, not an authorization boundary).
    """
    try:
        from agent_utilities.knowledge_graph.core.session import GraphSession

        session = GraphSession.from_ambient()
    except Exception:  # noqa: BLE001 — no ambient session is the common case
        return "", "", ""
    principal = str(getattr(session.actor, "actor_id", "") or "")
    return str(session.tenant or ""), principal, str(session.policy_version or "")


def resolve_prompt_cache_key(
    *,
    system_prompt: str | None = None,
    model_identity: str | None = None,
    tenant: str | None = None,
    principal: str | None = None,
    policy_version: str | int | None = None,
) -> PromptCacheKey:
    """Build a :class:`PromptCacheKey`, filling unset scoping fields from the ambient
    :class:`~agent_utilities.knowledge_graph.core.session.GraphSession` when one is bound."""
    amb_tenant, amb_principal, amb_policy = _ambient_tenant_principal_policy()
    return PromptCacheKey(
        tenant=tenant if tenant is not None else amb_tenant,
        principal=principal if principal is not None else amb_principal,
        policy_version=(
            str(policy_version) if policy_version is not None else amb_policy
        ),
        model_identity=model_identity or "",
        prompt_fingerprint=_prompt_fingerprint(system_prompt),
        safety_posture=_safety_posture(),
    )


def _record_prompt_cache_op(outcome: str) -> None:
    try:
        from agent_utilities.observability.gateway_metrics import PROMPT_CACHE_OPS

        PROMPT_CACHE_OPS.labels(op="fold", outcome=outcome).inc()
    except Exception as exc:  # noqa: BLE001 — metrics must never break the caller
        logger.debug("prompt-cache telemetry recording failed: %s", exc)


def fold_prompt_cache_hint(
    settings: Any,
    *,
    system_prompt: str | None = None,
    model_identity: str | None = None,
    tenant: str | None = None,
    principal: str | None = None,
    policy_version: str | int | None = None,
) -> Any:
    """Return ``settings`` with provider-native prompt-cache directives folded in.

    CONCEPT:AU-ORCH.optimization.provider-prompt-cache — the live seam, called from
    ``agent_utilities.agent.sampling_profile.attach_profile_resolver`` on EVERY chat call exactly
    like the sibling ``kvcache.policy.fold_kv_hint``. Sets Anthropic's ``anthropic_cache``/
    ``anthropic_cache_instructions``/``anthropic_cache_tool_definitions`` (True, TTL='5m') and
    OpenAI's ``openai_prompt_cache_key`` (a tenant/policy-scoped routing hint) UNLESS the caller
    already set that specific key — never clobbers explicit per-call intent. Each provider's Model
    implementation reads only its own ``anthropic_*``/``openai_*``-prefixed keys and ignores the
    rest, so setting both unconditionally is safe regardless of which provider is actually being
    called (mirrors how ``AnthropicModelSettings``/``OpenAIChatModelSettings`` coexist in one
    ``ModelSettings`` dict by design).

    Disabled wholesale by ``AU_PROMPT_CACHE=false`` (the one opt-out — see module docstring);
    settings pass through completely untouched in that case. Never raises.
    """
    from pydantic_ai import ModelSettings as _MS

    merged: dict[str, Any] = dict(settings) if settings else {}
    if not setting("AU_PROMPT_CACHE", True):
        _record_prompt_cache_op("disabled")
        return _MS(**merged)  # type: ignore[typeddict-item]

    key = resolve_prompt_cache_key(
        system_prompt=system_prompt,
        model_identity=model_identity,
        tenant=tenant,
        principal=principal,
        policy_version=policy_version,
    )

    merged.setdefault("anthropic_cache", True)
    merged.setdefault("anthropic_cache_instructions", True)
    merged.setdefault("anthropic_cache_tool_definitions", True)
    merged.setdefault(
        "openai_prompt_cache_key", key.fingerprint[:_OPENAI_CACHE_KEY_LEN]
    )

    _record_prompt_cache_op("folded")
    return _MS(**merged)  # type: ignore[typeddict-item]


def prompt_cache_create_kwargs(
    create_kwargs: dict[str, Any],
    *,
    system_prompt: str | None = None,
    model_identity: str | None = None,
    tenant: str | None = None,
    principal: str | None = None,
    policy_version: str | int | None = None,
) -> dict[str, Any]:
    """The raw-``openai`` client equivalent of :func:`fold_prompt_cache_hint`.

    For the internal OpenAI-compatible chat-completion seam
    (``agent_utilities.knowledge_graph.retrieval.context_compiler_serving``), which calls
    ``client.chat.completions.create(**create_kwargs)`` directly rather than through pydantic-ai
    ``ModelSettings``. Returns a NEW dict with ``prompt_cache_key`` set (unless the caller already
    supplied one) — never mutates ``create_kwargs`` in place. Same ``AU_PROMPT_CACHE`` opt-out.
    """
    result = dict(create_kwargs)
    if not setting("AU_PROMPT_CACHE", True):
        return result
    if "prompt_cache_key" in result:
        return result
    key = resolve_prompt_cache_key(
        system_prompt=system_prompt,
        model_identity=model_identity,
        tenant=tenant,
        principal=principal,
        policy_version=policy_version,
    )
    result["prompt_cache_key"] = key.fingerprint[:_OPENAI_CACHE_KEY_LEN]
    return result


def usage_cache_tokens(usage: Any) -> tuple[int, int]:
    """``(cache_read_tokens, cache_write_tokens)`` off a pydantic-ai usage object.

    Defensive ``getattr`` — never raises, returns ``(0, 0)`` for any shape that doesn't carry these
    (older pydantic-ai, a non-usage object, ``None``).
    """
    read = int(getattr(usage, "cache_read_tokens", 0) or 0)
    write = int(getattr(usage, "cache_write_tokens", 0) or 0)
    return read, write


def record_prompt_cache_usage(*, provider: str | None, usage: Any) -> tuple[int, int]:
    """Record one call's provider prompt-cache usage into Prometheus telemetry.

    CONCEPT:AU-ORCH.optimization.provider-prompt-cache — "report cache-hit tokens in usage/cost
    telemetry so savings are measurable". Called from
    ``agent_utilities.harness.tracing._TracingModel.request`` on every traced LLM call. Returns the
    extracted ``(cache_read_tokens, cache_write_tokens)`` so the caller can also thread them onto a
    ``GenerationNode``. Best-effort — never raises.
    """
    read, write = usage_cache_tokens(usage)
    try:
        from agent_utilities.observability.gateway_metrics import (
            PROMPT_CACHE_OPS,
            PROMPT_CACHE_TOKENS_SAVED,
        )

        PROMPT_CACHE_OPS.labels(op="response", outcome="hit" if read else "miss").inc()
        if read:
            PROMPT_CACHE_TOKENS_SAVED.labels(provider=provider or "unknown").inc(read)
    except Exception as exc:  # noqa: BLE001 — metrics must never break the caller
        logger.debug("prompt-cache usage telemetry recording failed: %s", exc)
    return read, write
