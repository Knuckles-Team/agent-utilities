"""Dynamic, context-aware KV-cache-layering policy — per-execution cache-worthiness.

CONCEPT:AU-ORCH.optimization.kvcache-worthiness-policy — Dynamic context-aware per-execution KV-cache-layering cache-worthiness policy

A per-call decision engine that decides, for EACH LLM chat execution, whether its
KV-cache is worth **storing** into the decoupled LMCache layer (L1 CPU + L2 epistemic-graph, see ``kvcache/remote_backend.py`` /
``CONCEPT:AU-KG.backend.kvcache-vllm-connector``) rather than a hard, global on/off. The intelligence: reserve
the KV store cost (and cache pollution) for executions that are actually reused,
and skip it for one-off short single-shot prompts.

Why this is a *store*-side decision
-----------------------------------
LMCache **retrieval** is already opportunistic and free to gate: on a token-prefix
hash hit it reuses, otherwise it just misses. The only lever worth pulling per
request is the **store** side — writing a fresh prompt's KV blocks into the tiered
cache costs bandwidth/space and, for a one-off prompt that never recurs, is pure
pollution. vLLM + LMCache expose exactly this lever per request:

* vLLM ``ChatCompletionRequest.kv_transfer_params: dict | None`` (validated in the
  ``registry.arpa/vllm-lmcache:latest`` image) rides in the OpenAI ``extra_body``
  and is threaded into ``sampling_params.extra_args["kv_transfer_params"]``.
* LMCache's ``vllm_v1_adapter.extract_request_configs`` copies every ``lmcache.*``
  key out of it into the per-request ``request_configs``, and the save path honours
  ``request_configs["lmcache.skip_save"]`` — a hit means "do not write this
  request's KV blocks to the cache".

So a **non-cache-worthy** execution sets ``kv_transfer_params={"lmcache.skip_save":
true}`` and a cache-worthy one sets it ``false`` (explicit + observable; ``false``
is also LMCache's default, so retrieval/prefix-reuse is unaffected either way).

Signals (cache-worthy => store)
-------------------------------
* **Long shared/system prefix** — a big system prompt / agent preamble reused
  across calls (``>= KV_CACHE_MIN_PREFIX_TOKENS``).
* **Multi-turn conversation** — a growing message history reused each turn
  (``>= KV_CACHE_MIN_HISTORY_TURNS`` prior turns).
* **Large fixed context** — a big total prompt, e.g. a fixed RAG block
  (``>= KV_CACHE_MIN_CONTEXT_TOKENS``).

A one-off short single-shot prompt hits none of these => ``skip_save = true``.

Graceful by construction: if the served model is plain vLLM without an LMCache
connector, the ``kv_transfer_params`` dict is an ignored no-op — nothing reads
``extra_args["kv_transfer_params"]`` — so the hint is always safe to attach.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from agent_utilities.core.config import setting

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pydantic_ai import ModelSettings

# Roles whose executions are inherently reuse-heavy (long stable preamble +
# multi-turn), used as a weak prior when message-level signals are thin. Kept
# small and additive — absence never forces skip; presence only nudges toward
# store when other signals are borderline.
_REUSE_HEAVY_ROLES: frozenset[str] = frozenset(
    {"chat", "orchestrator", "planner", "router", "assistant", "conversation"}
)


class KVCacheDecision(BaseModel):
    """The per-execution verdict from :class:`KVCacheLayeringPolicy`.

    CONCEPT:AU-ORCH.optimization.kvcache-worthiness-policy. ``skip_save`` is the enforced lever (``lmcache.skip_save``);
    everything else is provenance so the decision is explainable and testable.
    """

    cache_worthy: bool = Field(
        description="True => store this execution's KV (skip_save False)."
    )
    skip_save: bool = Field(
        description="The value threaded to LMCache as 'lmcache.skip_save'. "
        "True => do NOT write this request's KV blocks to the cache."
    )
    score: float = Field(
        default=0.0,
        description="Aggregate cache-worthiness score (0.0-1.0), for ranking/telemetry.",
    )
    reasons: list[str] = Field(
        default_factory=list,
        description="Human-readable signals that drove the verdict.",
    )
    signals: dict[str, Any] = Field(
        default_factory=dict,
        description="Raw computed signals (token estimates, turn counts).",
    )
    enabled: bool = Field(
        default=True,
        description="False => the master switch KV_CACHE_LAYERING is off; the "
        "decision is inert and no hint should be attached.",
    )

    model_config = ConfigDict(extra="forbid")

    @property
    def kv_transfer_params(self) -> dict[str, Any]:
        """The vLLM/LMCache per-request control dict for this decision.

        Empty when the policy is disabled (attach nothing => opportunistic default).
        Otherwise ``{"lmcache.skip_save": <bool>}`` — explicit on both branches so
        the store decision is observable server-side and in tests.
        """
        if not self.enabled:
            return {}
        return {"lmcache.skip_save": self.skip_save}


def _estimate_tokens(text: str | None, chars_per_token: float) -> int:
    """Cheap dependency-free token estimate (chars / chars_per_token).

    Deliberately avoids importing a tokenizer on the inference hot path; the policy
    only needs order-of-magnitude thresholds, not exact counts.
    """
    if not text:
        return 0
    return int(len(text) / max(chars_per_token, 1.0))


def _history_text_and_turns(message_history: Any) -> tuple[str, int]:
    """Best-effort (concatenated-text, turn-count) from a pydantic-ai history.

    ``message_history`` is a list of ``ModelMessage`` (ModelRequest/ModelResponse)
    whose ``.parts`` carry ``.content``. We never hard-depend on those types: any
    shape is walked defensively and failures degrade to ``("", 0)`` so history
    parsing can never break a run.
    """
    if not message_history:
        return "", 0
    try:
        turns = len(message_history)
    except TypeError:
        return "", 0
    chunks: list[str] = []
    try:
        for msg in message_history:
            parts = getattr(msg, "parts", None)
            if not parts:
                continue
            for part in parts:
                content = getattr(part, "content", None)
                if isinstance(content, str):
                    chunks.append(content)
    except Exception:  # noqa: BLE001 - history text is a soft signal
        pass
    return " ".join(chunks), turns


class KVCacheLayeringPolicy:
    """Scores an LLM execution's KV-cache-worthiness (CONCEPT:AU-ORCH.optimization.kvcache-worthiness-policy).

    Thresholds are read live from :func:`agent_utilities.core.config.setting` at
    construction (so ``config.json`` / monkeypatched env apply), and may be
    overridden per-instance for testing. Construct-then-``decide`` is cheap; a fresh
    instance per call is fine.
    """

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        min_prefix_tokens: int | None = None,
        min_context_tokens: int | None = None,
        min_history_turns: int | None = None,
        chars_per_token: float | None = None,
    ) -> None:
        self.enabled = (
            setting("KV_CACHE_LAYERING", True) if enabled is None else enabled
        )
        self.min_prefix_tokens = (
            setting("KV_CACHE_MIN_PREFIX_TOKENS", 1024)
            if min_prefix_tokens is None
            else min_prefix_tokens
        )
        self.min_context_tokens = (
            setting("KV_CACHE_MIN_CONTEXT_TOKENS", 2048)
            if min_context_tokens is None
            else min_context_tokens
        )
        self.min_history_turns = (
            setting("KV_CACHE_MIN_HISTORY_TURNS", 1)
            if min_history_turns is None
            else min_history_turns
        )
        self.chars_per_token = (
            setting("KV_CACHE_CHARS_PER_TOKEN", 4.0)
            if chars_per_token is None
            else chars_per_token
        )

    def decide(
        self,
        *,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        message_history: Any = None,
        rag_context: str | None = None,
        role: str | None = None,
    ) -> KVCacheDecision:
        """Compute the cache-worthiness verdict for one execution.

        Cache-worthy (store) when ANY strong reuse signal fires: a long shared
        prefix, a multi-turn history, or a large total context. Otherwise the
        execution is treated as a one-off and ``skip_save`` is set to spare the
        cache the store cost and pollution.
        """
        if not self.enabled:
            return KVCacheDecision(
                cache_worthy=False,
                skip_save=False,
                enabled=False,
                reasons=["policy_disabled"],
            )

        prefix_tokens = _estimate_tokens(system_prompt, self.chars_per_token)
        rag_tokens = _estimate_tokens(rag_context, self.chars_per_token)
        user_tokens = _estimate_tokens(user_prompt, self.chars_per_token)
        history_text, turns = _history_text_and_turns(message_history)
        history_tokens = _estimate_tokens(history_text, self.chars_per_token)
        total_tokens = prefix_tokens + rag_tokens + user_tokens + history_tokens

        reasons: list[str] = []
        worthy = False

        long_prefix = prefix_tokens >= self.min_prefix_tokens
        if long_prefix:
            worthy = True
            reasons.append(
                f"long_shared_prefix({prefix_tokens}>={self.min_prefix_tokens}t)"
            )

        multi_turn = turns >= self.min_history_turns and turns > 0
        if multi_turn:
            worthy = True
            reasons.append(f"multi_turn_conversation({turns}turns)")

        large_context = total_tokens >= self.min_context_tokens
        if large_context:
            worthy = True
            reasons.append(
                f"large_fixed_context({total_tokens}>={self.min_context_tokens}t)"
            )

        large_rag = rag_tokens >= self.min_prefix_tokens
        if large_rag:
            worthy = True
            reasons.append(f"large_rag_context({rag_tokens}t)")

        # Weak role prior: only tips a borderline (no strong signal but non-trivial
        # context) execution toward store; never overrides a clear one-off.
        role_reuse = role is not None and role.lower() in _REUSE_HEAVY_ROLES
        if not worthy and role_reuse and total_tokens >= self.min_context_tokens // 2:
            worthy = True
            reasons.append(f"reuse_heavy_role({role})")

        if not worthy:
            reasons.append("one_off_short_prompt")

        # Score: fraction of the total against the context threshold, capped, plus a
        # multi-turn bonus. Purely for telemetry/ranking; the verdict is boolean.
        score = min(1.0, total_tokens / max(self.min_context_tokens, 1))
        if multi_turn:
            score = min(1.0, score + 0.25)

        return KVCacheDecision(
            cache_worthy=worthy,
            skip_save=not worthy,
            score=round(score, 4),
            reasons=reasons,
            signals={
                "prefix_tokens": prefix_tokens,
                "rag_tokens": rag_tokens,
                "user_tokens": user_tokens,
                "history_tokens": history_tokens,
                "history_turns": turns,
                "total_tokens": total_tokens,
            },
        )


# A module-level default instance is intentionally NOT cached: thresholds are read
# live so a config reload / test monkeypatch takes effect on the next call.


def decide(
    *,
    system_prompt: str | None = None,
    user_prompt: str | None = None,
    message_history: Any = None,
    rag_context: str | None = None,
    role: str | None = None,
) -> KVCacheDecision:
    """Convenience: score one execution with a fresh policy (live thresholds)."""
    return KVCacheLayeringPolicy().decide(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        message_history=message_history,
        rag_context=rag_context,
        role=role,
    )


def fold_kv_hint(
    settings: Any,
    *,
    system_prompt: str | None = None,
    user_prompt: str | None = None,
    message_history: Any = None,
    rag_context: str | None = None,
    role: str | None = None,
) -> ModelSettings:
    """Return ``settings`` with this execution's KV-cache hint folded into extra_body.

    CONCEPT:AU-ORCH.optimization.kvcache-worthiness-policy — the live seam. Takes the per-call base ``ModelSettings``
    (dict-shaped), runs :class:`KVCacheLayeringPolicy`, and dict-merges
    ``{"kv_transfer_params": {"lmcache.skip_save": <bool>}}`` into ``extra_body``
    WITHOUT disturbing any pre-existing knob (reasoning_effort, priority, top_k...).
    When the master switch is off the settings pass through untouched. Best-effort:
    the caller wraps this so any failure leaves the run's settings as-is.
    """
    decision = decide(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        message_history=message_history,
        rag_context=rag_context,
        role=role,
    )

    from pydantic_ai import ModelSettings as _MS

    merged: dict[str, Any] = dict(settings) if settings else {}
    if not decision.enabled:
        return _MS(**merged)  # type: ignore[typeddict-item]

    extra: dict[str, Any] = dict(merged.get("extra_body") or {})
    kv: dict[str, Any] = dict(extra.get("kv_transfer_params") or {})
    kv.update(decision.kv_transfer_params)
    extra["kv_transfer_params"] = kv
    merged["extra_body"] = extra
    return _MS(**merged)  # type: ignore[typeddict-item]
