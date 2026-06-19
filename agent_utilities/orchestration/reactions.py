#!/usr/bin/python
from __future__ import annotations

"""Agent reactions / emotes — a first-class orchestrator output (CONCEPT:ECO-4.79/4.80).

Reactions used to live ONLY in the messaging layer: the instinctive-reaction heuristic,
the available-emoji choices, and the Telegram ``setMessageReaction`` call all sat inside
``messaging/`` (CONCEPT:ECO-4.60). That made "react with 👍" a messaging-only feature —
no other entrypoint (webui / terminal-ui / geniusbot / ``agent_server.py``) could inherit it.

This module promotes reactions to a **core capability of the universal orchestrator** so any
agent turn can emit one and every entrypoint renders it (the *Universal capability — ONE
core, thin entrypoints* rule). Three pieces, all torch-free and dependency-light:

* :class:`AgentReaction` (CONCEPT:ECO-4.79) — first-class reaction/emote output any agent
  turn can emit alongside or instead of text, inherited by every entrypoint.
  Shape: ``{emote, target_message_id?, intensity?}``. Optional and lightweight — a turn that
  doesn't react simply produces ``None``.
* :class:`EmoteRegistry` (CONCEPT:ECO-4.80) — one emote registry and governance gate shared
  by all renderers, governed through the existing :class:`ActionPolicy`
  (``reaction`` kind). No per-surface emote list.
* :func:`decide_reaction` (CONCEPT:ECO-4.79) — the instinctive-reaction heuristic, moved
  here from ``messaging/router.py``. A cheap, model-agnostic, tool-free completion that
  works on ANY model (including local serves that can't call tools), so the decision is made
  once in core and the renderers just paint it.

Renderers (the ONLY per-surface code) consume :class:`AgentReaction`:

* Telegram/messaging → ``send_reaction`` / ``setMessageReaction`` (CONCEPT:ECO-4.81).
* ``agent-webui`` → emoji reaction chips · ``agent-terminal-ui`` → inline emote glyph ·
  ``geniusbot`` → desktop reaction affordance · ``agent_server.py`` → ``reaction`` envelope
  field. Their renderer contract is documented in
  ``docs/architecture/reactions.md`` (those are separate repos; this module defines the
  contract, it does not render for them).
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agent_utilities.core.config import setting

if TYPE_CHECKING:  # pragma: no cover - typing only
    from agent_utilities.security.request_identity import ActorContext

logger = logging.getLogger(__name__)


# ── The core output type (CONCEPT:ECO-4.79) ──────────────────────────────────
@dataclass(slots=True)
class AgentReaction:
    """A structured reaction an agent turn emits — a first-class orchestrator output.

    CONCEPT:ECO-4.79 — emitted alongside (or instead of) the text of a turn. Every
    entrypoint renders it natively (chat reaction, webui chip, TUI glyph, API field).

    Attributes:
        emote: The emoji / emote to react with (e.g. ``"👍"``). The single required field.
        target_message_id: The id of the inbound message being reacted to, when the medium
            attaches reactions to a specific message (chat). ``None`` for media that show a
            standalone reaction (a terminal glyph next to the turn).
        intensity: Optional strength in ``[0.0, 1.0]`` a renderer may map to a visual weight
            (a larger chip, a repeated glyph). ``None`` means "renderer default".
    """

    emote: str
    target_message_id: str | None = None
    intensity: float | None = None

    def __post_init__(self) -> None:
        self.emote = (self.emote or "").strip()
        if self.intensity is not None:
            # Clamp rather than reject — a reaction is cosmetic, never a hard error.
            self.intensity = max(0.0, min(1.0, float(self.intensity)))

    def is_valid(self) -> bool:
        """True when there is an emote to render."""
        return bool(self.emote)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for the API/A2A response envelope and webui/terminal renderers."""
        out: dict[str, Any] = {"emote": self.emote}
        if self.target_message_id is not None:
            out["target_message_id"] = self.target_message_id
        if self.intensity is not None:
            out["intensity"] = self.intensity
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentReaction:
        """Rebuild from a serialized envelope (renderer / cross-process)."""
        return cls(
            emote=str(data.get("emote", "")),
            target_message_id=(
                str(data["target_message_id"])
                if data.get("target_message_id") is not None
                else None
            ),
            intensity=(
                float(data["intensity"]) if data.get("intensity") is not None else None
            ),
        )


# ── The one emote registry + governance (CONCEPT:ECO-4.80) ───────────────────
# The single source of available emotes, shared by every renderer. NO per-surface list.
# Keep this small and universally renderable (chat reactions, a TUI glyph, an API field) —
# platform reaction sets vary, so a backend that cannot render a given emote degrades to its
# nearest supported one or drops it (renderer concern), but the *menu* lives here, once.
_DEFAULT_EMOTES: tuple[str, ...] = (
    "👍",  # acknowledge a request / command
    "❤️",  # thanks / praise
    "🎉",  # good news / a win
    "👀",  # starting to look into something
    "🔥",  # strong agreement / impressive
    "😂",  # something funny
    "🙏",  # gratitude / please
    "✅",  # done / confirmed
    "🤔",  # thinking / uncertain
    "😢",  # bad news / sympathy
)


@dataclass
class EmoteRegistry:
    """The ONE registry of available emotes + governance over their use (CONCEPT:ECO-4.80).

    A single source every renderer reads from — there is no per-surface emote list. Whether a
    given principal/context may emit a reaction is decided through the existing
    :class:`~agent_utilities.orchestration.action_policy.ActionPolicy` gate (the ``reaction``
    action kind), so reactions inherit the same autonomy tiers / rate limits as every other
    fleet action instead of inventing a parallel permission model.
    """

    emotes: list[str] = field(default_factory=lambda: list(_DEFAULT_EMOTES))
    engine: Any = None

    _instance: EmoteRegistry | None = None

    @classmethod
    def instance(cls, engine: Any = None) -> EmoteRegistry:
        """Process-wide singleton (the one shared menu)."""
        if cls._instance is None:
            cls._instance = cls(engine=engine)
        elif engine is not None and cls._instance.engine is None:
            cls._instance.engine = engine
        return cls._instance

    def available(self) -> list[str]:
        """The full menu of emotes a renderer can paint."""
        return list(self.emotes)

    def is_known(self, emote: str) -> bool:
        """True when ``emote`` is in the menu (governance is a separate check)."""
        return (emote or "").strip() in self.emotes

    def allows(
        self,
        emote: str,
        *,
        actor: ActorContext | None = None,
        context: str | None = None,
    ) -> bool:
        """Whether ``actor`` may emit ``emote`` in ``context`` — the governance gate.

        CONCEPT:ECO-4.80 — reuses the ActionPolicy decision point (kind ``reaction``) rather
        than a bespoke per-surface allowlist. An unknown emote is never allowed. When the
        policy gate is unavailable (zero-infra default), reactions are allowed (cosmetic,
        fail-open) — they never block or mutate anything a human cares about.
        """
        emote = (emote or "").strip()
        if not emote or emote not in self.emotes:
            return False
        try:
            from agent_utilities.orchestration.action_policy import (
                ActionRequest,
                get_action_policy,
            )

            decision = get_action_policy(self.engine).decide(
                ActionRequest(
                    kind="reaction",
                    target=context or "chat",
                    source=(getattr(actor, "subject", None) or "orchestrator"),
                    params={"emote": emote},
                )
            )
            # Reactions are cosmetic: only a hard FORBIDDEN denies them. auto / auto_notify /
            # approval_required all permit the (low-risk) emote — we do not file an approval
            # for an emoji. A missing rule defaults to allow (the gate's own default tier).
            return str(getattr(decision, "decision", "allow")).lower() != "deny"
        except Exception as exc:  # noqa: BLE001 — governance optional; fail-open for cosmetics
            logger.debug(
                "[ECO-4.80] reaction governance unavailable (%s); allowing", exc
            )
            return True


# ── The instinctive-reaction heuristic, moved into core (CONCEPT:ECO-4.79) ───
# Model-agnostic: a cheap tool-free completion so reactions work on ANY model (including local
# serves that cannot do tool calls). This is the heuristic that used to live in
# ``messaging/router.py::_decide_reaction`` — it now lives in core so EVERY entrypoint, not
# just messaging, produces reactions from the same one decision.
_REACTION_SYSTEM = (
    "You decide whether to react to a user's message with a single emoji, the way a "
    "thoughtful assistant would. Reply with ONE emoji if a reaction fits (e.g. 👍 to "
    "acknowledge a request/command, ❤️ for thanks or praise, 🎉 for good news, 👀 when "
    "starting to look into something), or the exact word NONE if no reaction fits. Output "
    "only the emoji or NONE — nothing else."
)


def _reactions_enabled() -> bool:
    """Opt-out switch (``REACTIONS=0``). Default ON — reactions are a native turn output."""
    # ``MESSAGING_REACTIONS`` kept as a recognized alias so the prior messaging opt-out still
    # disables them after the capability moved to core (No-Legacy: one behavior, two readers
    # only because the var name was the operator-facing contract).
    for key in ("REACTIONS", "MESSAGING_REACTIONS"):
        if str(setting(key, "1")).strip().lower() in ("0", "false", "no", "off"):
            return False
    return True


async def decide_reaction(
    content: str,
    *,
    registry: EmoteRegistry | None = None,
    actor: ActorContext | None = None,
    context: str | None = None,
    target_message_id: str | None = None,
) -> AgentReaction | None:
    """Instinctively decide whether a turn should react, returning an :class:`AgentReaction`.

    CONCEPT:ECO-4.79 — first-class reaction output any agent turn can emit, shared by every
    entrypoint. Returns ``None`` when no reaction fits, when disabled, or when governance
    forbids the chosen emote. The
    LLM call is bounded (10s) so a slow/hung serve can never stall a caller that awaits it
    (callers should still run this off their critical path — it is cosmetic).
    """
    if not _reactions_enabled() or not content:
        return None
    reg = registry or EmoteRegistry.instance()
    try:
        from pydantic_ai import Agent

        from agent_utilities.core.model_factory import create_model

        agent = Agent(create_model(), system_prompt=_REACTION_SYSTEM)
        result = await asyncio.wait_for(agent.run(content), timeout=10.0)
        out = str(getattr(result, "output", result)).strip()
    except Exception as exc:  # noqa: BLE001 — cosmetic; never propagate
        logger.debug("[ECO-4.79] reaction decision skipped: %s", exc)
        return None

    if not out or out.upper().startswith("NONE") or len(out) > 8:
        return None
    emote = out
    if not reg.is_known(emote):
        # The model is free to pick any emoji; if it is outside our menu, only emit it when
        # it is genuinely a single emoji (keeps the output a real reaction, not a sentence),
        # but governance still applies to in-menu choices below.
        if len(emote) > 4:
            return None
        return AgentReaction(emote=emote, target_message_id=target_message_id)
    if not reg.allows(emote, actor=actor, context=context):
        logger.debug(
            "[ECO-4.80] reaction %s denied by governance for %s", emote, context
        )
        return None
    return AgentReaction(emote=emote, target_message_id=target_message_id)
