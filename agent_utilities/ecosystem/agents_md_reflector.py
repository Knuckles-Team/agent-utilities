#!/usr/bin/python
from __future__ import annotations

"""AGENTS.md Reflector — Stop Hook for Self-Improving Configuration.

CONCEPT:ECO-4.17 — Self-Improving AGENTS.md Reflector

Registers as an AFTER_RUN hook that reflects on what happened during a session
and proposes AGENTS.md updates while the context is fresh.

Pipeline:
    Session transcript -> LLM Observer -> Proposed AGENTS.md diff
    -> `.agents/proposed_updates/YYYY-MM-DD.md` -> KG AgentsMdProposal node
"""

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..capabilities.hooks import HookInput, HookResult
    from ..knowledge_graph.core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)

__all__ = [
    "AgentsMdReflector",
    "AgentsMdProposal",
    "create_reflector_hook",
    "apply_proposal",
]

DEFAULT_AUTO_APPLY_CONFIDENCE = 0.0
DEFAULT_PROPOSAL_DIR = ".agents/proposed_updates"
DEFAULT_MAX_PROPOSALS = 5
DEFAULT_MIN_TURNS = 5

REFLECTOR_SYSTEM_PROMPT = """You are the AGENTS.md Reflector for agent-utilities.
Analyze a completed session and propose AGENTS.md updates.

Look for: new conventions, working commands, gotchas, contradicted rules, tool preferences.

Return JSON array:
[{"section":"## Section","action":"add|update|remove","content":"text","reasoning":"why","confidence":0.0-1.0}]

Rules: Only propose genuinely useful changes. Max 5 proposals. Be specific."""


@dataclass(frozen=True)
class AgentsMdProposal:
    """A proposed update to AGENTS.md from session reflection."""

    section: str
    action: str
    content: str
    reasoning: str
    confidence: float
    session_id: str = ""
    timestamp: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )

    @property
    def proposal_id(self) -> str:
        digest = hashlib.md5(
            f"{self.section}:{self.content}".encode(), usedforsecurity=False
        ).hexdigest()[:10]
        return f"amp_{digest}"

    def to_markdown(self) -> str:
        emoji = {"add": "+", "update": "~", "remove": "-"}.get(self.action, "?")
        return (
            f"### [{emoji}] {self.action.upper()} — {self.section}\n\n"
            f"**Confidence**: {self.confidence:.0%} | **Reasoning**: {self.reasoning}\n\n"
            f"```markdown\n{self.content}\n```\n"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "section": self.section,
            "action": self.action,
            "content": self.content,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
        }


class AgentsMdReflector:
    """Stop hook that proposes AGENTS.md updates from session learnings.

    CONCEPT:ECO-4.17 — Self-Improving AGENTS.md Reflector
    """

    def __init__(
        self,
        engine: IntelligenceGraphEngine | None = None,
        workspace_path: str | Path = ".",
        auto_apply_confidence: float = DEFAULT_AUTO_APPLY_CONFIDENCE,
        proposal_dir: str = DEFAULT_PROPOSAL_DIR,
        max_proposals: int = DEFAULT_MAX_PROPOSALS,
        min_turns: int = DEFAULT_MIN_TURNS,
    ) -> None:
        self.engine = engine
        self.workspace_path = Path(workspace_path).resolve()
        self.auto_apply_confidence = auto_apply_confidence
        self.proposal_dir = Path(proposal_dir)
        self.max_proposals = max_proposals
        self.min_turns = min_turns

    async def reflect(
        self,
        messages: list[dict[str, str]],
        *,
        session_id: str = "",
        source: str = "unknown",
    ) -> list[AgentsMdProposal]:
        """Analyze session transcript and propose AGENTS.md updates."""
        if len(messages) < self.min_turns:
            return []

        current_md = self._load_current_agents_md()
        transcript = "\n\n".join(
            f"[{m.get('role', '?').upper()}] {m.get('content', '')[:1500]}"
            for m in messages[-50:]
        )

        raw = await self._call_llm(transcript, current_md)
        proposals = self._parse_proposals(raw, session_id)
        if not proposals:
            return []

        self._write_proposals(proposals)
        self._persist_proposals(proposals, source)

        applied = [
            p
            for p in proposals
            if self.auto_apply_confidence > 0
            and p.confidence >= self.auto_apply_confidence
        ]
        for p in applied:
            try:
                apply_proposal(self.workspace_path, p)
            except Exception as e:
                logger.warning("[ECO-4.2] Auto-apply failed: %s", e)

        logger.info(
            "[ECO-4.2] Generated %d proposals (auto-applied: %d)",
            len(proposals),
            len(applied),
        )
        return proposals

    def as_hook(self):
        """Return a callable hook for HooksCapability registration."""

        async def _hook(input: HookInput) -> HookResult | None:
            from ..capabilities.hooks import HookEvent

            if input.event != HookEvent.AFTER_RUN:
                return None
            messages = _extract_messages(input.result)
            if messages:
                await self.reflect(
                    messages, session_id=str(getattr(input.ctx, "run_id", ""))
                )
            return None

        return _hook

    def _load_current_agents_md(self) -> str:
        for name in ["AGENTS.md", ".agents/AGENTS.md"]:
            p = self.workspace_path / name
            if p.is_file():
                try:
                    return p.read_text(encoding="utf-8")[:8000]
                except Exception:
                    pass
        return ""

    async def _call_llm(self, transcript: str, current_md: str) -> str:
        try:
            from pydantic_ai import Agent

            from ..core.config import DEFAULT_KG_MODEL_ID, DEFAULT_LLM_PROVIDER
            from ..core.model_factory import create_model

            model = create_model(
                provider=DEFAULT_LLM_PROVIDER, model_id=DEFAULT_KG_MODEL_ID
            )
            agent = Agent(model, system_prompt=REFLECTOR_SYSTEM_PROMPT)
            import nest_asyncio

            nest_asyncio.apply()
            result = agent.run_sync(
                f"## Current AGENTS.md\n```\n{current_md[:4000]}\n```\n\n## Session\n{transcript}"
            )
            return str(result.data)
        except Exception as e:
            logger.warning("[ECO-4.2] LLM call failed: %s", e)
            return "[]"

    def _parse_proposals(self, raw: str, session_id: str) -> list[AgentsMdProposal]:
        try:
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            items = json.loads(match.group()) if match else json.loads(raw)
        except (json.JSONDecodeError, TypeError, AttributeError):
            return []
        if not isinstance(items, list):
            return []
        proposals = []
        for item in items[: self.max_proposals]:
            try:
                p = AgentsMdProposal(
                    section=str(item.get("section", "## General")),
                    action=str(item.get("action", "add")).lower(),
                    content=str(item.get("content", "")),
                    reasoning=str(item.get("reasoning", "")),
                    confidence=float(item.get("confidence", 0.5)),
                    session_id=session_id,
                )
                if p.content.strip() and p.action in ("add", "update", "remove"):
                    proposals.append(p)
            except (ValueError, TypeError):
                pass
        return proposals

    def _write_proposals(self, proposals: list[AgentsMdProposal]) -> None:
        d = self.workspace_path / self.proposal_dir
        d.mkdir(parents=True, exist_ok=True)
        fp = d / f"{time.strftime('%Y-%m-%d', time.gmtime())}.md"
        parts = []
        if fp.exists():
            parts.append(fp.read_text(encoding="utf-8").rstrip() + "\n")
        else:
            parts.append(
                f"# AGENTS.md Proposals — {time.strftime('%Y-%m-%d')}\n\n"
                "> Auto-generated by AGENTS.md Reflector (CONCEPT:ECO-4.17)\n"
            )
        parts.append(f"\n## Session at {time.strftime('%H:%M:%S UTC')}\n")
        for p in proposals:
            parts.append(p.to_markdown())
        fp.write_text("\n".join(parts), encoding="utf-8")

    def _persist_proposals(
        self, proposals: list[AgentsMdProposal], source: str
    ) -> None:
        if not self.engine:
            return
        for p in proposals:
            try:
                self.engine.add_node(
                    p.proposal_id,
                    "agents_md_proposal",
                    {
                        "name": f"Proposal: {p.section}",
                        "description": p.reasoning,
                        "content": p.content,
                        "confidence": p.confidence,
                        "source": source,
                        "timestamp": p.timestamp,
                        "importance_score": p.confidence,
                        "applied": False,
                    },
                )
            except Exception as e:
                logger.debug("[ECO-4.2] Persist failed: %s", e)


def apply_proposal(workspace_path: str | Path, proposal: AgentsMdProposal) -> None:
    """Apply a single proposal to AGENTS.md."""
    path = Path(workspace_path) / "AGENTS.md"
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    if proposal.action == "add":
        if proposal.section in current:
            idx = current.index(proposal.section) + len(proposal.section)
            ns = current.find("\n## ", idx + 1)
            insert = f"\n{proposal.content}\n"
            current = (
                current[:ns] + insert + current[ns:] if ns != -1 else current + insert
            )
        else:
            current += f"\n\n{proposal.section}\n{proposal.content}\n"
        path.write_text(current, encoding="utf-8")
    elif proposal.action == "remove" and proposal.content in current:
        path.write_text(current.replace(proposal.content, ""), encoding="utf-8")


def create_reflector_hook(engine=None, workspace_path=".", **kw):
    """Convenience: create a reflector hook callable."""
    return AgentsMdReflector(
        engine=engine, workspace_path=workspace_path, **kw
    ).as_hook()


def _extract_messages(result: Any) -> list[dict[str, str]]:
    msgs: list[dict[str, str]] = []
    try:
        if hasattr(result, "all_messages"):
            for msg in result.all_messages():
                for part in getattr(msg, "parts", []):
                    text = getattr(part, "content", None)
                    if text:
                        msgs.append(
                            {"role": getattr(msg, "role", "?"), "content": text[:2000]}
                        )
    except Exception:
        pass
    return msgs
