"""CONCEPT:ORCH-1.92 — dispatch-tier warm-fork: share the warm agent-construction artifacts.

When a swarm fans out (``graph/parallel_engine.py``, 50–300 sub-agents) or a dispatch worker
handles many same-config turns, every ``create_agent`` rebuilds the **SkillsToolset** from
scratch — scanning the skill directories and parsing every ``SKILL.md``. That directory scan is
the dominant *deterministic, connection-free* slice of agent construction, identical for all
agents sharing a skill set, so it is exactly the "warm parent" to amortise across the fan-out
cohort (the agent-tier analogue of the snippet rungs' warm-fork).

Why a shared artifact and not an ``os.fork`` of the agent: forking the orchestrator process —
which holds a live asyncio loop and open MCP/stdio file descriptors — is unsafe, and the
in-process worker already amortises *imports*. The honest, safe win is to build the reusable,
read-only construction artifacts **once** and let each freshly-constructed ``Agent`` reuse them
(pydantic-ai toolsets are designed to be attached to many agents) while still opening its own
per-run MCP connections. The artifacts are pooled in the host :class:`WarmParentRegistry`
(CONCEPT:OS-5.58), so they are idle-reaped and counted alongside the sandbox warm parents.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _skills_key(skill_dirs: list[str]) -> str:
    # Order-independent content key: the same set of skill dirs => the same warm toolset.
    payload = "\x00".join(sorted(skill_dirs))
    return "skills_toolset:" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def get_or_build_skills_toolset(skill_dirs: list[str], factory: Any) -> Any:
    """Return a warm, shared ``SkillsToolset`` for ``skill_dirs`` (built once, then reused).

    ``factory`` is a zero-arg callable that builds a fresh toolset (so this module never imports
    ``pydantic_ai_skills`` itself). Misses build + register into the warm-parent registry; a
    registry/​import failure degrades to a fresh build — never blocks agent creation.
    """
    if not skill_dirs:
        return factory()
    try:
        from agent_utilities.runtime.warm_registry import WarmParentRegistry

        registry = WarmParentRegistry.get()
        key = _skills_key(skill_dirs)
        warm = registry.acquire(key)
        if warm is not None:
            logger.debug("reusing warm SkillsToolset (%d dirs)", len(skill_dirs))
            return warm
        toolset = factory()
        # Read-only construction artifact — nothing to tear down on reap.
        registry.register(key, toolset, close=lambda: None, kind="skills_toolset")
        return toolset
    except Exception as exc:  # noqa: BLE001 — warm cache is an optimisation, never a gate
        logger.debug("warm SkillsToolset unavailable, building fresh: %s", exc)
        return factory()
