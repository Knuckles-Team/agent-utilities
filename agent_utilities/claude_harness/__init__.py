#!/usr/bin/python
from __future__ import annotations

"""Claude Code CLI harness — safe unattended operation over the graph-os Loop.

This package lets the **Claude Code CLI itself** run unattended and safely while
driving the existing Loop engine (``knowledge_graph/research/loop_controller``).
It is the operator-surface counterpart of the fleet-level :mod:`ActionPolicy`
(``orchestration/action_policy``, OS-5.24): the same governance source of truth
now also fences the IDE harness.

Modules:

- :mod:`claude_fence` (CONCEPT:AU-OS.deployment.governance-derived-claude-code) — derive a hardened Claude Code
  ``settings.json`` permission fence (``allow``/``ask``/``deny`` +
  ``defaultMode``) and a ``.claudeignore`` from the live ``ActionPolicy`` plus
  the known-secret patterns. Self-updating: a new ``forbidden`` rule propagates
  into the static deny on the next run.
- :mod:`pretooluse_gate` (CONCEPT:AU-OS.deployment.dynamic-two-fail-closed) — the runtime PreToolUse hook body.
  Two layers: an always-on static secret/irreversible deny, then the live
  ``ActionPolicy.decide()`` governed verdict. Fail-closed (deny on any error).
- :mod:`overnight_runner` (CONCEPT:AU-AHE.harness.overnight-loop-driver) — the testable core the
  ``unattended-loop-driver`` skill leans on: drive ``LoopController`` cycles,
  commit per productive cycle, and write a morning summary into ``MEMORY.md``.

NOT to be confused with :mod:`agent_utilities.harness` — that is the AHE
agentic-evolution harness (evaluators, SWE-bench, scorers). This package is the
Claude Code *IDE* harness.
"""

from .claude_fence import (
    GATE_HOOK_COMMAND,
    build_claudeignore_text,
    build_settings_dict,
    is_secret_path,
    write_fence,
)

__all__ = [
    "GATE_HOOK_COMMAND",
    "build_claudeignore_text",
    "build_settings_dict",
    "is_secret_path",
    "write_fence",
]
