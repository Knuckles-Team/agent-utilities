#!/usr/bin/python
from __future__ import annotations

"""Graph Steps Module — Backward-Compatible Re-Export Shim.

This module previously contained all step definitions for the pydantic-graph
orchestrator (2,400+ lines). It has been decomposed into focused submodules:

- ``lifecycle.py``: Session lifecycle, policy guards, human-in-the-loop.
- ``planning.py``: Research, architecture, planning, memory selection.
- ``routing.py``: Core dispatch, routing, parallel execution, MCP routing.
- ``verification.py``: Quality gates, synthesis, error recovery, join.
- ``adaptive_agent_router.py``: Data-driven factory for specialist persona steps.

All public symbols are re-exported here so that existing imports
(e.g., ``from .steps import router_step``) continue to work.
New code should import from the specific submodule directly.
"""


import asyncio

from .adaptive_agent_router import *  # noqa: F401,F403
from .hierarchical_planner import *  # noqa: F401,F403

# Re-export all public symbols from submodules
from .lifecycle import *  # noqa: F401,F403
from .routing import *  # noqa: F401,F403
from .verification import *  # noqa: F401,F403

# Preserve the module-level lock that was previously defined here
lock = asyncio.Lock()
