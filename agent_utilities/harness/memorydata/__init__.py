"""Graph-os memory stack adapter for the MemoryData benchmark.

CONCEPT:AU-AHE.harness.hardening-transparency-surface/3.72/3.73/3.74 — a MemoryData-contract agent method over the graph-os
memory surfaces, a config-vs-family bake-off, a family-aware retrieval router, and a markdown
scoreboard. The adapter talks to a pluggable :class:`MemoryBackendClient` (mock for offline
tests, REST for a live engine), so the whole sweep is runnable and testable with nothing
deployed.

Integration point: MemoryData drives each method through
``AgentWrapper.send_message(message, memorizing=..., context_id=..., query_id=...,
eval_metadata=...)``; :class:`GraphOSMemoryMethod` implements that contract exactly.
"""

from agent_utilities.harness.memorydata.adapter import (
    RETRIEVAL_CONFIGS,
    GraphOSMemoryMethod,
)
from agent_utilities.harness.memorydata.bakeoff import BakeoffResult, run_bakeoff
from agent_utilities.harness.memorydata.client import (
    BackendUnavailable,
    GraphOSRestClient,
    MemoryBackendClient,
    MockBackendClient,
    build_client,
)
from agent_utilities.harness.memorydata.router_method import GraphOSRouterMethod
from agent_utilities.harness.memorydata.scoreboard import (
    MEMORYDATA_BASELINES,
    render_scoreboard,
)

__all__ = [
    "GraphOSMemoryMethod",
    "GraphOSRouterMethod",
    "RETRIEVAL_CONFIGS",
    "run_bakeoff",
    "render_scoreboard",
    "BakeoffResult",
    "MEMORYDATA_BASELINES",
    "MemoryBackendClient",
    "MockBackendClient",
    "GraphOSRestClient",
    "BackendUnavailable",
    "build_client",
]
