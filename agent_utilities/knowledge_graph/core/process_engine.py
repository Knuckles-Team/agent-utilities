"""The one construction site of the process-owned IntelligenceGraphEngine.

Both entrypoints that open the process engine use it: graph-os MCP
(``kg_server._get_engine``) and the hosted AU runtime
(``api.runtime.open_process_runtime``). Whichever runs first builds the
engine the same way. The winner is registered through
:meth:`IntelligenceGraphEngine.get_or_create`, so a second entrypoint reuses
it rather than racing another authority into existence (D-WD-7).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import IntelligenceGraphEngine

__all__ = ["open_process_engine"]


def open_process_engine(*, defer_background_start: bool) -> IntelligenceGraphEngine:
    """Return the process engine, constructing it on first use."""
    from agent_utilities.core.paths import ensure_dirs

    from ..backends import create_backend
    from .engine import IntelligenceGraphEngine

    def _factory() -> IntelligenceGraphEngine:
        ensure_dirs()
        return IntelligenceGraphEngine(
            backend=create_backend(), defer_background_start=defer_background_start
        )

    return IntelligenceGraphEngine.get_or_create(factory=_factory)
