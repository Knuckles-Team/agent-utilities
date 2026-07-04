"""FastAPI routers for agent utilities server.

CONCEPT:AU-ECO.interop.standardized-api-surface — Standardized API Surface

This package provides:
- Commands router — Slash command execution and autocomplete for client UIs
"""

from .benchmark import router as benchmark_router
from .commands import router as commands_router

__all__ = [
    "benchmark_router",
    "commands_router",
]
