"""FastAPI routers for agent utilities server.

CONCEPT:ECO-4.0 — Standardized API Surface

This package provides:
- Commands router — Slash command execution and autocomplete for client UIs
"""

from .commands import router as commands_router

__all__ = [
    "commands_router",
]
