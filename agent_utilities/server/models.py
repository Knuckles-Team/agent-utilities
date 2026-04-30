import logging
from collections.abc import Callable
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class CodemapRequest(BaseModel):
    """Request schema for codebase codemap generation."""

    prompt: str
    mode: Literal["fast", "smart"] = "smart"


class ReloadableApp:
    """ASGI application wrapper that supports manual hot-reloading.

    This wrapper allows swapping the underlying FastAPI application instance
    at runtime without restarting the physical server process.
    """

    def __init__(self, factory: Callable[[], FastAPI]):
        """Initialize the reloadable application.

        Args:
            factory: A function that returns a fresh FastAPI instance.
        """
        self.factory = factory
        self.app: FastAPI = self.factory()

    async def __call__(self, scope, receive, sender):
        """Standard ASGI entry point."""
        await self.app(scope, receive, sender)

    def reload(self):
        """Execute the factory to replace the current application state."""
        logger.info("Hot-reloading agent application...")
        self.app = self.factory()
