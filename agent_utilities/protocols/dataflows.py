"""Market Dataflows (ECO-4.8).

CONCEPT: ECO-4.8 Market Dataflows

Connects continuous ticker data into the graph orchestrator as a continuous
message stream rather than polling.
"""

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any


class TickerStreamConnector:
    """Connects to external market data streams and dispatches ticks."""

    def __init__(self, source_url: str):
        self.source_url = source_url
        self._subscribers: list[
            Callable[[str, float, float], Coroutine[Any, Any, None]]
        ] = []

    def subscribe(
        self, callback: Callable[[str, float, float], Coroutine[Any, Any, None]]
    ) -> None:
        """Register a callback (e.g., agent.receive_tick) to the stream."""
        self._subscribers.append(callback)

    async def _dispatch(self, symbol: str, price: float, timestamp: float) -> None:
        """Dispatch tick to all subscribers concurrently."""
        tasks = [sub(symbol, price, timestamp) for sub in self._subscribers]
        if tasks:
            await asyncio.gather(*tasks)

    async def start_streaming(self) -> None:
        """Begin streaming data (Mock implementation)."""
        import time

        # In production, this would be an async websocket client
        while True:
            # Mock tick
            await self._dispatch("BTC-USD", 65000.0, time.time())
            await asyncio.sleep(1.0)
