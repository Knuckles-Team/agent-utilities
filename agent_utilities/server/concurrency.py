"""Session Concurrency Manager for OS-5.3.

CONCEPT:AU-OS.safety.doom-loop-detection

Provides double-texting concurrency strategies (enqueue, reject, interrupt, rollback)
for Pydantic AI graph sessions. Supports both Asyncio (local) and Redis (distributed).
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

from fastapi import HTTPException

logger = logging.getLogger(__name__)


class BaseConcurrencyManager(ABC):
    @abstractmethod
    async def acquire(self, session_id: str, strategy: str = "enqueue") -> Any:
        pass

    @abstractmethod
    async def release(self, session_id: str) -> None:
        pass


class AsyncioConcurrencyManager(BaseConcurrencyManager):
    def __init__(self):
        self._locks: dict[str, asyncio.Lock] = {}
        self._active_tasks: dict[str, asyncio.Task] = {}

    def _get_lock(self, session_id: str) -> asyncio.Lock:
        if session_id not in self._locks:
            self._locks[session_id] = asyncio.Lock()
        return self._locks[session_id]

    async def acquire(self, session_id: str, strategy: str = "enqueue") -> None:
        lock = self._get_lock(session_id)
        current_task = asyncio.current_task()

        if lock.locked():
            if strategy == "reject":
                raise HTTPException(
                    status_code=409, detail="A run is already active for this session."
                )
            elif strategy in ("interrupt", "rollback"):
                active = self._active_tasks.get(session_id)
                if active and not active.done():
                    logger.info(
                        f"[{strategy.upper()}] Canceling active task for session {session_id}"
                    )
                    active.cancel()
                    # Wait briefly for cancellation to process
                    try:
                        await asyncio.wait_for(active, timeout=2.0)
                    except (TimeoutError, asyncio.CancelledError, Exception):
                        pass

        await lock.acquire()
        if current_task:
            self._active_tasks[session_id] = current_task

    async def release(self, session_id: str) -> None:
        lock = self._get_lock(session_id)
        if lock.locked():
            lock.release()
        self._active_tasks.pop(session_id, None)


class RedisConcurrencyManager(BaseConcurrencyManager):
    def __init__(self, redis_client: Any):
        self.redis = redis_client
        self.prefix = "acp:lock:"
        self._local_manager = AsyncioConcurrencyManager()

    async def acquire(self, session_id: str, strategy: str = "enqueue") -> None:
        # A distributed lock implies coordination across workers.
        # For 'interrupt', we would need to publish a cancel signal to a pub/sub channel.
        # For MVP OS-5.3, we use the local manager for task cancellation, plus a Redis distributed lock.
        await self._local_manager.acquire(session_id, strategy)

        lock_key = f"{self.prefix}{session_id}"
        # Simplified distributed lock acquisition loop
        while True:
            acquired = await self.redis.set(lock_key, "locked", nx=True, ex=300)
            if acquired:
                break
            if strategy == "reject":
                await self._local_manager.release(session_id)
                raise HTTPException(
                    status_code=409, detail="A run is already active for this session."
                )

            # Wait and retry for enqueue
            await asyncio.sleep(0.5)

    async def release(self, session_id: str) -> None:
        lock_key = f"{self.prefix}{session_id}"
        await self.redis.delete(lock_key)
        await self._local_manager.release(session_id)
