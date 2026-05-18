"""Distributed Agent State Manager (AHE-3.7).

CONCEPT: AHE-3.7 Distributed Agent State Manager

Enhances memory tiers by adding an OptimisticStateLocker to prevent race conditions
during high-frequency simulated execution. Optionally supports Redis for scalability.
"""

import time
from typing import Any


class OptimisticStateLocker:
    """Manages distributed state using optimistic locking with optional Redis support."""

    def __init__(
        self, use_redis: bool = False, redis_url: str = "redis://localhost:6379"
    ):
        self.use_redis = use_redis
        self._local_state: dict[str, dict[str, Any]] = {}
        self._redis_client = None

        if self.use_redis:
            try:
                import redis

                self._redis_client = redis.Redis.from_url(
                    redis_url, decode_responses=True
                )
            except ImportError:
                self.use_redis = False

    def get_state(self, key: str) -> dict[str, Any] | None:
        """Retrieve the current state and its version."""
        if self.use_redis and self._redis_client:
            import json

            val = self._redis_client.get(key)
            if val:
                return json.loads(val)
            return None

        return self._local_state.get(key)

    def update_state(
        self, key: str, new_data: dict[str, Any], expected_version: int
    ) -> bool:
        """Optimistically update state only if the expected version matches the current version."""
        current_state = self.get_state(key)
        current_version = current_state.get("version", 0) if current_state else 0

        if current_version != expected_version:
            return False

        new_state = {
            "data": new_data,
            "version": current_version + 1,
            "timestamp": time.time(),
        }

        if self.use_redis and self._redis_client:
            import json

            pipeline = self._redis_client.pipeline()
            try:
                pipeline.watch(key)
                val = pipeline.get(key)
                curr_v = json.loads(val).get("version", 0) if val else 0
                if curr_v != expected_version:
                    return False
                pipeline.multi()
                pipeline.set(key, json.dumps(new_state))
                pipeline.execute()
                return True
            except Exception:
                return False
        else:
            self._local_state[key] = new_state
            return True
