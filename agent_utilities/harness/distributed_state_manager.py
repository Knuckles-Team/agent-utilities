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


class BranchMergeStateLocker(OptimisticStateLocker):
    """Extends OptimisticStateLocker to support parallel state branching, staging,
    and concurrent merge resolution.

    CONCEPT: AHE-3.7 Distributed Agent State Manager - Concurrency Branching & Merging
    """

    def __init__(
        self, use_redis: bool = False, redis_url: str = "redis://localhost:6379"
    ):
        super().__init__(use_redis, redis_url)
        self._local_branches: dict[str, dict[str, Any]] = {}

    def get_branch_key(self, base_key: str, branch_name: str) -> str:
        return f"{base_key}:branch:{branch_name}"

    def fork_state(self, base_key: str, branch_name: str) -> dict[str, Any]:
        """Creates a parallel state branch from the base_key version.

        Returns the branched state dictionary containing 'data', 'base_version',
        and 'timestamp'.
        """
        base_state = self.get_state(base_key)
        if not base_state or not isinstance(base_state, dict):
            # Initialize empty base state if none exists
            base_state = {"data": {}, "version": 0, "timestamp": time.time()}
            self.update_state(base_key, {}, 0)

        base_data = base_state.get("data", {})
        if not isinstance(base_data, dict):
            base_data = {}

        version_val = base_state.get("version", 0)
        version_int = int(version_val) if isinstance(version_val, int | float) else 0

        # Create fork
        branched_state = {
            "data": dict(base_data),
            "base_version": version_int,
            "version": version_int,
            "timestamp": time.time(),
        }

        branch_key = self.get_branch_key(base_key, branch_name)
        if self.use_redis and self._redis_client:
            import json

            self._redis_client.set(branch_key, json.dumps(branched_state))
        else:
            self._local_branches[branch_key] = branched_state

        return branched_state

    def get_branch_state(
        self, base_key: str, branch_name: str
    ) -> dict[str, Any] | None:
        """Retrieve the branch state."""
        branch_key = self.get_branch_key(base_key, branch_name)
        if self.use_redis and self._redis_client:
            import json

            val = self._redis_client.get(branch_key)
            if val:
                return json.loads(val)
            return None
        return self._local_branches.get(branch_key)

    def update_branch_state(
        self, base_key: str, branch_name: str, new_data: dict[str, Any]
    ) -> bool:
        """Update a branch state's 'data' and bump its timestamp."""
        branch_key = self.get_branch_key(base_key, branch_name)
        branch_state = self.get_branch_state(base_key, branch_name)
        if not branch_state or not isinstance(branch_state, dict):
            return False

        branch_state["data"] = new_data
        branch_state["timestamp"] = time.time()

        if self.use_redis and self._redis_client:
            import json

            self._redis_client.set(branch_key, json.dumps(branch_state))
        else:
            self._local_branches[branch_key] = branch_state
        return True

    def merge_state(
        self, base_key: str, branch_name: str, resolver: Any = None
    ) -> bool:
        """Merges branched state back to base_key, resolving conflicts.

        If base_key version hasn't changed since fork_state, does a fast-forward.
        If base_key version has changed, uses the resolver callback or recursive dict-merge.
        """
        branch_state = self.get_branch_state(base_key, branch_name)
        if not branch_state or not isinstance(branch_state, dict):
            return False

        base_state = self.get_state(base_key)
        if not base_state or not isinstance(base_state, dict):
            base_state = {"data": {}, "version": 0, "timestamp": time.time()}

        b_version_val = base_state.get("version", 0)
        base_version = (
            int(b_version_val) if isinstance(b_version_val, int | float) else 0
        )

        fb_version_val = branch_state.get("base_version", 0)
        forked_base_version = (
            int(fb_version_val) if isinstance(fb_version_val, int | float) else 0
        )

        # Case 1: Fast-forward (no concurrent changes on base_key)
        if base_version == forked_base_version:
            branch_data = branch_state.get("data", {})
            if not isinstance(branch_data, dict):
                branch_data = {}
            success = self.update_state(base_key, branch_data, base_version)
            if success:
                branch_key = self.get_branch_key(base_key, branch_name)
                if self.use_redis and self._redis_client:
                    self._redis_client.delete(branch_key)
                elif branch_key in self._local_branches:
                    del self._local_branches[branch_key]
                return True
            return False

        # Case 2: Three-way merge / conflict resolution
        base_data = base_state.get("data", {})
        if not isinstance(base_data, dict):
            base_data = {}
        branch_data = branch_state.get("data", {})
        if not isinstance(branch_data, dict):
            branch_data = {}

        merged_data = dict(base_data)

        if resolver:
            try:
                merged_data = resolver(base_data, branch_data)
            except Exception:
                return False
        else:
            # Smart default dictionary merge
            for k, v in branch_data.items():
                if k not in merged_data:
                    merged_data[k] = v
                else:
                    if isinstance(merged_data[k], dict) and isinstance(v, dict):
                        merged_data[k] = self._recursive_merge(merged_data[k], v)
                    elif merged_data[k] == base_data.get(k):
                        merged_data[k] = v
                    elif v == base_data.get(k):
                        pass
                    else:
                        merged_data[k] = v

        success = self.update_state(base_key, merged_data, base_version)
        if success:
            branch_key = self.get_branch_key(base_key, branch_name)
            if self.use_redis and self._redis_client:
                self._redis_client.delete(branch_key)
            elif branch_key in self._local_branches:
                del self._local_branches[branch_key]
            return True

        return False

    def _recursive_merge(
        self, d1: dict[str, Any], d2: dict[str, Any]
    ) -> dict[str, Any]:
        result = dict(d1)
        for k, v in d2.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = self._recursive_merge(result[k], v)
            else:
                result[k] = v
        return result
