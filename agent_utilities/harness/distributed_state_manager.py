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
        self,
        use_redis: bool = False,
        redis_url: str | None = None,
        *,
        tls_profile: str | None = None,
        tls_profile_ref: str | None = None,
    ):
        self.use_redis = use_redis
        self._local_state: dict[str, dict[str, Any]] = {}
        self._redis_client: Any = None
        self._tls_trust: Any = None

        if self.use_redis:
            from urllib.parse import urlparse

            if not redis_url or urlparse(redis_url).scheme.casefold() != "rediss":
                raise ValueError("distributed Redis transport requires rediss://")
            try:
                import redis

                from agent_utilities.core.transport_security import (
                    resolve_configured_tls_profile,
                )

                self._tls_trust = resolve_configured_tls_profile(
                    "redis",
                    profile_name=tls_profile,
                    profile_ref=tls_profile_ref,
                )
                self._redis_client = redis.Redis.from_url(
                    redis_url,
                    decode_responses=True,
                    **self._tls_trust.redis_kwargs(),
                )
            except ImportError:
                self.use_redis = False
            except Exception:
                if self._tls_trust is not None:
                    self._tls_trust.cleanup()
                    self._tls_trust = None
                raise

    def close(self) -> None:
        """Close Redis and remove runtime TLS material."""
        if self._redis_client is not None:
            self._redis_client.close()
            self._redis_client = None
        if self._tls_trust is not None:
            self._tls_trust.cleanup()
            self._tls_trust = None

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
            # Immutable snapshot of base_data AT FORK TIME. "data" above is
            # the branch's own working copy and gets replaced wholesale by
            # every `update_branch_state` call; this field never changes
            # after fork and is the real common-ancestor reference a
            # three-way merge needs at merge time (see BUG-CX-037 /
            # `_default_dict_merge`) -- comparing against the CURRENT
            # base_data at merge time is not the same thing, since base_data
            # may have changed concurrently since the fork.
            "forked_data": dict(base_data),
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

    @staticmethod
    def _coerce_int_version(value: Any) -> int:
        return int(value) if isinstance(value, int | float) else 0

    def _delete_branch(self, base_key: str, branch_name: str) -> None:
        branch_key = self.get_branch_key(base_key, branch_name)
        if self.use_redis and self._redis_client:
            self._redis_client.delete(branch_key)
        elif branch_key in self._local_branches:
            del self._local_branches[branch_key]

    def _finalize_merge(
        self,
        base_key: str,
        branch_name: str,
        merged_data: dict[str, Any],
        base_version: int,
    ) -> bool:
        success = self.update_state(base_key, merged_data, base_version)
        if success:
            self._delete_branch(base_key, branch_name)
            return True
        return False

    def _fast_forward_merge(
        self,
        base_key: str,
        branch_name: str,
        branch_state: dict[str, Any],
        base_version: int,
    ) -> bool:
        branch_data = branch_state.get("data", {})
        if not isinstance(branch_data, dict):
            branch_data = {}
        return self._finalize_merge(base_key, branch_name, branch_data, base_version)

    def _default_dict_merge(
        self,
        base_data: dict[str, Any],
        branch_data: dict[str, Any],
        original_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Smart default dictionary merge. ``original_data`` is the
        # common-ancestor snapshot (base_data AT FORK TIME, see
        # ``fork_state``'s ``forked_data``) -- the real reference for
        # deciding "did base change this key since fork" / "did branch
        # change this key since fork". Comparing against the CURRENT
        # base_data (as this used to do, before BUG-CX-037) is trivially
        # always-true for merged_data[k] and made the branch's value win
        # unconditionally, discarding legitimate concurrent base changes.
        original_data = original_data if original_data is not None else {}
        merged_data = dict(base_data)
        for k, v in branch_data.items():
            if k not in merged_data:
                merged_data[k] = v
            elif isinstance(merged_data[k], dict) and isinstance(v, dict):
                merged_data[k] = self._recursive_merge(merged_data[k], v)
            elif merged_data[k] == original_data.get(k):
                # Base hasn't changed this key since fork -> branch's edit wins.
                merged_data[k] = v
            elif v == original_data.get(k):
                # Branch never touched this key -> keep base's concurrent edit.
                pass
            else:
                # Both sides changed the key to different values -> conflict;
                # branch wins (documented default resolution).
                merged_data[k] = v
        return merged_data

    def _resolve_conflict_merge(
        self,
        resolver: Any,
        base_data: dict[str, Any],
        branch_data: dict[str, Any],
        original_data: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any] | None, bool]:
        if resolver:
            try:
                return resolver(base_data, branch_data), True
            except Exception:
                return None, False
        return self._default_dict_merge(base_data, branch_data, original_data), True

    def _merged_state_or_none(
        self,
        resolver: Any,
        base_data: dict[str, Any],
        branch_data: dict[str, Any],
        original_data: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """The merged state, or ``None`` when the merge did not produce one.

        Collapses two failure modes the caller used to have to tell apart, and
        one it did not handle at all: `_resolve_conflict_merge`'s ``ok`` is
        False only for a resolver that RAISED, so a custom resolver that
        RETURNS None reached `update_state(base_key, None, ...)` and wrote None
        as the merged state. A merge that produced no state has not merged
        anything, so both are one ``None`` here.
        """
        merged_data, ok = self._resolve_conflict_merge(
            resolver, base_data, branch_data, original_data
        )
        return merged_data if ok else None

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

        base_version = self._coerce_int_version(base_state.get("version", 0))
        forked_base_version = self._coerce_int_version(
            branch_state.get("base_version", 0)
        )

        # Case 1: Fast-forward (no concurrent changes on base_key)
        if base_version == forked_base_version:
            return self._fast_forward_merge(
                base_key, branch_name, branch_state, base_version
            )

        # Case 2: Three-way merge / conflict resolution
        base_data = base_state.get("data", {})
        if not isinstance(base_data, dict):
            base_data = {}
        branch_data = branch_state.get("data", {})
        if not isinstance(branch_data, dict):
            branch_data = {}
        # Common-ancestor snapshot from fork time (see ``fork_state``). Falls
        # back to {} for a branch created before this field existed -- every
        # key then looks "changed" relative to the (empty) original, which
        # degrades to the pre-fix branch-wins-on-conflict behaviour rather
        # than crashing or fabricating a false common ancestor.
        original_data = branch_state.get("forked_data", {})
        if not isinstance(original_data, dict):
            original_data = {}

        merged_data = self._merged_state_or_none(
            resolver, base_data, branch_data, original_data
        )
        if merged_data is None:
            return False

        return self._finalize_merge(base_key, branch_name, merged_data, base_version)

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
