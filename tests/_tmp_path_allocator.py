"""Bounded-fanout implementation of pytest's ``tmp_path`` fixture.

Pytest's stock fixture places every test directory directly below one session
basetemp and enumerates that growing directory on each allocation.  Large
suites then turn otherwise independent test setup into quadratic directory
enumeration.
This plugin keeps pytest's per-session/per-xdist-worker basetemp authority, but
places a deterministic test-id directory under one of 256 bounded buckets.
"""

from __future__ import annotations

import base64
import hashlib
import re
import secrets
import shutil
import threading
from collections.abc import Generator
from pathlib import Path

import pytest

_BUCKET_DIRECTORY = "t"
_BUCKET_WIDTH = 2
# The bucket/hash components consume the same path budget pytest normally gives
# its 30-character test-name leaf.  The 64-bit allocation token needs 11
# URL-safe base64 characters; keeping five display characters preserves the
# Unix-socket path budget supplied by the lane's short basetemp contract.
_DISPLAY_NAME_LIMIT = 5
_ALLOCATION_TOKEN_BITS = 64
_ALLOCATION_TOKEN_LIMIT = 1 << _ALLOCATION_TOKEN_BITS
_MAX_LEAF_MKDIR_PROBES = 4
# A new bucket performs one mkdir for ``t`` and one for the bucket, followed by
# the fixed leaf budget.  pytest owns the already-existing basetemp itself.
_MAX_MKDIR_CALLS_PER_ALLOCATION = _MAX_LEAF_MKDIR_PROBES + 2
_NODE_NAME_UNSAFE = re.compile(r"[\W]")
_test_result_key = pytest.StashKey[dict[str, bool]]()


class BoundedTempPathAllocator:
    """Allocate unique test paths without directory enumeration.

    The hash is deterministic for a test node id, which makes a failed test's
    path easy to locate.  Each allocator starts a 64-bit counter at a random
    origin instead of ``0``, so recovery never relies on walking the retained
    ``-0``, ``-1``, ... namespace.  Atomic ``mkdir`` remains the authority
    across threads and processes: four distinct token candidates are attempted
    before the allocator raises rather than reusing an existing test directory.
    Xdist workers already receive separate pytest basetemps, so their bucket
    trees remain isolated by pytest's normal worker contract.
    """

    def __init__(self, basetemp: Path) -> None:
        self._root = basetemp / _BUCKET_DIRECTORY
        self._next_token = secrets.randbits(_ALLOCATION_TOKEN_BITS)
        self._created_buckets: set[Path] = set()
        self._lock = threading.Lock()

    def _take_allocation_token(self) -> str:
        """Return one distinct fixed-width token from this allocator's stream."""
        if self._next_token >= _ALLOCATION_TOKEN_LIMIT:
            raise RuntimeError(
                "bounded tmp_path allocation token space exhausted; "
                "start a fresh pytest basetemp"
            )

        token = (
            base64.urlsafe_b64encode(
                self._next_token.to_bytes(_ALLOCATION_TOKEN_BITS // 8, "big")
            )
            .decode("ascii")
            .rstrip("=")
        )
        self._next_token += 1
        return token

    def allocate(self, node_id: str) -> Path:
        """Create and return an empty, unique path for ``node_id``."""
        digest = hashlib.sha1(node_id.encode(), usedforsecurity=False).hexdigest()
        bucket = self._root / digest[:_BUCKET_WIDTH]
        display_name = _NODE_NAME_UNSAFE.sub("_", node_id.rsplit("::", 1)[-1])
        display_name = display_name[:_DISPLAY_NAME_LIMIT] or "tmp"
        stem = f"{display_name}-{digest[_BUCKET_WIDTH:10]}"

        with self._lock:
            if bucket not in self._created_buckets:
                self._root.mkdir(mode=0o700, exist_ok=True)
                bucket.mkdir(mode=0o700, exist_ok=True)
                self._created_buckets.add(bucket)

            for _ in range(_MAX_LEAF_MKDIR_PROBES):
                path = bucket / f"{stem}-{self._take_allocation_token()}"
                try:
                    path.mkdir(mode=0o700)
                except FileExistsError:
                    continue
                return path

            raise RuntimeError(
                "bounded tmp_path allocation collision probe budget exhausted "
                f"for {node_id!r}; no existing test directory was reused"
            )


@pytest.fixture(scope="session")
def _bounded_tmp_path_allocator(
    tmp_path_factory: pytest.TempPathFactory,
) -> BoundedTempPathAllocator:
    """Bind allocations to pytest's own lane- and xdist-isolated basetemp."""
    return BoundedTempPathAllocator(tmp_path_factory.getbasetemp())


@pytest.fixture
def tmp_path(
    request: pytest.FixtureRequest,
    _bounded_tmp_path_allocator: BoundedTempPathAllocator,
) -> Generator[Path]:
    """A drop-in ``tmp_path`` fixture with bounded directory fanout.

    It mirrors pytest's per-test ``tmp_path_retention_policy=failed`` cleanup
    behavior.  The ``all`` and ``none`` policy effects and session-basetemp
    lifecycle remain owned by pytest's ``TempPathFactory``, as they do for the
    stock fixture.
    """
    path = _bounded_tmp_path_allocator.allocate(request.node.nodeid)
    yield path

    result = request.node.stash[_test_result_key]
    try:
        if request.config.getini(
            "tmp_path_retention_policy"
        ) == "failed" and result.get("call", True):
            shutil.rmtree(path, ignore_errors=True)
    finally:
        del request.node.stash[_test_result_key]


@pytest.hookimpl(wrapper=True, trylast=True)
def pytest_runtest_makereport(
    item: pytest.Item,
    call: pytest.CallInfo[object],
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    """Record test-phase status for matching retention-policy cleanup."""
    report = yield
    item.stash.setdefault(_test_result_key, {})[report.when] = report.passed
    return report
