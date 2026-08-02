"""Bounded-fanout implementation of pytest's ``tmp_path`` fixture.

Pytest's stock fixture places every test directory directly below one session
basetemp and scans that whole directory on each allocation.  Large suites then
turn otherwise independent test setup into quadratic directory enumeration.
This plugin keeps pytest's per-session/per-xdist-worker basetemp authority, but
places a deterministic test-id directory under one of 256 bounded buckets.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import threading
from collections.abc import Generator
from pathlib import Path

import pytest

_BUCKET_DIRECTORY = "t"
_BUCKET_WIDTH = 2
# The bucket/hash components consume the same path budget pytest normally gives
# its 30-character test-name leaf.  Keeping this at 15 protects Unix-socket
# fixtures that already depend on the lane's short basetemp contract.
_DISPLAY_NAME_LIMIT = 15
_NODE_NAME_UNSAFE = re.compile(r"[\W]")
_test_result_key = pytest.StashKey[dict[str, bool]]()


class BoundedTempPathAllocator:
    """Allocate unique test paths without scanning a growing session root.

    The hash is deterministic for a test node id, which makes a failed test's
    path easy to locate.  The per-node attempt counter keeps reruns isolated;
    a fresh allocator also advances past an existing attempt without scanning
    the bucket.  Xdist workers already receive separate pytest basetemps, so
    their bucket trees remain isolated by pytest's normal worker contract.
    """

    def __init__(self, basetemp: Path) -> None:
        self._root = basetemp / _BUCKET_DIRECTORY
        self._attempts: dict[str, int] = {}
        self._lock = threading.Lock()

    def allocate(self, node_id: str) -> Path:
        """Create and return an empty, unique path for ``node_id``."""
        digest = hashlib.sha1(node_id.encode(), usedforsecurity=False).hexdigest()
        bucket = self._root / digest[:_BUCKET_WIDTH]
        display_name = _NODE_NAME_UNSAFE.sub("_", node_id.rsplit("::", 1)[-1])
        display_name = display_name[:_DISPLAY_NAME_LIMIT] or "test"
        stem = f"{display_name}-{digest[_BUCKET_WIDTH:10]}"

        with self._lock:
            bucket.mkdir(parents=True, exist_ok=True)
            attempt = self._attempts.get(node_id, 0)
            while True:
                path = bucket / f"{stem}-{attempt}"
                try:
                    path.mkdir(mode=0o700)
                except FileExistsError:
                    attempt += 1
                    continue
                self._attempts[node_id] = attempt + 1
                return path


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

    It mirrors pytest's ``tmp_path_retention_policy=failed`` cleanup behavior,
    while leaving the policy and the session basetemp lifecycle owned by pytest.
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
