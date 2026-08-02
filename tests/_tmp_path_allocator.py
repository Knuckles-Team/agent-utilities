"""Bounded-fanout implementation of pytest's ``tmp_path`` fixture.

Pytest's stock fixture places every test directory directly below one session
basetemp and enumerates that growing directory on each allocation.  Large
suites then turn otherwise independent test setup into quadratic directory
enumeration.
This plugin keeps pytest's per-session/per-xdist-worker basetemp authority, but
places a deterministic test-id shard below its own root and uses bounded token
slots for the allocation leaves.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
import shutil
import threading
from collections.abc import Generator
from pathlib import Path

import pytest

_ALLOCATION_ROOT_DIRECTORY = "t"
_NODE_SHARD_WIDTH = 2
_ALLOCATION_TOKEN_BITS = 64
_ALLOCATION_TOKEN_LIMIT = 1 << _ALLOCATION_TOKEN_BITS
_ALLOCATION_TOKEN_CHARACTERS = (_ALLOCATION_TOKEN_BITS + 5) // 6
_LEAF_TOKEN_WIDTH = 2
_ALLOCATION_GROUP_TOKEN_WIDTH = _ALLOCATION_TOKEN_CHARACTERS - _LEAF_TOKEN_WIDTH
# A 64-bit value occupies eleven base64 characters: the final character carries
# four data bits, so the final two-character slot has exactly ten bits of fanout.
_MAX_LEAF_DIRECTORY_FANOUT = 1 << (_LEAF_TOKEN_WIDTH * 6 - 2)
_MAX_LEAF_MKDIR_PROBES = 4
# A candidate may cross one 1,024-slot group boundary, creating the allocation
# root and two allocation parents before the fixed leaf probe budget is exhausted.
_MAX_MKDIR_CALLS_PER_ALLOCATION = _MAX_LEAF_MKDIR_PROBES + 3
_test_result_key = pytest.StashKey[dict[str, bool]]()


class BoundedTempPathAllocator:
    """Allocate unique test paths without directory enumeration.

    The first two SHA-1 hex characters deterministically shard a test node id.
    Each allocator starts a 64-bit counter at a random origin instead of ``0``;
    its full URL-safe base64 origin and the origin-relative group select an
    allocation parent.  The two-character relative slot is the leaf name, so
    every allocation parent has at most 1,024 possible child directories even
    across retained runs and independent processes.  Atomic ``mkdir`` remains
    the authority across threads and processes: four distinct token candidates
    are attempted before the allocator raises rather than reusing an existing
    test directory.  Xdist workers already receive separate pytest basetemps,
    so their allocation trees remain isolated by pytest's normal worker contract.
    """

    def __init__(self, basetemp: Path) -> None:
        self._root = basetemp / _ALLOCATION_ROOT_DIRECTORY
        self._token_origin = secrets.randbits(_ALLOCATION_TOKEN_BITS)
        self._origin_token = self._encode_token(self._token_origin)
        self._next_token = self._token_origin
        self._root_created = False
        self._created_allocation_parents: set[Path] = set()
        self._lock = threading.Lock()

    @staticmethod
    def _encode_token(value: int) -> str:
        """Return one fixed-width URL-safe base64 representation of ``value``."""
        return (
            base64.urlsafe_b64encode(value.to_bytes(_ALLOCATION_TOKEN_BITS // 8, "big"))
            .decode("ascii")
            .rstrip("=")
        )

    def _take_allocation_token(self) -> str:
        """Return one origin-relative token from this allocator's random stream."""
        if self._next_token >= _ALLOCATION_TOKEN_LIMIT:
            raise RuntimeError(
                "bounded tmp_path allocation token space exhausted; "
                "start a fresh pytest basetemp"
            )

        token = self._encode_token(self._next_token - self._token_origin)
        self._next_token += 1
        return token

    def allocate(self, node_id: str) -> Path:
        """Create and return an empty, unique path for ``node_id``."""
        digest = hashlib.sha1(node_id.encode(), usedforsecurity=False).hexdigest()
        node_shard = digest[:_NODE_SHARD_WIDTH]

        with self._lock:
            if not self._root_created:
                self._root.mkdir(mode=0o700, exist_ok=True)
                self._root_created = True

            for _ in range(_MAX_LEAF_MKDIR_PROBES):
                relative_token = self._take_allocation_token()
                allocation_parent = self._root / (
                    f"{node_shard}{self._origin_token}"
                    f"{relative_token[:_ALLOCATION_GROUP_TOKEN_WIDTH]}"
                )
                if allocation_parent not in self._created_allocation_parents:
                    allocation_parent.mkdir(mode=0o700, exist_ok=True)
                    self._created_allocation_parents.add(allocation_parent)

                path = allocation_parent / relative_token[-_LEAF_TOKEN_WIDTH:]
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
