"""Bounded-fanout implementation of pytest's ``tmp_path`` fixture.

Pytest's stock fixture places every test directory directly below one session
basetemp and enumerates that growing directory on each allocation.  Large
suites then turn otherwise independent test setup into quadratic directory
enumeration.  This plugin keeps pytest's per-session/per-xdist-worker basetemp
authority, but places each allocation in a compact radix tree with a hard cap
at every allocator-managed directory.
"""

from __future__ import annotations

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
_ALLOCATION_TOKEN_MASK = _ALLOCATION_TOKEN_LIMIT - 1
_ALLOCATION_TOKEN_CHARACTERS = _ALLOCATION_TOKEN_BITS // 4
_ALLOCATION_TOKEN_PARENT_COMPONENT_WIDTHS = (4, 4, 4)
_LEAF_TOKEN_WIDTH = 4
# A 64-bit value occupies sixteen lowercase hexadecimal characters. The
# fixed-width 4/4/4/4 radix layout keeps every token component case-fold stable
# and outside the Windows device-name alphabet while preserving a shallow tree.
_MAX_LEAF_DIRECTORY_FANOUT = 1 << (_LEAF_TOKEN_WIDTH * 4)
_MAX_TOKEN_DIRECTORY_FANOUT = 1 << (max(_ALLOCATION_TOKEN_PARENT_COMPONENT_WIDTHS) * 4)
_MAX_ROOT_DIRECTORY_FANOUT = 1 << (_NODE_SHARD_WIDTH * 4)
_MAX_LEAF_MKDIR_PROBES = 4
_ALLOCATION_TOKEN_PARENT_COMPONENTS = len(_ALLOCATION_TOKEN_PARENT_COMPONENT_WIDTHS)
_ALLOCATION_PATH_COMPONENTS = 2 + _ALLOCATION_TOKEN_PARENT_COMPONENTS + 1
# A collision can cross the modular 64-bit boundary, so every probe may need a
# distinct token branch. The bound covers ``t``, the node shard, three token
# parents, and the final leaf for each fixed collision probe.
_MAX_MKDIR_CALLS_PER_ALLOCATION = 1 + _MAX_LEAF_MKDIR_PROBES * (
    _ALLOCATION_TOKEN_PARENT_COMPONENTS + 2
)
_test_result_key = pytest.StashKey[dict[str, bool]]()


class _DirectoryBranch:
    """One cached allocator directory and its bounded direct descendants."""

    __slots__ = ("children", "path")

    def __init__(self, path: Path) -> None:
        self.path = path
        self.children: dict[str, _DirectoryBranch] = {}


class BoundedTempPathAllocator:
    """Allocate unique test paths without directory enumeration.

    The first two SHA-1 hex characters deterministically shard a test node id,
    bounding ``t`` at 256 children.  A random 64-bit origin chooses a cyclic
    permutation of a separate 64-bit ordinal stream: ``(origin + ordinal) mod
    2**64``.  Thus a high random origin does not shorten the stream; every
    allocator can consume all ``2**64`` ordinals exactly once. Fixed-width
    Lowercase hexadecimal token groups become radix-tree levels. This bounds
    every lower allocator-managed directory at 65,536 children while using a
    case-fold-stable alphabet that cannot encode Windows device-name components.

    Atomic ``mkdir`` remains the authority across threads and processes: four
    distinct token candidates are attempted before the allocator raises rather
    than reusing an existing test directory.  Xdist workers already receive
    separate pytest basetemps, so their allocation trees remain isolated by
    pytest's normal worker contract.
    """

    def __init__(self, basetemp: Path) -> None:
        self._root = basetemp / _ALLOCATION_ROOT_DIRECTORY
        self._token_origin = (
            secrets.randbits(_ALLOCATION_TOKEN_BITS) & _ALLOCATION_TOKEN_MASK
        )
        # The ordinal is deliberately independent of the random collision
        # namespace.  It always has the complete 64-bit allocation range.
        self._next_token = 0
        self._root_created = False
        self._directory_tree: dict[str, _DirectoryBranch] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _encode_token(value: int) -> str:
        """Return one fixed-width lowercase-hex representation of ``value``."""
        return f"{value:0{_ALLOCATION_TOKEN_CHARACTERS}x}"

    def _take_allocation_token(self) -> str:
        """Return one token from a full ordinal stream, permuted by its origin."""
        if self._next_token >= _ALLOCATION_TOKEN_LIMIT:
            raise RuntimeError(
                "bounded tmp_path allocation token space exhausted; "
                "start a fresh pytest basetemp"
            )

        token = self._encode_token(
            (self._token_origin + self._next_token) & _ALLOCATION_TOKEN_MASK
        )
        self._next_token += 1
        return token

    def _allocation_parent(self, node_shard: str, token: str) -> Path:
        """Create the bounded token branch and return its final parent."""
        parent = self._root
        branches = self._directory_tree
        branch = branches.get(node_shard)
        if branch is None:
            directory = parent / node_shard
            directory.mkdir(mode=0o700, exist_ok=True)
            branch = _DirectoryBranch(directory)
            branches[node_shard] = branch
        parent = branch.path
        branches = branch.children

        token_index = 0
        for component_width in _ALLOCATION_TOKEN_PARENT_COMPONENT_WIDTHS:
            component = token[token_index : token_index + component_width]
            branch = branches.get(component)
            if branch is None:
                directory = parent / component
                directory.mkdir(mode=0o700, exist_ok=True)
                branch = _DirectoryBranch(directory)
                branches[component] = branch
            parent = branch.path
            branches = branch.children
            token_index += component_width
        return parent

    def allocate(self, node_id: str) -> Path:
        """Create and return an empty, unique path for ``node_id``."""
        digest = hashlib.sha1(node_id.encode(), usedforsecurity=False).hexdigest()
        node_shard = digest[:_NODE_SHARD_WIDTH]

        with self._lock:
            if not self._root_created:
                self._root.mkdir(mode=0o700, exist_ok=True)
                self._root_created = True

            for _ in range(_MAX_LEAF_MKDIR_PROBES):
                token = self._take_allocation_token()
                allocation_parent = self._allocation_parent(node_shard, token)
                path = allocation_parent / token[-_LEAF_TOKEN_WIDTH:]
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
