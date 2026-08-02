"""Contract tests for the bounded-fanout ``tmp_path`` fixture."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from _tmp_path_allocator import BoundedTempPathAllocator

pytest_plugins = ("pytester",)


def _node_ids_covering_all_buckets() -> list[str]:
    """Return deterministic node ids that exercise all 256 hash buckets."""
    by_bucket: dict[str, str] = {}
    index = 0
    while len(by_bucket) < 256:
        node_id = f"tests/unit/test_case.py::test_case[{index}]"
        bucket = hashlib.sha1(node_id.encode(), usedforsecurity=False).hexdigest()[:2]
        by_bucket.setdefault(bucket, node_id)
        index += 1
    return list(by_bucket.values())


def test_allocator_uses_deterministic_bounded_fanout(tmp_path: Path) -> None:
    allocator = BoundedTempPathAllocator(tmp_path)
    paths = [
        allocator.allocate(node_id) for node_id in _node_ids_covering_all_buckets()
    ]

    relative_paths = [path.relative_to(tmp_path) for path in paths]
    assert len(set(paths)) == 256
    assert {path.parts[0] for path in relative_paths} == {"t"}
    assert len({path.parts[1] for path in relative_paths}) == 256
    assert all(len(path.parts) == 3 for path in relative_paths)
    assert set(tmp_path.iterdir()) == {tmp_path / "t"}


def test_allocator_keeps_reruns_and_fresh_instances_isolated(tmp_path: Path) -> None:
    node_id = "tests/unit/test_case.py::test_same_name"
    allocator = BoundedTempPathAllocator(tmp_path)
    first = allocator.allocate(node_id)
    second = allocator.allocate(node_id)
    recovered = BoundedTempPathAllocator(tmp_path).allocate(node_id)

    assert first.parent == second.parent == recovered.parent
    assert [path.name.rsplit("-", 1)[-1] for path in (first, second, recovered)] == [
        "0",
        "1",
        "2",
    ]
    assert len({first, second, recovered}) == 3


def test_tmp_path_fixture_is_wired_to_bounded_allocator(
    tmp_path: Path,
    tmp_path_factory,
) -> None:
    relative_path = tmp_path.relative_to(tmp_path_factory.getbasetemp())

    assert relative_path.parts[0] == "t"
    assert len(relative_path.parts) == 3
    assert len(relative_path.parts[1]) == 2


def test_tmp_path_factory_api_remains_pytest_owned(tmp_path_factory) -> None:
    factory_path = tmp_path_factory.mktemp("factory_contract")

    assert factory_path.parent == tmp_path_factory.getbasetemp()
    assert factory_path.name == "factory_contract0"


def test_allocator_preserves_pytest_unix_socket_path_budget(tmp_path_factory) -> None:
    basetemp = tmp_path_factory.getbasetemp()
    allocated = BoundedTempPathAllocator(basetemp).allocate(
        "tests/unit/test_really_long_module_name.py::test_really_long_case_name"
    )
    socket_name = "eg-12345678.sock"
    stock_longest_leaf = basetemp / ("x" * 30 + "0") / socket_name

    assert len(os.fsencode(str(allocated / socket_name))) <= len(
        os.fsencode(str(stock_longest_leaf))
    )


def test_failed_setup_keeps_tmp_path_retention_cleanup_sound(pytester) -> None:
    pytester.syspathinsert(Path(__file__).parents[1])
    pytester.makeconftest('pytest_plugins = ("_tmp_path_allocator",)')
    pytester.makepyfile(
        """
        import pytest

        @pytest.fixture
        def broken_after_tmp_path(tmp_path):
            raise RuntimeError("expected setup failure")

        def test_broken_fixture(broken_after_tmp_path):
            pass
        """
    )

    result = pytester.runpytest(
        "-q",
        "-p",
        "no:asyncio",
        "-o",
        "tmp_path_retention_policy=failed",
    )

    result.assert_outcomes(errors=1)
