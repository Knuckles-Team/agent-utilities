"""Contract tests for the bounded-fanout ``tmp_path`` fixture."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from _tmp_path_allocator import (
    _ALLOCATION_PATH_COMPONENTS,
    _ALLOCATION_TOKEN_LIMIT,
    _ALLOCATION_TOKEN_PARENT_COMPONENT_WIDTHS,
    _LEAF_TOKEN_WIDTH,
    _MAX_LEAF_DIRECTORY_FANOUT,
    _MAX_LEAF_MKDIR_PROBES,
    _MAX_MKDIR_CALLS_PER_ALLOCATION,
    _MAX_ROOT_DIRECTORY_FANOUT,
    _MAX_TOKEN_DIRECTORY_FANOUT,
    _TOKEN_DIRECTORY_COMPONENT_PREFIX,
    BoundedTempPathAllocator,
)

pytest_plugins = ("pytester",)

_RECORD_DIRECTORY_ENV = "BOUNDED_TMP_PATH_RECORD_DIRECTORY"
_RETENTION_CASES = frozenset({"call_pass", "call_failure", "setup_failure"})
_RETENTION_EXPECTATIONS = {
    "all": {"call_pass": True, "call_failure": True, "setup_failure": True},
    "failed": {"call_pass": False, "call_failure": True, "setup_failure": False},
    "none": {"call_pass": True, "call_failure": True, "setup_failure": True},
}
_WINDOWS_RESERVED_DIRECTORY_NAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{number}" for number in range(1, 10)),
        *(f"LPT{number}" for number in range(1, 10)),
    }
)


def _install_allocator_plugin(pytester, directory: str) -> Path:
    """Create one child-test directory with the public fixture plugin loaded."""
    pytester.makefile(
        ".py",
        **{
            f"{directory}/conftest": 'pytest_plugins = ("_tmp_path_allocator",)',
        },
    )
    return pytester.path / directory


def _prepare_child_pytest(pytester, monkeypatch) -> None:
    """Make the test-only plugin importable by child pytest subprocesses."""
    tests_directory = str(Path(__file__).parents[1])
    pytester.syspathinsert(tests_directory)
    current_python_path = os.environ.get("PYTHONPATH")
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join(
            path for path in (tests_directory, current_python_path) if path
        ),
    )


def _read_records(record_directory: Path) -> list[dict[str, str]]:
    """Read child-pytest fixture observations from their out-of-basetemp home."""
    records: list[dict[str, str]] = []
    for record_path in sorted(record_directory.glob("*.json")):
        payload = json.loads(record_path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        records.append({str(key): str(value) for key, value in payload.items()})
    return records


def _retention_paths(record_directory: Path) -> dict[str, Path]:
    """Return the exact child-test path allocated for each retention outcome."""
    records = _read_records(record_directory)
    paths = {record["case"]: Path(record["path"]) for record in records}
    assert paths.keys() == _RETENTION_CASES
    return paths


def _allocator_root(path: Path) -> Path:
    """Return the fixed-depth ``t`` root for one allocator-created path."""
    root = path.parents[_ALLOCATION_PATH_COMPONENTS - 2]
    assert root.name == "t"
    return root


def _allocator_relative_parts(path: Path) -> tuple[str, ...]:
    """Return the allocator's fixed-width relative layout components."""
    root = _allocator_root(path)
    return path.relative_to(root.parent).parts


def _allocation_token_from_parts(parts: tuple[str, ...]) -> str:
    """Recover the encoded token from its filesystem-safe path components."""
    return (
        "".join(
            component.removeprefix(_TOKEN_DIRECTORY_COMPONENT_PREFIX)
            for component in parts[2:-1]
        )
        + parts[-1]
    )


def _assert_bounded_fixture_paths(paths: dict[str, Path]) -> None:
    """Assert real fixture paths use the documented bounded-fanout layout."""
    for path in paths.values():
        parts = _allocator_relative_parts(path)
        assert parts[0] == "t"
        assert len(parts) == _ALLOCATION_PATH_COMPONENTS
        assert len(parts[1]) == 2
        assert tuple(len(part) for part in parts[2:-1]) == tuple(
            len(_TOKEN_DIRECTORY_COMPONENT_PREFIX) + width
            for width in _ALLOCATION_TOKEN_PARENT_COMPONENT_WIDTHS
        )
        assert all(
            part.startswith(_TOKEN_DIRECTORY_COMPONENT_PREFIX) for part in parts[2:-1]
        )
        assert len(parts[-1]) == _LEAF_TOKEN_WIDTH


def _assert_every_allocator_directory_is_bounded(root: Path) -> None:
    """Check the hard fanout cap at every allocator-managed tree level."""
    pending = [root]
    while pending:
        directory = pending.pop()
        child_directories = [child for child in directory.iterdir() if child.is_dir()]
        relative_depth = len(directory.relative_to(root).parts)
        if relative_depth == 0:
            cap = _MAX_ROOT_DIRECTORY_FANOUT
        elif relative_depth == _ALLOCATION_PATH_COMPONENTS - 2:
            cap = _MAX_LEAF_DIRECTORY_FANOUT
        else:
            cap = _MAX_TOKEN_DIRECTORY_FANOUT
        assert len(child_directories) <= cap
        if relative_depth < _ALLOCATION_PATH_COMPONENTS - 2:
            pending.extend(child_directories)


_RETENTION_SUITE = """
import json
import os
from pathlib import Path

import pytest


def _record(case: str, tmp_path: Path) -> None:
    record_path = Path(os.environ["BOUNDED_TMP_PATH_RECORD_DIRECTORY"]) / f"{case}.json"
    record_path.write_text(
        json.dumps({"case": case, "path": str(tmp_path)}), encoding="utf-8"
    )


def test_call_pass(tmp_path: Path) -> None:
    _record("call_pass", tmp_path)


def test_call_failure(tmp_path: Path) -> None:
    _record("call_failure", tmp_path)
    assert False, "expected call failure"


@pytest.fixture
def broken_after_tmp_path(tmp_path: Path) -> None:
    _record("setup_failure", tmp_path)
    raise RuntimeError("expected setup failure")


def test_setup_failure(broken_after_tmp_path: None) -> None:
    pass
"""


def _run_retention_suite(
    pytester,
    monkeypatch,
    *,
    child_directory: Path,
    record_directory: Path,
    policy: str,
) -> dict[str, Path]:
    """Run the three real fixture outcomes under one exact pytest policy."""
    record_directory.mkdir()
    monkeypatch.setenv(_RECORD_DIRECTORY_ENV, str(record_directory))
    result = pytester.runpytest_subprocess(
        "-q",
        "-p",
        "no:asyncio",
        "-o",
        f"tmp_path_retention_policy={policy}",
        child_directory,
        timeout=60,
    )

    result.assert_outcomes(passed=1, failed=1, errors=1)
    return _retention_paths(record_directory)


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
    root = tmp_path / "t"
    assert len(set(paths)) == 256
    assert {path.parts[0] for path in relative_paths} == {"t"}
    assert len({path.parts[1] for path in relative_paths}) == 256
    assert all(
        len(path.parts) == _ALLOCATION_PATH_COMPONENTS for path in relative_paths
    )
    assert all(len(path.parts[1]) == 2 for path in relative_paths)
    assert all(
        tuple(len(part) for part in path.parts[2:-1])
        == tuple(
            len(_TOKEN_DIRECTORY_COMPONENT_PREFIX) + width
            for width in _ALLOCATION_TOKEN_PARENT_COMPONENT_WIDTHS
        )
        and all(
            part.startswith(_TOKEN_DIRECTORY_COMPONENT_PREFIX)
            for part in path.parts[2:-1]
        )
        and len(path.parts[-1]) == _LEAF_TOKEN_WIDTH
        for path in relative_paths
    )
    assert set(tmp_path.iterdir()) == {root}
    assert len([child for child in root.iterdir() if child.is_dir()]) == 256
    _assert_every_allocator_directory_is_bounded(root)


@pytest.mark.parametrize("reserved_name", ("CON", "PRN", "AUX", "NUL"))
def test_allocator_avoids_windows_reserved_token_directories(
    tmp_path: Path, reserved_name: str
) -> None:
    """A token prefix cannot create a Windows device-name path component."""
    encoded_token = f"{reserved_name}{'A' * 8}"
    token_value = int.from_bytes(
        base64.urlsafe_b64decode(f"{encoded_token}="), byteorder="big"
    )
    allocator = BoundedTempPathAllocator(tmp_path)
    allocator._token_origin = token_value

    path = allocator.allocate("tests/unit/test_case.py::test_windows_names")
    parts = _allocator_relative_parts(path)

    assert allocator._encode_token(token_value) == encoded_token
    assert parts[2] == f"{_TOKEN_DIRECTORY_COMPONENT_PREFIX}{reserved_name}"
    assert all(
        component.upper() not in _WINDOWS_RESERVED_DIRECTORY_NAMES
        for component in parts[1:]
    )
    assert _allocation_token_from_parts(parts) == encoded_token


def test_allocator_bounds_every_directory_under_adversarial_node_prefix_spread(
    tmp_path: Path,
) -> None:
    """All 256 node shards cannot grow ``t`` beyond its fixed radix cap."""
    allocator = BoundedTempPathAllocator(tmp_path)
    node_ids = _node_ids_covering_all_buckets()
    paths = [allocator.allocate(node_id) for node_id in node_ids for _ in range(20)]
    root = tmp_path / "t"

    assert len(paths) == len(set(paths)) == 5_120
    root_children = [child for child in root.iterdir() if child.is_dir()]
    assert len(root_children) == _MAX_ROOT_DIRECTORY_FANOUT
    _assert_every_allocator_directory_is_bounded(root)


def test_allocator_keeps_reruns_and_fresh_instances_isolated(
    tmp_path: Path, monkeypatch
) -> None:
    node_id = "tests/unit/test_case.py::test_same_name"
    token_origins = iter((0, 0))
    monkeypatch.setattr(
        "_tmp_path_allocator.secrets.randbits", lambda _: next(token_origins)
    )
    allocator = BoundedTempPathAllocator(tmp_path)
    first = allocator.allocate(node_id)
    second = allocator.allocate(node_id)
    recovered = BoundedTempPathAllocator(tmp_path).allocate(node_id)

    assert first.parent == second.parent == recovered.parent
    assert len({first, second, recovered}) == 3


def test_allocator_bounds_fresh_recovery_after_many_retained_attempts(
    tmp_path: Path, monkeypatch
) -> None:
    """A fresh allocator does not walk every retained attempt by ``mkdir``."""
    node_id = "tests/unit/test_case.py::test_retained_attempts"
    token_origins = iter((0, 1 << 63))
    monkeypatch.setattr(
        "_tmp_path_allocator.secrets.randbits", lambda _: next(token_origins)
    )
    retained_allocator = BoundedTempPathAllocator(tmp_path)
    retained_paths = [retained_allocator.allocate(node_id) for _ in range(2_048)]
    fresh_allocator = BoundedTempPathAllocator(tmp_path)
    mkdir_calls: list[Path] = []
    original_mkdir = Path.mkdir

    def observe_mkdir(path: Path, *args, **kwargs) -> None:
        mkdir_calls.append(path)
        original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", observe_mkdir)
    allocated = fresh_allocator.allocate(node_id)

    assert len(retained_paths) == 2_048
    assert allocated not in retained_paths
    leaf_counts = Counter(path.parent for path in retained_paths)
    assert max(leaf_counts.values()) <= _MAX_LEAF_DIRECTORY_FANOUT
    assert allocated.is_dir()
    assert len(mkdir_calls) <= _MAX_MKDIR_CALLS_PER_ALLOCATION
    _assert_every_allocator_directory_is_bounded(tmp_path / "t")


def test_allocator_keeps_independent_allocators_concurrently_unique(
    tmp_path: Path, monkeypatch
) -> None:
    """Independent locks emulate the atomic mkdir boundary used by processes."""
    token_origins = iter([0] * _MAX_LEAF_MKDIR_PROBES)
    monkeypatch.setattr(
        "_tmp_path_allocator.secrets.randbits", lambda _: next(token_origins)
    )
    node_id = "tests/unit/test_case.py::test_independent_allocators"
    allocators = [
        BoundedTempPathAllocator(tmp_path) for _ in range(_MAX_LEAF_MKDIR_PROBES)
    ]

    with ThreadPoolExecutor(max_workers=_MAX_LEAF_MKDIR_PROBES) as executor:
        paths = list(
            executor.map(lambda allocator: allocator.allocate(node_id), allocators)
        )

    assert len(paths) == len(set(paths)) == _MAX_LEAF_MKDIR_PROBES
    assert all(path.is_dir() for path in paths)


def test_allocator_keeps_shared_basetemp_processes_unique(tmp_path: Path) -> None:
    """Use real OS processes to prove ``mkdir`` is the shared authority."""
    shared_basetemp = tmp_path / "shared-process-basetemp"
    shared_basetemp.mkdir()
    tests_directory = str(Path(__file__).parents[1])
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        path for path in (tests_directory, environment.get("PYTHONPATH")) if path
    )
    child_program = "\n".join(
        (
            "import sys",
            "from pathlib import Path",
            "import _tmp_path_allocator as allocator_module",
            "allocator_module.secrets.randbits = lambda _: 0",
            "allocator = allocator_module.BoundedTempPathAllocator(Path(sys.argv[1]))",
            "print(allocator.allocate('tests/unit/test_process.py::test_shared'))",
        )
    )
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", child_program, str(shared_basetemp)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
        )
        for _ in range(_MAX_LEAF_MKDIR_PROBES)
    ]
    outcomes = [process.communicate(timeout=30) for process in processes]

    assert all(process.returncode == 0 for process in processes), outcomes
    paths = [Path(stdout.strip()) for stdout, _ in outcomes]
    assert len(paths) == len(set(paths)) == _MAX_LEAF_MKDIR_PROBES
    assert len({path.parent for path in paths}) == 1
    assert all(path.is_dir() for path in paths)


def test_allocator_refuses_after_its_fixed_collision_budget(
    tmp_path: Path, monkeypatch
) -> None:
    """A saturated candidate window fails explicitly without reusing a path."""
    node_id = "tests/unit/test_case.py::test_collision_budget"
    token_origins = iter((0, 0))
    monkeypatch.setattr(
        "_tmp_path_allocator.secrets.randbits", lambda _: next(token_origins)
    )
    retained_allocator = BoundedTempPathAllocator(tmp_path)
    retained_paths = [
        retained_allocator.allocate(node_id) for _ in range(_MAX_LEAF_MKDIR_PROBES)
    ]
    fresh_allocator = BoundedTempPathAllocator(tmp_path)
    mkdir_calls: list[Path] = []
    original_mkdir = Path.mkdir

    def observe_mkdir(path: Path, *args, **kwargs) -> None:
        mkdir_calls.append(path)
        original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", observe_mkdir)
    with pytest.raises(RuntimeError, match="collision probe budget exhausted"):
        fresh_allocator.allocate(node_id)

    assert all(path.is_dir() for path in retained_paths)
    assert len(mkdir_calls) <= _MAX_MKDIR_CALLS_PER_ALLOCATION


def test_allocator_bounds_collision_calls_across_a_group_boundary(
    tmp_path: Path, monkeypatch
) -> None:
    """A four-probe collision window remains bounded across a leaf radix edge."""
    node_id = "tests/unit/test_case.py::test_group_boundary"
    token_origins = iter((0, 0))
    monkeypatch.setattr(
        "_tmp_path_allocator.secrets.randbits", lambda _: next(token_origins)
    )
    retained_allocator = BoundedTempPathAllocator(tmp_path)
    retained_allocator._next_token = _MAX_LEAF_DIRECTORY_FANOUT - 2
    retained_paths = [
        retained_allocator.allocate(node_id) for _ in range(_MAX_LEAF_MKDIR_PROBES)
    ]
    fresh_allocator = BoundedTempPathAllocator(tmp_path)
    fresh_allocator._next_token = _MAX_LEAF_DIRECTORY_FANOUT - 2
    mkdir_calls: list[Path] = []
    original_mkdir = Path.mkdir

    def observe_mkdir(path: Path, *args, **kwargs) -> None:
        mkdir_calls.append(path)
        original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", observe_mkdir)
    with pytest.raises(RuntimeError, match="collision probe budget exhausted"):
        fresh_allocator.allocate(node_id)

    assert len({path.parent for path in retained_paths}) == 2
    assert len(mkdir_calls) <= _MAX_MKDIR_CALLS_PER_ALLOCATION


def test_allocator_refuses_token_stream_overflow(tmp_path: Path) -> None:
    """A counter overflow is explicit rather than silently wrapping or reusing."""
    allocator = BoundedTempPathAllocator(tmp_path)
    allocator._next_token = _ALLOCATION_TOKEN_LIMIT

    with pytest.raises(RuntimeError, match="token space exhausted"):
        allocator.allocate("tests/unit/test_case.py::test_token_stream_overflow")


def test_allocator_full_stream_is_independent_of_max_random_origin(
    tmp_path: Path, monkeypatch
) -> None:
    """A maximal random seed wraps tokens but never shortens the ordinal stream."""
    maximum_origin = _ALLOCATION_TOKEN_LIMIT - 1
    monkeypatch.setattr(
        "_tmp_path_allocator.secrets.randbits", lambda _: maximum_origin
    )
    allocator = BoundedTempPathAllocator(tmp_path)
    node_id = "tests/unit/test_case.py::test_max_origin"

    first = allocator.allocate(node_id)
    second = allocator.allocate(node_id)
    assert _allocation_token_from_parts(
        _allocator_relative_parts(first)
    ) == allocator._encode_token(maximum_origin)
    assert _allocation_token_from_parts(
        _allocator_relative_parts(second)
    ) == allocator._encode_token(0)

    allocator._next_token = _ALLOCATION_TOKEN_LIMIT - 1
    final = allocator.allocate(node_id)
    assert _allocation_token_from_parts(
        _allocator_relative_parts(final)
    ) == allocator._encode_token(_ALLOCATION_TOKEN_LIMIT - 2)
    with pytest.raises(RuntimeError, match="token space exhausted"):
        allocator.allocate(node_id)


def test_allocator_creates_active_root_and_parent_once(
    tmp_path: Path, monkeypatch
) -> None:
    """Avoid a redundant ``mkdir(..., exist_ok=True)`` syscall per allocation."""
    mkdir_calls: list[Path] = []
    original_mkdir = Path.mkdir
    monkeypatch.setattr("_tmp_path_allocator.secrets.randbits", lambda _: 0)

    def observe_mkdir(path: Path, *args, **kwargs) -> None:
        mkdir_calls.append(path)
        original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", observe_mkdir)
    allocator = BoundedTempPathAllocator(tmp_path)
    first = allocator.allocate("tests/unit/test_case.py::test_same_bucket")
    root = _allocator_root(first)
    node_directory = root / _allocator_relative_parts(first)[1]
    root_mkdir_calls = mkdir_calls.count(root)
    node_mkdir_calls = mkdir_calls.count(node_directory)
    allocation_parent_mkdir_calls = mkdir_calls.count(first.parent)
    second = allocator.allocate("tests/unit/test_case.py::test_same_bucket")

    assert first.parent == second.parent
    assert root_mkdir_calls >= 1
    assert node_mkdir_calls >= 1
    assert allocation_parent_mkdir_calls >= 1
    assert mkdir_calls.count(root) == root_mkdir_calls
    assert mkdir_calls.count(node_directory) == node_mkdir_calls
    assert mkdir_calls.count(first.parent) == allocation_parent_mkdir_calls


def test_allocator_4000_allocation_measurement(tmp_path: Path) -> None:
    """Measure suite-scale allocation while asserting the bounded layout."""
    allocator = BoundedTempPathAllocator(tmp_path)
    node_ids = [
        f"tests/unit/test_many_cases.py::test_case[{index}]" for index in range(4000)
    ]

    started = time.perf_counter()
    paths = [allocator.allocate(node_id) for node_id in node_ids]
    elapsed = time.perf_counter() - started

    print(f"bounded_tmp_path_4000_allocations={elapsed:.6f}s")
    assert len(paths) == len(set(paths)) == 4000
    root = tmp_path / "t"
    assert (
        len([path for path in root.iterdir() if path.is_dir()])
        <= _MAX_ROOT_DIRECTORY_FANOUT
    )
    leaf_counts = Counter(path.parent for path in paths)
    assert max(leaf_counts.values()) <= _MAX_LEAF_DIRECTORY_FANOUT
    _assert_every_allocator_directory_is_bounded(root)


def test_tmp_path_fixture_is_wired_to_bounded_allocator(
    tmp_path: Path,
    tmp_path_factory,
) -> None:
    relative_path = tmp_path.relative_to(tmp_path_factory.getbasetemp())

    assert relative_path.parts[0] == "t"
    assert len(relative_path.parts) == _ALLOCATION_PATH_COMPONENTS
    assert len(relative_path.parts[1]) == 2
    assert tuple(len(part) for part in relative_path.parts[2:-1]) == tuple(
        len(_TOKEN_DIRECTORY_COMPONENT_PREFIX) + width
        for width in _ALLOCATION_TOKEN_PARENT_COMPONENT_WIDTHS
    )
    assert all(
        part.startswith(_TOKEN_DIRECTORY_COMPONENT_PREFIX)
        for part in relative_path.parts[2:-1]
    )
    assert len(relative_path.parts[-1]) == _LEAF_TOKEN_WIDTH


def test_tmp_path_factory_api_remains_pytest_owned(tmp_path_factory) -> None:
    factory_path = tmp_path_factory.mktemp("factory_contract")

    assert factory_path.parent == tmp_path_factory.getbasetemp()
    assert factory_path.name == "factory_contract0"


def test_allocator_preserves_pytest_unix_socket_path_budget(tmp_path_factory) -> None:
    basetemp = tmp_path_factory.getbasetemp()
    allocator = BoundedTempPathAllocator(basetemp)
    allocated_paths = [
        allocator.allocate(
            "tests/unit/test_really_long_module_name.py::test_really_long_case_name"
        ),
        allocator.allocate("tests/unit/test_case.py::"),
        allocator.allocate("tests/unit/test_case.py::漢漢漢漢漢"),
    ]
    socket_name = "eg-12345678.sock"
    stock_longest_leaf = basetemp / ("x" * 30 + "0") / socket_name

    assert all(
        part.isascii()
        for allocated in allocated_paths
        for part in allocated.relative_to(basetemp).parts
    )
    assert all(
        len(os.fsencode(str(allocated / socket_name)))
        <= len(os.fsencode(str(stock_longest_leaf)))
        for allocated in allocated_paths
    )


def test_tmp_path_fixture_matches_pytest_retention_contract(
    pytester, monkeypatch
) -> None:
    """Compare actual path retention with pytest's stock public fixture contract."""
    _prepare_child_pytest(pytester, monkeypatch)
    custom_directory = _install_allocator_plugin(pytester, "custom_retention")
    stock_directory = pytester.path / "stock_retention"
    pytester.makefile(
        ".py",
        **{
            "custom_retention/test_retention": _RETENTION_SUITE,
            "stock_retention/test_retention": _RETENTION_SUITE,
        },
    )
    assert stock_directory.is_dir()

    for policy, expected in _RETENTION_EXPECTATIONS.items():
        custom_paths = _run_retention_suite(
            pytester,
            monkeypatch,
            child_directory=custom_directory,
            record_directory=pytester.path / f"custom-{policy}-records",
            policy=policy,
        )
        stock_paths = _run_retention_suite(
            pytester,
            monkeypatch,
            child_directory=stock_directory,
            record_directory=pytester.path / f"stock-{policy}-records",
            policy=policy,
        )

        custom_presence = {case: path.is_dir() for case, path in custom_paths.items()}
        stock_presence = {case: path.is_dir() for case, path in stock_paths.items()}
        assert custom_presence == stock_presence == expected
        _assert_bounded_fixture_paths(custom_paths)


_RETRY_CONFTEST = """
import pytest
from _pytest.runner import runtestprotocol

pytest_plugins = ("_tmp_path_allocator",)


def pytest_configure(config) -> None:
    config.addinivalue_line("markers", "retry_once: execute this item twice")


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    if item.get_closest_marker("retry_once") is None:
        return None
    runtestprotocol(item, nextitem=nextitem, log=False)
    runtestprotocol(item, nextitem=nextitem, log=True)
    return True
"""


_RETRY_SUITE = """
import json
import os
from pathlib import Path

import pytest


_attempt = 0


@pytest.mark.retry_once
def test_retried_tmp_path(tmp_path: Path) -> None:
    global _attempt
    record_directory = Path(os.environ["BOUNDED_TMP_PATH_RECORD_DIRECTORY"])
    attempt = _attempt
    _attempt += 1
    (record_directory / f"retry-{attempt}.json").write_text(
        json.dumps({"attempt": attempt, "path": str(tmp_path)}), encoding="utf-8"
    )
    assert _attempt == 2
"""


def test_tmp_path_fixture_handles_a_real_retry_lifecycle(pytester, monkeypatch) -> None:
    """Exercise the same item twice through pytest's rerun protocol lifecycle."""
    _prepare_child_pytest(pytester, monkeypatch)
    retry_directory = pytester.path / "retry"
    pytester.makefile(
        ".py",
        **{
            "retry/conftest": _RETRY_CONFTEST,
            "retry/test_retry": _RETRY_SUITE,
        },
    )
    record_directory = pytester.path / "retry-records"
    record_directory.mkdir()
    monkeypatch.setenv(_RECORD_DIRECTORY_ENV, str(record_directory))

    result = pytester.runpytest_subprocess(
        "-q",
        "-p",
        "no:asyncio",
        "-o",
        "tmp_path_retention_policy=failed",
        retry_directory,
        timeout=60,
    )

    result.assert_outcomes(passed=1)
    records = sorted(
        _read_records(record_directory), key=lambda record: record["attempt"]
    )
    attempts = [record["attempt"] for record in records]
    paths = [Path(record["path"]) for record in records]
    assert attempts == ["0", "1"]
    assert paths[0] != paths[1]
    assert _allocator_root(paths[0]) == _allocator_root(paths[1])
    assert (
        _allocator_relative_parts(paths[0])[1] == _allocator_relative_parts(paths[1])[1]
    )
    assert paths[0].is_dir()
    assert not paths[1].exists()
    _assert_bounded_fixture_paths(
        {attempt: path for attempt, path in zip(attempts, paths, strict=True)}
    )


_XDIST_SUITE = """
import json
import os
from pathlib import Path

import pytest


@pytest.mark.parametrize("case", range(8))
def test_worker_tmp_path(tmp_path: Path, worker_id: str, case: int) -> None:
    record_path = (
        Path(os.environ["BOUNDED_TMP_PATH_RECORD_DIRECTORY"])
        / f"{worker_id}-{case}.json"
    )
    record_path.write_text(
        json.dumps({"worker": worker_id, "case": str(case), "path": str(tmp_path)}),
        encoding="utf-8",
    )
"""


def test_tmp_path_fixture_keeps_xdist_workers_isolated(pytester, monkeypatch) -> None:
    """Run the public fixture under two real xdist workers and inspect both roots."""
    _prepare_child_pytest(pytester, monkeypatch)
    xdist_directory = _install_allocator_plugin(pytester, "xdist")
    pytester.makefile(".py", **{"xdist/test_workers": _XDIST_SUITE})
    record_directory = pytester.path / "xdist-records"
    record_directory.mkdir()
    monkeypatch.setenv(_RECORD_DIRECTORY_ENV, str(record_directory))

    result = pytester.runpytest_subprocess(
        "-q",
        "-p",
        "no:asyncio",
        "-n",
        "2",
        xdist_directory,
        timeout=60,
    )

    result.assert_outcomes(passed=8)
    records = _read_records(record_directory)
    assert len(records) == 8
    worker_bases: dict[str, Path] = {}
    for record in records:
        path = Path(record["path"])
        worker = record["worker"]
        assert path.is_dir()
        _assert_bounded_fixture_paths({record["case"]: path})
        root = _allocator_root(path)
        worker_bases.setdefault(worker, root.parent)
        assert worker_bases[worker] == root.parent

    assert set(worker_bases) == {"gw0", "gw1"}
    assert len(set(worker_bases.values())) == 2


_REAL_FIXTURE_MEASUREMENT_SUITE = """
import json
import os
from pathlib import Path

import pytest


@pytest.mark.parametrize("case", range(64))
def test_fixture_measurement(tmp_path: Path, case: int) -> None:
    record_path = Path(os.environ["BOUNDED_TMP_PATH_RECORD_DIRECTORY"]) / f"{case}.json"
    record_path.write_text(json.dumps({"case": str(case), "path": str(tmp_path)}), encoding="utf-8")
"""


def test_real_tmp_path_fixture_measurement(pytester, monkeypatch) -> None:
    """Measure a bounded real-fixture run without making every gate a 4k child suite."""
    _prepare_child_pytest(pytester, monkeypatch)
    measurement_directory = _install_allocator_plugin(pytester, "measurement")
    pytester.makefile(
        ".py",
        **{"measurement/test_fixture_measurement": _REAL_FIXTURE_MEASUREMENT_SUITE},
    )
    record_directory = pytester.path / "measurement-records"
    record_directory.mkdir()
    monkeypatch.setenv(_RECORD_DIRECTORY_ENV, str(record_directory))

    result = pytester.runpytest_subprocess(
        "-q",
        "-p",
        "no:asyncio",
        "-o",
        "tmp_path_retention_policy=all",
        measurement_directory,
        timeout=60,
    )

    result.assert_outcomes(passed=64)
    records = _read_records(record_directory)
    paths = {record["case"]: Path(record["path"]) for record in records}
    print(f"bounded_tmp_path_real_fixture_64={result.duration:.6f}s")
    assert len(paths) == 64
    assert all(path.is_dir() for path in paths.values())
    _assert_bounded_fixture_paths(paths)
