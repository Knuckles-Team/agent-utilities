#!/usr/bin/python
"""Tests for the tiered RLM sandbox: analyzer, capability router, and backends.

CONCEPT:AU-ORCH.sandbox.tiered-rlm-sandbox — analyzer classifies a snippet's needs; the router picks the cheapest
capable backend and escalates on rejection; backends honor the (rejected vs reported vs
fatal) error contract. Higher tiers (monty/wasm/docker) are exercised with capability fakes
where the real dependency may be absent; the monty path runs for real when importable.
"""

from __future__ import annotations

import pytest

from agent_utilities.rlm.config import RLMConfig
from agent_utilities.rlm.sandboxes.analyzer import AstAnalyzer
from agent_utilities.rlm.sandboxes.base import (
    HELPER_NAMES,
    Sandbox,
    SandboxCapabilities,
    SandboxEnv,
    SandboxRejected,
    SandboxResult,
)
from agent_utilities.rlm.sandboxes.router import SandboxRouter
from agent_utilities.rlm.telemetry import SandboxFatalError

pytestmark = pytest.mark.concept("AU-ORCH.sandbox.tiered-rlm-sandbox")


# --- analyzer ---------------------------------------------------------------


def test_analyzer_classifies_imports_classes_async_helpers():
    a = AstAnalyzer()
    req = a.analyze(
        "import numpy\nimport json\nclass C:\n    pass\nx = await rlm_query('p')\n"
    )
    assert req.syntax_ok
    assert req.third_party_imports == {"numpy"}  # json is stdlib, excluded
    assert req.defines_classes
    assert req.uses_async
    assert req.helper_calls == {"rlm_query"}
    assert req.needs_third_party and req.needs_host_callbacks


def test_analyzer_plain_snippet_needs_nothing():
    req = AstAnalyzer().analyze("total = sum(range(10))\nprint(total)")
    assert req.syntax_ok
    assert not req.third_party_imports and not req.defines_classes
    assert not req.helper_calls


def test_analyzer_dataclass_decorator_flags_classes():
    # @dataclass expands to a ClassDef — monty can't run it, so it must be flagged.
    req = AstAnalyzer().analyze(
        "from dataclasses import dataclass\n@dataclass\nclass P:\n    x: int"
    )
    assert req.defines_classes


def test_analyzer_relative_import_not_third_party():
    req = AstAnalyzer().analyze("from . import sibling\nfrom .pkg import thing")
    assert req.third_party_imports == set()


def test_analyzer_bad_syntax():
    req = AstAnalyzer().analyze("def (:\n")
    assert req.syntax_ok is False


# --- router (capability fakes) ----------------------------------------------


def _fake(name, rank, *, host, third, classes, available=True, isolated=True):
    class _F(Sandbox):
        def is_available(self):
            return available

        async def execute(self, code, env):  # pragma: no cover - fakes are not executed
            return SandboxResult({}, "")

    f = _F()
    f.name = name
    f.capabilities = SandboxCapabilities(
        host_callbacks=host,
        third_party_libs=third,
        classes=classes,
        full_stdlib=(name != "monty"),
        network=False,
        isolated=isolated,
        preference_rank=rank,
    )
    return f


@pytest.fixture
def backends():
    return {
        "monty": _fake("monty", 0, host=True, third=False, classes=False),
        "wasm": _fake("wasm", 10, host=False, third=False, classes=True),
        "docker": _fake("docker", 20, host=True, third=True, classes=True),
    }


def _route(backends_map, code, **kw):
    router = SandboxRouter(list(backends_map.values()))
    return [b.name for b in router.select(code, **kw)]


@pytest.mark.parametrize(
    "code,expected",
    [
        ("total = sum(range(5))", ["monty", "wasm", "docker"]),
        (
            "x = await rlm_query('p')",
            ["monty", "docker"],
        ),  # helpers → host-capable
        ("import numpy\nnumpy.array([1])", ["docker"]),
        ("class P:\n    x = 1", ["wasm", "docker"]),
        (
            "class P:\n    x = 1\nFINAL_VAR('a', 1)",
            ["docker"],
        ),  # classes+helpers
        ("def (:\n", ["monty", "wasm", "docker"]),
    ],
)
def test_router_decisions(backends, code, expected):
    assert _route(backends, code) == expected


def test_router_skips_unavailable_backend():
    bk = {
        "monty": _fake(
            "monty", 0, host=True, third=False, classes=False, available=False
        ),
        "docker": _fake("docker", 20, host=True, third=True, classes=True),
    }
    assert _route(bk, "total = 1") == ["docker"]


def test_router_force_pins_backend(backends):
    assert _route(backends, "import numpy", force="docker") == ["docker"]


def test_router_force_unavailable_fails_closed():
    bk = {
        "monty": _fake(
            "monty", 0, host=True, third=False, classes=False, available=False
        ),
        "docker": _fake("docker", 20, host=True, third=True, classes=True),
    }
    with pytest.raises(SandboxFatalError, match="unavailable"):
        _route(bk, "total = 1", force="monty")


def test_router_excludes_unisolated_backend(backends):
    unisolated = _fake(
        "unconfined", 1, host=True, third=True, classes=True, isolated=False
    )
    router = SandboxRouter([unisolated, *backends.values()])
    assert [b.name for b in router.select("import numpy")] == ["docker"]


def test_router_without_available_isolation_fails_closed():
    with pytest.raises(SandboxFatalError, match="no approved isolated"):
        SandboxRouter([]).select("x = 1")


# --- config resolution ------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({}, "auto"),
        ({"sandbox": "monty"}, "monty"),
        ({"sandbox": "docker"}, "docker"),
    ],
)
def test_config_sandbox(kwargs, expected):
    assert RLMConfig(**kwargs).sandbox == expected


# --- MontySandbox (real, when importable) -----------------------------------


def _monty_or_skip():
    pytest.importorskip("pydantic_monty")
    from agent_utilities.rlm.sandboxes.monty_backend import MontySandbox

    return MontySandbox()


async def test_monty_runs_async_helpers_and_reports_runtime_error():
    sb = _monty_or_skip()
    calls = {"n": 0}
    sink = {}

    async def rlm_query(p, c=""):
        calls["n"] += 1
        return f"s[{len(c)}]"

    env = SandboxEnv(
        vars={"context": "Z" * 250, "depth": 0},
        helpers={
            "rlm_query": rlm_query,
            "FINAL_VAR": lambda n, v: sink.__setitem__(n, v),
        },
    )
    code = (
        "parts = []\n"
        "i = 0\n"
        "while i < len(context):\n"
        "    parts.append(await rlm_query('s', context[i:i+100]))\n"
        "    i += 100\n"
        "FINAL_VAR('answer', {'n': len(parts), 'first': parts[0]})\n"
    )
    res = await sb.execute(code, env)
    assert res.error is None
    assert sink["answer"] == {
        "n": 3,
        "first": "s[100]",
    }  # awaited → real value, not a future
    assert calls["n"] == 3

    # genuine runtime error is reported (not raised, not escalated)
    res2 = await sb.execute("x = 1 / 0", SandboxEnv(vars={}))
    assert res2.error and "division" in res2.error.lower()


async def test_monty_rejects_class_before_any_helper_fires():
    sb = _monty_or_skip()
    calls = {"n": 0}

    async def rlm_query(p, c=""):
        calls["n"] += 1
        return "x"

    env = SandboxEnv(vars={"context": "x"}, helpers={"rlm_query": rlm_query})
    with pytest.raises(SandboxRejected):
        await sb.execute("class C:\n    pass\nx = await rlm_query('a')", env)
    assert calls["n"] == 0  # construction-time rejection: zero side effects


def test_helper_names_cover_repl_namespace():
    # Guard against the HELPER_NAMES set drifting from the REPL's injected helpers.
    assert {"rlm_query", "graph_query", "FINAL_VAR", "sub_agent_call"} <= HELPER_NAMES
    assert len(HELPER_NAMES) == 9


# --- DockerSandbox (real, when a docker/podman daemon is reachable) ----------


def _docker_or_skip(**kw):
    from agent_utilities.rlm.sandboxes.docker_backend import DockerSandbox

    sb = DockerSandbox(**kw)
    if not sb.is_available():
        pytest.skip("no docker/podman daemon reachable")
    return sb


def test_docker_configuration_rejects_argv_injection_and_unbounded_resources():
    from agent_utilities.rlm.sandboxes.docker_backend import DockerSandbox

    with pytest.raises(ValueError, match="image"):
        DockerSandbox(image="--privileged")
    with pytest.raises(ValueError, match="memory"):
        DockerSandbox(memory="1t")
    with pytest.raises(ValueError, match="process"):
        DockerSandbox(pids_limit=1)


def test_production_docker_image_requires_digest(monkeypatch):
    from agent_utilities.rlm.sandboxes.docker_backend import DockerSandbox

    monkeypatch.setenv("APP_PROFILE", "production")
    with pytest.raises(ValueError, match="immutable"):
        DockerSandbox(image="python:3.12-slim")
    sandbox = DockerSandbox(image="registry.example.invalid/sandbox@sha256:" + "a" * 64)
    assert "@sha256:" in sandbox.image


async def test_docker_argv_enforces_read_only_confinement(monkeypatch, tmp_path):
    import agent_utilities.rlm.sandboxes.docker_backend as docker_backend

    captured = []

    class _Process:
        async def wait(self):
            return 0

    async def create(*argv, **_kwargs):
        captured.extend(argv)
        return _Process()

    monkeypatch.setattr(
        docker_backend.asyncio,
        "create_subprocess_exec",
        create,
    )
    monkeypatch.setattr(
        docker_backend._bridge,
        "read_result",
        lambda _path: ("", None, True),
    )
    sandbox = docker_backend.DockerSandbox()
    await sandbox._run_container("docker", tmp_path, "safeid")  # noqa: SLF001
    assert {"--network", "--read-only", "--ipc", "--tmpfs", "--ulimit"} <= set(captured)
    assert "none" in captured


@pytest.mark.integration
@pytest.mark.slow
async def test_docker_runs_class_and_serves_host_bridge_under_no_network():
    # The escalation tier: code monty can't run (a class) reaches the host helpers over the
    # UDS bridge even though the container has --network none.
    # Spins a REAL Docker container with a 90s sandbox budget; on a saturated box the
    # container startup + execution can exceed the fast suite's 60s per-test budget, so
    # (like ``test_docker_timeout_is_fatal``) this real-resource test is marked ``slow``
    # — excluded from the ``-m "not slow"`` pre-commit gate, exercised in the full run.
    sb = _docker_or_skip(timeout_secs=90)
    calls = {"n": 0}
    sink = {}

    async def rlm_query(p, c=""):
        calls["n"] += 1
        return f"host[{len(c)}]"

    env = SandboxEnv(
        vars={"context": "Q" * 150, "depth": 0},
        helpers={
            "rlm_query": rlm_query,
            "FINAL_VAR": lambda n, v: sink.__setitem__(n, v),
        },
    )
    code = (
        "class Acc:\n"
        "    def __init__(self): self.p = []\n"
        "acc = Acc()\n"
        "i = 0\n"
        "while i < len(context):\n"
        "    acc.p.append(await rlm_query('s', context[i:i+50]))\n"
        "    i += 50\n"
        "FINAL_VAR('answer', {'n': len(acc.p), 'first': acc.p[0]})\n"
    )
    res = await sb.execute(code, env)
    assert res.error is None, res.error
    assert sink["answer"] == {"n": 3, "first": "host[50]"}
    assert calls["n"] == 3


@pytest.mark.integration
@pytest.mark.slow
async def test_docker_network_is_isolated():
    # Real Docker container (--network none) with a 60s sandbox budget; same
    # saturation rationale as the sibling real-Docker tests — marked ``slow`` so
    # it never flakes the fast ``-m "not slow"`` pre-commit gate under load.
    sb = _docker_or_skip(timeout_secs=60)
    sink = {}
    code = (
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 53), timeout=3)\n"
        "    FINAL_VAR('net', 'REACHED')\n"
        "except Exception as e:\n"
        "    FINAL_VAR('net', 'blocked')\n"
    )
    env = SandboxEnv(
        vars={}, helpers={"FINAL_VAR": lambda n, v: sink.__setitem__(n, v)}
    )
    await sb.execute(code, env)
    assert sink.get("net") == "blocked"  # --network none


@pytest.mark.integration
@pytest.mark.slow
async def test_docker_timeout_is_fatal():
    # Spins a real Docker container and WAITS for an app-level timeout to fire on
    # an infinite loop. On a saturated box, container startup + the spin window
    # can exceed the fast suite's per-test budget, so this real-resource timing
    # test is marked ``slow`` (excluded from the ``-m "not slow"`` pre-commit
    # gate, exercised in the full/nightly run) rather than flaking under load.
    from agent_utilities.rlm.telemetry import SandboxFatalError

    sb = _docker_or_skip(timeout_secs=6)
    with pytest.raises(SandboxFatalError):
        await sb.execute("while True:\n    pass", SandboxEnv(vars={}))


# --- WasmSandbox (real, when wasmtime + a python.wasm payload are present) ----


def _wasm_or_skip(**kw):
    pytest.importorskip("wasmtime")
    from agent_utilities.rlm.sandboxes.wasm_backend import WasmSandbox

    sb = WasmSandbox(**kw)
    if not sb.is_available():
        pytest.skip(
            "no python.wasm payload (set RLM_WASM_PYTHON or run provision_rlm_wasm.py)"
        )
    return sb


@pytest.mark.integration
async def test_wasm_runs_isolated_full_stdlib_compute():
    # The isolated full-CPython tier: a class + stdlib, on seeded vars, communicating via stdout.
    sb = _wasm_or_skip(timeout_secs=20)
    code = (
        "import json, math\n"
        "class Stats:\n"
        "    def __init__(self, d): self.d = d\n"
        "    def summary(self): return {'n': len(self.d), 'sqrt': round(math.sqrt(len(self.d)), 1)}\n"
        "print(json.dumps(Stats(context).summary()))\n"
    )
    res = await sb.execute(
        code, SandboxEnv(vars={"context": list(range(144)), "depth": 0})
    )
    assert res.error is None, res.error
    import json as _json

    assert _json.loads(res.stdout.strip()) == {"n": 144, "sqrt": 12.0}


@pytest.mark.integration
async def test_wasm_reports_in_sandbox_error():
    sb = _wasm_or_skip(timeout_secs=20)
    res = await sb.execute("x = 1 / 0", SandboxEnv(vars={}))
    assert res.error and "division" in res.error.lower()


@pytest.mark.integration
async def test_wasm_timeout_is_rejected_not_fatal():
    # A WASI run has no side effects, so an overrun is a SAFE escalation (SandboxRejected),
    # unlike Docker where a dead container is fatal.
    sb = _wasm_or_skip(timeout_secs=3)
    with pytest.raises(SandboxRejected):
        await sb.execute("while True:\n    pass", SandboxEnv(vars={}))
