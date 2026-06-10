#!/usr/bin/python
"""Tests for the tiered RLM sandbox: analyzer, capability router, and backends.

CONCEPT:ORCH-1.38 — analyzer classifies a snippet's needs; the router picks the cheapest
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
from agent_utilities.rlm.sandboxes.local_backend import LocalSandbox
from agent_utilities.rlm.sandboxes.router import SandboxRouter

pytestmark = pytest.mark.concept("ORCH-1.38")


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


def _fake(name, rank, *, host, third, classes, available=True):
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
        network=(name == "local"),
        isolated=(name != "local"),
        preference_rank=rank,
    )
    return f


@pytest.fixture
def backends():
    return {
        "monty": _fake("monty", 0, host=True, third=False, classes=False),
        "wasm": _fake("wasm", 10, host=False, third=False, classes=True),
        "docker": _fake("docker", 20, host=True, third=True, classes=True),
        "local": LocalSandbox(),
    }


def _route(backends_map, code, **kw):
    router = SandboxRouter(list(backends_map.values()))
    return [b.name for b in router.select(code, **kw)]


@pytest.mark.parametrize(
    "code,expected",
    [
        ("total = sum(range(5))", ["monty", "wasm", "docker", "local"]),  # plain → all
        (
            "x = await rlm_query('p')",
            ["monty", "docker", "local"],
        ),  # helpers → host-capable
        ("import numpy\nnumpy.array([1])", ["docker", "local"]),  # 3p → docker only
        ("class P:\n    x = 1", ["wasm", "docker", "local"]),  # classes → no monty
        (
            "class P:\n    x = 1\nFINAL_VAR('a', 1)",
            ["docker", "local"],
        ),  # classes+helpers
        ("def (:\n", ["local"]),  # bad syntax → floor only
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
        "local": LocalSandbox(),
    }
    assert _route(bk, "total = 1") == ["docker", "local"]


def test_router_force_pins_backend(backends):
    assert _route(backends, "import numpy", force="docker") == ["docker"]


def test_router_force_unavailable_degrades_to_auto():
    bk = {
        "monty": _fake(
            "monty", 0, host=True, third=False, classes=False, available=False
        ),
        "docker": _fake("docker", 20, host=True, third=True, classes=True),
        "local": LocalSandbox(),
    }
    # forcing the down monty falls through to normal routing rather than dying
    assert _route(bk, "total = 1", force="monty") == ["docker", "local"]


def test_router_never_empty_and_floor_is_local(backends):
    # Even a snippet only local can satisfy keeps local at the tail.
    chain = _route(backends, "import numpy")
    assert chain[-1] == "local"


def test_router_requires_at_least_one_backend():
    with pytest.raises(ValueError):
        SandboxRouter([])


# --- LocalSandbox (verbatim _execute_local semantics) -----------------------


async def test_local_sandbox_runs_and_syncs_seeded_vars():
    sink = {}
    env = SandboxEnv(
        vars={"context": "abc", "depth": 0},
        helpers={"FINAL_VAR": lambda n, v: sink.__setitem__(n, v)},
    )
    res = await LocalSandbox().execute(
        "y = len(context)\nFINAL_VAR('answer', y)\nprint('ran', y)", env
    )
    assert res.error is None
    assert res.stdout.strip() == "ran 3"
    assert sink == {"answer": 3}
    # function-local 'y' is NOT persisted (matches legacy async-wrapped exec); seeds persist.
    assert "y" not in res.updated_vars
    assert res.updated_vars["context"] == "abc"
    # injected helpers are never synced back as state
    assert "FINAL_VAR" not in res.updated_vars


async def test_local_sandbox_captures_in_sandbox_error():
    res = await LocalSandbox().execute("raise ValueError('boom')", SandboxEnv(vars={}))
    assert res.error and "boom" in res.error
    assert "Traceback" in res.stdout  # surfaced to the model, not raised


# --- config resolution ------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({}, "auto"),
        ({"sandbox": "monty"}, "monty"),
        ({"sandbox": "docker"}, "docker"),
        ({"use_monty": True}, "monty"),
        ({"use_wasm": True}, "wasm"),
        ({"use_container": True}, "docker"),
        ({"use_monty": True, "use_container": True}, "monty"),  # monty wins
        ({"sandbox": "local", "use_monty": True}, "local"),  # explicit field wins
    ],
)
def test_config_resolved_sandbox(kwargs, expected):
    assert RLMConfig(**kwargs).resolved_sandbox() == expected


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
