"""Wiring tests for the non-blocking / chat-budget execution work.

Covers the P0/P1 items of ``docs/architecture/non-blocking-execution.md`` at the
wiring level (engine/vLLM may be offline — these assert the contract, not a live run):

* CONCEPT:ORCH-1.62 — the ``chat`` execution profile yields bounded node timeouts and is
  threaded from the messaging reply path into ``execute_agent``/``run_agent``/the config;
  the ``task`` profile keeps the long defaults; a backend timeout does NOT double-call.
* CONCEPT:ORCH-1.63 — the widened fast-path classifier catches normal simple questions and
  escalates tool/plan-shaped turns.
* CONCEPT:ORCH-1.64 — the built graph topology is cached across two turns of one config.
* CONCEPT:ORCH-1.65 — the hot-path KG resolution runs off the event loop.
* CONCEPT:KG-2.131 — the per-session memento cache primes from memory, refreshed in the bg.
"""

from __future__ import annotations

from typing import Any

import pytest

# ─────────────────────────── ORCH-1.62 — chat profile ───────────────────────────


def test_chat_profile_bounds_node_timeouts() -> None:
    from agent_utilities.orchestration.execution_profile import (
        CHAT_NODE_TIMEOUT_S,
        resolve_execution_profile,
    )

    chat = resolve_execution_profile("chat")
    assert chat.name == "chat"
    assert (
        chat.router_timeout is not None and chat.router_timeout <= CHAT_NODE_TIMEOUT_S
    )
    assert (
        chat.verifier_timeout is not None
        and chat.verifier_timeout <= CHAT_NODE_TIMEOUT_S
    )
    assert chat.fast_path_eligible is True
    assert chat.cheap_fallback is True


def test_task_profile_keeps_long_defaults() -> None:
    from agent_utilities.orchestration.execution_profile import (
        resolve_execution_profile,
    )

    for value in (None, "task", "unknown"):
        task = resolve_execution_profile(value)
        assert task.name == "task"
        # None timeouts → the long defaults are used by the builder/config.
        assert task.router_timeout is None
        assert task.verifier_timeout is None
        assert task.cheap_fallback is False


def test_chat_profile_stays_under_reply_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """The chat node budget must be far below the messaging reply timeout so a turn resolves
    INSIDE the graph instead of being killed and retried via plain-chat (CONCEPT:ORCH-1.62)."""
    from agent_utilities.orchestration.execution_profile import (
        resolve_execution_profile,
    )

    monkeypatch.setenv("MESSAGING_REPLY_TIMEOUT", "45")
    chat = resolve_execution_profile("chat")
    assert chat.router_timeout is not None
    assert chat.router_timeout < 45.0

    # A small reply budget shrinks the node budget so a single round still fits.
    monkeypatch.setenv("MESSAGING_REPLY_TIMEOUT", "8")
    chat_small = resolve_execution_profile("chat")
    assert chat_small.router_timeout is not None
    assert chat_small.router_timeout <= 8.0


def test_build_execution_config_applies_chat_profile() -> None:
    """The chat profile's bounded node timeouts reach the graph config (CONCEPT:ORCH-1.62)."""
    from agent_utilities.core.config import (
        DEFAULT_GRAPH_ROUTER_TIMEOUT,
        DEFAULT_GRAPH_VERIFIER_TIMEOUT,
    )
    from agent_utilities.orchestration.agent_runner import _build_execution_config

    meta: dict[str, Any] = {"type": "unknown", "capabilities": [], "tools": []}

    chat_cfg = _build_execution_config(
        None, "messaging-assistant", meta, execution_profile="chat", recent_mementos=[]
    )
    assert chat_cfg["execution_profile"] == "chat"
    assert chat_cfg["router_timeout"] < DEFAULT_GRAPH_ROUTER_TIMEOUT
    assert chat_cfg["verifier_timeout"] < DEFAULT_GRAPH_VERIFIER_TIMEOUT

    task_cfg = _build_execution_config(
        None, "some-agent", meta, execution_profile="task", recent_mementos=[]
    )
    assert task_cfg["execution_profile"] == "task"
    assert task_cfg["router_timeout"] == DEFAULT_GRAPH_ROUTER_TIMEOUT
    assert task_cfg["verifier_timeout"] == DEFAULT_GRAPH_VERIFIER_TIMEOUT


@pytest.mark.asyncio
async def test_execute_agent_threads_profile_to_run_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``Orchestrator.execute_agent(execution_profile=...)`` reaches ``run_agent`` unchanged
    (Universal capability — built once, threaded, not re-implemented per surface)."""
    from agent_utilities.orchestration import manager as mgr

    captured: dict[str, Any] = {}

    async def _fake_run_agent(**kwargs: Any) -> str:
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(mgr, "run_agent", _fake_run_agent)
    # Skip the prompt-injection scan (it needs no engine, but keep the test hermetic).
    monkeypatch.setattr(mgr.Orchestrator, "_scan_task", lambda self, task: None)

    orch = mgr.Orchestrator(engine=object())
    await orch.execute_agent(
        agent_name="messaging-assistant", task="hi", execution_profile="chat"
    )
    assert captured["execution_profile"] == "chat"


@pytest.mark.asyncio
async def test_timeout_does_not_double_call_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The messaging reply path passes ``chat`` and a backend timeout does NOT trigger a
    second full LLM call (CONCEPT:ORCH-1.62 — removes the measured double-LLM tax)."""
    import asyncio

    from agent_utilities.messaging import router as router_mod
    from agent_utilities.orchestration import manager as mgr

    seen_profile: dict[str, Any] = {}

    class _SlowOrch:
        def __init__(self, _engine: Any) -> None:
            pass

        async def execute_agent(self, **kwargs: Any) -> str:
            seen_profile.update(kwargs)
            await asyncio.sleep(10)
            return "never"

    plain_calls: list[str] = []

    async def _spy_plain(content: str, **_: Any) -> str:
        plain_calls.append(content)
        return "[local] should-not-happen"

    monkeypatch.setattr(mgr, "Orchestrator", _SlowOrch)
    monkeypatch.setattr(router_mod, "_plain_chat_reply", _spy_plain)
    monkeypatch.setenv("MESSAGING_REPLY_TIMEOUT", "0.2")

    reply = await router_mod._graph_agent_reply(
        object(), "hello there", session="messaging:telegram:1"
    )
    assert seen_profile.get("execution_profile") == "chat"
    assert plain_calls == []  # no second LLM call on a backend timeout
    assert "slowly" in reply.lower() or "try again" in reply.lower()


def test_is_backend_timeout_classifier() -> None:
    from agent_utilities.messaging.router import _is_backend_timeout

    assert _is_backend_timeout("Agent execution failed: router round timed out")
    assert _is_backend_timeout("Agent execution failed: CancelledError")
    assert not _is_backend_timeout("Agent execution failed: delegation exploded")


# ─────────────────────────── ORCH-1.63 — fast path ───────────────────────────


def test_fast_path_catches_normal_simple_questions() -> None:
    from agent_utilities.graph.routing.strategies.fast_path import is_trivial_query

    for q in (
        "hello there",
        "thanks!",
        "what's the status of the project?",
        "can you summarise this?",
        "who are you",
        "what is the capital of France",
    ):
        assert is_trivial_query(q), f"expected fast path for {q!r}"


def test_fast_path_escalates_tool_and_plan_turns() -> None:
    from agent_utilities.graph.routing.strategies.fast_path import (
        is_trivial_query,
        needs_full_orchestration,
    )

    for q in (
        "deploy the freshrss stack to r710",
        "restart the graph-os service",
        "/skill code-enhancer",
        "search the KG for recent papers and then ingest them",
        "fix the failing test in agent_runner and run the suite",
    ):
        assert needs_full_orchestration(q), f"expected escalation for {q!r}"
        assert not is_trivial_query(q), f"expected NOT fast-path for {q!r}"


# ─────────────────────────── ORCH-1.64 — graph cache ───────────────────────────


def test_built_graph_is_cached_across_two_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The structural topology is built ONCE for a routing-config and reused (CONCEPT:ORCH-1.64)."""
    from agent_utilities.graph import builder

    builder._GRAPH_CACHE.clear()
    # Validation mode skips the engine/registry init so the build is pure topology assembly.
    monkeypatch.setattr(builder, "DEFAULT_VALIDATION_MODE", True)
    # Avoid a fleet/agent discovery round-trip — the topology cache is what we exercise.
    monkeypatch.setattr(builder, "discover_agents", lambda *a, **k: {})

    build_config_calls = {"n": 0}
    real_build_config = builder._build_graph_config

    def _counting_config(**kwargs: Any) -> dict[str, Any]:
        build_config_calls["n"] += 1
        return real_build_config(**kwargs)

    monkeypatch.setattr(builder, "_build_graph_config", _counting_config)

    tag_prompts = {"messaging-assistant": "Specialized agent"}

    g1, _c1 = builder.create_graph_agent(
        tag_prompts=tag_prompts, mcp_config="", mcp_url=""
    )
    g2, _c2 = builder.create_graph_agent(
        tag_prompts=tag_prompts, mcp_config="", mcp_url=""
    )

    # Both turns build a config (config is per-run), but the GRAPH OBJECT is the same instance
    # — the structural topology was reused from the cache, not rebuilt.
    assert g1 is g2, "graph topology must be reused from the cache on the second turn"
    assert build_config_calls["n"] == 2  # config built per-turn (cheap), topology once


def test_graph_cache_key_is_structural_only() -> None:
    from agent_utilities.graph.builder import _graph_cache_key

    base = dict(
        router_model="r",
        agent_model="m",
        routing_strategy="hybrid",
        sub_agents=None,
        custom_nodes=None,
    )
    k1 = _graph_cache_key(tag_prompts={"a": "x", "b": "y"}, **base)
    k2 = _graph_cache_key(tag_prompts={"b": "y", "a": "x"}, **base)  # order-independent
    k3 = _graph_cache_key(tag_prompts={"a": "x"}, **base)  # different tag set
    assert k1 == k2
    assert k1 != k3


# ─────────────────────────── ORCH-1.65 — off-loop KG ───────────────────────────


@pytest.mark.asyncio
async def test_resolve_agent_runs_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``run_agent`` must resolve the agent via ``to_thread`` so the sync KG round-trips never
    stall the loop (CONCEPT:ORCH-1.65)."""
    import asyncio
    import threading

    from agent_utilities.orchestration import agent_runner

    main_thread = threading.get_ident()
    resolve_threads: list[int] = []

    def _resolver(engine: Any, name: str) -> dict[str, Any]:
        resolve_threads.append(threading.get_ident())
        return {"type": "unknown", "capabilities": [], "tools": []}

    monkeypatch.setattr(agent_runner, "_resolve_agent_from_kg", _resolver)
    # Short-circuit everything after resolution so we only exercise the off-loop hop.
    monkeypatch.setattr(agent_runner, "_get_or_create_engine", lambda: object())

    async def _fake_prime(*_a: Any, **_k: Any) -> list[str]:
        return []

    monkeypatch.setattr(agent_runner, "_prime_recent_mementos", _fake_prime)

    async def _fake_graph(**_kwargs: Any) -> dict[str, Any]:
        return {"results": {"output": "done"}}

    monkeypatch.setattr(agent_runner, "_execute_graph", _fake_graph)
    monkeypatch.setattr(agent_runner, "_record_execution_trace", lambda *a, **k: None)
    monkeypatch.setattr(agent_runner, "_write_step_credit", lambda *a, **k: None)

    out = await agent_runner.run_agent(
        agent_name="messaging-assistant", task="hello there", engine=object()
    )
    assert out == "done"
    assert resolve_threads, "resolver was not called"
    assert resolve_threads[0] != main_thread, (
        "_resolve_agent_from_kg ran on the event loop"
    )

    # Sanity: to_thread really hands off.
    assert await asyncio.to_thread(threading.get_ident) != main_thread


# ─────────────────────────── KG-2.131 — memento cache ───────────────────────────


def test_session_memento_cache_lru_and_copy() -> None:
    from agent_utilities.knowledge_graph.memory.session_memento_cache import (
        SessionMementoCache,
    )

    cache = SessionMementoCache(max_sessions=2)
    assert cache.get("s") is None
    cache.put("s", ["a", "b"])
    got = cache.get("s")
    assert got == ["a", "b"]
    got.append("mutation")
    assert cache.get("s") == ["a", "b"], "cache must return a copy, not the stored list"

    # LRU eviction.
    cache.put("s2", ["c"])
    cache.get("s")  # touch s → s2 is now LRU
    cache.put("s3", ["d"])
    assert cache.get("s2") is None
    assert cache.get("s") == ["a", "b"]
    assert cache.get("s3") == ["d"]


@pytest.mark.asyncio
async def test_prime_recent_mementos_reads_cache_without_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cache hit must NOT touch the backend (zero I/O on the hot path, CONCEPT:KG-2.131)."""
    from agent_utilities.knowledge_graph.memory import session_memento_cache as smc
    from agent_utilities.orchestration import agent_runner

    smc.SessionMementoCache.instance().clear()
    smc.SessionMementoCache.instance().put("sess", ["primed memento"])

    def _boom(*_a: Any, **_k: Any) -> list[str]:
        raise AssertionError("get_recent_mementos must not be called on a cache hit")

    import agent_utilities.knowledge_graph.memory.memento_compressor as mc

    monkeypatch.setattr(mc, "get_recent_mementos", _boom)

    out = await agent_runner._prime_recent_mementos(object(), "sess")
    assert out == ["primed memento"]


@pytest.mark.asyncio
async def test_prime_recent_mementos_cold_miss_fetches_off_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading

    import agent_utilities.knowledge_graph.memory.memento_compressor as mc
    from agent_utilities.knowledge_graph.memory import session_memento_cache as smc
    from agent_utilities.orchestration import agent_runner

    smc.SessionMementoCache.instance().clear()
    main_thread = threading.get_ident()
    fetch_threads: list[int] = []

    def _fetch(engine: Any, source: str, limit: int) -> list[str]:
        fetch_threads.append(threading.get_ident())
        return ["fetched"]

    monkeypatch.setattr(mc, "get_recent_mementos", _fetch)

    out = await agent_runner._prime_recent_mementos(object(), "cold-sess")
    assert out == ["fetched"]
    assert fetch_threads and fetch_threads[0] != main_thread, (
        "cold fetch ran on the loop"
    )
    # Now cached for the next turn.
    assert smc.SessionMementoCache.instance().get("cold-sess") == ["fetched"]


def test_refresh_session_memento_cache_populates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The background ``_persist_and_enrich`` pass refreshes the cache so turn N+1 reads
    turn N's memento from memory (CONCEPT:KG-2.131 / ECO-4.74)."""
    import agent_utilities.knowledge_graph.memory.memento_compressor as mc
    from agent_utilities.knowledge_graph.memory.session_memento_cache import (
        SessionMementoCache,
        refresh_session_memento_cache,
    )

    SessionMementoCache.instance().clear()
    monkeypatch.setattr(mc, "get_recent_mementos", lambda *a, **k: ["m-from-bg"])

    out = refresh_session_memento_cache(object(), "bg-sess")
    assert out == ["m-from-bg"]
    assert SessionMementoCache.instance().get("bg-sess") == ["m-from-bg"]


# ───────────────── ORCH-1.67/1.68 — dynamic per-job execution shape ─────────────────


def test_plan_shape_trivial_turn_is_lean() -> None:
    """A trivial conversational/Q&A turn gets the lean shape: direct-completion on a local
    model with NO KG agent resolution, NO usage-guard LLM round, NO discovery/verifier, and
    reasoning OFF (CONCEPT:ORCH-1.67/1.68)."""
    from agent_utilities.orchestration.execution_profile import plan_execution_shape

    for q in ("what is 2 plus 2?", "hello", "what can you do?"):
        s = plan_execution_shape(q, profile_hint="chat")
        assert s.direct_complete is True, q
        assert s.skip_usage_guard is True, q
        assert s.resolve_agent is False, q
        assert s.run_discovery is False, q
        assert s.run_verifier is False, q
        assert s.enable_reasoning is False, q
        assert s.origin == "heuristic"
        # The chat hint still bounds the node budget.
        assert s.router_timeout is not None


def test_plan_shape_real_task_keeps_full_graph() -> None:
    """A tool/plan-shaped turn keeps the full graph: resolve the specialist, run the
    usage-guard / discovery / verifier, reasoning ON (CONCEPT:ORCH-1.67/1.68)."""
    from agent_utilities.orchestration.execution_profile import plan_execution_shape

    for q in (
        "deploy the service and then restart it",
        "analyze this repo and create a migration plan",
    ):
        s = plan_execution_shape(q, profile_hint="chat")
        assert s.direct_complete is False, q
        assert s.resolve_agent is True, q
        assert s.skip_usage_guard is False, q
        assert s.run_discovery is True, q
        assert s.run_verifier is True, q
        assert s.enable_reasoning is True, q


def test_plan_shape_defaults_preserve_full_behaviour() -> None:
    """An ExecutionProfile built without the planner keeps the prior full-graph behaviour so
    direct callers that plan no shape are unchanged (CONCEPT:ORCH-1.67)."""
    from agent_utilities.orchestration.execution_profile import (
        resolve_execution_profile,
    )

    for value in (None, "task", "chat"):
        p = resolve_execution_profile(value)
        assert p.direct_complete is False
        assert p.skip_usage_guard is False
        assert p.resolve_agent is True
        assert p.run_discovery is True
        assert p.run_verifier is True
        assert p.enable_reasoning is True
        assert p.origin == "preset"


def test_graph_deps_carries_execution_shape() -> None:
    """The shape threads onto GraphDeps so graph nodes can read it (CONCEPT:ORCH-1.68)."""
    from agent_utilities.graph.state import GraphDeps
    from agent_utilities.orchestration.execution_profile import plan_execution_shape

    shape = plan_execution_shape("what is 2 plus 2?", profile_hint="chat")
    deps = GraphDeps(
        tag_prompts={}, tag_env_vars={}, mcp_toolsets=[], execution_shape=shape
    )
    assert deps.execution_shape is shape
    assert deps.execution_shape.direct_complete is True
    # The default (no shape planned) stays None so direct callers keep full-graph behaviour.
    assert (
        GraphDeps(tag_prompts={}, tag_env_vars={}, mcp_toolsets=[]).execution_shape
        is None
    )


# ───────── ORCH-1.68 — memory_selection doc scan is pruned/capped/off-loop ─────────


def test_memory_selection_doc_scan_pruned_and_capped(tmp_path) -> None:
    """The workspace doc inventory must prune vendor/build trees and cap its output, not
    rglob the whole 234-repo tree + read_text every file on the event loop (CONCEPT:ORCH-1.68).
    """
    from agent_utilities.graph.hierarchical_planner import (
        _DOC_SCAN_CAP,
        _scan_workspace_docs,
    )

    (tmp_path / "README.md").write_text("---\ndescription: root readme\n---\n")
    # A vendor tree that MUST be pruned (never descended into).
    nm = tmp_path / "node_modules" / "pkg"
    nm.mkdir(parents=True)
    (nm / "IGNORED.md").write_text("must not be scanned")
    # More docs than the cap, to prove the scan stops early.
    for i in range(_DOC_SCAN_CAP + 25):
        (tmp_path / f"doc{i}.md").write_text("hi")

    out = _scan_workspace_docs(str(tmp_path))
    assert len(out) <= _DOC_SCAN_CAP, "doc scan exceeded its cap"
    assert not any("IGNORED.md" in line for line in out), "node_modules was not pruned"
