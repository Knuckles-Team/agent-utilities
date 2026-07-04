"""Wiring tests for the non-blocking / chat-budget execution work.

Covers the P0/P1 items of ``docs/architecture/non-blocking-execution.md`` at the
wiring level (engine/vLLM may be offline — these assert the contract, not a live run):

* CONCEPT:AU-ORCH.execution.chat-profile-timeouts — the ``chat`` execution profile yields bounded node timeouts and is
  threaded from the messaging reply path into ``execute_agent``/``run_agent``/the config;
  the ``task`` profile keeps the long defaults; a backend timeout does NOT double-call.
* CONCEPT:AU-ORCH.routing.original-rule-was-far — the widened fast-path classifier catches normal simple questions and
  escalates tool/plan-shaped turns.
* CONCEPT:AU-ORCH.routing.structural-build-reuse — the built graph topology is cached across two turns of one config.
* CONCEPT:AU-ORCH.routing.offload-sync-roundtrip — the hot-path KG resolution runs off the event loop.
* CONCEPT:AU-KG.memory.refresh-per-session-memento — the per-session memento cache primes from memory, refreshed in the bg.
"""

from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _reset_shape_state() -> Any:
    """Each test starts from a clean shape planner. The recipe cache and the learned shape
    policy are process-global, so a test that runs a turn (recording an outcome via
    ``record_shape_outcome``) must not bias another test's classification
    (CONCEPT:AU-ORCH.execution.planner-failure-feedback/1.72)."""
    from agent_utilities.orchestration.execution_profile import (
        reset_recipe_cache,
        reset_shape_policy,
    )

    reset_recipe_cache()
    reset_shape_policy()
    yield
    reset_recipe_cache()
    reset_shape_policy()


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


def test_chat_profile_stays_under_reply_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """The chat node budget must be far below the messaging reply timeout so a turn resolves
    INSIDE the graph instead of being killed and retried via plain-chat (CONCEPT:AU-ORCH.execution.chat-profile-timeouts)."""
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
    """The chat profile's bounded node timeouts reach the graph config (CONCEPT:AU-ORCH.execution.chat-profile-timeouts)."""
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


def test_build_execution_config_injects_code_context_prime() -> None:
    """The task-start code_context prime reaches the run's tag_prompts (CONCEPT:AU-KG.retrieval.synthesized-cited-answer)."""
    from agent_utilities.orchestration.agent_runner import _build_execution_config

    meta: dict[str, Any] = {"type": "unknown", "capabilities": [], "tools": []}
    cfg = _build_execution_config(
        None,
        "some-agent",
        meta,
        execution_profile="task",
        recent_mementos=[],
        code_context_prime="run_agent is defined at engine.py:1378.",
    )
    assert "code_context" in cfg["tag_prompts"]
    assert "engine.py:1378" in cfg["tag_prompts"]["code_context"]
    assert "code_context" in cfg["valid_domains"]


@pytest.mark.asyncio
async def test_prime_code_context_skips_chat_profile() -> None:
    """The prime is skipped on the latency-sensitive chat profile (CONCEPT:AU-KG.retrieval.synthesized-cited-answer)."""
    from agent_utilities.orchestration.agent_runner import _prime_code_context

    out = await _prime_code_context(
        object(), "how does the messaging reply path work", execution_profile="chat"
    )
    assert out is None


@pytest.mark.asyncio
async def test_prime_code_context_synthesizes_for_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On a task run the prime returns the synthesized answer + citations + cap id."""
    from agent_utilities.orchestration import agent_runner

    fake = {
        "anchors": [{"symbol": "run_agent"}],
        "answer": "`run_agent` is the entry point.",
        "citations": [{"symbol": "run_agent", "file": "/x/engine.py", "line": 1378}],
        "capability_id": "code_context:how:run_agent",
    }
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.retrieval.code_context.build_code_context",
        lambda *a, **k: fake,
    )
    out = await agent_runner._prime_code_context(
        object(), "how does run_agent work", execution_profile="task"
    )
    assert out is not None
    assert "entry point" in out
    assert "/x/engine.py:1378" in out
    assert "code_context:how:run_agent" in out


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
    second full LLM call (CONCEPT:AU-ORCH.execution.chat-profile-timeouts — removes the measured double-LLM tax)."""
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


def test_fast_path_escalates_structural_turns() -> None:
    """Structural escalation only (CONCEPT:EG-ORCH.routing.lexical-capability-escalation/ORCH-1.73): slash-command, multi-clause, or
    over-length. Capability/action turns are NOT escalated here anymore — the engine lexical
    gate handles them against the live KG (see test_cascade_lexical_gate_hit_escalates)."""
    from agent_utilities.graph.routing.strategies.fast_path import (
        is_trivial_query,
        needs_full_orchestration,
    )

    for q in (
        "/skill code-enhancer",  # slash command
        "search the KG for recent papers and then ingest them",  # multi-clause ("and then")
        "please " + "really " * 45 + "do it",  # over-length
    ):
        assert needs_full_orchestration(q), f"expected structural escalation for {q!r}"
        assert not is_trivial_query(q), f"expected NOT fast-path for {q!r}"

    # A bare action/capability turn no longer trips the STRUCTURAL gate — it is structurally
    # trivial and escalates (if at all) via the KG lexical gate, not this module.
    for q in (
        "deploy the freshrss stack to r710",
        "restart the graph-os service",
    ):
        assert not needs_full_orchestration(q), (
            f"expected NO structural escalation for {q!r}"
        )


# ─────────────────────────── ORCH-1.64 — graph cache ───────────────────────────


def test_built_graph_is_cached_across_two_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The structural topology is built ONCE for a routing-config and reused (CONCEPT:AU-ORCH.routing.structural-build-reuse)."""
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
        name="A",
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
    # The graph ``name`` is stamped onto the returned graph, so two same-topology
    # graphs with different names are distinct objects and MUST NOT share a cache
    # entry (CONCEPT:AU-ORCH.routing.structural-build-reuse — otherwise a cached agent leaks the prior name).
    k_other_name = _graph_cache_key(
        tag_prompts={"a": "x", "b": "y"}, **{**base, "name": "B"}
    )
    assert k1 != k_other_name


# ─────────────────────────── ORCH-1.65 — off-loop KG ───────────────────────────


@pytest.mark.asyncio
async def test_resolve_agent_runs_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``run_agent`` must resolve the agent via ``to_thread`` so the sync KG round-trips never
    stall the loop (CONCEPT:AU-ORCH.routing.offload-sync-roundtrip). Uses a non-trivial (specialist-targeting) task because
    CONCEPT:AU-ORCH.execution.direct-completion-shape skips KG resolution entirely for a trivial/direct-completion turn, so the
    off-loop hop is only exercised when the shape sets ``resolve_agent=True``. Targets a named
    specialist (NOT the universal ``messaging-assistant``, which CONCEPT:AU-ORCH.execution.passthrough-identity exempts from
    resolution as a pass-through identity)."""
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
        agent_name="portainer-agent",
        task="deploy the service and restart the database now",
        engine=object(),
    )
    assert out == "done"
    assert resolve_threads, "resolver was not called"
    assert resolve_threads[0] != main_thread, (
        "_resolve_agent_from_kg ran on the event loop"
    )

    # Sanity: to_thread really hands off.
    assert await asyncio.to_thread(threading.get_ident) != main_thread


async def test_passthrough_agent_skips_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CONCEPT:AU-ORCH.execution.passthrough-identity — the universal ``messaging-assistant`` is a pass-through identity:
    even on a non-trivial (resolve_agent=True) turn it must NOT hit ``_resolve_agent_from_kg``
    (that ~21 s semantic search both wastes time and mis-binds it to ``prepare_messages``)."""
    from agent_utilities.orchestration import agent_runner

    called: list[str] = []

    def _resolver(engine: Any, name: str) -> dict[str, Any]:
        called.append(name)
        return {"type": "unknown", "capabilities": [], "tools": []}

    monkeypatch.setattr(agent_runner, "_resolve_agent_from_kg", _resolver)
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
        agent_name="messaging-assistant",
        task="deploy the service and restart the database now",
        engine=object(),
    )
    assert out == "done"
    assert called == [], "pass-through messaging-assistant must skip KG resolution"


def test_reply_budget_scales_with_shape() -> None:
    """CONCEPT:AU-ORCH.execution.passthrough-identity — the reply budget is derived from the shape: a direct/lean turn is
    short and answered inline; a full multi-agent tool turn earns a much larger budget and is
    delivered as a deferred follow-up."""
    import dataclasses

    from agent_utilities.orchestration.execution_profile import (
        _FULL_FIELDS,
        _LEAN_FIELDS,
        resolve_execution_profile,
    )

    base = resolve_execution_profile("chat")
    lean = dataclasses.replace(base, **_LEAN_FIELDS)
    full = dataclasses.replace(base, **_FULL_FIELDS)

    assert lean.is_interactive and lean.reply_budget_s <= 30.0
    assert not full.is_interactive and full.reply_budget_s >= 120.0
    # A full turn must earn a strictly larger budget than a lean one.
    assert full.reply_budget_s > lean.reply_budget_s


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
    """A cache hit must NOT touch the backend (zero I/O on the hot path, CONCEPT:AU-KG.memory.refresh-per-session-memento)."""
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
    turn N's memento from memory (CONCEPT:AU-KG.memory.refresh-per-session-memento / ECO-4.74)."""
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
    reasoning OFF (CONCEPT:AU-ORCH.execution.dynamic-execution-profile/1.68)."""
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
    usage-guard / discovery / verifier, reasoning ON (CONCEPT:AU-ORCH.execution.dynamic-execution-profile/1.68)."""
    from agent_utilities.orchestration.execution_profile import plan_execution_shape

    for q in (
        "deploy the service and then restart it",
        "analyze this repo and then create a migration plan",
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
    direct callers that plan no shape are unchanged (CONCEPT:AU-ORCH.execution.dynamic-execution-profile)."""
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
    """The shape threads onto GraphDeps so graph nodes can read it (CONCEPT:AU-ORCH.execution.direct-completion-shape)."""
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
    rglob the whole 234-repo tree + read_text every file on the event loop (CONCEPT:AU-ORCH.execution.direct-completion-shape).
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


# ───────── ORCH-1.69/1.70 — escalating cascade + Rust stage-2 + recipe cache ─────────


class _FakeGraphCompute:
    """Stand-in for engine.graph_compute exposing the lexical gate (CONCEPT:EG-ORCH.routing.lexical-capability-escalation)."""

    def __init__(self, terms: Any) -> None:
        self._terms = terms

    def match_ontology_terms(self, query: str) -> Any:
        if isinstance(self._terms, Exception):
            raise self._terms
        return self._terms


class _FakeSearchEngine:
    """Minimal engine exposing search_hybrid (stage 2) and, optionally, the lexical gate.

    ``lexical`` (when given) is the list of capability terms the ontology gate returns for
    any query; absent → no ``graph_compute`` attribute, so the gate reports "no capability".
    """

    def __init__(self, hits: Any, lexical: Any = None) -> None:
        self._hits = hits
        if lexical is not None:
            self.graph_compute = _FakeGraphCompute(lexical)

    def search_hybrid(self, query: str, top_k: int = 8) -> Any:
        if isinstance(self._hits, Exception):
            raise self._hits
        return self._hits


def test_signal_strength_grades() -> None:
    """The graded classifier is now PURELY STRUCTURAL (CONCEPT:EG-ORCH.routing.lexical-capability-escalation/ORCH-1.73): 0 = no
    structural signal (domain escalation is the lexical gate's job), 2+ = clearly complex."""
    from agent_utilities.graph.routing.strategies.fast_path import (
        orchestration_signal_strength as strength,
    )

    assert strength("hello") == 0
    assert strength("what is 2 plus 2?") == 0
    # Action vocabulary no longer scores here — it lives in the KG lexical gate.
    assert strength("search the docs") == 0
    assert strength("deploy the freshrss stack") == 0
    assert (
        strength("deploy the service and then restart it") >= 2
    )  # multi-clause ("and then") → complex
    assert strength("/deploy foo") >= 2  # slash command


def test_focused_tools_shape_carries_named_servers() -> None:
    """CONCEPT:AU-ORCH.execution.focused-tools-altitude — a lexical hit naming concrete server(s) yields the FOCUSED-TOOLS
    altitude: origin='lexical' + tool_servers = the distinct matched servers, best-score first."""
    from agent_utilities.orchestration.execution_profile import (
        plan_execution_shape,
        reset_recipe_cache,
    )

    # A turn naming BOTH portainer and github → bind both, github lower score sorts after.
    matches = [
        {
            "term": "github",
            "node_type": "Tool",
            "mcp_server": "github-mcp",
            "score": 6.0,
        },
        {
            "term": "portainer",
            "node_type": "Tool",
            "mcp_server": "portainer-mcp",
            "score": 9.0,
        },
        {
            "term": "portainer_stack",
            "node_type": "Tool",
            "mcp_server": "portainer-mcp",
            "score": 15.0,
        },
    ]
    eng = _FakeSearchEngine([], lexical=matches)
    reset_recipe_cache()
    shape = plan_execution_shape(
        "list my portainer stacks and the github issues",
        profile_hint="chat",
        engine=eng,
    )
    assert shape.direct_complete is False
    assert shape.origin == "lexical"
    # distinct, de-duped, best-score first (portainer_stack score 15 → portainer-mcp first)
    assert shape.tool_servers == ("portainer-mcp", "github-mcp")


def test_focused_tools_wins_over_structural_signal() -> None:
    """CONCEPT:AU-ORCH.execution.focused-tools-altitude — a turn that NAMES concrete capabilities takes the focused-tools
    altitude even when it is multi-clause + over-length (strength>=2). The real Telegram
    regression: 'fetch my github issues? ... ? list my portainer stacks?' was going to the
    full planning graph instead of one parallel tool loop."""
    from agent_utilities.graph.routing.strategies.fast_path import (
        orchestration_signal_strength,
    )
    from agent_utilities.orchestration.execution_profile import (
        plan_execution_shape,
        reset_recipe_cache,
    )

    msg = (
        "Can you use the github mcp to fetch me the open issues for my Knuckles-Team "
        "organization repositories? Is there a way to get issues wholistically open across "
        "all projects in the organization? Can you list the stacks I have running on portainer?"
    )
    assert orchestration_signal_strength(msg) >= 2  # multi-clause + over-length
    eng = _FakeSearchEngine(
        RuntimeError("semantic search must NOT run — focused-tools wins"),
        lexical=[
            {"term": "github", "mcp_server": "github-mcp", "score": 6.0},
            {"term": "portainer", "mcp_server": "portainer-mcp", "score": 9.0},
        ],
    )
    reset_recipe_cache()
    shape = plan_execution_shape(msg, profile_hint="chat", engine=eng)
    assert shape.origin == "lexical"
    assert shape.direct_complete is False
    assert set(shape.tool_servers) == {"portainer-mcp", "github-mcp"}


def test_focused_tools_resolver_dedups_and_orders() -> None:
    from agent_utilities.orchestration.execution_profile import (
        _lexical_capability_servers,
    )

    eng = _FakeSearchEngine(
        [],
        lexical=[
            {"term": "github", "mcp_server": "github-mcp", "score": 6.0},
            {"term": "portainer", "mcp_server": "portainer-mcp", "score": 9.0},
            {
                "term": "skillonly",
                "mcp_server": "",
                "score": 4.0,
            },  # no server → dropped
        ],
    )
    assert _lexical_capability_servers(eng, "x") == ["portainer-mcp", "github-mcp"]


def test_cascade_light_and_complex_skip_stage2() -> None:
    """Strength 0 → confident lean; strength ≥2 → confident full. Neither touches the engine."""
    from agent_utilities.orchestration.execution_profile import (
        plan_execution_shape,
        reset_recipe_cache,
    )

    reset_recipe_cache()
    boom = _FakeSearchEngine(
        RuntimeError("engine must NOT be called for confident jobs")
    )

    light = plan_execution_shape("what is 2 plus 2?", profile_hint="chat", engine=boom)
    assert light.direct_complete is True and light.origin == "heuristic"

    reset_recipe_cache()
    full = plan_execution_shape(
        "deploy the service and then restart it", profile_hint="chat", engine=boom
    )
    assert full.direct_complete is False and full.origin == "heuristic"
    assert full.confidence >= 0.9


# A substantial turn (> MAX_TRIVIAL_WORDS) that names no capability lexically — reaches stage 2.
_SUBSTANTIAL = "can you find everything related to the archived revenue records please"


def test_cascade_lexical_gate_hit_escalates() -> None:
    """CONCEPT:EG-ORCH.routing.lexical-capability-escalation — a turn naming a real fleet capability escalates via the FREE lexical
    gate (no vector search), the path that fixes the portainer/github classification bug."""
    from agent_utilities.orchestration.execution_profile import (
        plan_execution_shape,
        reset_recipe_cache,
    )

    reset_recipe_cache()
    # search_hybrid must NOT be consulted when the lexical gate already matched.
    eng = _FakeSearchEngine(
        RuntimeError("semantic search must NOT run after a lexical hit"),
        lexical=[
            {"term": "portainer", "node_type": "Tool", "label": "portainer_stack"}
        ],
    )
    shape = plan_execution_shape(
        "list the stacks I have on portainer", profile_hint="chat", engine=eng
    )
    assert shape.direct_complete is False
    assert shape.origin == "lexical"


def test_cascade_stage2_hits_keep_full() -> None:
    """A substantial turn that named no capability but whose KG search finds some IS a tool task."""
    from agent_utilities.orchestration.execution_profile import (
        plan_execution_shape,
        reset_recipe_cache,
    )

    reset_recipe_cache()
    eng = _FakeSearchEngine(
        [{"name": "archive_search"}, {"name": "fetch"}], lexical=[]
    )  # lexical miss → falls to stage 2
    shape = plan_execution_shape(_SUBSTANTIAL, profile_hint="chat", engine=eng)
    assert shape.direct_complete is False
    assert shape.origin == "designate"


def test_cascade_stage2_empty_downgrades_to_lean() -> None:
    """A substantial turn whose lexical gate AND KG search find nothing is conversational → lean."""
    from agent_utilities.orchestration.execution_profile import (
        plan_execution_shape,
        reset_recipe_cache,
    )

    reset_recipe_cache()
    shape = plan_execution_shape(
        _SUBSTANTIAL, profile_hint="chat", engine=_FakeSearchEngine([], lexical=[])
    )
    assert shape.direct_complete is True
    assert shape.origin == "designate-empty"


def test_cascade_short_turn_skips_stage2() -> None:
    """A SHORT turn that names no capability never pays the semantic tier — it leans (free)."""
    from agent_utilities.orchestration.execution_profile import (
        plan_execution_shape,
        reset_recipe_cache,
    )

    reset_recipe_cache()
    # search_hybrid raising would surface if stage 2 were (wrongly) reached for a short turn.
    eng = _FakeSearchEngine(
        RuntimeError("stage 2 must NOT run for a short turn"), lexical=[]
    )
    shape = plan_execution_shape("list my stacks", profile_hint="chat", engine=eng)
    assert shape.direct_complete is True
    assert shape.origin == "heuristic"


def test_cascade_stage2_unavailable_keeps_full() -> None:
    """If KG search is unavailable, a substantial ambiguous turn keeps the safe full graph."""
    from agent_utilities.orchestration.execution_profile import (
        plan_execution_shape,
        reset_recipe_cache,
    )

    reset_recipe_cache()
    eng = _FakeSearchEngine(RuntimeError("search down"), lexical=[])
    shape = plan_execution_shape(_SUBSTANTIAL, profile_hint="chat", engine=eng)
    assert shape.direct_complete is False
    assert (
        shape.origin == "heuristic"
    )  # unavailable → safe full default, not a downgrade


def test_recipe_cache_reuses_and_resets() -> None:
    """An identical job reuses the cached recipe (origin marked cache:…); reset clears it."""
    from agent_utilities.orchestration.execution_profile import (
        plan_execution_shape,
        reset_recipe_cache,
    )

    reset_recipe_cache()
    # A stage-2 job (substantial, lexical miss): caching it means the second call must NOT
    # hit the engine again.
    eng = _FakeSearchEngine([{"name": "x"}], lexical=[])
    first = plan_execution_shape(_SUBSTANTIAL, profile_hint="chat", engine=eng)
    assert first.origin == "designate"
    boom = _FakeSearchEngine(RuntimeError("cache must serve the 2nd call"), lexical=[])
    second = plan_execution_shape(_SUBSTANTIAL, profile_hint="chat", engine=boom)
    assert second.origin == "cache:designate"
    assert second.direct_complete == first.direct_complete

    reset_recipe_cache()
    third = plan_execution_shape(_SUBSTANTIAL, profile_hint="chat", engine=eng)
    assert third.origin == "designate"  # cache cleared → recomputed


def test_recipe_outcome_evicts_on_failure_keeps_on_success() -> None:
    """The cached recipe self-corrects: a failed run evicts it (re-plan next time); a
    successful run keeps it (reuse next time) — CONCEPT:AU-ORCH.execution.planner-failure-feedback."""
    from agent_utilities.orchestration.execution_profile import (
        plan_execution_shape,
        record_shape_outcome,
        reset_recipe_cache,
    )

    reset_recipe_cache()
    eng = _FakeSearchEngine([{"name": "x"}], lexical=[])
    cache_boom = _FakeSearchEngine(
        RuntimeError("must be served from cache"), lexical=[]
    )

    assert plan_execution_shape(
        _SUBSTANTIAL, profile_hint="chat", engine=eng
    ).origin == ("designate")
    # cached now
    assert (
        plan_execution_shape(
            _SUBSTANTIAL, profile_hint="chat", engine=cache_boom
        ).origin
        == "cache:designate"
    )

    # failure evicts -> next call re-plans (engine called again)
    record_shape_outcome(_SUBSTANTIAL, "chat", success=False)
    assert plan_execution_shape(
        _SUBSTANTIAL, profile_hint="chat", engine=eng
    ).origin == ("designate")

    # success keeps -> next call served from cache
    record_shape_outcome(_SUBSTANTIAL, "chat", success=True)
    assert (
        plan_execution_shape(
            _SUBSTANTIAL, profile_hint="chat", engine=cache_boom
        ).origin
        == "cache:designate"
    )


# ───────── ORCH-1.71 — learned shape policy (OutcomeRouter over the reward-EMA spine) ─────────


def test_outcome_router_prior_until_learned() -> None:
    """Neutral → return the heuristic prior; once an alternative's reward-EMA exceeds the
    prior's, select flips. outcome_reward ranks fast-success > slow-success > failure."""
    from agent_utilities.orchestration.outcome_router import (
        OutcomeRouter,
        outcome_reward,
    )

    r = OutcomeRouter("test")
    assert r.select("cls", "lean", ("lean", "full")) == "lean"
    assert r.select("cls", "full", ("lean", "full")) == "full"
    for _ in range(10):
        r.record("cls", "full", 1.0)
        r.record("cls", "lean", 0.0)
    assert r.select("cls", "lean", ("lean", "full")) == "full"  # learned override

    assert (
        outcome_reward(success=True, latency_s=1.0)
        > outcome_reward(success=True, latency_s=25.0)
        > outcome_reward(success=False, latency_s=1.0)
    )


def test_outcome_router_is_per_task_class() -> None:
    """Learning for one task-class must not leak into another."""
    from agent_utilities.orchestration.outcome_router import OutcomeRouter

    r = OutcomeRouter("test")
    for _ in range(10):
        r.record("A", "full", 1.0)
        r.record("A", "lean", 0.0)
    assert r.select("A", "lean", ("lean", "full")) == "full"  # learned for A
    assert r.select("B", "lean", ("lean", "full")) == "lean"  # B untouched → prior


def test_shape_policy_overlay_flips_after_learning() -> None:
    """The planner's heuristic shape is a prior the learned policy refines per task-class
    (CONCEPT:AU-ORCH.execution.shape-policy-learning); a reset restores the pure heuristic."""
    from dataclasses import replace

    from agent_utilities.orchestration.execution_profile import (
        _FULL_FIELDS,
        plan_execution_shape,
        record_shape_outcome,
        reset_recipe_cache,
        reset_shape_policy,
    )

    reset_shape_policy()
    reset_recipe_cache()
    q = "tell me what you think about this"  # strength 0 → heuristic prior = lean
    p0 = plan_execution_shape(q, profile_hint="chat")
    assert p0.direct_complete is True and p0.origin == "heuristic"

    full = replace(p0, **_FULL_FIELDS)
    lean = replace(p0)  # the lean shape actually planned
    for _ in range(8):
        record_shape_outcome(q, "chat", success=True, latency_s=2.0, shape=full)
        record_shape_outcome(q, "chat", success=False, latency_s=30.0, shape=lean)

    p1 = plan_execution_shape(q, profile_hint="chat")
    assert p1.direct_complete is False and p1.origin == "policy:full"

    reset_shape_policy()
    reset_recipe_cache()
    p2 = plan_execution_shape(q, profile_hint="chat")
    assert p2.direct_complete is True and p2.origin == "heuristic"
    reset_shape_policy()  # clean up so later tests see a neutral policy


def test_focused_tools_reply_budget_is_tighter_than_full() -> None:
    """CONCEPT:AU-ORCH.execution.focused-tools-altitude — a focused-tools turn (one parallel tool loop) gets a much tighter
    reply budget than a full multi-agent turn (~190s); it grows mildly per bound server."""
    from dataclasses import replace

    from agent_utilities.orchestration.execution_profile import (
        _FULL_FIELDS,
        resolve_execution_profile,
    )

    base = resolve_execution_profile("chat")
    full = replace(base, **_FULL_FIELDS)
    one = replace(full, tool_servers=("portainer-mcp",))
    two = replace(full, tool_servers=("portainer-mcp", "github-mcp"))
    assert full.reply_budget_s >= 150.0  # full apparatus
    assert one.reply_budget_s == 55.0  # 35 + 20*1
    assert two.reply_budget_s == 75.0  # 35 + 20*2
    assert two.reply_budget_s < full.reply_budget_s
