"""Plan 03 Step 3: unified routing package — Router composition + R1 fast-path.

Validates the strangler (historical imports still resolve) and the new
RoutingStrategy/Router framework with the extracted fast-path strategy.
"""

from __future__ import annotations

import types

import pytest

from agent_utilities.graph.routing import (
    FastPathStrategy,
    Router,
    RoutingConfig,
    RoutingStrategy,
    is_trivial_query,
    router_step,
)


def _ctx(query: str):
    return types.SimpleNamespace(state=types.SimpleNamespace(query=query))


def test_strangler_reexports_monolith():
    # The package re-exports the implementation's step function unchanged.
    import agent_utilities.graph._router_impl as impl

    assert router_step is impl.router_step


def test_is_trivial_query_rules():
    assert is_trivial_query("hello") is True
    assert is_trivial_query("hi there") is True
    assert is_trivial_query("thanks a lot") is True
    # Structural escalators are NOT trivial (slash-command / multi-clause / over-length).
    # Domain/action vocabulary is no longer scored here — escalation for a turn that names a
    # real capability is the KG lexical gate's job (CONCEPT:EG-010/ORCH-1.73).
    assert is_trivial_query("/optimize my portfolio") is False
    assert is_trivial_query("compute the frontier and then plot the results") is False
    assert is_trivial_query("") is False


def test_fast_path_strategy_is_protocol_member():
    assert isinstance(FastPathStrategy(), RoutingStrategy)


@pytest.mark.asyncio
async def test_fast_path_strategy_decides():
    s = FastPathStrategy()
    assert await s.decide(_ctx("hello")) == "fast_path"
    # Multi-clause → structural escalation → the fast path defers.
    assert (
        await s.decide(_ctx("design the schema and then build the scheduler")) is None
    )


@pytest.mark.asyncio
async def test_router_runs_pipeline_in_order():
    calls: list[str] = []

    class Recording:
        def __init__(self, name, decision):
            self.name = name
            self._decision = decision

        async def decide(self, ctx):
            calls.append(self.name)
            return self._decision

    # 'a' defers (None), 'b' decides -> 'c' must never run.
    router = Router(
        strategies=[Recording("a", None), Recording("b", "B"), Recording("c", "C")],
        config=RoutingConfig(pipeline=["a", "b", "c"]),
    )
    decision = await router.route(_ctx("x"))
    assert decision == "B"
    assert calls == ["a", "b"]  # short-circuits after first decision


@pytest.mark.asyncio
async def test_router_returns_none_when_all_defer():
    router = Router(
        strategies=[FastPathStrategy()],
        config=RoutingConfig(pipeline=["fast_path"]),
    )
    # Non-trivial (structural) query -> fast_path defers -> no decision (caller falls back).
    assert await router.route(_ctx("/build an OWL reasoner")) is None
