"""Tests for the quant ``add_node(id=...)`` kwarg-drift bug (exhaustive kg-*
validation): delegating a quant task (``quant`` MCP tool, ``domain='orchestrate'``,
``action='regime'``) crashed with::

    IntelligenceGraphEngine.add_node() got an unexpected keyword argument 'id'

Root cause: ``RegimeDetector._persist_to_kg`` / ``DebateEngine._persist_to_kg`` /
``StrategyEngine.register_strategy`` / ``StrategyEngine.record_backtest`` called
``self.engine.add_node(id=..., node_type=..., <extra kwargs>...)`` against the
REAL ``IntelligenceGraphEngine.add_node(self, node_id, node_type,
properties=None, ephemeral=False, *, session=None)`` — whose first positional
param is ``node_id``, not ``id``, and which has no ``**kwargs`` catch-all for
free-form properties.

These tests use a strict fake engine whose ``add_node``/``add_edge`` mirror the
REAL engine signatures exactly (unlike ``GraphComputeEngine``, which accepts
arbitrary kwargs) — a wrong-kwarg call raises ``TypeError`` against it exactly
as it did against the real engine, so a regression back to ``id=`` is caught
immediately. Each test drives the REAL, already-wired call site (not a
reimplementation) — the exact method that crashed.
"""

from __future__ import annotations

import asyncio
import json

import pandas as pd

from agent_utilities.domains.finance.debate_engine import (
    DebateContext,
    DebateEngine,
    DebateSession,
    RiskVeto,
)
from agent_utilities.domains.finance.market_data import DataRegistry, SyntheticProvider
from agent_utilities.domains.finance.regime_detector import RegimeDetector
from agent_utilities.domains.finance.strategy_engine import (
    StrategyEngine,
    StrategyMetrics,
)


class _StrictRealSignatureEngine:
    """Fake engine whose ``add_node``/``add_edge``/``query_cypher`` mirror the
    REAL ``IntelligenceGraphEngine`` signatures (see ``knowledge_graph/core/
    engine.py``): ``add_node(node_id, node_type, properties=None, ...)`` — NOT
    ``GraphComputeEngine``'s permissive ``**kwargs`` form. Calling this with a
    stray ``id=`` kwarg (the bug) raises ``TypeError`` exactly like production.
    """

    def __init__(self) -> None:
        self.nodes: list[dict] = []
        self.edges: list[tuple[str, str, str]] = []
        self.cypher_calls: list[tuple[str, dict]] = []

    def add_node(
        self, node_id, node_type, properties=None, ephemeral=False, *, session=None
    ):
        self.nodes.append(
            {"node_id": node_id, "node_type": node_type, "properties": properties or {}}
        )

    def add_edge(
        self,
        source,
        target,
        rel_type="",
        ephemeral=False,
        *,
        session=None,
        **properties,
    ):
        self.edges.append((source, target, rel_type))

    def query_cypher(self, query, params=None, *args, **kwargs):
        self.cypher_calls.append((query, params or {}))
        return []


# ── RegimeDetector (the exact "quant orchestrate/regime" crash site) ─────────
def test_regime_detector_persists_via_real_add_node_signature():
    engine = _StrictRealSignatureEngine()
    detector = RegimeDetector(engine)

    prices = [100.0 + i * 0.5 for i in range(60)]  # steady uptrend
    df = pd.DataFrame({"Close": prices})

    regime = detector.detect_regime(df, ticker="AAPL")

    assert regime != "unknown"
    assert len(engine.nodes) == 1
    node = engine.nodes[0]
    assert node["node_id"] == "Regime_AAPL"
    assert node["node_type"] == "MarketRegime"
    assert node["properties"]["ticker"] == "AAPL"
    assert node["properties"]["regime_type"] == regime
    assert "volatility" in node["properties"]


def test_quant_orchestrate_regime_live_path(monkeypatch):
    """The FULL live path: the registered ``quant`` MCP tool, ``domain=
    'orchestrate'``, ``action='regime'`` — exactly the call that crashed
    (``Action 'regime' failed in domain 'orchestrate': ... unexpected keyword
    argument 'id'``). Data fetch is scoped to the deterministic
    ``SyntheticProvider`` (no network) so the test is hermetic."""
    from agent_utilities.domains.finance import quant_mcp_tools

    class _CollectingMCP:
        def __init__(self):
            self.tools = {}

        def tool(self, *args, **kwargs):
            def _deco(fn):
                self.tools[fn.__name__] = fn
                return fn

            return _deco

    engine = _StrictRealSignatureEngine()
    mcp = _CollectingMCP()
    quant = quant_mcp_tools.register_quant_tools(mcp, engine_default=engine)

    monkeypatch.setattr(
        quant_mcp_tools,
        "DataRegistry",
        lambda *a, **k: DataRegistry(providers=[SyntheticProvider()]),
    )

    result = quant(domain="orchestrate", action="regime", ticker="AAPL")

    assert "failed in domain" not in result, result
    assert "unexpected keyword argument" not in result, result
    assert "Current market regime for AAPL" in result
    assert len(engine.nodes) == 1
    assert engine.nodes[0]["node_type"] == "MarketRegime"


# ── DebateEngine (same bug class, second live site) ──────────────────────────
def test_debate_engine_persists_via_real_add_node_signature():
    engine = _StrictRealSignatureEngine()
    debate = DebateEngine(engine=engine)

    context = DebateContext(ticker="TSLA", asset_class="equity")
    session = DebateSession(session_id="s1", context=context)
    session.final_decision = "BUY"
    session.risk_assessment = RiskVeto(
        approved=True, reasoning="Solid fundamentals", max_position_size=0.1
    )

    debate._persist_to_kg(session)

    assert len(engine.nodes) == 2
    assert engine.nodes[0]["node_id"] == "Debate_s1_TSLA"
    assert engine.nodes[0]["node_type"] == "DebateSession"
    assert engine.nodes[0]["properties"]["ticker"] == "TSLA"
    assert engine.nodes[0]["properties"]["decision"] == "BUY"
    assert engine.nodes[1]["node_id"] == "Debate_s1_TSLA_Risk"
    assert engine.nodes[1]["node_type"] == "RiskAssessment"
    assert engine.edges == [("Debate_s1_TSLA", "Debate_s1_TSLA_Risk", "EVALUATED_BY")]


# ── StrategyEngine (third live site: graph_analyze(action="quant_strategy")) ─
def test_strategy_engine_register_persists_via_real_add_node_signature():
    engine = _StrictRealSignatureEngine()
    se = StrategyEngine(engine)

    sid = se.register_strategy(
        name="Momentum V1", code_ref="strategies/momentum.py", author="quant"
    )

    assert sid == "Strat_Momentum_V1"
    assert len(engine.nodes) == 1
    assert engine.nodes[0]["node_id"] == sid
    assert engine.nodes[0]["node_type"] == "TradingStrategy"
    assert engine.nodes[0]["properties"]["author"] == "quant"


def test_strategy_engine_record_backtest_persists_and_promotes():
    engine = _StrictRealSignatureEngine()
    se = StrategyEngine(engine)

    metrics = StrategyMetrics(
        sharpe=2.5,
        max_drawdown=-0.10,
        win_rate=0.55,
        profit_factor=1.5,
        total_trades=100,
    )
    promotable = se.record_backtest("Strat_X", metrics)

    assert promotable is True
    assert len(engine.nodes) == 1
    assert engine.nodes[0]["node_type"] == "BacktestResult"
    assert engine.edges == [(engine.nodes[0]["node_id"], "Strat_X", "VALIDATES")]
    # promotion gate fired the status-update Cypher write
    assert len(engine.cypher_calls) == 1


def test_graph_analyze_quant_strategy_live_path(monkeypatch):
    """The FULL live path: ``graph_analyze(action='quant_strategy', ...)`` —
    exactly the call site in ``mcp/tools/analysis_tools.py`` that hit the same
    ``add_node(id=...)`` kwarg drift."""
    from agent_utilities.mcp import kg_server

    kg_server.ensure_tools_registered()
    engine = _StrictRealSignatureEngine()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)

    tool = kg_server.REGISTERED_TOOLS["graph_analyze"]
    out = asyncio.run(tool(action="quant_strategy", query="Strat_Y", top_k=10))

    assert "Error" not in out, out
    payload = json.loads(out)
    assert payload["strategy_id"] == "Strat_Y"
    assert len(engine.nodes) == 1
    assert engine.nodes[0]["node_type"] == "BacktestResult"
