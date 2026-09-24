"""Finance agent roles (KG-2.6) -- orchestration and LLM-driven analysis only.

Deterministic finance math lives in epistemic-graph (``FinanceMarket``,
``FinanceSignalModels`` and the ``Finance*`` kernels); market data and order
execution live in the fleet connectors (``emerald-exchange``,
``market-data-mcp``). What stays here are the agent roles that coordinate that
work: the trading swarm and its calibration, the investor debate and persona
heuristics, the research autopilot, filing and forensic evidence for the
debate personas, the Kronos forecaster and the flip explainer (EH-423 /
AUD-30).
"""

from __future__ import annotations

import importlib
from typing import Any

# CONCEPT:AU-KG.domains.lazy-symbol-loading — each symbol resolves to its
# submodule on first attribute access, so importing the package stays cheap and
# never fails for a missing optional extra.
_SYMBOL_MODULES: dict[str, str] = {
    "AgentSignal": "trading_swarm",
    "AutopilotConfig": "research_autopilot",
    "BacktestMetrics": "research_autopilot",
    "CalibrationScore": "calibration_tracker",
    "CalibrationTracker": "calibration_tracker",
    "CallRecord": "calibration_tracker",
    "CandleType": "kronos_forecaster",
    "DEFAULT_BEAR_PERSONA": "investor_debate",
    "DEFAULT_BULL_PERSONA": "investor_debate",
    "Evidence": "flip_explainer",
    "FilingDiff": "filing_diff",
    "FilingDiffAgent": "filing_diff",
    "FilingDiffFinding": "filing_diff",
    "FilingDiffResult": "filing_diff",
    "FlipExplanation": "flip_explainer",
    "FlipMath": "flip_explainer",
    "ForecastResult": "kronos_forecaster",
    "ForensicScreener": "forensic_screener",
    "ForensicVerdict": "forensic_screener",
    "Heuristic": "persona_heuristics",
    "HeuristicResult": "persona_heuristics",
    "Hypothesis": "research_autopilot",
    "HypothesisResult": "research_autopilot",
    "HypothesisStatus": "research_autopilot",
    "INVESTOR_PERSONAS": "investor_debate",
    "KLineToken": "kronos_forecaster",
    "KLineTokenizer": "kronos_forecaster",
    "KronosForecaster": "kronos_forecaster",
    "KronosPredictor": "kronos_forecaster",
    "PERSONA_HEURISTICS": "persona_heuristics",
    "PersonaEvaluation": "persona_heuristics",
    "PersonaRole": "investor_debate",
    "ResearchAutopilot": "research_autopilot",
    "ResearchReport": "research_autopilot",
    "SPECIALIST_ROLES": "investor_debate",
    "SimpleBacktester": "research_autopilot",
    "SourcedClaim": "flip_explainer",
    "SwarmAgent": "trading_swarm",
    "SwarmConfig": "trading_swarm",
    "SwarmConsensus": "trading_swarm",
    "SwarmDecision": "trading_swarm",
    "SwarmRole": "trading_swarm",
    "TradingSwarm": "trading_swarm",
    "apply_calibration_to_swarm": "calibration_tracker",
    "brier_score": "calibration_tracker",
    "build_financial_debate_team": "investor_debate",
    "calibrated_role_weights": "calibration_tracker",
    "diff_filing_sections": "filing_diff",
    "evaluate_all": "persona_heuristics",
    "evaluate_persona": "persona_heuristics",
    "explain_flip": "flip_explainer",
    "flip_math": "flip_explainer",
    "keep_sourced": "flip_explainer",
    "list_personas": "persona_heuristics",
    "load_persona_prompt": "investor_debate",
    "persona_archetype": "investor_debate",
    "persona_for_role": "investor_debate",
    "persona_heuristics_batch": "persona_heuristics",
    "persona_system_prompt": "investor_debate",
    "seed_financial_debate_team": "investor_debate",
    "seed_persona_heuristics": "persona_heuristics",
}


def __getattr__(name: str) -> Any:
    """Lazily import a finance symbol from its submodule on first access."""
    module = _SYMBOL_MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = importlib.import_module(f".{module}", __name__)
    value = getattr(mod, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_SYMBOL_MODULES))


__all__ = [
    "AgentSignal",
    "AutopilotConfig",
    "BacktestMetrics",
    "CalibrationScore",
    "CalibrationTracker",
    "CallRecord",
    "CandleType",
    "DEFAULT_BEAR_PERSONA",
    "DEFAULT_BULL_PERSONA",
    "Evidence",
    "FilingDiff",
    "FilingDiffAgent",
    "FilingDiffFinding",
    "FilingDiffResult",
    "FlipExplanation",
    "FlipMath",
    "ForecastResult",
    "ForensicScreener",
    "ForensicVerdict",
    "Heuristic",
    "HeuristicResult",
    "Hypothesis",
    "HypothesisResult",
    "HypothesisStatus",
    "INVESTOR_PERSONAS",
    "KLineToken",
    "KLineTokenizer",
    "KronosForecaster",
    "KronosPredictor",
    "PERSONA_HEURISTICS",
    "PersonaEvaluation",
    "PersonaRole",
    "ResearchAutopilot",
    "ResearchReport",
    "SPECIALIST_ROLES",
    "SimpleBacktester",
    "SourcedClaim",
    "SwarmAgent",
    "SwarmConfig",
    "SwarmConsensus",
    "SwarmDecision",
    "SwarmRole",
    "TradingSwarm",
    "apply_calibration_to_swarm",
    "brier_score",
    "build_financial_debate_team",
    "calibrated_role_weights",
    "diff_filing_sections",
    "evaluate_all",
    "evaluate_persona",
    "explain_flip",
    "flip_math",
    "keep_sourced",
    "list_personas",
    "load_persona_prompt",
    "persona_archetype",
    "persona_for_role",
    "persona_heuristics_batch",
    "persona_system_prompt",
    "seed_financial_debate_team",
    "seed_persona_heuristics",
]
