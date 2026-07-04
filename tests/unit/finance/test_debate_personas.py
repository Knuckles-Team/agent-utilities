"""Persona-voiced debate wiring (CONCEPT:AU-KG.research.research-pipeline-runner).

Verifies the Bull/Bear DebateEngine actually loads investor-persona prompt
bodies as its system prompt — closing the loop so each side argues in a
specific investor's voice. Runs fully offline (no LLM): we assert on the
prompt-building + labelling, which is where the wire lives.
"""

from __future__ import annotations

from agent_utilities.domains.finance.debate_engine import (
    _GENERIC_BEAR_PROMPT,
    _GENERIC_BULL_PROMPT,
    DebateEngine,
)
from agent_utilities.domains.finance.investor_debate import (
    DEFAULT_BEAR_PERSONA,
    DEFAULT_BULL_PERSONA,
    persona_archetype,
    persona_system_prompt,
)


def test_persona_system_prompt_loads_real_body():
    sp = persona_system_prompt("buffett_investor")
    assert sp  # non-empty
    # carries the persona's actual identity/voice, not the generic prompt
    assert "Buffett" in sp or "value" in sp.lower()
    assert "archetype" in sp.lower()
    assert "never invent" in sp.lower()  # the grounding guardrail


def test_persona_system_prompt_missing_returns_empty():
    assert persona_system_prompt("does_not_exist_persona") == ""


def test_persona_archetype():
    assert persona_archetype("buffett_investor") == "BuffettInvestor"
    assert persona_archetype("missing") == "missing"  # graceful


def test_engine_without_personas_uses_generic_voice():
    eng = DebateEngine()
    assert eng._bull_system_prompt() == _GENERIC_BULL_PROMPT
    assert eng._bear_system_prompt() == _GENERIC_BEAR_PROMPT
    assert eng._bull_label() == "Bull Researcher"
    assert eng._bear_label() == "Bear Researcher"


def test_with_personas_binds_investor_voices():
    eng = DebateEngine.with_personas(bull="buffett_investor", bear="burry_investor")
    bull_sp = eng._bull_system_prompt()
    bear_sp = eng._bear_system_prompt()
    # Each side now speaks in its investor's voice, not the generic prompt
    assert bull_sp != _GENERIC_BULL_PROMPT
    assert bear_sp != _GENERIC_BEAR_PROMPT
    assert bull_sp == persona_system_prompt("buffett_investor")
    assert bear_sp == persona_system_prompt("burry_investor")
    # labels carry the archetype so the audit trail shows who argued
    assert eng._bull_label() == "Bull Researcher (BuffettInvestor)"
    assert eng._bear_label() == "Bear Researcher (BurryInvestor)"


def test_with_personas_defaults_to_buffett_vs_burry():
    eng = DebateEngine.with_personas()
    assert eng.bull_persona == DEFAULT_BULL_PERSONA == "buffett_investor"
    assert eng.bear_persona == DEFAULT_BEAR_PERSONA == "burry_investor"


def test_bull_fallback_argument_carries_persona_label():
    # No LLM/model available in unit env → generation hits the fallback path,
    # which must still stamp the persona label (live-path behaviour).
    from agent_utilities.domains.finance.debate_engine import DebateContext

    eng = DebateEngine.with_personas(bull="damodaran_investor")
    ctx = DebateContext(ticker="TSLA", asset_class="equity")
    arg = eng._generate_bull_argument(ctx, [], 1)
    assert arg.role == "Bull Researcher (DamodaranInvestor)"
