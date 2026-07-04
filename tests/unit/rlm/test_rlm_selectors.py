#!/usr/bin/python
"""Tests for RLM config selectors: configurable max_turns + lossless/compaction.

CONCEPT:AU-ORCH.execution.predict-rlm-runtime
"""

import pytest

from agent_utilities.rlm.config import RLMConfig
from agent_utilities.rlm.repl import RLMEnvironment

pytestmark = pytest.mark.concept("AU-ORCH.execution.predict-rlm-runtime")


# --- configurable max_turns -------------------------------------------------


def test_max_turns_default_and_override():
    assert RLMConfig().max_turns == 5
    assert RLMConfig(max_turns=3).max_turns == 3


def test_env_reads_max_turns_from_config():
    env = RLMEnvironment(config=RLMConfig(max_turns=2))
    assert env.max_turns == 2  # live: run_full_rlm bounds its loop by self.max_turns


# --- lossless-vs-compaction selector ---------------------------------------


def test_strategy_rlm_lossless_when_triggered():
    cfg = RLMConfig(
        enabled=False, trigger_on_large_output=True, max_context_threshold=100
    )
    assert cfg.select_long_context_strategy(output_size=150) == "rlm_lossless"


def test_strategy_memento_compaction_when_large_but_not_triggered():
    cfg = RLMConfig(
        enabled=False,
        trigger_on_large_output=False,  # size does not trigger RLM
        compaction_threshold=1000,
    )
    assert cfg.select_long_context_strategy(output_size=2000) == "memento_compaction"


def test_strategy_none_when_small():
    cfg = RLMConfig(
        enabled=False, trigger_on_large_output=False, compaction_threshold=1000
    )
    assert cfg.select_long_context_strategy(output_size=50) == "none"


def test_strategy_enabled_always_lossless():
    assert (
        RLMConfig(enabled=True).select_long_context_strategy(output_size=0)
        == "rlm_lossless"
    )
