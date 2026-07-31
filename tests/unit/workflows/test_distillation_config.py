"""D-32(b): ``_load_distillation_config`` reads real ``AgentConfig`` fields.

Previously imported a nonexistent ``agent_utilities.config`` module and read a
``.raw`` dict shape that doesn't exist on ``AgentConfig`` (an env-var-driven
``BaseSettings``) — every call fell through the bare ``except`` to hardcoded
defaults, silently discarding any configured override.

@pytest.mark.concept("AU-AHE.optimization.workflow-distillation")
"""

from __future__ import annotations

import pytest

from agent_utilities.workflows.distillation_hook import (
    DEFAULT_PROMOTION_THRESHOLD,
    DEFAULT_QUALITY_MINIMUM,
    _load_distillation_config,
)

pytestmark = pytest.mark.concept("AU-AHE.optimization.workflow-distillation")


def test_defaults_when_unset(monkeypatch):
    monkeypatch.delenv("DISTILLATION_PROMOTION_THRESHOLD", raising=False)
    monkeypatch.delenv("DISTILLATION_QUALITY_SCORE_MINIMUM", raising=False)
    threshold, quality = _load_distillation_config()
    assert threshold == DEFAULT_PROMOTION_THRESHOLD
    assert quality == DEFAULT_QUALITY_MINIMUM


def test_env_override_is_actually_read(monkeypatch):
    """The real regression: an explicit override used to be silently ignored
    because the import path/attribute shape it read never existed."""
    monkeypatch.setenv("DISTILLATION_PROMOTION_THRESHOLD", "9")
    monkeypatch.setenv("DISTILLATION_QUALITY_SCORE_MINIMUM", "0.42")
    threshold, quality = _load_distillation_config()
    assert threshold == 9
    assert quality == pytest.approx(0.42)
