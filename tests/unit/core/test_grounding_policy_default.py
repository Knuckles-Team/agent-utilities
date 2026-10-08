"""The deployment grounding default applies only outside an explicit scope."""

from __future__ import annotations

import pytest

from agent_utilities.core import contextual_model
from agent_utilities.core.contextual_model import (
    current_grounding_policy,
    use_grounding_policy,
)


def test_unscoped_calls_use_the_deployment_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        contextual_model, "setting", lambda name, default=None: "best_effort"
    )
    assert current_grounding_policy() == "best_effort"


def test_unconfigured_default_stays_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(contextual_model, "setting", lambda name, default=None: default)
    assert current_grounding_policy() == "required"


def test_an_explicit_scope_overrides_the_deployment_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        contextual_model, "setting", lambda name, default=None: "best_effort"
    )
    with use_grounding_policy("required"):
        assert current_grounding_policy() == "required"
    assert current_grounding_policy() == "best_effort"
