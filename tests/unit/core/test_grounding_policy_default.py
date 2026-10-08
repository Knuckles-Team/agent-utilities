"""The deployment grounding default applies only outside an explicit scope."""

from __future__ import annotations

import contextvars

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
    assert contextvars.Context().run(current_grounding_policy) == "best_effort"


def test_unconfigured_default_stays_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(contextual_model, "setting", lambda name, default=None: default)
    assert contextvars.Context().run(current_grounding_policy) == "required"


def test_an_explicit_scope_overrides_the_deployment_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        contextual_model, "setting", lambda name, default=None: "best_effort"
    )

    def scoped() -> tuple[str, str, str]:
        with use_grounding_policy("required"):
            explicit = current_grounding_policy()
        with use_grounding_policy():
            deferred = current_grounding_policy()
        return explicit, deferred, current_grounding_policy()

    assert contextvars.Context().run(scoped) == (
        "required",
        "best_effort",
        "best_effort",
    )
