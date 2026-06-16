"""Shared action-dispatch helpers: discovery, aliases, did-you-mean."""

from __future__ import annotations

import pytest

from agent_utilities.mcp.action_dispatch import (
    DISCOVERY_ACTIONS,
    canonicalize,
    dispatch,
    public_actions,
    resolve_action,
    suggest,
    unknown_action_error,
)


class _Client:
    def get_movie(self, **kwargs):
        return {"called": "get_movie", "kwargs": kwargs}

    def add_movie(self, **kwargs):
        return {"called": "add_movie"}

    def _private(self):  # excluded from discovery
        return None


def test_public_actions_excludes_private():
    actions = public_actions(_Client())
    assert "get_movie" in actions and "add_movie" in actions
    assert "_private" not in actions


def test_suggest_finds_close_match():
    assert "get_movie" in suggest("get_movei", ["get_movie", "add_movie"])


def test_canonicalize_plural_and_alias():
    valid = ["get_movie", "get_series"]
    assert canonicalize("get_movies", valid) == "get_movie"  # plural -> singular
    assert canonicalize("get_movie", valid) == "get_movie"  # identity
    assert canonicalize("films", valid, aliases={"films": "get_movie"}) == "get_movie"
    assert canonicalize("nope", valid) is None


def test_dispatch_discovery_returns_actions():
    for keyword in DISCOVERY_ACTIONS:
        res = dispatch(_Client(), keyword, {}, service="radarr")
        assert res["service"] == "radarr"
        assert "get_movie" in res["actions"]


def test_dispatch_resolves_plural_and_calls():
    res = dispatch(_Client(), "get_movies", {"tmdbId": 1}, service="radarr")
    assert res == {"called": "get_movie", "kwargs": {"tmdbId": 1}}


def test_dispatch_unknown_raises_rich_error():
    with pytest.raises(ValueError) as exc:
        dispatch(_Client(), "get_movei", {}, service="radarr")
    msg = str(exc.value)
    assert "list_actions" in msg and "get_movie" in msg


def test_dispatch_result_coercer_applied():
    class _Model:
        def model_dump(self):
            return {"dumped": True}

    class _C:
        def get(self, **kwargs):
            return _Model()

    res = dispatch(_C(), "get", {}, result_coercer=lambda r: r.model_dump())
    assert res == {"dumped": True}


def test_resolve_action_for_explicit_dispatch():
    valid = ["list_workflows", "list_runs", "get_run"]
    # discovery
    disc = resolve_action("list_actions", valid, service="actions")
    assert disc == {"service": "actions", "actions": sorted(valid)}
    # canonical passthrough
    assert resolve_action("list_runs", valid) == "list_runs"
    # unknown -> rich error
    with pytest.raises(ValueError) as exc:
        resolve_action("list_run", valid, service="actions")
    assert "Did you mean" in str(exc.value)


def test_unknown_action_error_without_matches():
    err = unknown_action_error("zzzzz", ["get_movie"], target="radarr")
    assert "Unknown action 'zzzzz' on radarr" in str(err)
    assert "list_actions" in str(err)
