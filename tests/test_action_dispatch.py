"""Shared action-dispatch helpers: discovery, aliases, did-you-mean."""

from __future__ import annotations

import threading

import anyio
import pytest

from agent_utilities.mcp.action_dispatch import (
    DISCOVERY_ACTIONS,
    canonicalize,
    dispatch,
    dispatch_async,
    is_destructive_action,
    parse_json_object,
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


@pytest.mark.parametrize("value", [None, "", "   ", b"", bytearray()])
def test_parse_json_object_accepts_empty_values(value):
    assert parse_json_object(value) == {}


def test_parse_json_object_returns_object():
    assert parse_json_object('{"page": 2}') == {"page": 2}


@pytest.mark.parametrize("value", ['{"token": "private"', '["private"]'])
def test_parse_json_object_rejects_invalid_shape_without_echo(value):
    with pytest.raises(ValueError) as exc:
        parse_json_object(value)

    assert "private" not in str(exc.value)


def test_parse_json_object_rejects_oversized_input_before_parsing():
    with pytest.raises(ValueError, match="input limit"):
        parse_json_object("{" + " " * (64 * 1024) + "}")


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


def test_dispatch_folds_flat_fields_into_body_param():
    """dispatch() self-heals the REST-body convention like run_blocking does."""

    class _JiraLike:
        def create_issue(self, update_history=None, payload=None):
            return {"update_history": update_history, "payload": payload}

        def get_all_projects(self):
            return ["AU", "KAN"]

    c = _JiraLike()
    # LLM's flat fields folded into the single `payload` body param
    res = dispatch(c, "create_issue", {"fields": {"summary": "S"}}, service="jira")
    assert res == {"update_history": None, "payload": {"fields": {"summary": "S"}}}
    # read path (no body param) is a strict no-op
    assert dispatch(c, "get_all_projects", {}, service="jira") == ["AU", "KAN"]


def test_dispatch_async_offloads_sync_method():
    main_thread = threading.get_ident()

    class _SyncClient:
        def get_record(self):
            return {"thread": threading.get_ident()}

    result = anyio.run(dispatch_async, _SyncClient(), "get_record")

    assert result["thread"] != main_thread


def test_dispatch_async_awaits_native_method_and_async_coercer():
    class _AsyncClient:
        async def get_record(self, *, record_id):
            return {"record_id": record_id}

    async def _coerce(result):
        return {"coerced": result}

    async def _run():
        return await dispatch_async(
            _AsyncClient(),
            "get_record",
            {"record_id": "abc"},
            result_coercer=_coerce,
        )

    result = anyio.run(_run)

    assert result == {"coerced": {"record_id": "abc"}}


@pytest.mark.parametrize("action", ["delete_record", "projects_delete"])
def test_is_destructive_action_matches_exact_tokens(action):
    assert is_destructive_action(action) is True


def test_is_destructive_action_prefers_operation_metadata():
    assert is_destructive_action("undelete_record") is False
    assert is_destructive_action("get_record", {"http": "DELETE"}) is True
    assert is_destructive_action("delete_record", {"destructive": False}) is False


class _ElicitResult:
    action = "decline"
    data = True


class _DecliningContext:
    def __init__(self):
        self.asked = []

    async def elicit(self, message, response_type=bool):
        self.asked.append(message)
        return _ElicitResult()


def test_dispatch_async_requires_confirmation_before_destructive_invocation():
    class _TrackedClient:
        def __init__(self):
            self.called = []

        async def delete_record(self):
            self.called.append("delete_record")

    client = _TrackedClient()
    ctx = _DecliningContext()

    async def _run():
        return await dispatch_async(client, "delete_record", ctx=ctx)

    result = anyio.run(_run)

    assert result == {"cancelled": True, "operation": "delete_record"}
    assert client.called == []
    assert len(ctx.asked) == 1


def test_dispatch_async_denies_headless_destructive_invocation():
    class _DeleteClient:
        async def delete_record(self):
            raise AssertionError("destructive method must not run")

    result = anyio.run(dispatch_async, _DeleteClient(), "delete_record")

    assert result == {"cancelled": True, "operation": "delete_record"}
