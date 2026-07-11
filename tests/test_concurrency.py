"""run_blocking offloads sync work off the event loop."""

from __future__ import annotations

import threading

import anyio
import pytest

from agent_utilities.mcp.concurrency import _wrap_data_kwargs, run_blocking
from agent_utilities.mcp_utilities import run_blocking as run_blocking_reexport


def test_run_blocking_runs_in_worker_thread_and_returns():
    async def main():
        loop_thread = threading.current_thread().name

        def work(a, b, *, c):
            return (threading.current_thread().name, a + b + c)

        worker_thread, total = await run_blocking(work, 1, 2, c=3)
        assert total == 6
        assert worker_thread != loop_thread  # ran off the event loop thread

    anyio.run(main)


def test_run_blocking_propagates_exceptions():
    async def main():
        def boom():
            raise ValueError("nope")

        with pytest.raises(ValueError, match="nope"):
            await run_blocking(boom)

    anyio.run(main)


def test_reexport_is_same_callable():
    assert run_blocking_reexport is run_blocking


class _Api:
    def create_work_item(self, project_id, data):  # takes a data dict
        return {"project_id": project_id, "data": data}

    def update_work_item(self, project_id, work_item_id, data):
        return (project_id, work_item_id, data)

    def list_work_items(self, project_id, cursor=None):  # flat, no data param
        return (project_id, cursor)

    def jira_create_issue(self, update_history=None, payload=None):  # payload body
        return {"update_history": update_history, "payload": payload}

    def flexible(self, project_id, **kw):  # VAR_KEYWORD accepts anything
        return kw


def test_wrap_data_folds_stray_fields_into_data():
    api = _Api()
    # LLM passes flat REST-shaped fields -> folded under `data`
    assert _wrap_data_kwargs(
        api.create_work_item, (), {"project_id": "P", "name": "T", "description": "D"}
    ) == {"project_id": "P", "data": {"name": "T", "description": "D"}}
    # two path params kept out of `data`
    assert _wrap_data_kwargs(
        api.update_work_item, (), {"project_id": "P", "work_item_id": "W", "state": "x"}
    ) == {"project_id": "P", "work_item_id": "W", "data": {"state": "x"}}
    # payload-convention client (atlassian codegen): stray fields -> `payload`
    assert _wrap_data_kwargs(
        api.jira_create_issue, (), {"fields": {"summary": "S"}}
    ) == {"payload": {"fields": {"summary": "S"}}}


def test_wrap_data_is_noop_when_not_applicable():
    api = _Api()
    # already-correct `data` passthrough
    payload = {"project_id": "P", "data": {"name": "T"}}
    assert _wrap_data_kwargs(api.create_work_item, (), dict(payload)) == payload
    # method without a `data` param — strict no-op
    flat = {"project_id": "P", "cursor": "c"}
    assert _wrap_data_kwargs(api.list_work_items, (), dict(flat)) == flat
    # VAR_KEYWORD method — no-op (accepts the fields itself)
    var = {"project_id": "P", "x": 1}
    assert _wrap_data_kwargs(api.flexible, (), dict(var)) == var
    # positional args present — no-op
    assert _wrap_data_kwargs(api.create_work_item, ("P",), {"name": "T"}) == {"name": "T"}


def test_run_blocking_applies_data_wrap_end_to_end():
    async def main():
        result = await run_blocking(
            _Api().create_work_item, project_id="P", name="T", description="D"
        )
        assert result == {"project_id": "P", "data": {"name": "T", "description": "D"}}

    anyio.run(main)
