"""CONCEPT:ECO-4.44 — optional browser tier: pluggable driver, Null floor, policy-gated."""

from __future__ import annotations

from agent_utilities.runtime import DevWorkspace, LocalWorkspace
from agent_utilities.runtime.browser_tier import NullBrowserDriver
from agent_utilities.runtime.events import (
    ACTION_ADAPTER,
    BrowseAction,
    BrowserObservation,
    mutating_action_name,
)


def test_browse_action_in_union_and_policy_mapping():
    back = ACTION_ADAPTER.validate_python({"kind": "browse", "url": "http://x"})
    assert isinstance(back, BrowseAction)
    assert mutating_action_name(BrowseAction(url="x")) == "workspace.browse"


async def test_null_driver_reports_not_provisioned():
    async with DevWorkspace(LocalWorkspace(), run_id="b1") as ws:
        obs = await ws.act(BrowseAction(url="http://example.com"))
        assert isinstance(obs, BrowserObservation)
        assert "not provisioned" in obs.error


async def test_attached_driver_handles_browse():
    class _FakeDriver:
        async def browse(self, action):
            return BrowserObservation(
                url=action.url, status=200, title="OK", text="hello world"
            )

    ws = DevWorkspace(LocalWorkspace(), run_id="b2", browser=_FakeDriver())
    async with ws:
        obs = await ws.act(BrowseAction(url="http://example.com"))
        assert obs.status == 200
        assert obs.text == "hello world"


async def test_null_driver_direct():
    obs = await NullBrowserDriver().browse(BrowseAction(url="u"))
    assert obs.url == "u" and obs.error
