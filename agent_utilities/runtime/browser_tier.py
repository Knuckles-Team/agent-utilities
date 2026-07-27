"""CONCEPT:AU-ECO.toolkit.browser-agent-tier — Optional browser-agent tier for the developer-workspace runtime.

A pluggable :class:`BrowserDriver` lets the SWE agent issue ``BrowseAction``s (navigate a docs
page, reproduce a web bug) without dragging playwright/browsergym into the core edit→test
critical path. The default :class:`NullBrowserDriver` returns a clear "not provisioned"
observation, so the action surface exists everywhere and a real driver (e.g. the agent-browser
skill) is wired in only where browsing is actually needed. ``BrowseAction`` is gated by
``ActionPolicy`` as ``workspace.browse`` (OS-5.24).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .events import BrowseAction, BrowserObservation


@runtime_checkable
class BrowserDriver(Protocol):
    async def browse(self, action: BrowseAction) -> BrowserObservation: ...


class NullBrowserDriver:
    """The always-available floor: advertises the tier but performs no browsing."""

    name = "null"

    async def browse(self, action: BrowseAction) -> BrowserObservation:
        return BrowserObservation(
            url=action.url,
            error="browser tier not provisioned (attach a BrowserDriver to enable ECO-4.44)",
        )
