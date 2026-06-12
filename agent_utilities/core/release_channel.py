#!/usr/bin/python
from __future__ import annotations

"""Release-Channel System (CONCEPT:OS-5.13).

Palantir AIP ships capabilities on release *tracks* (stable / beta / edge) so a
component can be exposed to early adopters without affecting the stable surface.
agent-utilities synthesizes agents, skills, tools, and workflows dynamically —
so the analogous need is to *gate* a synthesized/registered component by channel:
an ``edge`` agent should be discoverable on the ``edge`` (and ``beta``) channel,
but invisible on the default ``stable`` channel.

This module provides:

  - :class:`ReleaseChannel` — the ordered stable < beta < edge enum.
  - :func:`active_channel` — resolves the active channel from
    ``AGENT_UTILITIES_RELEASE_CHANNEL`` (env), falling back to config and then to
    ``stable`` (the conservative default).
  - :func:`channel_visible` — the gate: is a component tagged ``component_channel``
    visible on the ``active`` channel? An ``edge`` component is visible only on
    ``edge``; a ``beta`` component on ``beta`` or ``edge``; ``stable`` everywhere.
  - :func:`release_channel` — a decorator stamping ``__release_channel__`` on a
    function/class so a registry/discovery pass can read + gate it.
  - :class:`ChannelRegistry` — an in-memory registry that filters registered
    components by the active channel (used by discovery paths).

The gate is wired into the live KG-driven specialist designation path
(``graph/routing/enrichers/capability_designation.py``): callable nodes tagged
with a ``release_channel`` property that is not visible on the active channel are
excluded from the designation index — so channel actually filters what an agent
can be routed to, not merely stored as metadata.
"""

import logging
from collections.abc import Callable
from enum import IntEnum
from typing import Any, TypeVar

from agent_utilities.core.config import setting

logger = logging.getLogger(__name__)

ENV_RELEASE_CHANNEL = "AGENT_UTILITIES_RELEASE_CHANNEL"
RELEASE_CHANNEL_ATTR = "__release_channel__"

T = TypeVar("T")


class ReleaseChannel(IntEnum):
    """Ordered release tracks (stable < beta < edge). CONCEPT:OS-5.13.

    Higher channels are *more* permissive: running on ``edge`` exposes
    everything; running on ``stable`` (the default) exposes only stable
    components. A component's own channel is the *minimum* channel on which it
    becomes visible.
    """

    STABLE = 0
    BETA = 1
    EDGE = 2

    @classmethod
    def parse(cls, value: Any, default: ReleaseChannel | None = None) -> ReleaseChannel:
        """Coerce a string/int/enum into a :class:`ReleaseChannel`.

        Unknown values fall back to ``default`` (or ``STABLE``) — never raises,
        so a typo in config can never crash discovery.
        """
        if isinstance(value, ReleaseChannel):
            return value
        if isinstance(value, bool):
            return cls.STABLE
        if isinstance(value, int):
            try:
                return cls(value)
            except ValueError:
                return default or cls.STABLE
        if isinstance(value, str):
            key = value.strip().upper()
            # Common aliases.
            aliases = {
                "GA": "STABLE",
                "PROD": "STABLE",
                "PRODUCTION": "STABLE",
                "PREVIEW": "BETA",
                "CANARY": "EDGE",
                "NIGHTLY": "EDGE",
                "EXPERIMENTAL": "EDGE",
            }
            key = aliases.get(key, key)
            try:
                return cls[key]
            except KeyError:
                return default or cls.STABLE
        return default or cls.STABLE


# Cached active channel (reset via reset_active_channel for tests / reconfig).
_ACTIVE: ReleaseChannel | None = None


def active_channel(refresh: bool = False) -> ReleaseChannel:
    """Resolve the active release channel. CONCEPT:OS-5.13.

    Resolution order: ``AGENT_UTILITIES_RELEASE_CHANNEL`` env → ``release_channel``
    in the loaded config (best-effort) → ``stable``. Cached after first read;
    pass ``refresh=True`` (or call :func:`reset_active_channel`) to re-resolve.
    """
    global _ACTIVE
    if _ACTIVE is not None and not refresh:
        return _ACTIVE

    raw = setting(ENV_RELEASE_CHANNEL)
    if not raw:
        raw = _channel_from_config()
    _ACTIVE = ReleaseChannel.parse(raw, default=ReleaseChannel.STABLE)
    logger.debug("Active release channel resolved: %s", _ACTIVE.name)
    return _ACTIVE


def _channel_from_config() -> str | None:
    """Best-effort read of ``release_channel`` from ``config.json`` (optional)."""
    try:
        import json

        from agent_utilities.core.paths import config_dir

        path = config_dir() / "config.json"
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        val = data.get("release_channel") if isinstance(data, dict) else None
        return str(val) if val else None
    except Exception:  # noqa: BLE001 — config optional; default to stable
        return None


def reset_active_channel() -> None:
    """Clear the cached active channel (re-resolved on next call)."""
    global _ACTIVE
    _ACTIVE = None


def set_active_channel(channel: Any) -> ReleaseChannel:
    """Override the active channel in-process (used by tests / runtime reconfig)."""
    global _ACTIVE
    _ACTIVE = ReleaseChannel.parse(channel, default=ReleaseChannel.STABLE)
    return _ACTIVE


def channel_visible(
    component_channel: Any,
    active: ReleaseChannel | None = None,
) -> bool:
    """Is a component on ``component_channel`` visible on the ``active`` channel?

    A component is visible iff ``active >= component_channel`` — i.e. an ``edge``
    component is hidden on ``stable``/``beta`` and visible only on ``edge``; a
    ``stable`` component is visible everywhere. CONCEPT:OS-5.13.
    """
    comp = ReleaseChannel.parse(component_channel, default=ReleaseChannel.STABLE)
    act = active if active is not None else active_channel()
    return act >= comp


def release_channel(channel: Any) -> Callable[[T], T]:
    """Decorator stamping a component's release channel. CONCEPT:OS-5.13.

    Usage::

        @release_channel("edge")
        class ExperimentalSkill: ...

    A discovery/registration pass reads :data:`RELEASE_CHANNEL_ATTR` (or calls
    :func:`get_component_channel`) and gates the component via
    :func:`channel_visible`.
    """
    resolved = ReleaseChannel.parse(channel, default=ReleaseChannel.STABLE)

    def _decorator(obj: T) -> T:
        try:
            setattr(obj, RELEASE_CHANNEL_ATTR, resolved)
        except (AttributeError, TypeError):
            logger.debug("cannot stamp release channel on %r", obj)
        return obj

    return _decorator


def get_component_channel(
    obj: Any, default: Any = ReleaseChannel.STABLE
) -> ReleaseChannel:
    """Read a component's release channel from its decorator stamp or metadata.

    Checks (in order): the :data:`RELEASE_CHANNEL_ATTR` decorator stamp, a
    ``release_channel`` attribute, and — for dict-shaped node props — the
    ``release_channel`` / ``channel`` keys. Defaults to ``stable``.
    """
    if isinstance(obj, dict):
        raw = obj.get("release_channel") or obj.get("channel")
        return ReleaseChannel.parse(raw, default=ReleaseChannel.parse(default))
    raw = getattr(obj, RELEASE_CHANNEL_ATTR, None)
    if raw is None:
        raw = getattr(obj, "release_channel", None)
    return ReleaseChannel.parse(raw, default=ReleaseChannel.parse(default))


def component_visible(obj: Any, active: ReleaseChannel | None = None) -> bool:
    """Convenience: is ``obj`` (decorated/dict/attr) visible on ``active``?"""
    return channel_visible(get_component_channel(obj), active)


class ChannelRegistry:
    """An in-memory registry that gates components by the active channel. CONCEPT:OS-5.13.

    Components register with a name + (decorated/explicit) channel; :meth:`active`
    returns only those visible on the active channel, so discovery surfaces an
    ``edge`` component only when running on ``edge``.
    """

    def __init__(self) -> None:
        self._components: dict[str, tuple[Any, ReleaseChannel]] = {}

    def register(self, name: str, component: Any, channel: Any = None) -> None:
        """Register ``component`` under ``name`` at ``channel`` (or its stamp)."""
        ch = (
            ReleaseChannel.parse(channel)
            if channel is not None
            else get_component_channel(component)
        )
        self._components[name] = (component, ch)

    def channel_of(self, name: str) -> ReleaseChannel | None:
        entry = self._components.get(name)
        return entry[1] if entry else None

    def is_visible(self, name: str, active: ReleaseChannel | None = None) -> bool:
        """Whether the named component is visible on the active channel."""
        entry = self._components.get(name)
        if entry is None:
            return False
        return channel_visible(entry[1], active)

    def active(self, active: ReleaseChannel | None = None) -> dict[str, Any]:
        """Return ``{name: component}`` for components visible on the channel."""
        act = active if active is not None else active_channel()
        return {
            name: comp
            for name, (comp, ch) in self._components.items()
            if channel_visible(ch, act)
        }

    def all(self) -> dict[str, Any]:
        """Return every registered component regardless of channel."""
        return {name: comp for name, (comp, _ch) in self._components.items()}

    def __len__(self) -> int:
        return len(self._components)
