"""Shared action-routing helpers for MCP servers.

Many MCP servers expose a single ``<service>_action`` tool that dispatches a
free-form ``action`` string to an underlying client. Without help, a typo or an
intuitive-but-wrong name (e.g. ``get_movies`` when the method is ``get_movie``)
yields an opaque "unknown action" error and there is no way to discover the valid
names. These helpers standardize three behaviours across the fleet:

1. **Discovery** — ``action`` in :data:`DISCOVERY_ACTIONS` returns the valid names.
2. **Aliasing** — intuitive plurals resolve to the real singular method, plus any
   explicit ``aliases`` mapping a server supplies.
3. **Did-you-mean** — an unknown action raises a rich error with close matches and
   a pointer to ``list_actions``.

Two entry points cover the two dispatch shapes in the fleet:

- :func:`dispatch` — for **getattr-dynamic** servers (``getattr(client, action)``).
  A full drop-in that introspects the client, resolves, calls, and errors richly.
- :func:`resolve_action` — for **explicit if/elif** servers. Call it at the top of
  the tool with the known action list; it returns the discovery payload, the
  canonical action string (which the existing if/elif then handles), or raises.

CONCEPT:ECO-4.0 — MCP Standardized Interfaces
"""

from __future__ import annotations

import difflib
from collections.abc import Callable, Iterable, Mapping
from typing import Any

#: Action strings that request the list of valid actions instead of executing one.
DISCOVERY_ACTIONS = ("list_actions", "help", "actions")


def public_actions(client: Any) -> list[str]:
    """Sorted public, callable attribute names on a client object."""
    return sorted(
        name
        for name in dir(client)
        if not name.startswith("_") and callable(getattr(client, name, None))
    )


def suggest(action: str, valid: Iterable[str], *, n: int = 3) -> list[str]:
    """Close matches for ``action`` among ``valid`` (difflib, cutoff 0.6)."""
    return difflib.get_close_matches(action, list(valid), n=n)


def _plural_candidates(action: str) -> list[str]:
    """Singular forms to try for an intuitive plural (get_movies -> get_movie)."""
    if not action.endswith("s"):
        return []
    candidates = [action[:-1]]
    if action.endswith("es"):
        candidates.append(action[:-2])
    return candidates


def canonicalize(
    action: str,
    valid: Iterable[str],
    *,
    aliases: Mapping[str, str] | None = None,
) -> str | None:
    """Resolve ``action`` to a name present in ``valid``, or ``None``.

    Tries, in order: the action itself, an explicit ``aliases`` entry, then
    plural->singular forms.
    """
    valid_set = set(valid)
    if action in valid_set:
        return action
    if aliases and action in aliases and aliases[action] in valid_set:
        return aliases[action]
    for candidate in _plural_candidates(action):
        if candidate in valid_set:
            return candidate
    return None


def unknown_action_error(
    action: str, valid: Iterable[str], *, target: str = ""
) -> ValueError:
    """Build a rich ValueError with did-you-mean hints and a discovery pointer."""
    valid = list(valid)
    matches = suggest(action, valid)
    hint = f" Did you mean: {', '.join(matches)}?" if matches else ""
    where = f" on {target}" if target else ""
    return ValueError(
        f"Unknown action '{action}'{where}.{hint} "
        f"Call with action='list_actions' to see all {len(valid)} available actions."
    )


def resolve_action(
    action: str,
    valid_actions: Iterable[str],
    *,
    aliases: Mapping[str, str] | None = None,
    service: str = "",
) -> str | dict:
    """Resolve an action for an explicit if/elif dispatcher.

    Returns a ``{"service", "actions"}`` discovery payload when ``action`` is a
    discovery keyword, the canonical action string otherwise, or raises
    :func:`unknown_action_error`. The caller's existing if/elif runs on the
    returned canonical string.
    """
    valid = list(valid_actions)
    if action in DISCOVERY_ACTIONS:
        return {"service": service, "actions": sorted(valid)}
    canonical = canonicalize(action, valid, aliases=aliases)
    if canonical is None:
        raise unknown_action_error(action, valid, target=service or "this tool")
    return canonical


def dispatch(
    client: Any,
    action: str,
    kwargs: Mapping[str, Any] | None = None,
    *,
    aliases: Mapping[str, str] | None = None,
    service: str = "",
    result_coercer: Callable[[Any], Any] | None = None,
) -> Any:
    """Resolve and execute ``action`` on a getattr-dynamic ``client``.

    Handles discovery, alias/plural resolution, the call, and rich errors. If
    ``result_coercer`` is given it is applied to the return value (e.g. to call
    ``.model_dump()``); otherwise the raw result is returned.
    """
    actions = public_actions(client)
    target = service or type(client).__name__
    if action in DISCOVERY_ACTIONS:
        return {"service": service or target, "actions": actions}

    canonical = canonicalize(action, actions, aliases=aliases)
    if canonical is None:
        raise unknown_action_error(action, actions, target=target)

    method = getattr(client, canonical)
    result = method(**(dict(kwargs) if kwargs else {}))
    if result_coercer is not None:
        return result_coercer(result)
    return result
