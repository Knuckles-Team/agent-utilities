"""Regression test for the ``gateway-widgets`` extra (production log-flood fix).

Every widget in ``agent_utilities/gateway/widgets/`` that talks to a real
service does an IN-PROCESS import of that service's connector package inside
``fetch_data()`` (e.g. ``caddy.py``: ``from caddy_mcp.api_client import Api as
CaddyApi``). Before the ``gateway-widgets`` extra existed, none of these
connector packages were declared anywhere, so every configured widget logged
``ModuleNotFoundError: No module named '<connector>_mcp'`` on every
``WidgetAggregator`` cache-refresh cycle (``gateway/aggregator.py``,
``_cache_ttl = 10.0``) — this is the bug this extra + the widget import-alias
fixes resolve.

This test statically extracts each widget's connector import (module +
imported symbol) and actually performs it, so a regression in either the
extra's version floors OR a widget's import statement (e.g. importing a
class name the connector package does not export — the second, subtler bug
found alongside the missing-dependency one; see the ``gateway-widgets``
comment in ``pyproject.toml``) fails this test instead of silently flooding
production logs again.

Skips (not failures) when ``gateway-widgets`` isn't installed in the current
environment (``python3 scripts/uv_workspace.py run --extra serving ...`` or
``--extra gateway-widgets`` installs it) and xfails the small number of
widgets with a KNOWN, separately-tracked breakage that a dependency
declaration alone cannot fix (see ``KNOWN_BROKEN`` below).
"""

from __future__ import annotations

import importlib

import pytest

# widget module name -> (connector module, imported symbol)
# Mirrors the actual `from <module> import <symbol>` line inside each
# widget's fetch_data(). Kept as literal data (not re-parsed via ast) so this
# test also catches a widget import silently drifting out of sync with what
# is actually shipped.
WIDGET_CONNECTOR_IMPORTS: dict[str, tuple[str, str]] = {
    "ansible_tower": ("ansible_tower_mcp.api_client", "Api"),
    "caddy": ("caddy_mcp.api_client", "Api"),
    "documentdb": ("documentdb_mcp.api_client", "DocumentDBApi"),
    "erpnext": ("erpnext_agent.api_client", "Api"),
    "github": ("github_agent.api_client", "Api"),
    "gitlab": ("gitlab_api.api_client", "Api"),
    "home_assistant": ("home_assistant_agent.api_client", "HomeAssistantApi"),
    "jellyfin": ("jellyfin_mcp.api_client", "Api"),
    "keycloak": ("keycloak_agent.api_client", "Api"),
    "langfuse": ("langfuse_agent.api_client", "LangfuseApi"),
    "listmonk": ("listmonk_api.api_client", "ListmonkAPI"),
    "mattermost": ("mattermost_mcp.api_client", "Api"),
    "mealie": ("mealie_mcp.api_client", "Api"),
    "microsoft": ("microsoft_agent.api_client", "MicrosoftGraphApi"),
    "nextcloud": ("nextcloud_agent.api_client", "NextcloudAPI"),
    "openbao": ("openbao_mcp.api_client", "Api"),
    "owncast": ("owncast_agent.api_client", "OwncastApi"),
    "plane": ("plane_agent.api_client", "Api"),
    "portainer": ("portainer_agent.api_client", "PortainerApi"),
    "postiz": ("postiz_agent.api_client", "PostizApi"),
    "qbittorrent": ("qbittorrent_agent.api_client", "QbittorrentApi"),
    "servicenow": ("servicenow_api.api_client", "Api"),
    "technitium": ("technitium_dns_mcp.api_client", "Api"),
    "twenty": ("twenty_mcp.api_client", "Api"),
    "uptime_kuma": ("uptime_kuma_agent.auth", "get_client"),
    "wger": ("wger_agent.api_client", "WgerApi"),
}

# Widgets whose import is declared correctly above but is KNOWN to still fail
# for a reason a pyproject.toml dependency declaration cannot fix. See the
# `gateway-widgets` extra's comment in pyproject.toml for the full write-up.
KNOWN_BROKEN: dict[str, str] = {
    # PyPI's portainer-agent (currently 1.1.0, the newest published) imports
    # `agent_utilities.http`, which no longer exists (renamed to
    # `agent_utilities.httpsupport`). The local sibling checkout
    # (agent-packages/agents/portainer-agent, 2.1.0) already carries the fix,
    # but that version is not yet published, and PyPI has nothing newer.
    "portainer": "portainer-agent 1.1.0 (PyPI) imports the removed "
    "agent_utilities.http module; fixed in the unpublished 2.1.0",
}

# Widgets whose connector package has no `api_client` (or equivalent) module
# at ANY version, local or published — a real widget bug, not a missing
# dependency. Declared here only so this test documents them; they are
# deliberately absent from WIDGET_CONNECTOR_IMPORTS above.
NO_API_CLIENT_MODULE = frozenset(
    {
        "arr",
        "atlassian",
        "container_manager",
        "media_downloader",
        "repository_manager",
        "tunnel_manager",
        "vector_db",
    }
)

# Widgets whose connector package does not exist locally or on PyPI at all.
NO_CONNECTOR_PACKAGE = frozenset({"ear", "sentry", "zulip"})


def _connector_installed() -> bool:
    """True when the ``gateway-widgets`` extra's packages are importable."""
    try:
        importlib.import_module("caddy_mcp")
    except ModuleNotFoundError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _connector_installed(),
    reason="gateway-widgets extra not installed in this environment "
    "(run via `python3 scripts/uv_workspace.py run --extra serving ...` "
    "or `--extra gateway-widgets`)",
)


@pytest.mark.parametrize(
    "widget_name",
    sorted(WIDGET_CONNECTOR_IMPORTS),
)
def test_widget_connector_import_resolves(widget_name: str) -> None:
    """The widget's declared connector module + symbol actually import."""
    module_name, symbol = WIDGET_CONNECTOR_IMPORTS[widget_name]
    if widget_name in KNOWN_BROKEN:
        pytest.xfail(KNOWN_BROKEN[widget_name])

    module = importlib.import_module(module_name)
    assert hasattr(module, symbol), (
        f"{widget_name}.py imports `{symbol}` from `{module_name}`, but that "
        f"module does not export it -- this is exactly the class-name-"
        f"mismatch bug the `Api as {symbol}` alias fix corrected for the "
        f"other widgets; check {module_name}'s actual public API."
    )


def test_widget_connector_inventory_is_exhaustive() -> None:
    """Every widget file with a real connector import is accounted for.

    Guards against a new widget being added with an in-process connector
    import that nobody declared in `gateway-widgets` (the original bug).
    """
    import inspect

    from agent_utilities.gateway import widgets as widgets_pkg

    package_dir = inspect.getfile(widgets_pkg).rsplit("/", 1)[0]
    import os

    accounted_for = (
        set(WIDGET_CONNECTOR_IMPORTS)
        | KNOWN_BROKEN.keys()
        | NO_API_CLIENT_MODULE
        | NO_CONNECTOR_PACKAGE
    )
    # Widgets that talk over plain HTTP (self._http_client) with no
    # in-process connector-package import at all -- nothing to declare.
    no_import_widgets = {
        "archivebox",
        "audio_transcriber",
        "data_science",
        "emerald_exchange",
        "genius_agent",
        "gitlab",  # aliased to gitlab_api above via WIDGET_CONNECTOR_IMPORTS
        "google_workspace",
        "legal_peripherals",
        "lgtm",
        "listmonk",  # aliased above
        "media_downloader",  # NO_API_CLIENT_MODULE
        "ollama",
        "repository_manager",  # NO_API_CLIENT_MODULE
        "scholarx",
        "searxng",
        "servicenow",  # aliased above
        "stirlingpdf",
        "systems_manager",
        "teleport",
        "tunnel_manager",  # NO_API_CLIENT_MODULE
    }

    for filename in sorted(os.listdir(package_dir)):
        if not filename.endswith(".py") or filename in {"__init__.py", "base.py"}:
            continue
        widget_name = filename[: -len(".py")]
        with open(os.path.join(package_dir, filename), encoding="utf-8") as fh:
            source = fh.read()
        has_connector_import = ".api_client import" in source or (
            "from " in source and ".auth import get_client" in source
        )
        if not has_connector_import:
            continue
        assert widget_name in accounted_for or widget_name in no_import_widgets, (
            f"{filename} has an in-process connector import that this test "
            "does not account for -- add it to WIDGET_CONNECTOR_IMPORTS (and "
            "the `gateway-widgets` extra in pyproject.toml) or to one of the "
            "documented exclusion sets."
        )
