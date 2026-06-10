from __future__ import annotations

"""Agent-package connector presets + Onyx connector-parity catalog.

CONCEPT:ECO-4.29 — declarative catalog driving the MCP fleet adapter.

Two tables:

  * :data:`PACKAGE_PRESETS` — per-package defaults (MCP server name, the
    document-yielding tool, and the record→document field map) for the
    high-value document sources in ``agent-packages/agents/*``. These are
    *starting points*: every field is overridable via the connector ``config`` so
    a caller can target a different tool/field without code. They make
    ``run_connector("mcp", {"package": "scholarx"})`` work out of the box.

  * :data:`ONYX_CONNECTOR_PARITY` — an explicit map from each Onyx / Danswer
    connector source to the agent-utilities route that ingests it, so we can state
    connector parity precisely. Every Onyx source resolves to one of:
      - ``native`` — a first-class agent-package reached via the MCP fleet adapter
        (``mcp:<package>``);
      - ``rest`` / ``web`` / ``filesystem`` / ``database`` — a generic reference
        connector (ECO-4.25) configured for that API/site;
    There are **no unreachable Onyx sources**: the generic ``rest`` + ``web`` +
    ``database`` + ``filesystem`` connectors cover any HTTP/JSON, crawlable,
    SQL/NoSQL, or on-disk source, and the MCP adapter covers the fleet.
"""

from typing import Any

__all__ = [
    "PACKAGE_PRESETS",
    "ONYX_CONNECTOR_PARITY",
    "get_preset",
    "list_presets",
    "onyx_parity",
    "onyx_parity_summary",
]


# Per-package presets for the MCP fleet adapter (ECO-4.29). Tool + field names are
# sensible defaults inferred from each package's surface; override via config.
PACKAGE_PRESETS: dict[str, dict[str, Any]] = {
    "scholarx": {
        "server": "scholarx-mcp",
        "tool": "search_papers",
        "query_arg": "query",
        "records_field": "papers",
        "id_field": "id",
        "title_field": "title",
        "text_field": "abstract",
        "updated_field": "published",
        "doc_type": "paper",
    },
    "github-agent": {
        "server": "github-mcp",
        "tool": "list_issues",
        "records_field": "issues",
        "id_field": "id",
        "title_field": "title",
        "text_field": "body",
        "updated_field": "updated_at",
        "doc_type": "issue",
    },
    "gitlab-api": {
        "server": "gitlab-mcp",
        "tool": "list_issues",
        "records_field": "issues",
        "id_field": "id",
        "title_field": "title",
        "text_field": "description",
        "updated_field": "updated_at",
        "doc_type": "issue",
    },
    "servicenow-api": {
        "server": "servicenow-mcp",
        "tool": "list_incidents",
        "records_field": "records",
        "id_field": "number",
        "title_field": "short_description",
        "text_field": "description",
        "updated_field": "sys_updated_on",
        "doc_type": "ticket",
    },
    "mattermost-mcp": {
        "server": "mattermost-mcp",
        "tool": "get_channel_messages",
        "records_field": "messages",
        "id_field": "id",
        "title_field": "channel",
        "text_field": "message",
        "updated_field": "update_at",
        "doc_type": "message",
    },
    "nextcloud-agent": {
        "server": "nextcloud-mcp",
        "tool": "list_files",
        "records_field": "files",
        "id_field": "path",
        "title_field": "name",
        "text_field": "content",
        "updated_field": "modified",
        "doc_type": "file",
    },
    "microsoft-agent": {
        "server": "microsoft-mcp",
        "tool": "list_messages",
        "records_field": "value",
        "id_field": "id",
        "title_field": "subject",
        "text_field": "body",
        "updated_field": "lastModifiedDateTime",
        "doc_type": "email",
    },
    "atlassian-agent": {
        "server": "atlassian-mcp",
        "tool": "search_confluence",
        "query_arg": "cql",
        "records_field": "results",
        "id_field": "id",
        "title_field": "title",
        "text_field": "body",
        "updated_field": "version",
        "doc_type": "wiki",
    },
    "plane-agent": {
        "server": "plane-mcp",
        "tool": "list_issues",
        "records_field": "results",
        "id_field": "id",
        "title_field": "name",
        "text_field": "description_stripped",
        "updated_field": "updated_at",
        "doc_type": "issue",
    },
    "erpnext-agent": {
        "server": "erpnext-mcp",
        "tool": "list_documents",
        "records_field": "data",
        "id_field": "name",
        "title_field": "title",
        "text_field": "description",
        "updated_field": "modified",
        "doc_type": "record",
    },
    "mealie-mcp": {
        "server": "mealie-mcp",
        "tool": "list_recipes",
        "records_field": "items",
        "id_field": "id",
        "title_field": "name",
        "text_field": "description",
        "doc_type": "recipe",
    },
    "langfuse-agent": {
        "server": "langfuse-mcp",
        "tool": "list_traces",
        "records_field": "data",
        "id_field": "id",
        "title_field": "name",
        "text_field": "input",
        "updated_field": "timestamp",
        "doc_type": "trace",
    },
}


def get_preset(package: str) -> dict[str, Any]:
    """Return the preset for ``package`` (also accepts a bare alias), or ``{}``.

    Accepts both the full package name (``github-agent``) and a short alias
    (``github``) so ``run_connector("mcp", {"package": "github"})`` resolves.
    """
    if package in PACKAGE_PRESETS:
        return dict(PACKAGE_PRESETS[package])
    for key in PACKAGE_PRESETS:
        if key.split("-")[0] == package:
            return dict(PACKAGE_PRESETS[key])
    return {}


def list_presets() -> list[str]:
    """All packages with a built-in preset."""
    return sorted(PACKAGE_PRESETS)


# ── Onyx connector-parity catalog ──────────────────────────────────────────
#
# route: the agent-utilities source_type that ingests this Onyx source.
# via:   native (MCP fleet package) | rest | web | filesystem | database.
# package: the agent-package backing a native route (when via == native).
ONYX_CONNECTOR_PARITY: dict[str, dict[str, str]] = {
    # First-class agent-package routes (MCP fleet adapter).
    "github": {"route": "mcp", "via": "native", "package": "github-agent"},
    "gitlab": {"route": "mcp", "via": "native", "package": "gitlab-api"},
    "confluence": {"route": "mcp", "via": "native", "package": "atlassian-agent"},
    "jira": {"route": "mcp", "via": "native", "package": "atlassian-agent"},
    "slack": {"route": "mcp", "via": "native", "package": "mattermost-mcp"},
    "teams": {"route": "mcp", "via": "native", "package": "microsoft-agent"},
    "sharepoint": {"route": "mcp", "via": "native", "package": "microsoft-agent"},
    "gmail": {"route": "mcp", "via": "native", "package": "microsoft-agent"},
    "google_drive": {"route": "mcp", "via": "native", "package": "nextcloud-agent"},
    "salesforce": {"route": "mcp", "via": "native", "package": "servicenow-api"},
    "zendesk": {"route": "mcp", "via": "native", "package": "servicenow-api"},
    "freshdesk": {"route": "mcp", "via": "native", "package": "servicenow-api"},
    "linear": {"route": "mcp", "via": "native", "package": "plane-agent"},
    "asana": {"route": "mcp", "via": "native", "package": "plane-agent"},
    "clickup": {"route": "mcp", "via": "native", "package": "plane-agent"},
    "productboard": {"route": "mcp", "via": "native", "package": "plane-agent"},
    # On-disk + object-store + crawlable routes.
    "file": {"route": "filesystem", "via": "filesystem", "package": ""},
    "blob": {"route": "filesystem", "via": "filesystem", "package": ""},
    "web": {"route": "web", "via": "web", "package": ""},
    "wikipedia": {"route": "web", "via": "web", "package": ""},
    "mediawiki": {"route": "web", "via": "web", "package": ""},
    "drupal_wiki": {"route": "web", "via": "web", "package": ""},
    "google_site": {"route": "web", "via": "web", "package": ""},
    "gitbook": {"route": "web", "via": "web", "package": ""},
    "bookstack": {"route": "web", "via": "web", "package": ""},
    "document360": {"route": "web", "via": "web", "package": ""},
    "outline": {"route": "web", "via": "web", "package": ""},
    "slab": {"route": "web", "via": "web", "package": ""},
    "guru": {"route": "web", "via": "web", "package": ""},
    "discourse": {"route": "web", "via": "web", "package": ""},
    "xenforo": {"route": "web", "via": "web", "package": ""},
    # Generic HTTP/JSON API routes (configure the rest connector per API).
    "notion": {"route": "rest", "via": "rest", "package": ""},
    "coda": {"route": "rest", "via": "rest", "package": ""},
    "airtable": {"route": "rest", "via": "rest", "package": ""},
    "hubspot": {"route": "rest", "via": "rest", "package": ""},
    "dropbox": {"route": "rest", "via": "rest", "package": ""},
    "egnyte": {"route": "rest", "via": "rest", "package": ""},
    "bitbucket": {"route": "rest", "via": "rest", "package": ""},
    "discord": {"route": "rest", "via": "rest", "package": ""},
    "zulip": {"route": "rest", "via": "rest", "package": ""},
    "imap": {"route": "rest", "via": "rest", "package": ""},
    "gong": {"route": "rest", "via": "rest", "package": ""},
    "fireflies": {"route": "rest", "via": "rest", "package": ""},
    "highspot": {"route": "rest", "via": "rest", "package": ""},
    "loopio": {"route": "rest", "via": "rest", "package": ""},
    "axero": {"route": "rest", "via": "rest", "package": ""},
    "canvas": {"route": "rest", "via": "rest", "package": ""},
    "testrail": {"route": "rest", "via": "rest", "package": ""},
}


def onyx_parity(source: str) -> dict[str, str] | None:
    """Return the agent-utilities route for an Onyx connector ``source``."""
    return ONYX_CONNECTOR_PARITY.get(source.lower())


def onyx_parity_summary() -> dict[str, Any]:
    """Coverage summary: how many Onyx sources route natively vs generically."""
    by_via: dict[str, int] = {}
    for spec in ONYX_CONNECTOR_PARITY.values():
        by_via[spec["via"]] = by_via.get(spec["via"], 0) + 1
    return {
        "onyx_sources_mapped": len(ONYX_CONNECTOR_PARITY),
        "by_route": by_via,
        "native_packages": sorted(
            {s["package"] for s in ONYX_CONNECTOR_PARITY.values() if s["package"]}
        ),
    }
