from __future__ import annotations

"""Universal MCP-tool ingestion source — any fleet MCP server as a KG source.

CONCEPT:AU-KG.ingest.mcp-tool-connector — MCP Tool Source Connector

One declarative adapter that turns **any** MCP server's paginated, record-listing
tool into a Knowledge-Graph document source — sql-mcp, objectstore-mcp,
servicenow-api, and the rest of the ~58-server fleet — replacing the idea of
hand-writing per-database/per-SaaS native drivers. Where :mod:`mcp_package`
(ECO-4.29) targets *search-shaped* fleet tools with a fixed one-call contract,
this connector models the full ingestion-source contract:

  * **Action-routed fleet envelopes** — the fleet convention is one tool taking
    ``action`` + ``params_json`` (a JSON string); ``params_style="json"`` encodes
    the declarative ``params`` dict accordingly (``params_style="args"`` spreads
    them as plain tool arguments for non-fleet servers).
  * **Pagination** — ``cursor`` (token in the response or keyset from the last
    record), ``page`` (page-number or offset with exhaustion detection), or
    ``none``; with ``max_pages`` / ``max_records`` backstops.
  * **Session lifecycle** — one MCP client session per run (``load``) or per
    batch (``poll``), reused across every page and detail call, closed cleanly.
  * **Incremental poll** — an ``updated_since_param`` binds the prior checkpoint
    watermark into the tool params so re-polls are server-side deltas
    (ECO-4.26); an in-memory ``updated_field`` filter is the belt to that brace.
  * **Two-phase list+get** — an optional ``detail`` call fetches each record's
    body (objectstore ``objects get``, attachment downloads, …) inside the same
    session, with ``{field}`` templating from the listed record.
  * **Permission seam** — ``acl_*`` field maps project ACL-ish record fields
    onto :class:`ExternalAccess`, feeding the ECO-4.28 permission sync.
  * **SQL table sweeps** — a ``sql_table`` block bootstraps a keyset-paginated
    ``SELECT`` against sql-mcp, discovering columns via ``sql_schema`` when not
    given, so "ingest this table" is one config dict.

Transport resolution (first match wins): an injected ``client`` target (an
in-process ``FastMCP`` instance in tests), an explicit ``url``, an explicit
``command``/``args``/``env`` stdio spec, or a ``server`` name resolved through the
workspace ``mcp_config.json`` (the same source the multiplexer uses). No package
import of any fleet repo — runtime MCP calls only.
"""

import json
import logging
import os
import re
from collections.abc import Iterator
from typing import Any

from ..base import (
    CheckpointedBatch,
    ConnectorCheckpoint,
    ExternalAccess,
    LoadConnector,
    PollConnector,
    SourceDocument,
)
from ..registry import register_source
from .mcp_package import _decode_tool_result, _load_mcp_config, _run_async
from .rest import _dig

logger = logging.getLogger(__name__)

__all__ = [
    "McpToolSourceConnector",
    "McpToolSourceError",
    "MCP_TOOL_PRESETS",
    "SOURCE_PRESET_PROVIDER_GROUP",
    "get_tool_preset",
    "list_tool_presets",
    "all_tool_presets",
    "reset_contributed_presets_cache",
    "call_tool_once",
]


class McpToolSourceError(RuntimeError):
    """A transport/tool failure while draining an MCP-backed source.

    Raised for session, tool-call, and pagination failures so the ingestion
    adaptor reports a typed, actionable drain error (CONCEPT:AU-KG.ingest.mcp-tool-connector) (config
    mistakes raise ``ValueError`` at build time, per the framework convention).
    """


# ── Recipe presets (data, not code) ─────────────────────────────────────────
#
# Named partial configs for the common fleet sources (CONCEPT:AU-KG.ingest.mcp-tool-connector), in the
# same spirit as ECO-4.29's PACKAGE_PRESETS: every key is overridable, and the
# caller extends a preset with the run-specific bits (table, bucket, params).
MCP_TOOL_PRESETS: dict[str, dict[str, Any]] = {
    # sql-mcp: declarative whole-table sweep. Extend with
    #   {"sql_table": {"table": "articles", "key_column": "id",
    #    "text_column": "body", "updated_column": "updated_at"}}
    # Columns are discovered via sql_schema when not listed explicitly.
    "sql-table": {
        "server": "sql-mcp",
        "tool": "sql_query",
        "action": "execute",
        "doc_type": "record",
    },
    # Harness-run traces from any fleet evolution/governance server (CONCEPT:AU-KG.retrieval.harness-grounding).
    # Grounds harness evolution in the WHOLE connector fleet, not a single benchmark
    # verifier — the cross-team evidence substrate HarnessX (arXiv:2606.14249) lacks.
    # Extend with {"server": "<your-mcp>", "params": {"status": "completed"}}.
    "harness-runs": {
        "server": "governance-api",
        "tool": "harness_runs",
        "action": "list",
        "records_path": "runs",
        "id_field": "run_id",
        "title_field": "harness_id",
        "updated_field": "completed_at",
        "pagination": "cursor",
        "cursor_param": "after_id",
        "cursor_path": "next_cursor",
        "more_path": "has_more",
        "doc_type": "harness_run",
    },
    # sql-mcp: hand-written SELECT with keyset pagination. Extend with
    #   {"params": {"sql": "SELECT id, title, body FROM t WHERE id > :after
    #                       ORDER BY id", "params": {"after": 0},
    #               "max_rows": 500},
    #    "cursor_record_field": "id", "text_field": "body"}
    "sql-query": {
        "server": "sql-mcp",
        "tool": "sql_query",
        "action": "execute",
        "pagination": "cursor",
        "cursor_param": "params.after",
        "more_path": "truncated",
        "doc_type": "record",
    },
    # objectstore-mcp: prefix sweep — paginated `objects list`, then a text-mode
    # size-capped `objects get` per key inside the same session. Extend with
    #   {"params": {"bucket": "docs", "prefix": "kb/", "max_keys": 200}}
    "objectstore-prefix": {
        "server": "objectstore-mcp",
        "tool": "objects",
        "action": "list",
        "records_path": "objects",
        "id_field": "key",
        "title_field": "key",
        "updated_field": "last_modified",
        "pagination": "cursor",
        "cursor_param": "token",
        "cursor_path": "next_token",
        "more_path": "truncated",
        "doc_type": "file",
        "detail": {
            "tool": "objects",
            "action": "get",
            "params": {"bucket": "{bucket}", "key": "{key}", "mode": "text"},
            "text_path": "content",
        },
    },
    # nextcloud-agent: configurable folder ingest — list a folder's files, then
    # read each into a Document/Chunk. Opt-in: nothing ingests until a folder is
    # configured. Extend with the folder(s) to index, e.g.
    #   {"params": {"path": "/Documents"}}
    "nextcloud-files": {
        "server": "nextcloud-agent",
        "tool": "nextcloud_files",
        "action": "list_files",
        "id_field": "path",
        "title_field": "name",
        "updated_field": "last_modified",
        "doc_type": "file",
        "detail": {
            "tool": "nextcloud_files",
            "action": "read_file",
            "params": {"path": "{path}"},
            "text_path": "content",
        },
    },
    # archivebox-api: every preserved web snapshot enumerated as a record. ArchiveBox
    # is the fast ingestion source for pages we've archived (no live re-crawl). List
    # via ``archivebox_core get_snapshots`` (DRF ``{"results": [...]}`` with
    # limit/offset paging); the snapshot ``url`` rides in metadata. ``extracted_text``
    # is used as the body when ArchiveBox inlines it (``with_archiveresults``); when
    # it doesn't, the snapshot's preserved body is retrieved robustly by
    # ``web_fetch.resolve_web_fetch(url, prefer="archivebox")`` in the sync path
    # (``_sync_archivebox``), which also fires research-paper extraction for archived
    # roundups. Delta = ``created_at__gte``; "pull all" is a full drain; restrict with
    # {"params": {"tag": "research"}}.
    "archivebox": {
        "server": "archivebox-api",
        "tool": "archivebox_core",
        "action": "get_snapshots",
        "params": {"with_archiveresults": True},
        "records_path": "results",
        "id_field": "abid",
        "title_field": "title",
        "text_field": "extracted_text",
        "updated_field": "modified_at",
        "pagination": "page",
        "page_kind": "offset",
        "page_param": "offset",
        "page_size_param": "limit",
        "page_size": 100,
        "doc_type": "webpage",
    },
    # mealie-mcp: recipes as documents. ``mealie_recipes`` is an action-routed
    # tool (action + params_json envelope), so the fleet-default params_style
    # "json" drives it directly. get_recipes returns Mealie's ``{"items": [...],
    # "page", "per_page", "total"}`` page envelope; descriptions are the body.
    # Extend with richer filters, e.g. {"params": {"categories": "dinner"}}.
    "mealie-recipes": {
        "server": "mealie-mcp",
        "tool": "mealie_recipes",
        "action": "get_recipes",
        "params_style": "json",
        "params": {"per_page": 50},
        "records_path": "items",
        "id_field": "slug",
        "title_field": "name",
        "text_field": "description",
        "updated_field": "updateAt",
        "doc_type": "record",
        "pagination": "page",
        "page_kind": "number",
        "page_param": "page",
        "page_size_param": "per_page",
        "page_size": 50,
        "start_page": 1,
    },
    # searxng-mcp: privacy-respecting metasearch — one `web_search` call,
    # each result (url/title/content snippet) becomes a document. Extend with
    #   {"params": {"query": "..."}}  (plus optional categories/engines/
    #    language/pageno — the tool takes plain arguments, hence "args").
    "searxng-search": {
        "server": "searxng-mcp",
        "tool": "web_search",
        "params_style": "args",
        "records_path": "results",
        "id_field": "url",
        "title_field": "title",
        "text_field": "content",
        "doc_type": "web",
    },
    # servicenow-api: any Table-API table via sysparm offset paging. Extend with
    #   {"params": {"table": "incident"},
    #    "updated_since_param": "sysparm_query"} as needed.
    "servicenow-table": {
        "server": "servicenow-mcp",
        "tool": "servicenow_table_api",
        "action": "get_table",
        "records_path": "result",
        "id_field": "sys_id",
        "title_field": "short_description",
        "text_field": "description",
        "updated_field": "sys_updated_on",
        "pagination": "page",
        "page_kind": "offset",
        "page_param": "sysparm_offset",
        "page_size_param": "sysparm_limit",
        "page_size": 100,
        "doc_type": "ticket",
    },
    # github-mcp: list repositories as Document records. ``github_repos`` with
    # action="list" returns ``{"status", "data": [Repository.model_dump()...]}``
    # (github_agent/mcp/mcp_repo.py — list → client.get_repositories()). Repository
    # fields are flat: id/name/full_name/description/updated_at
    # (github_response_models.py). Extend with the listing scope, e.g.
    #   {"params": {"org": "Knuckles-Team", "type": "all"}}
    "github-repos": {
        "server": "github-mcp",
        "tool": "github_repos",
        "action": "list",
        "records_path": "data",
        "id_field": "id",
        "title_field": "full_name",
        "text_field": "description",
        "updated_field": "updated_at",
        "doc_type": "repository",
    },
    # GitLab API-OBJECT ingestion (issues / MRs as documents) — the metadata side
    # of "assimilate all of GitLab" (CONCEPT:AU-KG.backend.declared-columns-so-schema). The resolved CODE graph comes
    # from the dedicated `gitlab` source-sync handler + `index_repository`, not from
    # this preset. `{project_id}` is bound by the caller's params.
    "gitlab-issues": {
        "server": "gitlab-api",
        "tool": "mcp_issues",
        "action": "list",
        "params": {"project_id": "{project_id}"},
        "records_path": "data",
        "id_field": "id",
        "title_field": "title",
        "text_field": "description",
        "updated_field": "updated_at",
        "doc_type": "ticket",
    },
    "gitlab-merge-requests": {
        "server": "gitlab-api",
        "tool": "mcp_merge_requests",
        "action": "list",
        "params": {"project_id": "{project_id}"},
        "records_path": "data",
        "id_field": "id",
        "title_field": "title",
        "text_field": "description",
        "updated_field": "updated_at",
        "doc_type": "merge_request",
    },
    # okta-mcp: list users (identities) as Document records. ``okta_users`` with
    # action="list" returns the API-client envelope ``{"data": [user...],
    # "count", "truncated", "next_cursor"}`` (okta_agent/api/api_client_base.py
    # envelope()/paginate(); the client auto-follows Okta's ``after`` cursor up to
    # max_items, so no connector-level pagination is needed). Okta user objects
    # carry a flat ``id`` and ``lastUpdated`` with login/email under ``profile``
    # (dotted field maps are supported). Extend with a filter, e.g.
    #   {"params": {"filter": "status eq \"ACTIVE\""}}
    "okta-users": {
        "server": "okta-mcp",
        "tool": "okta_users",
        "action": "list",
        "records_path": "data",
        "id_field": "id",
        "title_field": "profile.login",
        "text_field": "profile.email",
        "updated_field": "lastUpdated",
        "doc_type": "identity",
    },
    # keycloak-mcp: list realm users (identities) as Document records.
    # ``keycloak_agent_users`` with action="list_users" returns the Keycloak Admin
    # API's bare JSON array of UserRepresentations — so ``records_path`` stays ""
    # (the whole result is the list). Fields id/username/email are flat
    # (keycloak_agent/api/api_client_users.list_users → GET /admin/realms/{realm}/users).
    # Realm is required — extend with {"params": {"realm": "master"}}.
    "keycloak-users": {
        "server": "keycloak-mcp",
        "tool": "keycloak_agent_users",
        "action": "list_users",
        "id_field": "id",
        "title_field": "username",
        "text_field": "email",
        "doc_type": "identity",
    },
    # ── PulseLink open-web/social sources (CONCEPT:AU-ECO.connector.mcp-tool-connector) ─────────────────
    #
    # pulselink-mcp is the keyless-first reach server (sibling to scholarx). Its
    # ``pulse_search`` / ``pulse_list`` tools return a uniform
    # ``{documents: [{id,title,url,text,author,created_at,metrics,extra}],
    #    next_cursor}`` envelope, so every source is the SAME flat field map +
    # cursor pagination — only ``source`` (and channel/query) differs. Auth is
    # handled inside pulselink via the shared CredentialProvider (OS-5.38); these
    # presets carry NO secrets. Extend a search preset with the query, e.g.
    #   {"params": {"query": "agentic retrieval"}}
    # and a list preset with the channel, e.g. {"params": {"channel": "MachineLearning"}}.
    **{
        f"pulselink-{src}": {
            "server": "pulselink-mcp",
            "tool": "pulse_search",
            "params_style": "args",
            "params": {"source": src},
            "records_path": "documents",
            "id_field": "id",
            "title_field": "title",
            "text_field": "text",
            "updated_field": "created_at",
            "pagination": "cursor",
            "cursor_param": "cursor",
            "cursor_path": "next_cursor",
            "doc_type": "social_post" if src in ("x", "reddit") else "web",
        }
        for src in (
            "x",
            "reddit",
            "hackernews",
            "web",
            "news",
            "github",
            "exa",
            "bilibili",
            "xiaohongshu",
            "xueqiu",
        )
    },
    # YouTube: search returns metadata only; a detail phase pulls the transcript
    # per video via pulse_fetch ({id} templates the record's id field).
    "pulselink-youtube": {
        "server": "pulselink-mcp",
        "tool": "pulse_search",
        "params_style": "args",
        "params": {"source": "youtube"},
        "records_path": "documents",
        "id_field": "id",
        "title_field": "title",
        "updated_field": "created_at",
        "pagination": "cursor",
        "cursor_param": "cursor",
        "cursor_path": "next_cursor",
        "doc_type": "video",
        "detail": {
            "tool": "pulse_fetch",
            "params_style": "args",
            "params": {"source": "youtube", "target": "{id}"},
            "text_path": "text",
        },
    },
    # List-style sources: enumerate a channel/feed/node via pulse_list. Extend
    # with {"params": {"channel": "<subreddit | RSS feed URL | V2EX node>"}}.
    **{
        f"pulselink-{src}": {
            "server": "pulselink-mcp",
            "tool": "pulse_list",
            "params_style": "args",
            "params": {"source": src},
            "records_path": "documents",
            "id_field": "id",
            "title_field": "title",
            "text_field": "text",
            "updated_field": "created_at",
            "pagination": "cursor",
            "cursor_param": "cursor",
            "cursor_path": "next_cursor",
            "doc_type": "web",
        }
        for src in ("rss", "v2ex")
    },
    # FreshRSS (CONCEPT:AU-KG.compute.homelab-rss-reader-as): the homelab RSS reader as a gated world-model
    # source. Drives the freshrss-mcp ``freshrss_reader`` action-routed tool
    # (action=stream_contents) over the Google-Reader API; each entry becomes a
    # ``news_article`` Document. The preset is decoupled from raw GReader param
    # names — the tool maps ``newer_than``→GReader ``ot`` (unix-seconds delta
    # watermark) and ``continuation``→GReader ``c``. NOT a mirror: items pass the
    # world-model relevance gate in ``_sync_freshrss`` (KG-2.116). Restrict to one
    # category with {"params": {"stream_id": "user/-/label/Markets & Finance"}}.
    "freshrss": {
        "server": "freshrss-mcp",
        "tool": "freshrss_reader",
        "action": "stream_contents",
        "params_style": "json",
        "params": {"count": 100, "order": "o"},
        "records_path": "items",
        "id_field": "id",
        "title_field": "title",
        "text_field": "text",
        "updated_field": "published",
        "updated_since_param": "newer_than",
        "pagination": "cursor",
        "cursor_param": "continuation",
        "cursor_path": "continuation",
        "doc_type": "news_article",
    },
    # ── Atlassian + Plane issue trackers / wiki (CONCEPT:AU-KG.compute.confluence-first-class-delta/2.124/2.125) ──
    #
    # All three drive an action-routed fleet tool whose result is the api-client
    # envelope ``{status_code, data, message}`` — so every records/cursor path is
    # prefixed ``data.``. The ``server`` is overridden per instance by the sync
    # handler (a second Atlassian site / Plane workspace = a second ``*-mcp`` server),
    # and the handler injects the per-run ``params`` (JQL / space-id / project-id).
    #
    # Jira issues as typed entities: the ``jira`` delta handler (``_sync_jira``)
    # rebuilds issue/person/epic nodes from each record (carried in the Document's
    # ``metadata.record``); ``text_field`` is the always-present summary so no issue
    # is dropped. JQL ``updated >=`` (server delta) is built by the handler; cursor =
    # the enhanced-search ``nextPageToken`` (absent on the last page → exhausted).
    "jira": {
        "server": "atlassian-mcp",
        "tool": "atlassian_jira_issue",
        "action": "search_and_reconsile_issues_using_jql",
        "params_style": "json",
        "params": {
            "jql": 'created >= "1970-01-01" ORDER BY updated DESC',
            "max_results": 100,
            "fields": [
                "summary",
                "updated",
                "status",
                "issuetype",
                "assignee",
                "reporter",
                "created",
                "parent",
                "description",
                "labels",
                "priority",
            ],
        },
        "records_path": "data.issues",
        "id_field": "key",
        "title_field": "fields.summary",
        "text_field": "fields.summary",
        "updated_field": "fields.updated",
        "pagination": "cursor",
        "cursor_param": "next_page_token",
        "cursor_path": "data.nextPageToken",
        "doc_type": "issue",
    },
    # Confluence pages as a FULL MIRROR of ``:ConfluencePage`` Documents. Cloud v2
    # has no CQL search, so list via ``get_pages`` sorted by recency; the body lands
    # in ``body.storage.value`` (``body_format=storage``). Cloud v2 paginates by a
    # next-page URL under ``_links.next`` — ``cursor_from_query`` extracts the bare
    # ``cursor`` token. Delta = the connector's ``version.createdAt`` since-filter +
    # the write-layer content-hash (KG_WRITE_DELTA). The handler injects ``space-id``.
    "confluence": {
        "server": "atlassian-mcp",
        "tool": "atlassian_confluence_page",
        "action": "get_pages",
        "params_style": "json",
        "params": {"body_format": "storage", "sort": "-modified-date", "limit": 100},
        "records_path": "data.results",
        "id_field": "id",
        "title_field": "title",
        "text_field": "body.storage.value",
        "updated_field": "version.createdAt",
        "pagination": "cursor",
        "cursor_param": "cursor",
        "cursor_path": "data._links.next",
        "cursor_from_query": "cursor",
        "doc_type": "wiki",
    },
    # Plane work items as typed entities (``_sync_plane`` rebuilds issue/project nodes
    # from ``metadata.record``). Requires a ``project_id`` (injected per project by the
    # handler); ``per_page``/``cursor`` page the workspace. Depends on plane-agent's
    # ``list_work_items`` returning the raw envelope (results + ``next_cursor`` +
    # ``next_page_results``) — the fidelity fix that also restores ``updated_at``.
    "plane": {
        "server": "plane-mcp",
        "tool": "plane_work_items",
        "action": "list_work_items",
        "params_style": "json",
        "params": {"per_page": 100},
        "records_path": "data.results",
        "id_field": "id",
        "title_field": "name",
        "text_field": "name",
        "updated_field": "updated_at",
        "pagination": "cursor",
        "cursor_param": "cursor",
        "cursor_path": "data.next_cursor",
        "more_path": "data.next_page_results",
        "doc_type": "issue",
    },
    # ── Ops / platform connectors as typed OWL entities (CONCEPT:AU-KG.compute.dockerhub-repositories–2.161) ──
    #
    # Each preset is the DRAIN half (list records via the fleet ``*-mcp`` tool); the
    # matching ``_sync_*`` delta handler in ``core/source_sync.py`` rebuilds typed
    # entities from ``metadata.record`` and ingests them under an OWL class — the
    # Document path makes everything a ``:Document``, whereas these are first-class
    # classes (:Repository / :Trace / :Entity / :Person …) so the reasoner can act on
    # them. The dict-shaped / bare-list connectors (tunnel-manager, uptime-kuma,
    # technitium-dns) have NO preset here — their handlers call the tool directly via
    # ``call_tool_once`` because their result is not a flat record list.
    #
    # DockerHub repositories → :Repository / :ContainerImage (CONCEPT:AU-KG.compute.dockerhub-repositories). The
    # ``hub_repos`` tool is action-routed (action + params_json); a ``namespace`` is
    # required (injected per-namespace by the handler). DRF page envelope under
    # ``data.results`` (page-number paging). ``text_field`` is the description.
    "dockerhub-repos": {
        "server": "dockerhub-mcp",
        "tool": "hub_repos",
        "action": "list",
        "params_style": "json",
        "params": {"page_size": 100},
        "records_path": "data.results",
        "id_field": "name",
        "title_field": "name",
        "text_field": "description",
        "updated_field": "last_updated",
        "pagination": "page",
        "page_kind": "number",
        "page_param": "page",
        "page_size_param": "page_size",
        "page_size": 100,
        "start_page": 1,
        "doc_type": "container_image",
    },
    # Langfuse traces → :Trace (CONCEPT:AU-KG.compute.langfuse-traces-observations). ``langfuse_observability`` is an
    # action-routed tool that takes plain keyword args (``params_style='args'``):
    # ``action='trace_list'`` + page/limit. Langfuse returns ``{"data": [...], "meta":
    # {...}}``; page-number paging (a short final page ends the sweep). The handler also
    # drains ``legacy_observations_v1_get_many`` → :Observation / :Generation.
    "langfuse-traces": {
        "server": "langfuse-mcp",
        "tool": "langfuse_observability",
        "action": "trace_list",
        "params_style": "args",
        "params": {"limit": 100},
        "records_path": "data",
        "id_field": "id",
        "title_field": "name",
        "text_field": "input",
        "updated_field": "timestamp",
        "pagination": "page",
        "page_kind": "number",
        "page_param": "page",
        "page_size_param": "limit",
        "page_size": 100,
        "start_page": 1,
        "doc_type": "trace",
    },
    # Uses the STABLE legacy ``/api/public/observations`` route (page-number paging),
    # not the ``v2`` observations endpoint — ``/api/public/v2/observations`` is absent
    # on older self-hosted Langfuse and 404s, which burned the ingestion-sweep retries.
    "langfuse-observations": {
        "server": "langfuse-mcp",
        "tool": "langfuse_observability",
        "action": "legacy_observations_v1_get_many",
        "params_style": "args",
        "params": {"limit": 100},
        "records_path": "data",
        "id_field": "id",
        "title_field": "name",
        "text_field": "input",
        "updated_field": "startTime",
        "pagination": "page",
        "page_kind": "number",
        "page_param": "page",
        "page_size_param": "limit",
        "page_size": 100,
        "start_page": 1,
        "doc_type": "observation",
    },
    # Home Assistant states → :Device / :Entity (CONCEPT:AU-KG.compute.home-assistant-states). ``home_assistant_states``
    # action-routed; ``list_states`` returns a BARE LIST of HAState (``records_path=""`` =
    # the whole result), no pagination. Each entity_id maps to an :Entity; its device-class
    # attribute rolls up to a :Device. ``text_field`` is the state value.
    "home-assistant-states": {
        "server": "home-assistant-mcp",
        "tool": "home_assistant_states",
        "action": "list_states",
        "params_style": "json",
        "id_field": "entity_id",
        "title_field": "entity_id",
        "text_field": "state",
        "updated_field": "last_updated",
        "doc_type": "entity",
    },
    # Twenty CRM people/companies/opportunities → :Person / :Company / :Opportunity
    # (CONCEPT:AU-KG.compute.twenty-crm-people-companies). ``twenty_crm`` action-routed (action + params_json); each
    # ``get_<object>`` returns the Twenty REST envelope ``{"data": {"<object>": [...]},
    # "pageInfo": {...}}``. Cursor paging via ``pageInfo.endCursor`` /
    # ``pageInfo.hasNextPage`` (Twenty's keyset ``starting_after``).
    "twenty-people": {
        "server": "twenty-mcp",
        "tool": "twenty_crm",
        "action": "get_people",
        "params_style": "json",
        "params": {"limit": 60},
        "records_path": "data.people",
        "id_field": "id",
        "title_field": "name.firstName",
        "text_field": "jobTitle",
        "updated_field": "updatedAt",
        "pagination": "cursor",
        "cursor_param": "starting_after",
        "cursor_path": "pageInfo.endCursor",
        "more_path": "pageInfo.hasNextPage",
        "doc_type": "person",
    },
    "twenty-companies": {
        "server": "twenty-mcp",
        "tool": "twenty_crm",
        "action": "get_companies",
        "params_style": "json",
        "params": {"limit": 60},
        "records_path": "data.companies",
        "id_field": "id",
        "title_field": "name",
        "text_field": "name",
        "updated_field": "updatedAt",
        "pagination": "cursor",
        "cursor_param": "starting_after",
        "cursor_path": "pageInfo.endCursor",
        "more_path": "pageInfo.hasNextPage",
        "doc_type": "company",
    },
    "twenty-opportunities": {
        "server": "twenty-mcp",
        "tool": "twenty_crm",
        "action": "get_opportunities",
        "params_style": "json",
        "params": {"limit": 60},
        "records_path": "data.opportunities",
        "id_field": "id",
        "title_field": "name",
        "text_field": "name",
        "updated_field": "updatedAt",
        "pagination": "cursor",
        "cursor_param": "starting_after",
        "cursor_path": "pageInfo.endCursor",
        "more_path": "pageInfo.hasNextPage",
        "doc_type": "opportunity",
    },
    # ── Finance / document / genealogy connectors (CONCEPT:AU-KG.compute.audiobookshelf-libraries-books-authors–2.166) ────────
    #
    # Same DRAIN-half pattern as the ops connectors above: the preset lists records via
    # the fleet ``*-mcp`` tool, the matching ``_sync_*`` handler rebuilds typed entities.
    # Audiobookshelf (libraries→items, multi-step) and gramps (Response.data envelope
    # per object) have NO preset — their handlers call the tool directly via
    # ``call_tool_once`` because the result is multi-step / wrapped.
    #
    # Firefly III accounts/transactions/budgets → :Account / :Transaction / :Budget
    # (CONCEPT:AU-KG.compute.firefly-iii-accounts-transactions). The ``*_operations`` tools are action-routed (action +
    # params_json); each returns the Laravel JSON:API envelope ``{"data": [{"id","type",
    # "attributes":{…}}], "meta":{"pagination":{…}}}``. Page-number paging nests under the
    # tool's ``params`` query dict (``params.page`` / ``params.limit``); a short final page
    # ends the sweep. ``text_field`` digs the JSON:API ``attributes`` block.
    "firefly-accounts": {
        "server": "firefly-iii-mcp",
        "tool": "accounts_operations",
        "action": "list_account",
        "params_style": "json",
        "params": {"params": {"limit": 100}},
        "records_path": "data",
        "id_field": "id",
        "title_field": "attributes.name",
        "text_field": "attributes.type",
        "updated_field": "attributes.updated_at",
        "pagination": "page",
        "page_kind": "number",
        "page_param": "params.page",
        "page_size_param": "params.limit",
        "page_size": 100,
        "start_page": 1,
        "doc_type": "account",
    },
    "firefly-transactions": {
        "server": "firefly-iii-mcp",
        "tool": "transactions_operations",
        "action": "list_transaction",
        "params_style": "json",
        "params": {"params": {"limit": 100}},
        "records_path": "data",
        "id_field": "id",
        "title_field": "attributes.group_title",
        "text_field": "attributes.group_title",
        "updated_field": "attributes.updated_at",
        "pagination": "page",
        "page_kind": "number",
        "page_param": "params.page",
        "page_size_param": "params.limit",
        "page_size": 100,
        "start_page": 1,
        "doc_type": "transaction",
    },
    "firefly-budgets": {
        "server": "firefly-iii-mcp",
        "tool": "budgets_operations",
        "action": "list_budget",
        "params_style": "json",
        "params": {"params": {"limit": 100}},
        "records_path": "data",
        "id_field": "id",
        "title_field": "attributes.name",
        "text_field": "attributes.name",
        "updated_field": "attributes.updated_at",
        "pagination": "page",
        "page_kind": "number",
        "page_param": "params.page",
        "page_size_param": "params.limit",
        "page_size": 100,
        "start_page": 1,
        "doc_type": "budget",
    },
    # Paperless-ngx documents/correspondents/tags → :Document / :Correspondent / :Tag
    # (CONCEPT:AU-KG.compute.paperless-ngx-documents-correspondents). The ``document_operations`` tool is action-routed (action +
    # params_json) and ALREADY paginates internally (``_fetch_all`` follows DRF ``next``),
    # returning a FLAT LIST — so ``records_path=""`` (the whole result) and
    # ``pagination='none'``. ``ordering=-modified`` surfaces the freshest first.
    "paperless-documents": {
        "server": "paperless-ngx-mcp",
        "tool": "document_operations",
        "action": "list_documents",
        "params_style": "json",
        "params": {"ordering": "-modified"},
        "id_field": "id",
        "title_field": "title",
        "text_field": "content",
        "updated_field": "modified",
        "doc_type": "document",
    },
    "paperless-correspondents": {
        "server": "paperless-ngx-mcp",
        "tool": "document_operations",
        "action": "list_correspondents",
        "params_style": "json",
        "id_field": "id",
        "title_field": "name",
        "text_field": "name",
        "doc_type": "correspondent",
    },
    "paperless-tags": {
        "server": "paperless-ngx-mcp",
        "tool": "document_operations",
        "action": "list_tags",
        "params_style": "json",
        "id_field": "id",
        "title_field": "name",
        "text_field": "name",
        "doc_type": "tag",
    },
    # ── Connector dual-role presets (CONCEPT:AU-KG.ingest.mcp-tool-connector) ────────────────────────────
    #
    # The same ~58-server fleet that exposes LIVE MCP tools is ALSO an ingestion
    # surface: every connector with a record-listing tool gets a declarative preset
    # here, so "a connector repo" is BOTH a live tool AND a KG source with no new
    # transport code. Each preset below was verified against the connector's actual
    # tool surface (server name from its ``mcp_config.json``, the list tool + action
    # from its condensed MCP handler / api-client). Tool/action names are NOT
    # invented — connectors whose surface couldn't yield a record list (action/
    # compute/config tools, or list rows with no text body) are intentionally left
    # out and tracked as follow-ups. Default ``params`` are sensible templates the
    # caller overrides per instance; pagination is left ``none`` (one batch) unless
    # the connector's client auto-paginates or returns the full set in one call, so
    # no preset risks passing an unknown pagination kwarg to a fleet tool.
    # jellyfin-mcp: media library items → :MediaItem. ``jellyfin_library`` is
    # action-routed; ``get_items`` maps to GET /Items (envelope ``{"Items":[...],
    # "TotalRecordCount"}``). The default params surface real media (recursive, with
    # the Overview field) rather than the root virtual folders; StartIndex/Limit are
    # Jellyfin's standard offset paging.
    "jellyfin-media": {
        "server": "jellyfin-mcp",
        "tool": "jellyfin_library",
        "action": "get_items",
        "params_style": "json",
        "params": {
            "Recursive": True,
            "IncludeItemTypes": "Movie,Series,Episode,MusicAlbum,Audio",
            "Fields": "Overview",
            "Limit": 200,
        },
        "records_path": "Items",
        "id_field": "Id",
        "title_field": "Name",
        "text_field": "Overview",
        "updated_field": "DateCreated",
        "pagination": "page",
        "page_kind": "offset",
        "page_param": "StartIndex",
        "page_size_param": "Limit",
        "page_size": 200,
        "doc_type": "media_item",
    },
    # portainer-agent: Docker containers per endpoint → :Container. ``portainer_docker``
    # action ``docker_list_containers`` wraps the bare Docker list as ``{"data":[...]}``.
    # An ``endpoint_id`` is REQUIRED (injected per Portainer endpoint by the caller);
    # the Docker API returns every container in one call (no pagination).
    "portainer-containers": {
        "server": "portainer-agent",
        "tool": "portainer_docker",
        "action": "docker_list_containers",
        "params_style": "json",
        "params": {"endpoint_id": "{endpoint_id}", "all": True},
        "records_path": "data",
        "id_field": "Id",
        "title_field": "Names",
        "text_field": "Image",
        "updated_field": "Created",
        "doc_type": "container",
    },
    # rom-manager (RomM): game ROMs → :Game. ``romm_roms`` action ``list`` (→ GET
    # /api/roms) returns the limit/offset page envelope ``{"items":[...], "total"}``.
    "rom-manager-roms": {
        "server": "rom-manager",
        "tool": "romm_roms",
        "action": "list",
        "params_style": "json",
        "params": {"limit": 100},
        "records_path": "items",
        "id_field": "id",
        "title_field": "name",
        "text_field": "summary",
        "updated_field": "updated_at",
        "pagination": "page",
        "page_kind": "offset",
        "page_param": "offset",
        "page_size_param": "limit",
        "page_size": 100,
        "doc_type": "game",
    },
    # listmonk-api: newsletter subscribers / campaigns → :Person / :Campaign. The
    # ``listmonk_*`` tools wrap the client list as ``{"results":[...]}``.
    "listmonk-subscribers": {
        "server": "listmonk-api",
        "tool": "listmonk_subscribers",
        "action": "get_subscribers",
        "params_style": "json",
        "records_path": "results",
        "id_field": "id",
        "title_field": "name",
        "text_field": "email",
        "updated_field": "updated_at",
        "doc_type": "subscriber",
    },
    "listmonk-campaigns": {
        "server": "listmonk-api",
        "tool": "listmonk_campaigns",
        "action": "get_campaigns",
        "params_style": "json",
        "records_path": "results",
        "id_field": "id",
        "title_field": "name",
        "text_field": "subject",
        "updated_field": "updated_at",
        "doc_type": "campaign",
    },
    # wger-agent: training routines → :Routine. ``wger_routine`` action ``get_routines``
    # returns the Django-REST page envelope (``{"results":[...]}``).
    "wger-routines": {
        "server": "wger-agent",
        "tool": "wger_routine",
        "action": "get_routines",
        "params_style": "json",
        "records_path": "results",
        "id_field": "id",
        "title_field": "name",
        "text_field": "description",
        "updated_field": "created",
        "doc_type": "routine",
    },
    # ansible-tower-mcp (AWX/Tower): inventories / job-templates / projects →
    # :Inventory / :JobTemplate / :Project. Every ``list_*`` action calls the client's
    # ``handle_pagination`` which walks ALL pages and returns a FLAT list — so
    # ``records_path=""`` and ``pagination='none'`` (already complete). Records carry
    # the standard AWX id/name/description/modified fields.
    "ansible-tower-inventories": {
        "server": "ansible-tower-mcp",
        "tool": "ansible_tower_inventory",
        "action": "list_inventories",
        "params_style": "json",
        "id_field": "id",
        "title_field": "name",
        "text_field": "description",
        "updated_field": "modified",
        "doc_type": "inventory",
    },
    "ansible-tower-job-templates": {
        "server": "ansible-tower-mcp",
        "tool": "ansible_tower_job_templates",
        "action": "list_job_templates",
        "params_style": "json",
        "id_field": "id",
        "title_field": "name",
        "text_field": "description",
        "updated_field": "modified",
        "doc_type": "job_template",
    },
    "ansible-tower-projects": {
        "server": "ansible-tower-mcp",
        "tool": "ansible_tower_projects",
        "action": "list_projects",
        "params_style": "json",
        "id_field": "id",
        "title_field": "name",
        "text_field": "description",
        "updated_field": "modified",
        "doc_type": "project",
    },
    # camunda-mcp: user tasks → :UserTask. ``camunda_task`` action ``list`` (Camunda 7
    # GET /task) returns a bare JSON array (``records_path=""``). The default
    # ``platform`` is "7"; tasks carry name/description/created/lastUpdated.
    "camunda-tasks": {
        "server": "camunda-mcp",
        "tool": "camunda_task",
        "action": "list",
        "params_style": "json",
        "id_field": "id",
        "title_field": "name",
        "text_field": "description",
        "updated_field": "lastUpdated",
        "doc_type": "user_task",
    },
    # salesforce-agent: any sObject → :Record, SOQL-driven (the entity + returned
    # fields are whatever the caller's SOQL selects). Mirrors the ``sql-query`` preset:
    # ``salesforce_soql`` action ``query`` returns ``{"records":[...], "totalSize",
    # "nextRecordsUrl"}`` (capped at SALESFORCE_MAX_QUERY_RECORDS). Override ``soql``
    # for the object, e.g. {"params": {"soql": "SELECT Id,Name,Description,
    # LastModifiedDate FROM Account"}}.
    "salesforce-sobject": {
        "server": "salesforce-agent",
        "tool": "salesforce_soql",
        "action": "query",
        "params_style": "json",
        "params": {
            "soql": (
                "SELECT Id, Name, Description, LastModifiedDate "
                "FROM Account ORDER BY LastModifiedDate DESC"
            ),
            "max_records": 200,
        },
        "records_path": "records",
        "id_field": "Id",
        "title_field": "Name",
        "text_field": "Description",
        "updated_field": "LastModifiedDate",
        "doc_type": "sobject",
    },
    # erpnext-agent: any Frappe DocType → :Record, doctype-driven. ``erpnext_agent_resource``
    # action ``list_documents`` (GET /api/resource/{doctype}) returns ``{"data":[...]}``.
    # A ``doctype`` is REQUIRED and ``fields`` MUST be requested for a title/body to
    # appear (Frappe returns name-only otherwise), so override per doctype, e.g.
    # {"params": {"doctype": "Item", "fields": ["name","item_name","description","modified"]}}
    # and {"text_field": "description", "title_field": "item_name"}.
    "erpnext-doctype": {
        "server": "erpnext-agent",
        "tool": "erpnext_agent_resource",
        "action": "list_documents",
        "params_style": "json",
        "params": {"doctype": "{doctype}", "limit_page_length": 100},
        "records_path": "data",
        "id_field": "name",
        "title_field": "name",
        "text_field": "name",
        "doc_type": "erpnext_doctype",
    },
}


# ── Per-repo preset contribution (CONCEPT:AU-KG.ingest.mcp-tool-connector) ──────────────────────────
#
# A connector package can ship its OWN ingestion preset RIGHT BESIDE its MCP tool
# ("2 actions from the same repo") instead of registering it in this central dict.
# It declares a setuptools entry-point in the SAME data-only style the hub already
# uses for fleet skills/prompts (CONCEPT:AU-OS.deployment.agent-factory-autoload, ``core/providers.py``)::
#
#     # in the connector package's pyproject.toml
#     [project.entry-points."agent_utilities.source_connector_providers"]
#     jellyfin-mcp = "jellyfin_mcp.ingestion"
#
# pointing at a data subpackage that contains a ``mcp_source_presets.json`` file —
# a JSON object of ``{preset_name: {server, tool, action, field-map, ...}}`` with
# exactly the schema of an entry in :data:`MCP_TOOL_PRESETS`. The hub resolves the
# data dir via ``importlib.resources`` (it never imports the connector's business
# logic), reads the JSON, and merges it into the catalog. Contributed presets take
# precedence over the central dict (they live WITH the connector and track its tool
# surface); the central dict is the fallback for connectors that don't ship one.
# Discovery is failure-isolated and cached for the process.
SOURCE_PRESET_PROVIDER_GROUP = "agent_utilities.source_connector_providers"
_PRESET_DATA_FILE = "mcp_source_presets.json"
_contributed_presets_cache: dict[str, dict[str, Any]] | None = None


def _load_contributed_presets() -> dict[str, dict[str, Any]]:
    """Resolve + merge every fleet-contributed ``mcp_tool`` preset (cached)."""
    global _contributed_presets_cache
    if _contributed_presets_cache is not None:
        return _contributed_presets_cache
    presets: dict[str, dict[str, Any]] = {}
    try:
        from agent_utilities.core.providers import iter_provider_dirs

        for _name, data_dir in iter_provider_dirs(SOURCE_PRESET_PROVIDER_GROUP):
            data_file = data_dir / _PRESET_DATA_FILE
            if not data_file.is_file():
                continue
            try:
                loaded = json.loads(data_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                logger.debug("[KG-2.59] bad contributed presets at %s", data_file)
                continue
            if isinstance(loaded, dict):
                for key, value in loaded.items():
                    if isinstance(value, dict):
                        presets[str(key)] = dict(value)
    except Exception:  # noqa: BLE001 — one bad provider never breaks the catalog
        logger.debug("[KG-2.59] contributed-preset discovery failed", exc_info=True)
    _contributed_presets_cache = presets
    return presets


def reset_contributed_presets_cache() -> None:
    """Clear the contributed-preset cache (tests / after an install)."""
    global _contributed_presets_cache
    _contributed_presets_cache = None


def all_tool_presets() -> dict[str, dict[str, Any]]:
    """The full catalog: central presets overlaid with per-repo contributions."""
    merged: dict[str, dict[str, Any]] = dict(MCP_TOOL_PRESETS)
    merged.update(_load_contributed_presets())  # contributed wins (lives w/ connector)
    return merged


def get_tool_preset(name: str) -> dict[str, Any]:
    """Return a copy of the named preset, or ``{}`` when unknown.

    A per-repo contributed preset (entry-point) takes precedence over the central
    :data:`MCP_TOOL_PRESETS`; the central dict is the fallback.
    """
    contributed = _load_contributed_presets()
    if name in contributed:
        return dict(contributed[name])
    return dict(MCP_TOOL_PRESETS.get(name, {}))


def list_tool_presets() -> list[str]:
    """All preset names — central presets plus per-repo contributions."""
    return sorted(set(MCP_TOOL_PRESETS) | set(_load_contributed_presets()))


# ── helpers ──────────────────────────────────────────────────────────────────

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _ident(name: str, what: str) -> str:
    """Validate a SQL identifier from config (defense against SQL splicing)."""
    if not _IDENT_RE.match(name or ""):
        raise ValueError(f"sql_table {what} {name!r} is not a plain SQL identifier")
    return name


def _extract_query_param(url: str, key: str) -> str | None:
    """Pull one query-param value out of a URL/relative link.

    Cloud v2 APIs (e.g. Confluence) return the next page as a *URL* under
    ``_links.next`` (``/wiki/api/v2/pages?cursor=ABC&limit=100``) rather than a bare
    token, so a connector that needs the token sets ``cursor_from_query`` to extract
    it. Returns ``None`` when the param is absent (→ pagination is exhausted).
    """
    from urllib.parse import parse_qs, urlparse

    try:
        values = parse_qs(urlparse(url).query).get(key)
    except (ValueError, TypeError):
        return None
    return values[0] if values else None


def _set_path(target: dict[str, Any], dotted: str, value: Any) -> None:
    """Set ``value`` at a dotted path inside ``target``, creating dicts."""
    parts = dotted.split(".")
    cur = target
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


class _TemplateScope(dict):
    """``format_map`` scope: record fields first, then the base tool params."""

    def __init__(self, record: dict[str, Any], params: dict[str, Any]):
        super().__init__()
        self._record = record
        self._params = params

    def __missing__(self, key: str) -> Any:
        if key in self._record:
            return self._record[key]
        if key in self._params:
            return self._params[key]
        raise KeyError(key)


def _render(value: Any, record: dict[str, Any], params: dict[str, Any]) -> Any:
    """Substitute ``{field}`` placeholders in string values from record/params."""
    if isinstance(value, str) and "{" in value:
        try:
            return value.format_map(_TemplateScope(record, params))
        except KeyError as exc:
            raise ValueError(
                f"detail param template {value!r} references unknown field {exc}"
            ) from exc
    return value


def _decode(result: Any) -> Any:
    """Decode a fastmcp ``CallToolResult`` (structured data preferred)."""
    data = getattr(result, "data", None)
    if data is not None:
        return data
    return _decode_tool_result(result)


@register_source("mcp_tool")
class McpToolSourceConnector(LoadConnector, PollConnector):
    """Drive any MCP server's record-listing tool as a document source.

    See the module docstring for the design (CONCEPT:AU-KG.ingest.mcp-tool-connector). Config keys
    (every preset key is overridable by an explicit one):

    Transport (first match wins):
        client: Injected fastmcp ``Client`` target (e.g. an in-process
            ``FastMCP`` instance) — offline/test transport.
        url: Explicit HTTP/streamable endpoint.
        command / args / env: Explicit stdio server spec.
        server: Server name resolved via the workspace ``mcp_config.json``
            (also tries ``<server>-mcp``).
        timeout: Per-call timeout in seconds (default 60).

    Tool call:
        preset: Name of a shipped partial config (see ``MCP_TOOL_PRESETS``).
        tool: The MCP tool to call (required).
        action / action_param: Action-routing value + argument name
            (fleet convention; ``action_param`` defaults to ``action``).
        params: Declarative tool parameters.
        params_style: ``json`` (fleet: JSON-encode ``params`` into
            ``params_arg``, default ``params_json``) or ``args`` (spread
            ``params`` as plain tool arguments).
        arguments: Extra top-level tool arguments (``connection``, ``store``).

    Records + field map:
        records_path: Dotted path to the record list in the result ("" = the
            result itself). A ``{columns, rows}`` tabular envelope (sql-mcp) is
            zipped into row dicts automatically.
        id_field / title_field / text_field / updated_field: dotted field maps.
        doc_type: Document type hint.
        metadata_fields: Record fields copied into document metadata (default:
            the whole record sans the text body).
        acl_public_field / acl_users_field / acl_groups_field /
        acl_markings_field: ACL-ish record fields → :class:`ExternalAccess`
            for the ECO-4.28 permission sync.

    Detail (two-phase list+get):
        detail: ``{tool, action?, params, text_path, title_path?}`` — called
            once per record inside the same session; string params support
            ``{field}`` templating from the record then the base params.

    Pagination:
        pagination: ``none`` | ``cursor`` | ``page``.
        cursor_param: Dotted path *inside params* the cursor is sent as.
        cursor_path: Dotted path in the response carrying the next cursor.
        cursor_record_field: Fallback — cursor taken from the last record
            (keyset pagination).
        cursor_from_query: When the ``cursor_path`` value is a next-page *URL*
            (Cloud v2 ``_links.next``), the query-param name to extract the bare
            cursor token from; absence of the param ends pagination.
        more_path: Dotted path to a boolean "has more" flag; when present and
            falsy the sweep stops regardless of cursor.
        page_param / page_size_param / page_size / page_kind / start_page:
            page-number (``number``) or offset (``offset``) paging; exhaustion
            when a page returns fewer than ``page_size`` records.
        max_pages / max_records / batch_size: volume controls.

    Incremental:
        updated_since_param: Dotted path inside ``params`` bound to the prior
            checkpoint watermark, so re-polls are server-side deltas.

    SQL sweeps:
        sql_table: ``{table, key_column='id', text_column, title_column?,
            updated_column?, columns?, schema?, page_size=500, start_after=0,
            connection?}`` — bootstraps a keyset-paginated SELECT against
            sql-mcp; columns discovered via ``sql_schema`` when not listed.
    """

    provider = "MCP Tool Source"

    def configure(  # noqa: PLR0915 — flat declarative-config binding
        self,
        *,
        preset: str = "",
        client: Any = None,
        url: str = "",
        command: str = "",
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        server: str = "",
        timeout: float = 60.0,
        tool: str = "",
        action: str = "",
        action_param: str = "action",
        params: dict[str, Any] | None = None,
        params_style: str = "json",
        params_arg: str = "params_json",
        arguments: dict[str, Any] | None = None,
        records_path: str = "",
        id_field: str = "id",
        title_field: str = "title",
        text_field: str = "text",
        updated_field: str = "",
        doc_type: str = "document",
        metadata_fields: list[str] | None = None,
        acl_public_field: str = "",
        acl_users_field: str = "",
        acl_groups_field: str = "",
        acl_markings_field: str = "",
        detail: dict[str, Any] | None = None,
        pagination: str = "none",
        cursor_param: str = "",
        cursor_path: str = "",
        cursor_record_field: str = "",
        cursor_from_query: str = "",
        more_path: str = "",
        page_param: str = "",
        page_size_param: str = "",
        page_size: int = 100,
        page_kind: str = "number",
        start_page: int = 0,
        max_pages: int = 100,
        max_records: int = 0,
        batch_size: int = 200,
        updated_since_param: str = "",
        sql_table: dict[str, Any] | None = None,
        **_: object,
    ) -> None:
        # Merge the preset under the explicit config (explicit keys win).
        if preset:
            base = get_tool_preset(preset)
            if not base:
                raise ValueError(
                    f"Unknown mcp_tool preset {preset!r}. "
                    f"Available: {', '.join(list_tool_presets())}"
                )
            merged = {
                **base,
                **{k: v for k, v in self._config.items() if k != "preset"},
            }
            # Nested dicts merge shallowly so a caller can extend preset params.
            for key in ("params", "arguments"):
                if isinstance(base.get(key), dict):
                    merged[key] = {**base[key], **(self._config.get(key) or {})}
            self._config = merged
            self.configure(**merged)
            return

        if not tool and not sql_table:
            raise ValueError(
                "McpToolSourceConnector requires a 'tool' (or a 'sql_table' block)"
            )
        if params_style not in ("json", "args"):
            raise ValueError("params_style must be 'json' or 'args'")
        if pagination not in ("none", "cursor", "page"):
            raise ValueError("pagination must be 'none', 'cursor', or 'page'")
        if page_kind not in ("number", "offset"):
            raise ValueError("page_kind must be 'number' or 'offset'")
        if client is None and not (url or command or server):
            raise ValueError(
                "McpToolSourceConnector needs a transport: one of "
                "'client', 'url', 'command', or 'server'"
            )

        self._injected_client = client
        self.url = url
        self.command = command
        self.command_args = list(args or [])
        self.command_env = dict(env or {})
        self.server = server
        self.timeout = float(timeout)
        self.tool = tool or "sql_query"
        self.action = action or ("execute" if sql_table else "")
        self.action_param = action_param
        self.params = dict(params or {})
        self.params_style = params_style
        self.params_arg = params_arg
        self.extra_arguments = dict(arguments or {})
        self.records_path = records_path
        self.id_field = id_field
        self.title_field = title_field
        self.text_field = text_field
        self.updated_field = updated_field
        self.doc_type = doc_type
        self.metadata_fields = list(metadata_fields or [])
        self.acl_public_field = acl_public_field
        self.acl_users_field = acl_users_field
        self.acl_groups_field = acl_groups_field
        self.acl_markings_field = acl_markings_field
        self.detail = dict(detail or {})
        self.pagination = pagination
        self.cursor_param = cursor_param
        self.cursor_path = cursor_path
        self.cursor_record_field = cursor_record_field
        self.cursor_from_query = cursor_from_query
        self.more_path = more_path
        self.page_param = page_param
        self.page_size_param = page_size_param
        self.page_size = max(1, int(page_size))
        self.page_kind = page_kind
        self.start_page = int(start_page)
        self.max_pages = max(1, int(max_pages))
        self.max_records = max(0, int(max_records))
        self.batch_size = max(1, int(batch_size))
        self.updated_since_param = updated_since_param
        self.sql_table = dict(sql_table or {})
        self._sql_ready = not self.sql_table
        self._sql = ""
        self._sql_since = ""

    @property
    def name(self) -> str:
        return f"mcp_tool:{self.server or self.url or 'inline'}/{self.tool}"

    def health_check(self) -> bool:
        return bool(self.tool) and (
            self._injected_client is not None
            or bool(self.url or self.command or self.server)
        )

    # ── transport ────────────────────────────────────────────────────────────

    def _client_target(self) -> Any:
        """Resolve the fastmcp ``Client`` target from the configured transport."""
        if self._injected_client is not None:
            return self._injected_client
        if self.url:
            return self.url
        if self.command:
            env = {k: os.path.expandvars(str(v)) for k, v in self.command_env.items()}
            return {
                "mcpServers": {
                    "source": {
                        "command": self.command,
                        "args": self.command_args,
                        "env": env,
                    }
                }
            }
        servers = _load_mcp_config()
        cfg = servers.get(self.server) or servers.get(f"{self.server}-mcp")
        if not cfg:
            raise McpToolSourceError(
                f"MCP server {self.server!r} not found in mcp_config.json; "
                "pass an explicit 'url'/'command' or an injected 'client'."
            )
        return {"mcpServers": {self.server: dict(cfg)}}

    def _open_client(self) -> Any:
        """Build the fastmcp client for one run (lazy import, clear error).

        Authenticate to JWT-protected fleet servers with the service-account bearer
        (CONCEPT:AU-OS.identity.so-jwt-protected-children) — opt-in via ``MCP_CLIENT_AUTH=oidc-client-credentials``,
        mirroring the multiplexer. A url-based target uses the URL directly so the
        per-request ``httpx.Auth`` applies; a mint failure / disabled auth degrades
        to no auth (an unauthenticated server is unaffected). An injected client is
        used as-is.
        """
        try:
            from fastmcp import Client
        except ImportError as exc:  # pragma: no cover - env without fastmcp
            raise McpToolSourceError(
                "McpToolSourceConnector needs 'fastmcp' (install agent-utilities[mcp])."
            ) from exc
        target = self._client_target()
        if self._injected_client is not None:
            return Client(target, timeout=self.timeout)

        auth = None
        try:
            from agent_utilities.mcp.client_credentials import bearer_auth

            auth = bearer_auth(None)
        except Exception:  # noqa: BLE001 — auth is best-effort/opt-in
            auth = None
        if auth is None:
            return Client(target, timeout=self.timeout)

        # Resolve a single url-based server config to its URL so the bearer applies.
        url = target if isinstance(target, str) else None
        if url is None and isinstance(target, dict):
            servers = target.get("mcpServers") or {}
            if len(servers) == 1:
                cfg = next(iter(servers.values()))
                if isinstance(cfg, dict) and cfg.get("url"):
                    url = str(cfg["url"])
        if url and url.startswith(("http://", "https://")):
            try:
                return Client(url, auth=auth, timeout=self.timeout)
            except TypeError:  # pragma: no cover - older fastmcp without auth=
                pass
        return Client(target, timeout=self.timeout)

    # ── tool-call plumbing ───────────────────────────────────────────────────

    def _build_arguments(
        self, params: dict[str, Any], *, tool_action: str = "", style: str = ""
    ) -> dict[str, Any]:
        """Assemble the tool's argument dict from action + params + extras."""
        style = style or self.params_style
        arguments: dict[str, Any] = dict(self.extra_arguments)
        action = tool_action or self.action
        if action:
            arguments[self.action_param] = action
        if style == "json":
            arguments[self.params_arg] = json.dumps(params, default=str)
        else:
            arguments.update(params)
        return arguments

    async def _call(self, client: Any, tool: str, arguments: dict[str, Any]) -> Any:
        try:
            return _decode(await client.call_tool(tool, arguments))
        except McpToolSourceError:
            raise
        except Exception as exc:
            raise McpToolSourceError(
                f"MCP tool {tool!r} on {self.server or self.url or 'inline'} "
                f"failed: {exc}"
            ) from exc

    def _records(self, result: Any) -> list[dict[str, Any]]:
        """Extract the record list; a {columns, rows} envelope is zipped."""
        data = _dig(result, self.records_path) if self.records_path else result
        if (
            isinstance(data, dict)
            and isinstance(data.get("columns"), list)
            and isinstance(data.get("rows"), list)
        ):
            cols = [str(c) for c in data["columns"]]
            return [
                dict(zip(cols, row, strict=False))
                for row in data["rows"]
                if isinstance(row, list)
            ]
        if isinstance(data, list):
            return [r for r in data if isinstance(r, dict)]
        return []

    # ── record → document ────────────────────────────────────────────────────

    def _external_access(self, record: dict[str, Any]) -> ExternalAccess:
        if not (
            self.acl_public_field
            or self.acl_users_field
            or self.acl_groups_field
            or self.acl_markings_field
        ):
            return ExternalAccess.public()

        def _principals(field: str) -> list[str]:
            raw = _dig(record, field) if field else None
            if isinstance(raw, str):
                return [p.strip() for p in raw.split(",") if p.strip()]
            if isinstance(raw, list):
                return [str(p) for p in raw if p]
            return []

        public = True
        if self.acl_public_field:
            public = bool(_dig(record, self.acl_public_field))
        users = _principals(self.acl_users_field)
        groups = _principals(self.acl_groups_field)
        markings = _principals(self.acl_markings_field)
        if users or groups:
            public = False if not self.acl_public_field else public
        return ExternalAccess(
            is_public=public,
            user_emails=users,
            group_ids=groups,
            markings=markings,
        )

    async def _fetch_detail(
        self, client: Any, record: dict[str, Any]
    ) -> tuple[str | None, str | None]:
        """Run the per-record detail call → (text, title); None text = skip."""
        spec = self.detail
        detail_params = {
            k: _render(v, record, self.params)
            for k, v in dict(spec.get("params") or {}).items()
        }
        arguments = self._build_arguments(
            detail_params,
            tool_action=spec.get("action", ""),
            style=spec.get("params_style", self.params_style),
        )
        result = await self._call(client, spec.get("tool") or self.tool, arguments)
        text = _dig(result, spec.get("text_path", "text"))
        title = _dig(result, spec["title_path"]) if spec.get("title_path") else None
        return (
            text if isinstance(text, str) else None,
            str(title) if title else None,
        )

    def _to_document(
        self, record: dict[str, Any], text: str | None = None, title: str | None = None
    ) -> SourceDocument | None:
        rid = _dig(record, self.id_field)
        body = text if text is not None else _dig(record, self.text_field)
        if rid is None or not isinstance(body, str) or not body.strip():
            return None
        doc_title = title or _dig(record, self.title_field)
        updated = _dig(record, self.updated_field) if self.updated_field else None
        if self.metadata_fields:
            meta_record = {f: _dig(record, f) for f in self.metadata_fields}
        else:
            meta_record = {k: v for k, v in record.items() if k != self.text_field}
        return SourceDocument(
            id=str(rid),
            source_uri=f"mcp-tool://{self.server or self.url or 'inline'}/{self.tool}/{rid}",
            title=str(doc_title) if doc_title else str(rid),
            text=body,
            doc_type=self.doc_type,
            metadata={
                "server": self.server or self.url,
                "tool": self.tool,
                "record": meta_record,
            },
            external_access=self._external_access(record),
            updated_at=str(updated) if updated is not None else None,
        )

    # ── sql_table bootstrap ──────────────────────────────────────────────────

    async def _prepare_sql_table(self, client: Any) -> None:
        """Turn a ``sql_table`` block into a keyset-paginated sql_query sweep.

        Table/column identifiers come from operator config (CONCEPT:AU-KG.ingest.mcp-tool-connector)
        and are validated as plain identifiers; values always travel as bound
        parameters via sql-mcp's ``params``. Columns are discovered through
        ``sql_schema`` (action=columns) inside the same session when not given.
        """
        spec = self.sql_table
        table = _ident(str(spec.get("table", "")), "table")
        schema = str(spec.get("schema", "") or "")
        if schema:
            _ident(schema, "schema")
        key = _ident(str(spec.get("key_column", "id")), "key_column")
        page_size = max(1, int(spec.get("page_size", 500)))

        columns = [str(c) for c in (spec.get("columns") or [])]
        if not columns:
            schema_params: dict[str, Any] = {"table": table}
            if schema:
                schema_params["schema"] = schema
            arguments = self._build_arguments(schema_params, tool_action="columns")
            if spec.get("connection"):
                arguments["connection"] = str(spec["connection"])
            described = await self._call(client, "sql_schema", arguments)
            if isinstance(described, dict):
                described = described.get("result", described.get("columns", []))
            columns = [
                str(c.get("name"))
                for c in (described if isinstance(described, list) else [])
                if isinstance(c, dict) and c.get("name")
            ]
        if not columns:
            raise McpToolSourceError(
                f"sql_table column discovery returned nothing for {table!r}"
            )
        for col in columns:
            _ident(col, "column")

        text_col = str(spec.get("text_column", "") or "")
        if not text_col:
            raise ValueError("sql_table requires a 'text_column'")
        _ident(text_col, "text_column")
        title_col = str(spec.get("title_column", "") or "")
        updated_col = str(spec.get("updated_column", "") or "")
        for needed in (key, text_col, title_col, updated_col):
            if needed and needed not in columns:
                columns.append(_ident(needed, "column"))

        qualified = f"{schema}.{table}" if schema else table
        select = ", ".join(columns)
        self._sql = (
            f"SELECT {select} FROM {qualified} "  # noqa: S608 — identifiers validated above
            f"WHERE {key} > :after ORDER BY {key}"
        )
        self._sql_since = (
            f"SELECT {select} FROM {qualified} "  # noqa: S608 — identifiers validated above
            f"WHERE {key} > :after AND {updated_col} > :since ORDER BY {key}"
            if updated_col
            else ""
        )

        self.params = {
            "sql": self._sql,
            "params": {"after": spec.get("start_after", 0)},
            "max_rows": page_size,
        }
        self.records_path = ""
        self.id_field = key
        self.text_field = text_col
        self.title_field = title_col or key
        self.updated_field = updated_col
        self.pagination = "cursor"
        self.cursor_param = "params.after"
        self.cursor_record_field = key
        self.more_path = "truncated"
        if updated_col:
            self.updated_since_param = "params.since"
        if spec.get("connection"):
            self.extra_arguments.setdefault("connection", str(spec["connection"]))
        self._sql_ready = True

    # ── pagination drain ─────────────────────────────────────────────────────

    def _page_params(self, state: dict[str, Any], since: str | None) -> dict[str, Any]:
        """Per-page params: base + cursor/page position + since watermark."""
        params = json.loads(json.dumps(self.params, default=str))  # deep copy
        if since and self._sql_since:
            params["sql"] = self._sql_since
            _set_path(params, "params.since", since)
        elif since and self.updated_since_param:
            _set_path(params, self.updated_since_param, since)
        if self.pagination == "cursor" and state.get("cursor") is not None:
            if not self.cursor_param:
                raise ValueError("cursor pagination requires 'cursor_param'")
            _set_path(params, self.cursor_param, state["cursor"])
        elif self.pagination == "page":
            if not self.page_param:
                raise ValueError("page pagination requires 'page_param'")
            page = int(state.get("page", self.start_page))
            value = page * self.page_size if self.page_kind == "offset" else page
            _set_path(params, self.page_param, value)
            if self.page_size_param:
                _set_path(params, self.page_size_param, self.page_size)
        return params

    def _advance(
        self, state: dict[str, Any], result: Any, records: list[dict[str, Any]]
    ) -> bool:
        """Advance pagination ``state`` in place; return True when exhausted."""
        if self.pagination == "none" or not records:
            return True
        if self.pagination == "page":
            if len(records) < self.page_size:
                return True
            state["page"] = int(state.get("page", self.start_page)) + 1
            return False
        # cursor mode
        if self.more_path:
            more = _dig(result, self.more_path) if isinstance(result, dict) else None
            if not more:
                return True
        nxt: Any = None
        if self.cursor_path and isinstance(result, dict):
            nxt = _dig(result, self.cursor_path)
            if nxt is not None and self.cursor_from_query:
                # The cursor is embedded in a next-page URL (Confluence ``_links.next``);
                # extract the bare token, or None when the link omits it (exhausted).
                nxt = _extract_query_param(str(nxt), self.cursor_from_query)
        if nxt is None and self.cursor_record_field:
            nxt = _dig(records[-1], self.cursor_record_field)
        if nxt is None or nxt == state.get("cursor"):
            return True
        state["cursor"] = nxt
        return False

    def _drain(
        self,
        state: dict[str, Any],
        *,
        since: str | None,
        limit: int,
    ) -> tuple[list[SourceDocument], dict[str, Any], bool, str | None]:
        """One session: pull pages from ``state`` until limit/exhaustion.

        Returns ``(documents, new_state, exhausted, max_updated_seen)``. The
        session is opened once, reused for every page and detail call, and
        closed before returning (CONCEPT:AU-KG.ingest.mcp-tool-connector session lifecycle).
        """

        async def run() -> tuple[
            list[SourceDocument], dict[str, Any], bool, str | None
        ]:
            docs: list[SourceDocument] = []
            new_state = dict(state)
            exhausted = False
            max_updated: str | None = None
            async with self._open_client() as client:
                if not self._sql_ready:
                    await self._prepare_sql_table(client)
                pages = 0
                while pages < self.max_pages:
                    params = self._page_params(new_state, since)
                    result = await self._call(
                        client, self.tool, self._build_arguments(params)
                    )
                    records = self._records(result)
                    for record in records:
                        if since and self.updated_field:
                            # Filter before the detail fetch so an unchanged
                            # record costs zero extra tool calls on a re-poll.
                            updated = _dig(record, self.updated_field)
                            if updated is not None and str(updated) <= str(since):
                                continue
                        text = title = None
                        if self.detail:
                            try:
                                text, title = await self._fetch_detail(client, record)
                            except McpToolSourceError as exc:
                                logger.warning(
                                    "[KG-2.59] detail fetch failed for %s: %s",
                                    _dig(record, self.id_field),
                                    exc,
                                )
                                continue
                            if text is None:
                                continue
                        doc = self._to_document(record, text, title)
                        if doc is None:
                            continue
                        docs.append(doc)
                        if doc.updated_at is not None and (
                            max_updated is None
                            or str(doc.updated_at) > str(max_updated)
                        ):
                            max_updated = doc.updated_at
                    pages += 1
                    exhausted = self._advance(new_state, result, records)
                    if exhausted:
                        break
                    if limit and len(docs) >= limit:
                        break
                # max_pages backstop: exhausted stays False so a later poll
                # resumes from the advanced cursor/page state.
            return docs, new_state, exhausted, max_updated

        return _run_async(run())

    # ── LoadConnector / PollConnector ────────────────────────────────────────

    def load(self) -> Iterator[SourceDocument]:
        """Full sweep: one session across every page, capped by max_records."""
        docs, _, _, _ = self._drain({}, since=None, limit=self.max_records)
        if self.max_records:
            docs = docs[: self.max_records]
        yield from docs

    def poll(self, checkpoint: ConnectorCheckpoint | None = None) -> CheckpointedBatch:
        """One batch per call: resume pagination, bind the since-watermark.

        (CONCEPT:AU-KG.ingest.mcp-tool-connector + ECO-4.26) Pagination position lives in
        ``checkpoint.state``; the watermark only advances once a sweep
        exhausts, so a resumed mid-sweep run keeps filtering against the
        watermark of the previous *completed* sweep.
        """
        prior_watermark = checkpoint.watermark if checkpoint else None
        state = dict(checkpoint.state) if checkpoint else {}
        pending = state.pop("pending_watermark", None)

        docs, new_state, exhausted, max_updated = self._drain(
            state, since=prior_watermark, limit=self.batch_size
        )
        if self.max_records:
            docs = docs[: self.max_records]

        candidates = [w for w in (pending, max_updated, prior_watermark) if w]
        high_water = max(candidates, key=str) if candidates else None
        if exhausted:
            cp = ConnectorCheckpoint(has_more=False, watermark=high_water)
        else:
            new_state["pending_watermark"] = high_water
            cp = ConnectorCheckpoint(
                has_more=True,
                cursor=str(new_state.get("cursor") or "") or None,
                watermark=prior_watermark,
                state=new_state,
            )
        return CheckpointedBatch(documents=docs, checkpoint=cp)


async def call_tool_once(
    *,
    tool: str,
    server: str = "",
    action: str = "",
    params: dict[str, Any] | None = None,
    client: Any = None,
    url: str = "",
    command: str = "",
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
    params_style: str = "json",
    params_arg: str = "params_json",
    action_param: str = "action",
    timeout: float = 60.0,
) -> Any:
    """One-shot fleet MCP tool call — the **write-side twin** of the KG-2.59 source
    connector (CONCEPT:AU-KG.ontology.batch-actions-executor).

    The source connector *reads* records out of a fleet MCP server; this calls one
    fleet tool *once* and returns its decoded result, reusing the very same transport
    resolution (``server`` resolved through ``mcp_config``, or an injected ``client`` /
    explicit ``url``/``command``), the fleet ``action`` + ``params_json`` argument
    assembly, and the result decode. It exists so a *governed* ontology Action can push
    a mutation back to any fleet system without a bespoke client — symmetric with how
    ingestion pulls from one.

    Args mirror the connector: identify the server (``server`` / ``client`` / ``url`` /
    ``command``), the ``tool`` and its routing ``action``, and the ``params`` dict
    (JSON-encoded into ``params_json`` under the fleet convention by default).
    """
    conn = McpToolSourceConnector(
        server=server,
        tool=tool,
        action=action,
        client=client,
        url=url,
        command=command,
        args=args,
        env=env,
        params_style=params_style,
        params_arg=params_arg,
        action_param=action_param,
        timeout=timeout,
    )
    arguments = conn._build_arguments(dict(params or {}))
    async with conn._open_client() as open_client:
        return await conn._call(open_client, tool, arguments)
