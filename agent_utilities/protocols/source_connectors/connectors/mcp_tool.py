from __future__ import annotations

"""Universal MCP-tool ingestion source — any fleet MCP server as a KG source.

CONCEPT:KG-2.59 — MCP Tool Source Connector

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
    "get_tool_preset",
    "list_tool_presets",
]


class McpToolSourceError(RuntimeError):
    """A transport/tool failure while draining an MCP-backed source.

    Raised for session, tool-call, and pagination failures so the ingestion
    adaptor reports a typed, actionable drain error (CONCEPT:KG-2.59) (config
    mistakes raise ``ValueError`` at build time, per the framework convention).
    """


# ── Recipe presets (data, not code) ─────────────────────────────────────────
#
# Named partial configs for the common fleet sources (CONCEPT:KG-2.59), in the
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
}


def get_tool_preset(name: str) -> dict[str, Any]:
    """Return a copy of the named preset, or ``{}`` when unknown."""
    return dict(MCP_TOOL_PRESETS.get(name, {}))


def list_tool_presets() -> list[str]:
    """All shipped preset names."""
    return sorted(MCP_TOOL_PRESETS)


# ── helpers ──────────────────────────────────────────────────────────────────

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _ident(name: str, what: str) -> str:
    """Validate a SQL identifier from config (defense against SQL splicing)."""
    if not _IDENT_RE.match(name or ""):
        raise ValueError(f"sql_table {what} {name!r} is not a plain SQL identifier")
    return name


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

    See the module docstring for the design (CONCEPT:KG-2.59). Config keys
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
    priority = 60

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
        """Build the fastmcp client for one run (lazy import, clear error)."""
        try:
            from fastmcp import Client
        except ImportError as exc:  # pragma: no cover - env without fastmcp
            raise McpToolSourceError(
                "McpToolSourceConnector needs 'fastmcp' "
                "(install agent-utilities[mcp])."
            ) from exc
        return Client(self._client_target(), timeout=self.timeout)

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

        Table/column identifiers come from operator config (CONCEPT:KG-2.59)
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
        closed before returning (CONCEPT:KG-2.59 session lifecycle).
        """

        async def run() -> (
            tuple[list[SourceDocument], dict[str, Any], bool, str | None]
        ):
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

        (CONCEPT:KG-2.59 + ECO-4.26) Pagination position lives in
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
