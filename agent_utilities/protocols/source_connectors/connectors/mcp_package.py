from __future__ import annotations

"""Agent-package fleet connector — any MCP package as a document source.

CONCEPT:AU-ECO.connector.mcp-package-adapter — MCP Agent-Package Connector Adapter

The agent ecosystem ships ~50 sibling packages under ``agent-packages/agents/*``
(``scholarx``, ``github-agent``, ``gitlab-api``, ``servicenow-api``,
``mattermost-mcp``, ``nextcloud-agent``, ``microsoft-agent``, ``atlassian-agent``,
…), each exposing a FastMCP server. Rather than hand-write 50 connectors, this
single adapter reaches *any* of them through its MCP server, calls a declared
document-yielding tool, and maps the JSON result to :class:`SourceDocument`s — so
the entire fleet becomes ingestable with **config, not code** (see
:mod:`package_manifest` for the preset catalog and the Onyx parity map).

Two ways to supply the MCP transport:

  * **Injected** ``call_tool(tool_name, args) -> result`` — a plain sync callable
    (used by tests; no servers spawned, fully offline).
  * **Production** — when no ``call_tool`` is given, the adapter spawns the
    package's MCP server over stdio (reading its ``command``/``args``/``env`` from
    the workspace ``mcp_config.json``, the same source the multiplexer uses) and
    calls the tool. The stdio session runs in a dedicated event loop thread so the
    sync ``load`` / ``poll`` surface is safe to call from inside the async
    ingestion adaptor.
"""

import json
import logging
import os
from collections.abc import Callable, Iterator
from typing import Any

from agent_utilities.core.config import setting

from ..base import (
    CheckpointedBatch,
    ConnectorCheckpoint,
    LoadConnector,
    PollConnector,
    SourceDocument,
    default_external_access,
)
from ..registry import register_source

logger = logging.getLogger(__name__)

# An injected tool caller: (tool_name, arguments) -> decoded result.
CallToolFn = Callable[[str, dict[str, Any]], Any]


def _run_async(coro: Any) -> Any:
    """Run a coroutine to completion, safe whether or not a loop is running.

    The connector surface is synchronous but the MCP client is async. When called
    from inside a running loop (the async ingestion adaptor), the coroutine is run
    on a fresh loop in a worker thread to avoid ``asyncio.run`` re-entrancy.
    """
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(coro)).result()


def _load_mcp_config() -> dict[str, Any]:
    """Load the workspace ``mcp_config.json`` (env override → root default)."""
    path = setting("MCP_CONFIG_PATH") or setting("MCP_CONFIG")
    if not path:
        path = os.path.join(
            setting("WORKSPACE_PATH", "/home/apps/workspace"), "mcp_config.json"
        )
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh).get("mcpServers", {})
    except Exception as exc:  # noqa: BLE001 — missing config → no production transport
        logger.warning("[ECO-4.29] could not read mcp_config %s: %s", path, exc)
        return {}


def _decode_tool_result(result: Any) -> Any:
    """Decode an MCP ``call_tool`` result into a Python object.

    Handles the FastMCP ``CallToolResult`` (``.content`` list of text blocks),
    a raw JSON string, or an already-decoded object.
    """
    content = getattr(result, "content", None)
    if content is not None:
        texts = [getattr(c, "text", "") for c in content if getattr(c, "text", "")]
        joined = "\n".join(texts).strip()
        if joined:
            try:
                return json.loads(joined)
            except (ValueError, TypeError):
                return joined
        # structured content fallback
        sc = getattr(result, "structured_content", None) or getattr(
            result, "structuredContent", None
        )
        if sc is not None:
            return sc
        return joined
    if isinstance(result, str):
        try:
            return json.loads(result)
        except (ValueError, TypeError):
            return result
    return result


def _default_call_tool(server_name: str) -> CallToolFn:
    """Build a production stdio MCP tool-caller for ``server_name``.

    Spawns the package's MCP server (per ``mcp_config.json``), initializes a
    session, calls the tool, decodes the result, and tears down — one short-lived
    session per call (connectors poll infrequently, so this is acceptable and
    avoids holding child processes open).
    """
    servers = _load_mcp_config()
    cfg = servers.get(server_name) or servers.get(f"{server_name}-mcp")
    if not cfg or not cfg.get("command"):
        raise KeyError(
            f"No MCP server {server_name!r} in mcp_config.json (need a 'command'). "
            "Pass call_tool=... to use this connector offline."
        )

    def _call(tool_name: str, arguments: dict[str, Any]) -> Any:
        async def _run() -> Any:
            from mcp import ClientSession, StdioServerParameters, stdio_client

            env = os.environ.copy()
            for k, v in (cfg.get("env") or {}).items():
                env[k] = os.path.expandvars(str(v))
            params = StdioServerParameters(
                command=cfg["command"], args=cfg.get("args", []), env=env
            )
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return await session.call_tool(tool_name, arguments)

        return _decode_tool_result(_run_async(_run()))

    return _call


@register_source("mcp")
class MCPPackageConnector(LoadConnector, PollConnector):
    """Adapt an MCP agent-package's document-yielding tool into a connector.

    CONCEPT:AU-ECO.connector.mcp-package-adapter.

    Config:
        package: Logical package name (e.g. ``scholarx``); resolves a preset from
            :mod:`package_manifest` for the remaining fields when they are omitted.
        server: MCP server name in ``mcp_config.json`` (defaults to
            ``<package>-mcp``).
        tool: The tool to call (e.g. ``search_papers``).
        args: Static arguments passed to the tool.
        query_arg: Name of the tool argument carrying the search query (optional).
        query: Search query value bound to ``query_arg``.
        records_field: Dotted path to the array of records in the tool result
            (default: the result is itself the list).
        id_field / title_field / text_field / updated_field: record→document map.
        cursor_arg / cursor_field: pagination argument + response cursor field.
        call_tool: Optional injected ``(tool, args) -> result`` for offline use.
    """

    provider = "MCP Agent-Package"

    def configure(
        self,
        *,
        package: str = "",
        server: str = "",
        tool: str = "",
        args: dict[str, Any] | None = None,
        query_arg: str = "",
        query: str = "",
        records_field: str = "",
        id_field: str = "id",
        title_field: str = "title",
        text_field: str = "text",
        updated_field: str = "",
        cursor_arg: str = "",
        cursor_field: str = "",
        doc_type: str = "document",
        call_tool: CallToolFn | None = None,
        **_: object,
    ) -> None:
        # Merge a preset from the parity catalog when only a package is named.
        if package and not tool:
            from .package_manifest import get_preset

            preset = get_preset(package)
            if preset:
                server = server or preset.get("server", "")
                tool = tool or preset.get("tool", "")
                records_field = records_field or preset.get("records_field", "")
                id_field = preset.get("id_field", id_field)
                title_field = preset.get("title_field", title_field)
                text_field = preset.get("text_field", text_field)
                updated_field = updated_field or preset.get("updated_field", "")
                cursor_field = cursor_field or preset.get("cursor_field", "")
                cursor_arg = cursor_arg or preset.get("cursor_arg", "")
                query_arg = query_arg or preset.get("query_arg", "")
                doc_type = preset.get("doc_type", doc_type)
                args = {**preset.get("args", {}), **(args or {})}

        if not tool:
            raise ValueError(
                "MCPPackageConnector requires a 'tool' (or a known 'package' preset)"
            )

        self.package = package
        self.server = server or (f"{package}-mcp" if package else "")
        self.tool = tool
        self.args = dict(args or {})
        self.query_arg = query_arg
        self.query = query
        self.records_field = records_field
        self.id_field = id_field
        self.title_field = title_field
        self.text_field = text_field
        self.updated_field = updated_field
        self.cursor_arg = cursor_arg
        self.cursor_field = cursor_field
        self.doc_type = doc_type
        self._call_tool = call_tool

    @property
    def name(self) -> str:
        return f"mcp:{self.package or self.server or self.tool}"

    def _caller(self) -> CallToolFn:
        if self._call_tool is None:
            self._call_tool = _default_call_tool(self.server)
        return self._call_tool

    def health_check(self) -> bool:
        return bool(self.tool) and (self._call_tool is not None or bool(self.server))

    # -- record → document mapping -----------------------------------------

    def _records(self, result: Any) -> list[dict[str, Any]]:
        from .rest import _dig  # reuse the dotted-path resolver

        data = _dig(result, self.records_field) if self.records_field else result
        if isinstance(data, dict):
            # Some tools wrap the list under a single obvious key.
            for key in ("results", "items", "records", "data", "documents"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
        if isinstance(data, list):
            return [r for r in data if isinstance(r, dict)]
        return []

    def _to_document(self, record: dict[str, Any]) -> SourceDocument | None:
        from .rest import _dig

        rid = _dig(record, self.id_field)
        text = _dig(record, self.text_field)
        if rid is None or not isinstance(text, str) or not text.strip():
            return None
        title = _dig(record, self.title_field)
        updated = _dig(record, self.updated_field) if self.updated_field else None
        return SourceDocument(
            id=str(rid),
            source_uri=f"mcp://{self.package or self.server}/{rid}",
            title=str(title) if title else str(rid),
            text=text,
            doc_type=self.doc_type,
            metadata={"package": self.package, "tool": self.tool, "raw": record},
            # No ACL surface on this connector's record shape (CONCEPT:AU-P0-4):
            # fail-closed default (quarantined), not silently public. Set
            # CONNECTOR_DEFAULT_PUBLIC=true to opt a dev/local deployment back
            # into the legacy public-by-default behavior.
            external_access=default_external_access(),
            updated_at=str(updated) if updated is not None else None,
        )

    def _call(
        self, cursor: str | None = None
    ) -> tuple[list[SourceDocument], str | None]:
        from .rest import _dig

        arguments = dict(self.args)
        if self.query_arg and self.query:
            arguments[self.query_arg] = self.query
        if self.cursor_arg and cursor:
            arguments[self.cursor_arg] = cursor
        result = self._caller()(self.tool, arguments)
        records = self._records(result)
        docs = [d for d in (self._to_document(r) for r in records) if d is not None]
        next_cursor = None
        if self.cursor_field:
            top = _dig(result, self.cursor_field) if isinstance(result, dict) else None
            next_cursor = str(top) if top else None
        return docs, next_cursor

    # -- LoadConnector / PollConnector -------------------------------------

    def load(self) -> Iterator[SourceDocument]:
        docs, _ = self._call()
        yield from docs

    def poll(self, checkpoint: ConnectorCheckpoint | None = None) -> CheckpointedBatch:
        """One tool call per poll, carrying a pagination cursor when supported.

        CONCEPT:AU-ECO.connector.preset-cursor-poll — when the tool/preset exposes a cursor, ``poll`` advances
        a page at a time; otherwise it returns a single batch with ``has_more``
        false. ``seen_ids`` dedups across polls for cursor-less sources.
        """
        cursor = checkpoint.cursor if checkpoint else None
        prior_ids = set(checkpoint.seen_ids) if checkpoint else set()
        docs, next_cursor = self._call(cursor)
        fresh = [d for d in docs if d.id not in prior_ids]
        new_ids = prior_ids | {d.id for d in docs}
        cp = ConnectorCheckpoint(
            has_more=bool(next_cursor),
            cursor=next_cursor,
            seen_ids=sorted(new_ids)
            if not next_cursor
            else checkpoint.seen_ids
            if checkpoint
            else [],
        )
        return CheckpointedBatch(documents=fresh, checkpoint=cp)
