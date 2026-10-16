"""In-process fake of the host-provided tool client session (tests only).

agent-utilities carries no MCP client: graph-os hands the harness an async
context manager exposing ``list_tools``/``call_tool``/``read_resource``
(:class:`agent_utilities.host_ports.ToolClient`). Tests inject this fake as that
session. Tools register with ``@server.tool`` (or ``server.tool(fn, name=...)``)
and are listed with a JSON input schema derived from their signature, the same
protocol view a real client reports.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from pydantic import create_model

from agent_utilities.host_ports import HostToolError


def _input_schema(fn: Callable[..., Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for name, parameter in inspect.signature(fn).parameters.items():
        annotation = (
            Any
            if parameter.annotation is inspect.Parameter.empty
            else parameter.annotation
        )
        default = (
            ... if parameter.default is inspect.Parameter.empty else parameter.default
        )
        fields[name] = (annotation, default)
    schema = create_model(f"{fn.__name__}_arguments", **fields).model_json_schema()
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
    return schema


class FakeToolServer:
    """A named set of tools/resources served through the ToolClient shape."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._tools: dict[str, Callable[..., Any]] = {}
        self._resources: dict[str, Callable[[], Any]] = {}

    def tool(
        self, fn: Callable[..., Any] | None = None, *, name: str | None = None
    ) -> Any:
        def register(target: Callable[..., Any]) -> Callable[..., Any]:
            self._tools[name or target.__name__] = target
            return target

        return register(fn) if fn is not None else register

    def resource(self, uri: str) -> Callable[[Callable[[], Any]], Callable[[], Any]]:
        def register(target: Callable[[], Any]) -> Callable[[], Any]:
            self._resources[uri] = target
            return target

        return register

    async def __aenter__(self) -> FakeToolServer:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def list_tools(self) -> list[Any]:
        return [
            SimpleNamespace(
                name=tool_name,
                description=inspect.getdoc(fn) or "",
                inputSchema=_input_schema(fn),
                annotations=None,
                meta=None,
            )
            for tool_name, fn in self._tools.items()
        ]

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None, **_kwargs: Any
    ) -> Any:
        fn = self._tools.get(name)
        if fn is None:
            raise HostToolError(f"Unknown tool: {name}")
        try:
            value = fn(**dict(arguments or {}))
            if inspect.isawaitable(value):
                value = await value
        except Exception as exc:
            raise HostToolError(f"Error calling tool {name!r}: {exc}") from exc
        text = value if isinstance(value, str) else json.dumps(value, default=str)
        return SimpleNamespace(
            data=value,
            structured_content={"result": value},
            content=[SimpleNamespace(type="text", text=text)],
            is_error=False,
        )

    async def read_resource(self, uri: str) -> list[Any]:
        reader = self._resources.get(uri)
        if reader is None:
            raise HostToolError(f"Unknown resource: {uri}")
        value = reader()
        if inspect.isawaitable(value):
            value = await value
        return [SimpleNamespace(uri=uri, text=str(value), mime_type="text/plain")]
