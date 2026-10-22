"""In-run client tools: apply host selection and limits without local effects."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic_ai.toolsets import FunctionToolset

from agent_utilities import host_ports


@dataclass(frozen=True)
class ClientToolLimits:
    """Received ceilings; defaults apply only when the host omits limits."""

    request_bytes: int = 65_536
    output_bytes: int = 65_536
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        for value in (self.request_bytes, self.output_bytes):
            if type(value) is not int or value <= 0:
                raise ValueError("client byte limits must be positive integers")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("client timeout must be finite and positive")


class ClientCapabilityToolset(FunctionToolset[Any]):
    """Forward only host-selected tools over one already bound client session.

    Search and patch are explicit host extensions; the host must refuse them
    when the connected client did not declare those capabilities. No ACP
    client capability, permission, or workspace grant is inferred here.
    """

    def __init__(
        self,
        transport: host_ports.ClientCapabilityTransport | None,
        selected: tuple[str, ...],
        *,
        limits: ClientToolLimits | None = None,
        run_id: str,
    ) -> None:
        super().__init__()
        if transport is None:
            raise host_ports.ClientCapabilityRefused("client_disconnected")
        self._transport = transport
        self._limits = limits or ClientToolLimits()
        self._run_id = run_id
        available: dict[str, Callable[..., Awaitable[str]]] = {
            "read_file": self.read_file,
            "search_files": self.search_files,
            "apply_patch": self.apply_patch,
            "write_file": self.write_file,
            "run_command": self.run_command,
        }
        if len(set(selected)) != len(selected) or set(selected) - available.keys():
            raise ValueError("unknown or duplicate client tool selection")
        for name in selected:
            self.add_function(available[name], name=name)

    async def _call(self, method: str, params: dict[str, Any]) -> str:
        request_bytes = len(json.dumps(params, ensure_ascii=False).encode("utf-8"))
        outcome, output = "request_limit", ""
        if request_bytes <= self._limits.request_bytes:
            outcome, output = await self._dispatch(method, params)
        host_ports.emit_run_event(
            {
                "type": "client_capability",
                "run_id": self._run_id,
                "method": method,
                "request_bytes": request_bytes,
                "output_bytes": len(output.encode("utf-8")),
                "outcome": outcome,
            }
        )
        if outcome != "completed":
            return json.dumps({"error": "client_capability_refused", "reason": outcome})
        return output

    async def _dispatch(self, method: str, params: dict[str, Any]) -> tuple[str, str]:
        try:
            async with asyncio.timeout(self._limits.timeout_seconds):
                output = await self._transport.call(method, params)
        except host_ports.ClientCapabilityRefused as exc:
            return exc.reason, ""
        except TimeoutError:
            # The host may have dispatched the effect. Do not retry or claim rollback.
            return "timeout", ""
        except Exception:
            return "transport_error", ""
        if not isinstance(output, str):
            return "invalid_response", ""
        if len(output.encode("utf-8")) > self._limits.output_bytes:
            return "output_limit", ""
        return "completed", output

    async def read_file(self, path: str) -> str:
        """Read text from the connected client's workspace."""
        return await self._call("fs/read_text_file", {"path": path})

    async def search_files(self, pattern: str) -> str:
        """Search via the client's explicitly declared search extension."""
        return await self._call("fs/search_files", {"pattern": pattern})

    async def apply_patch(self, patch: str) -> str:
        """Submit a patch for host approval and client application."""
        return await self._call("fs/apply_patch", {"patch": patch})

    async def write_file(self, path: str, content: str) -> str:
        """Submit text for host approval and client writing."""
        return await self._call(
            "fs/write_text_file", {"path": path, "content": content}
        )

    async def run_command(self, command: str, args: list[str] | None = None) -> str:
        """Submit a command for host approval and bounded client execution."""
        return await self._call(
            "terminal/create",
            {
                "command": command,
                "args": args or [],
                "outputByteLimit": self._limits.output_bytes,
                "timeoutSeconds": self._limits.timeout_seconds,
            },
        )
