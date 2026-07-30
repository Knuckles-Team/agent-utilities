"""In-process ACP JSON-RPC wire harness for integration tests."""

from __future__ import annotations

import asyncio
import contextlib
import socket
from types import TracebackType
from typing import Any

import acp
from acp.client.connection import ClientSideConnection


class WireClient(acp.Client):
    """Record session updates received through the real ACP codec."""

    def __init__(self) -> None:
        self.updates: list[Any] = []

    def on_connect(self, connection: Any) -> None:
        return None

    async def session_update(
        self,
        session_id: str,
        update: Any,
        **kwargs: Any,
    ) -> None:
        self.updates.append(update)


class WireAgent:
    """Connect an ACP adapter and client through the SDK's JSON-RPC router."""

    def __init__(self, adapter: acp.Agent, client: WireClient | None = None) -> None:
        self._adapter = adapter
        self._client = client or WireClient()
        self._server: asyncio.Task[None] | None = None
        self._writers: tuple[asyncio.StreamWriter, asyncio.StreamWriter] | None = None

    async def __aenter__(self) -> tuple[ClientSideConnection, WireClient]:
        agent_socket, client_socket = socket.socketpair()
        agent_reader, agent_writer = await asyncio.open_connection(sock=agent_socket)
        client_reader, client_writer = await asyncio.open_connection(sock=client_socket)
        self._writers = (client_writer, agent_writer)
        self._server = asyncio.create_task(
            acp.run_agent(
                self._adapter,
                input_stream=agent_writer,
                output_stream=agent_reader,
                use_unstable_protocol=True,
            )
        )
        connection = acp.connect_to_agent(
            self._client,
            input_stream=client_writer,
            output_stream=client_reader,
            use_unstable_protocol=True,
        )
        return connection, self._client

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        assert self._server is not None and self._writers is not None
        self._server.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._server
        for writer in self._writers:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
