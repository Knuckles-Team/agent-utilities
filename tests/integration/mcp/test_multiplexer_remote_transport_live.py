"""Live proof that au's hardened HTTP client drives MCP SDK v2's remote transport.

CONCEPT:AU-ECO.mcp.protocol-compat-bridge

The multiplexer's remote path (the whole deployed fleet —
``deploy/mcp-fleet.registry.yml`` defaults every service to
``transport: streamable-http``) now hands SDK v2's ``streamable_http_client`` a
client built by :func:`agent_utilities.core.http_client.create_async_http_client`
with the full security posture (pinned TLS trust, DNS-pinned egress, no ambient
proxy, no redirects). SDK v2 type-hints that parameter as ``httpx2.AsyncClient``
while au's whole client stack — including the DNS-pinning and air-gap guard
transports — is built on ``httpx``. The two are *different packages*, so the
handover works only by duck typing and nothing but a live round trip proves it
still does.

This spawns a real fastmcp-4 streamable-http server and drives initialize +
list_tools + call_tool through exactly that client.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_SERVER_SCRIPT = textwrap.dedent(
    """
    import sys

    from fastmcp import FastMCP

    mcp = FastMCP("au-multiplexer-live-http")

    @mcp.tool
    def echo(value: str) -> str:
        \"\"\"Echo the value back.\"\"\"
        return f"echo:{value}"

    if __name__ == "__main__":
        mcp.run(
            transport="http",
            host="127.0.0.1",
            port=int(sys.argv[1]),
            show_banner=False,
        )
    """
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_port(port: int, deadline_s: float = 30.0) -> None:
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        with socket.socket() as sock:
            sock.settimeout(0.5)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.2)
    raise AssertionError(f"fastmcp-4 http server never opened port {port}")


@pytest.fixture
def live_http_server(tmp_path: Path):
    port = _free_port()
    script = tmp_path / "fastmcp4_http_server.py"
    script.write_text(_SERVER_SCRIPT, encoding="utf-8")
    process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
        [sys.executable, str(script), str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_port(port)
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - stubborn child
            process.kill()


@pytest.mark.asyncio
async def test_hardened_client_round_trips_sdk_v2_streamable_http(
    live_http_server: str,
) -> None:
    """au's ``create_async_http_client`` output is accepted by SDK v2's
    ``streamable_http_client`` and completes a full MCP round trip."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    from agent_utilities.core.http_client import create_async_http_client

    http_client = create_async_http_client(
        timeout=30.0,
        headers={"X-Fleet-Tenant": "acme"},
        trust_env=False,
        follow_redirects=False,
        pin_egress=True,
        allow_loopback=True,
        allowed_private_hosts=("127.0.0.1",),
    )
    async with http_client:
        async with streamable_http_client(
            live_http_server, http_client=http_client
        ) as streams:
            # SDK v2 yields (read, write); v1's third get_session_id is gone.
            assert len(streams) == 2
            read_stream, write_stream = streams
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools = await session.list_tools()
                assert "echo" in {tool.name for tool in tools.tools}
                result = await session.call_tool("echo", {"value": "live"})
                assert result.content[0].text == "echo:live"
