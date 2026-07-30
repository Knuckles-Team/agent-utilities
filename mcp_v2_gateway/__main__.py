"""Run the MCP v2 compatibility gateway over stateless Streamable HTTP."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .gateway import (
    GraphOSV2Gateway,
    HTTPGatewayResponse,
    StreamableHTTPGateway,
    StreamableHTTPGraphOSClient,
)


class _GatewayHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphos-mcp-url", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8005, type=int)
    parser.add_argument(
        "--allowed-origin",
        action="append",
        default=[],
        help="Exact allowed Origin value; repeat for multiple trusted origins.",
    )
    parser.add_argument("--max-request-bytes", default=1_048_576, type=int)
    args = parser.parse_args()
    gateway = GraphOSV2Gateway(StreamableHTTPGraphOSClient(args.graphos_mcp_url))
    transport = StreamableHTTPGateway(
        gateway,
        allowed_origins=args.allowed_origin,
        max_request_bytes=args.max_request_bytes,
    )

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 0:
                    raise ValueError
            except ValueError:
                response = HTTPGatewayResponse(status_code=400)
            else:
                if length > args.max_request_bytes:
                    self.close_connection = True
                    self._send(
                        HTTPGatewayResponse(
                            status_code=413,
                            headers={"Cache-Control": "no-store"},
                        )
                    )
                    return
                body = self.rfile.read(length)
                response = asyncio.run(
                    transport.handle(
                        path=self.path,
                        headers=list(self.headers.items()),
                        body=body,
                    )
                )
            self._send(response)

        def do_GET(self) -> None:  # noqa: N802
            self._send(
                HTTPGatewayResponse(
                    status_code=405 if self.path == "/mcp" else 404,
                    headers={"Cache-Control": "no-store"},
                )
            )

        def do_DELETE(self) -> None:  # noqa: N802
            self.do_GET()

        def _send(self, response: HTTPGatewayResponse) -> None:
            encoded = (
                json.dumps(response.body, separators=(",", ":")).encode()
                if response.body is not None
                else b""
            )
            self.send_response(response.status_code)
            for name, value in response.headers.items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            if encoded:
                self.wfile.write(encoded)

        def log_message(self, _format: str, *_args: Any) -> None:
            # Request lines can carry sensitive query values at reverse proxies;
            # the gateway emits no access logs by default.
            return

    logging.basicConfig(level=logging.WARNING)
    server = _GatewayHTTPServer((args.host, args.port), Handler)

    def stop_server(_signum: int, _frame: Any) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop_server)
    signal.signal(signal.SIGINT, stop_server)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
