"""Run the MCP v2 compatibility gateway over stateless Streamable HTTP."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .gateway import GraphOSV2Gateway, StreamableHTTPGraphOSClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphos-mcp-url", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8005, type=int)
    args = parser.parse_args()
    gateway = GraphOSV2Gateway(StreamableHTTPGraphOSClient(args.graphos_mcp_url))

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/mcp":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 1_048_576:
                    raise ValueError
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise ValueError
            except (ValueError, json.JSONDecodeError):
                response: dict[str, Any] = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32600, "message": "Invalid Request"},
                }
            else:
                response = asyncio.run(
                    gateway.dispatch(
                        request, authorization=self.headers.get("Authorization")
                    )
                )
            encoded = json.dumps(response, separators=(",", ":")).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, _format: str, *_args: Any) -> None:
            # Request lines can carry sensitive query values at reverse proxies;
            # the gateway emits no access logs by default.
            return

    logging.basicConfig(level=logging.WARNING)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
