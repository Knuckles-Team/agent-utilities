#!/usr/bin/env python
"""Validate an mcp_config.json against a Caddyfile (and optionally live endpoints).

The MCP multiplexer fans out to many child servers declared by ``url`` in an
``mcp_config.json``. Managed ``url`` values are streamable-http endpoints
reverse-proxied by Caddy. This tool reconciles the two:

  * every streamable-http ``url`` host must map to a Caddy reverse_proxy route;
  * Caddy routes under an optional managed suffix with no config entry are reported;
  * with ``--live`` each endpoint is probed with an MCP ``initialize`` and must
    answer (HTTP 200), catching routed-but-dead backends (e.g. a 502).

Exit code is non-zero if any invalid/unreachable entry is found, so it can guard
a pre-commit hook or CI job. Stdlib only — no third-party dependencies.

CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog — dynamic MCP tool gateway

Examples
--------
    python scripts/validate_mcp_config.py \
        --config mcp_config_claude.json --caddyfile services/caddy/Caddyfile
    python scripts/validate_mcp_config.py --config mcp_config_claude.json \
        --caddyfile services/caddy/Caddyfile --live --json
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import ssl
import urllib.error
import urllib.request
from urllib.parse import urlsplit

# A streamable-http transport is implied by a ``url`` or one of these transports.
_REMOTE_TRANSPORTS = {"streamable-http", "streamable_http", "http", "sse"}
_INITIALIZE = json.dumps(
    {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "validate_mcp_config", "version": "0"},
        },
    }
).encode()
_MAX_PROBE_RESPONSE_BYTES = 1 * 1024 * 1024


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        raise RuntimeError("remote redirect was rejected")


def _tls_context() -> ssl.SSLContext:
    cafile = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    capath = os.environ.get("SSL_CERT_DIR")
    return ssl.create_default_context(cafile=cafile or None, capath=capath or None)


def _validated_probe_host(url: str) -> str:
    try:
        parsed = urlsplit(url)
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError:
        return ""
    host = (parsed.hostname or "").lower().rstrip(".")
    if (
        parsed.scheme not in {"http", "https"}
        or not host
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or not (1 <= port <= 65_535)
        or len(url) > 8_192
    ):
        return ""
    if parsed.scheme == "http":
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = host == "localhost"
        if not loopback:
            return ""
    return host


def parse_caddy_hosts(caddyfile_text: str) -> dict[str, str | None]:
    """Map every Caddy site address → its reverse_proxy upstream (or None).

    Handles the standard ``host { ... reverse_proxy upstream ... }`` block form,
    including multiple space/comma-separated addresses on one site block.
    """
    hosts: dict[str, str | None] = {}
    for match in re.finditer(
        r"(?ms)^[ \t]*([^\n{#][^\n{]*?)\s*\{(.*?)^\}", caddyfile_text
    ):
        addrs, body = match.group(1), match.group(2)
        rp = re.search(r"reverse_proxy\s+([^\s\n]+)", body)
        upstream = rp.group(1) if rp else None
        for addr in re.split(r"[,\s]+", addrs.strip()):
            host = _host_of(addr)
            if host:
                hosts[host] = upstream
    return hosts


def _host_of(url_or_addr: str) -> str:
    """Extract the bare hostname from a URL or Caddy site address."""
    s = url_or_addr.strip()
    s = s.split("://", 1)[-1]  # drop scheme
    s = s.split("/", 1)[0]  # drop path
    s = s.split(":", 1)[0]  # drop port
    return s


def config_url_entries(config: dict) -> dict[str, str]:
    """Server-name → url for every streamable-http child in the config."""
    out: dict[str, str] = {}
    for name, cfg in (config.get("mcpServers") or {}).items():
        url = (cfg or {}).get("url", "")
        transport = str((cfg or {}).get("transport", "")).lower()
        if url or transport in _REMOTE_TRANSPORTS:
            if url:
                out[name] = url
    return out


def live_probe(url: str, timeout: float) -> tuple[bool, str]:
    """POST an MCP ``initialize`` and report (ok, detail). A healthy
    streamable-http server answers 200 (text/event-stream)."""
    host = _validated_probe_host(url)
    if not host or not 0.1 <= timeout <= 120:
        return False, "endpoint policy rejected"
    req = urllib.request.Request(
        url,
        data=_INITIALIZE,
        method="POST",
        headers={
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        },
    )
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=_tls_context()),
        _RejectRedirects(),
    )
    try:
        with opener.open(req, timeout=timeout) as resp:
            if _validated_probe_host(resp.geturl()) != host:
                return False, "endpoint origin changed"
            if len(resp.read(_MAX_PROBE_RESPONSE_BYTES + 1)) > _MAX_PROBE_RESPONSE_BYTES:
                return False, "response exceeded boundary"
            return (200 <= resp.status < 300), f"HTTP {resp.status}"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except Exception:  # connection refused, DNS, timeout, …
        return False, "request failed"


def validate(
    config: dict,
    caddy_hosts: dict[str, str | None],
    *,
    live: bool = False,
    timeout: float = 10.0,
    managed_suffix: str = "",
) -> dict:
    """Reconcile config urls against Caddy; optionally live-probe each endpoint."""
    entries = config_url_entries(config)
    invalid: dict[str, str] = {}  # url host not routed by Caddy
    unreachable: dict[str, str] = {}  # routed but the endpoint did not answer
    ok: list[str] = []

    for name, url in sorted(entries.items()):
        host = _host_of(url)
        if host not in caddy_hosts:
            invalid[name] = url
            continue
        if live:
            healthy, detail = live_probe(url, timeout)
            if not healthy:
                unreachable[name] = f"{url} ({detail})"
                continue
        ok.append(name)

    # Managed Caddy routes with no config entry — a coverage gap.
    config_hosts = {_host_of(u) for u in entries.values()}
    suffix = managed_suffix.strip().lower()
    missing = sorted(
        h
        for h in caddy_hosts
        if suffix and h.lower().endswith(suffix) and h not in config_hosts
    )

    return {
        "total": len(entries),
        "ok": ok,
        "invalid": invalid,
        "unreachable": unreachable,
        "missing_from_config": missing,
        "passed": not invalid and not unreachable,
    }


def _render(report: dict, live: bool) -> str:
    lines = [
        f"Checked {report['total']} streamable-http url entries "
        f"({len(report['ok'])} ok, {len(report['invalid'])} invalid"
        + (f", {len(report['unreachable'])} unreachable" if live else "")
        + ")."
    ]
    if report["invalid"]:
        lines.append("\nINVALID — url host has no Caddy route:")
        lines += [f"  ✗ {n}: {u}" for n, u in sorted(report["invalid"].items())]
    if report["unreachable"]:
        lines.append("\nUNREACHABLE — routed by Caddy but endpoint did not answer:")
        lines += [f"  ✗ {n}: {d}" for n, d in sorted(report["unreachable"].items())]
    if report["missing_from_config"]:
        lines.append("\nCADDY routes with no config entry (coverage gap):")
        lines += [f"  • {h}" for h in report["missing_from_config"]]
    lines.append("\nPASS ✓" if report["passed"] else "\nFAIL ✗")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True, help="path to mcp_config*.json")
    p.add_argument("--caddyfile", required=True, help="path to the Caddyfile")
    p.add_argument(
        "--live", action="store_true", help="also probe each endpoint (MCP initialize)"
    )
    p.add_argument("--timeout", type=float, default=10.0, help="per-probe timeout (s)")
    p.add_argument(
        "--managed-suffix",
        default="",
        help="optional hostname suffix whose Caddy routes must all appear in config",
    )
    p.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = p.parse_args(argv)

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)
    with open(args.caddyfile, encoding="utf-8") as f:
        caddy_hosts = parse_caddy_hosts(f.read())

    report = validate(
        config,
        caddy_hosts,
        live=args.live,
        timeout=args.timeout,
        managed_suffix=args.managed_suffix,
    )
    print(json.dumps(report, indent=2) if args.json else _render(report, args.live))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
