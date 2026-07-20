#!/usr/bin/env python3
"""Discover and pin governed connector MCP tool schemas.

The command starts (or connects to) each connector through the operator's MCP
configuration, calls ``list_tools`` once, and writes only deterministic SHA-256
fingerprints to the connector-owned ``connectors/tool_schema_fingerprints.json``.
No endpoint, credential, response record, local path, or trace content is
written.  Discovery is sequential by design so fleet certification stays safe
on resource-constrained workstations.

Examples::

    python scripts/certify_connector_tool_schemas.py --connector-root CONNECTOR
    python scripts/certify_connector_tool_schemas.py --all --agents-root AGENTS
    python scripts/certify_connector_tool_schemas.py --all --agents-root AGENTS --check
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_utilities.protocols.source_connectors.connectors.mcp_package import (  # noqa: E402
    _load_mcp_config,
)
from agent_utilities.protocols.source_connectors.tool_schema import (  # noqa: E402
    canonical_input_schema,
    compatibility_fingerprint,
)


def _module_dir(connector_root: Path) -> Path:
    matches = sorted(
        path.parent.parent
        for path in connector_root.glob("*/connectors/mcp_source_presets.json")
    )
    if len(matches) != 1:
        raise RuntimeError(
            "connector must contain exactly one */connectors/mcp_source_presets.json"
        )
    return matches[0]


def _presets(connector_root: Path) -> tuple[Path, dict[str, Any]]:
    module_dir = _module_dir(connector_root)
    path = module_dir / "connectors" / "mcp_source_presets.json"
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise RuntimeError("mcp_source_presets.json must be an object")
    presets = {
        str(name): value
        for name, value in loaded.items()
        if not str(name).startswith("_") and isinstance(value, dict)
    }
    if not presets:
        raise RuntimeError("connector declares no source presets")
    return module_dir, presets


def _target(server: str, config: dict[str, Any]) -> Any:
    entry = config.get(server) or config.get(f"{server}-mcp")
    if not isinstance(entry, dict):
        raise RuntimeError(f"MCP configuration has no server entry for {server!r}")
    # Pass the entry to FastMCP without logging or serializing it; it may contain
    # secret references or process environment values.
    return {"mcpServers": {server: dict(entry)}}


async def _discover(connector_root: Path) -> dict[str, str]:
    module_dir, presets = _presets(connector_root)
    del module_dir
    server_names = {str(preset.get("server") or "") for preset in presets.values()}
    if "" in server_names or len(server_names) != 1:
        raise RuntimeError("one connector bundle must resolve to exactly one MCP server")
    server = next(iter(server_names))
    config = _load_mcp_config()

    try:
        from fastmcp import Client
    except ImportError as exc:
        raise RuntimeError("install agent-utilities[mcp] to certify live MCP schemas") from exc

    async with Client(_target(server, config)) as client:
        result = await client.list_tools()
    tools = getattr(result, "tools", result)
    by_name = {str(getattr(tool, "name", "")): tool for tool in tools}
    required = sorted({str(preset.get("tool") or "") for preset in presets.values()})
    missing = [name for name in required if not name or name not in by_name]
    if missing:
        raise RuntimeError(f"live server is missing signed source tools: {missing!r}")
    return {
        name: compatibility_fingerprint(
            name, canonical_input_schema(by_name[name], include_presentation=False)
        )
        for name in required
    }


def _local_factory_target(connector_root: Path, server: str) -> str:
    """Resolve a connector's MCP module without importing the package here.

    Fleet certification must not accumulate 50+ connector dependency graphs in
    one interpreter.  The selected module is imported by one short-lived worker
    process and discarded before the next connector is inspected.
    """

    try:
        import tomllib

        project = tomllib.loads(
            (connector_root / "pyproject.toml").read_text(encoding="utf-8")
        ).get("project", {})
    except (OSError, ValueError) as exc:
        raise RuntimeError("connector project metadata is unavailable") from exc
    scripts = project.get("scripts", {}) if isinstance(project, dict) else {}
    if not isinstance(scripts, dict):
        raise RuntimeError("connector project scripts are unavailable")

    candidates: list[tuple[int, str]] = []
    for name, target in scripts.items():
        if not isinstance(name, str) or not isinstance(target, str) or ":" not in target:
            continue
        lowered = name.lower()
        score = 0
        if lowered == server.lower():
            score += 30
        if "mcp" in lowered:
            score += 20
        if target.rsplit(":", 1)[1] == "mcp_server":
            score += 100
        candidates.append((score, target.split(":", 1)[0]))
    if not candidates:
        raise RuntimeError("connector declares no importable MCP script")
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][1]


async def _discover_local(connector_root: Path) -> dict[str, str]:
    """Inspect the real local FastMCP surface in an isolated process.

    This is the low-resource certification path for a source checkout fleet. It
    calls ``get_mcp_instance`` and ``list_tools`` but never starts a transport or
    contacts an upstream system. Only tool names and structural digests cross
    the process boundary.
    """

    connector_root = connector_root.resolve()
    _, presets = _presets(connector_root)
    server_names = {str(preset.get("server") or "") for preset in presets.values()}
    if "" in server_names or len(server_names) != 1:
        raise RuntimeError("one connector bundle must resolve to exactly one MCP server")
    server = next(iter(server_names))
    required = sorted({str(preset.get("tool") or "") for preset in presets.values()})
    if any(not name for name in required):
        raise RuntimeError("connector preset has no tool name")
    module = _local_factory_target(connector_root, server)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (
            str(connector_root),
            str(ROOT),
            env.get("PYTHONPATH", ""),
        )
        if part
    )
    env["AGENT_UTILITIES_CERTIFY_MODULE"] = module
    env["AGENT_UTILITIES_CERTIFY_TOOLS"] = json.dumps(required)
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(Path(__file__).resolve()),
        "--local-worker",
        env=env,
        cwd=connector_root,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError("local MCP schema worker failed")
    marker = b"AGENT_UTILITIES_SCHEMA_CERTIFICATION="
    lines = [line for line in stdout.splitlines() if line.startswith(marker)]
    if len(lines) != 1:
        raise RuntimeError("local MCP schema worker returned no certification payload")
    payload = json.loads(lines[0][len(marker) :].decode("utf-8"))
    if not isinstance(payload, dict) or set(payload) != set(required):
        raise RuntimeError("local MCP schema worker returned an incomplete tool set")
    return {str(name): str(digest) for name, digest in payload.items()}


async def _local_worker() -> int:
    """Short-lived local FastMCP inspector used by :func:`_discover_local`."""

    import importlib

    module_name = os.environ.get("AGENT_UTILITIES_CERTIFY_MODULE", "")
    requested = json.loads(os.environ.get("AGENT_UTILITIES_CERTIFY_TOOLS", "[]"))
    if not module_name or not isinstance(requested, list):
        raise RuntimeError("local certification worker contract is incomplete")
    sys.argv = ["connector-schema-certification"]
    module = importlib.import_module(module_name)
    factory = getattr(module, "get_mcp_instance", None)
    if not callable(factory):
        raise RuntimeError("connector MCP module has no get_mcp_instance factory")
    created = factory()
    if isinstance(created, tuple):
        mcp = next(
            (item for item in created if callable(getattr(item, "list_tools", None))),
            None,
        )
    else:
        mcp = created
    if mcp is None or not callable(getattr(mcp, "list_tools", None)):
        raise RuntimeError("connector MCP factory returned no FastMCP surface")
    result = await mcp.list_tools()
    tools = getattr(result, "tools", result)
    by_name = {str(getattr(tool, "name", "")): tool for tool in tools}
    missing = [name for name in requested if name not in by_name]
    if missing:
        raise RuntimeError("connector MCP surface is missing a source tool")
    payload = {
        name: compatibility_fingerprint(
            name, canonical_input_schema(by_name[name], include_presentation=False)
        )
        for name in requested
    }
    print(
        "AGENT_UTILITIES_SCHEMA_CERTIFICATION="
        + json.dumps(payload, sort_keys=True, separators=(",", ":"))
    )
    return 0


def _render(connector: str, tools: dict[str, str]) -> str:
    return json.dumps(
        {
            "schema_version": "1",
            "connector": connector,
            "algorithm": "agent-utilities:mcp-tool-schema-compat:v1",
            "tools": dict(sorted(tools.items())),
        },
        indent=2,
        sort_keys=True,
    ) + "\n"


async def _one(connector_root: Path, *, check: bool, local: bool) -> None:
    connector_root = connector_root.resolve()
    module_dir, _ = _presets(connector_root)
    output = module_dir / "connectors" / "tool_schema_fingerprints.json"
    discovered = (
        await _discover_local(connector_root)
        if local
        else await _discover(connector_root)
    )
    rendered = _render(connector_root.name, discovered)
    if check:
        if not output.is_file() or output.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(
                f"{connector_root.name}: committed tool-schema fingerprints differ from live MCP"
            )
        print(f"verified {connector_root.name}: live schema pins match", flush=True)
        return
    output.write_text(rendered, encoding="utf-8")
    print(
        f"certified {connector_root.name}: "
        f"wrote {len(json.loads(rendered)['tools'])} tool pin(s)",
        flush=True,
    )


async def _main(args: argparse.Namespace) -> int:
    if args.all:
        if args.agents_root is None:
            raise RuntimeError("--all requires --agents-root")
        roots = sorted(
            path.parent.parent.parent
            for path in args.agents_root.glob("*/*/connectors/mcp_source_presets.json")
        )
    elif args.connector_root is not None:
        roots = [args.connector_root]
    else:
        raise RuntimeError("pass --connector-root or --all --agents-root")
    failures: list[str] = []
    for root in roots:
        # Deliberately sequential: each connector process is fully closed before
        # the next one starts, avoiding fleet-wide memory/CPU spikes.
        try:
            await _one(root, check=args.check, local=args.local)
        except Exception as exc:  # noqa: BLE001 - report provider, never raw details
            failures.append(root.name)
            print(
                f"certification failed for {root.name} ({type(exc).__name__})",
                file=sys.stderr,
                flush=True,
            )
    if failures:
        print(
            f"connector schema certification failed for {len(failures)} provider(s)",
            file=sys.stderr,
        )
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connector-root", type=Path)
    parser.add_argument("--agents-root", type=Path)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--local",
        action="store_true",
        help=(
            "inspect each checkout's real FastMCP list_tools surface in one "
            "short-lived process (no transport or upstream connection)"
        ),
    )
    parser.add_argument("--local-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.local_worker:
            return asyncio.run(_local_worker())
        return asyncio.run(_main(args))
    except Exception as exc:  # noqa: BLE001 - never echo config, paths, or endpoints
        print(
            f"connector schema certification failed ({type(exc).__name__})",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
