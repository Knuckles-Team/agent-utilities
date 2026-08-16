#!/usr/bin/env python3
"""Reject ``print()``/``sys.stdout.write()`` in the served MCP surface (B-19).

On the ``stdio`` transport, real stdout IS the JSON-RPC channel. Stdout purity
there is owned fd-level by the vendored MCP SDK's own ``stdio_server()`` (see
the "Stdio JSON-RPC purity" note in ``agent_utilities/mcp/server_factory.py``)
for the scope of ``mcp.run(transport="stdio")`` — but that protection only
starts once serving begins. Code that runs before it (engine bootstrap, a
co-service thread started moments before ``mcp.run()``) has no runtime net at
all, so the only real fix is: never write a ``print()`` into the served
surface in the first place. Catching that here, at commit time, replaces the
deleted ``protect_stdio_jsonrpc()`` process-wide runtime monkeypatch (B-19)
with the design the program calls for: a cheap static check in the fast
pre-commit tier instead of intercepting ``print`` at runtime.

Scope: ``agent_utilities/mcp/`` — the package that literally IS the MCP
server (``kg_server.py``, ``harness_server.py``, ``server_factory.py``,
``co_service_supervisor.py``, ``multiplexer.py``, ``tools/*``, …) — not the
whole ``agent_utilities`` tree, most of which is CLI/deployment tooling that
prints to a human's terminal and is never reached by a process serving MCP
over stdio. A handful of doc/report generator scripts inside ``mcp/`` are the
same kind of human-facing CLI tool (they run standalone via
``if __name__ == "__main__":``, never imported by the served surface) and are
explicitly excluded below.

Usage:
  python3 scripts/check_no_stdout_writes.py

Exit 0 = no stdout writes found in the served surface, 1 = violation(s) found.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVED_PACKAGE = ROOT / "agent_utilities" / "mcp"

# Standalone, human-facing CLI/doc-generation tools that live inside
# ``agent_utilities/mcp/`` for packaging convenience but are never imported by
# a process serving MCP over stdio — each has its own ``if __name__ ==
# "__main__":`` entry point and is run directly by a developer, not by
# ``kg_server``/``harness_server``. Printing to a human's terminal is exactly
# what these are for.
_ALLOWLIST = {
    "agent_utilities/mcp/check_env_var_drift.py",
    "agent_utilities/mcp/readme_env_vars.py",
    "agent_utilities/mcp/readme_mcp_examples.py",
    "agent_utilities/mcp/readme_tools.py",
    "agent_utilities/mcp/skill_coverage.py",
}


def _iter_py_files(package: Path) -> list[Path]:
    return sorted(p for p in package.rglob("*.py") if p.is_file())


def _is_stdout_write(node: ast.Call) -> str | None:
    """Return a short description if ``node`` writes to stdout, else ``None``."""
    func = node.func
    if isinstance(func, ast.Name) and func.id == "print":
        for kw in node.keywords:
            if kw.arg == "file" and not (
                isinstance(kw.value, ast.Attribute)
                and kw.value.attr == "stdout"
                and isinstance(kw.value.value, ast.Name)
                and kw.value.value.id == "sys"
            ):
                # An explicit non-stdout file= target (e.g. file=sys.stderr) —
                # not a violation.
                return None
        return "print(...)"
    if (
        isinstance(func, ast.Attribute)
        and func.attr == "write"
        and isinstance(func.value, ast.Attribute)
        and func.value.attr == "stdout"
        and isinstance(func.value.value, ast.Name)
        and func.value.value.id == "sys"
    ):
        return "sys.stdout.write(...)"
    return None


def validate(package: Path) -> list[str]:
    """Return one message per stdout-writing call found under ``package``."""
    errors: list[str] = []
    for path in _iter_py_files(package):
        rel = path.relative_to(package.parents[1]).as_posix()
        if rel in _ALLOWLIST:
            continue
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                kind = _is_stdout_write(node)
                if kind is not None:
                    errors.append(f"{rel}:{node.lineno}: {kind} writes to stdout")
    return errors


def main() -> int:
    errors = validate(SERVED_PACKAGE)
    if errors:
        print(
            "Stdout write(s) found in the served MCP surface "
            f"({SERVED_PACKAGE.relative_to(ROOT)}):\n"
        )
        for error in errors:
            print(f"  {error}")
        print(
            "\nOn the stdio transport, real stdout IS the JSON-RPC channel — a "
            "stray print() here corrupts the protocol frame stream. Route "
            "diagnostics through `logging` (stderr) instead. See the 'Stdio "
            "JSON-RPC purity' note in agent_utilities/mcp/server_factory.py."
        )
        return 1
    print(f"OK — no stdout writes in {SERVED_PACKAGE.relative_to(ROOT)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
