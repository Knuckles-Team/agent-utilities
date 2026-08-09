#!/usr/bin/env python3
"""Reject HTTP client construction outside the governed egress factory."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "agent_utilities"
FACTORY = Path("agent_utilities/core/http_client.py")


def _tracked_or_walked_py_files(package: Path) -> list[Path]:
    """``.py`` files under ``package``, preferring the git-tracked set (BUG-043).

    A raw ``rglob`` also picks up gitignored, generated build output, which
    can carry a stale, already-fixed direct-HTTP-construction violation.
    Falls back to a filesystem walk only when ``package`` is not inside a
    git working tree (e.g. a synthetic test fixture).
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(package), "ls-files", "--", "*.py"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        tracked = [package / line for line in out.splitlines() if line]
        if tracked:
            return [p for p in tracked if p.is_file()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return sorted(package.rglob("*.py"))

# Files whose direct construction of a blocked HTTP client is intentional and
# justified inline at the call site, not just here (mirrors
# ``check_context_compiler_boundary.py``'s ``RAW_PROVIDER_ALLOWLIST`` idiom).
_ALLOWLIST = {
    # EphemeralLoopbackOidcAuthority._verified_json: a purpose-built,
    # process-local, ephemeral-CA-pinned mTLS loopback authority (skill
    # certification only) that deliberately reads AT MOST
    # ``_MAX_BODY_BYTES + 1`` bytes off the raw socket to bound memory use
    # regardless of what the (self-generated, self-controlled) peer sends --
    # a property ``core.http_client``'s httpx-based factory does not offer
    # as a simple synchronous read cap. Not general outbound egress: the
    # host is always the loopback authority this same process just bound.
    "agent_utilities/deployment/certification_oidc.py",
}

# D-CIM-4 perf: every ``_BLOCKED`` pattern is rooted at one of these six
# modules (``_qualified`` only ever resolves a call target through an alias
# that ``_imports`` traced back to an actual ``import``/``from ... import``
# statement). A file that never mentions one of these module names in its
# source text therefore cannot import, and so cannot call through, any
# blocked constructor — skip the alias-collection + call-scan walks for it.
# Purely a redundant-work cut: it never narrows what a real match requires,
# and ``ast.parse``'s own syntax-error detection still runs unconditionally.
_TRIGGER_MARKERS = ("aiohttp", "requests", "httpx", "urllib3", "urllib", "http.client")

_BLOCKED = {
    "aiohttp.ClientSession",
    "http.client.HTTPConnection",
    "http.client.HTTPSConnection",
    "httpx.AsyncClient",
    "httpx.Client",
    "requests.Session",
    "requests.delete",
    "requests.get",
    "requests.head",
    "requests.options",
    "requests.patch",
    "requests.post",
    "requests.put",
    "requests.request",
    "urllib.request.urlopen",
    "urllib3.PoolManager",
}


def _imports(tree: ast.AST) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                if item.asname:
                    # ``import a.b.c as x`` binds ``x`` directly to the full
                    # dotted target.
                    aliases[item.asname] = item.name
                else:
                    # ``import a.b.c`` (no asname) binds ONLY the top-level
                    # name ``a`` in the local namespace -- Python attribute
                    # access (``a.b.c``) reaches the submodule from there.
                    # The alias must therefore resolve to ``a``, not the full
                    # ``item.name`` ("a.b.c"): storing the full dotted path
                    # here made ``_qualified`` double-count the submodule
                    # segment (e.g. producing "http.client.client.HTTPConnection"
                    # instead of "http.client.HTTPConnection"), silently
                    # blinding this check to EVERY unaliased dotted-submodule
                    # import -- including both ``http.client.*`` and
                    # ``urllib.request.*`` entries in ``_BLOCKED``.
                    root = item.name.split(".")[0]
                    aliases[root] = root
        elif isinstance(node, ast.ImportFrom) and node.module:
            for item in node.names:
                aliases[item.asname or item.name] = f"{node.module}.{item.name}"
    return aliases


def _qualified(node: ast.expr, aliases: dict[str, str]) -> str:
    parts: list[str] = []
    cursor: ast.expr = node
    while isinstance(cursor, ast.Attribute):
        parts.append(cursor.attr)
        cursor = cursor.value
    if not isinstance(cursor, ast.Name):
        return ""
    root = aliases.get(cursor.id)
    if root is None:
        return ""
    return ".".join([root, *reversed(parts)])


def validate(package: Path = PACKAGE) -> list[str]:
    """Return stable violations for direct outbound HTTP constructors/calls."""

    errors: list[str] = []
    for path in _tracked_or_walked_py_files(package):
        try:
            relative = path.relative_to(ROOT)
        except ValueError:
            relative = Path(package.name) / path.relative_to(package)
        if relative == FACTORY or "__pycache__" in relative.parts:
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as exc:
            errors.append(f"{relative.as_posix()}: parse failed ({type(exc).__name__})")
            continue
        try:
            tree = ast.parse(source, filename=str(relative))
        except SyntaxError as exc:
            errors.append(f"{relative.as_posix()}: parse failed ({type(exc).__name__})")
            continue
        if not any(marker in source for marker in _TRIGGER_MARKERS):
            continue
        aliases = _imports(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            target = _qualified(node.func, aliases)
            if target in _BLOCKED and relative.as_posix() not in _ALLOWLIST:
                errors.append(
                    f"{relative.as_posix()}:{node.lineno}: direct {target}; "
                    "use core.http_client"
                )
    return errors


def main() -> int:
    errors = validate()
    if errors:
        print("HTTP egress boundary failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("HTTP egress boundary passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
