#!/usr/bin/env python3
"""Enforce the staged HTTP transport strangler and dependency closure.

NE-015 deliberately keeps ``httpx`` installed for this migration wave: the
resolved lock still contains third-party packages that require its concrete
types, while the application surface is being moved behind
``httpsupport``/``core.http_client``.  This gate protects that boundary in two
ways:

* every existing direct ``httpx``/``httpx2`` import is an explicit, reviewed
  exception (a new production import fails immediately); and
* the lock cannot stop shipping ``httpx`` while any resolved package still
  names it as a dependency.

The allowlist is intentionally exact rather than a directory exemption.  A
future migration removes one entry and this file's stale-entry check requires
the corresponding evidence to be updated at the same time.  This is a ratchet
for the strangler seam, not a claim that all concrete imports can be removed in
one change.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "agent_utilities"
PYPROJECT = ROOT / "pyproject.toml"
LOCK = ROOT / "uv.lock"

# Each entry is a deliberate concrete-package dependency.  Keep the reason
# next to the exception so a later strangler wave can remove it with its
# owning behavior, rather than broadening a global exemption.
DIRECT_IMPORT_ALLOWLIST: dict[str, str] = {
    "agent_utilities/core/embedding_utilities.py": "TYPE_CHECKING-only SDK client annotations",
    "agent_utilities/core/http_client.py": "httpx transport/factory authority",
    "agent_utilities/core/model_factory.py": "provider SDK timeout/limits concrete types",
    "agent_utilities/gateway_client/client.py": "TYPE_CHECKING-only injected transport annotation",
    "agent_utilities/harness/memorydata/client.py": "lazy optional dependency error taxonomy",
    "agent_utilities/httpsupport/client.py": "public BaseApiClient concrete response/transport contract",
    "agent_utilities/httpsupport/httpx_adapter.py": "sanctioned httpx adapter",
    "agent_utilities/httpsupport/httpx2_adapter.py": "sanctioned httpx2 adapter",
    "agent_utilities/knowledge_graph/backends/sparql/jena_fuseki_backend.py": "lazy optional Fuseki dependency guard",
    "agent_utilities/knowledge_graph/pipeline/phases/embedding.py": "legacy core-factory HTTPError translation",
    "agent_utilities/kvcache/remote_backend.py": "public client injection and concrete limits/auth/error types",
    "agent_utilities/mcp/client_credentials.py": "local httpx.Auth implementation",
    "agent_utilities/mcp/httpx_boundary.py": "httpx-to-httpx2 auth boundary adapter",
    "agent_utilities/mcp/remote_oauth_broker.py": "public OAuth client/auth flow concrete types",
    "agent_utilities/mcp/toolset_factory.py": "rebuilds local httpx.Timeout at SDK boundary",
    "agent_utilities/protocols/source_connectors/http_safety.py": "HTTPX URL/exception and transport compatibility",
    "agent_utilities/security/browser_auth.py": "lazy OAuth/PKCE transport pending auth-specific parity",
    "agent_utilities/security/oauth_client_credentials.py": "local httpx.Auth implementation",
    "agent_utilities/security/oidc_discovery.py": "TYPE_CHECKING-only response annotations",
}


def _top_level_name(module: str) -> str:
    return module.split(".", 1)[0]


def _direct_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = _top_level_name(alias.name)
                if root in {"httpx", "httpx2"}:
                    found.add(root)
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = _top_level_name(node.module)
            if root in {"httpx", "httpx2"}:
                found.add(root)
    return found


def _runtime_files(package: Path) -> list[Path]:
    if not package.is_dir():
        raise FileNotFoundError(f"runtime package is not a directory: {package}")
    return sorted(path for path in package.rglob("*.py") if "__pycache__" not in path.parts)


def _import_inventory(package: Path) -> dict[str, set[str]]:
    inventory: dict[str, set[str]] = {}
    for path in _runtime_files(package):
        imports = _direct_imports(path)
        if imports:
            try:
                relative = path.relative_to(ROOT)
            except ValueError:
                relative = Path(package.name) / path.relative_to(package)
            inventory[relative.as_posix()] = imports
    return inventory


def _dependency_name(value: Any) -> str | None:
    if isinstance(value, dict):
        value = value.get("name")
    if not isinstance(value, str):
        return None
    return re.split(r"[<>=!~;\[]", value, maxsplit=1)[0].strip().lower() or None


def _lock_consumers(lock_path: Path) -> tuple[bool, set[str]]:
    data = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    packages = data.get("package", [])
    has_httpx = False
    consumers: set[str] = set()
    for package in packages:
        if not isinstance(package, dict):
            continue
        name = str(package.get("name", "")).lower()
        if name == "httpx":
            has_httpx = True
            continue
        dependencies = list(package.get("dependencies", []))
        optional = package.get("optional-dependencies", {})
        if isinstance(optional, dict):
            for values in optional.values():
                if isinstance(values, list):
                    dependencies.extend(values)
        metadata = package.get("metadata", {})
        if isinstance(metadata, dict):
            dependencies.extend(metadata.get("requires-dist", []))
        if any(_dependency_name(item) == "httpx" for item in dependencies):
            consumers.add(name)
    return has_httpx, consumers


def _declared_httpx(pyproject_path: Path) -> bool:
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = data.get("project", {})
    values: list[Any] = list(project.get("dependencies", []))
    optional = project.get("optional-dependencies", {})
    if isinstance(optional, dict):
        for specs in optional.values():
            if isinstance(specs, list):
                values.extend(specs)
    return any(_dependency_name(value) == "httpx" for value in values)


def validate(
    *,
    package: Path = PACKAGE,
    pyproject: Path = PYPROJECT,
    lock: Path = LOCK,
) -> list[str]:
    """Return deterministic direct-import and dependency-closure violations."""

    inventory = _import_inventory(package)
    found_paths = set(inventory)
    allowlisted_paths = set(DIRECT_IMPORT_ALLOWLIST)
    errors: list[str] = []

    for path in sorted(found_paths - allowlisted_paths):
        modules = ", ".join(sorted(inventory[path]))
        errors.append(
            f"{path}: direct {modules} import; use agent_utilities.httpsupport "
            "or core.http_client, or add a reviewed concrete-boundary exception"
        )
    for path in sorted(allowlisted_paths - found_paths):
        errors.append(f"{path}: stale HTTP transport allowlist entry")

    has_httpx, consumers = _lock_consumers(lock)
    if consumers and not has_httpx:
        rendered = ", ".join(sorted(consumers))
        errors.append(
            "uv.lock omits httpx while resolved packages still depend on it: "
            f"{rendered}"
        )
    if (found_paths or consumers) and not _declared_httpx(pyproject):
        errors.append(
            "runtime HTTP consumers remain but pyproject.toml declares no httpx "
            "dependency; do not remove the package before the closure reaches zero"
        )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="check-http-transport-closure")
    parser.add_argument("--package", type=Path, default=PACKAGE)
    parser.add_argument("--pyproject", type=Path, default=PYPROJECT)
    parser.add_argument("--lock", type=Path, default=LOCK)
    args = parser.parse_args(argv)
    try:
        errors = validate(package=args.package, pyproject=args.pyproject, lock=args.lock)
    except (OSError, tomllib.TOMLDecodeError, SyntaxError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 1
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "ok": True,
                "allowlistedFiles": len(DIRECT_IMPORT_ALLOWLIST),
                "message": "HTTP transport closure passed",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
