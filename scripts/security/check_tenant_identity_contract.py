#!/usr/bin/env python3
"""Static fail-closed contract for served graph tenant identity.

This gate binds the REST and MCP surfaces to one server-minted actor/session
authority. It intentionally checks source structure instead of importing the
runtime, so it remains deterministic and cannot contact an identity provider.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


class TenantIdentityContractError(ValueError):
    """A served identity or tenant-isolation invariant has regressed."""


_FILES = {
    "gateway": Path("agent_utilities/gateway/graph_api.py"),
    "mcp": Path("agent_utilities/mcp/kg_server.py"),
    "identity": Path("agent_utilities/security/request_identity.py"),
}
_RETIRED_AUTHORITY_SWITCHES = (
    "KG_" + "SERVED_PROFILE",
    "KG_AUTH_REQUIRED",
    "KG_ACL_DEFAULT_ALLOW",
    "MCP_ALLOW_UNAUTHENTICATED_" + "REMOTE",
)


def _function(tree: ast.AST, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            return node
    raise TenantIdentityContractError(f"required function {name} is absent")


def _call_lines(function: ast.AST, name: str) -> list[int]:
    lines: list[int] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if isinstance(target, ast.Name) and target.id == name:
            lines.append(node.lineno)
        elif isinstance(target, ast.Attribute) and target.attr == name:
            lines.append(node.lineno)
    return sorted(lines)


def _calls(function: ast.AST, name: str) -> list[ast.Call]:
    """Return calls to ``name`` inside ``function`` in source order."""

    calls: list[ast.Call] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if isinstance(target, ast.Name) and target.id == name:
            calls.append(node)
        elif isinstance(target, ast.Attribute) and target.attr == name:
            calls.append(node)
    return sorted(calls, key=lambda node: node.lineno)


def _read_sources(root: Path) -> dict[str, str]:
    sources: dict[str, str] = {}
    for name, relative in _FILES.items():
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise TenantIdentityContractError(f"required {name} source is absent")
        source = path.read_text(encoding="utf-8")
        if len(source.encode("utf-8")) > 2 * 1024 * 1024:
            raise TenantIdentityContractError(f"required {name} source is oversized")
        sources[name] = source
    return sources


def check_sources(sources: dict[str, str]) -> dict[str, Any]:
    """Check an in-memory source set, including meta-test mutations."""

    if set(sources) != set(_FILES):
        raise TenantIdentityContractError("identity source set is incomplete")
    try:
        trees = {name: ast.parse(source) for name, source in sources.items()}
    except SyntaxError as exc:
        raise TenantIdentityContractError(
            "identity source is not valid Python"
        ) from exc

    combined = "\n".join(sources.values())
    retired = [name for name in _RETIRED_AUTHORITY_SWITCHES if name in combined]
    if retired:
        raise TenantIdentityContractError("retired authority switch is present")

    gateway = sources["gateway"]
    register = _function(trees["gateway"], "register_graph_routes")
    middleware_lines = _call_lines(register, "add_middleware")
    if (
        not middleware_lines
        or "app.add_middleware(ActorIdentityMiddleware)" not in gateway
        or "from agent_utilities.security.request_identity import ActorIdentityMiddleware"
        not in gateway
    ):
        raise TenantIdentityContractError("graph routes lack server-minted identity")

    mcp = sources["mcp"]
    execute = _function(trees["mcp"], "_execute_tool")
    reject_lines = _call_lines(execute, "_reject_caller_authority")
    scope_lines = _call_lines(execute, "verified_tool_session_scope")
    if not reject_lines or not scope_lines or min(reject_lines) >= min(scope_lines):
        raise TenantIdentityContractError(
            "MCP dispatch does not reject caller authority before session scope"
        )
    if (
        '_CALLER_AUTHORITY_FIELDS = frozenset({"_actor", "_roles", "_tenant"})'
        not in mcp
    ):
        raise TenantIdentityContractError(
            "caller authority field rejection is incomplete"
        )
    scope = _function(trees["mcp"], "verified_tool_session_scope")
    scope_source = ast.get_source_segment(mcp, scope) or ""
    for marker in (
        "current_session()",
        "Verified GraphSession required",
        "session.engine_verified_context()",
        "current_actor()",
    ):
        if marker not in scope_source:
            raise TenantIdentityContractError(
                "verified MCP session boundary is incomplete"
            )

    identity = sources["identity"]
    if "UNAUTHENTICATED_PATHS: frozenset[str] = HEALTH_PATHS" not in identity:
        raise TenantIdentityContractError(
            "unauthenticated route set exceeds health probes"
        )
    mint = _function(trees["identity"], "mint_graph_session")
    projections = _calls(mint, "_mint_graph_session")
    if len(projections) != 1:
        raise TenantIdentityContractError(
            "served session minting does not use the verified projection exactly once"
        )
    projection = projections[0]
    projection_keywords = {keyword.arg for keyword in projection.keywords}
    if (
        not projection.args
        or not isinstance(projection.args[0], ast.Name)
        or projection.args[0].id != "actor"
        or projection_keywords != {"audience", "policy_version"}
    ):
        raise TenantIdentityContractError(
            "served session minting does not bind verified authority inputs"
        )

    verified_mint = _function(trees["identity"], "_mint_graph_session")
    mint_source = ast.get_source_segment(identity, verified_mint) or ""
    for marker in (
        "if not actor.authenticated:",
        'if not str(actor.actor_id or "").strip():',
        "if not tenant:",
        "if not audience or not policy_version:",
        "tenant_graph_name(tenant",
        "GraphSession(",
        "with use_session(unrouted):",
        "resolve_placement(",
        "return unrouted.with_route(",
    ):
        if marker not in mint_source:
            raise TenantIdentityContractError("session minting authority is incomplete")
    middleware = _function(trees["identity"], "__call__")
    middleware_source = ast.get_source_segment(identity, middleware) or ""
    if (
        "if path not in UNAUTHENTICATED_PATHS:" not in middleware_source
        or "actor_from_bearer_token" not in middleware_source
        or "mint_graph_session(actor)" not in middleware_source
    ):
        raise TenantIdentityContractError(
            "request identity middleware does not fail closed"
        )
    return {"ok": True, "boundaries": ["rest", "mcp", "tenant-session"]}


def check(root: Path) -> dict[str, Any]:
    """Check the repository identity boundary."""

    return check_sources(_read_sources(root))


def self_check(root: Path) -> None:
    """Prove representative identity regressions are rejected."""

    sources = _read_sources(root)
    mutations: list[dict[str, str]] = []
    missing_middleware = dict(sources)
    missing_middleware["gateway"] = missing_middleware["gateway"].replace(
        "app.add_middleware(ActorIdentityMiddleware)",
        "pass  # identity middleware removed",
        1,
    )
    mutations.append(missing_middleware)
    caller_authority = dict(sources)
    caller_authority["mcp"] = caller_authority["mcp"].replace(
        "_reject_caller_authority(kwargs)",
        "pass  # caller authority accepted",
        1,
    )
    mutations.append(caller_authority)
    unverified_mint = dict(sources)
    unverified_mint["identity"] = unverified_mint["identity"].replace(
        "if not actor.authenticated:",
        "if actor.authenticated:  # inverted verified-identity boundary",
        1,
    )
    mutations.append(unverified_mint)
    retired_switch = dict(sources)
    retired_switch["identity"] += "\n" + "KG_" + "SERVED_PROFILE = True\n"
    mutations.append(retired_switch)
    for mutation in mutations:
        try:
            check_sources(mutation)
        except TenantIdentityContractError:
            continue
        raise TenantIdentityContractError(
            "identity gate self-check accepted a regression"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="check-tenant-identity-contract")
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = check(args.repository_root)
        if args.self_check:
            self_check(args.repository_root)
            report["selfCheck"] = True
    except Exception as exc:  # noqa: BLE001 - privacy-safe gate boundary
        print(json.dumps({"ok": False, "error": type(exc).__name__}, sort_keys=True))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
