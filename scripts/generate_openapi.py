#!/usr/bin/env python3
"""Generate the published OpenAPI reference for the served Agent Web Dashboard.

The app already serves a live OpenAPI 3.1 document at ``/openapi.json`` (with
Swagger UI at ``/docs`` and ReDoc at ``/redoc``). None of that was previously
published in the documentation *site* — this script closes that gap by
projecting the same document, generated from code rather than hand-copied,
into two committed artifacts:

* ``docs/reference/openapi.json`` — the raw spec, byte-for-byte what
  ``app.openapi()`` returns (keys sorted, pretty-printed, no timestamp/host/
  build-path/version drift), so it can be diffed in review like any other
  generated artifact in this repo.
* ``docs/reference/api.md`` — a generated MkDocs page that renders it as a
  browsable catalog (grouped by top-level path) plus an explicit,
  generically-derived note about which mounted surfaces carry no OpenAPI
  schema yet (see :func:`schemaless_routes`) rather than presenting a partial
  spec as if it were the whole API.

Both artifacts are built the same way a drift guard in agent-webui itself
builds the app (``agent_webui/__tests__/test_canonical_gateway_mount.py``):
the real ``agent_webui.server.create_agent_web_app`` factory, a deterministic
``TestModel`` agent (no LLM credentials or network calls required), and a
loopback listener. Browser SSO is forced unconfigured so generation never
depends on ambient OIDC secrets reachable from the host running this script
-- that would make the artifact depend on *where* it was generated, which is
exactly the kind of drift this script exists to prevent. SSO is pure ASGI
middleware; it contributes no route and no schema either way, so this does
not change what gets documented.

Run ``python scripts/generate_openapi.py --write`` after any change that
could alter the served API surface. CI and pre-commit should use the
default, side-effect-free ``--check``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
SPEC_PATH = DOCS / "reference" / "openapi.json"
PAGE_PATH = DOCS / "reference" / "api.md"

_HTTP_METHODS = frozenset(
    {"get", "put", "post", "delete", "options", "head", "patch", "trace"}
)

_WRITE_CMD = "`python scripts/generate_openapi.py --write`"

# Sanity floor, not a coverage target. `agent_webui.server.create_agent_web_app`
# mounts several distinct optional/mandatory route groups (the service
# dashboard API, the canonical Knowledge Graph REST surface via
# `register_graph_routes`, ...) behind their own import/registration guards.
# A guard that fails soft still returns a *working* app -- just one missing
# an entire surface -- so `build_app()` here can succeed and `app.openapi()`
# can return a well-formed, syntactically valid spec that is nonetheless
# missing 50+ paths. That happened for real: a single shared
# `except ImportError` used to wrap both the optional dashboard import and
# the mandatory canonical KG REST registration in `create_agent_web_app`, so
# any ImportError in the dashboard import alone silently dropped
# `/api/registry/*` (12), `/api/ontology/*` (22), `/api/research/*` (7), and
# `/api/dashboard/*` (14) from the generated spec -- 55+ paths -- while this
# script still printed "Wrote ..." and exited 0. The committed spec normally
# carries ~190 paths; this floor sits comfortably above what remains after
# that specific truncation (~136) with headroom for ordinary path churn, so
# it fails on a dropped *surface*, not a dropped *route*.
_MIN_EXPECTED_PATHS = 150


class IncompleteSurfaceError(RuntimeError):
    """The built app's OpenAPI spec falls below the sanity floor.

    Raised instead of silently writing/checking a truncated spec -- see
    ``_MIN_EXPECTED_PATHS``.
    """


def _assert_surface_complete(spec: dict[str, Any]) -> None:
    """Fail closed if ``spec`` looks like a mounted surface went missing.

    Called on every ``--write`` and every ``--check`` (both funnel through
    ``_expected()``) so a truncated spec can never be committed as
    documentation nor silently pass validation.
    """
    path_count = len(spec.get("paths", {}))
    if path_count < _MIN_EXPECTED_PATHS:
        raise IncompleteSurfaceError(
            f"Refusing to use an OpenAPI spec with only {path_count} path(s) "
            f"(expected at least {_MIN_EXPECTED_PATHS}). This means a "
            "mounted API surface (the service dashboard API, the canonical "
            "KG REST surface, ...) failed to register on the built app -- "
            "not that the documented API legitimately shrank. Check "
            "agent_webui.server logs / build_app() output for a dashboard "
            "'not available' message or a canonical KG REST surface "
            "RuntimeError, fix the underlying mount failure, and rerun."
        )


def build_app() -> Any:
    """Construct the Agent Web Dashboard FastAPI app the way production does.

    See the module docstring for why SSO is forced unconfigured and the
    listener is pinned to loopback: both keep this deterministic across
    machines without changing the route/schema surface being documented.
    """
    from agent_webui.server import create_agent_web_app
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    with patch("agent_webui.oidc_session.load_settings", return_value=None):
        return create_agent_web_app(
            Agent(TestModel()),
            {"get_path": lambda value: value},
            listener_host="127.0.0.1",
        )


def openapi_spec() -> dict[str, Any]:
    """Return the app's own ``app.openapi()`` document, unmodified."""
    app = build_app()
    spec = app.openapi()
    if not isinstance(spec, dict):  # pragma: no cover - defensive
        raise TypeError("app.openapi() did not return a mapping")
    return spec


def render_spec(spec: dict[str, Any]) -> str:
    """Deterministic serialization: sorted keys, no trailing whitespace drift."""
    return json.dumps(spec, indent=2, sort_keys=True) + "\n"


def schemaless_routes(app: Any, spec: dict[str, Any]) -> tuple[int, list[str]]:
    """Generically derive which mounted routes carry no OpenAPI schema.

    Several fleet route tables (the canonical Knowledge Graph REST surface
    under ``/api/graph``, the fleet supervisory plane under ``/api/engine``,
    and others — ``register_graph_routes`` / ``kg_server._mount_rest_routes``)
    are mounted as raw Starlette routes so the SAME route code serves both
    webui and gateway clients; raw Starlette routes carry no request/response
    types for FastAPI to introspect, so they are invisible to
    ``app.openapi()``. This walks the live ``app.routes`` and diffs them
    against the spec's own path set instead of hardcoding a route list, so
    the count shrinks on its own as parallel work adds typed schemas to those
    surfaces -- it never needs to be hand-maintained here.

    Excluded, because they are not a coverage gap: FastAPI's own
    ``include_in_schema=False`` meta-routes (``/openapi.json``, ``/docs``,
    ``/redoc``, ...) and WebSocket routes, which structurally have no REST
    operation schema to begin with.
    """
    from fastapi.routing import APIRoute, APIWebSocketRoute

    schema_paths = set(spec.get("paths", {}))
    prefixes: set[str] = set()
    count = 0
    for route in app.routes:
        path = getattr(route, "path", None)
        if not path or path in schema_paths:
            continue
        if isinstance(route, (APIRoute, APIWebSocketRoute)):
            continue
        if getattr(route, "include_in_schema", True) is False:
            continue
        count += 1
        parts = [part for part in path.strip("/").split("/") if part]
        prefix = "/" + "/".join(parts[:2]) if len(parts) > 1 else "/" + "".join(parts)
        prefixes.add(prefix)
    return count, sorted(prefixes)


def _group_key(path: str) -> str:
    parts = [part for part in path.strip("/").split("/") if part]
    if len(parts) > 1:
        return "/" + "/".join(parts[:2])
    return "/" + "".join(parts) if parts else "/"


def _operations(spec: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    """Flatten ``spec['paths']`` to ``(group, method, path, summary)`` rows."""
    rows: list[tuple[str, str, str, str]] = []
    for path, methods in spec.get("paths", {}).items():
        if not isinstance(methods, dict):
            continue
        for method, operation in methods.items():
            if method.lower() not in _HTTP_METHODS or not isinstance(operation, dict):
                continue
            summary = str(operation.get("summary") or "").strip()
            rows.append((_group_key(path), method.upper(), path, summary))
    rows.sort(key=lambda row: (row[0], row[2], row[1]))
    return rows


def _escape_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").strip()


def render_page(
    spec: dict[str, Any], *, schemaless_count: int, schemaless_prefixes: list[str]
) -> str:
    rows = _operations(spec)
    info = spec.get("info", {}) if isinstance(spec.get("info"), dict) else {}
    title = str(info.get("title") or "")
    openapi_version = str(spec.get("openapi") or "")
    summaries = sum(1 for _, _, _, summary in rows if summary)

    grouped: dict[str, list[tuple[str, str, str]]] = {}
    for group, method, path, summary in rows:
        grouped.setdefault(group, []).append((method, path, summary))

    lines = [
        "# API Reference",
        "",
        f"> **GENERATED — do not edit by hand.** Run {_WRITE_CMD}. Source: "
        "the served app's own `app.openapi()`, projected from "
        "`agent_webui.server.create_agent_web_app` exactly as production "
        "builds it — not hand-copied.",
        "",
        f'This is the generated reference for **{title or "the served app"}** '
        f"(OpenAPI {openapi_version or 'unknown'}) — {len(spec.get('paths', {}))} "
        f"paths, {len(rows)} operations, {summaries} with a summary. The running "
        "server's own interactive `/docs` (Swagger UI) and `/redoc` pages "
        "reflect the identical document live; this page is the static, "
        "diffable snapshot checked into docs so it cannot silently drift from "
        "the code that generates it.",
        "",
        "## Coverage note",
        "",
    ]

    if schemaless_count:
        prefix_list = ", ".join(f"`{prefix}`" for prefix in schemaless_prefixes)
        lines += [
            f"**This reference does not yet cover the whole live API.** "
            f"{schemaless_count} route(s) across {len(schemaless_prefixes)} "
            "mounted path prefix(es) below are served by the app but carry no "
            "OpenAPI schema, so `app.openapi()` — and therefore this page — "
            "cannot see them:",
            "",
            prefix_list,
            "",
            "Those surfaces (the canonical Knowledge Graph REST/graph-execution "
            "routes, among others) are mounted as raw Starlette routes rather "
            "than typed FastAPI routers, so they carry no request/response "
            "schema for FastAPI to introspect — not an omission from this "
            "generator. This count is recomputed from the live app every time "
            f"{_WRITE_CMD} runs (see `schemaless_routes()` in "
            "`scripts/generate_openapi.py`), so as typed schemas land for "
            "those routes this note shrinks and eventually disappears on its "
            "own, with no route list to hand-maintain here.",
            "",
        ]
    else:
        lines += [
            "Every route mounted on the live app now carries an OpenAPI "
            "schema — this page covers the whole served surface.",
            "",
        ]

    lines += [
        "The full spec (not just this rendering of it) is published at "
        "[`openapi.json`](openapi.json).",
        "",
        "## Operations by path prefix",
        "",
        "| Path prefix | Operations |",
        "|---|---|",
    ]
    for group in sorted(grouped):
        lines.append(f"| `{_escape_cell(group)}` | {len(grouped[group])} |")
    lines.append("")

    lines += [
        "## All operations",
        "",
        "| Method | Path | Summary |",
        "|---|---|---|",
    ]
    for group in sorted(grouped):
        for method, path, summary in grouped[group]:
            lines.append(
                f"| {method} | `{_escape_cell(path)}` | {_escape_cell(summary)} |"
            )
    lines.append("")
    return "\n".join(lines)


def _expected() -> dict[Path, str]:
    spec = openapi_spec()
    _assert_surface_complete(spec)
    app = build_app()
    schemaless_count, schemaless_prefixes = schemaless_routes(app, spec)
    return {
        SPEC_PATH: render_spec(spec),
        PAGE_PATH: render_page(
            spec,
            schemaless_count=schemaless_count,
            schemaless_prefixes=schemaless_prefixes,
        ),
    }


def check() -> list[str]:
    errors: list[str] = []
    for path, content in _expected().items():
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            errors.append(
                f"{path.relative_to(ROOT).as_posix()} is stale; run {_WRITE_CMD}"
            )
    return errors


def write() -> None:
    for path, content in _expected().items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write", action="store_true", help="Regenerate the OpenAPI artifacts."
    )
    parser.add_argument(
        "--check", action="store_true", help="Validate without writing (default)."
    )
    args = parser.parse_args()

    try:
        if args.write:
            write()
            if not args.check:
                print(
                    f"Wrote {SPEC_PATH.relative_to(ROOT)} and {PAGE_PATH.relative_to(ROOT)}."
                )
                return 0

        errors = check()
    except IncompleteSurfaceError as exc:
        print(
            "OpenAPI reference contract FAILED (refusing to write/check a "
            "truncated spec):"
        )
        print(f"  - {exc}")
        return 1

    if errors:
        print("OpenAPI reference contract FAILED:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("OpenAPI reference contract PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
