#!/usr/bin/env python3
"""OpenAPI coverage ratchet — every mounted HTTP route must appear in the
generated OpenAPI spec, with a beautiful (summary + description) operation.

THE MEASURED PROBLEM
---------------------
The live spec (``/openapi.json``) is OpenAPI 3.1.0 and looks reasonably
documented — but two mounting styles coexist on the same app:

* ``agent_utilities/gateway/registry_api.py`` (and ``ontology_api.py`` /
  ``research_api.py`` / ``fleet.py``) build a real FastAPI ``APIRouter`` and
  call ``add_api_route(...)`` -> FastAPI's ``get_openapi()`` walks
  ``isinstance(route, fastapi.routing.APIRoute)`` and documents it.
* ``agent_utilities/mcp/kg_server.py``'s ``_mount_rest_routes`` (see the
  ``route()`` helper at the raw ``app.add_route(prefix + path, handler,
  methods=methods)`` call) builds plain ``starlette.routing.Route`` objects.
  FastAPI's schema generator skips anything that is not an ``APIRoute`` —
  full stop — so these routes are live, callable, and 100% invisible to
  ``/docs``. This is the entire canonical Knowledge Graph REST surface
  (``/api/graph/*``, ``/api/sessions``, ``/api/goals``, ``/api/tools``, the
  local SPARQL endpoint...): a developer opening ``/docs`` sees only the
  FastAPI-native routers and never learns the raw-mounted surface exists.

This script closes that gap permanently: it enumerates every concrete route
actually mounted on the app (recursing into ``Mount``s), diffs that table
against ``app.openapi()``'s path table, and ratchets the two resulting
finding sets (routes missing from the spec entirely, and *documented*
operations missing a ``summary`` or a non-empty ``description`` — an
undescribed operation is a finding, not a pass) against a committed
baseline, exactly like ``scripts/check_wiring.py``'s
``scripts/wire_first_baseline.json`` idiom: new entries beyond the baseline
fail the gate; the baseline only ever shrinks via ``--update-baseline``,
after a genuine fix.

WHAT "THE APP" IS HERE, AND WHY (IMPORTANT SCOPE NOTE)
--------------------------------------------------------
Building the literal top-level production entry point
(``agent_utilities.server.create_agent_server``) is too heavy for a
pre-commit gate: it calls ``ensure_local_engine()`` and spawns a real
lifecycle-coupled embedded Knowledge Graph engine process. Instead this
script calls ``agent_utilities.server.app.build_agent_app`` directly — the
SAME function ``create_agent_server`` calls to actually construct the
``FastAPI`` instance — with ``create_agent`` (LLM/provider bootstrap) and
``agent_to_epistemic_a2a`` (the FastA2A protocol sub-app, which needs a live
broker/storage + process identity) patched out, mirroring the exact patch
set already proven safe in
``tests/integration/core/test_api_endpoints.py``'s ``client`` fixture. This
is a real production code path, not a hand-assembled stand-in — every
router registration, every ``kg_server._mount_rest_routes`` call, every
gateway registrar (``register_graph_routes`` -> fleet/ontology/research/
registry/remote-oauth) runs for real, unmocked.

What this DOES NOT cover, on purpose:

* ``enable_web_ui=False`` — **agent-webui is out of scope for this lane**
  (five parallel lanes own it right now; this script's owner may not edit
  it). Its own FastAPI sub-app (mounted at ``/`` in production, including
  its own ``add_pydantic_routes``-style raw mounts and whatever serves
  ``/api/chat``) is therefore NOT enumerated or measured here. Audit that
  surface with its own gate, in agent-webui's own repo/lane.
* The ``/api/enhanced`` facade router (``server/routers/enhanced.py``) is
  only included inside the ``enable_web_ui`` branch of
  ``build_agent_app``, so with web UI disabled it is not mounted at all —
  which is correct for this gate's purpose: that facade is being deleted,
  not a canonical surface worth ratcheting.
* The ``/a2a`` mount is a **separate, independent** FastA2A protocol
  application with its own OpenAPI schema at its own ``/a2a/openapi.json``.
  Starlette/FastAPI's ``app.openapi()`` never spans a ``Mount`` boundary —
  a sub-app's routes are structurally never going to appear in the parent
  spec, by framework design, not by the bug this gate exists to catch. It
  is walked (so nothing is silently invisible to a human reading this
  script) but excluded from the ratchet via the explicit allowlist below;
  audit it separately at its own ``/openapi.json``.

If ``agent_utilities.server`` cannot be imported at all (e.g. a minimal
profile with the ``[server]`` extra absent), the gate says so loudly and
exits 0 — NOT ENFORCED, never a silent pass disguised as a real one (see
``check_liveness.py``'s docstring for why that distinction matters here).

Usage::

    python scripts/check_openapi_coverage.py                  # report + ratchet gate
    python scripts/check_openapi_coverage.py --json
    python scripts/check_openapi_coverage.py --update-baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
# Ensure the repo root resolves FIRST so `agent_utilities` imports below always
# bind to THIS checkout's source, never a stale/partial editable install
# elsewhere on the interpreter's path (see check_surface_parity.py).
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _gate_interpreter import require_project_interpreter  # noqa: E402

# Re-exec under this repo's declared interpreter BEFORE importing
# agent_utilities.server — under an out-of-contract python this gate would
# not fail, it would just stop functioning (see scripts/_gate_interpreter.py).
require_project_interpreter(ROOT)

BASELINE = Path(__file__).resolve().parent / "openapi_coverage_baseline.json"

#: Explicit, commented allowlist (never a silent skip — item 5 of the gate
#: spec). Every entry names exactly why it can never gain a documented
#: OpenAPI operation, or why documenting it here would be structurally
#: wrong. Exact-path entries match one path; prefix entries (ending "*")
#: match any path starting with that prefix.
ALLOWLIST: tuple[tuple[str, str], ...] = (
    # FastAPI's own doc/schema endpoints — self-referential, not part of the
    # API being documented (there is no operation for "the docs page").
    ("/openapi.json", "the OpenAPI document itself"),
    ("/docs", "Swagger UI"),
    ("/docs/oauth2-redirect", "Swagger UI OAuth2 redirect helper"),
    ("/redoc", "ReDoc UI"),
    # Prometheus text-exposition endpoint. Deliberately mounted with
    # `include_in_schema=False` (agent_utilities/gateway/graph_api.py) —
    # this is a universal Prometheus convention (scrape target, not a JSON
    # REST operation), not the raw-Starlette-mount bug this gate targets.
    ("/metrics", "Prometheus scrape endpoint (include_in_schema=False by design)"),
    # A separate, independent FastA2A protocol application mounted at
    # /a2a. Starlette Mounts are an OpenAPI boundary FastAPI's app.openapi()
    # never spans by design — /a2a/* has its own complete OpenAPI schema at
    # its own /a2a/openapi.json. Not the bug this gate exists to catch; walked
    # for visibility, never silently, but excluded from the ratchet.
    ("/a2a/*", "separate FastA2A sub-app with its own independent OpenAPI schema"),
)

_HTTP_METHODS = frozenset(
    {"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE"}
)


def _allowlisted(path: str) -> str | None:
    """Return the matching allowlist reason for *path*, or None."""
    for entry, reason in ALLOWLIST:
        if entry.endswith("*"):
            if path.startswith(entry[:-1]):
                return reason
        elif path == entry:
            return reason
    return None


def _build_target_app() -> Any:
    """Build the real production FastAPI app, LLM/A2A bootstrap patched out.

    See the module docstring's "WHAT 'THE APP' IS HERE" section for exactly
    what this covers and what it deliberately does not.
    """
    from unittest.mock import MagicMock, patch

    from fastapi import FastAPI

    mock_agent = MagicMock()
    mock_agent.name = "OpenAPICoverageGateProbe"
    # A trivial stand-in FastA2A app — its content is irrelevant since /a2a/*
    # is allowlisted out of the ratchet (see ALLOWLIST above); it only needs
    # to be mountable.
    a2a_stub = FastAPI()

    with (
        patch(
            "agent_utilities.server.app.create_agent",
            return_value=(mock_agent, []),
        ),
        patch("agent_utilities.core.workspace.initialize_workspace"),
        patch(
            "agent_utilities.server.app.load_identity",
            return_value={"name": "OpenAPICoverageGateProbe"},
        ),
        patch("agent_utilities.server.app.get_skills_path", return_value=None),
        patch(
            "agent_utilities.protocols.a2a_epistemic.agent_to_epistemic_a2a",
            return_value=a2a_stub,
        ),
    ):
        from agent_utilities.server import build_agent_app

        app = build_agent_app(
            provider="test-provider",
            model_id="test-model",
            host="127.0.0.1",  # loopback: avoids the non-loopback TLS guard
            enable_web_ui=False,  # agent-webui is out of scope for this lane
            enable_acp=False,
            enable_otel=False,
            name="OpenAPICoverageGateProbe",
        )
    return app


def enumerate_routes(app: Any, prefix: str = "") -> list[dict[str, Any]]:
    """Walk ``app.routes``, recursing into ``Mount``s, and return every
    concrete (method, path) endpoint as a dict with enough detail to explain
    a finding: ``{"method", "path", "kind", "mount_of"}``.

    ``kind`` is ``"api_route"`` for a FastAPI ``APIRoute`` (schema-capable by
    construction) or ``"starlette_route"`` for a plain Starlette ``Route``
    (structurally invisible to ``app.openapi()`` — this is exactly the bug
    class this gate exists to catch). A ``Mount`` whose target exposes no
    ``.routes`` (e.g. ``StaticFiles``) is recorded once as a synthetic
    ``"opaque_mount"`` entry rather than silently dropped.
    """
    from fastapi.routing import APIRoute
    from starlette.routing import Mount, Route

    entries: list[dict[str, Any]] = []
    for route in getattr(app, "routes", []):
        if isinstance(route, Mount):
            full_prefix = prefix + route.path
            sub_app = route.app
            if hasattr(sub_app, "routes"):
                entries.extend(enumerate_routes(sub_app, prefix=full_prefix))
            else:
                entries.append(
                    {
                        "method": "*",
                        "path": full_prefix,
                        "kind": "opaque_mount",
                        "mount_of": type(sub_app).__name__,
                    }
                )
            continue

        methods = sorted(getattr(route, "methods", None) or [])
        path = prefix + (getattr(route, "path", "") or "")
        if not methods or not path:
            continue
        kind = "api_route" if isinstance(route, APIRoute) else (
            "starlette_route" if isinstance(route, Route) else type(route).__name__
        )
        for method in methods:
            # HEAD is auto-added by Starlette for every GET route and never
            # gets its own OpenAPI operation — comparing it would manufacture
            # a permanent, unfixable "finding" for every single GET.
            if method == "HEAD":
                continue
            entries.append(
                {"method": method, "path": path, "kind": kind, "mount_of": None}
            )
    return entries


def _documented_pairs(spec: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """Map (METHOD, path) -> operation object for every documented operation."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for path, methods in spec.get("paths", {}).items():
        if not isinstance(methods, dict):
            continue
        for method, op in methods.items():
            if method.upper() not in _HTTP_METHODS or not isinstance(op, dict):
                continue
            out[(method.upper(), path)] = op
    return out


def compute_findings(app: Any | None = None) -> dict[str, Any]:
    """Diff *app*'s route table against its own generated spec and return the
    two ratcheted finding sets plus the raw counts used in the report.

    ``app`` defaults to the real production app (:func:`_build_target_app`);
    passing an app explicitly is how tests exercise this against a small,
    deliberately-constructed app instead of paying for the full build.
    """
    if app is None:
        app = _build_target_app()
    route_table = enumerate_routes(app)
    spec = app.openapi()
    documented = _documented_pairs(spec)

    excluded: list[dict[str, Any]] = []
    undocumented: set[str] = set()
    seen_table_pairs: set[tuple[str, str]] = set()
    for entry in route_table:
        if entry["kind"] == "opaque_mount":
            reason = _allowlisted(entry["path"]) or "opaque Mount (no .routes)"
            excluded.append({**entry, "reason": reason})
            continue
        pair = (entry["method"], entry["path"])
        seen_table_pairs.add(pair)
        reason = _allowlisted(entry["path"])
        if reason is not None:
            excluded.append({**entry, "reason": reason})
            continue
        if pair not in documented:
            undocumented.add(f"{pair[0]} {pair[1]}")

    missing_description: set[str] = set()
    for (method, path), op in documented.items():
        if _allowlisted(path) is not None:
            continue
        summary = str(op.get("summary") or "").strip()
        description = str(op.get("description") or "").strip()
        if not summary or not description:
            missing_description.add(f"{method} {path}")

    return {
        "undocumented_routes": undocumented,
        "missing_description": missing_description,
        "total_routes": len(seen_table_pairs),
        "total_documented_operations": len(documented),
        "total_excluded": len(excluded),
        "excluded": excluded,
    }


def _load_baseline() -> dict[str, list[str]]:
    if not BASELINE.exists():
        return {"undocumented_routes": [], "missing_description": []}
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def _write_baseline(findings: dict[str, Any]) -> None:
    BASELINE.write_text(
        json.dumps(
            {
                "_comment": (
                    "OpenAPI coverage ratchet baseline (scripts/"
                    "check_openapi_coverage.py). Frozen backlog of routes "
                    "mounted on the production app but absent from its "
                    "generated OpenAPI spec (the raw-Starlette-add_route "
                    "bug), and documented operations missing a summary or "
                    "description. Burn down toward empty as the parallel "
                    "gateway/registry_api schema lanes land. New entries "
                    "beyond this baseline fail the gate. Refresh with "
                    "--update-baseline only after genuinely fixing or "
                    "deliberately accepting a new backlog item, never to "
                    "silence a regression."
                ),
                "undocumented_routes": sorted(findings["undocumented_routes"]),
                "missing_description": sorted(findings["missing_description"]),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


#: The two categories ratcheted independently, wire-first-gate style: a NEW
#: entry in *either* fails the gate on its own; they never offset each other.
_CATEGORIES = ("undocumented_routes", "missing_description")


def evaluate(
    findings: dict[str, Any], baseline: dict[str, list[str]]
) -> dict[str, Any]:
    """Pure ratchet decision — no I/O, no printing.

    Returns a structured result (``exit_code`` plus, per category, ``total``/
    ``baselined``/``new``/``fixed`` lists) so both ``main()`` and tests act on
    real data rather than parsed log text. ``exit_code`` is 1 iff at least one
    category has a NEW (unbaselined) finding.
    """
    result: dict[str, Any] = {}
    any_new = False
    for category in _CATEGORIES:
        current = findings[category]
        base = set(baseline.get(category, []))
        new = sorted(current - base)
        fixed = sorted(base - current)
        result[category] = {
            "total": len(current),
            "baselined": len(base & current),
            "new": new,
            "fixed": fixed,
        }
        any_new = any_new or bool(new)
    result["exit_code"] = 1 if any_new else 0
    return result


def _print_category(label: str, category_result: dict[str, Any]) -> None:
    new = category_result["new"]
    fixed = category_result["fixed"]
    if new:
        print(f"\n{label}: {len(new)} NEW finding(s) beyond baseline:")
        for item in new:
            print(f"  + {item}")
    removed_note = f", {len(fixed)} fixed since baseline" if fixed else ""
    print(
        f"{label}: {category_result['total']} total "
        f"({category_result['baselined']} baselined{removed_note})."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenAPI coverage ratchet gate.")
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Write the current finding sets as the new baseline and exit 0.",
    )
    args = parser.parse_args()

    try:
        findings = compute_findings()
    except ImportError as exc:
        # Say NOT ENFORCED, loudly — mirrors check_liveness.py's contract: a
        # skip must never be mistaken for a pass.
        print(
            "openapi-coverage gate NOT ENFORCED (exit 0, nothing was checked): "
            f"could not build the production app ({exc!r}). Install the "
            "`server` extra this repo declares to enable this gate."
        )
        return 0

    if args.update_baseline:
        _write_baseline(findings)
        print(
            "openapi-coverage baseline updated: "
            f"{len(findings['undocumented_routes'])} undocumented route(s), "
            f"{len(findings['missing_description'])} operation(s) missing "
            f"summary/description -> {BASELINE.name}"
        )
        return 0

    baseline = _load_baseline()
    result = evaluate(findings, baseline)

    if args.json:
        print(
            json.dumps(
                {
                    "total_routes": findings["total_routes"],
                    "total_documented_operations": findings[
                        "total_documented_operations"
                    ],
                    "total_excluded": findings["total_excluded"],
                    "undocumented_routes": sorted(findings["undocumented_routes"]),
                    "missing_description": sorted(findings["missing_description"]),
                    "baseline": baseline,
                    "result": result,
                },
                indent=2,
            )
        )
        return result["exit_code"]

    print("=" * 72)
    print("OpenAPI coverage gate")
    print("=" * 72)
    print(
        f"routes on app: {findings['total_routes']}   "
        f"documented operations: {findings['total_documented_operations']}   "
        f"excluded (allowlisted/opaque mounts): {findings['total_excluded']}"
    )

    _print_category(
        "Undocumented routes (mounted, absent from OpenAPI spec)",
        result["undocumented_routes"],
    )
    _print_category(
        "Documented operations missing summary/description",
        result["missing_description"],
    )

    print()
    if result["exit_code"]:
        print(
            "FAIL — OpenAPI coverage REGRESSED vs baseline. Give the new "
            "route(s) a real FastAPI-schema-capable mount (APIRouter."
            "add_api_route, not Starlette add_route) and/or a summary + "
            "description, or — if genuinely accepted — "
            "`python scripts/check_openapi_coverage.py --update-baseline`."
        )
        return 1
    print("OK — no NEW OpenAPI coverage regression beyond the frozen baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
