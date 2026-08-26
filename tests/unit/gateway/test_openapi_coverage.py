"""Proves the OpenAPI coverage ratchet gate (scripts/check_openapi_coverage.py)
actually catches a regression — not just that it runs.

CONTEXT: this repo has been bitten before by gates that look green while
enforcing nothing (three of them, per the workspace's own retro notes). Every
test here asserts on the checker's structured result / exit code, never on
log text, per that lesson.

These tests build small, hand-assembled Starlette/FastAPI apps — NOT the
production app — so they stay fast and prove the mechanism (route-table vs
spec diff, allowlist, ratchet) in isolation. The real production measurement
is `scripts/check_openapi_coverage.py`'s own `compute_findings()` default
path (`_build_target_app()`), exercised by running the script directly (see
the gate's own report), not by this unit test.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from fastapi import APIRouter, FastAPI
from starlette.responses import JSONResponse


def _load_module():
    """Load ``scripts/check_openapi_coverage.py`` by path (not a package
    import) — matches the existing ``tests/unit/scripts/test_check_ontology.py``
    convention for testing a standalone gate script directly."""
    source = Path(__file__).parents[3] / "scripts" / "check_openapi_coverage.py"
    spec = importlib.util.spec_from_file_location("check_openapi_coverage", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gate():
    return _load_module()


# --- small hand-built apps -------------------------------------------------


async def _raw_handler(request):  # noqa: ARG001 - Starlette handler signature
    return JSONResponse({"ok": True})


def _app_with_undocumented_raw_route() -> FastAPI:
    """A FastAPI app with ONE route mounted the buggy way: raw Starlette
    ``add_route`` (exactly ``kg_server.py``'s ``_mount_rest_routes`` pattern) —
    structurally invisible to ``app.openapi()``."""
    app = FastAPI(title="probe")
    app.add_route("/api/graph/query", _raw_handler, methods=["POST"])
    return app


def _app_with_documented_route() -> FastAPI:
    """The same operation, mounted the correct way: a real FastAPI
    ``APIRouter.add_api_route`` with a summary + description, exactly like
    ``registry_api.py``."""
    app = FastAPI(title="probe")
    router = APIRouter()

    async def query_handler() -> dict[str, Any]:
        return {"ok": True}

    router.add_api_route(
        "/api/graph/query",
        query_handler,
        methods=["POST"],
        summary="Query the graph",
        description="Runs a bilateral graph query against the active engine.",
    )
    app.include_router(router)
    return app


def _app_with_documented_but_undescribed_route() -> FastAPI:
    """A real FastAPI route (schema-capable) but with no summary/description
    — the SECOND finding category (not the raw-mount bug)."""
    app = FastAPI(title="probe")
    router = APIRouter()

    async def handler() -> dict[str, Any]:
        return {"ok": True}

    router.add_api_route("/api/undescribed", handler, methods=["GET"])
    app.include_router(router)
    return app


# --- enumerate_routes / compute_findings mechanics -------------------------


def test_raw_starlette_route_is_flagged_undocumented(gate):
    app = _app_with_undocumented_raw_route()
    findings = gate.compute_findings(app=app)
    assert "POST /api/graph/query" in findings["undocumented_routes"]


def test_equivalent_fastapi_route_is_not_flagged(gate):
    app = _app_with_documented_route()
    findings = gate.compute_findings(app=app)
    assert findings["undocumented_routes"] == set()
    assert "POST /api/graph/query" not in findings["missing_description"]


def test_documented_route_missing_summary_and_description_is_flagged(gate):
    app = _app_with_documented_but_undescribed_route()
    findings = gate.compute_findings(app=app)
    # A real APIRoute IS in the spec, so it must never appear as "undocumented"
    # (that finding is reserved for routes structurally absent from the spec).
    assert "GET /api/undescribed" not in findings["undocumented_routes"]
    assert "GET /api/undescribed" in findings["missing_description"]


def test_allowlisted_doc_paths_are_never_findings(gate):
    """FastAPI's own /openapi.json, /docs, /redoc are plain Starlette Routes
    (self-referential), so without the allowlist they would be flagged on
    EVERY app. Prove they are excluded, not accidentally passing because
    nothing else is being measured."""
    app = _app_with_undocumented_raw_route()  # any app with docs enabled
    findings = gate.compute_findings(app=app)
    for path in ("/openapi.json", "/docs", "/redoc"):
        for method in ("GET", "POST"):
            assert f"{method} {path}" not in findings["undocumented_routes"]


# --- the checker's actual FAIL/PASS verdict (exit code / structured result) -


def test_checker_fails_on_undocumented_route(gate):
    app = _app_with_undocumented_raw_route()
    findings = gate.compute_findings(app=app)
    empty_baseline: dict[str, list[str]] = {
        "undocumented_routes": [],
        "missing_description": [],
    }
    result = gate.evaluate(findings, empty_baseline)
    assert result["exit_code"] == 1
    assert "POST /api/graph/query" in result["undocumented_routes"]["new"]


def test_checker_passes_on_documented_route(gate):
    app = _app_with_documented_route()
    findings = gate.compute_findings(app=app)
    empty_baseline: dict[str, list[str]] = {
        "undocumented_routes": [],
        "missing_description": [],
    }
    result = gate.evaluate(findings, empty_baseline)
    assert result["exit_code"] == 0
    assert result["undocumented_routes"]["new"] == []
    assert result["missing_description"]["new"] == []


# --- the baseline ratchet itself --------------------------------------------


def test_new_route_beyond_baseline_fails(gate):
    """A baseline that already accepts a DIFFERENT undocumented route must
    still fail when a NEW, unbaselined one shows up."""
    app = _app_with_undocumented_raw_route()
    findings = gate.compute_findings(app=app)
    baseline_missing_this_route: dict[str, list[str]] = {
        "undocumented_routes": ["GET /api/some/other/pre-existing/route"],
        "missing_description": [],
    }
    result = gate.evaluate(findings, baseline_missing_this_route)
    assert result["exit_code"] == 1
    assert "POST /api/graph/query" in result["undocumented_routes"]["new"]


def test_same_count_against_matching_baseline_passes(gate):
    """The exact same finding, once accepted into the baseline, must pass —
    proving the ratchet is set-based (not merely a count), and that fixing
    it is provable too (baselined-but-no-longer-current -> reported as
    'fixed', never re-flagged as new)."""
    app = _app_with_undocumented_raw_route()
    findings = gate.compute_findings(app=app)
    baseline_with_this_route: dict[str, list[str]] = {
        "undocumented_routes": ["POST /api/graph/query"],
        "missing_description": [],
    }
    result = gate.evaluate(findings, baseline_with_this_route)
    assert result["exit_code"] == 0
    assert result["undocumented_routes"]["new"] == []
    assert result["undocumented_routes"]["baselined"] == 1

    # Now prove the OTHER direction: fixing the route (mounting it correctly)
    # against the SAME baseline reports it as fixed, not as a new problem.
    fixed_app = _app_with_documented_route()
    fixed_findings = gate.compute_findings(app=fixed_app)
    fixed_result = gate.evaluate(fixed_findings, baseline_with_this_route)
    assert fixed_result["exit_code"] == 0
    assert fixed_result["undocumented_routes"]["fixed"] == ["POST /api/graph/query"]


def test_baseline_set_ratchet_is_not_fooled_by_matching_count(gate):
    """A count-only ratchet would be fooled by route A becoming documented
    while route B (never seen before) becomes undocumented, net count
    unchanged. This gate is set-based: prove that exact swap still fails."""
    app = FastAPI(title="probe")
    app.add_route("/api/brand/new/route", _raw_handler, methods=["POST"])
    findings = gate.compute_findings(app=app)
    # Baseline accepts ONE undocumented route, but a DIFFERENT one than the
    # app actually has — same count (1), different member.
    baseline_same_count_different_route: dict[str, list[str]] = {
        "undocumented_routes": ["POST /api/graph/query"],
        "missing_description": [],
    }
    result = gate.evaluate(findings, baseline_same_count_different_route)
    assert result["exit_code"] == 1
    assert result["undocumented_routes"]["new"] == ["POST /api/brand/new/route"]
    assert result["undocumented_routes"]["fixed"] == ["POST /api/graph/query"]


# --- baseline file round-trip (--update-baseline) ---------------------------


def test_update_baseline_writes_current_findings_and_gate_then_passes(gate, tmp_path):
    app = _app_with_undocumented_raw_route()
    findings = gate.compute_findings(app=app)

    baseline_path = tmp_path / "openapi_coverage_baseline.json"
    original = gate.BASELINE
    try:
        gate.BASELINE = baseline_path
        gate._write_baseline(findings)
        assert baseline_path.exists()

        written = json.loads(baseline_path.read_text(encoding="utf-8"))
        assert written["undocumented_routes"] == ["POST /api/graph/query"]

        reloaded = gate._load_baseline()
        result = gate.evaluate(findings, reloaded)
        assert result["exit_code"] == 0
    finally:
        gate.BASELINE = original


# --- real allowlist entries used by the production script ------------------


def test_a2a_mount_is_walked_but_allowlisted(gate):
    """/a2a is a separate FastA2A sub-app with its own independent OpenAPI
    schema — Starlette Mounts are a boundary app.openapi() never spans. Prove
    it is walked (visible to enumerate_routes, never silently dropped) but
    excluded from the ratchet via the explicit allowlist."""
    outer = FastAPI(title="probe")
    inner = FastAPI(title="a2a-probe")
    inner.add_route("/whatever", _raw_handler, methods=["POST"])
    outer.mount("/a2a", inner)

    table = gate.enumerate_routes(outer)
    assert any(e["path"] == "/a2a/whatever" for e in table), (
        "Mount recursion must still enumerate sub-app routes — "
        "exclusion belongs to the allowlist, not the walker."
    )

    findings = gate.compute_findings(app=outer)
    assert "POST /a2a/whatever" not in findings["undocumented_routes"]


def test_metrics_endpoint_is_allowlisted(gate):
    app = FastAPI(title="probe")

    async def metrics(request):  # noqa: ARG001
        return JSONResponse({})

    app.add_api_route("/metrics", metrics, methods=["GET"], include_in_schema=False)
    findings = gate.compute_findings(app=app)
    assert "GET /metrics" not in findings["undocumented_routes"]
