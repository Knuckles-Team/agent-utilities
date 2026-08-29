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
import subprocess
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


def test_checker_flags_new_undocumented_route_vs_an_empty_head(gate):
    app = _app_with_undocumented_raw_route()
    findings = gate.compute_findings(app=app)
    empty_head: dict[str, list[str]] = {
        "undocumented_routes": [],
        "missing_description": [],
    }
    new = gate._diff_new_findings(findings, empty_head)
    assert "POST /api/graph/query" in new["undocumented_routes"]


def test_checker_reports_nothing_new_on_documented_route(gate):
    app = _app_with_documented_route()
    findings = gate.compute_findings(app=app)
    empty_head: dict[str, list[str]] = {
        "undocumented_routes": [],
        "missing_description": [],
    }
    new = gate._diff_new_findings(findings, empty_head)
    assert new["undocumented_routes"] == []
    assert new["missing_description"] == []


# --- diff-scoped comparison against HEAD (the ratchet's replacement) -------


def test_new_route_absent_from_head_is_new(gate):
    """A HEAD finding-set missing this route entirely must report it as
    NEW — mirrors the retired ratchet's "unbaselined -> fails" behavior,
    now expressed as "absent at HEAD -> fails"."""
    app = _app_with_undocumented_raw_route()
    findings = gate.compute_findings(app=app)
    head_missing_this_route: dict[str, list[str]] = {
        "undocumented_routes": ["GET /api/some/other/pre-existing/route"],
        "missing_description": [],
    }
    new = gate._diff_new_findings(findings, head_missing_this_route)
    assert "POST /api/graph/query" in new["undocumented_routes"]


def test_same_finding_present_at_head_is_not_new(gate):
    """The exact same finding, already present at HEAD, must NOT be
    reported as new — proving the comparison is set-based (not merely a
    count) and that fixing it is provable too: once the route is mounted
    correctly, it simply no longer appears in either set."""
    app = _app_with_undocumented_raw_route()
    findings = gate.compute_findings(app=app)
    head_with_this_route: dict[str, list[str]] = {
        "undocumented_routes": ["POST /api/graph/query"],
        "missing_description": [],
    }
    new = gate._diff_new_findings(findings, head_with_this_route)
    assert new["undocumented_routes"] == []

    # Now prove the OTHER direction: fixing the route (mounting it
    # correctly) against the SAME head set reports nothing new either — it
    # simply isn't in the CURRENT set any more.
    fixed_app = _app_with_documented_route()
    fixed_findings = gate.compute_findings(app=fixed_app)
    fixed_new = gate._diff_new_findings(fixed_findings, head_with_this_route)
    assert fixed_new["undocumented_routes"] == []


def test_diff_is_not_fooled_by_a_matching_count(gate):
    """A count-only comparison would be fooled by route A becoming
    documented while route B (never seen before) becomes undocumented, net
    count unchanged. This comparison is set-based: prove that exact swap
    still reports the new member."""
    app = FastAPI(title="probe")
    app.add_route("/api/brand/new/route", _raw_handler, methods=["POST"])
    findings = gate.compute_findings(app=app)
    # HEAD has ONE undocumented route, but a DIFFERENT one than the app
    # actually has now — same count (1), different member.
    head_same_count_different_route: dict[str, list[str]] = {
        "undocumented_routes": ["POST /api/graph/query"],
        "missing_description": [],
    }
    new = gate._diff_new_findings(findings, head_same_count_different_route)
    assert new["undocumented_routes"] == ["POST /api/brand/new/route"]


def test_snapshot_runner_collects_census_without_executing_main(gate, tmp_path):
    """The HEAD child imports the gate; it must never execute its diffing main."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "check_openapi_coverage.py").write_text(
        "def _load_findings():\n"
        "    return {\n"
        "        'undocumented_routes': {'POST /api/graph/query'},\n"
        "        'missing_description': {'GET /api/undescribed'},\n"
        "    }\n"
        "def main():\n"
        "    raise RuntimeError('recursive snapshot main executed')\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )

    assert gate._run_gate_snapshot(tmp_path) == {
        "undocumented_routes": ["POST /api/graph/query"],
        "missing_description": ["GET /api/undescribed"],
    }


# --- retired flag ------------------------------------------------------


def test_update_baseline_flag_is_retired():
    """The retired flag must REFUSE, not silently do nothing — the same
    convention the liveness/complexity/swallowed-error/wire-first gates
    adopted when their baselines were removed."""
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[3] / "scripts" / "check_openapi_coverage.py"),
            "--update-baseline",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "RETIRED" in result.stderr


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
