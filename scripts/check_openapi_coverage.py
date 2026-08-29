#!/usr/bin/env python3
"""OpenAPI coverage gate — every mounted HTTP route must appear in the
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

This script enumerates every concrete route actually mounted on the app
(recursing into ``Mount``s) and diffs that table against ``app.openapi()``'s
path table, producing two finding sets: routes missing from the spec
entirely, and *documented* operations missing a ``summary`` or a non-empty
``description`` (an undescribed operation is a finding, not a pass).

NO BASELINE HERE ANY MORE (retired ratchet — see check_swallowed_errors.py's
module docstring for the fully-worked-out rationale this gate now follows)
--------------------------------------------------------------------------
This gate used to freeze both finding sets into
``scripts/openapi_coverage_baseline.json`` and fail only on a member absent
from that file. That is a ratchet, which this project does not allow: a
frozen count hides whether the real backlog is shrinking, growing, or just
sitting there — measured for the sibling ``check_swallowed_errors.py`` gate,
the frozen count silently diverged from reality while the gate kept saying
"no new sites"; measured here (2026-08-28, see the lane report this commit
belongs to), the two counts happened to still agree exactly (235
undocumented, 32 missing-description, byte-for-byte the same as the frozen
file) — but agreeing today is not a property of a ratchet, it is a
coincidence a ratchet cannot tell you is true without a fresh count, which
is the whole complaint.

What replaces it, matching the same shape now used repo-wide (liveness,
complexity, swallowed-errors, wire-first):

* an **unconditional census** — the real totals, printed on every run, pass
  or fail. Nothing is written to disk, so no number can go stale.
* **diff-scoped enforcement**, keyed by content (``METHOD /path`` — a route
  path has no "enclosing symbol" for an extract-method refactor to perturb;
  unlike Wire-First's per-symbol key this one was never at risk from that
  class of bug in the first place). "Diff-scoped" here means: compare the
  route/operation set THIS commit produces against the set the SAME gate
  script, unmodified, produces when run against the repository AS OF HEAD —
  i.e. this script imports its own ``HEAD`` revision inside a ``git archive``
  snapshot and invokes only its census function (see ``_run_gate_snapshot``)
  rather than
  hand-maintaining a second comparison implementation that could drift from
  the first. A finding present now but absent at HEAD is new debt and fails
  the commit; the rest is pre-existing backlog, printed, not hidden.
* the expensive half of that (importing and constructing the whole
  production FastAPI app a SECOND time) only runs when a file this gate's
  findings can actually depend on changed since HEAD
  (``_relevant_route_files_changed``) — the same scope this gate's own
  pre-commit ``files:`` pattern already uses, so on every commit that
  wouldn't even trigger this hook, the comparison is a free no-op.
* retired flags (``--update-baseline``) exit **2** with an explanation.

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
  not a canonical surface worth measuring.
* The ``/a2a`` mount is a **separate, independent** FastA2A protocol
  application with its own OpenAPI schema at its own ``/a2a/openapi.json``.
  Starlette/FastAPI's ``app.openapi()`` never spans a ``Mount`` boundary —
  a sub-app's routes are structurally never going to appear in the parent
  spec, by framework design, not by the bug this gate exists to catch. It
  is walked (so nothing is silently invisible to a human reading this
  script) but excluded via the explicit allowlist below; audit it
  separately at its own ``/openapi.json``.

If ``agent_utilities.server`` cannot be imported at all (e.g. a minimal
profile with the ``[server]`` extra absent), the gate says so loudly and
exits 0 — NOT ENFORCED, never a silent pass disguised as a real one (see
``check_liveness.py``'s docstring for why that distinction matters here).

Usage::

    python scripts/check_openapi_coverage.py                  # report + diff-scoped gate
    python scripts/check_openapi_coverage.py --json

Exit 0 = this change added no NEW undocumented route / undescribed
operation, 1 = it did (or the HEAD comparison could not be safely made while
inside a real git work tree — a degraded read must never read as a pass),
2 = a retired flag was passed.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
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

#: Explicit, commented allowlist (never a silent skip). Every entry names
#: exactly why it can never gain a documented OpenAPI operation, or why
#: documenting it here would be structurally wrong. Exact-path entries match
#: one path; prefix entries (ending "*") match any path starting with that
#: prefix.
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
    # for visibility, never silently, but excluded from the finding sets.
    ("/a2a/*", "separate FastA2A sub-app with its own independent OpenAPI schema"),
)

_HTTP_METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE"})

#: Files this gate's findings can actually depend on. Mirrors the
#: pre-commit hook's own `files:` scope — see `_relevant_route_files_changed`.
_RELEVANT_PREFIXES = (
    "agent_utilities/server/",
    "agent_utilities/gateway/",
)
_RELEVANT_EXACT = ("agent_utilities/mcp/kg_server.py",)

_CATEGORIES = ("undocumented_routes", "missing_description")


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
    # is allowlisted out of the findings (see ALLOWLIST above); it only needs
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


def _route_kind(route: Any, api_route_cls: type, route_cls: type) -> str:
    """``"api_route"`` (schema-capable) / ``"starlette_route"`` (invisible to
    ``app.openapi()`` — the bug class this gate exists to catch) / the raw
    type name for anything else."""
    if isinstance(route, api_route_cls):
        return "api_route"
    if isinstance(route, route_cls):
        return "starlette_route"
    return type(route).__name__


def _mount_entries(route: Any, prefix: str) -> list[dict[str, Any]]:
    """Entries contributed by one ``Mount`` — recurse into a sub-app's own
    routes, or record a single synthetic ``opaque_mount`` entry when the
    mounted target exposes no ``.routes`` (e.g. ``StaticFiles``) so it is
    never silently dropped."""
    full_prefix = prefix + route.path
    sub_app = route.app
    if hasattr(sub_app, "routes"):
        return enumerate_routes(sub_app, prefix=full_prefix)
    return [
        {
            "method": "*",
            "path": full_prefix,
            "kind": "opaque_mount",
            "mount_of": type(sub_app).__name__,
        }
    ]


def _concrete_route_entries(
    route: Any, prefix: str, api_route_cls: type, route_cls: type
) -> list[dict[str, Any]]:
    """One entry per HTTP method on a non-``Mount`` route. ``HEAD`` is
    auto-added by Starlette for every ``GET`` route and never gets its own
    OpenAPI operation — comparing it would manufacture a permanent,
    unfixable "finding" for every single GET, so it is skipped here."""
    methods = sorted(getattr(route, "methods", None) or [])
    path = prefix + (getattr(route, "path", "") or "")
    if not methods or not path:
        return []
    kind = _route_kind(route, api_route_cls, route_cls)
    return [
        {"method": method, "path": path, "kind": kind, "mount_of": None}
        for method in methods
        if method != "HEAD"
    ]


def enumerate_routes(app: Any, prefix: str = "") -> list[dict[str, Any]]:
    """Walk ``app.routes``, recursing into ``Mount``s, and return every
    concrete (method, path) endpoint as a dict with enough detail to explain
    a finding: ``{"method", "path", "kind", "mount_of"}``.
    """
    from fastapi.routing import APIRoute
    from starlette.routing import Mount, Route

    entries: list[dict[str, Any]] = []
    for route in getattr(app, "routes", []):
        if isinstance(route, Mount):
            entries.extend(_mount_entries(route, prefix))
            continue
        entries.extend(_concrete_route_entries(route, prefix, APIRoute, Route))
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


def _classify_route_table(
    route_table: list[dict[str, Any]], documented: dict[tuple[str, str], Any]
) -> tuple[list[dict[str, Any]], set[str], set[tuple[str, str]]]:
    """Split the mounted route table into (excluded, undocumented_labels,
    seen_pairs) against the documented (method, path) map."""
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
    return excluded, undocumented, seen_table_pairs


def _find_missing_description(
    documented: dict[tuple[str, str], dict[str, Any]],
) -> set[str]:
    missing: set[str] = set()
    for (method, path), op in documented.items():
        if _allowlisted(path) is not None:
            continue
        summary = str(op.get("summary") or "").strip()
        description = str(op.get("description") or "").strip()
        if not summary or not description:
            missing.add(f"{method} {path}")
    return missing


def compute_findings(app: Any | None = None) -> dict[str, Any]:
    """Diff *app*'s route table against its own generated spec and return the
    two finding sets plus the raw counts used in the report.

    ``app`` defaults to the real production app (:func:`_build_target_app`);
    passing an app explicitly is how tests exercise this against a small,
    deliberately-constructed app instead of paying for the full build.
    """
    if app is None:
        app = _build_target_app()
    route_table = enumerate_routes(app)
    spec = app.openapi()
    documented = _documented_pairs(spec)
    excluded, undocumented, seen_table_pairs = _classify_route_table(
        route_table, documented
    )
    missing_description = _find_missing_description(documented)

    return {
        "undocumented_routes": undocumented,
        "missing_description": missing_description,
        "total_routes": len(seen_table_pairs),
        "total_documented_operations": len(documented),
        "total_excluded": len(excluded),
        "excluded": excluded,
    }


# ── Diff-scoped enforcement (the ratchet's replacement) ───────────────────


def _git(*args: str, cwd: str | None = None) -> subprocess.CompletedProcess:
    """Run git from the repo toplevel with repo-relative paths — see
    ``check_swallowed_errors.py``'s identically-named helper for why every
    invocation runs from the resolved toplevel and never ``git -C <subdir>``
    (GIT_DIR/GIT_INDEX_FILE ambient-env hazard)."""
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
    )


def _repo_root() -> str | None:
    r = _git("rev-parse", "--show-toplevel")
    out = r.stdout.strip()
    return out if r.returncode == 0 and out else None


def _relevant_route_files_changed(root: str) -> bool:
    """True iff a file this gate's findings can actually depend on differs
    from HEAD (staged or working tree). Narrows the expensive half of the
    comparison (importing and constructing the whole FastAPI app a second
    time) to commits that could plausibly move the finding set at all —
    the same scope this gate's own pre-commit ``files:`` pattern already
    uses, computed independently here so the standalone script is correct
    even when invoked outside that hook."""
    changed: set[str] = set()
    for args in (
        ("diff", "--cached", "--name-only", "--diff-filter=ACMR", "HEAD"),
        ("diff", "--name-only", "HEAD"),
    ):
        r = _git(*args, cwd=root)
        if r.returncode == 0:
            changed.update(r.stdout.splitlines())
    return any(
        p.startswith(_RELEVANT_PREFIXES) or p in _RELEVANT_EXACT for p in changed
    )


def _archive_head_bytes(root: str) -> bytes | None:
    proc = subprocess.run(
        [
            "git",
            "archive",
            "HEAD",
            "--",
            "agent_utilities",
            "scripts",
            "pyproject.toml",
        ],
        cwd=root,
        capture_output=True,
        check=False,
    )
    return proc.stdout if proc.returncode == 0 and proc.stdout else None


def _materialize_head_snapshot(root: str, dest: Path) -> bool:
    """Extract ``agent_utilities/`` + ``scripts/`` + ``pyproject.toml`` AS OF
    HEAD into ``dest``. Returns False when the archive could not be produced
    at all (e.g. a shallow/synthetic repo with no HEAD) so the caller can
    treat the comparison as unavailable rather than silently diffing against
    an empty tree."""
    data = _archive_head_bytes(root)
    if data is None:
        return False
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        tf.extractall(dest, filter="data")
    return True


def _snapshot_subprocess_environment() -> dict[str, str]:
    """Return an environment detached from the caller's Git worktree."""
    # Hooks can export worktree-specific Git variables (especially
    # GIT_INDEX_FILE). The materialized snapshot is deliberately not a Git
    # worktree and must never resolve back into the caller's repository or
    # index through ambient process state.
    return {
        name: value for name, value in os.environ.items() if not name.startswith("GIT_")
    }


def _run_gate_snapshot(snapshot_root: Path) -> dict[str, list[str]] | None:
    """Run this gate's HEAD census once in an isolated subprocess.

    The runner imports the archived script as a module and calls only
    ``_load_findings``. It must not execute that script's ``main``: ``main``
    performs another HEAD comparison, recursively spawning snapshots until
    the host is exhausted. Importing in a subprocess still gives the HEAD
    tree its own module cache and bootstrap without contaminating this
    process. Returns None when the snapshot is unavailable or cannot emit a
    valid census.
    """
    script = snapshot_root / "scripts" / "check_openapi_coverage.py"
    if not script.exists():
        return None
    census_runner = """
import json
import runpy
import sys

gate = runpy.run_path(sys.argv[1], run_name="_openapi_snapshot_gate")
findings = gate["_load_findings"]()
if findings is None:
    raise SystemExit(1)
print(json.dumps({
    "undocumented_routes": sorted(findings["undocumented_routes"]),
    "missing_description": sorted(findings["missing_description"]),
}))
"""
    proc = subprocess.run(
        [sys.executable, "-c", census_runner, str(script)],
        cwd=snapshot_root,
        capture_output=True,
        text=True,
        check=False,
        env=_snapshot_subprocess_environment(),
    )
    try:
        data = json.loads(proc.stdout)
    except (json.JSONDecodeError, ValueError):
        return None
    return {c: list(data.get(c, [])) for c in _CATEGORIES}


def _diff_new_findings(
    findings: dict[str, Any], head: dict[str, list[str]]
) -> dict[str, list[str]]:
    """Pure set-diff of the current finding sets against a HEAD finding map
    (``{category: [...]}``, e.g. one gate run's ``--json`` output) — kept
    separate from the git/subprocess plumbing in
    :func:`_new_findings_vs_head` so it is directly unit-testable without a
    real git repo (see ``tests/unit/gateway/test_openapi_coverage.py``)."""
    return {c: sorted(findings[c] - set(head.get(c, []))) for c in _CATEGORIES}


def _new_findings_vs_head(
    root: str, findings: dict[str, Any]
) -> dict[str, list[str]] | None:
    """``{category: [new...]}`` vs the same gate run against HEAD's own
    source, or None when the comparison could not be safely made (the HEAD
    snapshot could not be built or run) — the caller must treat that as a
    degraded read, never a silent pass (AGENTS.md "Fail closed"). Skips the
    expensive snapshot entirely (returns all-empty) when nothing this gate's
    findings could depend on has changed since HEAD."""
    if not _relevant_route_files_changed(root):
        return {c: [] for c in _CATEGORIES}
    with tempfile.TemporaryDirectory(prefix="au-openapi-head-") as tmp:
        dest = Path(tmp)
        if not _materialize_head_snapshot(root, dest):
            return None
        head = _run_gate_snapshot(dest)
    if head is None:
        return None
    return _diff_new_findings(findings, head)


def _print_census(findings: dict[str, Any]) -> None:
    """Print the real numbers. Always. This never fails the run."""
    print(
        f"routes on app: {findings['total_routes']}   "
        f"documented operations: {findings['total_documented_operations']}   "
        f"excluded (allowlisted/opaque mounts): {findings['total_excluded']}"
    )
    print(
        "Undocumented routes (mounted, absent from OpenAPI spec): "
        f"{len(findings['undocumented_routes'])} total."
    )
    print(
        "Documented operations missing summary/description: "
        f"{len(findings['missing_description'])} total."
    )


def _print_new(label: str, new: list[str]) -> None:
    if not new:
        return
    print(f"\n{label}: {len(new)} NEW finding(s) since HEAD:")
    for item in sorted(new):
        print(f"  + {item}")


def _emit_json(findings: dict[str, Any], new: dict[str, list[str]] | None) -> int:
    print(
        json.dumps(
            {
                "total_routes": findings["total_routes"],
                "total_documented_operations": findings["total_documented_operations"],
                "total_excluded": findings["total_excluded"],
                "undocumented_routes": sorted(findings["undocumented_routes"]),
                "missing_description": sorted(findings["missing_description"]),
                "new": new,
            },
            indent=2,
        )
    )
    if new is None:
        return 1
    return 1 if any(new[c] for c in _CATEGORIES) else 0


def _emit_report(
    findings: dict[str, Any], new: dict[str, list[str]] | None, root: str | None
) -> int:
    print("=" * 72)
    print("OpenAPI coverage gate")
    print("=" * 72)
    _print_census(findings)

    if new is None:
        print(
            "\nFAIL — could not build the HEAD-snapshot comparison app; a "
            "degraded read must never be treated as a pass. Re-run, or "
            "investigate `git archive HEAD` / the snapshot subprocess."
        )
        return 1

    if root is None:
        print("\n(not inside a work tree — diff-scoped enforcement skipped)")

    _print_new(
        "Undocumented routes (mounted, absent from OpenAPI spec)",
        new["undocumented_routes"],
    )
    _print_new(
        "Documented operations missing summary/description",
        new["missing_description"],
    )

    print()
    if any(new[c] for c in _CATEGORIES):
        print(
            "FAIL — OpenAPI coverage REGRESSED since HEAD. Give the new "
            "route(s) a real FastAPI-schema-capable mount (APIRouter."
            "add_api_route, not Starlette add_route) and/or a summary + "
            "description, or — if genuinely accepted — document it with "
            "`# noqa`-style justification at the call site (see AGENTS.md)."
        )
        return 1
    print("OK — no NEW OpenAPI coverage regression since HEAD.")
    return 0


def _load_findings() -> dict[str, Any] | None:
    """``compute_findings()``, or None if the production app could not be
    built at all (missing ``[server]`` extra) — the caller reports NOT
    ENFORCED rather than treating that as either a pass or a fail."""
    try:
        return compute_findings()
    except ImportError as exc:
        print(
            "openapi-coverage gate NOT ENFORCED (exit 0, nothing was checked): "
            f"could not build the production app ({exc!r}). Install the "
            "`server` extra this repo declares to enable this gate."
        )
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenAPI coverage gate.")
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    parser.add_argument(
        "--update-baseline", action="store_true", help=argparse.SUPPRESS
    )
    args = parser.parse_args()

    if args.update_baseline:
        print(
            "--update-baseline is RETIRED. This gate has no baseline: it "
            "prints the real census every run and enforces diff-scoped "
            "against HEAD, so there is nothing to freeze. See the module "
            "docstring.",
            file=sys.stderr,
        )
        return 2

    findings = _load_findings()
    if findings is None:
        return 0

    root = _repo_root()
    new = (
        {c: [] for c in _CATEGORIES}
        if root is None
        else _new_findings_vs_head(root, findings)
    )

    if args.json:
        return _emit_json(findings, new)
    return _emit_report(findings, new, root)


if __name__ == "__main__":
    sys.exit(main())
