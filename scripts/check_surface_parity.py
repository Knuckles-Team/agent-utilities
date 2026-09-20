#!/usr/bin/env python3
"""Surface-parity scan — enforce the "Two surfaces by default" edict.

Referenced by AGENTS.md ("Two surfaces by default — every feature reachable via
the gateway AND MCP"). Every capability we build must be reachable from BOTH the
API gateway (REST) and the MCP server. Because both surfaces dispatch through the
same ``_execute_tool`` single-source-of-truth, "reachable from the surface" is one
import-graph reachability question rooted at the surface entry points.

This script runs two regression checks:

1. **Tool <-> route drift** (hard gate, absolute, unconditional). Imports
   ``kg_server`` and compares the runtime execution table with the immutable,
   profile-aware ToolSpec universe, then asserts every enabled tool has a REST
   twin in ``ACTION_TOOL_ROUTES`` and vice-versa. The repo is at zero for this
   today and stays there — no exceptions, no baseline, ever.

2. **Feature reachability** (census + diff-scoped gate — see "Why there is no
   baseline here" below). Builds the static import graph (reusing
   ``check_wiring.build_graph``), seeds roots at the *surface* modules (the MCP
   server + the gateway routers), BFS-walks reachability, and flags every
   **capability module** that is reachable from NO surface root — i.e. a
   feature that exists in code but is exposed on neither the gateway nor MCP.

Capability modules are those under the user-facing feature packages
(``CAPABILITY_PREFIXES``). Pure infrastructure (config, security, http,
observability, model plumbing, tests, the surface code itself) is excluded — it
is not a "feature" with an operator surface.

Why there is no baseline here any more (D-WD5-RAT-03)
-------------------------------------------------------
This gate used to freeze its unexposed set into
``scripts/surface_parity_baseline.txt`` and fail only on an entry absent from
that file (a ratchet, keyed by module path — which project convention forbids
outright: see ``scripts/check_swallowed_errors.py``'s module docstring for the
first, and worst, example of why a frozen debt file is worse than no gate at
all). Measured before retiring it: the baseline held exactly 45 entries and the
live unexposed set was *also* exactly 45 — bit-for-bit in sync, no drift, no
silently-fixed entries hiding behind it. That is unusual (every other ratchet
retired in this program was hiding either false positives or already-fixed
debt) but it does not make freezing the set correct: a ratchet that happens to
be accurate today still hides tomorrow's drift the moment it stops being
watched, and it is not this project's call to make an exception for one that
currently isn't lying.

One entry in that 45, ``.../writeback/asset_mirror.py``, was never real debt at
all: its own baseline comment said so — the capability it fronts IS exposed via
``graph_writeback(asset_mirror=true)``, this module is just the co-located CLI
transport, unreachable by static import only because of the documented
dynamic-registration blind spot. It is promoted into ``INFRA_MODULES`` below
(a real fix, not a tracked exception) rather than left as a census entry
someone has to keep re-explaining.

What replaces the ratchet:

* an **unconditional census**: every unexposed capability module is printed on
  *every* run, pass or fail — never written to disk, so it cannot go stale;
* **diff-scoped enforcement**: a capability module that is *newly added* by
  this change (absent from ``HEAD``) and unexposed fails the commit. A
  module that already existed and is already unexposed is backlog — visible
  above, not gated — burning it down is deliberate cleanup, not a side effect
  of an unrelated commit;
* the tool<->route drift check, which was already unconditional and already at
  zero, is untouched.

Known, accepted blind spot (documented, not silently missed): an *existing*
capability module whose last surface-import edge this diff removes becomes
newly unexposed without being "added", so it is not caught at commit time —
only on the next run's census, where the unexposed count visibly grows.
Catching that would require rebuilding the whole reachability graph from the
``HEAD`` tree on every run (a second full package AST walk) to protect against
a failure mode a reviewer already sees directly in the diff (an import line
disappearing). Traded off deliberately for the same reason the swallowed-error
gate only diffs each file's own blob rather than the whole repo's history.

Blind spots inherited from ``check_wiring`` (static-import view: decorator/
pkgutil registration, lazy/string imports, external callers) are unchanged. A
module reachable only via dynamic registration is a false positive — verify the
registration runs on a surface path, then add it to ``PLUGIN_PACKAGES`` (or, if
it is not really a standalone feature, ``INFRA_MODULES``) with a comment,
exactly as ``asset_mirror.py`` was.

Usage::

    python scripts/check_surface_parity.py            # census + diff-scoped gate
    python scripts/check_surface_parity.py --json

Exit 0 = no drift and this change added no unwired capability module, 1 = it
did (or drift exists), 2 = a retired flag was passed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent

# Reuse the import-graph machinery from the Wire-First tool (same directory).
sys.path.insert(0, str(Path(__file__).resolve().parent))
# Ensure the repo root itself resolves FIRST so `agent_utilities` imports below
# (kg_server, gateway routers) always bind to THIS checkout's source, never to a
# stale/partial `agent_utilities` that happens to be on the ambient interpreter's
# site-packages (e.g. an old editable install) — see check_concepts.py, which
# already does this. Without it, a shared/dev box can resolve a partial merged
# namespace package and fail with a spurious "cannot import name ... (unknown
# location)" ImportError that has nothing to do with a real surface-parity drift.
sys.path.insert(0, str(ROOT))
from _gate_interpreter import require_project_interpreter  # noqa: E402

# Re-exec under this repo's declared interpreter BEFORE importing the MCP
# server surface: under an out-of-contract python this gate does not fail,
# it stops functioning. See scripts/_gate_interpreter.py.
require_project_interpreter(ROOT)

from check_wiring import bfs_hops, build_graph  # noqa: E402

# Modules that constitute the operator surface. Reachability is measured FROM
# these — anything they can import is considered exposed on gateway + MCP (both
# dispatch through the shared _execute_tool core that lives in kg_server).
SURFACE_ROOTS = (
    "agent_utilities/mcp/kg_server.py",
    "agent_utilities/gateway/graph_api.py",
    "agent_utilities/gateway/ontology_api.py",
    "agent_utilities/gateway/fleet.py",
    "agent_utilities/gateway/api.py",
    "agent_utilities/gateway/usage_api.py",
    "agent_utilities/gateway/artifacts_api.py",
    "agent_utilities/server/app.py",
)

# Packages whose modules are user-facing FEATURES — each should be reachable from
# the surface. A capability that lives here but is unreachable from a surface root
# is the violation this gate exists to catch.
CAPABILITY_PREFIXES = (
    "agent_utilities/knowledge_graph/extraction/",
    "agent_utilities/knowledge_graph/enrichment/",
    "agent_utilities/knowledge_graph/ingestion/",
    "agent_utilities/knowledge_graph/distillation/",
    "agent_utilities/knowledge_graph/assimilation/",
    "agent_utilities/knowledge_graph/ontology/",
    "agent_utilities/knowledge_graph/kb/",
    "agent_utilities/knowledge_graph/retrieval/",
    "agent_utilities/knowledge_graph/research/",
    "agent_utilities/knowledge_graph/search_synthesis/",
    "agent_utilities/knowledge_graph/orchestration/",
    "agent_utilities/knowledge_graph/maintenance/",
    "agent_utilities/knowledge_graph/live_artifacts/",
    "agent_utilities/knowledge_graph/streams/",
    "agent_utilities/protocols/source_connectors/",
    "agent_utilities/harness/",
    "agent_utilities/rlm/",
    "agent_utilities/workflows/",
    "agent_utilities/domains/",
)

# Cross-package operator capabilities that live beside orchestration plumbing.
# Listing these explicitly avoids the old blind spot without pretending every
# protocol, dataclass, allocator, or controller in ``orchestration/`` needs a
# standalone tool.
EXPLICIT_CAPABILITY_MODULES = frozenset(
    {"agent_utilities/orchestration/agent_digital_twin.py"}
)

# Within a capability package, these are not standalone features (helpers, types,
# fixtures, package inits) — excluded from the reachability requirement.
EXCLUDE_SUFFIXES = (
    "/__init__.py",
    "/conftest.py",
    "/__main__.py",
    "/models.py",  # pydantic data models — not an operator-invokable feature
    "_models.py",
    "/errors.py",  # exception types
    "_adapter.py",  # transport adapters (kafka/nats) — infra, not a feature
)
EXCLUDE_SUBSTRINGS = (
    "/tests/",
    "/test_",
    "/_fixtures",
)

# Infrastructure modules that are plumbing, not operator-invokable features:
# internal controllers, diagnostics, training/eval primitives, ontology data,
# the durable-queue orchestrator. Excluded from the capability scan by exact path.
INFRA_MODULES = frozenset(
    {
        "agent_utilities/domains/finance/quant_ontology.py",
        "agent_utilities/knowledge_graph/ingestion/batch_orchestrator.py",
        "agent_utilities/knowledge_graph/orchestration/voi_budget_controller.py",
        "agent_utilities/knowledge_graph/retrieval/embedding_diagnostics.py",
        "agent_utilities/harness/evaluators.py",
        "agent_utilities/harness/reasoning_effort.py",
        "agent_utilities/harness/reliability_corpus.py",
        "agent_utilities/harness/replay_buffer.py",
        "agent_utilities/harness/scaling_laws.py",
        "agent_utilities/harness/variant_pool.py",
        # Deterministic scoring helpers consumed by native optimizer jobs; these
        # contain no independently invokable optimizer or operator action.
        "agent_utilities/harness/policy_optimization.py",
        "agent_utilities/knowledge_graph/extraction/extraction_optimizer.py",
        # EH-274 follow-up: a pure graph-traversal algorithm (pass-through-
        # collapse BFS) shared by two extractors/ modules (aris.py,
        # camunda.py, both already covered by PLUGIN_PACKAGES below). It has
        # no operator-invokable action of its own -- just a helper function
        # called from within those extractors.
        "agent_utilities/knowledge_graph/enrichment/graph_collapse.py",
        # D-WD5-RAT-03: not a separate capability — the run_asset_mirror
        # capability itself IS exposed on both surfaces via
        # graph_writeback(asset_mirror=true) + its REST twin. This module is
        # only the co-located `python -m ...writeback.asset_mirror` CLI
        # transport for one multi-SoR mirror pass, unreachable by static
        # import for the same reason as PLUGIN_PACKAGES (a CLI entry point,
        # not an import edge). Formerly carried in the now-retired
        # surface_parity_baseline.txt as if it were debt; it never was.
        "agent_utilities/knowledge_graph/enrichment/writeback/asset_mirror.py",
    }
)

# Plugin packages whose members self-register via a decorator + pkgutil/import
# discovery that runs on a live surface path (e.g. ``@register_source`` +
# ``discover()``). Static import cannot see these edges, so members are exposed
# as long as the package's discovery loader is itself reachable — which the
# surface roots guarantee. Treated as exposed to avoid systematic false
# positives (the static-import blind spot documented in check_wiring).
PLUGIN_PACKAGES = (
    "agent_utilities/protocols/source_connectors/connectors/",
    "agent_utilities/knowledge_graph/enrichment/extractors/",
)


def _is_capability(rel: str) -> bool:
    if rel not in EXPLICIT_CAPABILITY_MODULES and not rel.startswith(
        CAPABILITY_PREFIXES
    ):
        return False
    if rel.startswith(PLUGIN_PACKAGES):
        return False
    if rel in INFRA_MODULES:
        return False
    if rel.endswith(EXCLUDE_SUFFIXES):
        return False
    return all(s not in rel for s in EXCLUDE_SUBSTRINGS)


def _toolspec_drift(tools: set[str], *, tool_specs_module: ModuleType) -> list[str]:
    """Drift between the runtime tool set and the immutable ToolSpec universe."""
    TOOL_SPECS_BY_NAME = tool_specs_module.TOOL_SPECS_BY_NAME
    INTENT_VERBS = tool_specs_module.INTENT_VERBS
    errors: list[str] = []
    unknown = tools - set(TOOL_SPECS_BY_NAME)
    if unknown:
        errors.append(
            "Runtime tools absent from ToolSpec: " + ", ".join(sorted(unknown))
        )
    features = frozenset(
        spec.feature
        for name, spec in TOOL_SPECS_BY_NAME.items()
        if name in tools and spec.feature is not None
    )
    intent_present = set(INTENT_VERBS) & tools
    if intent_present and intent_present != set(INTENT_VERBS):
        errors.append(
            "Runtime intent overlay is partial: " + ", ".join(sorted(intent_present))
        )
    expected = set(
        tool_specs_module.canonical_tool_names(
            features=features,
            include_intent=bool(intent_present),
        )
    )
    missing_runtime = expected - tools
    if missing_runtime:
        errors.append(
            "Canonical ToolSpecs absent from runtime: "
            + ", ".join(sorted(missing_runtime))
        )
    return errors


def _route_drift(tools: set[str], mapped: set[str]) -> list[str]:
    """Drift between the runtime tool set and the REST route table."""
    errors: list[str] = []
    missing = tools - mapped
    if missing:
        errors.append(
            "MCP tools with no REST twin (ACTION_TOOL_ROUTES): "
            + ", ".join(sorted(missing))
        )
    phantom = mapped - tools
    if phantom:
        errors.append(
            "REST routes for non-existent MCP tools: " + ", ".join(sorted(phantom))
        )
    return errors


def _check_tool_route_drift() -> list[str]:
    """Return human-readable drift errors between MCP tools and REST routes."""
    try:
        from agent_utilities.mcp import kg_server  # noqa: PLC0415
        from agent_utilities.mcp import tool_specs as tool_specs_module  # noqa: PLC0415

        kg_server.ensure_tools_registered()
        tools = set(kg_server.REGISTERED_TOOLS)
        mapped = set(kg_server.ACTION_TOOL_ROUTES)
    except Exception as exc:  # noqa: BLE001 — surface import failure as an error
        return [f"could not import kg_server surface: {exc!r}"]

    return _toolspec_drift(tools, tool_specs_module=tool_specs_module) + _route_drift(
        tools, mapped
    )


def compute_unexposed() -> tuple[list[str], int, int]:
    """Return (unexposed capability modules, total capability count, roots used)."""
    graph, modules = build_graph()
    roots = {r for r in SURFACE_ROOTS if r in modules}
    dist = bfs_hops(graph, roots)
    capabilities = sorted(m for m in modules if _is_capability(m))
    unexposed = [m for m in capabilities if m not in dist]
    return unexposed, len(capabilities), len(roots)


# ── Diff-scoped enforcement (the ratchet's replacement) ───────────────────


def _git(*args: str, cwd: str | None = None) -> subprocess.CompletedProcess:
    """Run git from the repo toplevel with repo-relative paths.

    Mirrors ``check_swallowed_errors.py``'s ``_git``: git exports
    ``GIT_DIR``/``GIT_INDEX_FILE``/``GIT_WORK_TREE`` into every hook
    subprocess, so every invocation here runs from the resolved toplevel via
    ``cwd=``, never ``git -C <subdir>`` (BUG-180's failure shape).
    """
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
    )


def _repo_root() -> str | None:
    r = _git("rev-parse", "--show-toplevel")
    out = r.stdout.strip()
    return out if r.returncode == 0 and out else None


def _added_py_files(root: str) -> list[str]:
    """Repo-relative ``agent_utilities/**.py`` paths newly ADDED relative to
    ``HEAD`` — the union of the staged and working-tree diffs (mirrors
    ``check_swallowed_errors._changed_py_files``: during a commit pre-commit
    has stashed unstaged edits so the two agree; a manual ``--all-files`` run
    needs the working-tree one)."""
    paths: set[str] = set()
    for args in (
        ("diff", "--cached", "--name-only", "--diff-filter=A", "HEAD"),
        ("diff", "--name-only", "--diff-filter=A", "HEAD"),
    ):
        r = _git(*args, "--", "agent_utilities", cwd=root)
        if r.returncode == 0:
            paths.update(line for line in r.stdout.splitlines() if line.endswith(".py"))
    return sorted(paths)


def new_capability_violations(unexposed: set[str]) -> list[str]:
    """Capability modules this change ADDS that are unexposed right now.

    A module already present at ``HEAD`` and already unexposed is backlog —
    reported by :func:`compute_unexposed`'s unconditional census, never
    gated here. See the module docstring's "known, accepted blind spot" for
    what this deliberately does not catch.
    """
    root = _repo_root()
    if root is None:
        return []
    return sorted(m for m in _added_py_files(root) if m in unexposed)


def _print_json_report(
    *,
    n_roots: int,
    drift: list[str],
    total_caps: int,
    unexposed: list[str],
    new_violations: list[str],
) -> None:
    print(
        json.dumps(
            {
                "surface_roots": n_roots,
                "tool_route_drift": drift,
                "capability_modules": total_caps,
                "unexposed": sorted(unexposed),
                "new_violations": new_violations,
            },
            indent=2,
        )
    )


def _print_human_report(
    *,
    n_roots: int,
    drift: list[str],
    total_caps: int,
    unexposed: list[str],
    new_violations: list[str],
) -> None:
    print(f"surface roots: {n_roots}   capability modules: {total_caps}")
    if drift:
        print("\nTOOL<->ROUTE DRIFT (hard failure):")
        for e in drift:
            print(f"  - {e}")
    print(
        f"\nunexposed capabilities (unconditional census — real backlog, "
        f"not gated): {len(unexposed)}"
    )
    for m in sorted(unexposed):
        print(f"  - {m}")
    if not new_violations:
        return
    print(
        f"\nNEW unwired capability module(s) added by this change "
        f"({len(new_violations)}):"
    )
    for m in new_violations:
        print(f"  + {m}")
    print(
        "\nEach must be reachable from a surface root (gateway/MCP) before "
        "merge — wire it in, or if it is deliberately not user-invokable, "
        "add it to INFRA_MODULES/PLUGIN_PACKAGES/EXCLUDE_* with a reason, "
        "matching asset_mirror.py."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Surface-parity scan.")
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args.update_baseline:
        print(
            "--update-baseline is RETIRED. This gate has no baseline: the "
            "unexposed capability set is printed as an unconditional census "
            "every run, and only a capability module ADDED by this change "
            "fails the diff-scoped check. See the module docstring.",
            file=sys.stderr,
        )
        return 2

    drift = _check_tool_route_drift()
    unexposed, total_caps, n_roots = compute_unexposed()
    new_violations = new_capability_violations(set(unexposed))
    report = _print_json_report if args.json else _print_human_report
    report(
        n_roots=n_roots,
        drift=drift,
        total_caps=total_caps,
        unexposed=unexposed,
        new_violations=new_violations,
    )

    return 1 if (drift or new_violations) else 0


if __name__ == "__main__":
    sys.exit(main())
