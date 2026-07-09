#!/usr/bin/env python3
"""Surface-parity scan — enforce the "Two surfaces by default" edict.

Referenced by AGENTS.md ("Two surfaces by default — every feature reachable via
the gateway AND MCP"). Every capability we build must be reachable from BOTH the
API gateway (REST) and the MCP server. Because both surfaces dispatch through the
same ``_execute_tool`` single-source-of-truth, "reachable from the surface" is one
import-graph reachability question rooted at the surface entry points.

This script runs two regression checks:

1. **Tool <-> route drift** (hard gate). Imports ``kg_server`` and asserts every
   MCP tool in ``REGISTERED_TOOLS`` has a REST twin in ``ACTION_TOOL_ROUTES`` and
   vice-versa, mirroring ``tests/unit/test_gateway_mcp_parity.py`` so the
   invariant is enforceable from CI/pre-commit as a fast standalone script too.

2. **Feature reachability** (ratchet gate). Builds the static import graph (reusing
   ``check_wiring.build_graph``), seeds roots at the *surface* modules (the MCP
   server + the gateway routers), BFS-walks reachability, and flags every
   **capability module** that is reachable from NO surface root — i.e. a feature
   that exists in code but is exposed on neither the gateway nor MCP. Known and
   accepted exceptions live in ``scripts/surface_parity_baseline.txt`` (one
   repo-relative path per line); the gate fails only on *new* violations, so the
   baseline can be ratcheted down to zero over time.

Capability modules are those under the user-facing feature packages
(``CAPABILITY_PREFIXES``). Pure infrastructure (config, security, http,
observability, model plumbing, tests, the surface code itself) is excluded — it
is not a "feature" with an operator surface.

Blind spots are inherited from ``check_wiring`` (static-import view: decorator/
pkgutil registration, lazy/string imports, external callers). A module reachable
only via dynamic registration is a false positive — verify the registration runs
on a surface path, then add it to the baseline with a comment.

Usage::

    python scripts/check_surface_parity.py            # report + ratchet gate
    python scripts/check_surface_parity.py --json
    python scripts/check_surface_parity.py --update-baseline   # accept current set
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
from check_wiring import bfs_hops, build_graph  # noqa: E402
BASELINE = Path(__file__).resolve().parent / "surface_parity_baseline.txt"

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
    if not rel.startswith(CAPABILITY_PREFIXES):
        return False
    if rel.startswith(PLUGIN_PACKAGES):
        return False
    if rel in INFRA_MODULES:
        return False
    if rel.endswith(EXCLUDE_SUFFIXES):
        return False
    return all(s not in rel for s in EXCLUDE_SUBSTRINGS)


def _load_baseline() -> set[str]:
    if not BASELINE.exists():
        return set()
    out: set[str] = set()
    for line in BASELINE.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.add(line)
    return out


def _check_tool_route_drift() -> list[str]:
    """Return human-readable drift errors between MCP tools and REST routes."""
    errors: list[str] = []
    try:
        from agent_utilities.mcp import kg_server  # noqa: PLC0415

        kg_server.ensure_tools_registered()
        tools = set(kg_server.REGISTERED_TOOLS)
        mapped = set(kg_server.ACTION_TOOL_ROUTES)
    except Exception as exc:  # noqa: BLE001 — surface import failure as an error
        return [f"could not import kg_server surface: {exc!r}"]

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


def compute_unexposed() -> tuple[list[str], int, int]:
    """Return (unexposed capability modules, total capability count, roots used)."""
    graph, modules = build_graph()
    roots = {r for r in SURFACE_ROOTS if r in modules}
    dist = bfs_hops(graph, roots)
    capabilities = sorted(m for m in modules if _is_capability(m))
    unexposed = [m for m in capabilities if m not in dist]
    return unexposed, len(capabilities), len(roots)


def main() -> int:
    parser = argparse.ArgumentParser(description="Surface-parity scan.")
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Write the current unexposed set to the baseline file and exit 0.",
    )
    args = parser.parse_args()

    drift = _check_tool_route_drift()
    unexposed, total_caps, n_roots = compute_unexposed()

    if args.update_baseline:
        body = (
            "# Surface-parity baseline — capability modules not yet reachable from\n"
            "# the gateway/MCP surface. Goal: ratchet to empty. See\n"
            "# scripts/check_surface_parity.py and AGENTS.md 'Two surfaces by default'.\n"
        )
        body += "".join(f"{m}\n" for m in sorted(unexposed))
        BASELINE.write_text(body, encoding="utf-8")
        print(f"wrote baseline with {len(unexposed)} entr(ies): {BASELINE}")
        return 0

    baseline = _load_baseline()
    new_violations = sorted(set(unexposed) - baseline)
    fixed = sorted(baseline - set(unexposed))

    if args.json:
        print(
            json.dumps(
                {
                    "surface_roots": n_roots,
                    "tool_route_drift": drift,
                    "capability_modules": total_caps,
                    "unexposed": sorted(unexposed),
                    "baselined": sorted(baseline),
                    "new_violations": new_violations,
                    "fixed_since_baseline": fixed,
                },
                indent=2,
            )
        )
    else:
        print(f"surface roots: {n_roots}   capability modules: {total_caps}")
        if drift:
            print("\nTOOL<->ROUTE DRIFT (hard failure):")
            for e in drift:
                print(f"  - {e}")
        print(
            f"\nunexposed capabilities: {len(unexposed)} "
            f"(baselined {len(baseline & set(unexposed))}, "
            f"new {len(new_violations)})"
        )
        if new_violations:
            print("\nNEW unexposed capability modules (must be wired to a surface):")
            for m in new_violations:
                print(f"  + {m}")
        if fixed:
            print(f"\nfixed since baseline ({len(fixed)}) — run --update-baseline:")
            for m in fixed:
                print(f"  - {m}")

    return 1 if (drift or new_violations) else 0


if __name__ == "__main__":
    sys.exit(main())
