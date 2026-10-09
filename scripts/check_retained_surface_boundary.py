#!/usr/bin/env python3
from __future__ import annotations

"""Architecture gate for AU-BOUNDARY-R043's retained agent surface.

AU-BOUNDARY-R043 ("Retained agent surface is declared and import-bounded",
``specs/au-boundary-deconstruction/requirements.md``) names the packages Agent
Utilities keeps as its agent library -- agent execution and orchestration,
evaluation and self-improvement, runtime and developer tooling, and prompts /
pricing / policy data -- and requires that they "import the engine only
through its public client and contain no serving, storage, or graph-compute
implementation."

This gate checks the two parts of that sentence that are safe to enforce as a
hard, zero-tolerance gate today without a large migration first:

  * **Serving** -- a retained-surface file must not import
    ``agent_utilities.server`` or ``agent_utilities.gateway``. Those packages
    relocate to graph-os under AU-BOUNDARY-R015/R001; the retained surface is
    a library, not a host.
  * **Storage** -- a retained-surface file must not import a raw database
    driver (``sqlite3``, ``psycopg2``, ``psycopg``) to run its own durable
    store. Durable state moves to the epistemic graph under the sibling
    AU-BOUNDARY requirements (R020, R036, R040).

A third clause, "graph-compute implementation" / "the engine only through its
public client", is deliberately NOT enforced here yet: most retained packages
currently reach ``agent_utilities.knowledge_graph`` directly (coverage.md
documents this file by file, e.g. "20 of 40 direct modules [under graph/]
import knowledge_graph and must reach it only through the engine's public
client"), and the generated public client that replaces those imports is
still being built under the sibling au-semantic-client spec. Gating that
clause now would fail at HEAD for reasons this PR cannot fix in one slice;
it is left as follow-on scope once that client exists.

Two pre-existing, reviewed exceptions are allowlisted below, each already
documented inline at its call site. This is a fixed, named list -- never a
self-updating baseline -- and a new entry requires the same inline
justification.

Usage:
  python3 scripts/check_retained_surface_boundary.py            # check
  python3 scripts/check_retained_surface_boundary.py --list-trees  # show what's scanned

Exit 0 = no retained-surface file imports a serving or storage module outside
the allowlist, 1 = a violation was found.
"""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts._git_scan import tracked_or_walked  # noqa: E402

# Exactly the packages AU-BOUNDARY-R043 names as the retained agent surface,
# relative to ``agent_utilities/``.
RETAINED_SURFACE = (
    "agent",
    "graph",
    "capabilities",
    "patterns",
    "workflows",
    "tools",
    "core/checkpoint",
    "core/execution",
    "harness",
    "measurement",
    "rlm",
    "runtime",
    "cli",
    "claude_harness",
    "integrations",
    "analysis",
    "agent_chat",
    "prompts",
    "prompting",
    "pricing",
    "policies",
)

SERVING_MODULES = ("agent_utilities.server", "agent_utilities.gateway")
STORAGE_MODULES = ("sqlite3", "psycopg2", "psycopg")

# Reviewed, justified exceptions. Each entry names the exact file and the
# exact module it may import; nothing else is exempt. A removal of the
# justifying comment at the call site should remove the matching entry here.
ALLOWLIST: dict[str, set[str]] = {
    # Lazily reused for benchmark scoring, with a documented local fallback
    # when the gateway/engine import graph is unavailable -- never the sole
    # path (agent_utilities/harness/memorydata/adapter.py:_scoring_helpers).
    "agent_utilities/harness/memorydata/adapter.py": {"agent_utilities.server"},
    # Durable trace/state store pending its AU-BOUNDARY-R020/R040 migration to
    # the epistemic graph (coverage.md: harness/ "Includes trace, state and
    # memory modules that hold durable records").
    "agent_utilities/harness/continuous_evaluation_engine.py": {"sqlite3"},
}


def _scan_roots() -> list[Path]:
    return [ROOT / "agent_utilities" / part for part in RETAINED_SURFACE]


def _tracked_or_walked_py_files(scan_root: Path) -> list[Path]:
    return tracked_or_walked(scan_root, "*.py", root=ROOT)


def _imported_root_modules(tree: ast.AST) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _source_violations(relative: str, source: str) -> list[str]:
    try:
        tree = ast.parse(source, filename=relative)
    except SyntaxError as exc:
        return [f"{relative}: parse failed: {exc}"]

    allowed = ALLOWLIST.get(relative, set())
    failures: list[str] = []
    for module in sorted(_imported_root_modules(tree)):
        forbidden = next(
            (
                target
                for target in (*SERVING_MODULES, *STORAGE_MODULES)
                if module == target or module.startswith(f"{target}.")
            ),
            None,
        )
        if forbidden is None or forbidden in allowed:
            continue
        kind = "serving" if forbidden in SERVING_MODULES else "storage"
        failures.append(f"{relative}: forbidden {kind} import {module!r}")
    return failures


def violations() -> list[str]:
    failures: list[str] = []
    for scan_root in _scan_roots():
        if not scan_root.is_dir():
            continue
        for path in _tracked_or_walked_py_files(scan_root):
            if "__pycache__" in path.parts:
                continue
            relative = path.relative_to(ROOT).as_posix()
            try:
                source = path.read_text(encoding="utf-8")
            except OSError as exc:
                failures.append(f"{relative}: read failed: {exc}")
                continue
            failures.extend(_source_violations(relative, source))
    return failures


def main() -> int:
    if "--list-trees" in sys.argv[1:]:
        for scan_root in _scan_roots():
            print(scan_root.relative_to(ROOT).as_posix())
        return 0

    failures = violations()
    if failures:
        print("Retained-surface boundary gate failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("Retained-surface boundary gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
