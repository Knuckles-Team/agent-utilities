"""Retirement proofs for the AU-boundary R028 family rows (1, 3, 4, 5).

The rows name pre-3.0.0-cut file paths under ``kg/``, ``observability/`` and
``protocols/``.  Those top-level trees were removed with the 3.0.0 cut, so
each retirement is pinned from the live tree: the named files must be absent
relative to the repository root, and no source module may still import the
retired module names.  For the quantum row the row's named symbols must
additionally be unreferenced anywhere in the source tree.

The importer census scans every top-level package directory present in the
checkout (the ``agent_utilities`` package today; any sibling package dir that
appears later is picked up automatically). Tests and generated build output are
excluded.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Top-level package dirs (excluding the test tree) scanned for lingering importers.
PACKAGE_DIRS = [
    d
    for d in sorted(REPO_ROOT.iterdir())
    if d.is_dir()
    and d.name not in {"tests", "build", "dist"}
    and not d.name.startswith(".")
    and any(d.rglob("*.py"))
]


def _source_files() -> list[Path]:
    files: list[Path] = []
    for directory in PACKAGE_DIRS:
        files.extend(sorted(directory.rglob("*.py")))
    return files


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def _assert_retired_absent(*relative_paths: str) -> None:
    present = [rel for rel in relative_paths if (REPO_ROOT / rel).exists()]
    assert not present, f"retired files still present: {present}"


def _touches(name: str, module_names: tuple[str, ...]) -> bool:
    return any(
        name == module or name.startswith(module + ".") for module in module_names
    )


def _importer_hits(
    module_names: tuple[str, ...],
    from_map: dict[str, tuple[str, ...]] | None = None,
) -> list[str]:
    """Return 'file:line:what' hits where a source line imports a retired name.

    Catches ``import <module>`` (including deeper submodules), ``from
    <module> import ...``, ``from <package> import <module>`` via *from_map*,
    and string-form dynamic imports such as
    ``importlib.import_module("<module>")``.
    """
    from_map = from_map or {}
    hits: list[str] = []
    for path in _source_files():
        text = path.read_text(encoding="utf-8")
        tree = None
        try:
            tree = ast.parse(text)
        except SyntaxError:
            pass
        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if _touches(alias.name, module_names):
                            hits.append(
                                f"{_rel(path)}:{node.lineno}: import {alias.name}"
                            )
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if _touches(module, module_names):
                        hits.append(
                            f"{_rel(path)}:{node.lineno}: from {module} import ..."
                        )
                    elif module in from_map:
                        for alias in node.names:
                            if alias.name in from_map[module]:
                                hits.append(
                                    f"{_rel(path)}:{node.lineno}: from {module} import {alias.name}"
                                )
        for lineno, line in enumerate(text.splitlines(), start=1):
            if "import" not in line:
                continue
            for name in module_names:
                if name in line:
                    hits.append(f"{_rel(path)}:{lineno}: {line.strip()}")
    return hits


def _symbol_hits(symbols: tuple[str, ...]) -> list[str]:
    hits: list[str] = []
    for path in _source_files():
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            for symbol in symbols:
                if re.search(rf"\b{re.escape(symbol)}\b", line):
                    hits.append(f"{_rel(path)}:{lineno}: {line.strip()}")
    return hits


@pytest.mark.spec("AU-BOUNDARY-R028.1")
def test_standardization_retirement_holds():
    """The three retired kg/standardization modules are absent and unimported.

    The row's "(excludes drift.py)" clause is stale: ``drift.py`` is itself
    absent from the post-3.0.0 tree, so only the three named files are
    asserted by this row.
    """
    _assert_retired_absent(
        "kg/standardization/cohorts.py",
        "kg/standardization/consolidation.py",
        "kg/standardization/standards.py",
    )
    hits = _importer_hits(
        (
            "kg.standardization.cohorts",
            "kg.standardization.consolidation",
            "kg.standardization.standards",
        ),
        from_map={"kg.standardization": ("cohorts", "consolidation", "standards")},
    )
    assert not hits, f"importers of retired standardization modules: {hits}"


@pytest.mark.spec("AU-BOUNDARY-R028.3")
def test_quantum_retirement_holds():
    """The kg/quantum package and observability/quantum_trace.py are absent.

    No module may import the retired names, and the row's named symbols must
    be unreferenced anywhere in the source tree.
    """
    _assert_retired_absent(
        "kg/quantum/budget.py",
        "kg/quantum/__init__.py",
        "observability/quantum_trace.py",
    )
    hits = _importer_hits(
        ("kg.quantum.budget", "kg.quantum", "observability.quantum_trace"),
        from_map={"kg": ("quantum",)},
    )
    assert not hits, f"importers of retired quantum modules: {hits}"
    symbol_hits = _symbol_hits(
        ("QuantumBudgetExceeded", "reserve_quantum_budget", "persist_quantum_job"),
    )
    assert not symbol_hits, f"references to retired quantum symbols: {symbol_hits}"


@pytest.mark.spec("AU-BOUNDARY-R028.4")
def test_security_retirement_holds():
    """All four retired kg/security modules are absent and unimported."""
    _assert_retired_absent(
        "kg/security/cognitive_trap_defense.py",
        "kg/security/policy_ingestor.py",
        "kg/security/graph_validator.py",
        "kg/security/rule_ingestor.py",
    )
    hits = _importer_hits(
        (
            "kg.security.cognitive_trap_defense",
            "kg.security.policy_ingestor",
            "kg.security.graph_validator",
            "kg.security.rule_ingestor",
        ),
        from_map={
            "kg.security": (
                "cognitive_trap_defense",
                "policy_ingestor",
                "graph_validator",
                "rule_ingestor",
            ),
        },
    )
    assert not hits, f"importers of retired security modules: {hits}"


@pytest.mark.spec("AU-BOUNDARY-R028.5")
def test_enterprise_protocols_retirement_holds():
    """The protocols/enterprise package and read_models are absent and unimported."""
    _assert_retired_absent(
        "protocols/enterprise/__init__.py",
        "protocols/enterprise/read_models.py",
    )
    hits = _importer_hits(
        ("protocols.enterprise.read_models", "protocols.enterprise"),
        from_map={"protocols": ("enterprise",)},
    )
    assert not hits, f"importers of retired enterprise protocols: {hits}"
