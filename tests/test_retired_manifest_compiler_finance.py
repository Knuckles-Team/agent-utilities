"""Retirement-census proofs for the ontology manifest compiler and the
fabricated-data finance modules.

Both retirements are already achieved on ``origin/main`` (post-3.0.0 cut):
``agent_utilities/knowledge_graph/ontology/manifest_compiler.py`` is gone,
and the finance modules that produced fabricated data are gone with it.
These tests bind that state:

* the retired path is absent relative to the repo root (and no same-named
  file reappeared anywhere in the package), and
* an import census over ``agent_utilities/`` and ``tests/`` finds zero
  callers: an AST scan of ``import``/``from`` statements plus a line scan
  for dynamic (string-literal) references to the full dotted paths.

The named-module list for the finance census is taken verbatim from the
``AU-BOUNDARY-R041`` row in
``specs/au-boundary-deconstruction/requirements.md``.

Pure Python: no network, no git subprocesses.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Top-level Python package directories present on origin/main.
PACKAGE_DIRS = ("agent_utilities", "tests")

MANIFEST_COMPILER = "manifest_compiler"
MANIFEST_COMPILER_DOTTED = "agent_utilities.knowledge_graph.ontology.manifest_compiler"

# The exact module names the AU-BOUNDARY-R041 row in requirements.md names.
R041_NAMED_MODULES = (
    "trade_journal",
    "strategy_export",
    "strategy_sharing",
    "strategy_engine",
    "streaming",
    "market_feeds",
    "market_data",
    "crypto_connector",
    "exchange_bridge",
    "banking",
    "payments",
    "quant_tasks",
)


def _r041_canonical_path(module_name: str) -> Path:
    """Pre-cut location: 11 modules under domains/finance/, quant_tasks under
    knowledge_graph/core/ (plan.md inventory)."""
    if module_name == "quant_tasks":
        return (
            REPO_ROOT
            / "agent_utilities"
            / "knowledge_graph"
            / "core"
            / "quant_tasks.py"
        )
    return REPO_ROOT / "agent_utilities" / "domains" / "finance" / f"{module_name}.py"


def _python_files() -> list[Path]:
    files: list[Path] = []
    for name in PACKAGE_DIRS:
        base = REPO_ROOT / name
        if base.is_dir():
            files.extend(sorted(base.rglob("*.py")))
    return files


def _import_census(
    module_names: tuple[str, ...], dotted_paths: tuple[str, ...]
) -> list[str]:
    """Return one evidence line per import of any retired module name."""
    hits: list[str] = []
    for path in _python_files():
        source = path.read_text(encoding="utf-8")
        rel = path.relative_to(REPO_ROOT)
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            tree = None
        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        dotted = alias.name
                        if dotted in module_names or any(
                            dotted.endswith(f".{name}") for name in module_names
                        ):
                            hits.append(f"{rel}:{node.lineno}: import {dotted}")
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    if mod in module_names or any(
                        mod.endswith(f".{name}") or mod.startswith(f"{name}.")
                        for name in module_names
                    ):
                        hits.append(f"{rel}:{node.lineno}: from {mod} import ...")
                    for alias in node.names:
                        if alias.name in module_names:
                            hits.append(
                                f"{rel}:{node.lineno}: from {mod or '<relative>'} import {alias.name}"
                            )
        for lineno, line in enumerate(source.splitlines(), start=1):
            if "import" not in line:
                continue
            for dotted in dotted_paths:
                if dotted in line:
                    hits.append(f"{rel}:{lineno}: {line.strip()}")
    return hits


@pytest.mark.spec("AU-BOUNDARY-R030.6")
def test_manifest_compiler_deleted_without_importers():
    """The retired manifest compiler is absent and no caller remains."""
    retired = (
        REPO_ROOT
        / "agent_utilities"
        / "knowledge_graph"
        / "ontology"
        / "manifest_compiler.py"
    )
    assert not retired.exists(), (
        f"retired path still present: {retired.relative_to(REPO_ROOT)}"
    )
    strays = [
        p
        for p in (REPO_ROOT / "agent_utilities").rglob("*.py")
        if p.stem == MANIFEST_COMPILER
    ]
    assert strays == [], (
        f"manifest_compiler.py reappeared: {[str(p.relative_to(REPO_ROOT)) for p in strays]}"
    )
    hits = _import_census((MANIFEST_COMPILER,), (MANIFEST_COMPILER_DOTTED,))
    assert hits == [], (
        "import census found callers of manifest_compiler:\n" + "\n".join(hits)
    )


@pytest.mark.spec("AU-BOUNDARY-R041")
@pytest.mark.parametrize("module_name", R041_NAMED_MODULES)
def test_retired_finance_module_absent_without_importers(module_name):
    """Each named fabricated-data finance module is gone and unreferenced."""
    canonical = _r041_canonical_path(module_name)
    assert not canonical.exists(), (
        f"retired path still present: {canonical.relative_to(REPO_ROOT)}"
    )
    strays = [
        p
        for p in (REPO_ROOT / "agent_utilities").rglob("*.py")
        if p.stem == module_name
    ]
    assert strays == [], (
        f"{module_name}.py reappeared: {[str(p.relative_to(REPO_ROOT)) for p in strays]}"
    )
    if module_name == "quant_tasks":
        dotted = ("agent_utilities.knowledge_graph.core.quant_tasks",)
    else:
        dotted = (f"agent_utilities.domains.finance.{module_name}",)
    hits = _import_census((module_name,), dotted)
    assert hits == [], f"import census found callers of {module_name}:\n" + "\n".join(
        hits
    )
