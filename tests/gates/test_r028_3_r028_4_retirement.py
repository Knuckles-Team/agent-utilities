"""Acceptance tests for AU-BOUNDARY-R028.3 and AU-BOUNDARY-R028.4.

R028.3: ``kg/quantum/{budget,__init__}.py`` and ``observability/quantum_trace.py``
are retired: none of the three files exists, and no production module imports
the deleted files or references their symbols
(``reserve_quantum_budget``/``QuantumBudgetExceeded``/``persist_quantum_job``).

R028.4: ``kg/security/cognitive_trap_defense.py`` is deleted (it had only a
test importer). The remaining three -- ``policy_ingestor.py``,
``rule_ingestor.py``, ``graph_validator.py`` -- stay, because each still has
a real production importer with no epistemic-graph client equivalent yet
(``engine_ingestion.py`` imports the first two, ``pipeline/phases/validate.py``
imports the third); this test pins exactly that blocked state.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.gates._deletion_support import imports_any as _imports_any

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "agent_utilities"


@pytest.mark.spec("AU-BOUNDARY-R028.3")
def test_r028_3_quantum_modules_deleted_and_unreferenced() -> None:
    for rel in (
        "knowledge_graph/quantum/budget.py",
        "knowledge_graph/quantum/__init__.py",
        "observability/quantum_trace.py",
    ):
        assert not (PACKAGE_ROOT / rel).exists(), f"{rel} should be deleted"

    offending: list[str] = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        if _imports_any(
            path, ("knowledge_graph.quantum", "observability.quantum_trace")
        ):
            offending.append(path.relative_to(PACKAGE_ROOT).as_posix())
    assert offending == [], f"stale importer(s) of deleted quantum modules: {offending}"

    for path in PACKAGE_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for symbol in (
            "reserve_quantum_budget",
            "QuantumBudgetExceeded",
            "persist_quantum_job",
        ):
            assert symbol not in text, (
                f"{path.relative_to(PACKAGE_ROOT)} still references retired "
                f"symbol {symbol!r}"
            )


@pytest.mark.spec("AU-BOUNDARY-R028.4")
def test_r028_4_cognitive_trap_defense_deleted_and_remaining_trio_blocked() -> None:
    # The one module with only a test importer is gone.
    target = PACKAGE_ROOT / "knowledge_graph/security/cognitive_trap_defense.py"
    assert not target.exists(), "cognitive_trap_defense.py should be deleted"

    # The other three remain, each with the real production importer that
    # blocks deletion until EG exposes an equivalent.
    remaining = {
        "knowledge_graph/security/policy_ingestor.py": (
            "knowledge_graph/core/engine_ingestion.py",
        ),
        "knowledge_graph/security/rule_ingestor.py": (
            "knowledge_graph/core/engine_ingestion.py",
        ),
        "knowledge_graph/security/graph_validator.py": (
            "knowledge_graph/pipeline/phases/validate.py",
        ),
    }
    for module_rel, importer_rels in remaining.items():
        assert (PACKAGE_ROOT / module_rel).exists(), (
            f"{module_rel} unexpectedly deleted"
        )
        module_name = module_rel[:-3].replace("/", ".")
        short_name = module_name.rsplit(".", 1)[-1]
        for importer_rel in importer_rels:
            importer_path = PACKAGE_ROOT / importer_rel
            assert importer_path.exists(), importer_rel
            assert _imports_any(importer_path, (module_name, f".{short_name}")), (
                f"expected {importer_rel} to still import {module_rel} "
                "(no EG client equivalent yet)"
            )
