"""Source contract tests for the exact-artifact closure layer."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "check_exact_artifact_closure.py"


def _load():
    specification = importlib.util.spec_from_file_location(
        "check_exact_artifact_closure", SCRIPT
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_current_exact_artifact_closure_contract_is_complete() -> None:
    assert _load().check_contract() == ()


def test_campaign_orchestrator_order_is_source_frozen(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load()
    original = module.ORCHESTRATOR.read_text(encoding="utf-8")
    altered = original.replace(
        '    "performance",\n    "fault-restart",',
        '    "fault-restart",\n    "performance",',
        1,
    )
    assert altered != original
    orchestrator = tmp_path / "run_exact_engine_campaigns.py"
    orchestrator.write_text(altered, encoding="utf-8")
    monkeypatch.setattr(module, "ORCHESTRATOR", orchestrator)

    assert "closure-engine-campaign-orchestrator" in module.check_contract()


def test_campaign_orchestrator_subprocess_boundary_is_source_frozen(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load()
    original = module.ORCHESTRATOR.read_text(encoding="utf-8")
    altered = original.replace("shell=False", "shell=True", 1)
    assert altered != original
    orchestrator = tmp_path / "run_exact_engine_campaigns.py"
    orchestrator.write_text(altered, encoding="utf-8")
    monkeypatch.setattr(module, "ORCHESTRATOR", orchestrator)

    assert "closure-engine-campaign-orchestrator" in module.check_contract()


def test_every_source_exact_gate_has_release_authority(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load()
    original = module.COMPATIBILITY.read_text(encoding="utf-8")
    altered = original.replace(
        '    "G-01": ("certification:exactArtifactClosureEvidence",),\n',
        "",
        1,
    )
    assert altered != original
    compatibility = tmp_path / "check_compatibility.py"
    compatibility.write_text(altered, encoding="utf-8")
    monkeypatch.setattr(module, "COMPATIBILITY", compatibility)

    assert "closure-release-gate-inventory" in module.check_contract()
