from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = (
        Path(__file__).resolve().parents[3]
        / "scripts/check_native_change_envelope_boundary.py"
    )
    spec = importlib.util.spec_from_file_location("native_change_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repository_satisfies_native_change_boundary() -> None:
    assert _module().violations() == []


def test_gate_detects_public_sequential_bypass(tmp_path, monkeypatch) -> None:
    module = _module()
    ingest = tmp_path / "envelope_ingest.py"
    ingest.write_text(
        """
def _apply_native_change_envelope():
    client.changes.apply({})
def ingest_envelope():
    _apply_write()
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "INGEST", ingest)
    failures = module.violations()
    assert any("sequential durability steps" in failure for failure in failures)
