from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).parents[2]


def _gate_module() -> ModuleType:
    source = ROOT / "scripts" / "check_version_consistency.py"
    spec = importlib.util.spec_from_file_location("check_version_consistency", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_version_has_one_authority() -> None:
    gate = _gate_module()

    assert gate.validate(ROOT) == []
