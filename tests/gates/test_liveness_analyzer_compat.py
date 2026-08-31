"""Regression coverage for the liveness gate's analyzer compatibility seam."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_liveness.py"


def _load_gate():
    spec = importlib.util.spec_from_file_location(
        "_check_liveness_analyzer_compat_under_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_analyzer(
    tmp_path: Path, *, with_helper: bool, omit: str | None = None
) -> Path:
    declarations = {
        "_facade_branches": "def _facade_branches(node):\n    return {}\n",
        "_facade_except_handlers": (
            "def _facade_except_handlers(node):\n    return {}\n"
        ),
        "_does_real_work": "def _does_real_work(node):\n    return False\n",
        "_returns_canned_payload": (
            "def _returns_canned_payload(node):\n    return False\n"
        ),
        "_decorator_names": "def _decorator_names(node):\n    return set()\n",
        "_is_test": "def _is_test(path):\n    return False\n",
        "_SURFACE_PARTS": "_SURFACE_PARTS = frozenset()\n",
        "_SURFACE_DECORATORS": "_SURFACE_DECORATORS = frozenset()\n",
        "_INFO_NAMES_RE": '_INFO_NAMES_RE = re.compile(r"^$")\n',
        "_PLACEHOLDER_RE": '_PLACEHOLDER_RE = re.compile(r"TODO")\n',
    }
    source = "import re\n\n"
    source += "\n".join(
        declaration for name, declaration in declarations.items() if name != omit
    )
    if with_helper:
        source += (
            "\n\nimport hashlib\n\n"
            "def _stable_finding_id(text):\n"
            '    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:8]\n'
        )
    path = tmp_path / "analyze_liveness.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def test_placeholder_ids_are_stable_with_or_without_private_helper(tmp_path):
    gate = _load_gate()
    marker = "  TODO: retain this marker identity across analyzer versions  "
    ids = {}
    for with_helper in (False, True):
        analyzer = gate._import_analyzer(
            _write_analyzer(tmp_path / str(with_helper), with_helper=with_helper)
        )
        source = f"before = 1\n{marker}\nafter = 2\n"
        ids[with_helper] = list(gate._placeholder_ids(analyzer, source))

    stable_id = hashlib.sha256(marker.strip().encode("utf-8")).hexdigest()[:8]
    expected = [gate._hash("placeholder", stable_id)]
    assert ids[False] == expected
    assert ids[True] == expected


def test_import_remains_fail_closed_for_missing_required_surface(tmp_path):
    gate = _load_gate()
    analyzer_path = _write_analyzer(tmp_path, with_helper=False, omit="_PLACEHOLDER_RE")

    with pytest.raises(SystemExit) as exc_info:
        gate._import_analyzer(analyzer_path)

    assert exc_info.value.code == 2


def test_present_but_invalid_private_helper_remains_fail_closed(tmp_path):
    gate = _load_gate()
    analyzer = gate._import_analyzer(_write_analyzer(tmp_path, with_helper=True))
    analyzer._stable_finding_id = None

    with pytest.raises(SystemExit) as exc_info:
        list(gate._placeholder_ids(analyzer, "TODO: invalid helper"))

    assert exc_info.value.code == 2
