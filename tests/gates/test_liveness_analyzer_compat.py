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


# ── EH-330: placeholder detection needs CODE-position context ───────────────
# `_PLACEHOLDER_RE` matches raw text; these tests prove `_placeholder_ids` no
# longer counts a hit sitting in docstring prose or in a non-marker comment,
# while still catching a real stub (`raise NotImplementedError`, a bare
# `pass`/`...` body, or a leading work-marker tag) — using the
# REAL vendored `_PLACEHOLDER_RE` (copied verbatim) and a real
# `_decorator_names`, not the minimal single-marker fixture above, so the
# masking logic is exercised against the actual trigger phrases.


def _write_real_placeholder_analyzer(tmp_path: Path) -> Path:
    source = """
import re

def _facade_branches(node):
    return {}

def _facade_except_handlers(node):
    return {}

def _does_real_work(node):
    return False

def _returns_canned_payload(node):
    return False

def _decorator_names(node):
    out = set()
    for d in getattr(node, "decorator_list", []) or []:
        t = d.func if isinstance(d, __import__("ast").Call) else d
        if isinstance(t, __import__("ast").Name):
            out.add(t.id)
        elif isinstance(t, __import__("ast").Attribute):
            out.add(t.attr)
    return out

def _is_test(path):
    return False

_SURFACE_PARTS = frozenset()
_SURFACE_DECORATORS = frozenset()
_INFO_NAMES_RE = re.compile(r"^$")
_PLACEHOLDER_RE = re.compile(
    r"\\b(TODO|FIXME|XXX|HACK|stub(?:bed|s)?|placeholder|mock(?:ed)?|dummy|"
    r"for now|not[ _-]?implemented|coming soon|hard[ -]?coded|sample data|"
    r"example only|fake|lorem ipsum|in (?:a|the) real|real implementation|"
    r"would (?:be|go) here|replace this|simulate[d]?)\\b",
    re.IGNORECASE,
)
"""
    path = tmp_path / "analyze_liveness.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


@pytest.fixture
def real_placeholder_analyzer(tmp_path):
    gate = _load_gate()
    analyzer = gate._import_analyzer(_write_real_placeholder_analyzer(tmp_path))
    return gate, analyzer


@pytest.mark.spec("AU-QUAL-R007")
def test_docstring_prose_is_not_a_placeholder_finding(real_placeholder_analyzer):
    gate, an = real_placeholder_analyzer
    src = (
        "def handler():\n"
        '    """This capability is designed, not implemented yet."""\n'
        "    return 1\n"
    )
    assert list(gate._placeholder_ids(an, src)) == []


@pytest.mark.spec("AU-QUAL-R007")
def test_prose_comment_mid_sentence_is_not_a_placeholder_finding(
    real_placeholder_analyzer,
):
    gate, an = real_placeholder_analyzer
    src = (
        "def handler():\n"
        "    # this path is designed, not implemented, see EH-270\n"
        "    return 1\n"
    )
    assert list(gate._placeholder_ids(an, src)) == []


def test_raise_not_implemented_error_is_a_placeholder_finding(
    real_placeholder_analyzer,
):
    gate, an = real_placeholder_analyzer
    src = "def handler():\n    raise NotImplementedError\n"
    assert len(list(gate._placeholder_ids(an, src))) == 1


def test_leading_todo_marker_comment_is_a_placeholder_finding(
    real_placeholder_analyzer,
):
    gate, an = real_placeholder_analyzer
    src = "def handler():\n    # TODO: implement\n    return 1\n"
    assert len(list(gate._placeholder_ids(an, src))) == 1


def test_bare_pass_body_is_a_placeholder_finding(real_placeholder_analyzer):
    gate, an = real_placeholder_analyzer
    src = "def handler():\n    pass\n"
    assert len(list(gate._placeholder_ids(an, src))) == 1


def test_bare_ellipsis_body_is_a_placeholder_finding(real_placeholder_analyzer):
    gate, an = real_placeholder_analyzer
    src = "def handler():\n    ...\n"
    assert len(list(gate._placeholder_ids(an, src))) == 1


def test_protocol_method_ellipsis_body_is_not_a_placeholder_finding(
    real_placeholder_analyzer,
):
    gate, an = real_placeholder_analyzer
    src = (
        "from typing import Protocol\n\n"
        "class Port(Protocol):\n"
        "    def search(self, request) -> int: ...\n"
    )
    assert list(gate._placeholder_ids(an, src)) == []


def test_abstractmethod_ellipsis_body_is_not_a_placeholder_finding(
    real_placeholder_analyzer,
):
    gate, an = real_placeholder_analyzer
    src = (
        "from abc import abstractmethod\n\n"
        "class Base:\n"
        "    @abstractmethod\n"
        "    def search(self, request) -> int: ...\n"
    )
    assert list(gate._placeholder_ids(an, src)) == []


def test_placeholder_string_literal_used_as_value_is_still_a_finding(
    real_placeholder_analyzer,
):
    gate, an = real_placeholder_analyzer
    src = "def handler():\n    return 'placeholder response'\n"
    assert len(list(gate._placeholder_ids(an, src))) == 1
