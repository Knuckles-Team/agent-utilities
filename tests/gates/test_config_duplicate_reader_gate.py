"""Gate: AU's retained core/config.py has no duplicate environment reader.

Binds AU-BOUNDARY-R016.1 -- the regression invariant AU's agent-and-model
settings portion of core/config.py must keep once connector-base settings
move to the agent connector SDK and hosting settings move to graph-os.
"""

from __future__ import annotations

import ast
import tempfile
from collections import Counter
from pathlib import Path

import pytest

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "agent_utilities" / "core" / "config.py"


def _env_var_names(source: str) -> list[str]:
    """Return every literal env-var name read via os.getenv/os.environ.get/[]."""

    tree = ast.parse(source)
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            func = node.func
            attr = func.attr
            owner = func.value
            is_os_getenv = (
                attr == "getenv"
                and isinstance(owner, ast.Name)
                and owner.id == "os"
            )
            is_environ_get = (
                attr == "get"
                and isinstance(owner, ast.Attribute)
                and owner.attr == "environ"
                and isinstance(owner.value, ast.Name)
                and owner.value.id == "os"
            )
            if (is_os_getenv or is_environ_get) and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    names.append(first.value)
        elif isinstance(node, ast.Subscript):
            value = node.value
            if (
                isinstance(value, ast.Attribute)
                and value.attr == "environ"
                and isinstance(value.value, ast.Name)
                and value.value.id == "os"
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)
            ):
                names.append(node.slice.value)
    return names


def duplicate_env_readers(source: str) -> list[str]:
    counts = Counter(_env_var_names(source))
    return sorted(name for name, count in counts.items() if count > 1)


@pytest.mark.spec("AU-BOUNDARY-R016.1")
def test_config_has_no_duplicate_environment_reader() -> None:
    source = _CONFIG_PATH.read_text(encoding="utf-8")
    assert duplicate_env_readers(source) == []


@pytest.mark.spec("AU-BOUNDARY-R016.1")
def test_gate_detects_an_injected_duplicate_reader() -> None:
    injected = (
        "import os\n"
        "a = os.getenv('DUPLICATE_VAR')\n"
        "b = os.environ.get('DUPLICATE_VAR')\n"
    )
    assert duplicate_env_readers(injected) == ["DUPLICATE_VAR"]
