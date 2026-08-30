"""Regression tests for the alias-aware caller discovery command."""

from pathlib import Path

from scripts.find_callers import find_callers

SYMBOL = "agent_utilities.security.threat_defense_engine.GuardrailEngine"


def _write_caller(root: Path, source: str) -> None:
    caller = root / "src" / "caller.py"
    caller.parent.mkdir(parents=True)
    caller.write_text(source, encoding="utf-8")


def test_finds_alias_calls_references_and_dynamic_dispatch(tmp_path: Path) -> None:
    _write_caller(
        tmp_path,
        """
from agent_utilities.security.threat_defense_engine import GuardrailEngine as Engine
from agent_utilities.security import threat_defense_engine as module


def use() -> object:
    Engine()
    module.GuardrailEngine()
    value = Engine
    getattr(module, "GuardrailEngine")
    monkeypatch.setattr(module, "GuardrailEngine", object)
    return value
""",
    )

    hits = find_callers(SYMBOL, roots=("src",), repo_root=tmp_path)

    assert {(hit.kind, hit.snippet) for hit in hits} == {
        ("call", "Engine(...)"),
        ("call", "module.GuardrailEngine(...)"),
        ("reference", "Engine"),
        ("monkeypatch", "setattr(module, 'GuardrailEngine', ...)"),
        ("getattr", "getattr(module, 'GuardrailEngine')"),
    }
    assert {hit.file for hit in hits} == {"src/caller.py"}


def test_does_not_report_unresolved_attribute_with_matching_suffix(
    tmp_path: Path,
) -> None:
    _write_caller(
        tmp_path,
        """
def use(local: object) -> object:
    return local.GuardrailEngine()
""",
    )

    assert find_callers(SYMBOL, roots=("src",), repo_root=tmp_path) == []
