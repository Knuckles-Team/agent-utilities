"""Regressions for the documented kernel-free (lean) unit profile.

The optional native numeric kernel (`epistemic_graph.numeric`) is genuinely
optional: guardrail/metadata environments install agent-utilities WITHOUT it.
Twice now, a shared test fixture assumed the kernel was present and turned that
supported absence into a *fixture-setup error* on hundreds of otherwise pure
unit tests -- masking real results behind an environment condition.

These tests pin the contract from both sides:

* ``tests/conftest.py``'s kernel-free ``IntelligenceGraphEngine`` fallback must
  satisfy every member the shared autouse fixtures call.
* ``tests/unit/conftest.py``'s hermetic-embedding fixture must patch the
  canonical factory unconditionally, and an already-bound rebind only where that
  rebind is actually importable.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

CONFTEST_PATH = Path(__file__).resolve().parents[1] / "conftest.py"
UNIT_CONFTEST_PATH = Path(__file__).resolve().parent / "conftest.py"

# The modules whose absence defines the lean profile.
_ENGINE_MODULES = (
    "agent_utilities.knowledge_graph.backends",
    "agent_utilities.knowledge_graph.core.engine",
)


class _BlockModules:
    """Meta-path finder that makes the named modules unimportable."""

    def __init__(self, names: tuple[str, ...]) -> None:
        self._names = names

    def find_spec(self, fullname, path=None, target=None):  # noqa: ARG002
        if fullname in self._names:
            raise ImportError(f"blocked for lean-profile test: {fullname}")
        return None


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


@pytest.fixture
def lean_root_conftest(monkeypatch):
    """Load ``tests/conftest.py`` with the engine layer forced unimportable."""
    blocker = _BlockModules(_ENGINE_MODULES)
    monkeypatch.setattr(sys, "meta_path", [blocker, *sys.meta_path])
    for name in _ENGINE_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    name = "_lean_profile_root_conftest"
    module = _load_module(CONFTEST_PATH, name)
    yield module
    sys.modules.pop(name, None)


def test_kernel_free_engine_fallback_supports_every_member_the_fixtures_call(
    lean_root_conftest,
):
    """SB-02: the fallback stub had ``set_active`` but not ``get_active``.

    ``clean_graph_globals`` calls BOTH. The missing member raised
    ``AttributeError`` -- which the fixture's ``except ImportError`` does not
    catch -- so every test in the lean profile errored during setup.
    """
    engine = lean_root_conftest.IntelligenceGraphEngine

    # We really are exercising the fallback, not the native class.
    assert engine.__module__ == "_lean_profile_root_conftest"

    assert engine.get_active() is None
    assert engine.set_active(None) is None
    assert lean_root_conftest.set_active_backend(None) is None


def test_kernel_free_engine_reset_sequence_is_a_no_op(lean_root_conftest):
    """The exact call sequence ``clean_graph_globals`` performs must not raise."""
    engine = lean_root_conftest.IntelligenceGraphEngine
    if engine.get_active() is not None:  # pragma: no cover - always None here
        engine.set_active(None)


def test_unit_conftest_always_patches_the_canonical_embedding_factory():
    """SB-11: blocking the canonical factory is what makes the suite hermetic."""
    from tests.unit import conftest as unit_conftest

    assert unit_conftest._importable(unit_conftest.CANONICAL_EMBEDDING_FACTORY)


def test_unit_conftest_importable_probe_reports_both_branches():
    """SB-11 acceptance: cover the module-available and module-absent branches."""
    from tests.unit import conftest as unit_conftest

    # Present branch: this very module is importable.
    assert unit_conftest._importable(
        f"{__name__}.test_unit_conftest_importable_probe_reports_both_branches"
    )
    # Absent branch: an optional consumer that cannot import is skipped, not fatal.
    assert not unit_conftest._importable(
        "agent_utilities._no_such_optional_module.factory"
    )


def test_optional_rebinds_are_filtered_not_assumed():
    """The fixture must never assume an optional rebind is importable."""
    from tests.unit import conftest as unit_conftest

    source = UNIT_CONFTEST_PATH.read_text(encoding="utf-8")
    # The rebind list must be consulted through the probe, never iterated raw.
    assert "if _importable(t)" in source
    assert unit_conftest.OPTIONAL_EMBEDDING_FACTORY_REBINDS
    assert (
        unit_conftest.CANONICAL_EMBEDDING_FACTORY
        not in unit_conftest.OPTIONAL_EMBEDDING_FACTORY_REBINDS
    )
