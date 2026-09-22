"""Meta-test: the Wire-First gate (``scripts/check_wiring.py``'s D-OB-9/13/16
sweeps) trips on synthetic fixtures shaped like the real defects it exists to
catch, and stays clean on wired-up equivalents. "A gate that can't fail is
not a gate" (see ``test_swallowed_errors_gate.py``, the template for this
file).

Imported directly (not via subprocess, unlike most ``scripts/check_*.py``
meta-tests) because the checks under test take fixture ``Path`` overrides
(``src_dir``/``tests_dir``/``display_root``) rather than a CLI ``--root``
flag — ``scripts/`` is not a package, so ``importlib`` loads it by path.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_wiring.py"


def _load_check_wiring():
    spec = importlib.util.spec_from_file_location("_check_wiring_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check_wiring = _load_check_wiring()


# ---------------------------------------------------------------------------
# D-OB-13a — uncollected test files
# ---------------------------------------------------------------------------


def _synthetic_config_paths(tmp_path):
    """An isolated ``pytest.ini``/``.config/pre-commit.yaml``/workflows-dir
    triple, so ``find_orphaned_test_files`` can be exercised without falling
    back to reading THIS repo's own live config (which would make the
    fixture's verdict depend on whatever ``testpaths`` this repo currently
    happens to declare, rather than on the synthetic scenario under test)."""
    pytest_ini = tmp_path / "pytest.ini"
    pytest_ini.write_text(
        "[pytest]\ntestpaths = tests/unit tests/integration tests/retrieval\n"
    )
    return {
        "pytest_ini": pytest_ini,
        "precommit_config": tmp_path / ".config" / "pre-commit.yaml",  # absent -> empty
        "workflows_dir": tmp_path / ".github" / "workflows",  # absent -> empty
    }


def test_orphan_gate_trips_on_a_loose_test_file_outside_testpaths(tmp_path):
    """A ``test_*.py`` sitting directly under a synthetic ``tests/`` root
    (not ``tests/unit``/``tests/integration``/``tests/retrieval``, and not
    named in any pre-commit/CI pytest invocation) must be flagged — this is
    exactly the ``tests/test_multiplexer_transports.py`` shape D-OB-13
    found.
    """
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_orphan.py").write_text("def test_x():\n    assert True\n")

    orphans = check_wiring.find_orphaned_test_files(
        tests_dir=tests_dir, display_root=tmp_path, **_synthetic_config_paths(tmp_path)
    )
    assert "tests/test_orphan.py" in orphans


def test_orphan_gate_does_not_flag_a_file_under_tests_unit(tmp_path):
    """A test file under ``tests/unit`` (a real ``testpaths`` entry) is
    never flagged — the gate must not cry wolf on ordinary, collected
    tests."""
    tests_dir = tmp_path / "tests"
    (tests_dir / "unit").mkdir(parents=True)
    (tests_dir / "unit" / "test_ok.py").write_text("def test_x():\n    assert True\n")

    orphans = check_wiring.find_orphaned_test_files(
        tests_dir=tests_dir, display_root=tmp_path, **_synthetic_config_paths(tmp_path)
    )
    assert "tests/unit/test_ok.py" not in orphans


# ---------------------------------------------------------------------------
# D-OB-13b — MagicMock(spec=[]) / patch(create=True) mock hygiene
# ---------------------------------------------------------------------------


def test_mock_hygiene_gate_trips_on_spec_empty_list(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_bad_mock.py").write_text(
        "from unittest.mock import MagicMock\n\n"
        "def test_x():\n"
        "    fake = MagicMock(spec=[])\n"
    )
    issues = check_wiring.find_mock_hygiene_issues(
        tests_dir=tests_dir, display_root=tmp_path
    )
    assert any(shape == "spec=[]" for _rel, _line, shape in issues)


def test_mock_hygiene_gate_ignores_a_docstring_mentioning_create_true(tmp_path):
    """A docstring merely mentioning ``create=True`` (explaining why the
    file does NOT use that shape, as the real ``test_graph_iter.py`` does)
    must never be flagged — this is AST ``ast.Call`` matching, not a raw
    line-regex, specifically to avoid that false positive."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_docstring_only.py").write_text(
        '"""Explains why we do NOT use patch(..., create=True) here."""\n\n'
        "def test_x():\n"
        "    assert True\n"
    )
    issues = check_wiring.find_mock_hygiene_issues(
        tests_dir=tests_dir, display_root=tmp_path
    )
    assert issues == []


def test_mock_hygiene_gate_trips_on_patch_create_true(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_bad_patch.py").write_text(
        "from unittest.mock import patch\n\n"
        "def test_x():\n"
        "    with patch('some.module.Thing', create=True):\n"
        "        pass\n"
    )
    issues = check_wiring.find_mock_hygiene_issues(
        tests_dir=tests_dir, display_root=tmp_path
    )
    assert any(shape == "create=True" for _rel, _line, shape in issues)


# ---------------------------------------------------------------------------
# D-OB-16 — silently-swallowed optional-extra import guards
# ---------------------------------------------------------------------------


def test_extras_gating_gate_trips_on_silent_import_error_pass(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_silent_skip.py").write_text(
        "def test_x():\n"
        "    try:\n"
        "        import some_optional_extra\n"
        "    except ImportError:\n"
        "        pass\n"
    )
    guards = check_wiring.find_silent_import_guards(
        tests_dir=tests_dir, display_root=tmp_path
    )
    assert guards != []


def test_extras_gating_gate_does_not_flag_a_visible_pytest_skip(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_visible_skip.py").write_text(
        "import pytest\n\n"
        "def test_x():\n"
        "    try:\n"
        "        import some_optional_extra\n"
        "    except ImportError:\n"
        "        pytest.skip('some_optional_extra not installed')\n"
    )
    guards = check_wiring.find_silent_import_guards(
        tests_dir=tests_dir, display_root=tmp_path
    )
    assert guards == []


# ---------------------------------------------------------------------------
# D-OB-9 — public symbol with no non-test caller
# ---------------------------------------------------------------------------


def test_symbol_gate_trips_on_a_class_referenced_only_from_tests(tmp_path):
    """The exact D-OB-9 shape: a class fully built, fully unit-tested, and
    never constructed anywhere else in ``agent_utilities/``."""
    src_dir = tmp_path / "agent_utilities"
    src_dir.mkdir()
    (src_dir / "orphan_module.py").write_text(
        "class NeverCalledPolicy:\n"
        "    def decide_something_distinctive(self):\n"
        "        return True\n"
    )
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_orphan_module.py").write_text(
        "from agent_utilities.orphan_module import NeverCalledPolicy\n\n"
        "def test_x():\n"
        "    assert NeverCalledPolicy().decide_something_distinctive()\n"
    )

    findings = check_wiring.find_test_only_symbols(
        src_dir=src_dir, tests_dir=tests_dir, display_root=tmp_path
    )
    symbols = {f["symbol"] for f in findings}
    assert "NeverCalledPolicy" in symbols


def test_symbol_gate_does_not_flag_a_class_with_a_live_caller(tmp_path):
    """The same class, but with a second module in agent_utilities/ that
    actually constructs it — the wired equivalent — must NOT be flagged."""
    src_dir = tmp_path / "agent_utilities"
    src_dir.mkdir()
    (src_dir / "wired_module.py").write_text(
        "class LiveCalledPolicy:\n"
        "    def decide_something_distinctive(self):\n"
        "        return True\n"
    )
    (src_dir / "live_caller.py").write_text(
        "from agent_utilities.wired_module import LiveCalledPolicy\n\n"
        "def run():\n"
        "    return LiveCalledPolicy().decide_something_distinctive()\n"
    )
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_wired_module.py").write_text(
        "from agent_utilities.wired_module import LiveCalledPolicy\n\n"
        "def test_x():\n"
        "    assert LiveCalledPolicy().decide_something_distinctive()\n"
    )

    findings = check_wiring.find_test_only_symbols(
        src_dir=src_dir, tests_dir=tests_dir, display_root=tmp_path
    )
    symbols = {f["symbol"] for f in findings}
    assert "LiveCalledPolicy" not in symbols


def test_symbol_gate_ignores_a_comment_only_mention(tmp_path):
    """A symbol name that appears ONLY inside a comment in another file
    (never actually constructed/called) must still be flagged — a raw
    line-regex over source text would have missed this by counting the
    comment as a "reference"; the real ``AdmissionPolicy`` instance hid
    behind exactly this shape (mentioned in a comment in
    ``engine_tasks.py``, never constructed) until this gate's tokenize-based
    scan was written to see past comments."""
    src_dir = tmp_path / "agent_utilities"
    src_dir.mkdir()
    (src_dir / "orphan_module2.py").write_text(
        "class StillNeverCalledPolicy:\n"
        "    def decide_something_else_distinctive(self):\n"
        "        return True\n"
    )
    (src_dir / "mentions_only.py").write_text(
        "# TODO: wire in StillNeverCalledPolicy() here eventually\n"
        "def run():\n"
        "    return None\n"
    )
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_orphan_module2.py").write_text(
        "from agent_utilities.orphan_module2 import StillNeverCalledPolicy\n\n"
        "def test_x():\n"
        "    assert StillNeverCalledPolicy().decide_something_else_distinctive()\n"
    )

    findings = check_wiring.find_test_only_symbols(
        src_dir=src_dir, tests_dir=tests_dir, display_root=tmp_path
    )
    symbols = {f["symbol"] for f in findings}
    assert "StillNeverCalledPolicy" in symbols


def test_symbol_gate_ignores_a_property_name_collision_with_an_unrelated_call(
    tmp_path,
):
    """The exact D-OB-9 false positive this fix closes: a ``@property``
    accessor (``Widget.fingerprint``) has zero real callers anywhere, but an
    UNRELATED class/method defined in a test file happens to share the bare
    name ``fingerprint`` and is invoked with call syntax
    (``loader.fingerprint(1)``). A property can never legitimately be
    referenced with call syntax — before this fix, the bare ``.name(``-call
    counter conflated the two (matching on the trailing token alone, with
    no class/module qualification) and flagged the property as test-only
    purely because of the name collision. Reproduces the real regression:
    ``scripts/dual_principal_validation.py``'s module-level ``fingerprint()``
    helper, called as ``mod.fingerprint(...)`` in its test, flipped THREE
    unrelated ``*.fingerprint`` properties (``PromptCacheKey``,
    ``SemanticCacheKey``, ``OAuthGrantBinding``) to "test-only" — verified
    this fixture trips on the pre-fix code before the gate was patched."""
    src_dir = tmp_path / "agent_utilities"
    src_dir.mkdir()
    (src_dir / "widget.py").write_text(
        "class Widget:\n"
        "    @property\n"
        "    def fingerprint(self) -> str:\n"
        "        return 'w'\n"
    )
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_unrelated.py").write_text(
        "class Loader:\n"
        "    def fingerprint(self, x):\n"
        "        return x\n\n"
        "def test_x():\n"
        "    loader = Loader()\n"
        "    assert loader.fingerprint(1) == 1\n"
    )

    findings = check_wiring.find_test_only_symbols(
        src_dir=src_dir, tests_dir=tests_dir, display_root=tmp_path
    )
    symbols = {f["symbol"] for f in findings}
    assert "Widget.fingerprint" not in symbols


def test_symbol_gate_still_trips_on_a_genuine_test_only_method(tmp_path):
    """A genuinely test-only REGULAR (non-property) method must still be
    flagged — proving the property exclusion above does not weaken
    detection for the much larger non-property case this gate exists to
    catch."""
    src_dir = tmp_path / "agent_utilities"
    src_dir.mkdir()
    (src_dir / "policy.py").write_text(
        "class OrphanPolicy:\n"
        "    def evaluate_distinctively(self):\n"
        "        return True\n"
    )
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_policy.py").write_text(
        "from agent_utilities.policy import OrphanPolicy\n\n"
        "def test_x():\n"
        "    assert OrphanPolicy().evaluate_distinctively()\n"
    )

    findings = check_wiring.find_test_only_symbols(
        src_dir=src_dir, tests_dir=tests_dir, display_root=tmp_path
    )
    symbols = {f["symbol"] for f in findings}
    assert "OrphanPolicy.evaluate_distinctively" in symbols


def test_symbol_gate_accepts_methods_on_explicitly_reexported_public_classes(
    tmp_path,
):
    """A public cross-repository class port needs no AU-local caller.

    The class is explicitly exported through a thin adapter module and the
    package API's ``__all__``. Its public methods are therefore callable by a
    consumer outside this repository. An unexported sibling remains subject
    to D-OB-9 even when a test invokes it.
    """
    src_dir = tmp_path / "agent_utilities"
    api_dir = src_dir / "api"
    api_dir.mkdir(parents=True)
    (api_dir / "__init__.py").write_text(
        "from .ports import CatalogPort\n__all__ = ['CatalogPort']\n"
    )
    (api_dir / "ports.py").write_text(
        "from ._implementation import CatalogPort\n__all__ = ['CatalogPort']\n"
    )
    (api_dir / "_implementation.py").write_text(
        "class CatalogPort:\n"
        "    def read_current_records(self):\n"
        "        return ()\n\n"
        "class UnexportedHelper:\n"
        "    def unused_helper_operation(self):\n"
        "        return None\n"
    )
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_public_port.py").write_text(
        "from agent_utilities.api import CatalogPort\n"
        "from agent_utilities.api._implementation import UnexportedHelper\n\n"
        "def test_ports():\n"
        "    CatalogPort().read_current_records()\n"
        "    UnexportedHelper().unused_helper_operation()\n"
    )

    findings = check_wiring.find_test_only_symbols(
        src_dir=src_dir, tests_dir=tests_dir, display_root=tmp_path
    )
    symbols = {f["symbol"] for f in findings}
    assert "CatalogPort.read_current_records" not in symbols
    assert "UnexportedHelper.unused_helper_operation" in symbols


def test_symbol_gate_does_not_cross_attribute_a_same_named_method_in_another_file(
    tmp_path,
):
    """D-OP-12: two UNRELATED classes in two DIFFERENT files each define a
    method with the same bare name (``parent_of``). Only one of them
    (``TenantRegistry.parent_of``) is ever actually called, and only from a
    test — the genuine D-OB-9 shape, which must be flagged. The other
    (``Lineage.parent_of``) has ZERO references anywhere, in production OR
    tests, and must NOT be flagged merely because a same-named method in an
    entirely different module gained a test caller.

    Reproduces the real defect by construction: before caller resolution
    was scoped to the defining file's own module, the bare-name pooled
    counters (``total_au_calls``/``total_test_calls``) meant a test calling
    ``TenantRegistry().parent_of()`` bumped ``test_refs`` for BOTH methods
    (they share the token ``parent_of``), producing a spurious "NEW
    finding" on ``Lineage.parent_of`` in a file this fixture's test file
    never imports and never opens."""
    src_dir = tmp_path / "agent_utilities"
    src_dir.mkdir()
    (src_dir / "tenant_registry.py").write_text(
        "class TenantRegistry:\n"
        "    def parent_of(self, tenant_id):\n"
        "        return tenant_id\n"
    )
    (src_dir / "concept_lineage.py").write_text(
        "class Lineage:\n"
        "    def parent_of(self, concept_id):\n"
        "        return concept_id\n"
    )
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_tenant_registry.py").write_text(
        "from agent_utilities.tenant_registry import TenantRegistry\n\n"
        "def test_x():\n"
        "    assert TenantRegistry().parent_of('t1') == 't1'\n"
    )

    findings = check_wiring.find_test_only_symbols(
        src_dir=src_dir, tests_dir=tests_dir, display_root=tmp_path
    )
    symbols = {f["symbol"] for f in findings}
    assert "TenantRegistry.parent_of" in symbols
    assert "Lineage.parent_of" not in symbols


def test_symbol_gate_scoped_resolution_still_trips_when_the_collision_partner_has_no_test_either(
    tmp_path,
):
    """Companion to the fixture above: if BOTH same-named methods are
    genuinely test-only (each reached only from a test file that imports
    its own module), scoped resolution must still flag both — proving the
    fix narrows false attribution without narrowing genuine detection."""
    src_dir = tmp_path / "agent_utilities"
    src_dir.mkdir()
    (src_dir / "tenant_registry2.py").write_text(
        "class TenantRegistry2:\n"
        "    def parent_of(self, tenant_id):\n"
        "        return tenant_id\n"
    )
    (src_dir / "concept_lineage2.py").write_text(
        "class Lineage2:\n"
        "    def parent_of(self, concept_id):\n"
        "        return concept_id\n"
    )
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_tenant_registry2.py").write_text(
        "from agent_utilities.tenant_registry2 import TenantRegistry2\n\n"
        "def test_x():\n"
        "    assert TenantRegistry2().parent_of('t1') == 't1'\n"
    )
    (tests_dir / "test_concept_lineage2.py").write_text(
        "from agent_utilities.concept_lineage2 import Lineage2\n\n"
        "def test_y():\n"
        "    assert Lineage2().parent_of('c1') == 'c1'\n"
    )

    findings = check_wiring.find_test_only_symbols(
        src_dir=src_dir, tests_dir=tests_dir, display_root=tmp_path
    )
    symbols = {f["symbol"] for f in findings}
    assert "TenantRegistry2.parent_of" in symbols
    assert "Lineage2.parent_of" in symbols


def _init_differential_repo(root: Path) -> None:
    source = root / "agent_utilities"
    tests = root / "tests"
    source.mkdir(parents=True)
    tests.mkdir()
    (source / "__init__.py").write_text("")
    (source / "old.py").write_text("def dormant():\n    return 1\n")
    (tests / "test_old.py").write_text(
        "from agent_utilities.old import dormant\n\n"
        "def test_old():\n    assert dormant() == 1\n"
    )
    _commit_differential_fixture(root)


def _differential_findings(root: Path) -> list[dict]:
    current = check_wiring.find_test_only_symbols(
        src_dir=root / "agent_utilities",
        tests_dir=root / "tests",
        display_root=root,
    )
    findings = check_wiring._new_symbol_findings_vs_head(str(root), current)
    assert findings is not None
    return findings


def test_differential_gate_preserves_findings_across_a_git_rename(tmp_path):
    """An exact move keeps the prior finding identity; path motion is not debt."""
    _init_differential_repo(tmp_path)
    subprocess.run(
        ["git", "mv", "agent_utilities/old.py", "agent_utilities/new.py"],
        cwd=tmp_path,
        check=True,
    )
    test_path = tmp_path / "tests" / "test_old.py"
    test_path.write_text(test_path.read_text().replace(".old import", ".new import"))
    subprocess.run(["git", "add", "--", str(test_path)], cwd=tmp_path, check=True)

    assert _differential_findings(tmp_path) == []


def test_differential_gate_still_finds_a_new_unwired_symbol(tmp_path):
    """Rename awareness must not hide a genuinely added test-only function."""
    _init_differential_repo(tmp_path)
    source = tmp_path / "agent_utilities" / "old.py"
    source.write_text(source.read_text() + "\ndef newly_unwired():\n    return 2\n")
    test_path = tmp_path / "tests" / "test_old.py"
    test_path.write_text(
        test_path.read_text().replace("import dormant", "import dormant, newly_unwired")
        + "\ndef test_new():\n    assert newly_unwired() == 2\n"
    )
    subprocess.run(
        ["git", "add", "--", str(source), str(test_path)], cwd=tmp_path, check=True
    )

    assert {entry["symbol"] for entry in _differential_findings(tmp_path)} == {
        "newly_unwired"
    }


def test_differential_gate_does_not_treat_a_cross_file_copy_as_a_rename(tmp_path):
    """Keeping the original and copying its unwired API creates new backlog."""
    _init_differential_repo(tmp_path)
    old_source = tmp_path / "agent_utilities" / "old.py"
    new_source = tmp_path / "agent_utilities" / "new.py"
    new_source.write_text(old_source.read_text())
    test_path = tmp_path / "tests" / "test_old.py"
    test_path.write_text(test_path.read_text().replace(".old import", ".new import"))
    subprocess.run(
        ["git", "add", "--", str(new_source), str(test_path)],
        cwd=tmp_path,
        check=True,
    )

    renames = check_wiring._current_to_head_renames(str(tmp_path))
    assert renames is not None
    assert "agent_utilities/new.py" not in renames
    head = {"file": "agent_utilities/old.py", "symbol": "dormant", "ordinal": 0}
    copied = {"file": "agent_utilities/new.py", "symbol": "dormant", "ordinal": 0}
    assert check_wiring._finding_key_at_head(
        copied, renames
    ) != check_wiring._finding_key(head)


def _commit_differential_fixture(root: Path) -> None:
    for args in (
        ("init", "-q"),
        ("config", "user.name", "Wire First Test"),
        ("config", "user.email", "wire-first@example.invalid"),
        ("add", "--", "agent_utilities", "tests"),
        ("commit", "-qm", "baseline"),
    ):
        subprocess.run(["git", *args], cwd=root, check=True)


def _set_ambient_git_repository_environment(monkeypatch) -> None:
    """Install real-repository git variables for bare-snapshot regressions."""
    real_git_dir = subprocess.run(
        ["git", "rev-parse", "--absolute-git-dir"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    monkeypatch.setenv("GIT_DIR", real_git_dir)
    monkeypatch.setenv("GIT_INDEX_FILE", f"{real_git_dir}/index")


def test_differential_gate_accounts_for_test_only_method_unmasked_by_removal(
    tmp_path,
):
    """Deleting an unrelated same-name caller must not manufacture new debt."""
    source = tmp_path / "agent_utilities"
    tests = tmp_path / "tests"
    source.mkdir()
    tests.mkdir()
    (source / "__init__.py").write_text("")
    (source / "target.py").write_text(
        "class Target:\n    def retire(self):\n        return True\n"
    )
    (source / "legacy.py").write_text(
        "class Legacy:\n    def retire(self):\n        return True\n"
    )
    (source / "service.py").write_text(
        "from agent_utilities.legacy import Legacy\n\n"
        "def run():\n    return Legacy().retire()\n"
    )
    (tests / "test_target.py").write_text(
        "from agent_utilities.target import Target\n\n"
        "def test_target():\n    assert Target().retire()\n"
    )
    _commit_differential_fixture(tmp_path)
    (source / "legacy.py").unlink()
    (source / "service.py").unlink()
    subprocess.run(
        ["git", "add", "-u", "--", "agent_utilities"], cwd=tmp_path, check=True
    )
    assert _differential_findings(tmp_path) == []


def test_differential_gate_still_finds_method_whose_real_caller_was_removed(
    tmp_path,
):
    """A removed caller importing the surviving definition remains new debt."""
    source = tmp_path / "agent_utilities"
    tests = tmp_path / "tests"
    source.mkdir()
    tests.mkdir()
    (source / "__init__.py").write_text("")
    (source / "target.py").write_text(
        "class Target:\n    def retire(self):\n        return True\n"
    )
    (source / "service.py").write_text(
        "from agent_utilities.target import Target\n\n"
        "def run():\n    return Target().retire()\n"
    )
    (tests / "test_target.py").write_text(
        "from agent_utilities.target import Target\n\n"
        "def test_target():\n    assert Target().retire()\n"
    )
    _commit_differential_fixture(tmp_path)
    (source / "service.py").unlink()
    subprocess.run(
        ["git", "add", "-u", "--", "agent_utilities"], cwd=tmp_path, check=True
    )

    assert {entry["symbol"] for entry in _differential_findings(tmp_path)} == {
        "Target",
        "Target.retire",
    }


def test_differential_gate_accounts_for_function_unmasked_by_removed_attribute(
    tmp_path,
):
    """An unrelated attribute token must not mask old function debt forever."""
    source = tmp_path / "agent_utilities"
    tests = tmp_path / "tests"
    source.mkdir()
    tests.mkdir()
    (source / "__init__.py").write_text("")
    (source / "target.py").write_text("def comparison():\n    return True\n")
    (source / "legacy.py").write_text(
        "class Legacy:\n    comparison: str\n\n"
        "def read(value):\n    return value.comparison\n"
    )
    (tests / "test_target.py").write_text(
        "from agent_utilities.target import comparison\n\n"
        "def test_target():\n    assert comparison()\n"
    )
    _commit_differential_fixture(tmp_path)
    (source / "legacy.py").unlink()
    subprocess.run(
        ["git", "add", "-u", "--", "agent_utilities"], cwd=tmp_path, check=True
    )

    assert _differential_findings(tmp_path) == []


# ---------------------------------------------------------------------------
# Regression lock — the real repo must stay green with nothing new since HEAD
# ---------------------------------------------------------------------------


def test_gate_passes_on_the_real_repo():
    """The combined report must exit 0 against the real repo with a clean
    tree — the regression lock proving the census + absolute-zero +
    diff-scoped mechanics work end-to-end, not just on synthetic fixtures
    (mirrors ``test_swallowed_errors_gate.py``'s equivalent test). This is
    NOT a claim the repo has no test-only symbols — it has 1107, and the
    gate prints all of them every run. It is a claim that leaving them
    alone does not fail."""
    import subprocess

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--wire-first-report"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# ---------------------------------------------------------------------------
# Extraction-invariance — WD4-RAT-02's key question for this gate
# ---------------------------------------------------------------------------


def _symbol_keys_for(root, src_text: str, test_text: str) -> set[str]:
    root.mkdir(parents=True)
    src_dir = root / "agent_utilities"
    src_dir.mkdir()
    (src_dir / "foo.py").write_text(src_text)
    tests_dir = root / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_foo.py").write_text(test_text)
    findings = check_wiring.find_test_only_symbols(
        src_dir=src_dir, tests_dir=tests_dir, display_root=root
    )
    return {check_wiring._finding_key(e) for e in findings}


_BAR_TEST = (
    "from agent_utilities.foo import Foo\n\n"
    "def test_x():\n"
    "    assert Foo().bar() == 1\n"
)


def test_symbol_key_survives_extraction_into_a_new_nested_helper(tmp_path):
    """The regression this gate's replacement was measured against, not
    assumed. The retired swallowed-error baseline keyed on the ENCLOSING
    SYMBOL, so a handler moved by extraction from ``create_agent`` into
    ``create_agent._setup_mcp_url_toolset`` re-keyed as brand-new debt (37
    phantom findings, D-SWG-1). This gate's key is ``(file, symbol,
    ordinal)`` where ``symbol`` for a method IS ``Class.method`` — the
    question is whether the SAME complexity-collapse technique (moving a
    flagged method's body into a NEW NESTED PRIVATE helper defined inside
    it) perturbs it. It must not: nested defs are invisible to
    ``_public_top_level_defs``/``_public_methods`` in the first place, so
    ``Foo.bar`` itself is untouched by the refactor."""
    before = _symbol_keys_for(
        tmp_path / "before",
        "class Foo:\n    def bar(self):\n        return 1\n",
        _BAR_TEST,
    )
    after = _symbol_keys_for(
        tmp_path / "after",
        "class Foo:\n"
        "    def bar(self):\n"
        "        def _bar_compute():\n"
        "            return 1\n"
        "        return _bar_compute()\n",
        _BAR_TEST,
    )
    assert before == after, (
        f"extraction into a new nested private helper must not manufacture "
        f"a phantom finding: {before} vs {after}"
    )


def test_symbol_key_does_change_on_a_genuine_enclosing_class_rename(tmp_path):
    """The other half, proving the invariance test above is not vacuous: a
    GENUINE rename of the enclosing class (a different, much rarer operation
    than private-helper extraction, and not the technique the
    complexity-collapse program's automation applies) DOES change the key —
    documented and expected, the same residual D-SWG-1-class instability the
    module docstring names, not silently swept under the rug."""
    before = _symbol_keys_for(
        tmp_path / "before",
        "class Foo:\n    def bar(self):\n        return 1\n",
        _BAR_TEST,
    )
    after = _symbol_keys_for(
        tmp_path / "after",
        "class FooRenamed:\n    def bar(self):\n        return 1\n",
        (
            "from agent_utilities.foo import FooRenamed\n\n"
            "def test_x():\n"
            "    assert FooRenamed().bar() == 1\n"
        ),
    )
    assert before != after, "a genuine class rename is expected to re-key"


# ---------------------------------------------------------------------------
# Retired flag
# ---------------------------------------------------------------------------


def test_update_wire_first_baseline_flag_is_retired():
    """The retired flag must REFUSE, not silently do nothing — the same
    convention the liveness/complexity/swallowed-error gates adopted when
    their baselines were removed."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--update-wire-first-baseline"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "RETIRED" in result.stderr


# ---------------------------------------------------------------------------
# GIT_DIR/GIT_INDEX_FILE ambient-env hazard (found by WD4-RAT-02's own plant
# proof under exported GIT_DIR/GIT_INDEX_FILE, not assumed)
# ---------------------------------------------------------------------------


def test_snapshot_symbol_scan_survives_ambient_git_dir_env(tmp_path, monkeypatch):
    """git sets GIT_DIR/GIT_INDEX_FILE/GIT_WORK_TREE in every hook
    subprocess (BUG-043/BUG-180). Left in the environment while scanning a
    bare ``git archive`` extraction (not a git repo itself),
    ``_tracked_or_walked``'s git-ls-files preference silently resolves
    against the AMBIENT (real) repository instead of erroring "not a git
    repository", and ``_scan_snapshot_for_test_only_symbols`` used to trust
    that wrong non-empty result — manufacturing a near-empty au_sources/
    tests set and reading EVERY current finding as new. Reproduced directly
    against the fixed function, without paying for a full repo scan: a real
    ambient GIT_DIR/GIT_INDEX_FILE (this very repo's own) must not blank out
    a scan of an unrelated synthetic snapshot directory. Found by running
    this exact scenario end to end (plant present, exported GIT_DIR/
    GIT_INDEX_FILE): 1 genuinely new finding read as 1108 (the entire
    backlog) before the fix in ``_scan_snapshot_for_test_only_symbols``."""
    snapshot = tmp_path / "snapshot"
    src_dir = snapshot / "agent_utilities"
    src_dir.mkdir(parents=True)
    (src_dir / "foo.py").write_text(
        "class Foo:\n    def bar(self):\n        return 1\n"
    )
    tests_dir = snapshot / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_foo.py").write_text(_BAR_TEST)

    _set_ambient_git_repository_environment(monkeypatch)

    findings = check_wiring._scan_snapshot_for_test_only_symbols(snapshot)
    symbols = {f["symbol"] for f in findings}
    assert "Foo.bar" in symbols, (
        "ambient GIT_DIR/GIT_INDEX_FILE must not blank out the snapshot scan"
    )


def test_snapshot_unmasking_context_survives_ambient_git_dir_env(tmp_path, monkeypatch):
    """Deletion accounting must read the same bare snapshot safely."""
    snapshot = tmp_path / "snapshot"
    src_dir = snapshot / "agent_utilities"
    src_dir.mkdir(parents=True)
    (src_dir / "target.py").write_text("def comparison():\n    return True\n")
    tests_dir = snapshot / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_target.py").write_text(
        "import agent_utilities.target as target\n\n"
        "def test_target():\n    assert target.comparison()\n"
    )

    _set_ambient_git_repository_environment(monkeypatch)

    sources, (idents, calls, imports, _, _) = check_wiring._read_head_snapshot_context(
        snapshot
    )

    assert "agent_utilities/target.py" in sources
    assert idents["tests/test_target.py"]["comparison"] > 0
    assert calls["tests/test_target.py"]["comparison"] > 0
    assert "agent_utilities.target" in imports["tests/test_target.py"]
