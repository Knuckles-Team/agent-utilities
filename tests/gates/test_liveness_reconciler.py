"""Meta-test: ``scripts/liveness_reconciler.py`` rescues a module/definition
genuinely reachable through each of the four dynamic-reference mechanisms it
was taught, and does NOT rescue a genuinely dead one. "A gate that can't fail
is not a gate" (see ``test_swallowed_errors_gate.py``, the template for this
file) — the failure mode this program has repeatedly found is a detector that
looks like it works but catches nothing (or, the opposite and more dangerous
direction here, a detector that starts rescuing everything and catches
nothing real).

Fixtures are synthetic (a throwaway ``agent_utilities``-shaped tree under
``tmp_path``), not the real repo, so these tests do not depend on the current
state of any real module and cannot be defeated by a future refactor of the
specific modules named in the design doc.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "liveness_reconciler.py"


def _load_reconciler():
    spec = importlib.util.spec_from_file_location(
        "_liveness_reconciler_under_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _build_fixture_repo(tmp_path: Path, *, extra_files: dict[str, str]) -> Path:
    """A minimal ``agent_utilities``-shaped repo root the reconciler's module
    functions can walk, with the reconciler's own ``REPO``/``SRC_DIR``/
    ``TESTS_DIR``/``PYPROJECT`` module attributes repointed at it (so the
    real repo's own thousands of files are never scanned by these tests)."""
    src = tmp_path / "agent_utilities"
    tests_dir = tmp_path / "tests"
    src.mkdir()
    tests_dir.mkdir()
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "fixture"\n')
    for rel, content in extra_files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return tmp_path


def _reconciler_for(tmp_path: Path, extra_files: dict[str, str]):
    lr = _load_reconciler()
    repo = _build_fixture_repo(tmp_path, extra_files=extra_files)
    lr.REPO = repo
    lr.SRC_DIR = repo / "agent_utilities"
    lr.TESTS_DIR = repo / "tests"
    lr.PYPROJECT = repo / "pyproject.toml"
    lr.check_wiring.ROOT = repo
    lr.check_wiring.SRC_DIR = lr.SRC_DIR
    lr.check_wiring.PYPROJECT = lr.PYPROJECT
    return lr


# ---------------------------------------------------------------------------
# True positive: a genuinely dead module/definition stays flagged.
# ---------------------------------------------------------------------------


def test_genuinely_dead_module_is_not_rescued(tmp_path):
    lr = _reconciler_for(
        tmp_path,
        extra_files={
            "agent_utilities/orphaned/__init__.py": "",
            "agent_utilities/orphaned/dead_mod.py": (
                "def never_called():\n    return 1\n"
            ),
            # Deliberately does NOT import `dead_mod` — a sibling module that
            # imports something ELSE, so the fixture package is non-trivial
            # without accidentally creating a real import edge to the module
            # under test (`from . import dead_mod` would itself be a genuine
            # import — see check_wiring.collect_imports's submodule-alias rule).
            "agent_utilities/orphaned/live_mod.py": "VALUE = 1\n",
        },
    )
    # `live_mod.py` imports `dead_mod` via a relative import — but nobody
    # imports `dead_mod` from OUTSIDE the package by its full dotted path,
    # and it has no entry-point/getattr/lazy-import reference either.
    details = {
        "orphan_modules": ["orphaned.dead_mod"],
        "dead_definitions": ["orphaned.dead_mod:never_called"],
    }
    recon = lr.reconcile(details)
    assert "orphaned.dead_mod" in recon["orphan_modules"]["still"]
    assert not recon["orphan_modules"]["rescued"]
    assert "orphaned.dead_mod:never_called" in recon["dead_definitions"]["still"]
    assert not recon["dead_definitions"]["rescued"]


# ---------------------------------------------------------------------------
# True negative #1: corrected static import resolution (the core bug fix).
# ---------------------------------------------------------------------------


def test_dotted_path_import_rescues_orphan_module(tmp_path):
    lr = _reconciler_for(
        tmp_path,
        extra_files={
            "agent_utilities/pkg/__init__.py": "",
            "agent_utilities/pkg/target.py": "VALUE = 1\n",
            "agent_utilities/caller.py": (
                "from agent_utilities.pkg.target import VALUE\n"
            ),
        },
    )
    details = {"orphan_modules": ["pkg.target"], "dead_definitions": []}
    recon = lr.reconcile(details)
    rescued = {r["module"]: r["mechanism"] for r in recon["orphan_modules"]["rescued"]}
    assert rescued.get("pkg.target") == "corrected-static-import-resolution"
    assert "pkg.target" not in recon["orphan_modules"]["still"]


# ---------------------------------------------------------------------------
# True negative #2: pyproject.toml entry-point (any group).
# ---------------------------------------------------------------------------


def test_entry_point_rescues_module_and_symbol_suffix(tmp_path):
    lr = _reconciler_for(
        tmp_path,
        extra_files={
            "agent_utilities/plugins/__init__.py": "",
            "agent_utilities/plugins/example.py": ("class ExamplePlugin:\n    pass\n"),
            "pyproject.toml": (
                '[project]\nname = "fixture"\n\n'
                '[project.entry-points."fixture.plugins"]\n'
                'example = "agent_utilities.plugins.example:ExamplePlugin"\n'
            ),
        },
    )
    details = {
        "orphan_modules": ["plugins.example"],
        "dead_definitions": ["plugins.example:ExamplePlugin"],
    }
    recon = lr.reconcile(details)
    om_rescued = {
        r["module"]: r["mechanism"] for r in recon["orphan_modules"]["rescued"]
    }
    dd_rescued = {
        r["definition"]: r["mechanism"] for r in recon["dead_definitions"]["rescued"]
    }
    assert om_rescued.get("plugins.example") == "pyproject-entry-point"
    assert (
        dd_rescued.get("plugins.example:ExamplePlugin")
        == "pyproject-entry-point-suffix"
    )


# ---------------------------------------------------------------------------
# True negative #3: getattr(module_or_obj, "Name") registry.
# ---------------------------------------------------------------------------


def test_getattr_registry_rescues_dead_definition(tmp_path):
    lr = _reconciler_for(
        tmp_path,
        extra_files={
            "agent_utilities/widgets/__init__.py": "",
            "agent_utilities/widgets/one.py": "class Widget:\n    pass\n",
            "agent_utilities/widgets/registry.py": (
                "import importlib\n\n"
                "def load(name):\n"
                "    module = importlib.import_module(f'agent_utilities.widgets.{name}')\n"
                "    return getattr(module, 'Widget', None)\n"
            ),
        },
    )
    details = {"orphan_modules": [], "dead_definitions": ["widgets.one:Widget"]}
    recon = lr.reconcile(details)
    dd_rescued = {
        r["definition"]: r["mechanism"] for r in recon["dead_definitions"]["rescued"]
    }
    assert dd_rescued.get("widgets.one:Widget") == "getattr-registry"


# ---------------------------------------------------------------------------
# True negative #4: __getattr__ lazy-import map (PEP 562).
# ---------------------------------------------------------------------------


def test_getattr_lazy_import_map_rescues_orphan_module(tmp_path):
    lr = _reconciler_for(
        tmp_path,
        extra_files={
            "agent_utilities/lazy/__init__.py": (
                "import importlib\n\n"
                "_ATTR_MAP = {\n"
                "    'do_thing': '.impl',\n"
                "}\n\n"
                "def __getattr__(name):\n"
                "    if name in _ATTR_MAP:\n"
                "        mod = importlib.import_module(_ATTR_MAP[name], __package__)\n"
                "        return getattr(mod, name)\n"
                "    raise AttributeError(name)\n"
            ),
            "agent_utilities/lazy/impl.py": "def do_thing():\n    return 1\n",
        },
    )
    details = {"orphan_modules": ["lazy.impl"], "dead_definitions": []}
    recon = lr.reconcile(details)
    om_rescued = {
        r["module"]: r["mechanism"] for r in recon["orphan_modules"]["rescued"]
    }
    assert om_rescued.get("lazy.impl") == "__getattr__-lazy-import-map"


# ---------------------------------------------------------------------------
# Fail-open: a genuinely dynamic, unresolvable import site.
# ---------------------------------------------------------------------------


def test_nonliteral_import_target_fails_open(tmp_path):
    lr = _reconciler_for(
        tmp_path,
        extra_files={
            "agent_utilities/discovery/__init__.py": (
                "import importlib\nimport pkgutil\n\n"
                "def load_all():\n"
                "    for m in pkgutil.iter_modules(__path__):\n"
                "        importlib.import_module(f'{__name__}.{m.name}')\n"
            ),
            "agent_utilities/discovery/plugin_a.py": "VALUE = 1\n",
        },
    )
    details = {"orphan_modules": ["discovery.plugin_a"], "dead_definitions": []}
    recon = lr.reconcile(details)
    om_rescued = {
        r["module"]: r["mechanism"] for r in recon["orphan_modules"]["rescued"]
    }
    assert (
        om_rescued.get("discovery.plugin_a") == "unresolvable-dynamic-import-fail-open"
    )
    assert recon["dynamic_unresolvable_sites"]


# ---------------------------------------------------------------------------
# Generated-file exclusion: only the file's OWN docstring counts, never a
# string literal elsewhere in the file that merely MENTIONS the marker text
# (the false positive this task found and fixed on `codebase_map_tools.py`,
# a GENERATOR whose output contains "do not edit... regenerate with").
# ---------------------------------------------------------------------------


def test_generated_docstring_excludes_but_mention_in_body_does_not(tmp_path):
    lr = _reconciler_for(
        tmp_path,
        extra_files={
            "agent_utilities/gen/__init__.py": "",
            "agent_utilities/gen/really_generated.py": (
                '"""Generated client. Regenerate with the codegen tool; do not edit."""\n'
                "class Thing:\n    pass\n"
            ),
            "agent_utilities/gen/generator_tool.py": (
                '"""Writes a markdown file for humans."""\n'
                "def emit():\n"
                "    return 'Do not edit manually — regenerate with emit()'\n"
            ),
        },
    )
    details = {
        "orphan_modules": ["gen.really_generated", "gen.generator_tool"],
        "dead_definitions": ["gen.really_generated:Thing", "gen.generator_tool:emit"],
    }
    recon = lr.reconcile(details)
    assert "gen.really_generated" in recon["orphan_modules"]["excluded_generated"]
    assert "gen.generator_tool" not in recon["orphan_modules"]["excluded_generated"]
    # generator_tool.py's `emit` is unrescued by any mechanism and its file is
    # NOT generated, so it stays a genuine (if synthetic) still-flagged finding.
    assert "gen.generator_tool" in recon["orphan_modules"]["still"]
