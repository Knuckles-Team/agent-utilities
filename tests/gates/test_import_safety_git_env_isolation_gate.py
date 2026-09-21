"""NE-044 sub-gate 3 acceptance (import safety, ``723f85ed``): does
``scripts/check_import_safety.py`` run "from the exact tree with NO ambient
Git-state dependency"?

``723f85ed`` fixed a DIFFERENT class of environment dependency for this gate
-- which INTERPRETER runs it (bare ``python3`` on PATH vs the repo's own
``.venv``), wired via ``.pre-commit-config.yaml``'s ``uv_workspace.py run``
wrapper, not a git-state dependency. This file proves the git-state half of
the "no ambient dependency" requirement holds too: unlike
``check_tracked_privacy.py``/``check_current_only_contract.py``
(``ee3814af``/``aad9ab52``, BUG-180), this gate never shells out to ``git``
at all -- it walks the ALREADY-IMPORTED package via ``pkgutil``/
``importlib``, so it has no ``GIT_DIR``/``GIT_INDEX_FILE`` exposure to begin
with. Both halves are checked here: a structural proof the script contains
no git subprocess call, and the empirical proof (spawned as a real
subprocess, exactly the pre-commit hook shape) that its verdict against the
real tree does not change under a poisoned ``GIT_DIR``/``GIT_INDEX_FILE``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE = _REPO_ROOT / "scripts" / "check_import_safety.py"

# The exact --exclude list .pre-commit-config.yaml's check-import-safety
# hook uses (minus --simulate-windows, which only changes WHICH modules are
# blocked, not the git-env-isolation question this file is scoped to) --
# kept in sync manually; a drift here would only make this test's own
# baseline noisier, not affect the git-env-isolation proof.
_EXCLUDES = [
    "agent_utilities.knowledge_graph.core.file_lock",
    "agent_utilities.__main__",
    "agent_utilities.agent.factory",
    "agent_utilities.server",
    "agent_utilities.mcp.toolset_factory",
    "agent_utilities.mcp.tools",
    "agent_utilities.mcp.verbose_tools",
    "agent_utilities.patterns",
    "agent_utilities.knowledge_graph.adaptation.trace_distiller",
    "agent_utilities.cli",
    "agent_utilities.core.unified_install",
]


def test_gate_source_never_shells_out_to_git() -> None:
    """Structural proof this gate is immune BY CONSTRUCTION, not by
    accident: it never imports ``subprocess`` and never invokes ``git`` --
    unlike its sibling gates (``check_tracked_privacy.py``/
    ``check_current_only_contract.py``), which had to be fixed to strip
    ``GIT_DIR``/``GIT_INDEX_FILE`` because they genuinely do shell out."""
    source = _MODULE.read_text(encoding="utf-8")
    assert "import subprocess" not in source
    assert "subprocess" not in source
    assert '"git"' not in source and "'git'" not in source


def _clean_env() -> dict[str, str]:
    env = dict(os.environ)
    for name in (
        "GIT_DIR",
        "GIT_INDEX_FILE",
        "GIT_WORK_TREE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_CEILING_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_NAMESPACE",
    ):
        env.pop(name, None)
    return env


def _poisoned_env(decoy_git_dir: Path) -> dict[str, str]:
    env = _clean_env()
    env["GIT_DIR"] = str(decoy_git_dir)
    env["GIT_INDEX_FILE"] = str(decoy_git_dir / "index")
    return env


def _run_gate(env: dict[str, str]) -> subprocess.CompletedProcess:
    args = [sys.executable, str(_MODULE), "--package", "agent_utilities"]
    for prefix in _EXCLUDES:
        args += ["--exclude", prefix]
    return subprocess.run(
        args,
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_gate_verdict_against_the_real_tree_is_identical_plain_vs_ambient_git_env(
    tmp_path: Path,
) -> None:
    """The empirical proof: run the REAL gate against the REAL tree twice --
    once plain, once with GIT_DIR/GIT_INDEX_FILE pointed at an unrelated
    decoy repository -- and assert the verdict is byte-for-byte identical."""
    decoy = tmp_path / "decoy-repo"
    decoy.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=decoy, check=True)

    plain = _run_gate(_clean_env())
    ambient = _run_gate(_poisoned_env(decoy / ".git"))

    assert plain.stdout == ambient.stdout
    assert plain.stderr == ambient.stderr
    assert plain.returncode == ambient.returncode
    assert "import-safety[native]: checked" in plain.stdout
