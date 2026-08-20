"""NE-044 sub-gate 3 acceptance (symlink gate, ``ebb7820b``, BUG-233): does
``scripts/check_skill_name_collision.py`` -- the gate that walks the
installable skill tree and symlink-classifies every symlinked directory it
finds (``_classify_symlinked_dir``) -- run "from the exact tree with NO
ambient Git-state dependency"?

Structurally this gate never shells out to ``git`` at all (it walks the
filesystem directly via ``os.walk``/``os.lstat``, matching against
``agents/``/``skills/``/``agent-utilities`` tree anchors), so it has no
``GIT_DIR``/``GIT_INDEX_FILE`` exposure to begin with -- proven here both
structurally and empirically (real subprocess, plain vs. poisoned
environment, against a synthetic fixture that exercises the ACTUAL
mechanism BUG-233 was about: a genuine skill-name collision plus a benign
in-tree symlink the walk must prune without double-counting).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE = _REPO_ROOT / "scripts" / "check_skill_name_collision.py"


def test_gate_source_never_shells_out_to_git() -> None:
    """Structural proof: this gate is immune by construction, not by
    accident -- it never imports ``subprocess`` at all."""
    source = _MODULE.read_text(encoding="utf-8")
    assert "import subprocess" not in source
    assert "subprocess" not in source


def _build_fleet_fixture(root: Path) -> None:
    """A minimal ``agents/``+``skills/`` fleet root with a REAL skill-name
    collision (two packages both declaring ``name: dup-skill``) and a
    benign in-tree symlink (BUG-233's own subject: a symlinked directory
    that resolves back inside the tracked ``agents/`` tree and must be
    pruned, not double-scanned)."""
    (root / "skills").mkdir(parents=True)  # required for _find_fleet_root

    pkg_a_skill = root / "agents" / "pkg-a" / "mod_a" / "skills" / "skill-a"
    pkg_a_skill.mkdir(parents=True)
    (pkg_a_skill / "SKILL.md").write_text(
        "---\nname: dup-skill\n---\nDoes a thing.\n", encoding="utf-8"
    )

    pkg_b_skill = root / "agents" / "pkg-b" / "mod_b" / "skills" / "skill-b"
    pkg_b_skill.mkdir(parents=True)
    (pkg_b_skill / "SKILL.md").write_text(
        "---\nname: dup-skill\n---\nDoes the same thing, differently.\n",
        encoding="utf-8",
    )

    # A benign symlink inside the tracked "agents" tree -- resolves back
    # inside root/agents, so _classify_symlinked_dir must prune it (not
    # fail closed, not double-count skill-a under a second path).
    link = root / "agents" / "pkg-a" / "mod_a" / "skills" / "linked-skill-a"
    link.symlink_to(pkg_a_skill, target_is_directory=True)


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


def _run_gate(*, fleet_root: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_MODULE), "--root", str(fleet_root), "--strict"],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_gate_verdict_against_a_real_collision_fixture_is_identical_plain_vs_ambient_git_env(
    tmp_path: Path,
) -> None:
    """The empirical proof against the gate's own real mechanism: a genuine
    skill-name collision plus a benign in-tree symlink must be reported
    identically whether the process's environment is clean or carries a
    poisoned GIT_DIR/GIT_INDEX_FILE pointed at an unrelated decoy repo."""
    fleet_root = tmp_path / "fleet"
    _build_fleet_fixture(fleet_root)

    decoy = tmp_path / "decoy-repo"
    decoy.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=decoy, check=True)

    plain = _run_gate(fleet_root=fleet_root, env=_clean_env())
    ambient = _run_gate(fleet_root=fleet_root, env=_poisoned_env(decoy / ".git"))

    assert plain.stdout == ambient.stdout
    assert plain.stderr == ambient.stderr
    assert plain.returncode == ambient.returncode

    # Not a trivially-empty comparison: the real collision must actually be
    # caught (--strict fails on ANY collision), and the benign symlink must
    # actually be pruned (reported, not treated as a failure).
    assert plain.returncode == 1
    assert "dup-skill" in plain.stdout
    assert "linked-skill-a" in plain.stdout
    assert "pruned, not scanned twice" in plain.stdout


def test_gate_verdict_against_the_real_tree_is_identical_plain_vs_ambient_git_env(
    tmp_path: Path,
) -> None:
    """Companion proof against THIS worktree's own real tree (no fleet root
    reachable from an isolated worktree, so the gate takes its early-exit
    "could not locate fleet root" path) -- identical either way, and not a
    crash."""
    decoy = tmp_path / "decoy-repo"
    decoy.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=decoy, check=True)

    def _run(env: dict[str, str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(_MODULE)],
            cwd=_REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

    plain = _run(_clean_env())
    ambient = _run(_poisoned_env(decoy / ".git"))

    assert plain.stdout == ambient.stdout
    assert plain.stderr == ambient.stderr
    assert plain.returncode == ambient.returncode
