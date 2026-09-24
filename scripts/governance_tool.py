"""Make repository-manager's development-governance package importable here.

Operator ruling OQ-3 moved agent-utilities' development governance — lane
arbitration, the OKF-CIS concept grammar and vocabularies, concept lineage and
reservation, and the lane-guard gate — to ``repository_manager.governance``.
This repository's own tooling (its concept gates, its docs generator, the
lane-guard entry point every fleet hook runs, and the test suite's shared
engine-daemon lease) consumes that package as a development tool, never from
``agent_utilities`` itself.

The package is resolved, in order, from:

1. ``$REPOSITORY_MANAGER_ROOT`` — an explicit repository-manager checkout;
2. ``agents/repository-manager`` beside this repository's canonical checkout
   (the ecosystem workspace layout; found through ``--git-common-dir`` so a
   linked worktree anywhere on disk resolves the same sibling);
3. the installed ``repository-manager`` distribution.

A checkout found by (1) or (2) is put first on ``sys.path`` so it wins over an
older installed release that predates the ``governance`` package.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

ROOT_ENV = "REPOSITORY_MANAGER_ROOT"
REPO = Path(__file__).resolve().parents[1]
_SIBLING = Path("agents") / "repository-manager"


class GovernanceUnavailable(RuntimeError):
    """repository-manager's governance package could not be imported."""


def _canonical_checkout(tree: Path) -> Path | None:
    proc = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=str(tree),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    return Path(proc.stdout.strip()).parent


def candidate_roots(tree: Path = REPO) -> list[Path]:
    """Checkouts that may hold ``repository_manager/governance``, in priority order."""
    roots: list[Path] = []
    configured = os.environ.get(ROOT_ENV, "").strip()
    if configured:
        roots.append(Path(configured).expanduser())
    canonical = _canonical_checkout(tree)
    for base in (canonical, tree):
        if base is not None:
            roots.append(base.parent / _SIBLING)
    return roots


def _has_governance(root: Path) -> bool:
    return (root / "repository_manager" / "governance" / "__init__.py").is_file()


def ensure_governance_importable(tree: Path = REPO) -> None:
    """Import ``repository_manager.governance`` or raise :class:`GovernanceUnavailable`."""
    root = next((r for r in candidate_roots(tree) if _has_governance(r)), None)
    if root is not None and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        importlib.import_module("repository_manager.governance")
    except ImportError as exc:
        raise GovernanceUnavailable(
            "repository-manager's development-governance package "
            "(repository_manager.governance) is not importable: set "
            f"{ROOT_ENV} to a repository-manager checkout, keep one at "
            f"<workspace>/{_SIBLING.as_posix()}, or install repository-manager "
            f"(searched {[str(r) for r in candidate_roots(tree)]}): {exc}"
        ) from exc


def governance(module: str) -> ModuleType:
    """``repository_manager.governance.<module>``, resolved as described above."""
    ensure_governance_importable()
    return importlib.import_module(f"repository_manager.governance.{module}")
