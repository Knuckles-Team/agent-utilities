#!/usr/bin/env python3
"""Repository-root hygiene gate.

The repo root is the first thing a reader of a public GitHub project sees, and
it is where scratch files accumulate fastest: a one-off proof artifact from a
CI experiment, a scratch note, a directory that belongs to the workspace rather
than to this package. Two such files (``runner_proof2.md``,
``reports_runner_proof.md`` -- both of which literally said "safe to delete" in
their own body) survived long enough to be published, and a stray ``plans/``
directory carried workspace-level program content into a package repo.

This gate enforces an **allowlist**, not a denylist. A denylist only catches the
junk somebody already thought of; an allowlist means a genuinely new root entry
has to be justified once, deliberately, by someone editing this file.

It reads the **tracked** file set (``git ls-files``), never the filesystem --
walking the filesystem makes a gate fire on build output and gitignored
artifacts, which is how gates earn a reputation for crying wolf and get
disabled (see BUG-043, where exactly that happened to the workspace_helpers
chokepoint gate).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Directories legitimately at the repo root. Each is either shipped, built, or
# read by tooling -- not a scratch space.
ALLOWED_DIRS: frozenset[str] = frozenset(
    {
        ".github",
        ".security",
        ".specify",
        "agent_utilities",  # the package itself
        "deploy",
        "docker",
        "docs",
        "examples",
        # A SEPARATE deployable with its own pyproject.toml and Dockerfile
        # (docker/mcp-v2-gateway.Dockerfile), running as its own container
        # alongside graph-os. It is deliberately NOT under agent_utilities/:
        # that package is the importable library, this is a distinct
        # distribution that must not be pulled into the library's wheel.
        "mcp_v2_gateway",
        "scripts",
        "tests",
    }
)

# Files legitimately at the repo root.
ALLOWED_FILES: frozenset[str] = frozenset(
    {
        # Documentation / project metadata
        "AGENTS.head.md",
        "AGENTS.md",
        "CHANGELOG.md",
        "CLAUDE.md",
        "CONTRIBUTING.md",
        "LICENSE",
        "README.md",
        "llms.txt",
        # Packaging / build
        "MANIFEST.in",
        "build_backend.py",
        "pyproject.toml",
        "requirements.txt",
        "uv.lock",
        # Image build: mirrors the workspace root's [tool.uv]
        # override-dependencies, which uv honours ONLY from a workspace root --
        # an image build resolves from this package alone. Consumed by
        # docker/Dockerfile via UV_OVERRIDE.
        "overrides.txt",
        # Tooling config
        "genesis.yaml",
        "mcp_config.bus.json",
        "mcp_config.example.json",
        "mkdocs.yml",
        "opencode.json",
        "pytest.ini",
        "vulture_whitelist.py",  # referenced by .pre-commit-config.yaml
    }
)

# Anything starting with "." at the root is dotfile config (.gitignore,
# .pre-commit-config.yaml, .bumpversion.cfg, ...). Those are conventional and
# self-describing, so they are allowed as a class rather than enumerated.


def tracked_root_entries() -> tuple[set[str], set[str]]:
    """Return (root_files, root_dirs) from the tracked file set.

    Resolves the repo root from this script's own location rather than trusting
    the caller's cwd: a hook can be invoked from anywhere, and ``git ls-files``
    run outside a work tree exits 128 and would surface as a stack trace rather
    than a usable message.
    """
    repo_root = Path(__file__).resolve().parent.parent
    out = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "-z"],
        capture_output=True,
        check=True,
        text=True,
    ).stdout
    files: set[str] = set()
    dirs: set[str] = set()
    for path in out.split("\0"):
        if not path:
            continue
        head, sep, _ = path.partition("/")
        if sep:
            dirs.add(head)
        else:
            files.add(head)
    return files, dirs


def main() -> int:
    files, dirs = tracked_root_entries()

    stray_files = sorted(f for f in files if not f.startswith(".") and f not in ALLOWED_FILES)
    stray_dirs = sorted(d for d in dirs if not d.startswith(".") and d not in ALLOWED_DIRS)

    if not stray_files and not stray_dirs:
        print(f"root hygiene: clean ({len(files)} root files, {len(dirs)} root dirs)")
        return 0

    print("FAIL: unexpected entries at the repository root.\n")
    for d in stray_dirs:
        print(f"  DIR   {d}/")
    for f in stray_files:
        print(f"  FILE  {f}")

    print(
        "\nPick the one that is true:\n"
        "  * it is scratch/proof output   -> delete it (it should never have been committed)\n"
        "  * it belongs to the workspace  -> move it to ${WORKSPACE_ROOT}/, not this package\n"
        "  * it belongs inside a package  -> move it under agent_utilities/ or scripts/\n"
        "  * it genuinely belongs at root -> add it to ALLOWED_FILES/ALLOWED_DIRS in\n"
        "    scripts/check_root_hygiene.py WITH a comment saying what reads it\n"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
