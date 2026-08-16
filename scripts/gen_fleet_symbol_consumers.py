#!/usr/bin/env python3
"""Companion generator for ``scripts/fleet_symbol_consumers.json`` — the
committed consumer index ``check_removed_symbol_consumers.py`` (CI, where the
fleet is NOT checked out) reads to know who imports ``agent_utilities``.

Why this exists: two real breaks in one day (2026-08-13) shipped a removed/
renamed public symbol and a newly-eager optional dependency, each into
production consumers, discovered only by accident (an unrelated uv sweep; a
manual blast-radius check). Neither had a control. agent-utilities is the
producer for ~70 fleet consumers (``agents/*`` + ``skills/*`` under
``agent-packages``, per ``workspace.yml``); this script builds the map of
"who imports what from us" that the gate cross-references.

What it scans
--------------
Every ``*.py`` file (including tests — a break in a consumer's *test* import
is still a break) under each repo listed in ``workspace.yml``'s
``subdirectories.agent-packages.subdirectories.{agents,skills}`` trees that
is present **locally** at ``<workspace-root>/agent-packages/{agents,skills}/
<repo>``. A repo in the manifest but not checked out locally is reported and
skipped (not silently ignored) — the index only ever reflects repos this
script could actually read.

Blind spots — say them, don't hide them
----------------------------------------
* **Static AST only.** ``importlib.import_module("agent_utilities....")`` /
  ``__import__`` / any string-composed import is invisible, same class of gap
  as ``check_wiring.py`` documents for the producer side.
* **Local checkout only.** A consumer repo not cloned under this workspace
  (e.g. only on another host) cannot be scanned; it is silently absent from
  the index unless someone runs this generator on a box where it *is*
  checked out. This is the single biggest source of index staleness/under-
  coverage — see ``check_removed_symbol_consumers.py``'s staleness handling.
* **Consumer-side aliasing.** ``from agent_utilities.mcp.server_factory
  import protect_stdio_jsonrpc as p`` is recorded under the ORIGINAL name
  (``protect_stdio_jsonrpc``), which is what matters for the producer-side
  removal check — the alias itself is irrelevant.

Usage::

    python3 scripts/gen_fleet_symbol_consumers.py                # report only
    python3 scripts/gen_fleet_symbol_consumers.py --update        # write the index
    python3 scripts/gen_fleet_symbol_consumers.py --json
    python3 scripts/gen_fleet_symbol_consumers.py --workspace-root /path
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

INDEX_PATH = ROOT / "scripts" / "fleet_symbol_consumers.json"
PACKAGE_NAME = "agent_utilities"

EXCLUDE_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".hypothesis",
    "build",
    "dist",
    ".eggs",
}


@dataclass(frozen=True)
class ConsumerHit:
    repo: str
    file: str  # repo-relative posix path
    line: int
    target: str  # fully-qualified dotted target, e.g. agent_utilities.mcp.server_factory.protect_stdio_jsonrpc


def _repo_name_from_url(url: str) -> str:
    name = str(url).rstrip("/").rsplit("/", 1)[-1]
    return name[:-4] if name.endswith(".git") else name


def _enumerate_fleet_repo_names(manifest: dict) -> dict[str, list[str]]:
    """``{"agents": [...], "skills": [...]}`` repo basenames from workspace.yml.

    Deliberately scoped to the fleet CONSUMER trees only (``agents/`` +
    ``skills/``) — not ``agent-utilities`` itself, and not the sibling
    producer-adjacent repos (``agent-webui``, ``epistemic-graph``, ...) at the
    ``agent-packages`` level, which are a different audience than "the ~70
    fleet consumers" this gate protects. Extend the ``for key in (...)`` tuple
    below if those should ever be covered too.
    """
    ap = (manifest.get("subdirectories", {}) or {}).get("agent-packages", {})
    subdirs = (ap.get("subdirectories", {}) or {}) if isinstance(ap, dict) else {}
    out: dict[str, list[str]] = {}
    for key in ("agents", "skills"):
        node = subdirs.get(key, {}) or {}
        names = []
        for repo in node.get("repositories", []) or []:
            url = repo.get("url") if isinstance(repo, dict) else repo
            if url:
                names.append(_repo_name_from_url(url))
        out[key] = sorted(names)
    return out


def _find_workspace_manifest() -> Path | None:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / "workspace.yml"
        if candidate.is_file():
            return candidate
    return None


def _iter_py_files(repo_dir: Path):
    for path in repo_dir.rglob("*.py"):
        if any(part in EXCLUDE_DIR_NAMES for part in path.parts):
            continue
        yield path


def _scan_file(repo: str, repo_dir: Path, path: Path) -> list[ConsumerHit]:
    rel = path.relative_to(repo_dir).as_posix()
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=rel)
    except (OSError, UnicodeDecodeError, SyntaxError):
        return []
    hits: list[ConsumerHit] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue
            mod = node.module or ""
            if mod != PACKAGE_NAME and not mod.startswith(PACKAGE_NAME + "."):
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                target = f"{mod}.{alias.name}"
                hits.append(ConsumerHit(repo, rel, node.lineno, target))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == PACKAGE_NAME or alias.name.startswith(
                    PACKAGE_NAME + "."
                ):
                    hits.append(ConsumerHit(repo, rel, node.lineno, alias.name))
    return hits


def _repo_head_sha(repo_dir: Path) -> str:
    try:
        res = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    return res.stdout.strip() or "unknown"


def build_index(workspace_root: Path) -> dict:
    manifest_path = _find_workspace_manifest()
    if manifest_path is None:
        manifest_path = workspace_root / "workspace.yml"
    if not manifest_path.is_file():
        raise SystemExit(f"cannot find workspace.yml (looked at {manifest_path})")

    import yaml  # local import: only needed by the generator, not the gate

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    fleet = _enumerate_fleet_repo_names(manifest)

    ap_root = workspace_root / "agent-packages"
    consumers: dict[str, list[dict]] = {}
    repos_meta: dict[str, dict] = {}
    missing_repos: list[str] = []

    for kind, names in fleet.items():
        for name in names:
            repo_dir = ap_root / kind / name
            if not repo_dir.is_dir():
                missing_repos.append(f"{kind}/{name}")
                continue
            repos_meta[name] = {
                "kind": kind,
                "head_sha": _repo_head_sha(repo_dir),
            }
            for path in _iter_py_files(repo_dir):
                for hit in _scan_file(name, repo_dir, path):
                    consumers.setdefault(hit.target, []).append(
                        {"repo": hit.repo, "file": hit.file, "line": hit.line}
                    )

    for target in consumers:
        consumers[target] = sorted(
            {(c["repo"], c["file"], c["line"]) for c in consumers[target]}
        )
        consumers[target] = [
            {"repo": r, "file": f, "line": ln} for (r, f, ln) in consumers[target]
        ]

    manifest_bytes = manifest_path.read_bytes()
    return {
        "_comment": (
            "Fleet consumer index for check_removed_symbol_consumers.py. Maps "
            "a fully-qualified agent_utilities dotted target (module path + "
            "symbol name, OR a bare module path for a whole-module import) to "
            "every agents/*  and skills/* repo+file+line that imports it. "
            "Regenerate with `python3 scripts/gen_fleet_symbol_consumers.py "
            "--update` after any fleet-wide checkout refresh; the gate FAILS "
            "closed if this file is missing or older than its staleness "
            "threshold — see that script's docstring."
        ),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "workspace_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "workspace_root": str(workspace_root),
        "repos_scanned": repos_meta,
        "repos_missing_locally": sorted(missing_repos),
        "consumers": {k: consumers[k] for k in sorted(consumers)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path("/home/apps/workspace"),
        help="Workspace root containing agent-packages/ (default: /home/apps/workspace).",
    )
    parser.add_argument("--update", action="store_true", help="Write the index file.")
    parser.add_argument(
        "--json", action="store_true", help="Emit the index as JSON to stdout."
    )
    args = parser.parse_args()

    index = build_index(args.workspace_root.resolve())

    if args.json:
        print(json.dumps(index, indent=2))
    else:
        n_targets = len(index["consumers"])
        n_hits = sum(len(v) for v in index["consumers"].values())
        print(
            f"scanned {len(index['repos_scanned'])} fleet repos "
            f"({len(index['repos_missing_locally'])} missing locally); "
            f"{n_targets} distinct agent_utilities targets imported, "
            f"{n_hits} import sites"
        )
        if index["repos_missing_locally"]:
            print("NOT checked out locally (excluded from this index):")
            for r in index["repos_missing_locally"]:
                print(f"  - {r}")

    if args.update:
        INDEX_PATH.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {INDEX_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
