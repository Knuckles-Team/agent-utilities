#!/usr/bin/env python3
"""Source delivery against fresh main; release and deferred APIs remain strict."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import date
from functools import cache
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STRICT_PATHS = frozenset(
    {
        "scripts/check_liveness.py",
        "scripts/liveness_reconciler.py",
        "scripts/liveness_deferred.py",
        "scripts/liveness_deferred.tsv",
        "scripts/check_wiring.py",
        "scripts/_git_subprocess_env.py",
        "tests/gates/test_liveness_deferred.py",
        ".github/workflows/release.yml",
    }
)


class CannotRun(RuntimeError):
    """The comparison cannot establish valid evidence."""


@cache
def _strict_gate():
    spec = importlib.util.spec_from_file_location(
        "_strict_source_liveness", REPO / "scripts/check_liveness.py"
    )
    if spec is None or spec.loader is None:
        raise CannotRun("strict liveness gate is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str) -> str:
    gate = _strict_gate()
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        env=gate._git_env.sanitized_git_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise CannotRun(f"git {args[0]} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _remote_main(repo: Path) -> str:
    rows = _git(
        repo, "ls-remote", "--exit-code", "origin", "refs/heads/main"
    ).splitlines()
    if len(rows) != 1 or rows[0].split()[1:] != ["refs/heads/main"]:
        raise CannotRun("could not resolve exactly one remote main")
    return rows[0].split()[0]


def _scope(repo: Path, base: str, candidate: str) -> list[str]:
    for oid in (base, candidate):
        if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", oid):
            raise CannotRun("base and candidate must be full immutable Git object IDs")
    if _git(repo, "cat-file", "-t", base) != "commit":
        raise CannotRun("base must be a commit")
    if _git(repo, "cat-file", "-t", candidate) not in {"commit", "tree"}:
        raise CannotRun("candidate must be a commit or frozen index tree")
    # Both sides of a move must be checked against the protected paths.
    rows = _git(
        repo, "diff", "--raw", "--no-abbrev", "--no-renames", base, candidate
    ).splitlines()
    return [_source_path(row) for row in rows]


def _source_path(row: str) -> str:
    metadata, separator, path = row.partition("\t")
    fields = metadata.split()
    if not separator or len(fields) != 5:
        raise CannotRun(f"unsupported source path/mode: {row}")
    if (
        fields[0][1:] not in {"000000", "100644", "100755"}
        or fields[1] not in {"000000", "100644", "100755"}
        or fields[4] not in {"A", "M", "D"}
        or path.startswith('"')
    ):
        raise CannotRun(f"unsupported source path/mode: {path}")
    return path


def _requires_strict(paths: list[str], obligations: str) -> bool:
    entries = _strict_gate().liveness_deferred.parse_entries(obligations)
    return any(
        path in STRICT_PATHS
        or path.startswith(("agent_utilities/core/registry/", "scripts/release/"))
        or any(fnmatch.fnmatchcase(path, entry.pattern) for entry in entries)
        for path in paths
    )


@contextmanager
def _snapshot(repo: Path, base: str, tree: str):
    # Only remove the worktree created inside this invocation's private directory.
    with tempfile.TemporaryDirectory(prefix="liveness-source-") as temporary:
        root = Path(temporary) / "tree"
        _git(repo, "worktree", "add", "--quiet", "--detach", str(root), base)
        try:
            _git(root, "read-tree", "--reset", "-u", tree)
            yield root
        finally:
            _git(repo, "worktree", "remove", "--force", str(root))


_SCAN = """
import contextlib, importlib.util, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('strict_snapshot', root / 'scripts/check_liveness.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
with contextlib.redirect_stdout(sys.stderr):
    analyzer = gate._import_analyzer(pathlib.Path(sys.argv[2]))
    counts, details, coverage = gate._run_census(pathlib.Path(sys.argv[2]))
    semantic = {}
    for rel in json.loads(sys.argv[3]):
        path = root / rel
        semantic[rel] = sorted(gate._file_findings(analyzer, rel, path.read_text()).findings) if path.exists() else []
print(json.dumps({'counts': counts, 'details': details, 'coverage': coverage, 'semantic': semantic}))
"""


def _scan(root: Path, analyzer: Path, paths: tuple[str, ...] | list[str] = ()) -> dict:
    result = subprocess.run(
        [sys.executable, "-c", _SCAN, str(root), str(analyzer), json.dumps(paths)],
        cwd=root,
        env=_strict_gate()._git_env.sanitized_git_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    print(result.stderr, end="", file=sys.stderr)
    if result.returncode:
        raise CannotRun(f"snapshot analyzer failed (exit {result.returncode})")
    try:
        report = json.loads(result.stdout)
        _validate_report(report, paths)
    except (KeyError, TypeError, ValueError) as exc:
        raise CannotRun("snapshot report is incomplete or malformed") from exc
    return report


def _identity_list(items) -> bool:
    return isinstance(items, list) and all(isinstance(item, str) for item in items)


def _validate_report(report: dict, paths) -> None:
    for category in _strict_gate().CATEGORIES:
        count, items = report["counts"][category], report["details"][category]
        if type(count) is not int or not _identity_list(items) or count != len(items):
            raise ValueError("incomplete findings")
    if type(report["coverage"]) is not bool:
        raise ValueError("missing coverage status")
    if set(report["semantic"]) != set(paths):
        raise ValueError("missing content identities")
    if not all(_identity_list(items) for items in report["semantic"].values()):
        raise ValueError("invalid content identities")


def _compare(before: dict, after: dict) -> bool:
    gate = _strict_gate()
    failed = gate._enforce_absolute(after["counts"], {})
    if before["coverage"] != after["coverage"]:
        raise CannotRun("coverage inputs differ")
    for category in gate.CATEGORIES:
        added = set(after["details"][category]) - set(before["details"][category])
        for identity in sorted(added):
            print(f"NEW {category}: {identity}")
        failed |= bool(added)
    for path, identities in after["semantic"].items():
        added = set(identities) - set(before["semantic"][path])
        for identity in sorted(added):
            print(f"NEW content finding: {path} {identity}")
        failed |= bool(added)
    return failed


def _deferrals(text: str, as_of: date) -> list:
    gate = _strict_gate()
    try:
        entries = gate.liveness_deferred.parse_entries(text)
    except ValueError as exc:
        raise CannotRun("deferrals are unparsable") from exc
    if any(not gate.liveness_deferred.is_well_formed(entry) for entry in entries):
        raise CannotRun("deferrals are malformed")
    return gate.liveness_deferred.stale_entries(entries, as_of)


def run(repo: Path, base: str, candidate: str, *, release: bool = False) -> int:
    paths = _scope(repo, base, candidate)
    if _remote_main(repo) != base:
        raise CannotRun("base is not fresh remote main")
    gate = _strict_gate()
    analyzer = gate._find_analyzer()
    if analyzer is None:
        raise CannotRun("required liveness analyzer is unavailable")
    analyzer_hash = hashlib.sha256(analyzer.read_bytes()).hexdigest()
    as_of = date.today()
    if (repo / "coverage.json").exists():
        raise CannotRun(
            "coverage.json is present outside the frozen comparison; use strict enforcement"
        )
    obligations = _git(repo, "show", f"{base}:scripts/liveness_deferred.tsv")
    strict = release or _requires_strict(paths, obligations)
    if strict:
        with _snapshot(repo, base, candidate) as root:
            print(
                "Strict liveness required: release or protected liveness surface",
                flush=True,
            )
            result = _strict_snapshot(root)
        _verify_evidence(repo, base, analyzer, analyzer_hash, as_of)
        return result
    before, after, stale = _paired_census(repo, base, candidate, paths, analyzer, as_of)
    _verify_evidence(repo, base, analyzer, analyzer_hash, as_of)
    failed = _compare(before, after)
    _print_verdict(base, candidate, as_of, analyzer_hash, stale, failed)
    return int(failed)


def _paired_census(repo, base, candidate, paths, analyzer, as_of):
    semantic_paths = [
        p for p in paths if p.startswith("agent_utilities/") and p.endswith(".py")
    ]
    with _snapshot(repo, base, base) as baseline_root:
        baseline_text = (baseline_root / "scripts/liveness_deferred.tsv").read_text()
        stale = _deferrals(baseline_text, as_of)
        before = _scan(baseline_root, analyzer, semantic_paths)
    with _snapshot(repo, base, candidate) as candidate_root:
        if (
            candidate_root / "scripts/liveness_deferred.tsv"
        ).read_text() != baseline_text:
            raise CannotRun("deferral obligations changed")
        after = _scan(candidate_root, analyzer, semantic_paths)
    return before, after, stale


def _verify_evidence(repo, base, analyzer, analyzer_hash, as_of):
    if date.today() != as_of:
        raise CannotRun("evaluation date changed; rerun both snapshots")
    if hashlib.sha256(analyzer.read_bytes()).hexdigest() != analyzer_hash:
        raise CannotRun("analyzer changed during comparison")
    if _remote_main(repo) != base:
        raise CannotRun("remote main moved during comparison; rebase and rerun")


def _print_verdict(base, candidate, as_of, analyzer_hash, stale, failed):
    print(
        f"Source comparison: base={base} candidate={candidate} date={as_of} analyzer={analyzer_hash}"
    )
    for entry in stale:
        print(
            f"UNRESOLVED EXPIRY: {entry.category} {entry.pattern} owner={entry.owner} review-by={entry.review_by}"
        )
    print(
        "Source differential: "
        + ("FAIL" if failed else "PASS")
        + "; this is not strict/release liveness approval."
    )


def _strict_snapshot(root: Path) -> int:
    return subprocess.run(
        [sys.executable, str(root / "scripts/check_liveness.py")],
        cwd=root,
        env=_strict_gate()._git_env.sanitized_git_env(),
        check=False,
    ).returncode


def _staged_tree(repo: Path) -> str:
    # Keep Git's alternate index for `git commit <paths>` and partial commits.
    # Only this capture inherits it; snapshot operations sanitize Git variables.
    root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if Path(root).resolve() != repo.resolve():
        raise CannotRun("hook index belongs to a different checkout")
    return subprocess.run(
        ["git", "write-tree"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hook", choices=("commit",))
    parser.add_argument("--base", help="fresh origin/main full commit ID")
    parser.add_argument(
        "--candidate",
        help="full candidate commit or frozen index tree ID",
    )
    args = parser.parse_args(argv)
    try:
        if args.hook:
            if args.base or args.candidate:
                raise CannotRun("hook snapshots cannot be overridden")
            candidate = _staged_tree(REPO)
            base = _remote_main(REPO)
            return run(REPO, base, candidate)
        if not args.base or not args.candidate:
            raise CannotRun("both base and candidate are required")
        return run(REPO, args.base, args.candidate)
    except (CannotRun, OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Source liveness CANNOT RUN: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
