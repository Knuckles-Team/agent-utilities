#!/usr/bin/env python3
"""No agent-facing ingest path calls a per-element engine method in a loop (B-11/GOC-68).

**Context.** `graph_write(action="bulk_ingest")` (`write_ingest_tools.py`) used to be
a plain Python ``for`` loop calling ``engine.add_node()`` once per element — nodes
only, non-atomic, no idempotency key, N round trips over the MessagePack-on-a-socket
transport to the engine. This violates the "batch, never per-element" edict written
in both `agent-utilities`' and `epistemic-graph`'s `AGENTS.md` (eg's states plainly:
*"N elements in a loop = N round-trips = catastrophic"*). `bulk_ingest` was rewritten
onto the engine's native atomic `BatchUpdate`/`ApplyChangeEnvelopes` primitives
(CONCEPT:AU-KG.ingest.envelope-atomic-transaction); this gate is the mechanical
enforcement GOC-68 asked for so the same defect cannot silently reappear — in
`bulk_ingest` itself, or anywhere else an agent-facing tool loops a per-element
engine write instead of batching.

**What it flags.** A static AST heuristic (matching every other `scripts/check_*.py`
gate in this repo — no dataflow/type analysis): a call whose dotted attribute name
ends in one of the per-element INGEST methods below, lexically enclosed by a
``for``/``async for`` loop, anywhere under ``agent_utilities/mcp/`` — the agent-facing
MCP tool surface. Deliberately scoped to CREATE/MERGE methods (the ingest direction
this defect class is about) — ``delete_node``/``delete_edge``/``remove_node``/
``remove_edge`` are a different operation family and out of this gate's scope.

    _INGEST_ATTR_SUFFIXES = {"add_node", "add_edge", "link_nodes",
                              "upsert_node", "upsert_edge", "_upsert_node"}

NOT A RATCHET ANY MORE (CX complexity-collapse program, WD4-RAT-01; supersedes the
frozen ``scripts/no_per_element_ingest_loop_baseline.txt``, now deleted). Measured
before retiring it: the frozen baseline held 7 entries, and the real current
population is **also 7 unique (file, label) sites — 9 raw call sites** once the two
files holding more than one identical-shaped site are counted individually
(``agent_manager.py``'s tool-registry sync and ``kg_server.py``'s capability-
ingest each have more than one call collapsed onto one baseline key). Unlike the
event-loop-blocking and swallowed-error ratchets, this population had not silently
drifted — every baselined entry still exists verbatim — but the same structural
defect was present: the key was ``(file, ENCLOSING FUNCTION, label)``, not invariant
under function extraction/renaming, so this gate could have manufactured the exact
same phantom-finding failure the moment anything in this small population was
refactored. It is retired proactively rather than reactively.

What replaces it, mirroring ``check_swallowed_errors.py``'s now-proven pattern:

* an **unconditional census** prints the real totals — call sites, a breakdown by
  ingest method, worst files — on EVERY run, pass or fail. Nothing is written to
  disk, so no number can go stale.
* enforcement is **diff-scoped**, recomputed live from the HEAD blob: a per-element
  ingest loop this change ADDS to a file under ``agent_utilities/mcp/`` fails the
  commit — including a DELIBERATELY reintroduced one, which is exactly what this
  gate exists to catch — while one it leaves alone does not. The comparison key is
  **content** — the finding's own label (built purely from the AST: the dotted
  ingest-method path) — never the enclosing function name and never a line number.
* no absolute (``HARD_ZERO_SHAPES``) invariant is defined here. Every ingest method
  this gate matches (``add_node``/``add_edge``/``link_nodes``/``upsert_node``/
  ``upsert_edge``/``_upsert_node``) is equally batchable and equally in-scope; none
  is categorically more dangerous than the others the way a bare ``except:`` or a
  literal ``time.sleep()`` is in the sibling gates, so there is no natural
  always-zero subset to defend absolutely. (The delete/remove family sits outside
  this gate's scope entirely — see above — so it isn't a candidate either: this gate
  never measures it, so it can't assert it stays at zero.)

Usage:
  python3 scripts/check_no_per_element_ingest_loop.py       # census + diff-scoped check
  python3 scripts/check_no_per_element_ingest_loop.py ROOT  # scan ROOT, report EVERY
                                                             # finding (test fixture dir)

Exit 0 = this change added no new per-element ingest loop, 1 = it did, 2 = a retired
flag was passed.

KNOWN SCOPE LIMIT (shared with `check_event_loop_blocking.py`): a per-element call
reached through a plain helper function three modules away, or a receiver whose base
name doesn't read as ``engine``/``client``/``backend``, is invisible to this AST
walk — a syntax scan, not a call graph.
"""

from __future__ import annotations

import argparse
import ast
import collections
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts._git_scan import tracked_or_walked  # noqa: E402

SCOPE = ROOT / "agent_utilities" / "mcp"

# Deliberately CREATE/MERGE only — see module docstring for why delete/remove is
# out of scope for this gate.
_INGEST_ATTR_SUFFIXES: frozenset[str] = frozenset(
    {"add_node", "add_edge", "link_nodes", "upsert_node", "upsert_edge", "_upsert_node"}
)


def _tracked_or_walked_py_files(target: Path) -> list[Path]:
    """``.py`` files under ``target``, preferring the git-tracked set (BUG-043) —
    a raw ``rglob`` also picks up gitignored/generated output that can carry a
    stale copy of an already-fixed file and reintroduce a cleared violation."""
    return tracked_or_walked(target, "*.py", root=ROOT)


def _is_test_path(path: Path) -> bool:
    return "/tests/" in str(path) or path.name.startswith("test_")


def _dotted_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return None


class Finding:
    __slots__ = ("path", "line", "func", "label")

    def __init__(self, path: Path, line: int, func: str, label: str) -> None:
        self.path = path
        self.line = line
        self.func = func
        self.label = label

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: [{self.func}] {self.label}"


def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _enclosing_loop(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    cur = parents.get(node)
    while cur is not None:
        if isinstance(cur, (ast.For, ast.AsyncFor)):
            return True
        cur = parents.get(cur)
    return False


def _enclosing_func_name(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    cur = parents.get(node)
    while cur is not None:
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return cur.name
        cur = parents.get(cur)
    return "<module>"


def _finding_for_call(
    node: ast.Call, path: Path, parents: dict[ast.AST, ast.AST]
) -> Finding | None:
    dotted = _dotted_name(node.func)
    if dotted is None:
        return None
    tail = dotted.split(".")[-1]
    if tail not in _INGEST_ATTR_SUFFIXES or not _enclosing_loop(node, parents):
        return None
    func_name = _enclosing_func_name(node, parents)
    return Finding(
        path, node.lineno, func_name, f"per-element ingest call in a loop: {dotted}"
    )


def _scan_tree(tree: ast.AST, path: Path) -> list[Finding]:
    parents = _parent_map(tree)
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            finding = _finding_for_call(node, path, parents)
            if finding is not None:
                findings.append(finding)
    return findings


def scan_file(path: Path) -> list[Finding]:
    try:
        src = path.read_text(encoding="utf-8")
    except OSError:
        return []
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError:
        return []
    return _scan_tree(tree, path)


def _all_findings() -> list[Finding]:
    """Every current candidate site under ``SCOPE`` — the unconditional census
    population. Never fails; nothing written to disk."""
    findings: list[Finding] = []
    for f in _tracked_or_walked_py_files(SCOPE):
        if _is_test_path(f):
            continue
        findings.extend(scan_file(f))
    return findings


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _print_census(findings: list[Finding]) -> None:
    """Print the real numbers. Always. This never fails the run."""
    methods = collections.Counter(f.label.rsplit(".", 1)[-1] for f in findings)
    files = collections.Counter(_rel(f.path) for f in findings)
    print(
        f"per-element-ingest census: {len(findings)} candidate call site(s) "
        f"in {len(files)} file(s)"
    )
    for method, count in methods.most_common():
        print(f"  {count:5d}  {method}")
    for rel, count in files.most_common(5):
        print(f"  top: {count:3d}  {rel}")


# ── Diff-scoped enforcement (the ratchet's replacement) ───────────────────
#
# See module docstring for the full rationale. In short: content-keyed (the
# finding's own label — never the enclosing function, never a line number),
# recomputed live against the HEAD blob per changed file, so extraction and
# renaming are invisible and only a genuinely added per-element ingest loop
# fails.


def _git(*args: str, cwd: str | None = None) -> subprocess.CompletedProcess:
    """Run git from the repo toplevel with repo-relative paths.

    git exports ``GIT_DIR``/``GIT_INDEX_FILE``/``GIT_WORK_TREE`` into every
    hook subprocess. Inherited blindly they silently re-root path resolution,
    which is how ~20 copied gate helpers once measured an empty universe and
    reported a confident clean verdict. They are kept (the index being
    committed IS the one to read) but every invocation runs from the resolved
    toplevel -- never ``git -C <subdir>``.
    """
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
    )


def _repo_root() -> str | None:
    r = _git("rev-parse", "--show-toplevel")
    out = r.stdout.strip()
    return out if r.returncode == 0 and out else None


def _changed_py_files(root: str) -> list[str]:
    """Repo-relative ``agent_utilities/mcp/**.py`` paths differing from HEAD.

    The union of the staged set and the working-tree set: during a commit
    pre-commit has stashed unstaged edits so the two agree, and on a manual
    ``--all-files`` run the working-tree set is the useful one.
    """
    paths: set[str] = set()
    for args in (
        ("diff", "--cached", "--name-only", "--diff-filter=ACMR", "HEAD"),
        ("diff", "--name-only", "HEAD"),
    ):
        r = _git(*args, "--", "agent_utilities/mcp", cwd=root)
        if r.returncode == 0:
            paths.update(line for line in r.stdout.splitlines() if line.endswith(".py"))
    return sorted(paths)


def _content_counts(rel: str, source: str) -> collections.Counter:
    """Findings in one file, counted by CONTENT (the finding's own label) rather
    than by location. Carries neither the enclosing function nor the line
    number, so extraction, renaming, and line motion are all invisible to it."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return collections.Counter()
    return collections.Counter(f.label for f in _scan_tree(tree, Path(rel)))


def _head_source(root: str, rel: str) -> str:
    """``rel`` as of HEAD, or empty when the file is newly added."""
    r = _git("show", f"HEAD:{rel}", cwd=root)
    return r.stdout if r.returncode == 0 else ""


def _added_findings(root: str, rel: str) -> list[tuple[str, int]]:
    """(label, added_count) for content this change ADDS to ``rel``."""
    before = _content_counts(rel, _head_source(root, rel))
    after = _content_counts(rel, Path(root, rel).read_text(encoding="utf-8"))
    return [
        (label, count - before[label])
        for label, count in after.items()
        if count > before[label]
    ]


def _report_added(added: list[tuple[str, str, int]]) -> int:
    print("Per-element ingest loop(s) ADDED by this change:\n")
    for rel, label, count in added:
        suffix = f" (x{count})" if count > 1 else ""
        print(f"  {rel}: {label}{suffix}")
    print(
        "\nRoute the batch through the engine's native atomic primitive instead "
        "of a per-element loop — batch_typed_mutations/BatchUpdate for the light "
        "path, ingest_graph_slice/ApplyChangeEnvelopes when evidence/policy/an "
        "idempotency key is involved — or, if this is a genuine one-off exception, "
        "say why in the commit message. See script docstring.\n"
        "Findings are compared per FILE by content, so MOVING an existing site "
        "between files reads as added in the destination."
    )
    return 1


def _scan_explicit_root(root_arg: str) -> int:
    """Absolute mode: report EVERY finding under an arbitrary path.

    Used by this gate's own tests to prove it trips on a known-bad fixture.
    """
    explicit_root = Path(root_arg)
    files = (
        [explicit_root]
        if explicit_root.is_file()
        else _tracked_or_walked_py_files(explicit_root)
    )
    findings: list[Finding] = []
    for f in files:
        findings.extend(scan_file(f))
    if not findings:
        print(f"OK — no per-element-ingest-loop candidate sites under {explicit_root}")
        return 0
    print("Per-element ingest loop candidate site(s) found:\n")
    for f in sorted(findings, key=lambda fd: (str(fd.path), fd.line)):
        print(f"  {f}")
    return 1


def _enforce_diff_scoped() -> int:
    root = _repo_root()
    if root is None:
        print("  (not inside a work tree — diff-scoped enforcement skipped)")
        return 0
    added: list[tuple[str, str, int]] = []
    for rel in _changed_py_files(root):
        if _is_test_path(Path(rel)) or not Path(root, rel).is_file():
            continue
        added.extend((rel, label, count) for label, count in _added_findings(root, rel))
    return _report_added(added) if added else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "root",
        nargs="?",
        help=(
            "Scan this path instead of agent_utilities/mcp/, reporting EVERY "
            "finding with no diff scoping — for an arbitrary fixture/test "
            "directory, mirroring check_swallowed_errors.py."
        ),
    )
    ap.add_argument("--update-baseline", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.update_baseline:
        print(
            "--update-baseline is RETIRED. This gate has no baseline: it "
            "prints the real census every run and enforces diff-scoped, so "
            "there is nothing to freeze. See the module docstring.",
            file=sys.stderr,
        )
        return 2

    if args.root:
        return _scan_explicit_root(args.root)

    findings = _all_findings()
    _print_census(findings)
    diff_scoped = _enforce_diff_scoped()
    if diff_scoped:
        return 1
    print("OK — no per-element ingest loop added by this change.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
