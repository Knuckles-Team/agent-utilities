#!/usr/bin/env python3
"""Event-loop-blocking heuristic scan (CONCEPT:AU-ORCH.execution.event-loop-blocking-sweep).

Context: graph-os has been liveness-killed in production (exit 137, `/health/ready`
"context deadline exceeded" while the engine logs a sustained ``engine_breaker: slow
engine call ... duration=3-8s`` stream) — the signature of synchronous KG/file/network
calls executed directly on the request-serving event loop, starving the health probe.
Three independent sweeps (agent-runner/router, governed_dynamic_workflow, and this one
covering the rest of ``mcp/tools/`` + the background loop controllers) each found
blocking sites the others missed, so the pattern is systemic rather than a handful of
one-off bugs — this is the guard that keeps it from silently regressing.

This is a **static AST heuristic**, matching every other ``scripts/check_*.py`` gate in
this repo: it flags a *known* blocking call (a synchronous KG/engine write or read, a
blocking file read/write, ``subprocess``, a synchronous HTTP client call, or
``time.sleep``) written as a literal call expression directly inside an ``async def``
body, with no thread hop in between. It is deliberately narrow — it does not attempt
dataflow/type analysis, so it cannot see a blocking call hidden behind an arbitrary
helper function of an unrecognized name three modules away. It also does not flag a
call already isolated via the sanctioned hop: :func:`agent_utilities.core.event_loop.
run_blocking_ordered`, ``asyncio.to_thread``, :func:`agent_utilities.mcp.concurrency.
run_blocking`/``invoke_client_method``, or ``loop.run_in_executor`` — including the
"nested sync closure" shape this sweep used throughout (define a ``def _do_thing():
...`` inside the ``async def``, then ``await run_blocking_ordered(_do_thing)``): a
nested function is exempted when its name is passed as a bare reference to one of
those hop calls anywhere in the same enclosing async function.

NOT A RATCHET ANY MORE (CX complexity-collapse program, WD4-RAT-01; supersedes the
frozen ``scripts/event_loop_blocking_baseline.txt``, now deleted). Measured before
retiring it: the baseline claimed 120 frozen entries; the real current population was
**109 unique (file, label) sites — 149 raw call sites** once duplicate calls sharing a
key are counted individually. 11 baselined entries no longer exist in the tree at all
(fixed since the baseline was last regenerated) and the gate's own output never said
so — the exact silent-shrinkage failure mode already found and fixed in
``check_swallowed_errors.py`` (D-SWG-1): a frozen count hides real progress instead of
reporting it. Worse, the baseline key was ``(file, ENCLOSING FUNCTION, label)`` — not
invariant under function extraction/renaming, so a pure refactor that renamed or split
the enclosing function could manufacture a phantom "new site" with no code change.

What replaces it, mirroring ``check_swallowed_errors.py``'s now-proven pattern:

* an **unconditional census** prints the real totals — call sites, a breakdown by
  shape, worst files — on EVERY run, pass or fail. Nothing is written to disk, so no
  number can go stale.
* enforcement is **diff-scoped**, recomputed live from the HEAD blob: a blocking call
  this change ADDS to a file fails the commit; one it leaves alone does not. The
  comparison key is **content** — the finding's own human-readable label (built purely
  from the AST: the call category + its dotted receiver path) — never the enclosing
  function name and never a line number, so extraction, renaming, reordering and line
  motion are all invisible to it and only a genuinely added blocking call fails.
* one shape, ``time.sleep`` inside an ``async def``, is enforced **absolutely**,
  repo-wide (``HARD_ZERO_SHAPES``): the repo is already at zero for it, and unlike an
  engine call that might legitimately need a closer look, a literal ``time.sleep()``
  on the request-serving loop has no legitimate reading — it blocks every in-flight
  request for its full duration, full stop.

Usage:
  python3 scripts/check_event_loop_blocking.py       # census + diff-scoped check
  python3 scripts/check_event_loop_blocking.py ROOT   # scan ROOT, report EVERY
                                                       # finding (test fixture dir)

Exit 0 = this change added no new blocking-call site, 1 = it did (or a hard-zero shape
exists), 2 = a retired flag was passed.

KNOWN SCOPE LIMIT — plain-``def`` tool handlers are NOT covered. This scanner walks
``ast.AsyncFunctionDef`` bodies only. A large share of this package's MCP tool handlers
(``graph_query``, ``graph_ask``, ``nl_query``, ``ask_data``, ``graph_table``,
``graph_search``, ``graph_code_nav``, ``graph_write``, ``graph_kv_checkpoint``, …) are
plain ``def``. On the REST/native-delegation path ``kg_server._execute_tool`` hops those
onto a thread; on the MCP wire path this package imports the standalone ``fastmcp``
distribution (``from fastmcp import FastMCP`` in ``agent_utilities/mcp/server_factory.py``),
whose ``FunctionTool._execute`` dispatches a sync body through
``call_sync_fn_in_threadpool`` whenever ``run_in_thread`` is set — and it defaults to
``True`` and is never overridden anywhere in this repo. So sync handlers are in fact
already thread-hopped on BOTH paths today. They are still out of scope for this scanner,
but as an untriaged population rather than an unprotected one.

SECOND SCOPE LIMIT — only ``engine.*``-shaped attribute calls and a fixed set of
``pathlib``/``open`` file operations are matched. A blocking call reached through a plain
helper function (``persist_facts(store, facts)``, which loops over synchronous KG writes)
is invisible to this AST walk; catching that class needs a call graph, not a syntax
scan. Recorded as D-W15-6 in ``reports/deferred/waves1-5-gate.md``.
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

PKG = ROOT / "agent_utilities"

# The one shape enforced absolutely, repo-wide, because the repo is already at zero
# for it and there is no legitimate reading of a literal `time.sleep()` directly on
# the request-serving event loop — see module docstring.
HARD_ZERO_SHAPES = frozenset({"time_sleep"})


def _tracked_or_walked_py_files(target: Path) -> list[Path]:
    """``.py`` files under ``target``, preferring the git-tracked set (BUG-043).

    A raw ``rglob`` also picks up gitignored, generated build output, which
    can carry a stale copy of an already-fixed source file and reintroduce a
    cleared violation. Falls back to a filesystem walk only when ``target``
    is not inside a git working tree (e.g. a synthetic test fixture).
    """
    return tracked_or_walked(target, "*.py", root=ROOT)


def _is_test_path(path: Path) -> bool:
    return "/tests/" in str(path) or path.name.startswith("test_")


# Hop helpers: passing the blocking callable as a bare reference to one of these is
# the sanctioned escape hatch. Matched by the CALLED function's simple name only
# (not fully-qualified) so both ``run_blocking_ordered(...)`` and
# ``event_loop.run_blocking_ordered(...)`` import spellings are recognized.
_HOP_CALL_NAMES = {
    "run_blocking_ordered",
    "run_blocking",
    "invoke_client_method",
    "to_thread",  # asyncio.to_thread
    "run_in_executor",
}

# (attribute suffix, human label) for KG/engine calls that are synchronous methods
# in this codebase (see agent_utilities/knowledge_graph/core/engine.py,
# core/graph_compute.py, _engine_protocol.py — all plain ``def``, never
# ``async def``).
_BLOCKING_ATTR_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("add_node", "engine write: add_node"),
    ("add_edge", "engine write: add_edge"),
    ("link_nodes", "engine write: link_nodes"),
    ("delete_node", "engine write: delete_node"),
    ("upsert_node", "engine write: upsert_node"),
    ("_upsert_node", "engine write: _upsert_node"),
    ("query_cypher", "engine read: query_cypher"),
    ("query_cypher_write", "engine write: query_cypher_write"),
    ("execute_cypher", "engine: execute_cypher"),
    ("run_cypher", "engine: run_cypher"),
    ("discover_agents", "registry: discover_agents"),
    ("get_discovery_registry", "registry: get_discovery_registry"),
    ("read_text", "blocking file read: Path.read_text"),
    ("write_text", "blocking file write: Path.write_text"),
    ("read_bytes", "blocking file read: Path.read_bytes"),
    ("write_bytes", "blocking file write: Path.write_bytes"),
    ("submit_task", "engine write: submit_task (durable WorkItem enqueue)"),
    ("get_blast_radius", "engine read: get_blast_radius"),
    ("search_hybrid", "engine read: search_hybrid"),
    ("get_shortest_path", "engine read: get_shortest_path"),
    ("execute_federated_query", "engine read: execute_federated_query"),
    ("get_text_embedding", "blocking remote-embedder call: get_text_embedding"),
    (
        "get_text_embedding_batch",
        "blocking remote-embedder call: get_text_embedding_batch",
    ),
)

_SUBPROCESS_FUNCS = {"run", "call", "check_call", "check_output", "Popen"}
_REQUESTS_FUNCS = {"get", "post", "put", "delete", "patch", "head"}


class Finding:
    __slots__ = ("path", "line", "func", "label")

    def __init__(self, path: Path, line: int, func: str, label: str) -> None:
        self.path = path
        self.line = line
        self.func = func
        self.label = label

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: [async def {self.func}] {self.label}"


def _label_shape(label: str) -> str:
    """A coarse category for one finding's ``label``, used for the census
    breakdown and the ``HARD_ZERO_SHAPES`` check. Pure string classification —
    the label itself is already fully determined by the AST (see ``_scan_call``),
    so this stays formatting-insensitive."""
    prefixes = (
        ("time.sleep", "time_sleep"),
        ("subprocess.", "subprocess"),
        ("requests.", "http_request"),
        ("open()", "open_file"),
        ("engine write", "engine_write"),
        ("engine read", "engine_read"),
        ("engine:", "engine_other"),
        ("registry:", "registry_call"),
        ("blocking remote-embedder", "embedder_call"),
    )
    for prefix, shape in prefixes:
        if label.startswith(prefix):
            return shape
    return "path_file_io" if "blocking file" in label else "other"


def _dotted_name(node: ast.AST) -> str | None:
    """Best-effort dotted-path string for a Call target, e.g. 'engine.query_cypher'."""
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return None


def _protected_nested_defs(async_fn: ast.AsyncFunctionDef) -> set[str]:
    """Nested ``def``/``async def`` names hopped via a bare-reference hop call.

    Matches the pattern this sweep used throughout: define a nested sync closure,
    then ``await run_blocking_ordered(_closure_name, ...)``. The closure's own
    blocking calls are exempt because the closure itself always runs off-loop.
    """
    protected: set[str] = set()
    for node in ast.walk(async_fn):
        if not isinstance(node, ast.Call):
            continue
        callee = node.func
        name = (
            callee.attr
            if isinstance(callee, ast.Attribute)
            else (callee.id if isinstance(callee, ast.Name) else None)
        )
        if name not in _HOP_CALL_NAMES:
            continue
        for arg in node.args:
            if isinstance(arg, ast.Name):
                protected.add(arg.id)
    return protected


def _label_for_named_call(dotted: str) -> str | None:
    """The label for a bare-module-function shape: ``time.sleep``, ``subprocess.*``,
    ``requests.*``. Split out of ``_scan_call`` to keep each check's branch count low."""
    base, tail = dotted.split(".")[0], dotted.split(".")[-1]
    if tail == "sleep" and base == "time":
        return "time.sleep (blocks the loop)"
    if base == "subprocess" and tail in _SUBPROCESS_FUNCS:
        return f"subprocess.{tail}"
    if base == "requests" and tail in _REQUESTS_FUNCS:
        return f"requests.{tail}"
    return None


def _label_for_attr_suffix(tail: str, dotted: str) -> str | None:
    """The label for a ``_BLOCKING_ATTR_SUFFIXES`` match, e.g. an engine/KG call."""
    for suffix, human_label in _BLOCKING_ATTR_SUFFIXES:
        if tail == suffix:
            return f"{human_label} ({dotted})"
    return None


def _scan_call(call: ast.Call, func_name: str, path: Path) -> Finding | None:
    if isinstance(call.func, ast.Name) and call.func.id == "open":
        return Finding(path, call.lineno, func_name, "open() (blocking file I/O)")
    dotted = _dotted_name(call.func)
    if dotted is None:
        return None
    tail = dotted.split(".")[-1]
    label = _label_for_named_call(dotted) or _label_for_attr_suffix(tail, dotted)
    return Finding(path, call.lineno, func_name, label) if label else None


def _visit_call_node(
    child: ast.Call, func_name: str, path: Path, protected: set[str], out: list[Finding]
) -> None:
    finding = _scan_call(child, func_name, path)
    if finding is not None:
        out.append(finding)
    for grandchild in ast.iter_child_nodes(child):
        _walk_async_body(grandchild, func_name, path, protected, out)


def _walk_async_body(
    node: ast.AST, func_name: str, path: Path, protected: set[str], out: list[Finding]
) -> None:
    """Walk statements of an async function, not descending into nested scopes
    (those are handled separately: skipped if hopped, else scanned standalone)."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.Call):
            _visit_call_node(child, func_name, path, protected, out)
            continue
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            if getattr(child, "name", "<lambda>") in protected:
                continue  # off-loaded via run_blocking_ordered/to_thread elsewhere
            # An un-hopped nested sync closure runs on the SAME thread as its
            # enclosing async function when called directly, so its blocking
            # calls are scanned too, just attributed to the outer function name
            # for a readable report.
        _walk_async_body(child, func_name, path, protected, out)


def _scan_tree(tree: ast.AST, path: Path) -> list[Finding]:
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef):
            protected = _protected_nested_defs(node)
            _walk_async_body(node, node.name, path, protected, findings)
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
    """Every current candidate site under ``PKG`` — the unconditional census
    population. Never fails; nothing written to disk."""
    findings: list[Finding] = []
    for f in _tracked_or_walked_py_files(PKG):
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
    shapes = collections.Counter(_label_shape(f.label) for f in findings)
    files = collections.Counter(_rel(f.path) for f in findings)
    print(
        f"event-loop-blocking census: {len(findings)} candidate call site(s) "
        f"in {len(files)} file(s)"
    )
    for shape, count in shapes.most_common():
        print(f"  {count:5d}  {shape}")
    for rel, count in files.most_common(5):
        print(f"  top: {count:3d}  {rel}")


def _report_hard_zero(findings: list[Finding]) -> int:
    """The one absolute invariant. The repo is at zero; keep it there."""
    offenders = sorted(
        (_rel(f.path), f.line, f.label)
        for f in findings
        if _label_shape(f.label) in HARD_ZERO_SHAPES
    )
    if not offenders:
        return 0
    print(f"\n{len(offenders)} call site(s) of an absolutely-forbidden shape:\n")
    for rel, line, label in offenders:
        print(f"  {rel}:{line} [{label}]")
    print(
        "\ntime.sleep() inside an `async def` blocks the ENTIRE event loop for every "
        "in-flight request for its full duration. Use `await asyncio.sleep(...)` "
        "instead, or hop the call off-loop."
    )
    return 1


# ── Diff-scoped enforcement (the ratchet's replacement) ───────────────────
#
# See module docstring for the full rationale. In short: content-keyed (the
# finding's own label — never the enclosing function, never a line number),
# recomputed live against the HEAD blob per changed file, so extraction and
# renaming are invisible and only a genuinely added blocking call fails.


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
    """Repo-relative ``agent_utilities/**.py`` paths differing from HEAD.

    The union of the staged set and the working-tree set: during a commit
    pre-commit has stashed unstaged edits so the two agree, and on a manual
    ``--all-files`` run the working-tree set is the useful one.
    """
    paths: set[str] = set()
    for args in (
        ("diff", "--cached", "--name-only", "--diff-filter=ACMR", "HEAD"),
        ("diff", "--name-only", "HEAD"),
    ):
        r = _git(*args, "--", "agent_utilities", cwd=root)
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
    print("Event-loop-blocking candidate site(s) ADDED by this change:\n")
    for rel, label, count in added:
        suffix = f" (x{count})" if count > 1 else ""
        print(f"  {rel}: {label}{suffix}")
    print(
        "\nHop the blocking call off the event loop — "
        "agent_utilities.core.event_loop.run_blocking_ordered, asyncio.to_thread, "
        "agent_utilities.mcp.concurrency.run_blocking/invoke_client_method, or "
        "loop.run_in_executor — or, if it is genuinely non-blocking / already "
        "isolated for a reason this AST heuristic can't see, say why in the commit "
        "message. See script docstring.\n"
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
        print(f"OK — no event-loop-blocking candidate sites under {explicit_root}")
        return 0
    print("Event-loop-blocking candidate site(s) found:\n")
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
            "Scan this path instead of agent_utilities/, reporting EVERY "
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
    hard_zero = _report_hard_zero(findings)
    diff_scoped = _enforce_diff_scoped()
    if hard_zero or diff_scoped:
        return 1
    print("OK — no event-loop-blocking site added by this change.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
