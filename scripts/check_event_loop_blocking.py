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

RATCHET, not report-only (D-G2C — closed the D-54 deferred item). A prior full sweep
(`0aae9c59` + this one) fixed the highest-value sites reachable in one pass, but this
codebase still has a large, untriaged population of matching call shapes (every
remaining ``async def`` MCP tool handler, every background daemon coroutine) that
nobody has individually verified is either genuinely non-blocking (e.g. the engine's
own async wire client) or already safe for another reason (a fresh per-task event loop
on a dedicated worker thread, as ``knowledge_graph/core/engine_tasks.py``'s
``_run_background_task`` is). Turning this into a hard gate against the WHOLE
population would fail on a mountain of pre-existing, untriaged code (noisy-always-
failing is worse than no gate) — so, mirroring ``check_no_env_sprawl.py``'s ratchet,
the pre-existing population is frozen in ``scripts/event_loop_blocking_baseline.txt``
and the gate fails only on *new* candidate sites not already in that baseline. Fixing
a baselined site (adding a real thread hop) is always allowed and shrinks the baseline
on the next ``--update-baseline``; regenerate after any such fix so the ratchet keeps
tightening.

Usage:
  python3 scripts/check_event_loop_blocking.py [PATH…]           # ratchet check (exit 1 on new sites)
  python3 scripts/check_event_loop_blocking.py --update-baseline # freeze the current set

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
already thread-hopped on BOTH paths today. (An earlier version of this note claimed they
were invoked inline via ``call_fn_with_arg_validation`` → ``return fn(**args)``; that is
the OTHER, SDK-bundled ``mcp.server.fastmcp`` implementation, which this package does not
use.) They are still out of scope for this scanner, but as an untriaged population rather
than an unprotected one — and widening to sync defs must not assume the inline-dispatch
risk this note used to assert.

SECOND SCOPE LIMIT — only ``engine.*``-shaped attribute calls and a fixed set of
``pathlib``/``open`` file operations are matched. A blocking call reached through a plain
helper function (``persist_facts(store, facts)``, which loops over synchronous KG writes)
is invisible to this AST walk; catching that class needs a call graph, not a syntax
scan. Recorded as D-W15-6 in ``reports/deferred/waves1-5-gate.md``.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "agent_utilities"


def _tracked_or_walked_py_files(target: Path) -> list[Path]:
    """``.py`` files under ``target``, preferring the git-tracked set (BUG-043).

    A raw ``rglob`` also picks up gitignored, generated build output, which
    can carry a stale copy of an already-fixed source file and reintroduce a
    cleared violation. Falls back to a filesystem walk only when ``target``
    is not inside a git working tree (e.g. a synthetic test fixture).
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(target), "ls-files", "--", "*.py"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        tracked = [target / line for line in out.splitlines() if line]
        if tracked:
            return [p for p in tracked if p.is_file()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return sorted(target.rglob("*.py"))
BASELINE = ROOT / "scripts" / "event_loop_blocking_baseline.txt"

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

    def key(self) -> tuple[str, str, str]:
        """Ratchet identity: (relpath, function, label) — deliberately WITHOUT the
        line number. A line number shifts on any unrelated edit above it in the
        same file, which would make the baseline spuriously "new" on every diff;
        (file, function, blocking-call-shape) is what actually identifies a site
        for triage purposes, mirroring ``check_no_env_sprawl.py``'s (relpath, KEY).
        """
        try:
            rel = str(self.path.resolve().relative_to(ROOT))
        except ValueError:
            rel = str(self.path)
        return (rel, self.func, self.label)


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


def _scan_call(call: ast.Call, func_name: str, path: Path) -> Finding | None:
    dotted = _dotted_name(call.func)
    if dotted is None:
        return None
    base = dotted.split(".")[0]
    tail = dotted.split(".")[-1]

    if tail == "sleep" and base == "time":
        return Finding(path, call.lineno, func_name, "time.sleep (blocks the loop)")
    if base == "subprocess" and tail in _SUBPROCESS_FUNCS:
        return Finding(path, call.lineno, func_name, f"subprocess.{tail}")
    if base == "requests" and tail in _REQUESTS_FUNCS:
        return Finding(path, call.lineno, func_name, f"requests.{tail}")
    if isinstance(call.func, ast.Name) and call.func.id == "open":
        return Finding(path, call.lineno, func_name, "open() (blocking file I/O)")
    for suffix, label in _BLOCKING_ATTR_SUFFIXES:
        if tail == suffix:
            return Finding(path, call.lineno, func_name, f"{label} ({dotted})")
    return None


def _walk_async_body(
    node: ast.AST, func_name: str, path: Path, protected: set[str], out: list[Finding]
) -> None:
    """Walk statements of an async function, not descending into nested scopes
    (those are handled separately: skipped if hopped, else scanned standalone)."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.Call):
            finding = _scan_call(child, func_name, path)
            if finding is not None:
                out.append(finding)
            for grandchild in ast.iter_child_nodes(child):
                _walk_async_body(grandchild, func_name, path, protected, out)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            nested_name = getattr(child, "name", "<lambda>")
            if nested_name in protected:
                continue  # off-loaded via run_blocking_ordered/to_thread elsewhere
            # An un-hopped nested sync closure runs on the SAME thread as its
            # enclosing async function when called directly, so its blocking
            # calls are scanned too, just attributed to the outer function name
            # for a readable report.
            _walk_async_body(child, func_name, path, protected, out)
        else:
            _walk_async_body(child, func_name, path, protected, out)


def scan_file(path: Path) -> list[Finding]:
    try:
        src = path.read_text(encoding="utf-8")
    except OSError:
        return []
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError:
        return []
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef):
            protected = _protected_nested_defs(node)
            _walk_async_body(node, node.name, path, protected, findings)
    return findings


def load_baseline() -> set[tuple[str, str, str]]:
    if not BASELINE.exists():
        return set()
    out: set[tuple[str, str, str]] = set()
    for line in BASELINE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) == 3:
            out.add((parts[0], parts[1], parts[2]))
    return out


def write_baseline(findings: list[Finding]) -> None:
    entries = sorted({finding.key() for finding in findings})
    body = "\n".join("\t".join(entry) for entry in entries)
    BASELINE.write_text(
        "# Frozen baseline of check_event_loop_blocking.py candidate sites\n"
        "# (ratchet — burn-down toward zero, D-G2C/D-54). Format: relpath\\tfunc\\tlabel.\n"
        "# New sites not listed here fail the gate; fixing a listed site (a real thread\n"
        "# hop via run_blocking_ordered/to_thread) is always allowed and should be\n"
        "# followed by --update-baseline to shrink this file. See script docstring.\n"
        + body
        + "\n",
        encoding="utf-8",
    )


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if a != "--update-baseline"]
    update_baseline = "--update-baseline" in argv[1:]
    targets = [Path(a) for a in args] or [PKG]
    all_findings: list[Finding] = []
    for target in targets:
        files = [target] if target.is_file() else _tracked_or_walked_py_files(target)
        for f in files:
            if "/tests/" in str(f) or f.name.startswith("test_"):
                continue
            all_findings.extend(scan_file(f))

    if update_baseline:
        write_baseline(all_findings)
        print(
            f"check_event_loop_blocking: baseline updated: "
            f"{len({f.key() for f in all_findings})} entries -> {BASELINE.name}"
        )
        return 0

    if not all_findings:
        print("check_event_loop_blocking: no candidate sites found.")
        return 0

    baseline = load_baseline()
    current_keys = {f.key() for f in all_findings}
    new_findings = [f for f in all_findings if f.key() not in baseline]

    if new_findings:
        print(
            f"check_event_loop_blocking: {len(new_findings)} NEW candidate site(s) "
            f"not in the frozen baseline ({BASELINE.name}):"
        )
        for finding in new_findings:
            print(f"  {finding}")
        print(
            "\nAdd a real thread hop (run_blocking_ordered / asyncio.to_thread) around "
            "the blocking call, or — if it is genuinely non-blocking / already isolated "
            "for a reason this AST heuristic can't see — regenerate the baseline with "
            "--update-baseline and say why in the commit message. See script docstring."
        )
        return 1

    removed = len(baseline) - len(baseline & current_keys)
    msg = f"check_event_loop_blocking: OK — no new candidate sites ({len(current_keys)} baselined"
    print(msg + (f", {removed} fixed since baseline)." if removed else ")."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
