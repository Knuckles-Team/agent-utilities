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

**RATCHET, not a blanket ban.** A full sweep of `agent_utilities/mcp/` at the time
this gate was added found several pre-existing per-element ingest loops OUTSIDE
`bulk_ingest`'s own scope (`agent_manager.py`'s tool-registry sync, `query_tools.py`'s
best-effort section-tree writer, `analysis_tools.py`'s change-coupling linker,
`kg_server.py`'s capability-declaration ingest, `governance_tools.py`'s veto write —
the last two are a single node+edge per call, not a real per-element loop, but the
others are genuine violations of the same class). Rewriting the whole population was
out of scope for the bulk_ingest fix that motivated this gate — mirroring
`check_event_loop_blocking.py`'s and `check_wiring.py`'s ratchet precedent (see their
docstrings), the pre-existing population is frozen in
`scripts/no_per_element_ingest_loop_baseline.txt` and the gate fails only on a *new*
site not already in that baseline (including a DELIBERATELY reintroduced one — that is
exactly what this gate exists to catch). Fixing a baselined site is always allowed and
shrinks the baseline on the next ``--update-baseline``.

Usage:
  python3 scripts/check_no_per_element_ingest_loop.py [PATH…]           # ratchet check
  python3 scripts/check_no_per_element_ingest_loop.py --update-baseline # freeze the current set

KNOWN SCOPE LIMIT (shared with `check_event_loop_blocking.py`): a per-element call
reached through a plain helper function three modules away, or a receiver whose base
name doesn't read as ``engine``/``client``/``backend``, is invisible to this AST
walk — a syntax scan, not a call graph.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts._git_scan import tracked_or_walked  # noqa: E402

SCOPE = ROOT / "agent_utilities" / "mcp"
BASELINE = ROOT / "scripts" / "no_per_element_ingest_loop_baseline.txt"

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

    def key(self) -> tuple[str, str, str]:
        """Ratchet identity: (relpath, function, label) — WITHOUT the line number,
        so an unrelated edit above the site doesn't spuriously invalidate the
        baseline (mirrors `check_event_loop_blocking.py`'s `Finding.key`)."""
        try:
            rel = str(self.path.resolve().relative_to(ROOT))
        except ValueError:
            rel = str(self.path)
        return (rel, self.func, self.label)


def scan_file(path: Path) -> list[Finding]:
    try:
        src = path.read_text(encoding="utf-8")
    except OSError:
        return []
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError:
        return []

    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    def _enclosing_loop(node: ast.AST) -> bool:
        cur = parents.get(node)
        while cur is not None:
            if isinstance(cur, (ast.For, ast.AsyncFor)):
                return True
            cur = parents.get(cur)
        return False

    def _enclosing_func_name(node: ast.AST) -> str:
        cur = parents.get(node)
        while cur is not None:
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return cur.name
            cur = parents.get(cur)
        return "<module>"

    findings: list[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        dotted = _dotted_name(node.func)
        if dotted is None:
            continue
        tail = dotted.split(".")[-1]
        if tail not in _INGEST_ATTR_SUFFIXES:
            continue
        if not _enclosing_loop(node):
            continue
        func_name = _enclosing_func_name(node)
        findings.append(
            Finding(
                path,
                node.lineno,
                func_name,
                f"per-element ingest call in a loop: {dotted}",
            )
        )
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
        "# Frozen baseline of check_no_per_element_ingest_loop.py candidate sites\n"
        "# (ratchet — burn down toward zero, B-11/GOC-68). Format: relpath\\tfunc\\tlabel.\n"
        "# A NEW site not listed here fails the gate — including a deliberately\n"
        "# reintroduced per-element ingest loop, which is exactly what this gate\n"
        "# exists to catch. Fixing a listed site (route it through batch_typed_mutations/\n"
        "# BatchUpdate or ingest_graph_slice/ApplyChangeEnvelopes instead) is always\n"
        "# allowed and should be followed by --update-baseline to shrink this file.\n"
        + body
        + "\n",
        encoding="utf-8",
    )


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if a != "--update-baseline"]
    update_baseline = "--update-baseline" in argv[1:]
    targets = [Path(a) for a in args] or [SCOPE]
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
            f"check_no_per_element_ingest_loop: baseline updated: "
            f"{len({f.key() for f in all_findings})} entries -> {BASELINE.name}"
        )
        return 0

    if not all_findings:
        print("check_no_per_element_ingest_loop: no candidate sites found.")
        return 0

    baseline = load_baseline()
    current_keys = {f.key() for f in all_findings}
    new_findings = [f for f in all_findings if f.key() not in baseline]

    if new_findings:
        print(
            f"check_no_per_element_ingest_loop: {len(new_findings)} NEW candidate "
            f"site(s) not in the frozen baseline ({BASELINE.name}):"
        )
        for finding in new_findings:
            print(f"  {finding}")
        print(
            "\nRoute the batch through the engine's native atomic primitive instead "
            "of a per-element loop — batch_typed_mutations/BatchUpdate for the light "
            "path, ingest_graph_slice/ApplyChangeEnvelopes when evidence/policy/an "
            "idempotency key is involved — or, if this is a genuine one-off exception, "
            "regenerate the baseline with --update-baseline and say why in the commit "
            "message. See script docstring."
        )
        return 1

    removed = len(baseline) - len(baseline & current_keys)
    msg = (
        f"check_no_per_element_ingest_loop: OK — no new candidate sites "
        f"({len(current_keys)} baselined"
    )
    print(msg + (f", {removed} fixed since baseline)." if removed else ")."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
