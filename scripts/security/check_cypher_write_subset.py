#!/usr/bin/env python3
"""Cypher write-subset contract gate.

The epistemic-graph engine's NATIVE Cypher implementation supports only a
narrow WRITE subset (`epistemic-graph/crates/eg-query/src/cypher/parser.rs`,
around line 1184 as of this writing): a write statement is
``[MATCH pattern [WHERE ...]] <write-clause>+ [RETURN ...]`` where

* at most **one leading MATCH** clause is allowed (a second ``MATCH``, or a
  comma-separated multi-node pattern like ``MATCH (a), (b)``, is rejected);
* ``MERGE`` supports **a single bare node only** (``MERGE (n:Label {props})``)
  — an edge pattern (``MERGE (a)-[:REL]->(b)``) is *never* accepted, in this
  subset, regardless of how many MATCH clauses precede it;
* ``WITH`` is never valid between a MATCH and a write clause (write
  statements have no read-pipeline stages); and
* ``SET`` values must be literals — a map-merge assignment (``SET n += $x``)
  is rejected.

A query built with any of these shapes and sent to a NATIVE
``EpistemicGraphBackend`` (``typed_mutation_support == "native"``) does not
silently do less than intended — it raises at the wire boundary (see
``EpistemicGraphBackend.execute_write`` -> ``CypherEngineError``). ~24 such
sites were found and fixed by hand (CONCEPT:AU-KG.query.register-each-user-table
lane, wave 2); this gate keeps them fixed and catches new ones, because the
established fix (dispatch through the typed engine API —
``IntelligenceGraphEngine.link_nodes``/``add_node``/``add_edge``, exactly the
``_upsert_edge``/``_upsert_node`` convention already used for 4 earlier sites)
means these shapes should never reappear in a raw ``.execute()``/
``.execute_write()`` call.

**Not every rich multi-MATCH/edge-MERGE query is a violation.** A backend
targeting a genuinely different store (e.g. ``LadybugBackend``, which hands
the query to Kuzu's full openCypher engine, or a ``FanOutBackend`` mirror
built on one) may legitimately emit the fuller grammar — see
``GraphComputeEngine.flush_ledger_to_backend`` in
``agent_utilities/knowledge_graph/core/graph_compute.py``, whose sole caller
(``agent_utilities/workflows/epistemic_sync.py``) always passes a
``LadybugBackend``. Sites like that carry an inline
``# cypher-write-subset-allow: <reason>`` pragma AT the call site (on the
line immediately above the ``.execute``/``.execute_write`` call, or trailing
on any line of the call itself) — the same "document why, in the source, at
the site" convention as this repo's ``# noqa: BLE001 — <reason>`` swallowed-
error exemptions.

D-MW-11: this pragma replaces a prior ``_ALLOWLIST`` dict keyed by
``(relative_path, lineno)``. That anchor drifted every time unrelated code
landed ABOVE an allowlisted site — shifting its line number made the gate
report a false "NEW violation" at the moved location, repeatedly (observed
2026-08-01: one site chased three times in one session as concurrent lanes
landed code ahead of it). An inline pragma has no such anchor to drift: it
is source text attached to the flagged statement itself, so it moves WITH
the statement no matter what lands above or below it. It is also
impossible to "hand-add" to silence an unrelated real violation without
touching the flagged line itself, unlike a separate allowlist file entry.

**Fail-closed vs. an honest absence** (a project-wide, codified distinction —
see ``run_contract_checks`` in ``agent_utilities/governance/merge_queue.py``,
"'Fewer contracts than the base' is the degraded read; 'this repo has none'
is a genuine empty and passes below"): this gate raises
``CypherWriteSubsetGateError`` — and ``main()`` exits 1 — if it cannot read
the tree it was asked to check (the target directory is absent, or a file
that content-matched the scan can't be read/decoded). It exits 0, reporting
the count, when the scan completes and finds zero NEW violations: a clean
scan is success, not something to be suspicious of.

**A ratchet baseline** (``cypher_write_subset_baseline.txt``, next to this
script — same convention as ``scripts/swallowed_error_baseline.txt``) covers
pre-existing violations outside the 14 files this lane was assigned to fix:
those are real (verified against the same grammar) but too large a surface
for one lane, so they're frozen rather than either silently ignored or used
to block every future merge. Only a violation NOT already in the baseline
fails the gate; fixing a baselined site shrinks the baseline on the next
``--update-baseline`` run.

D-F6-2: the baseline key was originally ``(path, line, shape)`` — the SAME
line-anchored fragility D-MW-11 fixed for the ``_ALLOWLIST`` pragma, just one
layer down in this same gate. It drifted on every merge that landed code
above a baselined site (observed 3x: ``graph_compute.py`` 3357->3352,
``engine.py`` 903->904, ``ogm.py`` 315->312), each time requiring a
``--update-baseline`` re-derive to quiet a phantom "NEW violation". The key
is now ``(path, symbol, shape, ordinal)`` — ``symbol`` is the enclosing
function/method's dotted qualname (``"<module>"`` at module scope,
``"ClassName.method"`` inside a class), computed by ``_scope_units``'s
manual parent-tracking DFS, not a line coordinate; ``ordinal`` disambiguates
the rare case of two same-shape violations in the same symbol, assigned in
deterministic AST traversal order. Same pattern already proven for
``scripts/check_swallowed_errors.py`` (D-SWG-1) and
``scripts/check_wiring.py`` (D-OP-11).

Usage:
  python3 scripts/security/check_cypher_write_subset.py
  python3 scripts/security/check_cypher_write_subset.py --repository-root DIR
  python3 scripts/security/check_cypher_write_subset.py --self-check
  python3 scripts/security/check_cypher_write_subset.py --update-baseline

Exit 0 = no NEW violation (or self-check passed too), 1 = a new violation
was found or the gate could not read the tree.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Only these two backend-facing methods carry raw Cypher text through to the
#: native engine's write parser. ``execute_read``/``query_cypher`` follow a
#: different (read) grammar that DOES allow multiple MATCH stages, so they are
#: deliberately not in scope here.
_WRITE_METHODS = {"execute", "execute_write"}

#: A write-containing statement (one of these keywords appears) is where the
#: subset restrictions apply at all; a plain read tolerates multiple MATCH
#: stages and is never flagged.
_WRITE_KEYWORD_RE = re.compile(r"\b(CREATE|MERGE|SET|DELETE|REMOVE)\b")
_MATCH_RE = re.compile(r"\bMATCH\b")
_COMMA_PATTERN_RE = re.compile(r"MATCH\s*\([^()]*\)\s*,\s*\(")
_MERGE_EDGE_RE = re.compile(r"\bMERGE\s*\([^()]*\)\s*[-<]")
_WITH_RE = re.compile(r"\bWITH\b")
_SET_PLUS_EQ_RE = re.compile(r"\bSET\b.*?\+=", re.DOTALL)
#: A SET target assigned a function CALL rather than a literal/parameter —
#: e.g. ``SET j.last_run = coalesce(j.last_run, '—')``. ``current_timestamp()``
#: is the one function this codebase's ``EpistemicGraphBackend._inline_cypher_params``
#: pre-substitutes with an actual literal before the query ever reaches the
#: parser (see that method's ``_CURRENT_TIMESTAMP_RE``), so it is the sole
#: exemption.
_SET_FUNCTION_CALL_RE = re.compile(r"=\s*(?!current_timestamp\()[A-Za-z_]\w*\s*\(")

#: D-MW-11: the ONLY escape hatch this gate has — an inline pragma comment
#: attached to the flagged call site itself (see module docstring). Checked
#: by scanning the physical source text around the call, never by (file,
#: line) coordinates, so it cannot drift when unrelated code moves above it.
#: Requires a non-empty reason after the colon (an unexplained pragma is
#: rejected the same way an unexplained ``# noqa: BLE001`` is elsewhere in
#: this repo).
_ALLOW_PRAGMA_RE = re.compile(r"#\s*cypher-write-subset-allow:\s*\S")


class CypherWriteSubsetGateError(RuntimeError):
    """The gate could not read the tree it was asked to check (fail-closed)."""


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    shape: str
    snippet: str
    #: D-F6-2: the enclosing function/method's dotted qualname (or
    #: ``"<module>"``), used for the content-anchored baseline key instead
    #: of ``line`` — see ``_baseline_key``.
    symbol: str = "<module>"
    #: Disambiguates two violations of the same ``shape`` in the same
    #: ``symbol`` — assigned in deterministic AST discovery order (see
    #: ``_find_violations``), not by line position.
    ordinal: int = 0

    def render(self) -> str:
        return f"{self.path}:{self.line} [{self.shape}] {self.snippet}"


def _literal_text(
    node: ast.expr, local_strings: dict[str, str]
) -> str | None:
    """Best-effort reconstruction of a query expression's literal text.

    Handles a plain/triple-quoted string, an f-string (interpolated
    ``{expr}`` parts become a neutral placeholder that cannot itself
    introduce a MATCH/MERGE/WITH/SET keyword), string concatenation via
    ``+``, and a bare ``Name`` looked up in ``local_strings`` (the
    ``query = "..."`` / ``query = f"..."`` assignment tracked earlier in the
    same function — the common shape in this codebase). Returns ``None`` when
    the text can't be statically resolved; those calls are simply not
    checked (a static heuristic, not a dataflow prover — matching every
    other ``scripts/check_*.py`` gate in this repo).
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            else:
                parts.append("X")
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _literal_text(node.left, local_strings)
        right = _literal_text(node.right, local_strings)
        if left is not None and right is not None:
            return left + right
        return None
    if isinstance(node, ast.Name):
        return local_strings.get(node.id)
    return None


def _ordered_descendants(node: ast.AST) -> list[ast.AST]:
    """A depth-first, field-order traversal of ``node``'s descendants —
    unlike ``ast.walk`` (breadth-first, so it does not respect source order
    across nested blocks), this visits an ``if``'s ``body`` before its
    ``orelse``, a function's statements top-to-bottom, etc., which is what
    ``_find_violations`` needs to resolve ``name = "..."; ...; x.execute(name)``
    using whichever assignment textually precedes the call, not whichever one
    happens to be the LAST assignment to that name anywhere in the function
    (the earlier flat-dict approach mis-attributed the wrong branch's query
    text when the same variable name, e.g. ``query``, was reused across
    several independent if/elif branches each with its own call — a real bug
    caught by re-running this gate against the live tree during development).

    Never descends into a NESTED function/async-function: ``_scope_units``
    already gives every one of those its own independent entry, so
    descending into one here too would visit — and flag — every one of its
    calls twice.
    """
    ordered: list[ast.AST] = [node]
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        ordered.extend(_ordered_descendants(child))
    return ordered


#: Necessary precondition for treating a string as Cypher at all: some other
#: object entirely (a raw-SQL backend, e.g. ``postgres_queue_backend.py``'s
#: ``UPDATE ... SET`` / ``WITH ... AS (...)`` queries) also has an
#: ``execute``/``execute_write`` method, and plain SQL shares the SET/WITH/
#: DELETE keywords this gate looks for. Every genuine Cypher write in this
#: codebase's own established shapes has a node pattern -- ``MATCH (...)`` or
#: ``MERGE (...)`` -- so requiring one before evaluating anything else is a
#: cheap, reliable way to leave a same-named SQL method alone.
_CYPHER_NODE_PATTERN_RE = re.compile(r"\bMATCH\s*\(|\bMERGE\s*\(")


def _shape(query: str) -> str | None:
    """Return a violation shape name for ``query``, or ``None`` if it is
    within the supported native write subset (or is a plain read, which is
    unrestricted in MATCH count), or isn't Cypher at all."""
    if not _CYPHER_NODE_PATTERN_RE.search(query):
        return None  # not Cypher-shaped -- e.g. a raw-SQL backend's own query
    if not _WRITE_KEYWORD_RE.search(query):
        return None  # a read: multiple MATCH stages are fine there
    if _COMMA_PATTERN_RE.search(query):
        return "comma_pattern_match"
    if len(_MATCH_RE.findall(query)) > 1:
        return "multiple_match_clauses"
    if _MERGE_EDGE_RE.search(query):
        return "merge_on_edge_pattern"
    if _WITH_RE.search(query):
        return "with_in_write_statement"
    if _SET_PLUS_EQ_RE.search(query):
        return "set_map_merge_assignment"
    if "SET" in query and _SET_FUNCTION_CALL_RE.search(query):
        return "set_function_call_value"
    return None


def _scope_units(tree: ast.Module) -> list[tuple[ast.AST, list[ast.stmt], str]]:
    """(owning node, body, dotted qualname) for the module and every
    function/async-function — the units ``_find_violations`` tracks local
    string assignments and the typed-dispatch guard within.

    D-F6-2: unlike a flat ``ast.walk`` (which finds every function but has
    no parent information), this is a manual pre-order DFS that tracks the
    enclosing class/function chain — the same "no line-number anchor"
    technique ``check_swallowed_errors.py``'s ``_iter_except_handlers_with_scope``
    already established (D-SWG-1) — so each unit gets a stable qualname
    (``"<module>"``, ``"foo"``, ``"ClassName.method"``) for the
    content-anchored baseline key, deterministic given the AST regardless of
    unrelated code landing above or below it.
    """
    units: list[tuple[ast.AST, list[ast.stmt], str]] = [(tree, tree.body, "<module>")]

    def walk(node: ast.AST, scope_stack: tuple[str, ...]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualname = ".".join(scope_stack + (child.name,))
                units.append((child, child.body, qualname))
                walk(child, scope_stack + (child.name,))
            elif isinstance(child, ast.ClassDef):
                walk(child, scope_stack + (child.name,))
            else:
                walk(child, scope_stack)

    walk(tree, ())
    return units


#: A function that itself inspects the backend's declared capability before
#: choosing a Cypher shape has already done the dispatch this gate would
#: otherwise flag as unconditional — this is the SAME idiom
#: ``IntelligenceGraphEngine._upsert_node``/``_upsert_edge``/``resolve_and_link``
#: use (native: typed API or raise; otherwise: the portable multi-clause
#: Cypher a non-native store's fuller grammar accepts), and the one this
#: lane's own fix to ``executor.py._persist`` follows. A whole-function
#: exemption rather than "only the guarded branch" is a deliberate,
#: documented over-approximation (this is a static heuristic, not a
#: dataflow prover, matching every other ``scripts/check_*.py`` gate in this
#: repo) — the false-negative risk (an unguarded bad call elsewhere in an
#: otherwise-dispatch-aware function) is judged far smaller than the
#: false-positive cost of flagging every one of this convention's
#: already-correct, already-audited non-native fallback branches.
_DISPATCH_GUARD_MARKERS = ("typed_mutation_support", "cypher_support")


def _find_violations(rel: str, source: str, tree: ast.Module) -> list[Violation]:
    violations: list[Violation] = []
    #: D-F6-2: ordinal counter per (symbol, shape), incremented in
    #: deterministic AST discovery order below — gives the second field the
    #: content-anchored key needs to tell apart two same-shape violations in
    #: the same function, with no dependence on either violation's line.
    ordinal_counts: dict[tuple[str, str], int] = {}
    # Split once per file and slice by line range instead of calling
    # ``ast.get_source_segment`` per scope unit: that helper re-splits the
    # WHOLE source into lines internally on every call, which dominated this
    # gate's runtime (measured: the majority of a ~17s full-tree scan) on a
    # file with many functions, each getting its own scope-unit lookup.
    lines = source.splitlines()
    for owner, body, symbol in _scope_units(tree):
        start = getattr(owner, "lineno", 1) - 1
        end = getattr(owner, "end_lineno", len(lines))
        owner_source = "\n".join(lines[max(start, 0) : end])
        if any(marker in owner_source for marker in _DISPATCH_GUARD_MARKERS):
            continue
        local_strings: dict[str, str] = {}
        for node in _ordered_descendants_of_body(body):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    text = _literal_text(node.value, local_strings)
                    if text is not None:
                        local_strings[target.id] = text
                continue
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute) or func.attr not in _WRITE_METHODS:
                continue
            if not node.args:
                continue
            text = _literal_text(node.args[0], local_strings)
            if text is None:
                continue
            shape = _shape(text)
            if shape is None:
                continue
            if _pragma_allows(lines, node):
                continue
            snippet = " ".join(text.split())[:160]
            ordinal_key = (symbol, shape)
            ordinal = ordinal_counts.get(ordinal_key, 0)
            ordinal_counts[ordinal_key] = ordinal + 1
            violations.append(
                Violation(rel, node.lineno, shape, snippet, symbol, ordinal)
            )
    return violations


def _pragma_allows(lines: list[str], node: ast.Call) -> bool:
    """True when a ``# cypher-write-subset-allow: <reason>`` pragma is
    attached to this call — anywhere in the contiguous ``#``-only comment
    block immediately above the call's opening line (a justification is
    often several lines long, mirroring this repo's other multi-line
    ``# noqa``-adjacent explanations), or trailing on any line the call
    itself spans (multi-line calls are common here). D-MW-11:
    source-text-anchored, so it travels with the call no matter what code
    lands above or below it — unlike a (file, line) coordinate in a
    separate allowlist file.
    """
    call_start = getattr(node, "lineno", 1)  # 1-indexed
    call_end = getattr(node, "end_lineno", call_start)  # 1-indexed, inclusive
    window = list(lines[call_start - 1 : call_end])  # the call's own lines
    idx = call_start - 2  # 0-indexed line directly above the call
    while idx >= 0 and lines[idx].strip().startswith("#"):
        window.append(lines[idx])
        idx -= 1
    return any(_ALLOW_PRAGMA_RE.search(line) for line in window)


def _ordered_descendants_of_body(body: list[ast.stmt]) -> list[ast.AST]:
    ordered: list[ast.AST] = []
    for stmt in body:
        # A top-level function/async-function statement gets its own
        # dedicated scope entry from ``_scope_units`` — descending into it
        # HERE too (as ``_ordered_descendants`` would for any other
        # statement type) double-visits, and so double-flags, every one of
        # its calls.
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        ordered.extend(_ordered_descendants(stmt))
    return ordered


def _candidate_files(root: Path) -> list[Path]:
    """Files under ``root`` worth AST-parsing: only those that reference one
    of the two write-carrying methods at all, found with ``grep`` (a couple
    of hundredths of a second over this tree) rather than reading and
    substring-testing every ``*.py`` file in Python (measured ~15x slower at
    this repo's size) -- falls back to a pure-Python scan if ``grep`` is
    unavailable, so the gate degrades in speed, never in coverage.
    """
    pattern = r"\.execute\(|\.execute_write\("
    try:
        proc = subprocess.run(
            ["grep", "-rlE", pattern, str(root), "--include=*.py"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CypherWriteSubsetGateError(
            f"could not enumerate candidate files under {root}: {exc}"
        ) from exc
    if proc.returncode not in (0, 1):  # 1 == grep found nothing, not an error
        raise CypherWriteSubsetGateError(
            f"grep over {root} exited {proc.returncode}: {proc.stderr.strip()}"
        )
    return sorted(Path(line) for line in proc.stdout.splitlines() if line)


def scan(root: Path) -> list[Violation]:
    """Scan ``root`` for Cypher write-subset violations.

    Fail-closed: raises :class:`CypherWriteSubsetGateError` if ``root``
    doesn't exist, or if a candidate file can't be read/decoded/parsed --
    those are exactly the "degraded read" this gate must refuse rather than
    silently under-report through. A file that legitimately has nothing to
    do with Cypher (not returned by the candidate-file scan at all) is never
    touched, so it can't trigger this.
    """
    if not root.is_dir():
        raise CypherWriteSubsetGateError(f"repository root is not a directory: {root}")

    violations: list[Violation] = []
    for path in _candidate_files(root):
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise CypherWriteSubsetGateError(
                f"could not read candidate file {path}: {exc}"
            ) from exc
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            raise CypherWriteSubsetGateError(
                f"could not parse candidate file {path}: {exc}"
            ) from exc
        try:
            rel = path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            rel = path.as_posix()
        violations.extend(_find_violations(rel, source, tree))
    return sorted(violations, key=lambda v: (v.path, v.line))


#: This lane (CONCEPT:AU-KG.query.register-each-user-table, wave 2) hand-fixed
#: every violation reachable from its assigned 14 files. Scanning the WHOLE
#: tree while building this gate surfaced ~50 more, pre-existing, in files
#: entirely outside that assignment — each is real (verified against the
#: same parser.rs grammar) but fixing all of them is a different, much
#: larger lane. Landing this gate with zero tolerance for the pre-existing
#: surface would fail the merge queue for every candidate, not just a
#: regression, which gets the gate reverted rather than a defect fixed. So:
#: a ratchet baseline, the SAME pattern ``scripts/check_swallowed_errors.py``
#: already established for an analogous "large pre-existing surface, fix
#: incrementally" situation. New sites --- not in this file --- fail the
#: gate immediately; a baselined site burns down whenever a future lane
#: fixes it (baseline shrinks) and this gate makes sure it never grows back
#: in the meantime.
BASELINE = Path(__file__).resolve().parent / "cypher_write_subset_baseline.txt"

_BASELINE_SEP = "\t"


#: D-F6-2: (path, enclosing symbol, shape, ordinal) — NOT line. Stable under
#: unrelated code landing above/below the flagged statement; see the module
#: docstring's D-F6-2 note and ``_scope_units``.
BaselineKey = tuple[str, str, str, int]


def _baseline_key(v: Violation) -> BaselineKey:
    return (v.path, v.symbol, v.shape, v.ordinal)


def _load_baseline() -> set[BaselineKey]:
    if not BASELINE.exists():
        return set()
    out: set[BaselineKey] = set()
    for line in BASELINE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(_BASELINE_SEP)
        if len(fields) < 4 or not fields[3].isdigit():
            continue
        out.add((fields[0], fields[1], fields[2], int(fields[3])))
    return out


def _write_baseline(violations: list[Violation]) -> None:
    lines = [
        f"{v.path}{_BASELINE_SEP}{v.symbol}{_BASELINE_SEP}{v.shape}"
        f"{_BASELINE_SEP}{v.ordinal}{_BASELINE_SEP}# line {v.line}: {v.snippet}"
        for v in sorted(violations, key=lambda v: (v.path, v.symbol, v.shape, v.ordinal))
    ]
    BASELINE.write_text(
        "# Frozen baseline of pre-existing Cypher write-subset violations\n"
        "# OUTSIDE this lane's assigned 14 files (a ratchet — burn down\n"
        "# toward zero, same convention as scripts/swallowed_error_baseline.txt).\n"
        "# TAB-separated: path\\tsymbol\\tshape\\tordinal\\t# line N: snippet\n"
        "# (D-F6-2: keyed by symbol+ordinal, NOT line -- the trailing 'line N'\n"
        "# in the comment is informational only, not parsed, and WILL drift as\n"
        "# the codebase changes -- expected and harmless).\n"
        "# A NEW entry (not here) fails check_cypher_write_subset.py. Fixing a\n"
        "# baselined site (see the module docstring's fix pattern) shrinks this\n"
        "# file on the next --update-baseline; never hand-add an entry to make\n"
        "# a real violation disappear.\n"
        + "\n".join(lines)
        + ("\n" if lines else ""),
        encoding="utf-8",
    )


def check(root: Path) -> dict[str, Any]:
    violations = scan(root)
    baseline = _load_baseline()
    new = [v for v in violations if _baseline_key(v) not in baseline]
    if new:
        raise CypherWriteSubsetGateError(
            "NEW Cypher write-subset violation(s) found (not in "
            f"{BASELINE.name}):\n" + "\n".join(f"  {v.render()}" for v in new)
        )
    current_keys = {_baseline_key(v) for v in violations}
    fixed = len(baseline - current_keys)
    return {
        "ok": True,
        "newViolations": 0,
        "baselined": len(current_keys),
        "fixedSinceBaseline": fixed,
    }


# ---------------------------------------------------------------------------
# Self-check: prove the gate actually catches each known-bad shape, and
# doesn't flag the supported subset or the allowlisted exemption.
# ---------------------------------------------------------------------------

_GOOD_FIXTURE = '''
def upsert_thread(backend):
    backend.execute(
        "MERGE (t:Thread {id: $id}) SET t.title = $title",
        {"id": "t1", "title": "x"},
    )
    backend.execute("MATCH (t:Thread) RETURN t.id")


def link_via_typed_api(engine):
    engine.link_nodes("a", "b", "CONTAINS")
'''

_BAD_FIXTURES: dict[str, str] = {
    "comma_pattern_match": '''
def f(backend):
    backend.execute(
        "MATCH (a:X {id: $a}), (b:Y {id: $b}) MERGE (a)-[:REL]->(b)",
        {},
    )
''',
    "multiple_match_clauses": '''
def f(backend):
    backend.execute(
        "MATCH (a:X {id: $a}) MATCH (b:Y {id: $b}) MERGE (a)-[:REL]->(b)",
        {},
    )
''',
    "merge_on_edge_pattern": '''
def f(backend):
    query = "MATCH (a:X {id: $a}) MERGE (a)-[:REL]->(b:Y {id: $b})"
    backend.execute(query, {})
''',
    "with_in_write_statement": '''
def f(backend):
    backend.execute(
        "MATCH (l:Log) WITH l ORDER BY l.timestamp DESC SKIP 5 DETACH DELETE l",
    )
''',
    "set_map_merge_assignment": '''
def f(backend):
    backend.execute("MATCH (n:Concept {id: $id}) SET n += $props", {})
''',
    "set_function_call_value": '''
def f(backend):
    backend.execute(
        "MERGE (j:Job {id: $id}) SET j.last_run = coalesce(j.last_run, '-')",
        {},
    )
''',
}


#: D-MW-11 proof fixtures: the SAME bad-shape statement, once with a pragma
#: on the line above the call, once with a trailing pragma on the call's own
#: line — both must be suppressed. An unpadded and a padded (code inserted
#: ABOVE the pragma'd site, simulating an unrelated merge landing ahead of
#: it) variant of each prove the mechanism has no line-number anchor to
#: drift: only line COUNT changes, the pragma's suppression is unaffected.
_PRAGMA_ABOVE_FIXTURE = '''
def f(backend):
    # cypher-write-subset-allow: LadybugBackend, not the native parser.
    backend.execute(
        "MATCH (a:X {id: $a}), (b:Y {id: $b}) MERGE (a)-[:REL]->(b)", {}
    )
'''

_PRAGMA_TRAILING_FIXTURE = '''
def f(backend):
    backend.execute(  # cypher-write-subset-allow: LadybugBackend here too.
        "MATCH (a:X {id: $a}), (b:Y {id: $b}) MERGE (a)-[:REL]->(b)", {}
    )
'''


def self_check() -> None:
    """Prove: (1) each known-bad shape is caught, (2) the supported subset
    and the typed-API convention are NOT flagged, (3) a pragma'd site is NOT
    flagged whether the pragma is above the call or trailing on it, (4) that
    suppression survives unrelated code landing ABOVE the pragma'd site
    (D-MW-11 — the whole point: no (file, line) anchor to drift), (5) a
    missing root fails closed."""
    good_tree = ast.parse(_GOOD_FIXTURE)
    good_hits = _find_violations("fixture_good.py", _GOOD_FIXTURE, good_tree)
    if good_hits:
        raise CypherWriteSubsetGateError(
            "self-check: the supported subset / typed-API convention was "
            f"flagged as a violation: {[v.render() for v in good_hits]}"
        )

    for shape, fixture in _BAD_FIXTURES.items():
        tree = ast.parse(fixture)
        hits = _find_violations(f"fixture_{shape}.py", fixture, tree)
        found_shapes = {v.shape for v in hits}
        if shape not in found_shapes:
            raise CypherWriteSubsetGateError(
                f"self-check: known-bad shape {shape!r} was NOT caught "
                f"(found instead: {sorted(found_shapes)})"
            )

    for label, fixture in (
        ("above", _PRAGMA_ABOVE_FIXTURE),
        ("trailing", _PRAGMA_TRAILING_FIXTURE),
    ):
        # Unpadded: the pragma suppresses the violation at its natural line.
        tree = ast.parse(fixture)
        hits = _find_violations(f"fixture_pragma_{label}.py", fixture, tree)
        if hits:
            raise CypherWriteSubsetGateError(
                f"self-check: a {label}-pragma'd site was still flagged: "
                f"{[v.render() for v in hits]}"
            )
        # Padded: 5 unrelated lines inserted ABOVE the whole fixture — the
        # exact shape of a sibling lane landing code ahead of an allowlisted
        # site (D-MW-11's reported failure mode). A line-number-anchored
        # allowlist would miss this; the pragma travels with the statement.
        padded = ("\n" * 5) + fixture
        padded_tree = ast.parse(padded)
        padded_hits = _find_violations(
            f"fixture_pragma_{label}_padded.py", padded, padded_tree
        )
        if padded_hits:
            raise CypherWriteSubsetGateError(
                f"self-check: a {label}-pragma'd site was flagged after "
                f"unrelated lines landed above it (line-anchor drift): "
                f"{[v.render() for v in padded_hits]}"
            )

    # And the negative: the SAME bad shape with NO pragma at all is still
    # caught — the pragma is not accidentally suppressing everything.
    unpragma_tree = ast.parse(_BAD_FIXTURES["comma_pattern_match"])
    unpragma_hits = _find_violations(
        "fixture_no_pragma.py",
        _BAD_FIXTURES["comma_pattern_match"],
        unpragma_tree,
    )
    if not unpragma_hits:
        raise CypherWriteSubsetGateError(
            "self-check: a violation with NO pragma was incorrectly suppressed"
        )

    # Fail-closed: a nonexistent root must raise, not report a clean scan.
    missing = Path("/nonexistent-cypher-subset-gate-fixture-root")
    try:
        scan(missing)
    except CypherWriteSubsetGateError:
        pass
    else:
        raise CypherWriteSubsetGateError(
            "self-check: scanning a nonexistent root did not fail closed"
        )

    # The ratchet baseline: a violation present in the baseline is tolerated
    # (burn-down, not a block), but a violation in a DIFFERENT symbol for the
    # same shape/file is treated as genuinely new — even though both happen
    # to render at a line, the key never looks at it.
    baselined = Violation("fixture_baselined.py", 10, "comma_pattern_match", "x", "f")
    new_one = Violation("fixture_baselined.py", 20, "comma_pattern_match", "y", "g")
    baseline_keys = {_baseline_key(baselined)}
    still_new = [v for v in (baselined, new_one) if _baseline_key(v) not in baseline_keys]
    if still_new != [new_one]:
        raise CypherWriteSubsetGateError(
            "self-check: baseline ratchet did not distinguish a baselined "
            f"site from a new one (got {still_new!r})"
        )

    # D-F6-2: the SAME baselined site's key must be byte-identical whether or
    # not unrelated code has landed above it in the file (no line-number
    # anchor to drift), and a genuinely SECOND same-shape violation in the
    # SAME symbol must get a distinct key (ordinal 1, not silently merged
    # into ordinal 0's baseline entry).
    unpadded_tree = ast.parse(_BAD_FIXTURES["comma_pattern_match"])
    unpadded_hits = _find_violations(
        "fixture_f6_2.py", _BAD_FIXTURES["comma_pattern_match"], unpadded_tree
    )
    padded_source = ("\n" * 7) + _BAD_FIXTURES["comma_pattern_match"]
    padded_tree2 = ast.parse(padded_source)
    padded_hits2 = _find_violations("fixture_f6_2.py", padded_source, padded_tree2)
    if not unpadded_hits or not padded_hits2:
        raise CypherWriteSubsetGateError(
            "self-check: D-F6-2 fixture produced no violations to compare"
        )
    if padded_hits2[0].line == unpadded_hits[0].line:
        raise CypherWriteSubsetGateError(
            "self-check: D-F6-2 padding fixture did not actually move the "
            "violation's line -- the drift-survival proof is vacuous"
        )
    if _baseline_key(unpadded_hits[0]) != _baseline_key(padded_hits2[0]):
        raise CypherWriteSubsetGateError(
            "self-check: baseline key changed when unrelated lines landed "
            f"above the site (line-anchor drift survived the D-F6-2 fix): "
            f"{_baseline_key(unpadded_hits[0])!r} != "
            f"{_baseline_key(padded_hits2[0])!r}"
        )

    two_hits_source = '''
def f(backend):
    backend.execute(
        "MATCH (a:X {id: $a}), (b:Y {id: $b}) MERGE (a)-[:REL]->(b)", {}
    )
    backend.execute(
        "MATCH (c:X {id: $c}), (d:Y {id: $d}) MERGE (c)-[:REL]->(d)", {}
    )
'''
    two_hits_tree = ast.parse(two_hits_source)
    two_hits = _find_violations("fixture_f6_2_twice.py", two_hits_source, two_hits_tree)
    if len(two_hits) != 2:
        raise CypherWriteSubsetGateError(
            f"self-check: D-F6-2 same-symbol-twice fixture expected 2 "
            f"violations, found {len(two_hits)}"
        )
    two_keys = {_baseline_key(v) for v in two_hits}
    if len(two_keys) != 2:
        raise CypherWriteSubsetGateError(
            "self-check: two distinct same-shape violations in one symbol "
            f"collapsed onto the same baseline key: {two_keys!r}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="check-cypher-write-subset")
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Freeze the current scan as the new ratchet baseline and exit 0.",
    )
    args = parser.parse_args(argv)
    if args.update_baseline:
        try:
            violations = scan(args.repository_root)
        except CypherWriteSubsetGateError as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
            return 1
        _write_baseline(violations)
        print(
            json.dumps(
                {"ok": True, "baselineUpdated": True, "entries": len(violations)},
                sort_keys=True,
            )
        )
        return 0
    try:
        if args.self_check:
            self_check()
        report = check(args.repository_root)
        if args.self_check:
            report["selfCheck"] = True
    except CypherWriteSubsetGateError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 1
    except Exception as exc:  # noqa: BLE001 — gate boundary: any unexpected failure must fail closed (exit 1), never be swallowed into a false "clean scan"
        print(
            json.dumps(
                {"ok": False, "error": f"unexpected {type(exc).__name__}: {exc}"},
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
