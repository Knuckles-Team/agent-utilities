#!/usr/bin/env python3
"""Liveness / dead-pathway gate — CENSUS + DIFF-SCOPED, NO BASELINE, NO CAPS.

(CONCEPT:AU-KG.maintenance.periodic-code-health consumer.)

WHAT THIS GATE MEASURES
-----------------------
Wire-First's static ``check_wiring.py`` catches modules with no import path.
This gate adds the layers it can't: the **typed-seam / contract-drift** scan
(public functions returning an untyped ``dict``/``list[dict]`` — the seam where
a producer writes ``_score`` and a consumer reads ``score`` and silently gets
``0.00``), the **facade** layer ("invoked but fake": a live-surface handler that
returns a canned payload while doing no real work) and, when a coverage report
is present, the **never-executed** layer.

It does NOT duplicate the raw detector — that lives once in the code-enhancer
skill (``universal_skills/.../code-enhancer/scripts/analyze_liveness.py``). This
gate locates it, runs it over ``agent_utilities/``, and RECONCILES its raw
``orphan_modules``/``dead_definitions`` findings
(``scripts/liveness_reconciler.py``) before reporting — the vendored analyzer
has a confirmed resolution bug (it compares a module path computed relative to
``agent_utilities/`` against import statements captured with the full
``agent_utilities.`` prefix, so this repo's dominant absolute-import style
almost never matches) plus a blind spot for string-dispatched registries
(entry-points, ``getattr`` registries, ``__getattr__`` lazy-import maps). See
``plans/graph-os-completion-program/designs/DEAD-CODE-INTENT-RECOVERY.md``.

WHY THE CAPS TABLE IS GONE (wD1 / WD1-GATE-02, 2026-08-27)
----------------------------------------------------------
This gate used to compare every category against a seven-integer ``CAPS`` table
and fail when a count exceeded its entry, telling the committer to "raise the
relevant CAPS entry". That is a ratchet — the mechanism this project bans
outright (NO RATCHETS: findings are debt to burn down, never to freeze) — and
moving it into the gate's own source did not change what it was. It has been
deleted, together with the count-vs-cap comparison it fed.

Deleting a ratchet is two jobs; doing only the first bricks every commit. What
replaces it is below, and it is the shape the two in-repo precedents already
established: ``scripts/check_complexity.py`` (absolute, whole-repo, census —
and it documents why ``--baseline``/``--write`` were retired and made to fail
loud) plus ``scripts/check_complexity_staged.py`` (diff-scoped, recomputed live
from git, nothing frozen on disk) as the thing that actually runs pre-commit.

★ THE MEASURED REASON A WHOLE-REPO COUNT CANNOT BE THE SIGNAL HERE
------------------------------------------------------------------
The cap that was red on ``main`` was ``facade_handlers`` at 109 against 98. It
was investigated before this gate was redesigned, and NOT ONE of those 11 is a
new dead pathway. Full accounting against the cap-setting tree (3b7186be2):

* ``mcp/tools/intent_tools.py``  **+12** — pure DOUBLE COUNTING. The analyzer
  walks every ``FunctionDef`` in the module, then walks each one's whole
  subtree, so a branch inside a NESTED function is attributed to the nested
  function AND to every enclosing one. ``2b7719220`` ("dispatch_intent CCN 82
  -> 23") extracted ``_intake`` / ``_restore_from_plan_ref`` /
  ``_select_tool_and_action`` / ``_policy_gate`` as CLOSURES. Measured after the
  refactor: ``dispatch_intent`` reports branches at lines
  {1185,1195,1225,1261,1360,1373,1391,1562,1568,1576,1582,1593} — which is
  exactly the union of the four helpers' own branch lines (2+2+3+5 = 12). Every
  finding is labelled twice.
* ``server/routers/commands.py`` **+8** — ``039c30d04`` ("execute_slash_command
  CCN 126 -> 4") split the mega-dispatcher into ``_cmd_*`` helpers, which turned
  ``response_md = "<empty-state text>"`` into ``return "<the same text>"``.
  Every one of the nine newly-flagged strings ("Graph backend not active — no
  live counts available.", "…no knowledge bases available.", "…cannot enqueue
  ingestion.", "No registered specialists or live dispatch workers.", …) is
  present VERBATIM in the cap-setting tree. No behaviour changed.
* ``analysis_tools`` **−5**, ``query_tools`` **−3**, ``write_ingest_tools``
  **−1** — the same effect in reverse: ``f0655146c`` moved
  ``_run_analysis_action`` out of ``register_analysis_tools``, deleting nine
  double-counted labels.

Net +11, from three complexity-collapse refactors, with zero new fake code. So
``facade_handlers`` is **not invariant under function extraction**: it counts
``(enclosing function, branch ordinal)`` labels, nesting multiplies them and
extraction re-partitions them. A whole-repo absolute count of it is instrument
noise in a repository that is being refactored on purpose, and gating a commit
on that number is gating on the instrument, not the code.

WHAT THE GATE DOES NOW
----------------------
1. **CENSUS, on every run, unconditionally.** The real number for all seven
   categories is printed, reconciled, with the worst findings named. Nothing is
   frozen, nothing is compared to a stored value, and the census NEVER fails the
   build. This is the no-ratchet policy in code: the true numbers are on screen
   every time, so the debt cannot become invisible.
2. **DIFF-SCOPED enforcement — the teeth.** For each ``agent_utilities/**.py``
   the commit touches, the "invoked but fake" layers (facade handlers +
   placeholder markers) are recomputed for BOTH the ``HEAD`` blob and the
   ``:index`` blob, live, from git. A finding that is present in the index and
   absent from HEAD fails the commit. Nothing is written to disk, no count is
   frozen, no finding is marked accepted; pre-existing debt stays visible in the
   census and must still be burned down deliberately. Identity is the
   CONTENT-HASH of the flagged code (the unparsed branch / except-handler /
   function body, or the analyzer's own content hash for a placeholder line),
   NOT ``symbol@ordinal`` — precisely because ``symbol@ordinal`` is what
   produced the 11 phantom findings dissected above. Under this identity the
   same three refactors are clean.
3. **Absolute invariants, zero slack.** ``never_executed`` must be 0. That is a
   real invariant, not "the number we happen to be at", and it has no slack to
   freeze. Any other absolute threshold is opt-in per invocation via
   ``--max category=N``, with NO default baked into this file, so a CI job can
   drive one category down deliberately without anything being frozen in source.
4. **Deferral expiry** (GOC-68: "a bare ratchet lets deferral become
   permanent"). ``scripts/liveness_deferred.tsv`` entries must carry an owner
   plus a review-by date or a PERMANENT justification; a past-due entry fails.
   Unchanged, and independent of everything above.

Reuse, not reimplementation: the per-file detectors are IMPORTED from the same
vendored ``analyze_liveness.py`` the census shells out to (one capability, one
entrypoint). If a required detector surface is missing the gate exits 2 — a
gate that could not run has not found nothing. The private stable-ID helper is
the one optional surface, with an exact content-hash compatibility adapter for
older universal-skills releases.

AMBIENT GIT ENVIRONMENT
-----------------------
git exports ``GIT_DIR``/``GIT_INDEX_FILE``/``GIT_WORK_TREE`` into every hook
subprocess. Inherited blindly by a whole-tree scanner they silently re-root path
resolution — that is BUG-180, confirmed live against THIS gate, which reported a
false ``orphan_modules: 4 -> 196`` regression with zero source changes. So the
two halves are treated differently and deliberately:

* the git blob reads MUST see the index being committed, so they keep the
  ambient variables and instead always run from the resolved toplevel with
  repo-relative paths (never ``git -C <subdir>``) — the reasoning
  ``check_complexity_staged.py`` already established;
* the whole-tree census runs with those variables stripped
  (``scripts/_git_subprocess_env.py``), in-process and in the analyzer
  subprocess, so no tree walk can be re-rooted.

If the code-enhancer skill is not installed the census cannot run; the gate says
NOT ENFORCED, loudly, and exits 0 rather than blocking CI (see ``_no_analyzer``).

Exit codes: 0 pass, 1 violation, 2 the gate could not run (an ENVIRONMENT fact,
never reported as a clean pass).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TARGET = REPO / "agent_utilities"

#: Reported by the census, in this order. No numbers live here — by policy.
CATEGORIES = (
    "orphan_modules",
    "dead_definitions",
    "never_executed",
    "untyped_seams",
    "orphan_read_keys",
    "facade_handlers",
    "placeholder_markers",
)

#: The only absolute invariant with no slack to freeze. A definition on a live
#: surface that a coverage report proves was never executed is unambiguous debt;
#: it is 0 today and there is no legitimate reason for it to rise.
HARD_ZERO = ("never_executed",)


def _fail_env(msg: str) -> None:
    print(f"liveness gate: CANNOT RUN: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _load_sibling(name: str):
    """Load a ``scripts/<name>.py`` sibling module by path — ``scripts`` ships an
    ``__init__.py`` (making it a package) but this gate must work regardless of
    how it is invoked (bare ``python3 scripts/check_liveness.py`` has no package
    context), the idiom ``tests/gates/test_wire_first_gate.py`` already proves."""
    spec = importlib.util.spec_from_file_location(
        f"_{name}_for_check_liveness", REPO / "scripts" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec — `liveness_deferred.py`'s
    # `@dataclass` decorator introspects `sys.modules[cls.__module__]` at
    # class-definition time; skipping this step makes that lookup return None
    # and crash (a real failure this gate's own dogfooding caught).
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


liveness_reconciler = _load_sibling("liveness_reconciler")
liveness_deferred = _load_sibling("liveness_deferred")
_git_env = _load_sibling("_git_subprocess_env")


# ── git plumbing ─────────────────────────────────────────────────────────────
# Ambient GIT_DIR/GIT_INDEX_FILE are KEPT here on purpose: the index these reads
# must see IS the one being committed. Safety comes from always running at the
# resolved toplevel with repo-relative paths. See the module docstring.


def _git(*args: str, cwd: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
    )


def repo_root() -> str:
    r = _git("rev-parse", "--show-toplevel", cwd=str(REPO))
    if r.returncode != 0 or not r.stdout.strip():
        _fail_env(f"not inside a work tree: {(r.stderr or '').strip()[:200]}")
    return r.stdout.strip()


def _changed_python(root: str) -> tuple[list[str], str]:
    """Repo-relative ``agent_utilities/**.py`` paths this commit touches.

    Uses the INDEX whenever ANYTHING is staged — that is what the commit will
    contain, so an unstaged edit can neither hide a violation nor invent one,
    independently of whether pre-commit's own stash ran. Only when the index is
    completely empty (a developer running this by hand, mid-edit) does it fall
    back to the working tree, so a manual run is still useful. Returns
    (paths, rev) where rev is ``:`` (the index) or ``""`` (the working tree).
    """

    def _py(out: str) -> list[str]:
        return [
            line
            for line in out.splitlines()
            if line.endswith(".py") and line.startswith("agent_utilities/")
        ]

    staged = _git("diff", "--cached", "--name-only", "--diff-filter=ACMR", cwd=root)
    if staged.returncode == 0 and staged.stdout.strip():
        return _py(staged.stdout), ":"
    unstaged = _git("diff", "--name-only", "--diff-filter=ACMR", "HEAD", cwd=root)
    if unstaged.returncode == 0:
        return _py(unstaged.stdout), ""
    return [], ":"


def _blob(root: str, rev: str, rel: str) -> str | None:
    """Source of ``rel`` at ``rev``, or None if git has no such blob.

    ``rev`` is ``":"`` (the index — git spells that ``:path``, with no second
    colon), ``""`` (the working tree, read from disk), or a real revision such
    as ``HEAD`` (``HEAD:path``).
    """
    if rev == "":
        try:
            return (Path(root) / rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
    spec = f":{rel}" if rev == ":" else f"{rev}:{rel}"
    r = _git("show", spec, cwd=root)
    return r.stdout if r.returncode == 0 else None


# ── per-file "invoked but fake" detection (imported, not reimplemented) ──────


def _find_analyzer() -> Path | None:
    spec = importlib.util.find_spec("universal_skills")
    locs = list(getattr(spec, "submodule_search_locations", []) or []) if spec else []
    for loc in locs:
        cand = Path(loc) / "core" / "code-enhancer" / "scripts" / "analyze_liveness.py"
        if cand.exists():
            return cand
    return None


_REQUIRED_ANALYZER_NAMES = (
    "_facade_branches",
    "_facade_except_handlers",
    "_does_real_work",
    "_returns_canned_payload",
    "_decorator_names",
    "_is_test",
    "_SURFACE_PARTS",
    "_SURFACE_DECORATORS",
    "_INFO_NAMES_RE",
    "_PLACEHOLDER_RE",
)


def _import_analyzer(path: Path):
    """Import the vendored detector as a module so its per-file passes can be
    reused verbatim. Duplicating them here would be a second copy of the
    detector; a missing required name means the vendored analyzer changed shape
    and this gate must say so rather than quietly enforce less. The private
    stable-ID helper is the one compatibility-adapted optional surface because
    universal-skills 1.2.x does not expose it."""
    spec = importlib.util.spec_from_file_location("_analyze_liveness_for_gate", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        _fail_env(f"could not load the detector at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover - defensive
        _fail_env(f"importing the detector at {path} failed: {exc!r}")
    missing = [n for n in _REQUIRED_ANALYZER_NAMES if not hasattr(module, n)]
    if missing:
        _fail_env(
            f"the detector at {path} no longer exposes {missing}. The diff-scoped "
            "layer cannot run and will NOT be reported as clean. Re-align this "
            "gate with the vendored analyzer."
        )
    return module


def _hash(kind: str, text: str) -> str:
    """Content-anchored finding id. See the docstring: identity must survive a
    function being extracted, renamed, or re-nested, because those moves are what
    produced 11 phantom regressions under the old ``symbol@ordinal`` identity."""
    digest = hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:16]
    return f"{kind}:{digest}"


def _fallback_stable_finding_id(text: str) -> str:
    """Mirror universal-skills' content-stable placeholder ID contract.

    universal-skills 1.2.x has the placeholder detector but not its private
    ``_stable_finding_id`` helper. Keep the gate's identity scheme identical to
    the newer analyzer so upgrading or downgrading that dependency does not
    turn an unchanged marker into a diff-scoped regression.
    """
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:8]


def _stable_analyzer_id(an, text: str) -> str:
    """Use the analyzer's stable-ID helper, with a strict compatibility shim."""
    if not hasattr(an, "_stable_finding_id"):
        return _fallback_stable_finding_id(text)
    helper = an._stable_finding_id
    if not callable(helper):
        _fail_env("the detector's optional _stable_finding_id is not callable")
    try:
        value = helper(text)
    except Exception as exc:  # pragma: no cover - defensive
        _fail_env(f"the detector's _stable_finding_id failed: {exc!r}")
    if not isinstance(value, str) or not value:
        _fail_env("the detector's _stable_finding_id returned an invalid ID")
    return value


def _substantive_constants(node) -> frozenset:
    """Non-empty strings and non-zero numbers under ``node``.

    This is the CONTENT of a canned payload — mirrors the vendored
    ``_has_substantive_literal``'s notion of "actual fabricated content, not an
    empty/neutral fallback". It is what the novelty rule below compares, because
    it is the one thing that survives every move the complexity-collapse program
    makes: hoisting a branch into its own function, turning
    ``response_md = "…"`` into ``return "…"``, re-nesting, renaming, reflowing.
    """
    out = set()
    for n in ast.walk(node):
        if not isinstance(n, ast.Constant):
            continue
        v = n.value
        if isinstance(v, str) and v.strip():
            out.add(("s", v))
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and v != 0:
            out.add(("n", v))
    return frozenset(out)


def _file_constants(tree) -> frozenset:
    return _substantive_constants(tree)


def _branch_index(tree: ast.Module) -> dict[int, list[ast.stmt]]:
    """``first-statement lineno -> that branch's statement list``.

    ``_facade_branches`` reports ``body[0].lineno``; this recovers the body so it
    can be hashed by content. Both the ``if`` body and a real ``else`` body are
    indexed, matching what the detector considers a branch.
    """
    out: dict[int, list[ast.stmt]] = {}
    for n in ast.walk(tree):
        if not isinstance(n, ast.If):
            continue
        if n.body:
            out.setdefault(n.body[0].lineno, n.body)
        if n.orelse:
            out.setdefault(n.orelse[0].lineno, n.orelse)
    return out


def _handler_index(tree: ast.Module) -> dict[int, ast.ExceptHandler]:
    return {n.lineno: n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)}


def _unparse(nodes) -> str:
    """Normalised source. ``ast.unparse`` erases formatting/comment churn, so a
    reflow or a black run cannot fabricate a "new" finding."""
    return "\n".join(ast.unparse(n) for n in nodes)


class _Scan:
    """One file's "invoked but fake" findings plus its whole literal universe."""

    __slots__ = ("findings", "constants")

    def __init__(self) -> None:
        #: {finding_id: frozenset of substantive constants in the flagged code}
        self.findings: dict[str, frozenset] = {}
        #: every substantive constant anywhere in the file
        self.constants: frozenset = frozenset()


def _placeholder_ids(an, src: str):
    """Admitted placeholder/stub tells in raw source. Identity is the vendored
    analyzer's own content hash of the matched line, so a marker that merely
    moves is the same finding."""
    for line in src.splitlines():
        if an._PLACEHOLDER_RE.search(line):
            yield _hash("placeholder", _stable_analyzer_id(an, line.strip()))


def _surface_functions(an, tree, path: Path):
    """The functions the vendored analyzer's layer 4 considers: those in a
    surface module or carrying a surface decorator, minus info/help handlers
    whose job IS to return static text."""
    is_surface_mod = bool(set(path.parts) & an._SURFACE_PARTS)
    for n in ast.walk(tree):
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        decorated = bool(an._decorator_names(n) & an._SURFACE_DECORATORS)
        if (is_surface_mod or decorated) and not an._INFO_NAMES_RE.search(n.name):
            yield n


def _flagged_nodes(an, fn, branches: dict, handlers: dict):
    """``(kind, [ast nodes])`` for every finding the detector reports inside one
    function. A whole-function facade short-circuits, exactly as the vendored
    layer 4 does, so its branches are not double-counted."""
    if not an._does_real_work(fn) and an._returns_canned_payload(fn):
        yield "facade-function", [fn]
        return
    for line in an._facade_branches(fn):
        body = branches.get(line)
        if body is not None:
            yield "facade-branch", body
    for line in an._facade_except_handlers(fn):
        handler = handlers.get(line)
        if handler is not None:
            yield "facade-except", handler.body


def _file_findings(an, rel: str, src: str) -> _Scan:
    """Content-anchored facade + placeholder findings for ONE file's source.

    Mirrors the vendored analyzer's layer-4 gating exactly (test files skipped;
    a function is considered only when it lives in a surface module or carries a
    surface decorator; info/help handlers exempt) but reuses that analyzer's own
    predicates rather than restating them.
    """
    scan = _Scan()
    path = Path(rel)
    if an._is_test(path):
        return scan

    for fid in _placeholder_ids(an, src):
        scan.findings[fid] = frozenset()

    try:
        tree = ast.parse(src)
    except SyntaxError:
        # An unparseable blob is not "clean". Report it as its own finding so a
        # broken file cannot slip through as zero findings.
        scan.findings[_hash("unparseable", rel)] = frozenset()
        return scan

    scan.constants = _file_constants(tree)
    branches = _branch_index(tree)
    handlers = _handler_index(tree)
    for fn in _surface_functions(an, tree, path):
        for kind, nodes in _flagged_nodes(an, fn, branches, handlers):
            consts = frozenset().union(*(_substantive_constants(n) for n in nodes))
            scan.findings[_hash(kind, _unparse(nodes))] = consts
    return scan


def _new_findings(before: _Scan, after: _Scan) -> list[str]:
    """Findings this change INTRODUCES, under a refactor-robust novelty rule.

    A finding is new when its id is absent from HEAD **and** the canned payload
    it flags contains at least one substantive literal that did not appear
    ANYWHERE in HEAD's version of the file.

    The second clause is what makes the gate survive the program running over
    this repository. The complexity-collapse refactors move canned text between
    an assignment and a return, and out of a mega-dispatcher into a helper; the
    detector then flags text it did not flag before, at a new location, with a
    new enclosing symbol. Measured over the three refactors that moved this
    metric (intent_tools 2b7719220, commands 039c30d04, analysis_tools
    f0655146c) the id-only rule reports 14 "new" findings, every one of which is
    pre-existing text that changed neither meaning nor behaviour. Requiring a
    literal that is genuinely new to the file reports 0 for all three, while a
    planted fabricated payload — which by construction introduces its own
    strings/numbers — still fails. A finding with no substantive literal at all
    (an empty envelope, a placeholder marker) falls back to the plain id
    comparison, so it can never be waved through by this clause.
    """
    out = []
    for fid, consts in sorted(after.findings.items()):
        if fid in before.findings:
            continue
        if consts and consts <= before.constants:
            continue  # every literal already lived in this file — relocated, not new
        out.append(fid)
    return out


# ── census ───────────────────────────────────────────────────────────────────


def _no_analyzer() -> int:
    # Say NOT ENFORCED, loudly. This branch is not hypothetical: the `guardrails`
    # dependency group `.github/workflows/guardrails.yml` syncs does NOT contain
    # `universal-skills` (it is only in the `agent-runtime` / `agent-headless`
    # extras), so that CI job takes this path and exits 0 — a gate that reads
    # green in the log while checking nothing (D-PCG-9). Until that group gains
    # the dependency the wording must make a skip impossible to mistake for a
    # pass.
    print(
        "liveness gate NOT ENFORCED (exit 0, nothing was checked): the "
        "code-enhancer detector (universal_skills) is not importable by "
        f"{sys.executable}. Add `universal-skills` to the environment this gate "
        "runs in — for CI that is the `guardrails` dependency group in "
        "pyproject.toml."
    )
    return 0


def _run_census(analyzer: Path) -> tuple[dict[str, int], dict[str, list[str]], bool]:
    """The whole-tree scan. Returns (counts, details, coverage_present).

    Runs with the repository-redirecting git variables STRIPPED — this walks the
    tree, and BUG-180 proved that inheriting them makes it walk the wrong one.
    """
    cmd = [sys.executable, str(analyzer), str(TARGET)]
    cov = REPO / "coverage.json"
    if cov.exists():
        cmd += ["--coverage", str(cov)]
    # No `--baseline` is passed: the analyzer is a finding SOURCE, never the
    # judge. Its own gate/regressed computation runs against the RAW (buggy)
    # orphan_modules/dead_definitions counts.
    res = subprocess.run(
        cmd, capture_output=True, text=True, env=_git_env.sanitized_git_env()
    )
    if res.returncode != 0:
        sys.stderr.write(res.stderr)
        # A crashing analyzer is NOT a pass. This once returned 0, which made a
        # broken detector indistinguishable from a clean tree — the same
        # silently-non-functional-gate failure mode `check_cpd.py` and
        # `check_surface_parity.py` were both in.
        _fail_env(
            f"the detector crashed (exit {res.returncode}); it enforced nothing. "
            "Fix the analyzer — do not gate around it."
        )
    report = json.loads(res.stdout)
    details = report.get("details", {})

    saved = {
        k: os.environ[k] for k in _git_env._DANGEROUS_GIT_ENV_VARS if k in os.environ
    }
    try:
        _git_env.strip_inherited_git_repository_env()
        recon = liveness_reconciler.reconcile(details)
    finally:
        os.environ.update(saved)

    om, dd = recon["orphan_modules"], recon["dead_definitions"]
    counts = dict(report["counts"])
    counts["orphan_modules"] = len(om["still"])
    counts["dead_definitions"] = len(dd["still"])
    details = dict(details)
    details["orphan_modules"] = list(om["still"])
    details["dead_definitions"] = list(dd["still"])

    print("Liveness census — the REAL numbers, nothing frozen, nothing compared:")
    print(
        f"  reconciliation (scripts/liveness_reconciler.py): "
        f"orphan_modules raw={len(report['details'].get('orphan_modules', [])):4d} "
        f"-> {counts['orphan_modules']:4d} (rescued={len(om['rescued'])}, "
        f"excluded_generated={len(om['excluded_generated'])}); "
        f"dead_definitions raw={len(report['details'].get('dead_definitions', [])):4d} "
        f"-> {counts['dead_definitions']:4d} (rescued={len(dd['rescued'])}, "
        f"excluded_generated={len(dd['excluded_generated'])})"
    )
    if recon["dynamic_unresolvable_sites"]:
        print(
            f"  {len(recon['dynamic_unresolvable_sites'])} genuinely dynamic import "
            "call site(s) (non-literal importlib.import_module()/__import__() "
            "target) — remaining findings in those package directories were "
            "treated fail-open (not dead), not guessed at."
        )
    for cat in CATEGORIES:
        items = details.get(cat, [])
        head = ", ".join(str(x) for x in items[:3])
        print(
            f"  {cat:<20} {counts.get(cat, 0):>6}" + (f"   e.g. {head}" if head else "")
        )
    if not cov.exists():
        print(
            "  NOTE: no coverage.json — `never_executed` is 0 because that layer "
            "had no input, not because it was proven clean."
        )
    return counts, details, cov.exists()


# ── deferrals ────────────────────────────────────────────────────────────────


def _check_deferrals() -> bool:
    """GOC-68: a deferral without an expiry becomes permanent. Failing entries
    are a policy violation, independent of any count."""
    failed = False
    try:
        entries = liveness_deferred.load_entries()
    except ValueError as exc:
        print(f"\n❌ liveness_deferred.tsv is unparsable: {exc}")
        return True

    malformed = [e for e in entries if not liveness_deferred.is_well_formed(e)]
    if malformed:
        failed = True
        print(
            "\n❌ scripts/liveness_deferred.tsv entries missing owner + "
            "(review-by OR PERMANENT reason):"
        )
        for e in malformed:
            print(f"  - line {e.line_no}: {e.category}\t{e.pattern}")

    stale = liveness_deferred.stale_entries(entries, date.today())
    if stale:
        failed = True
        print(
            "\n❌ scripts/liveness_deferred.tsv entries past their review-by date "
            "(bind, extend with a new reason, or mark PERMANENT with a justification):"
        )
        for e in stale:
            print(
                f"  - {e.category}\t{e.pattern} (owner={e.owner}, review-by={e.review_by})"
            )
    return failed


# ── enforcement ──────────────────────────────────────────────────────────────


ADVICE = """
Wire the new code into a live path, or make the handler do the work it claims to
do. Do NOT make this pass by widening a threshold and do NOT add a suppression
comment — an in-line suppression is a one-line baseline. There is no cap to
raise: this gate has no baseline and no cap table, by policy (NO RATCHETS —
findings are debt to burn down, never to freeze). If the placeholder is genuinely
intentional and time-boxed, record it with an owner and a review-by date in
scripts/liveness_deferred.tsv, which this same gate expires.
"""


def _check_one_file(an, root: str, rev: str, rel: str) -> bool:
    """Report one file's absolute findings and fail on the ones it introduces."""
    after_src = _blob(root, rev, rel)
    if after_src is None:
        # git listed this path as Added/Copied/Modified/Renamed, so the blob MUST
        # exist. Silently skipping here is how this layer was briefly vacuous
        # while still printing a clean verdict (caught by the planted-input
        # proof, which is exactly what that proof is for).
        _fail_env(f"could not read {rel} at {rev or 'the working tree'!r}")
    after = _file_findings(an, rel, after_src)
    before_src = _blob(root, "HEAD", rel)
    before = _file_findings(an, rel, before_src) if before_src is not None else _Scan()
    new = _new_findings(before, after)
    # The REAL absolute number for a touched file is printed even though this
    # layer does not fail on it — pre-existing debt stays on screen.
    print(
        f"  {rel}: {len(after.findings)} fake/placeholder finding(s) present"
        + (f", {len(new)} NEW in this change" if new else "")
    )
    for fid in new:
        print(f"      NEW  {fid}")
    return bool(new)


def _enforce_diff(an, root: str) -> bool:
    """FAIL on an "invoked but fake" finding this commit introduces."""
    files, rev = _changed_python(root)
    scope = "index" if rev == ":" else "working tree"
    if not files:
        print(
            "\nliveness(diff): no agent_utilities/**.py changed vs HEAD "
            "— nothing to compare"
        )
        return False
    print(
        f"\nliveness(diff): {len(files)} changed agent_utilities/**.py file(s) "
        f"({scope} vs HEAD), recomputed live — nothing frozen"
    )
    failed = False
    for rel in files:
        failed |= _check_one_file(an, root, rev, rel)
    if failed:
        print(
            "\n❌ liveness(diff) FAIL: this commit introduces code the "
            "'invoked but fake' detector flags — a live-surface handler returning "
            "a canned payload while doing no real work, a deceptive broad-except "
            "fallback, or a new placeholder/TODO marker. Identity is the content "
            "of the flagged code, so extracting or renaming a function cannot "
            "produce this; something genuinely new is being added."
        )
        print(ADVICE)
    return failed


def _enforce_absolute(counts: dict[str, int], maxima: dict[str, int]) -> bool:
    failed = False
    for cat in HARD_ZERO:
        if counts.get(cat, 0) > 0:
            failed = True
            print(
                f"\n❌ {cat} = {counts[cat]}, and the invariant is 0. This is not a "
                "cap that can be raised."
            )
    for cat, limit in maxima.items():
        now = counts.get(cat, 0)
        status = "OK" if now <= limit else "FAIL"
        print(f"\nliveness(absolute): {cat} = {now}, --max {limit} -> {status}")
        if now > limit:
            failed = True
            print(
                f"❌ {cat} exceeds the absolute limit this run was asked to "
                "enforce. Lower the finding count; the limit is an argument to "
                "this invocation and is never written back anywhere."
            )
    return failed


def _parse_maxima(raw: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in raw:
        cat, _, val = item.partition("=")
        if cat not in CATEGORIES or not val.isdigit():
            _fail_env(
                f"--max expects <category>=<int> with category in {list(CATEGORIES)}; "
                f"got {item!r}"
            )
        out[cat] = int(val)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Liveness census + diff-scoped gate")
    ap.add_argument(
        "--census",
        action="store_true",
        help="print the real numbers and exit 0 without enforcing anything",
    )
    ap.add_argument(
        "--max",
        action="append",
        default=[],
        metavar="CATEGORY=N",
        help=(
            "enforce an ABSOLUTE ceiling for one category on THIS invocation. "
            "No defaults exist and nothing is written back — a CI job passes the "
            "number it is driving down."
        ),
    )
    # RETIRED, kept so a re-introduction fails loudly rather than looking like a typo.
    ap.add_argument("--baseline", type=Path, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--update-baseline", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--write", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.baseline is not None or args.update_baseline or args.write:
        _fail_env(
            "--baseline/--update-baseline/--write are RETIRED, along with the CAPS "
            "table they replaced. A baseline converts a finding into invisible "
            "permanent debt (NO RATCHETS). This gate reports the real distribution "
            "and gates on what a commit ADDS; use --max CATEGORY=N for an absolute "
            "ceiling you intend to drive down."
        )

    maxima = _parse_maxima(args.max)
    analyzer = _find_analyzer()
    if analyzer is None:
        return _no_analyzer()

    root = repo_root()
    an = _import_analyzer(analyzer)
    counts, _details, _cov = _run_census(analyzer)

    if args.census:
        print("\nliveness: --census, nothing enforced (exit 0 by request)")
        return 0

    failed = _enforce_absolute(counts, maxima)
    failed |= _enforce_diff(an, root)
    failed |= _check_deferrals()

    if not failed:
        print("\nliveness gate: OK — census reported above, nothing new introduced")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
