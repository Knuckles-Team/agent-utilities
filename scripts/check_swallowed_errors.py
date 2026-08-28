#!/usr/bin/env python3
"""Swallowed-error guardrail gate (CONCEPT:AU-AHE.evaluation — diagnosability).

The single highest-leverage defect class found across a two-day debugging
session (2026-07-22/23): a handler that discards *why* an operation failed —
``except Exception: pass``, a bare ``except:``, an ``except ...: return`` with
no log, or a handler that logs only ``type(exc).__name__`` (the class name,
message dropped). Every major blocker in that session hid behind one of these
shapes; fixing the log line to include the real cause turned an
undiagnosable failure into a five-minute fix. A gate that catches this
pattern is worth more than any individual bug fix — this is that gate.

This gate is a **static heuristic**, not a dataflow prover (matching every
other ``scripts/check_*.py`` gate in this repo). For each ``except`` handler
that does not re-raise, it asks: does this handler ALREADY have a trail back
to the real cause? Two independent ways to have one:

1. **The justified ``# noqa: BLE001 — <reason>`` convention** — a comment on
   the ``except`` line documenting why swallowing here is deliberate and safe
   (already used at ~2000 sites in this codebase, e.g. "telemetry must never
   block startup", "duplicate registration tolerated"). Any non-empty reason
   after the ``BLE001`` marker (``-``/``—``/``:`` separator) counts.
2. **Cause-preserving logging** — the handler logs the bound exception itself
   (or ``str(exc)``/``repr(exc)``), or calls ``logger.exception(...)`` (which
   always attaches the current exception's traceback), rather than only
   ``type(exc).__name__``/``exc.__class__.__name__`` or nothing at all.
   NOTE: ``exc_info=True`` is NOT recognized as cause-preserving in this
   codebase specifically — ``core/log_privacy.py``'s process-wide LogRecord
   factory unconditionally nulls ``record.exc_info``/``exc_text``/
   ``stack_info`` for every ``agent_utilities.*`` logger (tracebacks embed
   host filesystem paths), so passing the exception object as a ``%s``/f-string
   argument is the only calling convention that actually survives to the log
   output — ``core/log_privacy.py``'s ``_sanitize_value`` renders it as
   ``f"{type(exc).__name__}: {sanitize_log_text(str(exc))}"``, preserving the
   message while still redacting endpoints/paths/emails.

A handler with NEITHER of those is a **violation**: a genuinely undocumented,
cause-dropping swallow. Flagged shapes (matching the exact patterns named in
the task this gate was built for):

* a bare ``except:`` (catches even ``SystemExit``/``KeyboardInterrupt``);
* ``except <Type>: pass`` (or ``pass`` as the sole statement);
* ``except <Type>: return`` with no value and no log call anywhere in the body;
* ``except ... as exc:`` whose only reference to ``exc`` in a log call is
  ``type(exc).__name__``/``exc.__class__.__name__`` (message dropped);
* ``except ... as exc:`` whose ONLY cause-preserving log call is at
  ``logger.debug(...)`` level, with no ``# noqa: BLE001 — <reason>``
  (``debug_only_swallow`` — see "Why DEBUG alone doesn't count" below,
  CONCEPT:AU-AHE.evaluation.debug-swallow-justification).

Deliberately NOT flagged: a handler that recovers with a **typed, narrow**
exception class doing ordinary control flow (``except ValueError: x = default``,
``except FileExistsError: ...``) — those aren't the "swallow a broad Exception
silently" antipattern this gate targets, and flagging every typed fallback
would swamp the signal in normal, idiomatic Python. A handler that re-raises
(anywhere in its body) is never a swallow at all.

Why DEBUG alone doesn't count (D-SWG-2, CONCEPT:AU-AHE.evaluation.
debug-swallow-justification): ``core/log_privacy.py``'s sanitizer makes a
handler's exception object the only calling convention that survives to the
log output at all — but *surviving to the log* and *surviving to a log
level anyone is actually watching in production* are different claims.
``agent_runner.py``'s ``:RunTrace`` write failure (D-DG-7) was
``except Exception as e: logger.debug(...)``: the exception object WAS
passed, cause-preservation as originally defined was satisfied, the gate
was green, and the failure was still invisible — DEBUG is off by default in
every production deployment this codebase ships. A handler whose *only*
cause-preserving log call is at DEBUG is therefore treated the same as an
undocumented swallow (`debug_only_swallow`) UNLESS it carries the existing
``# noqa: BLE001 — <reason>`` justification. This is deliberately not a
blanket ban on ``logger.debug``: a handler that also (or instead) logs at
``warning``/``error``/``exception``/``critical``/``info`` is unaffected, and
a handler with a real justification (e.g. "telemetry span close, expected to
fail when tracing is disabled, checked hourly by /health instead") stays
green by writing it down — the same bar every other swallow in this gate
already has to clear. The line is drawn on *loudness*, not on *level as a
lint target*: this keeps the common legitimate best-effort DEBUG swallow
quiet (one comment) while making the load-bearing one loud (a failing gate)
without re-flagging the ~39 ``log_type_name_only`` sites that have nothing
to do with level.

Because the codebase already carries a large number of un-annotated
`except Exception`-shaped handlers this gate's own author has not
individually triaged (the DEBT-2 sweep fixed every genuinely high-value site
it found — the public error-surface boundary, the MCP/graph-os boot chain,
the multiplexer child-lifecycle log lines, engine mutation paths — see the
sweep's final report for the full verdict table), the gate cannot demand an
absolute zero today. It is **not** a ratchet either. There is no baseline
file, nothing is written to disk, and no count is frozen. Instead:

* an **unconditional census** prints the real totals — sites, per-shape
  histogram, worst files — on EVERY run, pass or fail. Debt that is printed
  is debt someone can burn down; debt frozen in a file is debt nobody sees.
* enforcement is **diff-scoped**, recomputed live from the HEAD blob: a
  cause-dropping handler this change ADDS fails the commit, one it leaves
  alone does not.
* one shape (``HARD_ZERO_SHAPES``) is enforced **absolutely**, repo-wide,
  because the repo already sits at zero for it.

NOTE on why the old baseline key had to go (D-SWG-1 superseded). The frozen
key was ``(file, ENCLOSING SYMBOL, exception type text, shape, ordinal)``,
chosen because a line-number key manufactured a phantom finding on every
unrelated edit above a baselined handler. It fixed that and introduced a
worse one: **an enclosing symbol is not invariant under function
extraction.** When the complexity program split ~20 functions apart, every
handler that moved from ``create_agent`` into
``create_agent._setup_mcp_url_toolset`` re-keyed and read as brand-new debt.
That was measured, not assumed: of the 37 findings blocking every au commit,
**all 37** were re-keys — not one raised the count for its
``(file, type, shape)`` above what the baseline already held. Meanwhile the
real backlog had fallen 1,049 -> 584 and the gate's own output said only "no
new swallowed-error sites". Gating on a number that does not mean what its
name says gates the instrument, not the code.

The comparison key is therefore now **content**: ``(exception type text,
shape, the except line's own source text)``, compared per file against the
same file at HEAD. It carries no symbol and no line number, so extraction,
renaming, reordering and line motion are all invisible to it, and only a
genuinely added swallow fails. Moving an existing swallow to a DIFFERENT file
reads as added in the destination; that is deliberate — justify it with a
``# noqa: BLE001 — <reason>`` like any other.

Usage:
  python3 scripts/check_swallowed_errors.py       # census + diff-scoped check
  python3 scripts/check_swallowed_errors.py ROOT  # scan ROOT, report EVERY
                                                  # finding (test fixture dir)

Exit 0 = this change added no cause-dropping handler, 1 = it did (or a
hard-zero shape exists), 2 = a retired flag was passed.
"""

from __future__ import annotations

import argparse
import ast
import collections
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts._git_scan import tracked_or_walked  # noqa: E402

PKG = ROOT / "agent_utilities"
SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", "build", "dist"}

# Shapes that must never exist at all, at any count. A bare ``except:`` also
# catches ``SystemExit`` and ``KeyboardInterrupt``, so it can swallow a
# shutdown signal; the repo is at zero and stays there.
HARD_ZERO_SHAPES = frozenset({"bare_except"})


def _tracked_or_walked_py_files(target: Path) -> list[Path]:
    """``.py`` files under ``target``, preferring the git-tracked set (BUG-043).

    A raw ``rglob`` also picks up gitignored, generated build output, which
    can carry a stale copy of an already-fixed source file and reintroduce a
    cleared violation. Falls back to a filesystem walk only when ``target``
    is not inside a git working tree (e.g. a synthetic test fixture).
    """
    return tracked_or_walked(target, "*.py", root=ROOT)


# The justified-convention marker: a "noqa: BLE001"-style comment followed by
# a non-empty reason after a -/—/: separator. A bare marker with NO reason is
# deliberately NOT accepted — the convention this codebase established is
# "document why", not just "silence the lint".
_NOQA_BLE001_WITH_REASON_RE = re.compile(r"#\s*noqa:\s*BLE001\s*[-—:]\s*\S")

_LOG_METHODS = {"debug", "info", "warning", "error", "exception", "critical", "warn"}


def _is_log_call(call: ast.Call) -> tuple[bool, str | None]:
    """(is_a_log_call, method_name) for ``<something with 'log' in its name>.<level>(...)``."""
    fn = call.func
    if not isinstance(fn, ast.Attribute) or fn.attr not in _LOG_METHODS:
        return False, None
    base = fn.value
    base_name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", None)
    if base_name and "log" in base_name.lower():
        return True, fn.attr
    return False, None


def _expr_src(source_lines: list[str], node: ast.expr) -> str:
    try:
        return ast.get_source_segment("\n".join(source_lines), node) or ""
    except Exception:  # noqa: BLE001 - best-effort snippet only, never fatal
        return ""


def _has_raise(body: list[ast.stmt]) -> bool:
    return any(
        isinstance(n, ast.Raise)
        for n in ast.walk(ast.Module(body=body, type_ignores=[]))
    )


def _call_preserves_cause(
    call: ast.Call, bound_name: str, source_lines: list[str]
) -> bool:
    """True if this one log ``call`` references the bound exception itself
    (not just its type name)."""
    args_src = [_expr_src(source_lines, a) for a in call.args]
    kw_src = [_expr_src(source_lines, kw.value) for kw in call.keywords]
    joined = " ".join(args_src + kw_src)
    # The bound name appearing WITHOUT being wrapped in type(...)/.__class__
    # is cause-preserving (log_privacy.py's _sanitize_value renders a raw
    # BaseException as "Type: message", and str(exc)/repr(exc) obviously
    # carry the message too).
    bare_ref = re.search(rf"\b{re.escape(bound_name)}\b", joined)
    type_wrapped = re.search(
        rf"type\(\s*{re.escape(bound_name)}\s*\)|"
        rf"{re.escape(bound_name)}\.__class__",
        joined,
    )
    return bool(
        bare_ref
        and not (
            type_wrapped
            and not re.search(
                rf"str\(\s*{re.escape(bound_name)}\s*\)|repr\(\s*{re.escape(bound_name)}\s*\)",
                joined,
            )
        )
    )


def _cause_preservation_level(
    body: list[ast.stmt], bound_name: str | None, source_lines: list[str]
) -> str:
    """ "loud" (a non-DEBUG log call already carries the real cause forward —
    fully justified, no violation), "debug_only" (the ONLY cause-preserving
    log call(s) are at DEBUG level — D-SWG-2: invisible in production unless
    explicitly justified via ``# noqa: BLE001 — <reason>``), or "none" (no
    cause-preserving log call at all — handled by ``_shape`` instead).
    """
    levels = [
        _one_call_preservation(node, bound_name, source_lines)
        for node in ast.walk(ast.Module(body=body, type_ignores=[]))
        if isinstance(node, ast.Call)
    ]
    if "loud" in levels:
        return "loud"
    return "debug_only" if "debug_only" in levels else "none"


def _one_call_preservation(
    call: ast.Call, bound_name: str | None, source_lines: list[str]
) -> str:
    """``"loud"`` / ``"debug_only"`` / ``"none"`` for ONE call node."""
    is_log, method = _is_log_call(call)
    if not is_log:
        return "none"
    if method == "exception":
        # logger.exception(...) always attaches the current exception's
        # traceback/message regardless of what args are passed, and is never a
        # quiet level (it logs at ERROR).
        return "loud"
    if bound_name is None or not _call_preserves_cause(call, bound_name, source_lines):
        return "none"
    return "debug_only" if method == "debug" else "loud"


def _log_calls_in(body: list[ast.stmt]) -> list[ast.Call]:
    """Every ``<logger>.<level>(...)`` call anywhere in a handler body."""
    return [
        node
        for node in ast.walk(ast.Module(body=body, type_ignores=[]))
        if isinstance(node, ast.Call) and _is_log_call(node)[0]
    ]


def _call_args_src(call: ast.Call, source_lines: list[str]) -> str:
    """One string holding the source of every positional and keyword argument."""
    args = [_expr_src(source_lines, a) for a in call.args]
    kwargs = [_expr_src(source_lines, kw.value) for kw in call.keywords]
    return " ".join(args + kwargs)


def _drops_cause_to_type_name(
    call: ast.Call, bound_name: str, source_lines: list[str]
) -> bool:
    """True when this log call reduces the exception to its CLASS NAME.

    A call that logs the type name AND the exception itself is
    cause-preserving -- ``"failed (%s: %s)", type(e).__name__, e`` carries the
    real message. Only a call that names the class and nothing else drops the
    cause, so the bare binding must be absent once the type-name expressions
    themselves have been removed from the argument source.
    """
    type_name_expr = re.compile(
        rf"type\(\s*{re.escape(bound_name)}\s*\)\.__name__|"
        rf"{re.escape(bound_name)}\.__class__\.__name__"
    )
    joined = _call_args_src(call, source_lines)
    if not type_name_expr.search(joined):
        return False
    bare_exc = re.compile(rf"(?<![\w.]){re.escape(bound_name)}(?![\w(])")
    return not bare_exc.search(type_name_expr.sub("", joined))


def _is_type_name_only_swallow(
    body: list[ast.stmt], bound_name: str | None, source_lines: list[str]
) -> bool:
    """At least one log call exists and EVERY one of them drops the cause.

    If even one log call carries the real cause the handler is not
    cause-dropping, which is what this branch has always documented.
    """
    if not bound_name:
        return False
    log_calls = _log_calls_in(body)
    if not log_calls:
        return False
    dropping = sum(
        1 for call in log_calls if _drops_cause_to_type_name(call, bound_name, source_lines)
    )
    return dropping == len(log_calls)


def _is_silent_return(body: list[ast.stmt]) -> bool:
    """A lone bare ``return`` with no log call anywhere in the body."""
    if not (len(body) == 1 and isinstance(body[0], ast.Return)):
        return False
    return body[0].value is None and not _log_calls_in(body)


def _shape(
    body: list[ast.stmt], bound_name: str | None, source_lines: list[str]
) -> str | None:
    """Returns a violation shape name, or None if this handler isn't a target shape."""
    if len(body) == 1 and isinstance(body[0], ast.Pass):
        return "pass"
    if len(body) == 1 and isinstance(body[0], ast.Return) and body[0].value is None:
        # A lone bare `return`. A `return <value>` is NOT short-circuited here
        # -- it falls through to the type-name check below, exactly as before.
        return "return_none" if _is_silent_return(body) else None
    if _is_type_name_only_swallow(body, bound_name, source_lines):
        return "log_type_name_only"
    return None


def _iter_except_handlers_with_scope(
    node: ast.AST, scope_stack: tuple[str, ...] = ()
) -> list[tuple[ast.ExceptHandler, str]]:
    """Yields (handler, enclosing qualname) for every ``ExceptHandler`` under
    ``node``, e.g. ``"ClientFactory.create"`` or ``"<module>"``.

    ``ast.walk()`` gives no parent information, so this is a manual
    pre-order DFS that tracks the nearest enclosing function/class chain —
    the stable component of D-SWG-1's re-key (CONCEPT:AU-AHE.evaluation.
    swallow-baseline-stable-key). Traversal is deterministic given the AST
    (source order of sibling nodes), so it does not depend on line numbers.
    """
    out: list[tuple[ast.ExceptHandler, str]] = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.extend(
                _iter_except_handlers_with_scope(child, scope_stack + (child.name,))
            )
            continue
        if isinstance(child, ast.ExceptHandler):
            out.append((child, ".".join(scope_stack) if scope_stack else "<module>"))
        out.extend(_iter_except_handlers_with_scope(child, scope_stack))
    return out


def _exception_type_text(handler: ast.ExceptHandler) -> str:
    """A normalized, formatting-insensitive rendering of the caught type(s),
    e.g. ``"(ValueError, TypeError)"`` or ``"<bare>"`` for a bare ``except:``.
    Stable across reformatting because ``ast.unparse`` reconstructs from the
    parsed AST, not from source text."""
    if handler.type is None:
        return "<bare>"
    return re.sub(r"\s+", " ", ast.unparse(handler.type)).strip()


# HandlerKey: (file, enclosing symbol, exception type text, shape, ordinal).
# Stable under pure line motion (unrelated edits elsewhere in the file);
# changes only when the handler's own type/shape/enclosing-symbol/relative
# order among identically-shaped siblings actually changes. See the module
# docstring's "NOTE on the baseline key" for the full rationale.
HandlerKey = tuple[str, str, str, str, int]

def _handler_shape(
    node: ast.ExceptHandler, except_line: str, lines: list[str]
) -> str | None:
    """The violation shape for one handler, or ``None`` if it is not one.

    ``None`` covers all four not-a-violation cases: a justified
    ``# noqa: BLE001 — <reason>``, a handler that re-raises, one that already
    logs the real cause at a level someone watches, and one whose body is
    none of the target shapes.
    """
    if _NOQA_BLE001_WITH_REASON_RE.search(except_line):
        return None  # justified convention — already-fine
    if node.type is None:
        return "bare_except"
    if _has_raise(node.body):
        return None  # re-raises — not a swallow at all
    level = _cause_preservation_level(node.body, node.name, lines)
    if level == "loud":
        return None
    if level == "debug_only":
        return "debug_only_swallow"
    return _shape(node.body, node.name, lines)


def _find_violations(
    rel: str, source: str, tree: ast.Module
) -> list[tuple[HandlerKey, int, str, str]]:
    """Returns (key, line, shape, except_line_text) for every site in this file."""
    lines = source.splitlines()
    violations: list[tuple[HandlerKey, int, str, str]] = []
    ordinals: dict[tuple[str, str, str], int] = {}
    for node, symbol in _iter_except_handlers_with_scope(tree):
        except_line = lines[node.lineno - 1] if node.lineno - 1 < len(lines) else ""
        shape = _handler_shape(node, except_line, lines)
        if not shape:
            continue
        exc_types = _exception_type_text(node)
        group = (symbol, exc_types, shape)
        ordinal = ordinals.get(group, 0)
        ordinals[group] = ordinal + 1
        key: HandlerKey = (rel, symbol, exc_types, shape, ordinal)
        violations.append((key, node.lineno, shape, except_line.strip()))
    return violations


def scan(
    target: Path = PKG, *, display_root: Path = ROOT
) -> dict[HandlerKey, tuple[int, str, str]]:
    """Returns {key: (line, shape, except_line_text)} for every current site.

    ``target``/``display_root`` default to the real package/repo root but are
    overridable (e.g. from a test's ``tmp_path`` fixture) so this gate's logic
    can be proven to actually trip on a broken fixture — see
    ``tests/gates/test_swallowed_errors_gate.py`` ("a gate that cannot fail is
    not a gate").
    """
    found: dict[HandlerKey, tuple[int, str, str]] = {}
    for py in _tracked_or_walked_py_files(target):
        if any(part in SKIP_DIRS for part in py.parts):
            continue
        if not py.is_file():
            continue
        rel = py.relative_to(display_root).as_posix()
        try:
            source = py.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py))
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        for key, lineno, shape, text in _find_violations(rel, source, tree):
            found[key] = (lineno, shape, text)
    return found


# ── Diff-scoped enforcement (the ratchet's replacement) ───────────────────
#
# WHY THERE IS NO BASELINE HERE ANY MORE.
#
# This gate used to freeze its 1,049 findings into
# ``scripts/swallowed_error_baseline.txt`` and fail only on a key absent from
# that file. Two things were wrong with it, and the second one is what
# actually broke:
#
# 1. A frozen count is a ratchet, which this project does not allow: debt is
#    to be burned down deliberately and reported honestly, never made
#    invisible. The baseline hid that the real backlog had already fallen
#    from 1,049 to 584 -- 465 sites genuinely fixed, and the gate's own output
#    said only "no new swallowed-error sites".
#
# 2. The key was ``(file, ENCLOSING SYMBOL, exception type, shape, ordinal)``,
#    and an enclosing symbol is NOT invariant under function extraction. The
#    complexity program split ~20 functions apart; every handler that moved
#    from ``create_agent`` into ``create_agent._setup_mcp_url_toolset``
#    re-keyed and read as brand-new dead-stop debt. 37 findings, every one a
#    re-key -- measured: not one of them raised the count for its
#    ``(file, type, shape)`` above what the baseline already held. So the gate
#    was red on `main`, blocking every commit, over code that had not changed.
#    Same failure as the liveness CAPS ratchet: gating on a number that does
#    not mean what its name says gates the instrument, not the code.
#
# What replaces it:
#
#   * an UNCONDITIONAL CENSUS that prints the real totals every run and never
#     fails -- nothing is written to disk, so no count can ever go stale;
#   * DIFF-SCOPED enforcement recomputed live from the HEAD blob, keyed by
#     CONTENT (exception type, shape, the ``except`` line's own text) rather
#     than by enclosing symbol, so extraction, renaming and reordering are all
#     invisible and only a genuinely added swallow fails;
#   * one ABSOLUTE invariant, ``HARD_ZERO_SHAPES``, enforced repo-wide at
#     zero because the repo is already at zero for it.
#
# The property granted is "the backlog cannot grow", and the real number is
# printed on every single run so it cannot quietly stop shrinking either.


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
    """Findings in one file, counted by CONTENT rather than by location.

    The key is ``(exception type text, shape, the except line's own text)``.
    It deliberately carries neither the enclosing symbol nor the line number,
    because neither survives an extract-method refactor -- which is exactly
    what made the old baseline key produce 37 phantom findings.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return collections.Counter()
    return collections.Counter(
        (key[2], shape, text) for key, _line, shape, text in _find_violations(rel, source, tree)
    )


def _head_source(root: str, rel: str) -> str:
    """``rel`` as of HEAD, or empty when the file is newly added."""
    r = _git("show", f"HEAD:{rel}", cwd=root)
    return r.stdout if r.returncode == 0 else ""


def _added_findings(root: str, rel: str) -> list[tuple[str, str, str, int]]:
    """(exc_types, shape, text, added_count) for content this change ADDS."""
    before = _content_counts(rel, _head_source(root, rel))
    after = _content_counts(rel, Path(root, rel).read_text(encoding="utf-8"))
    return [
        (exc_types, shape, text, count - before[(exc_types, shape, text)])
        for (exc_types, shape, text), count in after.items()
        if count > before[(exc_types, shape, text)]
    ]


def _print_census(current: dict[HandlerKey, tuple[int, str, str]]) -> None:
    """Print the real numbers. Always. This never fails the run."""
    shapes = collections.Counter(key[3] for key in current)
    files = collections.Counter(key[0] for key in current)
    print(f"swallowed-error census: {len(current)} site(s) in {len(files)} file(s)")
    for shape, count in shapes.most_common():
        print(f"  {count:5d}  {shape}")
    for rel, count in files.most_common(5):
        print(f"  top: {count:3d}  {rel}")


def _report_added(added: list[tuple[str, str, str, str, int]]) -> int:
    print("Cause-dropping exception handler(s) ADDED by this change:\n")
    for rel, exc_types, shape, text, count in added:
        suffix = f" (x{count})" if count > 1 else ""
        print(f"  {rel}: [{shape}] except {exc_types}: {text}{suffix}")
    print(
        "\nEach must either (a) log the real cause at a level someone actually "
        "watches — pass the exception itself, not type(exc).__name__, and not "
        "ONLY at logger.debug (e.g. logger.warning('...: %s', exc)) — while "
        "staying best-effort, (b) re-raise where swallowing hides a genuine "
        "failure, or (c) document a deliberate best-effort swallow with "
        "`# noqa: BLE001 — <reason>`. See AGENTS.md.\n"
        "Findings are compared per FILE by content, so MOVING an existing "
        "swallow between files reads as added in the destination; justify it "
        "the same way."
    )
    return 1


def _report_hard_zero(current: dict[HandlerKey, tuple[int, str, str]]) -> int:
    """The one absolute invariant. The repo is at zero; keep it there."""
    offenders = sorted(
        (key[0], line, shape)
        for key, (line, shape, _text) in current.items()
        if shape in HARD_ZERO_SHAPES
    )
    if not offenders:
        return 0
    print(f"\n{len(offenders)} handler(s) of an absolutely-forbidden shape:\n")
    for rel, line, shape in offenders:
        print(f"  {rel}:{line} [{shape}]")
    print(
        "\nA bare `except:` also catches SystemExit and KeyboardInterrupt. "
        "Name the exception type you actually mean to handle."
    )
    return 1


def _scan_explicit_root(root_arg: str) -> int:
    """Absolute mode: report EVERY finding under an arbitrary path.

    Used by this gate's own tests to prove it trips on a known-bad fixture.
    """
    explicit_root = Path(root_arg)
    current = scan(explicit_root, display_root=explicit_root)
    if not current:
        print(f"OK — no swallowed-error sites under {explicit_root}")
        return 0
    print("Cause-dropping exception handler(s) found:\n")
    for key, (lineno, shape, text) in sorted(
        current.items(), key=lambda kv: (kv[0][0], kv[1][0])
    ):
        print(f"  {key[0]}:{lineno} [{shape}] {text}")
    return 1


def _enforce_diff_scoped() -> int:
    root = _repo_root()
    if root is None:
        print("  (not inside a work tree — diff-scoped enforcement skipped)")
        return 0
    added: list[tuple[str, str, str, str, int]] = []
    for rel in _changed_py_files(root):
        if not Path(root, rel).is_file():
            continue
        added.extend(
            (rel, exc_types, shape, text, count)
            for exc_types, shape, text, count in _added_findings(root, rel)
        )
    return _report_added(added) if added else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "root",
        nargs="?",
        help=(
            "Scan this path instead of agent_utilities/, reporting EVERY "
            "finding with no diff scoping — for an arbitrary fixture/test "
            "directory, mirroring check_identifier_interpolation.py."
        ),
    )
    ap.add_argument(
        "--update-baseline",
        action="store_true",
        help=argparse.SUPPRESS,
    )
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

    current = scan()
    _print_census(current)
    hard_zero = _report_hard_zero(current)
    diff_scoped = _enforce_diff_scoped()
    if hard_zero or diff_scoped:
        return 1
    print("OK — no swallowed-error site added by this change.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
