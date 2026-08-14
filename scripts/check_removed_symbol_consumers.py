#!/usr/bin/env python3
"""Removed/renamed/contract-changed-public-symbol-vs-fleet-consumers gate
(heavy/pre-push tier).

Four real breaks landed in agent-utilities within 24 hours (2026-08-13/14),
found by accident rather than by any control, of which only the first had a
control at the time:

1. Commit ``7d83cd42`` (B-19) deleted the public symbol
   ``agent_utilities.mcp.server_factory.protect_stdio_jsonrpc`` without
   migrating its consumers. ``agents/systems-manager`` imported it at MODULE
   scope (the whole MCP server raised ``ImportError`` at import and could not
   start); ``agents/tunnel-manager`` imported it inside a function (stdio
   transport broke). Found only because an unrelated ``uv`` sweep tripped
   over the ``ImportError``. Fixed in ``966fe1e`` / ``2afec44``. — Check 1.
2. The GOC-73 ``[graphos]`` extra split would have broken 65 repos importing
   ``knowledge_graph.memory.native_ingest``, because a package ``__init__``
   eagerly imported a module that newly required an optional dependency.
   Caught only by a manual blast-radius check. — Check 2.
3. ``native_ingest`` changed its RUNTIME CONTRACT: it started requiring an
   injected client exposing ``.supports()``/``.changes``/``.nodes``/``.rdf``
   and raising otherwise. Broke ``test_kg_ingest.py`` in 65 repos. The symbol
   itself never disappeared, so Checks 1/2 could not see it — see Check 3's
   docstring below for exactly how far static analysis can and cannot follow
   this class of break.
4. ``agent_utilities.http`` was renamed to ``agent_utilities.httpsupport``
   (commit ``8138dbad``, 2026-08-01) — a whole PACKAGE rename, not a symbol
   removal. ``portainer-mcp`` CrashLoopBackOff'd in production (19 restarts);
   ``kafka-mcp`` was latent (broke on next restart). Fixed downstream in
   ``9da27a0``/``8d88f42``. Check 1 already catches MOST of this shape
   incidentally (a whole-module-gone module reports every symbol the facade
   used to expose as "removed"), but misses a BARE whole-module import
   (``import agent_utilities.http``, with no trailing symbol — the consumer
   index records that under the bare module path, which Check 1 never
   queries) and a module with no public symbols of its own. Check 4 closes
   that gap and OWNS this shape's reporting (Check 1 defers to it for any
   module confirmed entirely gone, so the same break is not reported twice
   under two different messages — see "Fails closed" below and CONCEPT
   AU-OS.governance re: "one rule, one message").

agent-utilities is the producer for ~70 fleet consumers (``agents/*`` +
``skills/*``); this gate belongs here — the chokepoint — not replicated in
each consumer.

Five checks, run together, reported separately
------------------------------------------------
**Check 1 — removed/renamed public symbol (the primary check).** Compares the
resolved public API EXPOSURE (module-level ``def``/``class``/assignment not
prefixed with ``_``, plus anything in ``__all__``, plus names re-exported by
a top-level ``from x import y`` resolved TRANSITIVELY through facade modules
— see ``_symbol_surface.py``'s ``resolve_exposure``) of the working tree
against a base ref (``--base-ref``, default ``origin/main``). Any base-ref
exposed symbol absent from the same module at HEAD — because it was deleted
OR renamed (a rename is indistinguishable from a delete-plus-add under this
definition, and IS meant to be caught the same way), or because an upstream
module a facade re-exports from lost the symbol — is cross-referenced
against the committed fleet consumer index
(``scripts/fleet_symbol_consumers.json``, built by
``gen_fleet_symbol_consumers.py``). A hit fails the gate and names the
symbol, the base ref, and every consuming repo + file + line. Symbols
belonging to a module Check 4 has already confirmed is ENTIRELY gone are
skipped here (Check 4 owns and reports that shape, with a superset of
consumers — see break #4 above).

**Check 2 — transitive eager-import cost drift (narrower, explicitly bounded).**
For each module present at both refs, computes the set of external (non-
stdlib, non-``agent_utilities``) packages imported *unconditionally at module-
import time* (top-level ``import``/``from import`` statements only — not
inside ``try:`` or ``if TYPE_CHECKING:``, which are the standard "this is
optional" shapes). Classifies each package as "optional-only" at a given ref
by reading ``pyproject.toml``'s ``[project.dependencies]`` vs
``[project.optional-dependencies]`` at that ref. Flags a module that gained
an eager import of a package that is optional-only at HEAD but either wasn't
imported at all, or wasn't optional, at the base ref — IF that module (or
anything under it) has a recorded fleet consumer.

Check 2 does **NOT** cover: dynamic/string-composed imports
(``importlib.import_module(f"...")``, ``__import__``) — the exact mechanism
GOC-73's actual numeric-kernel loader uses, which is therefore invisible to
this static check (same documented blind spot as ``check_wiring.py``);
conditional imports that run on some-but-not-all *code paths* within a
function (only whole-module eager cost is modeled); or a dependency's own
transitive dependency changes (only directly-imported top-level names are
inspected). Treat a Check-2 pass as "no NEW eagerly-declared optional import
was added", not "this module's total import cost is unchanged".

**Check 3 — module-level function signature/contract change.** For every
module-level public function reachable as ``module.name`` at BOTH refs
(own definition or through a re-export facade, transitively resolved — see
``_symbol_surface.resolve_callable_surface``), diffs its caller-visible
parameter list: a parameter removed or renamed, an existing
positional-or-keyword parameter that became keyword-only (or was absorbed
into ``**kwargs``), a parameter that lost its default (became required), a
BRAND NEW parameter with no default (newly required — breaks every existing
caller), and a changed default-value expression (a softer signal — a caller
that doesn't pass that argument now gets different behavior). Cross-
referenced against the consumer index the same way as Check 1.

Check 3 is explicitly, honestly bounded to what a parameter-list AST diff can
see. It does **NOT** and cannot cover a RUNTIME/DUCK-TYPED contract change —
break #3 above (``native_ingest`` requiring an injected object to expose
``.supports()``/``.changes``/``.nodes``/``.rdf``) changed no parameter name,
kind, or default; the function still took the same arguments and raised only
once it *used* one at runtime. No static AST diff of a signature can see
that, and this gate does not claim to. It also does not cover: method
signatures on classes (see Check 5's scope), decorator-rewritten signatures
(a ``functools.wraps`` wrapper declared as ``(*args, **kwargs)`` is read
literally), or a parameter's TYPE narrowing/widening (only name/kind/default
are compared, never annotations).

**Check 4 — module-path rename/removal.** Computes the set of dotted module
paths present at the base ref's raw (unresolved) surface but absent at
HEAD's — the source file (or whole package directory) was deleted, moved, or
renamed away, not merely edited internally. For each such module, every
consumer-index entry whose target is that bare module path OR starts with
`"<module>."` (bare whole-module imports, symbol imports through it, AND
imports of its own submodules — see ``_consumers_for_module_prefix``, already
used by Check 2) is a violation. This is the ONLY check that catches a bare
``import agent_utilities.http`` with no symbol, and a module that exposes no
public symbols of its own (so Check 1 never had a `module.symbol` pair to
diff in the first place). See break #4 above — this is the check that would
have caught the ``http`` → ``httpsupport`` rename outright, without relying
on Check 1's incidental (and narrower) coverage of the same event.

**Check 5 — class/attribute surface (method/attribute removal).** For every
module-level public class reachable as ``module.name`` at BOTH refs (own
definition or through a re-export facade, transitively resolved — see
``_symbol_surface.resolve_class_surface``), diffs its set of public
method/attribute NAMES (``__init__``/``__call__`` included as genuine
contract surface; other dunders excluded). A method or attribute present at
base and gone at HEAD is a violation IF the class itself has a recorded
fleet consumer (i.e. some consumer imports the class) — cross-referenced at
CLASS-import granularity, because the consumer index records IMPORT
STATEMENTS, not attribute/method usage; it cannot confirm any given consumer
calls the specific removed member, only that the class is consumed
somewhere. Check 5 does NOT diff method signatures (only presence — see
Check 3 for the module-function-only signature diff) and does not catch a
whole class being renamed/removed (Check 1 already does — a class name is
itself a public symbol).

Fails closed on stale/missing data
------------------------------------
If the consumer index is missing, unparseable, or older than
``--max-index-age-days`` (default 30), the gate FAILS LOUDLY rather than
skipping — a gate that silently no-ops when its data is absent is the exact
anti-pattern this program keeps finding (three gates one session looked green
while enforcing nothing: never discovered, crashing, or blind). Regenerate
with ``python3 scripts/gen_fleet_symbol_consumers.py --update`` (needs the
fleet checked out locally — see that script's docstring for its own,
different staleness source: repos not present on this box are silently
absent from the index it produces, which is why age alone is not sufficient
and the index also records which repos it managed to scan).

Deliberately stdlib-only (see ``_symbol_surface.py``) — no ``import
agent_utilities``, so no managed-venv re-exec dance is needed.

Usage::

    python3 scripts/check_removed_symbol_consumers.py
    python3 scripts/check_removed_symbol_consumers.py --base-ref 7d83cd42^ \\
        --tree /path/to/checkout/at/7d83cd42
    python3 scripts/check_removed_symbol_consumers.py --json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _symbol_surface import (  # noqa: E402
    CallableSurface,
    ClassSurface,
    ModuleSurface,
    module_dotted_name,
    parse_module_surface,
    resolve_callable_surface,
    resolve_class_surface,
    resolve_exposure,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PACKAGE = "agent_utilities"
DEFAULT_INDEX = ROOT / "scripts" / "fleet_symbol_consumers.json"
DEFAULT_MAX_INDEX_AGE_DAYS = 30


# ---------------------------------------------------------------------------
# Git-ref-aware surface extraction
# ---------------------------------------------------------------------------


def _git(tree: Path, *args: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(tree), *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def _resolve_ref(tree: Path, ref: str) -> str | None:
    res = _git(tree, "rev-parse", "--verify", f"{ref}^{{commit}}")
    if res.returncode != 0:
        return None
    return res.stdout.strip()


def surface_at_ref(tree: Path, ref: str, package_name: str) -> dict[str, ModuleSurface]:
    """Public surface of ``package_name/`` at a git ref.

    Performance: reads every file's blob in ONE ``git cat-file --batch`` call
    (fed the blob SHAs from a single ``git ls-tree``), not one ``git show``
    subprocess per file. This is the single biggest cost in the whole gate —
    with ~1,500 ``.py`` files under ``agent_utilities/``, spawning a process
    per file (the original implementation) costs ~15-20ms of pure process
    overhead each, ~25-30s total; batching the same reads into one long-lived
    ``git cat-file --batch`` process cuts that to under a second. Correctness
    is unchanged: same file set, same content, same ``parse_module_surface``
    call per file — only the I/O strategy differs.
    """
    ls = _git(tree, "ls-tree", "-r", ref, "--", package_name)
    if ls.returncode != 0:
        raise RuntimeError(
            f"git ls-tree failed for ref {ref!r} in {tree}: {ls.stderr.strip()}"
        )
    entries: list[tuple[str, str]] = []  # (blob_sha, repo-relative path)
    for line in ls.stdout.splitlines():
        if "\t" not in line:
            continue
        meta, rel = line.split("\t", 1)
        meta_parts = meta.split(" ")
        if len(meta_parts) != 3:
            continue
        _mode, obj_type, sha = meta_parts
        if obj_type != "blob" or not rel.endswith(".py"):
            continue
        if rel.startswith(f"{package_name}/tests/") or "/tests/" in rel:
            continue
        entries.append((sha, rel))

    out: dict[str, ModuleSurface] = {}
    if not entries:
        return out

    batch_input = ("\n".join(sha for sha, _ in entries) + "\n").encode("utf-8")
    proc = subprocess.run(
        ["git", "-C", str(tree), "cat-file", "--batch"],
        input=batch_input,
        capture_output=True,
        check=False,
        timeout=120,
    )
    buf = proc.stdout
    pos = 0
    for sha, rel in entries:
        dotted = module_dotted_name(Path(package_name), Path(rel), package_name)
        nl = buf.find(b"\n", pos)
        if nl == -1:
            out[dotted] = ModuleSurface(
                dotted=dotted,
                relpath=rel,
                parse_error="git cat-file --batch: truncated output",
            )
            break
        header = buf[pos:nl].decode("utf-8", errors="replace")
        pos = nl + 1
        header_parts = header.split(" ")
        if (
            len(header_parts) != 3
            or header_parts[0] != sha
            or header_parts[1] != "blob"
        ):
            out[dotted] = ModuleSurface(
                dotted=dotted,
                relpath=rel,
                parse_error=f"git cat-file --batch: unexpected header {header!r}",
            )
            continue
        size = int(header_parts[2])
        content = buf[pos : pos + size]
        pos += size
        if buf[pos : pos + 1] == b"\n":
            pos += 1  # cat-file --batch appends one trailing newline per object
        try:
            source = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            out[dotted] = ModuleSurface(
                dotted=dotted, relpath=rel, parse_error=str(exc)
            )
            continue
        out[dotted] = parse_module_surface(source, dotted, rel, package_name)
    return out


def surface_on_disk(tree: Path, package_name: str) -> dict[str, ModuleSurface]:
    from _symbol_surface import surface_from_worktree  # local, avoid top-level dup

    return surface_from_worktree(tree / package_name, package_name)


# ---------------------------------------------------------------------------
# pyproject.toml dependency classification
# ---------------------------------------------------------------------------


def _normalize_pkg(name: str) -> str:
    for sep in ("[", ";", "=", "<", ">", "!", "~", " "):
        name = name.split(sep, 1)[0]
    return name.strip().lower().replace("_", "-")


def _classify_dependencies(pyproject_text: str) -> dict[str, dict]:
    """``{normalized_pkg_name: {"in_base": bool, "extras": set[str]}}``."""
    try:
        data = tomllib.loads(pyproject_text)
    except tomllib.TOMLDecodeError:
        return {}
    project = data.get("project", {}) or {}
    table: dict[str, dict] = {}
    for raw in project.get("dependencies", []) or []:
        pkg = _normalize_pkg(str(raw))
        table.setdefault(pkg, {"in_base": False, "extras": set()})
        table[pkg]["in_base"] = True
    for extra, deps in (project.get("optional-dependencies", {}) or {}).items():
        for raw in deps or []:
            pkg = _normalize_pkg(str(raw))
            table.setdefault(pkg, {"in_base": False, "extras": set()})
            table[pkg]["extras"].add(extra)
    return table


def _pyproject_text_at_ref(tree: Path, ref: str) -> str | None:
    res = _git(tree, "show", f"{ref}:pyproject.toml")
    return res.stdout if res.returncode == 0 else None


def _import_name_to_pkg_guess(import_name: str) -> str:
    """Best-effort import-name -> distribution-name guess (documented limitation)."""
    return import_name.replace("_", "-").lower()


# ---------------------------------------------------------------------------
# Consumer index
# ---------------------------------------------------------------------------


class IndexError_(RuntimeError):
    """Raised for a missing/unparseable/stale consumer index — always fatal."""


def load_consumer_index(path: Path, max_age_days: int) -> dict:
    if not path.exists():
        raise IndexError_(
            f"fleet consumer index not found at {path}. This gate fails CLOSED "
            "rather than silently skipping — regenerate with "
            "`python3 scripts/gen_fleet_symbol_consumers.py --update` "
            "(needs the agent-packages fleet checked out locally)."
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IndexError_(
            f"fleet consumer index at {path} is not valid JSON ({exc}); a "
            "corrupt index is treated as absent, not as zero consumers. "
            "Regenerate with `gen_fleet_symbol_consumers.py --update`."
        ) from exc

    generated_at = data.get("generated_at")
    if not generated_at:
        raise IndexError_(
            f"fleet consumer index at {path} has no 'generated_at' timestamp; "
            "cannot verify freshness, so it is treated as stale. Regenerate."
        )
    try:
        taken = datetime.fromisoformat(generated_at)
    except ValueError as exc:
        raise IndexError_(
            f"fleet consumer index at {path} has an unparseable 'generated_at' "
            f"({generated_at!r}: {exc}). Regenerate."
        ) from exc
    if taken.tzinfo is None:
        taken = taken.replace(tzinfo=UTC)
    age_days = (datetime.now(UTC) - taken).total_seconds() / 86400.0
    if age_days > max_age_days:
        raise IndexError_(
            f"fleet consumer index at {path} is {age_days:.1f} days old "
            f"(threshold {max_age_days}); treated as stale and therefore "
            "UNSAFE to trust — a removed symbol could have gained consumers "
            "since. Regenerate with `gen_fleet_symbol_consumers.py --update`."
        )
    if "consumers" not in data:
        raise IndexError_(
            f"fleet consumer index at {path} has no 'consumers' key; malformed. "
            "Regenerate."
        )
    return data


# ---------------------------------------------------------------------------
# Check 1 — removed/renamed public symbol
# ---------------------------------------------------------------------------


def find_removed_symbols(
    base: dict[str, ModuleSurface],
    head: dict[str, ModuleSurface],
    exclude_modules: frozenset[str] = frozenset(),
) -> list[tuple[str, str, str]]:
    """Return ``(fq_target, module, symbol)`` for every symbol exposed as
    ``module.symbol`` at the base ref but not at HEAD — computed on the
    RESOLVED exposure (``resolve_exposure``, own defs + transitively-resolved
    re-exports), not raw per-file definitions, so a facade module whose
    upstream re-export target lost the symbol is caught too.

    ``exclude_modules`` is the set of dotted module paths Check 4
    (``find_removed_modules``) has already confirmed are ENTIRELY gone at
    HEAD (the source file/package itself deleted or renamed away, not just
    edited) — those are reported once, by Check 4, with a superset of
    consumers (it also catches bare whole-module imports Check 1 structurally
    cannot see); skipping them here keeps each break to one message instead
    of two differently-worded ones for the same event.
    """
    base_exposure = resolve_exposure(base)
    head_exposure = resolve_exposure(head)
    removed: list[tuple[str, str, str]] = []
    for module, base_symbols in base_exposure.items():
        if module in exclude_modules:
            continue
        if base.get(module) is not None and base[module].parse_error:
            continue
        head_symbols = head_exposure.get(module, frozenset())
        for symbol in sorted(base_symbols - head_symbols):
            removed.append((f"{module}.{symbol}", module, symbol))
    return removed


# ---------------------------------------------------------------------------
# Check 2 — transitive eager-import cost drift
# ---------------------------------------------------------------------------


def find_eager_dependency_drift(
    base: dict[str, ModuleSurface],
    head: dict[str, ModuleSurface],
    base_deps: dict[str, dict],
    head_deps: dict[str, dict],
) -> list[tuple[str, list[str]]]:
    """Return ``(module, [newly-eager optional package names])`` for modules
    present at both refs whose head-eager-external-package set gained a
    package that is optional-only (in some extra, not in base deps) at HEAD
    and was not both present-and-optional at the base ref.
    """

    def is_optional_only(pkg_guess: str, deps: dict[str, dict]) -> bool:
        entry = deps.get(pkg_guess)
        if entry is None:
            return False  # unknown package: cannot claim it's optional-only
        return bool(entry["extras"]) and not entry["in_base"]

    out: list[tuple[str, list[str]]] = []
    for module, head_surface in head.items():
        base_surface = base.get(module)
        if base_surface is None or head_surface.parse_error or base_surface.parse_error:
            continue
        new_pkgs: list[str] = []
        for imp in sorted(head_surface.eager_external_packages):
            guess = _import_name_to_pkg_guess(imp)
            head_optional = is_optional_only(guess, head_deps)
            if not head_optional:
                continue
            was_eager_before = imp in base_surface.eager_external_packages
            was_optional_before = is_optional_only(guess, base_deps)
            if not (was_eager_before and was_optional_before):
                new_pkgs.append(guess)
        if new_pkgs:
            out.append((module, new_pkgs))
    return out


# ---------------------------------------------------------------------------
# Check 3 — module-level function signature/contract change
# ---------------------------------------------------------------------------


def _diff_signature(base_fn: CallableSurface, head_fn: CallableSurface) -> list[str]:
    """Human-readable reasons a function's caller-visible signature changed in
    a way an existing caller could not survive unmodified. See Check 3's
    module-docstring section for exactly what this can and cannot see.
    """
    reasons: list[str] = []
    base_by_name = {p.name: p for p in base_fn.params}
    head_by_name = {p.name: p for p in head_fn.params}

    for name, base_p in base_by_name.items():
        if base_p.kind in ("VAR_POSITIONAL", "VAR_KEYWORD"):
            continue
        head_p = head_by_name.get(name)
        if head_p is None:
            reasons.append(f"parameter '{name}' removed or renamed")
            continue
        if base_p.kind == "POSITIONAL_OR_KEYWORD" and head_p.kind == "KEYWORD_ONLY":
            reasons.append(
                f"parameter '{name}' became keyword-only (was positional-or-keyword)"
            )
        elif base_p.kind in (
            "POSITIONAL_ONLY",
            "POSITIONAL_OR_KEYWORD",
        ) and head_p.kind in (
            "VAR_KEYWORD",
            "VAR_POSITIONAL",
        ):
            reasons.append(f"parameter '{name}' removed (absorbed into *args/**kwargs)")
        if base_p.has_default and not head_p.has_default:
            reasons.append(f"parameter '{name}' lost its default (became required)")
        elif (
            base_p.has_default
            and head_p.has_default
            and base_p.default_repr != head_p.default_repr
        ):
            reasons.append(
                f"parameter '{name}' default changed ({base_p.default_repr!r} -> {head_p.default_repr!r})"
            )

    for name, head_p in head_by_name.items():
        if head_p.kind in ("VAR_POSITIONAL", "VAR_KEYWORD"):
            continue
        if name not in base_by_name and not head_p.has_default:
            reasons.append(f"new required parameter '{name}'")

    return reasons


def find_signature_breaks(
    base: dict[str, ModuleSurface], head: dict[str, ModuleSurface]
) -> list[tuple[str, list[str]]]:
    """Return ``(fq_target, [reasons])`` for every module-level public
    function reachable as ``module.name`` at BOTH refs (own definition or
    resolved transitively through a re-export facade — see
    ``resolve_callable_surface``) whose caller-visible parameter list changed.
    A function absent at HEAD is a removal, handled by Check 1/Check 4, not
    here (this only compares functions present on both sides).
    """
    base_callables = resolve_callable_surface(base)
    head_callables = resolve_callable_surface(head)
    out: list[tuple[str, list[str]]] = []
    for module, head_syms in head_callables.items():
        base_syms = base_callables.get(module, {})
        for name, head_fn in head_syms.items():
            base_fn = base_syms.get(name)
            if base_fn is None:
                continue  # new function, or its module/facade is new — not a break
            reasons = _diff_signature(base_fn, head_fn)
            if reasons:
                out.append((f"{module}.{name}", reasons))
    return out


# ---------------------------------------------------------------------------
# Check 4 — module-path rename/removal
# ---------------------------------------------------------------------------


def find_removed_modules(
    base: dict[str, ModuleSurface], head: dict[str, ModuleSurface]
) -> list[str]:
    """Dotted module paths present at the base ref's raw surface but entirely
    absent at HEAD's — the source file (or whole package) was deleted, moved,
    or renamed away. Complements Check 1: catches a bare whole-module import
    (``import agent_utilities.http``, recorded in the consumer index under
    the bare module path with no trailing symbol — Check 1 never queries that
    key) and a module that exposes no public symbols of its own (so no
    ``module.symbol`` pair ever existed for Check 1 to diff).
    """
    return sorted(m for m in base if m not in head)


# ---------------------------------------------------------------------------
# Check 5 — class/attribute surface (removed public methods/attributes)
# ---------------------------------------------------------------------------


def _diff_class_members(base_cls: ClassSurface, head_cls: ClassSurface) -> list[str]:
    removed_methods = sorted(base_cls.public_methods - head_cls.public_methods)
    removed_attrs = sorted(base_cls.public_attributes - head_cls.public_attributes)
    return [f"method '{m}'" for m in removed_methods] + [
        f"attribute '{a}'" for a in removed_attrs
    ]


def find_class_surface_removals(
    base: dict[str, ModuleSurface], head: dict[str, ModuleSurface]
) -> list[tuple[str, list[str]]]:
    """Return ``(fq_target, [removed member descriptions])`` for every
    module-level public class reachable as ``module.name`` at BOTH refs (own
    definition or resolved transitively through a re-export facade — see
    ``resolve_class_surface``) whose public method/attribute NAME set shrank.
    A whole class removed/renamed is Check 1's job (a class name is itself a
    public symbol); this only compares classes present on both sides.
    """
    base_classes = resolve_class_surface(base)
    head_classes = resolve_class_surface(head)
    out: list[tuple[str, list[str]]] = []
    for module, head_syms in head_classes.items():
        base_syms = base_classes.get(module, {})
        for name, head_cls in head_syms.items():
            base_cls = base_syms.get(name)
            if base_cls is None:
                continue
            removed = _diff_class_members(base_cls, head_cls)
            if removed:
                out.append((f"{module}.{name}", removed))
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _consumers_for(index: dict, key: str) -> list[dict]:
    return index.get("consumers", {}).get(key, [])


def _consumers_for_module_prefix(index: dict, module: str) -> dict[str, list[dict]]:
    prefix = module + "."
    hits: dict[str, list[dict]] = {}
    for key, cons in index.get("consumers", {}).items():
        if key == module or key.startswith(prefix):
            hits[key] = cons
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tree",
        type=Path,
        default=ROOT,
        help="Git worktree to scan (default: repo root).",
    )
    parser.add_argument(
        "--base-ref",
        default="origin/main",
        help="Base ref to diff against (default: origin/main).",
    )
    parser.add_argument(
        "--head-ref",
        default=None,
        help=(
            "Git ref to treat as HEAD instead of the working tree on disk. Normal "
            "pre-push usage leaves this unset (compares your dirty/staged working "
            "tree against --base-ref); pass it to replay a historical break "
            "entirely from git history, e.g. --base-ref 7d83cd42^ --head-ref 7d83cd42."
        ),
    )
    parser.add_argument(
        "--package", default=DEFAULT_PACKAGE, help="Package dir name to scan."
    )
    parser.add_argument(
        "--consumer-index",
        type=Path,
        default=None,
        help="Path to fleet_symbol_consumers.json.",
    )
    parser.add_argument(
        "--max-index-age-days", type=int, default=DEFAULT_MAX_INDEX_AGE_DAYS
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    t_start = time.monotonic()

    tree = args.tree.resolve()
    index_path = args.consumer_index or (
        tree / "scripts" / "fleet_symbol_consumers.json"
    )

    try:
        index = load_consumer_index(index_path, args.max_index_age_days)
    except IndexError_ as exc:
        print(f"REMOVED-SYMBOL-CONSUMER GATE FAILED (index): {exc}", file=sys.stderr)
        return 1

    resolved_base = _resolve_ref(tree, args.base_ref)
    if resolved_base is None:
        print(
            f"REMOVED-SYMBOL-CONSUMER GATE FAILED: base ref {args.base_ref!r} does not "
            f"resolve in {tree}. Fetch it first (e.g. `git fetch origin main`) — this "
            "gate refuses to silently skip the comparison.",
            file=sys.stderr,
        )
        return 1

    resolved_head = None
    if args.head_ref is not None:
        resolved_head = _resolve_ref(tree, args.head_ref)
        if resolved_head is None:
            print(
                f"REMOVED-SYMBOL-CONSUMER GATE FAILED: head ref {args.head_ref!r} does "
                f"not resolve in {tree}.",
                file=sys.stderr,
            )
            return 1
    else:
        package_dir = tree / args.package
        if not package_dir.is_dir():
            print(
                f"REMOVED-SYMBOL-CONSUMER GATE FAILED: package dir {package_dir} does not exist.",
                file=sys.stderr,
            )
            return 1

    base_surface = surface_at_ref(tree, resolved_base, args.package)
    head_surface = (
        surface_at_ref(tree, resolved_head, args.package)
        if resolved_head is not None
        else surface_on_disk(tree, args.package)
    )

    removed_modules = find_removed_modules(base_surface, head_surface)
    removed = find_removed_symbols(
        base_surface, head_surface, frozenset(removed_modules)
    )
    signature_breaks = find_signature_breaks(base_surface, head_surface)
    class_removals = find_class_surface_removals(base_surface, head_surface)

    base_pyproject = _pyproject_text_at_ref(tree, resolved_base)
    if resolved_head is not None:
        head_pyproject = _pyproject_text_at_ref(tree, resolved_head)
    else:
        head_pyproject_path = tree / "pyproject.toml"
        head_pyproject = (
            head_pyproject_path.read_text(encoding="utf-8")
            if head_pyproject_path.is_file()
            else None
        )
    base_deps = _classify_dependencies(base_pyproject) if base_pyproject else {}
    head_deps = _classify_dependencies(head_pyproject) if head_pyproject else {}
    drift = (
        find_eager_dependency_drift(base_surface, head_surface, base_deps, head_deps)
        if (base_deps and head_deps)
        else []
    )

    violations: list[dict] = []
    for fq_target, module, symbol in removed:
        consumers = _consumers_for(index, fq_target)
        if consumers:
            violations.append(
                {
                    "kind": "removed_symbol",
                    "target": fq_target,
                    "base_ref": args.base_ref,
                    "resolved_base_ref": resolved_base,
                    "consumers": consumers,
                }
            )

    for module, new_pkgs in drift:
        hits = _consumers_for_module_prefix(index, module)
        if hits:
            flat_consumers = [c for cons in hits.values() for c in cons]
            violations.append(
                {
                    "kind": "eager_dependency_drift",
                    "module": module,
                    "new_optional_packages": new_pkgs,
                    "consumers": flat_consumers,
                }
            )

    for fq_target, reasons in signature_breaks:
        consumers = _consumers_for(index, fq_target)
        if consumers:
            violations.append(
                {
                    "kind": "signature_change",
                    "target": fq_target,
                    "base_ref": args.base_ref,
                    "resolved_base_ref": resolved_base,
                    "reasons": reasons,
                    "consumers": consumers,
                }
            )

    for module in removed_modules:
        hits = _consumers_for_module_prefix(index, module)
        if hits:
            flat_consumers = sorted(
                {
                    (c["repo"], c["file"], c["line"])
                    for cons in hits.values()
                    for c in cons
                }
            )
            violations.append(
                {
                    "kind": "module_removed",
                    "module": module,
                    "base_ref": args.base_ref,
                    "resolved_base_ref": resolved_base,
                    "consumers": [
                        {"repo": r, "file": f, "line": ln}
                        for (r, f, ln) in flat_consumers
                    ],
                }
            )

    for fq_target, removed_members in class_removals:
        consumers = _consumers_for(index, fq_target)
        if consumers:
            violations.append(
                {
                    "kind": "class_surface_removed",
                    "target": fq_target,
                    "base_ref": args.base_ref,
                    "resolved_base_ref": resolved_base,
                    "removed_members": removed_members,
                    "consumers": consumers,
                }
            )

    elapsed_s = time.monotonic() - t_start

    if args.json:
        print(
            json.dumps(
                {
                    "base_ref": args.base_ref,
                    "resolved_base_ref": resolved_base,
                    "removed_symbols_total": len(removed),
                    "eager_drift_modules_total": len(drift),
                    "signature_breaks_total": len(signature_breaks),
                    "removed_modules_total": len(removed_modules),
                    "class_surface_removals_total": len(class_removals),
                    "violations": violations,
                    "elapsed_seconds": round(elapsed_s, 3),
                },
                indent=2,
            )
        )
    else:
        print(
            f"base ref {args.base_ref} -> {resolved_base}; "
            f"{len(removed)} removed/renamed public symbol(s), "
            f"{len(drift)} module(s) with new eager-optional-dependency imports, "
            f"{len(signature_breaks)} function signature change(s), "
            f"{len(removed_modules)} module(s) removed/renamed, "
            f"{len(class_removals)} class(es) with removed public member(s)"
        )
        for v in violations:
            if v["kind"] == "removed_symbol":
                print(
                    f"\nREMOVED PUBLIC SYMBOL STILL IMPORTED BY FLEET CONSUMERS:"
                    f"\n  {v['target']}"
                    f"\n    existed at base ref {v['base_ref']} ({v['resolved_base_ref']})"
                    f"\n    consumers:"
                )
                for c in v["consumers"]:
                    print(f"      - {c['repo']}: {c['file']}:{c['line']}")
            elif v["kind"] == "eager_dependency_drift":
                print(
                    f"\nMODULE GAINED EAGER IMPORT OF OPTIONAL-EXTRA PACKAGE(S), "
                    f"STILL IMPORTED BY FLEET CONSUMERS:"
                    f"\n  {v['module']}"
                    f"\n    newly eager: {', '.join(v['new_optional_packages'])}"
                    f"\n    consumers:"
                )
                for c in v["consumers"]:
                    print(f"      - {c['repo']}: {c['file']}:{c['line']}")
            elif v["kind"] == "signature_change":
                print(
                    f"\nPUBLIC FUNCTION SIGNATURE/CONTRACT CHANGED, "
                    f"STILL IMPORTED BY FLEET CONSUMERS:"
                    f"\n  {v['target']}"
                    f"\n    existed at base ref {v['base_ref']} ({v['resolved_base_ref']})"
                    f"\n    changes:"
                )
                for r in v["reasons"]:
                    print(f"      - {r}")
                print("    consumers:")
                for c in v["consumers"]:
                    print(f"      - {c['repo']}: {c['file']}:{c['line']}")
            elif v["kind"] == "module_removed":
                print(
                    f"\nMODULE REMOVED/RENAMED, STILL IMPORTED BY FLEET CONSUMERS:"
                    f"\n  {v['module']}"
                    f"\n    existed at base ref {v['base_ref']} ({v['resolved_base_ref']})"
                    f"\n    consumers:"
                )
                for c in v["consumers"]:
                    print(f"      - {c['repo']}: {c['file']}:{c['line']}")
            else:  # class_surface_removed
                print(
                    f"\nCLASS LOST PUBLIC METHOD(S)/ATTRIBUTE(S), "
                    f"CLASS STILL IMPORTED BY FLEET CONSUMERS:"
                    f"\n  {v['target']}"
                    f"\n    existed at base ref {v['base_ref']} ({v['resolved_base_ref']})"
                    f"\n    removed:"
                )
                for m in v["removed_members"]:
                    print(f"      - {m}")
                print(
                    "    consumers (import the CLASS; usage of the specific "
                    "removed member is not verifiable statically):"
                )
                for c in v["consumers"]:
                    print(f"      - {c['repo']}: {c['file']}:{c['line']}")
        if violations:
            print(
                f"\n{len(violations)} violation(s). Migrate the named consumers "
                "before this lands, or restore/re-export the symbol."
            )
        else:
            print("no violations")
        print(f"\ngate runtime: {elapsed_s:.2f}s")

    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
