#!/usr/bin/env python3
"""Removed/renamed-public-symbol-vs-fleet-consumers gate (heavy/pre-push tier).

Two real breaks landed in agent-utilities on 2026-08-13, both caught by
accident rather than by any control:

1. Commit ``7d83cd42`` (B-19) deleted the public symbol
   ``agent_utilities.mcp.server_factory.protect_stdio_jsonrpc`` without
   migrating its consumers. ``agents/systems-manager`` imported it at MODULE
   scope (the whole MCP server raised ``ImportError`` at import and could not
   start); ``agents/tunnel-manager`` imported it inside a function (stdio
   transport broke). Found only because an unrelated ``uv`` sweep tripped
   over the ``ImportError``. Fixed in ``966fe1e`` / ``2afec44``.
2. The GOC-73 ``[graphos]`` extra split would have broken 65 repos importing
   ``knowledge_graph.memory.native_ingest``, because a package ``__init__``
   eagerly imported a module that newly required an optional dependency.
   Caught only by a manual blast-radius check.

agent-utilities is the producer for ~70 fleet consumers (``agents/*`` +
``skills/*``); this gate belongs here — the chokepoint — not replicated in
each consumer.

Two checks, run together, reported separately
------------------------------------------------
**Check 1 — removed/renamed public symbol (the primary check).** Compares the
resolved public API EXPOSURE (module-level ``def``/``class``/assignment not
prefixed with ``_``, plus anything in ``__all__``, plus names re-exported by
a top-level ``from x import y`` resolved TRANSITIVELY through facade modules
— see ``_symbol_surface.py``'s ``resolve_exposure``) of the working tree
against a base ref (``--base-ref``, default ``origin/main``). Any base-ref
exposed symbol absent from the same module at HEAD — because it was deleted
OR renamed (a rename is indistinguishable from a delete-plus-add under this
definition, and IS meant to be caught the same way), because the whole
module file was deleted/renamed, or because an upstream module a facade
re-exports from lost the symbol — is cross-referenced against the committed
fleet consumer index (``scripts/fleet_symbol_consumers.json``, built by
``gen_fleet_symbol_consumers.py``). A hit fails the gate and names the
symbol, the base ref, and every consuming repo + file + line.

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
import tomllib
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _symbol_surface import (  # noqa: E402
    ModuleSurface,
    module_dotted_name,
    parse_module_surface,
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
    """Public surface of ``package_name/`` at a git ref (via ``git ls-tree``/``show``)."""
    ls = _git(tree, "ls-tree", "-r", "--name-only", ref, "--", package_name)
    if ls.returncode != 0:
        raise RuntimeError(
            f"git ls-tree failed for ref {ref!r} in {tree}: {ls.stderr.strip()}"
        )
    out: dict[str, ModuleSurface] = {}
    for rel in ls.stdout.splitlines():
        rel = rel.strip()
        if not rel.endswith(".py"):
            continue
        if rel.startswith(f"{package_name}/tests/") or "/tests/" in rel:
            continue
        show = _git(tree, "show", f"{ref}:{rel}")
        dotted = module_dotted_name(Path(package_name), Path(rel), package_name)
        if show.returncode != 0:
            out[dotted] = ModuleSurface(
                dotted=dotted, relpath=rel, parse_error=show.stderr.strip()
            )
            continue
        out[dotted] = parse_module_surface(show.stdout, dotted, rel, package_name)
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
    base: dict[str, ModuleSurface], head: dict[str, ModuleSurface]
) -> list[tuple[str, str, str]]:
    """Return ``(fq_target, module, symbol)`` for every symbol exposed as
    ``module.symbol`` at the base ref but not at HEAD — computed on the
    RESOLVED exposure (``resolve_exposure``, own defs + transitively-resolved
    re-exports), not raw per-file definitions, so a facade module whose
    upstream re-export target lost the symbol is caught too. Module deleted
    entirely counts: every one of its base-exposed symbols is "removed".
    """
    base_exposure = resolve_exposure(base)
    head_exposure = resolve_exposure(head)
    removed: list[tuple[str, str, str]] = []
    for module, base_symbols in base_exposure.items():
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
        "--tree", type=Path, default=ROOT, help="Git worktree to scan (default: repo root)."
    )
    parser.add_argument(
        "--base-ref", default="origin/main", help="Base ref to diff against (default: origin/main)."
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
    parser.add_argument("--package", default=DEFAULT_PACKAGE, help="Package dir name to scan.")
    parser.add_argument(
        "--consumer-index", type=Path, default=None, help="Path to fleet_symbol_consumers.json."
    )
    parser.add_argument(
        "--max-index-age-days", type=int, default=DEFAULT_MAX_INDEX_AGE_DAYS
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    tree = args.tree.resolve()
    index_path = args.consumer_index or (tree / "scripts" / "fleet_symbol_consumers.json")

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

    removed = find_removed_symbols(base_surface, head_surface)

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

    if args.json:
        print(
            json.dumps(
                {
                    "base_ref": args.base_ref,
                    "resolved_base_ref": resolved_base,
                    "removed_symbols_total": len(removed),
                    "eager_drift_modules_total": len(drift),
                    "violations": violations,
                },
                indent=2,
            )
        )
    else:
        print(
            f"base ref {args.base_ref} -> {resolved_base}; "
            f"{len(removed)} removed/renamed public symbol(s), "
            f"{len(drift)} module(s) with new eager-optional-dependency imports"
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
            else:
                print(
                    f"\nMODULE GAINED EAGER IMPORT OF OPTIONAL-EXTRA PACKAGE(S), "
                    f"STILL IMPORTED BY FLEET CONSUMERS:"
                    f"\n  {v['module']}"
                    f"\n    newly eager: {', '.join(v['new_optional_packages'])}"
                    f"\n    consumers:"
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

    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
