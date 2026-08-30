#!/usr/bin/env python3
"""AST-based caller-discovery helper (D-ORC-12).

The codebase's standing rule for any "enforce at the chokepoint" fix is
*grep every caller before wiring/removing a control* -- adopted after a
control wired at one entrypoint shipped and changed literally nothing
because six other callers bypassed it. But a plain text/regex grep for a
symbol name is unsound: it has silently missed real call sites three times
in this program alone --

  1. ``check_http_egress_boundary.py`` was blind to 2 of 16 blocked-client
     patterns because an unaliased dotted ``import a.b.c`` was stored under
     the wrong alias key (fixed; see that script's own comments).
  2. A caller grep for ``secured_reads.scope`` missed three real call sites
     that used ``import secured_reads as sr``, a bare
     ``from ... import secured_reads``, and
     ``monkeypatch.setattr(secured_reads, "scope", ...)``.
  3. The same shape recurred across the write-subset sites in the same lane.

Since every chokepoint fix is *validated* by its caller grep, an unsound
grep makes the validation unsound -- exactly the failure the rule exists to
prevent. This script is the generalized, reusable replacement: it resolves
imports/aliases/attribute chains statically via ``ast``, rather than
matching bare text, and additionally flags the two dynamic-dispatch shapes
that a static walk can only ever surface heuristically:
``monkeypatch.setattr(obj_or_module, "name", ...)`` and
``getattr(obj_or_module, "name")`` where the string literal matches the
symbol's simple name.

This does NOT claim perfect soundness (dynamic dispatch through a computed
string, ``**kwargs`` re-dispatch, or reflection defeats any static tool) --
but it closes the specific, repeatedly-observed gap: aliased imports.

Usage::

    python3 scripts/find_callers.py agent_utilities.security.threat_defense_engine.GuardrailEngine
    python3 scripts/find_callers.py secured_reads.scope --roots agent_utilities,tests
    python3 scripts/find_callers.py agent_utilities.mcp.kg_server._get_engine --json

Exit status is always 0 (this is a discovery tool, not a gate); pipe to
``--json`` for machine consumption.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

try:
    from .find_callers_ast import Hit, SearchContext, _file_hits, _FileImports
except ImportError:  # pragma: no cover - direct ``python scripts/find_callers.py``
    from find_callers_ast import Hit, SearchContext, _file_hits, _FileImports

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ROOTS = ("agent_utilities", "scripts", "tests", "examples")


def _collect_imports(tree: ast.AST) -> _FileImports:
    fi = _FileImports()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                if item.asname:
                    fi.aliases[item.asname] = item.name
                    fi.module_aliases[item.asname] = item.name
                else:
                    root = item.name.split(".")[0]
                    fi.aliases[root] = root
                    fi.module_aliases[item.name] = item.name
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            mod = "." * node.level + node.module
            for item in node.names:
                bound = item.asname or item.name
                if item.name == "*":
                    continue
                fi.aliases[bound] = f"{mod}.{item.name}"
                fi.module_aliases[bound] = mod
    return fi


def _python_files(roots: tuple[str, ...], repo_root: Path) -> list[Path]:
    files: list[Path] = []
    for root_name in roots:
        root_dir = repo_root / root_name
        if root_dir.exists():
            files.extend(root_dir.rglob("*.py"))
    return files


def _parse_candidate(path: Path, simple_name: str) -> ast.AST | None:
    if "/.venv/" in str(path) or "/target-isolated/" in str(path):
        return None
    try:
        src = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None
    if simple_name not in src:
        return None
    try:
        return ast.parse(src, filename=str(path))
    except SyntaxError:
        return None


def _first_alias(aliases: dict[str, str], target: str) -> str | None:
    return next(
        (local for local, imported in aliases.items() if imported == target), None
    )


def find_callers(
    symbol: str,
    roots: tuple[str, ...] = DEFAULT_ROOTS,
    repo_root: Path = ROOT,
) -> list[Hit]:
    """Find every static reference to ``symbol`` (a fully-qualified dotted
    path, e.g. ``pkg.mod.Class`` or ``pkg.mod.func``) across ``roots``,
    resolving import aliases per-file rather than matching bare text.
    """
    simple_name = symbol.rsplit(".", 1)[-1]
    module_path = symbol.rsplit(".", 1)[0] if "." in symbol else symbol
    hits: list[Hit] = []
    for path in _python_files(roots, repo_root):
        tree = _parse_candidate(path, simple_name)
        if tree is None:
            continue
        imports = _collect_imports(tree)
        context = SearchContext(
            tree=tree,
            symbol=symbol,
            simple_name=simple_name,
            module_path=module_path,
            imports=imports,
            relative_file=str(path.relative_to(repo_root)),
            direct_alias=_first_alias(imports.aliases, symbol),
            module_alias=_first_alias(imports.module_aliases, module_path),
            existing_hits=tuple(hits),
        )
        hits.extend(_file_hits(context))

    hits.sort(key=lambda h: (h.file, h.line))
    return hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "symbol",
        help="fully-qualified dotted symbol, e.g. pkg.mod.Class or pkg.mod.func",
    )
    parser.add_argument(
        "--roots",
        default=",".join(DEFAULT_ROOTS),
        help=f"comma-separated top-level dirs to scan (default: {','.join(DEFAULT_ROOTS)})",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    parser.add_argument(
        "--exclude-tests",
        action="store_true",
        help="drop hits under a 'tests' path segment (to answer 'any PRODUCTION caller?')",
    )
    args = parser.parse_args(argv)

    roots = tuple(r.strip() for r in args.roots.split(",") if r.strip())
    hits = find_callers(args.symbol, roots=roots)
    if args.exclude_tests:
        hits = [
            h
            for h in hits
            if "/tests/" not in f"/{h.file}" and not h.file.startswith("tests/")
        ]

    if args.json:
        print(json.dumps([h.__dict__ for h in hits], indent=2))
    else:
        if not hits:
            print(f"no static references found to {args.symbol!r} under {roots}")
        for h in hits:
            print(f"{h.file}:{h.line}: [{h.kind}] {h.snippet}")
        print(f"\n{len(hits)} hit(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
