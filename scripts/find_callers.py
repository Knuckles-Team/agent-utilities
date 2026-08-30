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
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ROOTS = ("agent_utilities", "scripts", "tests", "examples")


@dataclass
class Hit:
    file: str
    line: int
    kind: str  # "call" | "reference" | "monkeypatch" | "getattr"
    snippet: str


@dataclass
class _FileImports:
    # local name -> fully-qualified dotted path it refers to
    aliases: dict[str, str] = field(default_factory=dict)
    # local name -> module it was imported *from* (for from-imports of the
    # symbol's containing module itself, e.g. "from pkg import module as m")
    module_aliases: dict[str, str] = field(default_factory=dict)


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


def _dotted_from_attribute(node: ast.AST) -> str | None:
    """Reconstruct the dotted string an ``ast.Attribute``/``ast.Name`` chain spells."""
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    else:
        return None
    return ".".join(reversed(parts))


def _resolve(dotted: str, fi: _FileImports) -> str | None:
    """Resolve a locally-spelled dotted reference to its fully-qualified path,
    using this file's collected import aliases. Returns None if unresolvable
    (e.g. a purely local variable with no traced import)."""
    head, _, rest = dotted.partition(".")
    if head in fi.aliases:
        base = fi.aliases[head]
        return f"{base}.{rest}" if rest else base
    return None


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


def _call_hit(
    node: ast.AST,
    direct_alias: str | None,
    module_alias: str | None,
    symbol: str,
    simple_name: str,
    fi: _FileImports,
    rel: str,
) -> Hit | None:
    if not isinstance(node, ast.Call):
        return None
    if isinstance(node.func, ast.Name):
        if direct_alias and node.func.id == direct_alias:
            return Hit(rel, node.lineno, "call", f"{node.func.id}(...)")
        return None
    if not isinstance(node.func, ast.Attribute):
        return None
    dotted = _dotted_from_attribute(node.func)
    if dotted is None:
        return None
    if _resolve(dotted, fi) == symbol:
        return Hit(rel, node.lineno, "call", dotted + "(...)")
    if module_alias and dotted == f"{module_alias}.{simple_name}":
        return Hit(rel, node.lineno, "call", dotted + "(...)")
    return None


def _call_hits(
    tree: ast.AST,
    direct_alias: str | None,
    module_alias: str | None,
    symbol: str,
    simple_name: str,
    fi: _FileImports,
    rel: str,
) -> list[Hit]:
    hits: list[Hit] = []
    for node in ast.walk(tree):
        hit = _call_hit(
            node, direct_alias, module_alias, symbol, simple_name, fi, rel
        )
        if hit is not None:
            hits.append(hit)
    return hits


def _name_reference_hit(
    node: ast.Name,
    direct_alias: str | None,
    rel: str,
    call_lines: set[int],
) -> Hit | None:
    if direct_alias and node.id == direct_alias and node.lineno not in call_lines:
        return Hit(rel, node.lineno, "reference", node.id)
    return None


def _attribute_reference_hit(
    node: ast.Attribute,
    module_alias: str | None,
    symbol: str,
    simple_name: str,
    fi: _FileImports,
    rel: str,
    call_lines: set[int],
) -> Hit | None:
    dotted = _dotted_from_attribute(node)
    if dotted is None:
        return None
    is_match = _resolve(dotted, fi) == symbol or (
        module_alias and dotted == f"{module_alias}.{simple_name}"
    )
    if is_match and node.lineno not in call_lines:
        return Hit(rel, node.lineno, "reference", dotted)
    return None


def _reference_hit(
    node: ast.AST,
    direct_alias: str | None,
    module_alias: str | None,
    symbol: str,
    simple_name: str,
    fi: _FileImports,
    rel: str,
    call_lines: set[int],
) -> Hit | None:
    if isinstance(node, ast.Name):
        return _name_reference_hit(node, direct_alias, rel, call_lines)
    if not isinstance(node, ast.Attribute):
        return None
    return _attribute_reference_hit(
        node,
        module_alias,
        symbol,
        simple_name,
        fi,
        rel,
        call_lines,
    )


def _reference_hits(
    tree: ast.AST,
    direct_alias: str | None,
    module_alias: str | None,
    symbol: str,
    simple_name: str,
    fi: _FileImports,
    rel: str,
    call_lines: set[int],
) -> list[Hit]:
    hits: list[Hit] = []
    for node in ast.walk(tree):
        hit = _reference_hit(
            node,
            direct_alias,
            module_alias,
            symbol,
            simple_name,
            fi,
            rel,
            call_lines,
        )
        if hit is not None:
            hits.append(hit)
    return hits


def _target_dotted(node: ast.AST) -> str | None:
    if isinstance(node, (ast.Name, ast.Attribute)):
        return _dotted_from_attribute(node)
    return None


def _monkeypatch_hit(
    node: ast.AST,
    module_alias: str | None,
    module_path: str,
    simple_name: str,
    fi: _FileImports,
    rel: str,
) -> Hit | None:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return None
    if node.func.attr != "setattr" or len(node.args) < 2:
        return None
    name_arg = node.args[1]
    if not isinstance(name_arg, ast.Constant) or name_arg.value != simple_name:
        return None
    target_dotted = _target_dotted(node.args[0])
    target_resolved = _resolve(target_dotted, fi) if target_dotted else None
    if target_resolved != module_path and target_dotted != module_alias:
        return None
    return Hit(
        rel,
        node.lineno,
        "monkeypatch",
        f"setattr({target_dotted}, {simple_name!r}, ...)",
    )


def _getattr_hit(
    node: ast.AST,
    module_alias: str | None,
    module_path: str,
    simple_name: str,
    fi: _FileImports,
    rel: str,
) -> Hit | None:
    if (
        not isinstance(node, ast.Call)
        or not isinstance(node.func, ast.Name)
        or node.func.id != "getattr"
    ):
        return None
    if len(node.args) < 2:
        return None
    name_arg = node.args[1]
    if not isinstance(name_arg, ast.Constant) or name_arg.value != simple_name:
        return None
    target_dotted = _target_dotted(node.args[0])
    target_resolved = _resolve(target_dotted, fi) if target_dotted else None
    if target_resolved != module_path and target_dotted != module_alias:
        return None
    return Hit(
        rel,
        node.lineno,
        "getattr",
        f"getattr({target_dotted}, {simple_name!r})",
    )


def _dynamic_hits(
    tree: ast.AST,
    module_alias: str | None,
    module_path: str,
    simple_name: str,
    fi: _FileImports,
    rel: str,
) -> list[Hit]:
    hits: list[Hit] = []
    for node in ast.walk(tree):
        hit = _monkeypatch_hit(
            node, module_alias, module_path, simple_name, fi, rel
        )
        if hit is not None:
            hits.append(hit)
    for node in ast.walk(tree):
        hit = _getattr_hit(node, module_alias, module_path, simple_name, fi, rel)
        if hit is not None:
            hits.append(hit)
    return hits


def _file_hits(
    path: Path,
    simple_name: str,
    module_path: str,
    symbol: str,
    repo_root: Path,
    existing_hits: list[Hit],
) -> list[Hit]:
    tree = _parse_candidate(path, simple_name)
    if tree is None:
        return []
    fi = _collect_imports(tree)
    rel = str(path.relative_to(repo_root))
    direct_alias = _first_alias(fi.aliases, symbol)
    module_alias = _first_alias(fi.module_aliases, module_path)
    hits = _call_hits(
        tree, direct_alias, module_alias, symbol, simple_name, fi, rel
    )
    call_lines = {h.line for h in existing_hits if h.file == rel}
    call_lines.update(h.line for h in hits if h.file == rel)
    hits.extend(
        _reference_hits(
            tree,
            direct_alias,
            module_alias,
            symbol,
            simple_name,
            fi,
            rel,
            call_lines,
        )
    )
    hits.extend(
        _dynamic_hits(tree, module_alias, module_path, simple_name, fi, rel)
    )
    return hits


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
        hits.extend(
            _file_hits(path, simple_name, module_path, symbol, repo_root, hits)
        )

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
