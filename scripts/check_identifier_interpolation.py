#!/usr/bin/env python3
"""Identifier-interpolation guardrail gate.

Neither Cypher nor SQL/DDL accepts a bound parameter for an *identifier*
position (a node label, relationship type, property/column name, table,
schema, or extension) — only literal *values* bind. So a backend/pipeline
module that builds a label- or table-scoped query must interpolate that name
directly into an f-string. Left unvalidated, a caller/ontology/connector-
supplied name can break out of the intended query shape even though every
*value* in the same query is safely bound (see
``agent_utilities/security/identifiers.py``).

This gate is a **static heuristic**, not a dataflow prover (matching every
other ``scripts/check_*.py`` gate in this repo): it flags an f-string
identifier-position interpolation that has no *nearby* identifier-validation
call, so a newly-added, unguarded site fails the build instead of shipping
silently. It recognizes:

* the shared gate itself — ``validate_identifier`` / ``validate_sql_identifier``
  / ``quote_sql_identifier`` (``agent_utilities/security/identifiers.py``);
* the pre-existing per-backend validators this task centralizes
  (``_require_sql_identifier``, ``_require_age_graph_name``,
  ``_require_database_identifier``, ``_schema_identifier``, ``sanitize_label``);
* psycopg's safe quoting wrapper (``sql.Identifier(...)``);
* a direct ``SOMETHING_IDENTIFIER_RE.fullmatch(...)``-style regex check;
* one level of local indirection — a helper function *defined in the same
  file* that itself calls one of the above (e.g. a module's own
  ``_safe_graph_identifier()`` sanitizer) counts as a guard for its callers;
* a module-level constant assigned from a call to any recognized guard
  (e.g. ``_LABEL = validate_identifier("IngestManifest", kind="label")``)
  counts as pre-validated wherever that constant is later interpolated.

An interpolation is only a *candidate* when the f-string's own literal text
contains a query/DDL keyword (CREATE/ALTER/DROP/SELECT/INSERT/UPDATE/DELETE/
MERGE/MATCH/GRANT/CALL/…) AND the literal text immediately touching the ``{}``
gap looks like an identifier slot (a bare/backtick/double-quoted position, or
right after TABLE/INDEX/EXTENSION/POLICY/COLUMN/EXISTS/ON/FROM/TO/INTO) — or
the whole f-string is a "pure" dotted-name composition (``f"{schema}.{tbl}"``,
the ``::regclass`` cast shape) regardless of keyword content. This keeps the
gate from flagging ordinary log/error f-strings.

Scope: by default this scans the mirror-backend adapters
(``knowledge_graph/backends/**``, every current + future ``GraphBackend``) plus
the specific ETL/extraction/ingestion/pipeline modules this task centralized
(narrower than "all of ``agent_utilities``" — a directory-wide default would
also trip on unrelated dot-joined compositions elsewhere, e.g. a Python
import-path or dict-flattening f-string, that this gate cannot distinguish
from a real ``schema.table`` cast without deeper context; those stay a
follow-up audit item, not silently masked — see the task's final report).
RDF/SPARQL backends (``backends/sparql/``, ``backends/owl/``) are excluded:
they use IRIs, not SQL/Cypher identifiers, and already validate through a
different path (``_sparql_iri`` et al.). Pass an explicit ROOT to scan
anything else (e.g. a test fixture directory).

Usage:
  python3 scripts/check_identifier_interpolation.py [ROOT]

Exit 0 = no unguarded site found, 1 = at least one violation.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {"__pycache__", ".venv", "node_modules", "build", "dist"}

# Primitives that are ALWAYS a recognized guard, anywhere in the fleet. Every
# other local sanitizer is discovered transitively (see `_local_guard_names`)
# by calling one of these from its own body, so this list stays short.
BASE_GUARD_NAMES = {
    "validate_identifier",
    "validate_sql_identifier",
    "quote_sql_identifier",
    "_require_sql_identifier",
    "_require_age_graph_name",
    "_require_database_identifier",
    "_schema_identifier",
    "sanitize_label",
}

# A call `x.Identifier(...)` (psycopg's `sql.Identifier`) is always safe —
# it's the driver's own quoting primitive, not a raw string splice.
_SAFE_ATTR_CALLS = {"Identifier"}

_QUERY_KEYWORD_RE = re.compile(
    r"\b(CREATE|ALTER|DROP|SELECT|INSERT|UPDATE|DELETE|MERGE|MATCH|GRANT|"
    r"REVOKE|TRUNCATE|CALL)\b",
    re.IGNORECASE,
)
# Literal text ending in one of these means the very next `{}` gap is an
# identifier slot: a bare/backtick/double-quoted or Cypher-label-colon
# position. A bare trailing "." is deliberately EXCLUDED here — a lone dot
# also ends a SPARQL/N-Triples statement (``... .{gc}``), which is a
# value/close-brace position, not an identifier; the schema-qualified
# ``{schema}.{table}`` shape is instead caught structurally by
# `_is_pure_dotted_composition` below, which requires an ACTUAL dot constant
# between two interpolations rather than merely "no disqualifying text".
_SUFFIX_MARKER_RE = re.compile(r'["`:]$')
# Literal text ending in a Cypher node/relationship variable directly inside
# an (unclosed) paren/bracket — ``(n{label}``, ``(a{u_label_str}`` — the
# common "build an optional ``:Label`` fragment, then splice it after the
# bare pattern variable" idiom (the exact shape of the original
# ``ladybug_backend.prune()`` bug: the colon lives INSIDE the interpolated
# fragment's own value, not in this f-string's literal text, so the
# ``:``-suffix marker above can't see it — this structural marker catches it
# instead). Requires >=1 word char after the paren (the variable name) so a
# BARE ``(`` — e.g. building a SQL ``IN (?,?,?)`` placeholder list — does not
# match; that is a value position, not an identifier one.
_PAREN_VAR_MARKER_RE = re.compile(r"[(\[]\s*\w+$")
# Literal text ending in one of these DDL keywords (with a trailing space)
# also means the next `{}` gap is a bare (unquoted) identifier position.
_KEYWORD_MARKER_RE = re.compile(
    r"\b(TABLE|INDEX|EXTENSION|POLICY|COLUMN|EXISTS|ON|FROM|TO|INTO)\s?$",
    re.IGNORECASE,
)
# A regex object plausibly gating identifiers: `..._IDENTIFIER..._RE.fullmatch(...)`
# or `SOMETHING_IDENTIFIER.fullmatch(...)`.
_IDENTIFIER_REGEX_NAME_RE = re.compile(r"IDENTIFIER", re.IGNORECASE)


def _call_name(node: ast.expr) -> str | None:
    """Best-effort bare-or-attribute name of a Call's callee."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _iter_calls(node: ast.AST) -> list[ast.Call]:
    return [n for n in ast.walk(node) if isinstance(n, ast.Call)]


def _is_guard_call(call: ast.Call, guard_names: set[str]) -> bool:
    name = _call_name(call.func)
    if name in guard_names:
        return True
    if name in _SAFE_ATTR_CALLS and isinstance(call.func, ast.Attribute):
        return True
    if (
        name in ("fullmatch", "match", "search")
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and _IDENTIFIER_REGEX_NAME_RE.search(call.func.value.id)
    ):
        return True
    return False


def _local_guard_names(tree: ast.Module) -> set[str]:
    """One-hop transitive closure: a function defined in this file that itself
    calls a recognized guard (directly or via a fullmatch on an
    ``*IDENTIFIER*`` regex) is itself added to the recognized-guard set, so
    callers of *that* helper are considered guarded too (e.g. a module's own
    ``_safe_graph_identifier()`` wrapper)."""
    names = set(BASE_GUARD_NAMES)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if any(_is_guard_call(c, BASE_GUARD_NAMES) for c in _iter_calls(node)):
                names.add(node.name)
    return names


def _module_validated_constants(tree: ast.Module, guard_names: set[str]) -> set[str]:
    """Names assigned at module top level from a call to a recognized guard —
    e.g. ``_LABEL = validate_identifier("IngestManifest", kind="label")`` —
    are pre-validated everywhere else in the module they're interpolated."""
    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            if _is_guard_call(node.value, guard_names):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        found.add(tgt.id)
    return found


def _enclosing_scope(
    tree: ast.Module, lineno: int
) -> ast.FunctionDef | ast.AsyncFunctionDef | ast.Module:
    """The innermost function containing ``lineno``, else the module itself."""
    best: ast.FunctionDef | ast.AsyncFunctionDef | ast.Module = tree
    best_span = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            end = getattr(node, "end_lineno", None) or node.lineno
            if node.lineno <= lineno <= end:
                span = end - node.lineno
                if best_span is None or span < best_span:
                    best, best_span = node, span
    return best


def _scope_has_guard_call(scope: ast.AST, guard_names: set[str]) -> bool:
    return any(_is_guard_call(c, guard_names) for c in _iter_calls(scope))


def _joined_str_literal(node: ast.JoinedStr) -> str:
    return "".join(v.value for v in node.values if isinstance(v, ast.Constant))


def _is_pure_dotted_composition(node: ast.JoinedStr) -> bool:
    """True for an f-string that is ONLY ``{expr}.{expr}`` (an ACTUAL literal
    dot between two interpolations, e.g. ``f"{schema}.{tbl}"`` — the
    schema-qualified regclass-cast shape) — flagged regardless of keyword
    content since it never contains one. Requires a real ``"."`` constant to
    be present (not merely the absence of other text), so back-to-back
    interpolations with no separator at all (``f"{a}{b}"``, e.g. a delimiter
    or IRI composition) do NOT match."""
    fmt_count = sum(1 for v in node.values if isinstance(v, ast.FormattedValue))
    if fmt_count < 2:
        return False
    has_dot = False
    for v in node.values:
        if isinstance(v, ast.Constant):
            if v.value == ".":
                has_dot = True
            elif v.value != "":
                return False
    return has_dot


def _find_violations(rel: Path, tree: ast.Module) -> list[str]:
    violations: list[str] = []
    guard_names = _local_guard_names(tree)
    module_constants = _module_validated_constants(tree, guard_names)

    for node in ast.walk(tree):
        if not isinstance(node, ast.JoinedStr):
            continue
        literal = _joined_str_literal(node)
        has_query_keyword = bool(_QUERY_KEYWORD_RE.search(literal))
        pure_dotted = _is_pure_dotted_composition(node)
        if not has_query_keyword and not pure_dotted:
            continue

        prev_literal = ""
        for value in node.values:
            if isinstance(value, ast.Constant):
                prev_literal = value.value
                continue
            if not isinstance(value, ast.FormattedValue):
                continue
            is_identifier_slot = pure_dotted or bool(
                _SUFFIX_MARKER_RE.search(prev_literal)
                or _KEYWORD_MARKER_RE.search(prev_literal)
                or _PAREN_VAR_MARKER_RE.search(prev_literal)
            )
            prev_literal = ""
            if not is_identifier_slot:
                continue

            expr = value.value
            if isinstance(expr, ast.Name) and expr.id in module_constants:
                continue
            scope = _enclosing_scope(tree, value.lineno)
            if _scope_has_guard_call(scope, guard_names):
                continue

            try:
                snippet = ast.unparse(expr)
            except Exception:  # noqa: BLE001 - best-effort snippet only
                snippet = "<expr>"
            violations.append(
                f"{rel}:{value.lineno}: unguarded identifier interpolation "
                f"({{{snippet}}}) — call validate_identifier()/"
                f"validate_sql_identifier() (agent_utilities.security.identifiers) "
                f"before interpolating"
            )
    return violations


def _iter_py_files(target: Path, *, exclude_dirs: set[str]) -> list[Path]:
    if target.is_file():
        return [target] if target.suffix == ".py" else []
    if not target.is_dir():
        return []
    return sorted(
        p
        for p in target.rglob("*.py")
        if not any(part in SKIP_DIRS | exclude_dirs for part in p.parts)
    )


def scan(
    targets: list[Path], *, display_root: Path, exclude_dirs: set[str] = frozenset()
) -> list[str]:
    violations: list[str] = []
    seen: set[Path] = set()
    for target in targets:
        for path in _iter_py_files(target, exclude_dirs=exclude_dirs):
            if path in seen:
                continue
            seen.add(path)
            try:
                source = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            try:
                tree = ast.parse(source, filename=str(path))
            except SyntaxError:
                continue
            try:
                rel = path.relative_to(display_root)
            except ValueError:
                rel = path
            violations.extend(_find_violations(rel, tree))
    return sorted(violations)


# Default scope: the mirror-backend adapters (every ``GraphBackend``
# implementation, current + future) plus the specific ETL/extraction/
# ingestion/pipeline modules this task centralized. Deliberately narrower than
# "all of agent_utilities" — see the module docstring's "Scope" note.
_DEFAULT_TARGETS = (
    "knowledge_graph/backends",
    "knowledge_graph/migration.py",
    "knowledge_graph/etl/lineage.py",
    "knowledge_graph/extraction/job_manager.py",
    "knowledge_graph/ingestion/manifest.py",
    "knowledge_graph/pipeline/phases/sync.py",
)
# RDF/SPARQL backends use IRIs (``<...>``), not SQL/Cypher labels — a double
# quote there opens a SPARQL string *literal* (a value), not an identifier, so
# this gate's SQL/Cypher-shaped heuristics don't apply. Those backends have
# their own validated IRI path (``_sparql_iri`` et al., see
# ``tests/unit/knowledge_graph/test_query_construction_security.py``).
_EXCLUDE_DIRS = {"sparql", "owl"}


def main() -> int:
    if len(sys.argv) > 1:
        pkg_root = Path(sys.argv[1])
        targets = [pkg_root]
        exclude_dirs: set[str] = set()
    else:
        pkg_root = ROOT / "agent_utilities"
        targets = [pkg_root / t for t in _DEFAULT_TARGETS]
        exclude_dirs = _EXCLUDE_DIRS
    violations = scan(targets, display_root=pkg_root.parent, exclude_dirs=exclude_dirs)
    if violations:
        print("Unguarded identifier interpolation found:\n")
        for v in violations:
            print(f"  {v}")
        print(
            "\nEvery label/table/relationship-type/column interpolated into a "
            "Cypher/SQL/DDL f-string must be validated first — see "
            "agent_utilities/security/identifiers.py and AGENTS.md."
        )
        return 1
    scanned = ", ".join(str(t.relative_to(pkg_root.parent)) for t in targets)
    print(f"OK — no unguarded identifier interpolation under {scanned}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
