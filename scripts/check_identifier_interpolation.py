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
  (``_require_sql_identifier``, ``_require_identifier``, ``_require_age_graph_name``,
  ``_require_database_identifier``, ``_schema_identifier``, ``sanitize_label``,
  ``_sanitize_label``, ``safe_label``, ``_safe_rel_type``, ``_safe_ident`` — the
  ``safe_*``/``_safe_rel_type`` family is a fallback-not-raise variant: normalize,
  ``fullmatch`` check, else a safe generic label, rather than raising);
* psycopg's safe quoting wrapper (``sql.Identifier(...)``);
* a direct ``SOMETHING_IDENTIFIER_RE.fullmatch(...)``-style regex check;
* a bare ``ast.Name`` reference that follows the PEP8 module-constant
  convention (optional leading underscores, then ALL-CAPS + digits/
  underscores, e.g. ``TRACE_USED_TOOL_EDGE``) — a fixed, code-controlled
  compile-time symbol, never runtime/caller data, even when imported from
  another module this gate has no cross-module constant tracking for;
* a ``!r``/``!a``-converted ``FormattedValue`` — Python wraps the value in
  repr/ascii quoting syntax, never how a bare identifier is inlined;
* the WHOLE f-string when it is the sole argument to ``import_module(...)`` —
  a Python dotted import path, never a Cypher/SQL identifier position,
  regardless of the shape of the values composing it;
* a structurally non-identifier expression — a ``Call``/``BinOp``/``UnaryOp``
  result, a nested f-string, or a dunder name/attribute (``__name__``,
  ``o.__module__``) — none of which a real label/table name is ever inlined
  as (always a bare variable or simple attribute in this codebase's style);
* one level of local indirection — a helper function *defined in the same
  file* that itself calls one of the above (e.g. a module's own
  ``_safe_graph_identifier()`` sanitizer) counts as a guard for its callers;
* a guard call anywhere in an ENCLOSING lexical scope, not just the innermost
  one — a nested closure shares its enclosing function's exact variable
  bindings, so a guard call that already ran there before the closure was
  defined covers the closure too;
* a module-level constant assigned from a call to any recognized guard
  (e.g. ``_LABEL = validate_identifier("IngestManifest", kind="label")``)
  counts as pre-validated wherever that constant is later interpolated.

An interpolation is only a *candidate* when its f-string is **sink-reachable**
(BUG-232 — classify by SINK, not by f-string shape): the f-string's value must
be able to reach an actual query-execution call, not merely *look* query-shaped.
Reachability is computed once per file as a small worklist/fixed-point over the
file's own AST (still a single-file static heuristic, not cross-module
dataflow):

* **Seed** — every f-string passed directly as a positional/keyword argument to
  a call whose callee matches :data:`_SINK_CALL_NAMES` (the backends' actual
  execute/DDL entrypoints — ``execute``, ``executemany``, ``execute_batch``,
  ``execute_cypher``, ``query_cypher``, …) or a ``session.run(...)``/``tx.run(...)``
  call (the neo4j-driver shape, scoped to those receiver names so it doesn't
  collide with this codebase's many unrelated ``.run(...)`` calls — pydantic-ai
  ``agent.run(...)``, ``asyncio.run(...)``, ``subprocess.run(...)``). A bare
  ``execute``-prefixed name is deliberately **not** matched as a substring/regex —
  this codebase also has non-DB "execute this unit of work" dispatchers
  (``_execute_tool``, ``_execute_one``, ``_execute_claim``, …) with the same
  naming convention; the sink set is an explicit, grep-verified allow-list so
  those don't collide.
* **Propagate** — a name assigned directly from an f-string (``label = f"..."``)
  is itself sink-reachable if that name is later passed directly to a sink call,
  *or* if it is later interpolated into ANOTHER f-string that is (transitively)
  sink-reachable — the exact ``label`` → ``query`` → ``self.execute(query, …)``
  chain that is the original ``ladybug_backend.prune()`` bug this gate exists to
  catch (see the gate's own meta-test).

Only *within* a sink-reachable f-string does the existing per-slot shape check
run: the literal text immediately touching the ``{}`` gap must look like an
identifier slot (a bare/backtick/double-quoted position, or right after
TABLE/INDEX/EXTENSION/POLICY/COLUMN/EXISTS/ON/FROM/TO/INTO) — or the whole
f-string is a "pure" dotted-name composition (``f"{schema}.{tbl}"``, the
``::regclass`` cast shape). This two-stage design (sink-reachable, THEN
identifier-shaped) is what keeps the gate from flagging an ordinary log/error
f-string that merely *contains* a query-shaped word: the original false
positive here was ``intent_tools.py``'s approval-required error message —
never passed to any query call — which only tripped the gate because its text
happened to contain the English word "call" (matching the ``CALL`` Cypher
keyword) next to a JSON-quote-adjacent ``{chosen_tool}`` interpolation.

Scope: by default this scans ALL of ``agent_utilities/`` — every current and
future module that might f-string-interpolate a label/table/relationship-
type/column, not just the mirror-backend adapters. (Launched narrower —
``knowledge_graph/backends/**`` plus the specific ETL/extraction/ingestion/
pipeline modules the wave-0 task centralized — because a directory-wide
default also trips on unrelated dot-joined compositions elsewhere, e.g. a
Python import-path or dict-flattening f-string, that this gate cannot
distinguish from a real ``schema.table`` cast without deeper context. The
follow-up task widened the scope and either guarded or rewrote every site
the wider scan turned up — e.g. ``engine_surface_tools.py`` swapped a
diagnostic ``f"{a}.{m}"`` attribute-path composition for ``".".join((a, m))``
rather than leaving a false-positive-shaped f-string in place.) RDF/SPARQL
backends (``backends/sparql/``, ``backends/owl/``) are excluded: they use
IRIs, not SQL/Cypher identifiers, and already validate through a different
path (``_sparql_iri`` et al.). Pass an explicit ROOT to scan anything else
(e.g. a test fixture directory).

Usage:
  python3 scripts/check_identifier_interpolation.py [ROOT]

Exit 0 = no unguarded site found, 1 = at least one violation.
"""

from __future__ import annotations

import ast
import re
import sys
from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts._git_scan import tracked_or_walked  # noqa: E402

SKIP_DIRS = {"__pycache__", ".venv", "node_modules", "build", "dist"}

# Primitives that are ALWAYS a recognized guard, anywhere in the fleet. Every
# other local sanitizer is discovered transitively (see `_local_guard_names`)
# by calling one of these from its own body, so this list stays short.
BASE_GUARD_NAMES = {
    "validate_identifier",
    "validate_sql_identifier",
    "quote_sql_identifier",
    "_require_sql_identifier",
    "_require_identifier",
    "_require_age_graph_name",
    "_require_database_identifier",
    "_schema_identifier",
    "sanitize_label",
    "_sanitize_label",
    "safe_label",
    "_safe_rel_type",
    "_safe_ident",
}

# A call `x.Identifier(...)` (psycopg's `sql.Identifier`) is always safe —
# it's the driver's own quoting primitive, not a raw string splice.
_SAFE_ATTR_CALLS = {"Identifier"}

# Literal text ending in one of these means the very next `{}` gap is an
# identifier slot: a bare/backtick/double-quoted or Cypher-label-colon
# position. A bare trailing "." is deliberately EXCLUDED here — a lone dot
# also ends a SPARQL/N-Triples statement (``... .{gc}``), which is a
# value/close-brace position, not an identifier; the schema-qualified
# ``{schema}.{table}`` shape is instead caught structurally by
# `_is_pure_dotted_composition` below, which requires an ACTUAL dot constant
# between two interpolations rather than merely "no disqualifying text".
# Anchored on ``\Z`` (absolute end of string), NOT ``$`` — Python's ``$``
# (without MULTILINE) also matches just before a single trailing ``\n``, so
# any multi-line log/prompt/error-message literal ending "...: \n" (e.g.
# ``f"...following error:\n{ctx.state.error}\n..."``) would otherwise match
# the same as a genuine ``MERGE (n:{label})`` colon — a false positive the
# scope-widening task turned up in ``graph/_router_impl.py``.
_SUFFIX_MARKER_RE = re.compile(r'["`:]\Z')
# Literal text ending in a Cypher node/relationship variable directly inside
# an (unclosed) paren/bracket — ``(n{label}``, ``(a{u_label_str}`` — the
# common "build an optional ``:Label`` fragment, then splice it after the
# bare pattern variable" idiom (the exact shape of the original
# ``ladybug_backend.prune()`` bug: the colon lives INSIDE the interpolated
# fragment's own value, not in this f-string's literal text, so the
# ``:``-suffix marker above can't see it — this structural marker catches it
# instead). Requires >=1 word char after the paren (the variable name) so a
# BARE ``(`` — e.g. building a SQL ``IN (?,?,?)`` placeholder list — does not
# match; that is a value position, not an identifier one. ``\Z``-anchored for
# the same trailing-newline reason as ``_SUFFIX_MARKER_RE`` above.
_PAREN_VAR_MARKER_RE = re.compile(r"[(\[]\s*\w+\Z")
# Literal text ending in one of these DDL keywords (with a trailing space)
# also means the next `{}` gap is a bare (unquoted) identifier position.
# ``\Z``-anchored for the same trailing-newline reason as
# ``_SUFFIX_MARKER_RE`` above.
_KEYWORD_MARKER_RE = re.compile(
    r"\b(TABLE|INDEX|EXTENSION|POLICY|COLUMN|EXISTS|ON|FROM|TO|INTO)\s?\Z",
    re.IGNORECASE,
)
# A regex object plausibly gating identifiers: `..._IDENTIFIER..._RE.fullmatch(...)`
# or `SOMETHING_IDENTIFIER.fullmatch(...)`.
_IDENTIFIER_REGEX_NAME_RE = re.compile(r"IDENTIFIER", re.IGNORECASE)
# A PEP8-conventional module-level constant name (optional leading
# underscores, then ALL-CAPS + digits/underscores, e.g. ``_AGENT_LABEL``,
# ``TRACE_USED_TOOL_EDGE``, ``OUTCOME_NODE_LABEL``). Widening this gate's scope
# to all of ``agent_utilities`` surfaced many fixed, code-controlled label/
# edge-type/namespace constants defined once (often via a plain string
# literal, not a guard call) and imported into many consuming modules — this
# gate has no cross-module constant tracking, so each import site would
# otherwise need its own redundant guard call. A bare ``ast.Name`` reference
# that follows this naming convention is treated as a trusted compile-time
# symbol, matching how the rest of the fleet already relies on ALL-CAPS naming
# to signal "never reassigned, never runtime/caller data" (see AGENTS.md
# Naming). This is deliberately narrower than "any Name" — a lowercase local
# variable (``label``, ``table``, ``tbl``) still requires a real guard call.
_CONVENTIONAL_CONSTANT_RE = re.compile(r"^_*[A-Z][A-Z0-9_]*$")


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def _is_definitely_not_an_identifier_expr(expr: ast.expr) -> bool:
    """True when ``expr`` structurally cannot be a label/table/column name.

    Widening this gate's scope to all of ``agent_utilities`` surfaced dozens of
    "pure dotted composition" / keyword-marker false positives that are not
    Cypher/SQL at all: Python import-path construction
    (``import_module(f"{__name__}.{module_name}")``), JWT compact
    serialization (``f"{header}.{claims}"``), qualified-name error messages
    (``f"{o.__module__}.{o.__name__}"``), semver bumps
    (``f"{maj}.{min}.{int(patch) + 1}"``), Kafka/WS topic namespacing, and
    similar. A REAL identifier interpolation in this codebase's own style is
    always a bare variable or a simple attribute access that was validated (or
    is a recognized constant) — never the live result of a function call,
    arithmetic, or a dunder attribute. Excluding those structural shapes here
    only SKIPS flagging (never adds a flag), so it cannot mask a genuine
    identifier-shaped site — it only stops the heuristic from crying wolf on
    provably-non-identifier expressions.
    """
    if isinstance(expr, ast.Call | ast.BinOp | ast.UnaryOp | ast.JoinedStr):
        return True
    if isinstance(expr, ast.Name):
        return _is_dunder(expr.id)
    if isinstance(expr, ast.Attribute):
        return _is_dunder(expr.attr)
    return False


def _call_name(node: ast.expr) -> str | None:
    """Best-effort bare-or-attribute name of a Call's callee."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _iter_calls(node: ast.AST) -> list[ast.Call]:
    return [n for n in ast.walk(node) if isinstance(n, ast.Call)]


# Query-execution entrypoints (BUG-232 sink allow-list). Grep-verified against
# this codebase: each name below is the immediate call that hands a
# Cypher/SQL/DDL string to the underlying driver/connection — psycopg's
# ``cursor.execute``, kuzu's ``conn.execute``, sqlite3's ``.execute``, the
# shared ``GraphBackend`` ABC's ``execute``/``execute_read``/``execute_write``,
# and the private per-backend Cypher/AGE transpile-then-execute wrappers.
# Deliberately an EXPLICIT set, not an ``^execute`` prefix/substring regex:
# this codebase also names non-DB "execute this unit of work" dispatchers the
# same way (``_execute_tool`` — MCP tool dispatch; ``_execute_one``/
# ``_execute_batch`` — ontology action execution; ``_execute_claim``,
# ``_execute_result``, ``_execute_sub_call``, …) and a substring match would
# treat those as query sinks too, reintroducing a false-positive class shaped
# exactly like the one this allow-list was built to remove (BUG-232 —
# ``intent_tools.py`` calling ``kg_server._execute_tool(...)`` in the same
# scope as its approval-required error message).
_SINK_CALL_NAMES = {
    "execute",
    "executemany",
    "executescript",
    "execute_batch",
    "execute_read",
    "execute_write",
    "execute_ddl",
    "execute_query",
    "execute_cypher",
    "_execute_cypher",
    "execute_age",
    "_execute_age",
    "query_cypher",
    "_query_cypher_rows",
}
# The neo4j-driver ``session.run(query, params)`` / ``tx.run(...)`` shape is a
# real sink, but a bare ``run`` name is far too overloaded in this codebase to
# match unconditionally (pydantic-ai ``agent.run(...)``, ``asyncio.run(...)``,
# ``subprocess.run(...)``, an orchestrator's ``Graph.run(...)`` are all
# `.run(...)` calls that never touch Cypher/SQL) — so it is scoped to the
# conventional receiver names this codebase's own neo4j backend actually uses.
_SINK_RUN_RECEIVER_NAMES = {"session", "tx"}


def _is_sink_call(call: ast.Call) -> bool:
    """True when ``call`` is a recognized query/DDL-execution entrypoint."""
    name = _call_name(call.func)
    if name in _SINK_CALL_NAMES:
        return True
    return bool(
        name == "run"
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id in _SINK_RUN_RECEIVER_NAMES
    )


def _collect_name_defs(tree: ast.Module) -> dict[str, list[ast.JoinedStr]]:
    """Map ``name -> [f-strings directly assigned to it]`` across the whole
    file (``label = f"..."``, ``query: str = f"..."``) — the local, per-file
    dataflow edges the sink-reachability worklist below walks backward over.
    Multiple assignments to the same name are all kept (a coarse
    over-approximation consistent with this gate's whole-file, static-heuristic
    character elsewhere: it can only make MORE f-strings candidates, never
    fewer, so it cannot hide a genuine violation)."""
    defs: dict[str, list[ast.JoinedStr]] = {}
    for node in ast.walk(tree):
        target: ast.expr | None = None
        value: ast.expr | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        if isinstance(target, ast.Name) and isinstance(value, ast.JoinedStr):
            defs.setdefault(target.id, []).append(value)
    return defs


def _sink_reachable_ids(tree: ast.Module) -> set[int]:
    """``id()`` of every ``JoinedStr`` node that can reach a query-execution
    sink (BUG-232) — seeded from f-strings/names passed directly to a
    :func:`_is_sink_call` call, then propagated backward through
    :func:`_collect_name_defs` so a fragment (e.g. ``label``) spliced into a
    larger, sink-reachable f-string (e.g. ``query``) is reachable too — the
    ``label`` → ``query`` → ``self.execute(query, …)`` chain that is the
    original ``ladybug_backend.prune()`` bug this gate exists to catch."""
    name_defs = _collect_name_defs(tree)
    visited: set[int] = set()
    worklist: list[ast.JoinedStr] = []

    def _seed(expr: ast.expr) -> None:
        if isinstance(expr, ast.JoinedStr):
            if id(expr) not in visited:
                visited.add(id(expr))
                worklist.append(expr)
        elif isinstance(expr, ast.Name):
            for defn in name_defs.get(expr.id, ()):
                if id(defn) not in visited:
                    visited.add(id(defn))
                    worklist.append(defn)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_sink_call(node):
            for arg in node.args:
                _seed(arg)
            for kw in node.keywords:
                if kw.value is not None:
                    _seed(kw.value)

    while worklist:
        current = worklist.pop()
        for value in current.values:
            if isinstance(value, ast.FormattedValue) and isinstance(
                value.value, ast.Name
            ):
                for defn in name_defs.get(value.value.id, ()):
                    if id(defn) not in visited:
                        visited.add(id(defn))
                        worklist.append(defn)

    return visited


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


def _enclosing_scopes(
    tree: ast.Module, lineno: int
) -> list[ast.FunctionDef | ast.AsyncFunctionDef | ast.Module]:
    """Every function containing ``lineno``, innermost first, ending with the
    module itself — the full lexical nesting chain, not just the innermost
    function.

    A nested closure (e.g. a ``def _fetch(...)`` defined inside a method,
    capturing one of the method's own local variables) shares the EXACT same
    variable binding as its enclosing function — if the outer function already
    validated that variable before the closure was even defined, the value the
    closure later interpolates is the SAME validated value, not merely
    "probably fine by convention" (unlike the cross-*method* instance-attribute
    case, there is no possibility of a different call path bypassing it: a
    closure cannot observe its own free variable being reassigned by another
    call). So a guard call anywhere in an ENCLOSING scope also counts.
    """
    chain: list[ast.FunctionDef | ast.AsyncFunctionDef | ast.Module] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            end = getattr(node, "end_lineno", None) or node.lineno
            if node.lineno <= lineno <= end:
                chain.append(node)
    # getattr-based throughout (not direct .lineno access): chain's declared
    # element type includes ast.Module below (the fallback branch), which has
    # no .lineno of its own — even though only FunctionDef/AsyncFunctionDef
    # are ever actually appended above, before this sort runs.
    chain.sort(key=lambda n: getattr(n, "end_lineno", 0) - getattr(n, "lineno", 0))
    # Fall back to the module only when NO enclosing function exists (rare:
    # true module-top-level code) — NOT appended unconditionally, which would
    # make "any guard call anywhere in the file" satisfy every function's
    # scope check and defeat per-function scoping entirely.
    return chain or [tree]


def _scope_has_guard_call(scopes: Sequence[ast.AST], guard_names: set[str]) -> bool:
    return any(
        _is_guard_call(c, guard_names) for scope in scopes for c in _iter_calls(scope)
    )


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


def _build_parent_map(tree: ast.Module) -> dict[int, ast.AST]:
    """Map ``id(child) -> parent`` for every node (``ast`` gives no parent
    pointers by default) — used only for the narrow ``import_module(f"...")``
    exclusion below, so the rest of the gate stays parent-free/context-free."""
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent
    return parents


_IMPORT_MODULE_CALLEE_RE = re.compile(r"^import_module$")


def _is_import_module_path_argument(
    node: ast.JoinedStr, parents: dict[int, ast.AST]
) -> bool:
    """True when ``node`` (the WHOLE f-string) is the module-path argument to
    ``import_module(...)`` / ``importlib.import_module(...)`` — a Python
    dotted import path, never a Cypher/SQL identifier position. Recurring
    plugin/connector-discovery shape (``pkgutil.iter_modules`` + dynamic
    import) this gate's scope-widening turned up in half a dozen modules;
    handled structurally here (by SINK, not by value shape) rather than as
    per-site exclusions, since the same ``pkg.__name__``/``mod.name``
    attribute-chain shape is also a legitimate identifier shape in other
    contexts (e.g. ``self.table_name``) and can't be excluded on shape alone."""
    parent = parents.get(id(node))
    if not isinstance(parent, ast.Call):
        return False
    name = _call_name(parent.func)
    return bool(name and _IMPORT_MODULE_CALLEE_RE.match(name)) and parent.args == [node]


def _find_violations(rel: Path, tree: ast.Module) -> list[str]:
    violations: list[str] = []
    guard_names = _local_guard_names(tree)
    module_constants = _module_validated_constants(tree, guard_names)
    parents = _build_parent_map(tree)
    sink_reachable = _sink_reachable_ids(tree)

    for node in ast.walk(tree):
        if not isinstance(node, ast.JoinedStr):
            continue
        # BUG-232: candidacy is decided by SINK reachability, not by whether
        # the f-string's own text merely *looks* query-shaped (see module
        # docstring). A query-keyword word can appear in ordinary prose (the
        # ``intent_tools.py`` approval message contains "call") without the
        # f-string ever reaching a query-execution call.
        if id(node) not in sink_reachable:
            continue
        if _is_import_module_path_argument(node, parents):
            continue
        pure_dotted = _is_pure_dotted_composition(node)

        prev_literal = ""
        for value in node.values:
            if isinstance(value, ast.Constant):
                # ``ast.Constant.value`` is typed for ANY literal (int/float/…),
                # but a JoinedStr's own Constant children are always the
                # literal str text parts by grammar — checked explicitly so
                # mypy can narrow it rather than relying on that guarantee alone.
                prev_literal = value.value if isinstance(value.value, str) else ""
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
            # ``!r``/``!a`` (conversion codes for repr()/ascii()) wrap the
            # value in quote/escape syntax Python controls — never how a bare
            # Cypher/SQL identifier is inlined (``MERGE (n:'Label')`` isn't
            # valid Cypher; a real label position is always the bare value).
            # This also sidesteps the ``TO``/``ON``/``FROM``/``INTO`` keyword
            # markers colliding with ordinary English ("Set mode to
            # {value!r}.") that the scope-widening task turned up in a
            # test-pinned prompt string (``skills/runtime_validation.py``)
            # that could not simply be reworded.
            if value.conversion in (ord("r"), ord("a")):
                continue

            expr = value.value
            if isinstance(expr, ast.Name) and (
                expr.id in module_constants or _CONVENTIONAL_CONSTANT_RE.match(expr.id)
            ):
                continue
            if _is_definitely_not_an_identifier_expr(expr):
                continue
            scopes = _enclosing_scopes(tree, value.lineno)
            if _scope_has_guard_call(scopes, guard_names):
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


def _tracked_or_walked_py_files(target: Path) -> list[Path]:
    """``.py`` files under ``target``, preferring the git-tracked set (BUG-043).

    A raw ``rglob`` also picks up gitignored, generated build output, which
    can carry a stale copy of an already-fixed source file and reintroduce a
    cleared violation. Falls back to a filesystem walk only when ``target``
    is not inside a git working tree (e.g. a synthetic test fixture).
    """
    return tracked_or_walked(target, "*.py", root=ROOT)


def _iter_py_files(target: Path, *, exclude_dirs: AbstractSet[str]) -> list[Path]:
    if target.is_file():
        return [target] if target.suffix == ".py" else []
    if not target.is_dir():
        return []
    return sorted(
        p
        for p in _tracked_or_walked_py_files(target)
        if not any(part in SKIP_DIRS | exclude_dirs for part in p.parts)
    )


def scan(
    targets: list[Path],
    *,
    display_root: Path,
    exclude_dirs: AbstractSet[str] = frozenset(),
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


# Default scope: the ENTIRE agent_utilities package (every current + future
# module that might f-string-interpolate a label/table/relationship-type/
# column). Widened from the original narrower launch scope (mirror backends +
# the specific ETL/extraction/ingestion/pipeline modules the wave-0 task
# centralized) once every site the wider scan turned up was guarded or proven
# a non-identifier false positive — see AGENTS.md / this task's final report.
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
        targets = [pkg_root]
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
