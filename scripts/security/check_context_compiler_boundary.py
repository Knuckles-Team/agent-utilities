#!/usr/bin/env python3
from __future__ import annotations

"""Architecture gate for the mandatory ContextCompiler model boundary."""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "agent_utilities"
SCAN_ROOTS = (PACKAGE, ROOT / "scripts", ROOT / "tests", ROOT / "examples")
CANONICAL_AGENT_BOUNDARY = "agent_utilities/core/contextual_model.py"
RAW_PROVIDER_ALLOWLIST = {
    "agent_utilities/core/model_factory.py",
    "agent_utilities/knowledge_graph/retrieval/context_compiler_serving.py",
    # Model-catalog verification only (client.models.retrieve(model_id)) — it
    # never sends a completion/response request, so there is no inference to
    # govern; create_context_agent has nothing to wrap here.
    "agent_utilities/core/openai_catalog.py",
}
# Files whose Agent(...) construction is intentionally metadata-only (no
# model is ever invoked through it) and therefore cannot go through
# create_context_agent, which requires and governs an explicit model
# (contextual_model.py: "governed agent construction requires an explicit
# model"). Each entry must be justified inline at its call site, not just
# here.
# D-CIM-3 perf: every violation this module can report requires at least one
# of these substrings to appear literally in the file's source (an import
# mentions "pydantic_ai"; a construction mentions "Agent"/"OpenAI"/
# "AsyncOpenAI"/"Anthropic"; a raw call mentions "chat.completions.create" or
# "responses.create"; the routers-only raw-HTTP-transport check mentions one
# of the provider hostnames or the raw POST call shapes). A file containing
# NONE of these cannot trigger ANY check below — skip it before paying for
# `ast.parse` + tree walks at all. This is a cheap, correctness-preserving
# pre-filter (substring supersets, never narrows what a real check does),
# not a weakening of the gate.
_TRIGGER_MARKERS = (
    "pydantic_ai",
    "Agent",
    "OpenAI",
    "Anthropic",
    "chat.completions.create",
    "responses.create",
    "api.openai.com",
    "api.anthropic.com",
    "generativelanguage.googleapis.com",
    'client.stream("POST"',
    'client.post("POST"',
)
METADATA_ONLY_AGENT_ALLOWLIST = {
    # GraphOSWorkflowAgent.__init__ builds `Agent(model=None, ...,
    # defer_model_check=True)` purely as a WrapperAgent facade carrying a
    # step's name/description for host tooling (approval prompts, tracing) —
    # actual execution is dispatched to the GraphOS orchestrator, never to
    # this Agent instance's own model (it deliberately has none).
    "agent_utilities/capabilities/governed_dynamic_workflow.py",
}


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _function_ancestors(tree: ast.AST) -> dict[ast.AST, str]:
    """Map every descendant node to its nearest function name."""

    owners: dict[ast.AST, str] = {}

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.stack: list[str] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def generic_visit(self, node: ast.AST) -> None:
            if self.stack:
                owners[node] = self.stack[-1]
            super().generic_visit(node)

    Visitor().visit(tree)
    return owners


def _source_violations(relative: str, source: str) -> list[str]:
    """Return boundary violations for one Python source unit.

    Performance (D-CIM-3): this used to run FOUR independent full-tree
    traversals per file — ``_function_ancestors`` (its own recursive visitor)
    plus three separate ``ast.walk(tree)`` passes (imports, then assigns,
    then calls). Profiled on the ~3100-file scan (``agent_utilities`` +
    ``scripts`` + ``tests`` + ``examples``), that put this check at ~34s,
    too slow to add to the merge queue's 180s fast tier alongside its tests.
    Two changes cut that without changing which violations are reported:

    1. ``_function_ancestors`` is only ever consulted for ONE file — the
       canonical boundary module itself (to allow ``create_context_agent``'s
       own internal ``Agent(...)`` call) — yet it ran for every file. It is
       now computed only when ``canonical`` is true.
    2. The imports pass stays its OWN full walk, run to completion first —
       it must fully populate ``pydantic_modules``/``pydantic_agent_aliases``
       before any call is judged, and ``ast.walk``'s breadth-first order
       does not guarantee a deeper-nested aliased import is seen before an
       earlier-enqueued shallower call in the same file (only same-or-lesser
       depth is guaranteed relative to same-scope code — mixing them into
       one walk would be a false-negative risk on a zero-tolerance gate for
       that edge case, even though it does not occur in this repo today).
       The assigns and calls checks do not depend on each other's results,
       so THEY are merged into one combined second walk — cutting the
       original three full-tree walks to two.
    3. Every violation this function can report requires at least one of
       ``_TRIGGER_MARKERS`` to appear literally in the source text (see that
       constant's docstring). A file that parses cleanly but mentions none
       of them cannot trigger anything below the parse step, so the two
       tree walks are skipped for it entirely — the parse (and its
       syntax-error detection) still runs unconditionally for every file,
       so this narrows only redundant work, never the failure surface.
    """

    failures: list[str] = []
    try:
        tree = ast.parse(source, filename=relative)
    except SyntaxError as exc:
        return [f"{relative}: parse failed: {exc}"]

    # A file parses fine but mentions none of ``_TRIGGER_MARKERS`` — it
    # cannot trigger any check below (the ast-walk passes AND the
    # routers-only raw-HTTP-transport text check all require one of these
    # substrings to be present). Skip the two expensive tree walks; the
    # parse-failure detection above still runs for every file regardless, so
    # this never weakens THAT signal.
    if relative.startswith("agent_utilities/server/routers/"):
        router_markers = (
            "api.openai.com",
            "api.anthropic.com",
            "generativelanguage.googleapis.com",
            'client.stream("POST"',
            'client.post("POST"',
        )
        if any(marker in source for marker in router_markers) and (
            "compile_model_context(" not in source
        ):
            failures.append(
                f"{relative}: raw HTTP model transport bypasses ContextCompiler"
            )
    if not any(marker in source for marker in _TRIGGER_MARKERS):
        return failures

    canonical = relative == CANONICAL_AGENT_BOUNDARY
    owners: dict[ast.AST, str] = _function_ancestors(tree) if canonical else {}
    pydantic_modules: set[str] = set()
    pydantic_agent_aliases: set[str] = {"Agent", "PydanticAgent"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "pydantic_ai":
                    pydantic_modules.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and (
            node.module == "pydantic_ai"
            or bool(node.module and node.module.startswith("pydantic_ai."))
        ):
            for alias in node.names:
                if alias.name == "*" and not canonical:
                    failures.append(
                        f"{relative}:{node.lineno}: wildcard PydanticAI import may "
                        "re-export Agent; import the required non-Agent symbols"
                    )
                    continue
                if alias.name != "Agent":
                    continue
                local_name = alias.asname or alias.name
                pydantic_agent_aliases.add(local_name)
                if not canonical and relative not in METADATA_ONLY_AGENT_ALLOWLIST:
                    failures.append(
                        f"{relative}:{node.lineno}: direct PydanticAI Agent import "
                        f"{local_name!r}; use create_context_agent"
                    )

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if value is not None and _dotted(value).endswith(".Agent"):
                if any(
                    isinstance(target, ast.Name) and target.id == "Agent"
                    for target in targets
                ):
                    failures.append(
                        f"{relative}:{node.lineno}: direct Agent re-export is "
                        "forbidden; use create_context_agent"
                    )
        elif isinstance(node, ast.Call):
            called = _dotted(node.func)
            raw_call = called.endswith("chat.completions.create") or called.endswith(
                "responses.create"
            )
            raw_constructor = called in {"OpenAI", "AsyncOpenAI", "Anthropic"}
            if (
                relative.startswith("agent_utilities/")
                and (raw_call or raw_constructor)
                and relative not in RAW_PROVIDER_ALLOWLIST
            ):
                failures.append(
                    f"{relative}:{node.lineno}: raw model provider bypass {called}"
                )

            module_agent_call = any(
                called == f"{module_name}.Agent" for module_name in pydantic_modules
            )
            named_agent_call = called in pydantic_agent_aliases
            if not (module_agent_call or named_agent_call):
                continue
            if canonical and owners.get(node) == "create_context_agent":
                continue
            if relative in METADATA_ONLY_AGENT_ALLOWLIST:
                continue
            failures.append(
                f"{relative}:{node.lineno}: direct Agent construction {called}; "
                "use create_context_agent"
            )

    return failures


def violations() -> list[str]:
    failures: list[str] = []
    for scan_root in SCAN_ROOTS:
        for path in sorted(scan_root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            relative = path.relative_to(ROOT).as_posix()
            try:
                source = path.read_text(encoding="utf-8")
            except OSError as exc:
                failures.append(f"{relative}: read failed: {exc}")
                continue
            failures.extend(_source_violations(relative, source))

    factory = (PACKAGE / "core" / "model_factory.py").read_text(encoding="utf-8")
    if "wrap_model_with_context(model)" not in factory:
        failures.append("model_factory.py does not install mandatory context wrapper")
    wrapper = (PACKAGE / "core" / "contextual_model.py").read_text(encoding="utf-8")
    if "def create_context_agent(" not in wrapper:
        failures.append("contextual_model.py does not own the Agent constructor")
    if "model=wrap_model_with_context(model)" not in wrapper:
        failures.append("create_context_agent does not govern the supplied model")
    for transport in (
        "async def request(",
        "async def count_tokens(",
        "async def compact_messages(",
        "async def request_stream(",
    ):
        if transport not in wrapper:
            failures.append(
                f"contextual_model.py does not govern {transport.removesuffix('(')}"
            )
    graph_state = (PACKAGE / "graph" / "state.py").read_text(encoding="utf-8")
    if (
        "def __post_init__" not in graph_state
        or "governed(self.agent_model)" not in graph_state
    ):
        failures.append("GraphDeps leaves direct model-id construction ungoverned")
    serving = (
        PACKAGE
        / "knowledge_graph"
        / "retrieval"
        / "context_compiler_serving.py"
    ).read_text(encoding="utf-8")
    if "compile_model_context(" not in serving:
        failures.append("context serving does not compile governed evidence")
    return failures


def main() -> int:
    failures = violations()
    if failures:
        print("ContextCompiler boundary gate failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("ContextCompiler boundary gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
