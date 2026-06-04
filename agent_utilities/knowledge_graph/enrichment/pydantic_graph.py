"""Discover the pydantic-graph orchestration graph from code facts (KG-2.10).

``pydantic-ai`` / ``pydantic-graph`` model an orchestration graph as a set of
classes that subclass ``BaseNode`` — each node implements
``async def run(...) -> NextNodeType | End`` — assembled in a ``Graph(nodes=[...])``.

This module turns that *implicit* structure, recorded by the Rust parser as
``CodeEntity`` class facts (``bases``/``methods``/``name``/``kind``), into the
*explicit* orchestration substrate the rest of the KG already understands. The
discovered node set is rendered as a :class:`WorkflowSpec` (REUSED from
``orchestration``) and persisted through the one ``GraphBackend`` via
``registry.write_batch`` — same uniform path as every other source.

It also proposes concrete *evolutions* to the discovered flow via an injected
LLM function, so the system can reason over and improve its own orchestration.
Everything here is testable and offline: ``llm_fn`` is injected, no network.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from .models import CodeEntity
from .orchestration import WorkflowSpec, workflow_to_batch
from .registry import write_batch

# Base classes that mark a class as a pydantic-graph node. Matched on the short
# name (tolerant to dotted / generic-parameterised forms like
# ``BaseNode[State, Deps, Out]`` or ``pydantic_graph.BaseNode``).
_NODE_BASES = {"basenode", "node"}


def _short_base(base: str) -> str:
    """Reduce a base expression to its bare class name for tolerant matching."""
    b = str(base).strip()
    b = b.split("[", 1)[0]  # drop generic params: BaseNode[State] -> BaseNode
    b = b.rsplit(".", 1)[-1]  # drop module path: pydantic_graph.BaseNode -> BaseNode
    return b.strip().lower()


def _is_node_class(entity: CodeEntity) -> bool:
    if entity.kind != "class":
        return False
    return any(_short_base(b) in _NODE_BASES for b in entity.bases)


def discover_pydantic_graph(code: list[CodeEntity]) -> dict:
    """Identify pydantic-graph node classes among ``code`` (CONCEPT:KG-2.10).

    A node class is a ``CodeEntity`` with ``kind == "class"`` whose ``bases``
    include any of ``{"BaseNode", "Node"}`` (matched on the short name, so
    dotted / generic forms are tolerated).

    Returns a dict with::

        {
          "nodes":      [name, ...],            # node class names (in input order)
          "node_ids":   [id, ...],              # corresponding CodeEntity ids
          "file_paths": {name: file_path, ...}, # where each node lives
          "entrypoint": name | None,            # best-effort graph entry node
        }
    """
    nodes: list[str] = []
    node_ids: list[str] = []
    file_paths: dict[str, str] = {}
    for entity in code:
        if _is_node_class(entity):
            nodes.append(entity.name)
            node_ids.append(entity.id)
            file_paths[entity.name] = entity.file_path

    return {
        "nodes": nodes,
        "node_ids": node_ids,
        "file_paths": file_paths,
        "entrypoint": _detect_entrypoint(code, nodes),
    }


def _detect_entrypoint(code: list[CodeEntity], node_names: list[str]) -> str | None:
    """Best-effort guess of the graph entrypoint.

    Heuristics (in priority order):
      1. A ``*Graph*`` class (the assembly that owns ``Graph(nodes=[...])``).
      2. A node class whose name reads like a start (``Start``/``Begin``/``Entry``).
      3. A node class exposing a ``run``/``start`` method.
    """
    if not node_names:
        # The Graph assembly class may not be a node itself.
        for entity in code:
            if entity.kind == "class" and "graph" in entity.name.lower():
                return entity.name
        return None

    for entity in code:
        if entity.kind == "class" and "graph" in entity.name.lower():
            return entity.name

    start_re = re.compile(r"(start|begin|entry|init)", re.IGNORECASE)
    for name in node_names:
        if start_re.search(name):
            return name

    by_name = {e.name: e for e in code}
    for name in node_names:
        ent = by_name.get(name)
        methods = {m.lower() for m in (ent.methods if ent else [])}
        if methods & {"run", "start"}:
            return name

    return node_names[0]


def pydantic_graph_to_workflow(
    discovered: dict, name: str = "discovered-graph"
) -> WorkflowSpec:
    """Build a :class:`WorkflowSpec` from a discovery dict (CONCEPT:KG-2.10).

    Steps are the node class names; ``orchestrates`` are the node CodeEntity ids
    — so the workflow persists via the shared ``workflow_to_batch`` path and the
    ``ORCHESTRATES`` edges point at the actual code symbols.
    """
    return WorkflowSpec(
        name=name,
        steps=list(discovered.get("nodes", [])),
        orchestrates=list(discovered.get("node_ids", [])),
    )


def _build_evolution_prompt(discovered: dict, context: str) -> str:
    nodes = discovered.get("nodes", [])
    entry = discovered.get("entrypoint")
    parts = [
        "You are improving a pydantic-graph orchestration flow.",
        f"Discovered node classes (in declared order): {nodes}.",
    ]
    if entry:
        parts.append(f"Entrypoint: {entry}.")
    if context:
        parts.append(f"Additional context: {context}")
    parts.append(
        "Propose concrete improvements to the orchestration flow — "
        "reordering, adding, removing, or parallelizing steps. "
        "Respond ONLY with a JSON array of objects, each "
        '{"change": "...", "rationale": "..."}.'
    )
    return "\n".join(parts)


def _parse_proposals(raw: str) -> list[dict]:
    """Leniently extract a JSON array of proposals from an LLM response.

    Finds the first ``[ ... ]`` span and parses it; returns ``[]`` on any
    failure so a malformed model response never crashes the pipeline.
    """
    if not raw:
        return []
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        parsed = json.loads(raw[start : end + 1])
    except (ValueError, TypeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [p for p in parsed if isinstance(p, dict)]


def propose_workflow_evolution(
    discovered: dict, llm_fn: Callable[[str], str], context: str = ""
) -> list[dict]:
    """Ask an injected LLM to propose evolutions of the discovered flow.

    ``llm_fn`` maps a prompt string to a response string (injected for
    testability — no network here). The response is parsed leniently into a list
    of ``{"change", "rationale"}`` dicts; an empty list is returned on any parse
    failure. (CONCEPT:KG-2.10)
    """
    prompt = _build_evolution_prompt(discovered, context)
    try:
        raw = llm_fn(prompt)
    except Exception:  # pragma: no cover - llm transport
        return []
    return _parse_proposals(raw or "")


def persist(
    backend: Any, discovered: dict, name: str = "discovered-graph"
) -> tuple[int, int]:
    """Persist a discovered graph as a Workflow (+ORCHESTRATES) via write_batch.

    Converts through :func:`pydantic_graph_to_workflow` and the shared
    ``workflow_to_batch`` so the discovered orchestration graph lands in the KG
    through the one ``GraphBackend`` interface. Returns
    ``(nodes_written, edges_written)``. (CONCEPT:KG-2.10)
    """
    spec = pydantic_graph_to_workflow(discovered, name=name)
    batch = workflow_to_batch(spec)
    return write_batch(backend, batch)
