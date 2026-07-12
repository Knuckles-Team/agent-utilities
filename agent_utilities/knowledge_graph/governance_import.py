"""Governance-process importers/exporter — external process models ⇄ skill-workflows.

CONCEPT:AU-KG.ontology.connector-agnostic-proposal — the keystone reframe
(``reports/autonomous-sdlc-loop-design.md`` §6/§7.1, decision #4): every governing
process — Camunda BPMN, ARIS EPC, ArchiMate business process, OneTrust/ERPNext
approval flow — is translated INTO the fleet's one executable process language,
the ``:WorkflowDefinition``/``:WorkflowStep`` DAG (= ``:Procedure``), so an agent can
execute any of them via ``WorkflowRunner``. Their approval/sign-off constructs become
``kind="gate"`` steps (the suspend/resume gate of §7.1), and a ``REALIZES`` back-edge
records the descriptive→executable bridge (ORCH-1.41) for lineage/provenance.

Direction in (import):
    * **Camunda BPMN** → reuse :class:`ProcessPlanCompiler` (already imports a lifted
      ``:BusinessProcess`` subgraph). :meth:`GovernanceImporter.import_bpmn`.
    * **ARIS EPC** → walk the ``:EPCFunction``/``:flowsTo`` chain (aris-mcp's
      ``ingest_model_graph``), functions become steps, events/rules collapse away.
    * **ArchiMate** → walk ``Triggering``/``Flow`` from a ``:BusinessProcess`` element
      (archimate-mcp's direct element→element edges).
    * **OneTrust / ERPNext** → a single ``kind="gate"`` step whose completion is the
      assessment/workflow reaching an approved state.

Direction out (export):
    * :func:`export_workflow` serializes a stored ``:WorkflowDefinition`` back to
      **BPMN 2.0 XML** (gate steps → ``userTask``, others → ``serviceTask``), **JSON**,
      or a **SKILL.md** — the round-trip the fleet was missing (``export_workflow`` was
      a deprecated stub). This is what makes "we speak their language" bidirectional
      (e.g. push a fleet-authored governed workflow out to Camunda via ``camunda_deploy``).

Every importer is best-effort/guarded: a missing source subgraph, an absent engine,
or a malformed model degrades to an error dict rather than raising. Gate detection is
a shared heuristic (:func:`looks_like_gate`) over the source node's own type/name
(approval/review/sign-off/authorize/DPIA), overridable per import.
"""

from __future__ import annotations

import html
import json
import logging
import re
import time
from typing import TYPE_CHECKING, Any

from agent_utilities.knowledge_graph.process_plan_compiler import (
    ProcessCompilationError,
    ProcessPlanCompiler,
)

if TYPE_CHECKING:
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)

# Source-node signals that a step is a governance gate (approval/sign-off) rather
# than ordinary work — the words governance models use for human-decision steps.
_GATE_WORDS = (
    "approv",
    "review",
    "sign-off",
    "signoff",
    "sign off",
    "authoriz",
    "authorise",
    "dpia",
    "assessment",
    "gate",
    "consent",
    "decision",
)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_") or "step"


def looks_like_gate(props: dict[str, Any]) -> bool:
    """True when a source node reads as an approval/sign-off gate (heuristic)."""
    if props.get("is_gate") or props.get("kind") in ("gate", "approval"):
        return True
    hay = " ".join(
        str(props.get(k) or "")
        for k in ("name", "label", "task_type", "objectType", "type", "symbolName")
    ).lower()
    return any(w in hay for w in _GATE_WORDS)


class GovernanceImporter:
    """Import external governance process models into ``:WorkflowDefinition`` DAGs.

    CONCEPT:AU-KG.ontology.connector-agnostic-proposal — one convergent target, many front-ends.
    """

    def __init__(self, engine: IntelligenceGraphEngine) -> None:
        self.engine = engine

    # ------------------------------------------------------------------
    # graph reads (tolerant — compute graph then backend)
    # ------------------------------------------------------------------
    def _node_props(self, node_id: str) -> dict[str, Any]:
        graph = getattr(self.engine, "graph", None)
        if graph is not None:
            try:
                data = graph.nodes[node_id]
                if data:
                    return dict(data)
            except Exception:  # noqa: BLE001
                pass
        backend = getattr(self.engine, "backend", None)
        if backend is not None:
            try:
                rows = backend.execute(
                    "MATCH (p) WHERE p.id = $pid RETURN p", {"pid": node_id}
                )
                if rows and isinstance(rows[0].get("p"), dict):
                    return dict(rows[0]["p"])
            except Exception:  # noqa: BLE001
                pass
        return {}

    def _out(self, node_id: str) -> list[tuple[str, str]]:
        """``[(rel_type, target)]`` off ``node_id`` (compute graph, then backend)."""
        graph = getattr(self.engine, "graph", None)
        if graph is not None:
            try:
                return [
                    (
                        str(
                            (e or {}).get("type") or (e or {}).get("rel_type") or ""
                        ),
                        t,
                    )
                    for _s, t, e in graph.out_edges(node_id, data=True)
                ]
            except Exception:  # noqa: BLE001
                pass
        backend = getattr(self.engine, "backend", None)
        if backend is not None:
            try:
                rows = backend.execute(
                    "MATCH (a)-[r]->(b) WHERE a.id = $sid "
                    "RETURN type(r) AS rel, b.id AS tgt",
                    {"sid": node_id},
                )
                return [(str(r.get("rel")), str(r.get("tgt"))) for r in rows or []]
            except Exception:  # noqa: BLE001
                pass
        return []

    # ------------------------------------------------------------------
    # normalization → (tasks, flows) in the ProcessPlanCompiler shape
    # ------------------------------------------------------------------
    def _walk_sequence(
        self,
        roots: list[str],
        flow_rels: set[str],
        *,
        is_executable: Any,
    ) -> tuple[dict[str, dict[str, Any]], list[tuple[str, str, Any]]]:
        """BFS the flow graph from ``roots`` over ``flow_rels`` edges.

        Returns ``(tasks, flows)`` in the shape ``ProcessPlanCompiler._collapse_gateways``
        consumes: non-executable nodes are marked ``is_gateway`` so they collapse
        away, executable nodes keep their real props.
        """
        tasks: dict[str, dict[str, Any]] = {}
        flows: list[tuple[str, str, Any]] = []
        frontier = list(roots)
        seen: set[str] = set()
        while frontier:
            nid = frontier.pop(0)
            if nid in seen:
                continue
            seen.add(nid)
            props = self._node_props(nid)
            if not is_executable(props):
                props = {**props, "is_gateway": True}
            tasks[nid] = props
            for rel, tgt in self._out(nid):
                if rel in flow_rels:
                    flows.append((nid, tgt, None))
                    if tgt not in seen:
                        frontier.append(tgt)
        flows = [(s, t, c) for s, t, c in flows if s in tasks and t in tasks]
        return tasks, flows

    def _hasobject_children(self, model_id: str, link_rel: str) -> list[str]:
        return [t for rel, t in self._out(model_id) if rel == link_rel]

    # ------------------------------------------------------------------
    # importers
    # ------------------------------------------------------------------
    def import_epc(
        self, model_id: str, name: str | None = None, domain: str = "governance"
    ) -> dict[str, Any]:
        """Import an ARIS EPC model (``:ProcessModel``) as a ``:WorkflowDefinition``.

        Functions (``:EPCFunction``) become steps; events/rules (``:EPCEvent``/
        ``:EPCRule``) are the pass-through structure collapsed away. A function that
        reads as an approval/sign-off becomes a ``kind="gate"`` step.
        """
        children = self._hasobject_children(model_id, "hasObject")
        if not children:
            return {"error": f"EPC model {model_id!r} has no :hasObject objects", "source": model_id}

        def _is_fn(props: dict[str, Any]) -> bool:
            return str(props.get("type") or "").endswith("EPCFunction")

        tasks, flows = self._walk_sequence(children, {"flowsTo"}, is_executable=_is_fn)
        return self._compile_and_store(
            tasks, flows, model_id, name or self._node_props(model_id).get("name"), domain, "aris-epc"
        )

    def import_archimate(
        self, process_id: str, name: str | None = None, domain: str = "governance"
    ) -> dict[str, Any]:
        """Import an ArchiMate business process (``:BusinessProcess`` element) as a
        ``:WorkflowDefinition`` by walking ``Triggering``/``Flow`` edges."""
        props0 = self._node_props(process_id)
        if not props0:
            return {"error": f"ArchiMate element {process_id!r} not found", "source": process_id}

        def _is_behavior(props: dict[str, Any]) -> bool:
            t = str(props.get("type") or "")
            return t in ("BusinessProcess", "BusinessFunction", "BusinessInteraction", "ApplicationProcess")

        tasks, flows = self._walk_sequence(
            [process_id], {"Triggering", "Flow"}, is_executable=_is_behavior
        )
        return self._compile_and_store(
            tasks, flows, process_id, name or props0.get("name"), domain, "archimate"
        )

    async def import_bpmn(
        self, process_id: str, name: str | None = None, domain: str = "governance"
    ) -> dict[str, Any]:
        """Import a Camunda BPMN ``:BusinessProcess`` — delegates to the existing
        :class:`ProcessPlanCompiler` (which already writes the ``REALIZES`` bridge)."""
        try:
            compiler = ProcessPlanCompiler(self.engine)
            report = await compiler.compile_and_store(
                process_id, name=name, domain=domain
            )
            report["translator"] = "camunda-bpmn"
            return report
        except ProcessCompilationError as exc:
            return {"error": str(exc), "source": process_id, "translator": "camunda-bpmn"}

    def import_approval_gate(
        self,
        source_id: str,
        system: str,
        name: str | None = None,
        domain: str = "governance",
        capability: str | None = None,
    ) -> dict[str, Any]:
        """Import a OneTrust assessment / ERPNext (Frappe) approval workflow as a
        SINGLE ``kind="gate"`` step (§6.4 compliance-gate class).

        ``system`` is ``onetrust`` or ``erpnext``; ``capability`` overrides the
        default bound tool (``onetrust_assessments`` / ``erpnext frappe-workflow``).
        """
        props = self._node_props(source_id)
        label = name or props.get("name") or f"{system} approval"
        cap = capability or (
            "onetrust_assessments" if system == "onetrust" else "erpnext_workflow"
        )
        step = {
            "id": f"gate_{_slug(label)}",
            "label": str(label),
            "kind": "gate",
            "depends_on": [],
            "capability": cap,
        }
        return self._store_steps(
            [step], source_id, name or str(label), domain, f"{system}-approval"
        )

    # ------------------------------------------------------------------
    # shared build → store
    # ------------------------------------------------------------------
    def _compile_and_store(
        self,
        tasks: dict[str, dict[str, Any]],
        flows: list[tuple[str, str, Any]],
        source_id: str,
        source_name: Any,
        domain: str,
        translator: str,
    ) -> dict[str, Any]:
        if not tasks:
            return {"error": f"{translator}: no process structure for {source_id!r}", "source": source_id}
        try:
            order, deps, _branches = ProcessPlanCompiler._collapse_gateways(tasks, flows)
        except ProcessCompilationError as exc:
            return {"error": str(exc), "source": source_id, "translator": translator}
        if not order:
            return {"error": f"{translator}: only structural nodes, no steps", "source": source_id}

        steps: list[dict[str, Any]] = []
        for tid in order:
            props = tasks[tid]
            label = str(props.get("name") or props.get("label") or tid)
            steps.append(
                {
                    "id": _slug(label) or _slug(tid),
                    "label": label,
                    "kind": "gate" if looks_like_gate(props) else "task",
                    "depends_on": sorted(_slug(str(tasks[d].get("name") or d)) for d in deps[tid]),
                    "capability": None,
                }
            )
        return self._store_steps(
            steps, source_id, str(source_name or source_id), domain, translator
        )

    def _store_steps(
        self,
        steps: list[dict[str, Any]],
        source_id: str,
        name: str,
        domain: str,
        translator: str,
    ) -> dict[str, Any]:
        """Write a ``:WorkflowDefinition`` + ``:WorkflowStep`` DAG (in the WorkflowStore
        shape ``load_workflow`` reads) + a ``REALIZES`` edge back to the source."""
        wf_name = name or f"{translator}_{_slug(source_id)}"
        wf_id = f"workflow:{translator}:{_slug(wf_name)}"
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        gate_count = sum(1 for s in steps if s["kind"] in ("gate", "approval"))
        self.engine.add_node(
            wf_id,
            "WorkflowDefinition",
            properties={
                "name": wf_name,
                "description": f"Imported from {translator} source {source_id}",
                "domain": domain,
                "source": translator,
                "step_count": len(steps),
                "gate_count": gate_count,
                "created_at": ts,
                "last_used": ts,
                "version": 1,
            },
        )
        for i, s in enumerate(steps):
            step_id = f"{wf_id}:step:{i}"
            self.engine.add_node(
                step_id,
                "WorkflowStep",
                properties={
                    "node_id": s["id"],
                    "step_order": i,
                    "refined_subtask": s["label"],
                    "is_parallel": not s["depends_on"],
                    "depends_on_json": json.dumps(s["depends_on"]),
                    "kind": s["kind"],
                    "condition": "on_approval" if s["kind"] in ("gate", "approval") else "on_success",
                    "timeout": 120.0,
                    "status": "pending",
                    **({"boundCapability": s["capability"]} if s.get("capability") else {}),
                },
            )
            self.engine.link_nodes(
                wf_id, step_id, "HAS_STEP", properties={"step_order": i}
            )
        # The descriptive→executable bridge (ORCH-1.41).
        self.engine.link_nodes(wf_id, source_id, "REALIZES")
        logger.info(
            "[gov-import] %s → %s (%d steps, %d gates)",
            source_id,
            wf_id,
            len(steps),
            gate_count,
        )
        return {
            "workflow_id": wf_id,
            "name": wf_name,
            "step_count": len(steps),
            "gate_count": gate_count,
            "source": source_id,
            "translator": translator,
        }


# --------------------------------------------------------------------------- #
# exporter: :WorkflowDefinition → BPMN / JSON / SKILL.md (the missing round-trip)
# --------------------------------------------------------------------------- #
def _load_steps(engine: Any, name: str) -> tuple[str, list[dict[str, Any]]] | None:
    """Load a stored workflow's ``(wf_id, ordered steps)`` by name (tolerant)."""
    graph = getattr(engine, "graph", None)
    wf_id = None
    if graph is not None:
        try:
            for nid, data in graph.nodes(data=True):
                if (
                    str(data.get("type") or "") == "WorkflowDefinition"
                    and data.get("name") == name
                ):
                    wf_id = nid
                    break
        except Exception:  # noqa: BLE001
            wf_id = None
    if wf_id is None:
        backend = getattr(engine, "backend", None)
        if backend is not None:
            try:
                rows = backend.execute(
                    "MATCH (w:WorkflowDefinition) WHERE w.name = $n RETURN w.id AS id LIMIT 1",
                    {"n": name},
                )
                if rows:
                    wf_id = rows[0].get("id")
            except Exception:  # noqa: BLE001
                wf_id = None
    if wf_id is None:
        return None

    steps: list[tuple[int, dict[str, Any]]] = []
    if graph is not None:
        try:
            for _s, tgt, edata in graph.out_edges(wf_id, data=True):
                if str((edata or {}).get("type") or (edata or {}).get("rel_type") or "") == "HAS_STEP":
                    d = dict(graph.nodes[tgt])
                    steps.append((int(d.get("step_order", 0)), d))
        except Exception:  # noqa: BLE001
            steps = []
    if not steps:
        backend = getattr(engine, "backend", None)
        if backend is not None:
            try:
                rows = backend.execute(
                    "MATCH (w:WorkflowDefinition {id: $wid})-[:HAS_STEP]->(s:WorkflowStep) "
                    "RETURN s.node_id AS node_id, s.refined_subtask AS label, "
                    "s.step_order AS step_order, s.kind AS kind, "
                    "s.depends_on_json AS depends_on ORDER BY s.step_order",
                    {"wid": wf_id},
                )
                for r in rows or []:
                    steps.append((int(r.get("step_order", 0)), dict(r)))
            except Exception:  # noqa: BLE001
                steps = []
    steps.sort(key=lambda x: x[0])
    return str(wf_id), [d for _o, d in steps]


def _xml_escape(text: str) -> str:
    return html.escape(str(text), quote=True)


def export_workflow(engine: Any, name: str, fmt: str = "bpmn") -> dict[str, Any]:
    """Serialize a stored ``:WorkflowDefinition`` to ``bpmn`` | ``json`` | ``skill``.

    CONCEPT:AU-KG.ontology.connector-agnostic-proposal — the round-trip the fleet was missing
    (``export_workflow`` was a deprecated stub). Gate steps → BPMN ``userTask``
    (human decision), ordinary steps → ``serviceTask``; ``depends_on`` → sequence
    flows. Returns ``{format, name, content, step_count}`` or ``{error}``.
    """
    loaded = _load_steps(engine, name)
    if loaded is None:
        return {"error": f"WorkflowDefinition {name!r} not found"}
    wf_id, steps = loaded
    fmt = (fmt or "bpmn").lower()

    if fmt == "json":
        content = json.dumps(
            {
                "workflow_id": wf_id,
                "name": name,
                "steps": [
                    {
                        "id": s.get("node_id"),
                        "label": s.get("label") or s.get("refined_subtask"),
                        "kind": s.get("kind") or "task",
                        "depends_on": _decode_deps(s),
                    }
                    for s in steps
                ],
            },
            indent=2,
            default=str,
        )
    elif fmt == "skill":
        content = _to_skill_md(name, steps)
    else:
        content = _to_bpmn(name, steps)
    return {"format": fmt, "name": name, "content": content, "step_count": len(steps)}


def _decode_deps(step: dict[str, Any]) -> list[str]:
    raw = step.get("depends_on") or step.get("depends_on_json")
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str) and raw:
        try:
            return [str(x) for x in json.loads(raw)]
        except (json.JSONDecodeError, TypeError):
            return []
    return []


def _step_id(step: dict[str, Any], idx: int) -> str:
    return _slug(str(step.get("node_id") or step.get("label") or f"step{idx}"))


def _to_bpmn(name: str, steps: list[dict[str, Any]]) -> str:
    """Render an ordered step list as minimal BPMN 2.0 XML (userTask for gates)."""
    proc_id = _slug(name)
    ids = {_step_id(s, i): i for i, s in enumerate(steps)}
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<bpmn2:definitions '
        'xmlns:bpmn2="http://www.omg.org/spec/BPMN/20100524/MODEL" '
        f'id="{proc_id}_defs" targetNamespace="http://agent-utilities.dev/bpmn">',
        f'  <bpmn2:process id="{proc_id}" name="{_xml_escape(name)}" isExecutable="true">',
        '    <bpmn2:startEvent id="start"/>',
    ]
    flows: list[tuple[str, str, str | None]] = []
    for i, s in enumerate(steps):
        sid = _step_id(s, i)
        kind = str(s.get("kind") or "task").lower()
        tag = "userTask" if kind in ("gate", "approval") else "serviceTask"
        label = _xml_escape(str(s.get("label") or s.get("refined_subtask") or sid))
        lines.append(f'    <bpmn2:{tag} id="{sid}" name="{label}"/>')
        deps = [d for d in (_slug(x) for x in _decode_deps(s)) if d in ids]
        if not deps:
            flows.append(("start", sid, None))
        else:
            for d in deps:
                flows.append((d, sid, None))
    # steps that nothing depends on → flow to end
    has_succ = {src for src, _t, _c in flows}
    lines.append('    <bpmn2:endEvent id="end"/>')
    for i, s in enumerate(steps):
        sid = _step_id(s, i)
        if sid not in has_succ:
            flows.append((sid, "end", None))
    for j, (src, tgt, _c) in enumerate(flows):
        lines.append(
            f'    <bpmn2:sequenceFlow id="f{j}" sourceRef="{src}" targetRef="{tgt}"/>'
        )
    lines += ["  </bpmn2:process>", "</bpmn2:definitions>", ""]
    return "\n".join(lines)


def _to_skill_md(name: str, steps: list[dict[str, Any]]) -> str:
    """Render an ordered step list as a workflow SKILL.md (round-trips through the
    skill-workflow ingester, including gate steps via **Kind**)."""
    lines = [
        "---",
        f"name: {name}",
        "description: Exported from a KG WorkflowDefinition.",
        "skill_type: workflow",
        "---",
        "",
        f"# {name}",
        "",
        "## Steps",
        "",
    ]
    for i, s in enumerate(steps, start=1):
        label = str(s.get("label") or s.get("refined_subtask") or _step_id(s, i))
        deps = _decode_deps(s)
        head = f"### Step {i}: {label}"
        if deps:
            head += f" [depends_on: {', '.join(deps)}]"
        lines.append(head)
        kind = str(s.get("kind") or "task").lower()
        if kind in ("gate", "approval"):
            lines.append(f"**Kind**: {kind}")
        lines.append("")
    return "\n".join(lines)
