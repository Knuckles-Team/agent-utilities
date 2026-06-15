"""Connector → skill synthesis: mapped processes → propose-only skill candidates.

CONCEPT:KG-2.90 / KG-2.91 — a KG-native, propose-only background distiller that
turns the mapped processes of **all** connected systems (egeria / leanix / aris /
camunda) into NEW atomic-skill and skill-workflow PROPOSALS. The distiller is
generic over the ONTOLOGY (``BusinessProcess`` / ``BusinessTask`` / ``flowsTo`` /
``Capability``), never per-connector: every connector already normalizes into
those same ArchiMate/capability classes, so one ontology-driven pass covers them
all.

The heavy lifting — graph traversal, OWL reasoning, classification, dedup,
drafting — runs in the engine/loop. A human/Claude only reviews + approves; the
:meth:`ConnectorSkillDistiller.propose` step writes proposal nodes and provenance
edges (``AUTOMATES`` / ``DERIVED_FROM`` / ``COMPOSES``) and NOTHING lands in any
repo. Materialization (``draft_artifact`` → a staging dir, and on human approval
:class:`PhysicalDistillationEngine`) is the only file-touching step and it never
writes into a source repo.

Pipeline (each stage best-effort, composable, idempotent where it can be):

    discover()  → ontology classes (BusinessProcess flowsTo-chains, BusinessTask,
                  Capability) + unresolved ``manual:`` tasks from
                  ProcessPlanCompiler + OntologyReasoningDriver recurring patterns
    classify()  → single coherent action  → atomic-skill candidate;
                  flowsTo-chain (≥2 tasks) → skill-workflow candidate whose steps
                  map to existing atomic skills / co-created atomic candidates
    dedup()     → reuse ConceptMatcher / the skill registry to skip candidates
                  already covered by an existing skill; keep only the novel ones
    propose()   → write SkillProposal / SkillWorkflowProposal nodes through the
                  propose-only path with AUTOMATES + DERIVED_FROM edges
    draft_artifact() → render a reviewable SKILL.md into a STAGING dir (reusing
                  SkillGraphDistiller + the universal-skills templates), workflows
                  in the dual-mode (Claude + graph-os) format.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_utilities.models.knowledge_graph import RegistryEdgeType, RegistryNodeType

logger = logging.getLogger(__name__)

# Ontology classes the distiller reasons over (case-insensitive match against the
# live LPG ``type`` property). Connector-agnostic by construction — every
# connector lifts into these same classes.
_PROCESS_TYPE = "businessprocess"
_TASK_TYPE = "businesstask"
_CAPABILITY_TYPES = {"capability", "businesscapability"}

# A flowsTo-chain of at least this many tasks is a skill-workflow candidate;
# anything smaller (a lone coherent action) is an atomic-skill candidate.
_WORKFLOW_MIN_STEPS = 2


def _slug(text: str) -> str:
    """Kebab-case a label into a skill name token."""
    s = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return s or "skill"


@dataclass
class SkillCandidate:
    """A classified, pre-dedup skill candidate (atomic or workflow)."""

    candidate_id: str
    name: str  # kebab-case
    description: str
    kind: str  # "atomic" | "workflow"
    source_id: str  # the BusinessProcess/Task/Capability node it came from
    source_system: str  # provenance system label (e.g. camunda/aris/egeria/leanix)
    automates: str | None = None  # the process/capability node it would automate
    trigger_patterns: list[str] = field(default_factory=list)
    # workflow-only: ordered steps, each {name, atomic_id?, depends_on:[idx,...]}
    steps: list[dict[str, Any]] = field(default_factory=list)
    novelty: str = "novel"  # set by dedup: covered | related | novel
    rationale: str = ""

    def to_props(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "trigger_patterns": list(self.trigger_patterns),
            "proposal_status": "proposal",
            "provenance": f"{self.source_system}:{self.source_id}",
            "kind": self.kind,
            "novelty": self.novelty,
            "status": "proposal",
        }


@dataclass
class DistillReport:
    """Structured, JSON-able outcome of one distiller run."""

    discovered: int = 0
    atomic_candidates: int = 0
    workflow_candidates: int = 0
    covered_skipped: int = 0
    proposed: int = 0
    proposal_ids: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "discovered": self.discovered,
            "atomic_candidates": self.atomic_candidates,
            "workflow_candidates": self.workflow_candidates,
            "covered_skipped": self.covered_skipped,
            "proposed": self.proposed,
            "proposal_ids": self.proposal_ids,
            "artifacts": self.artifacts,
            "errors": self.errors[:10],
        }


class ConnectorSkillDistiller:
    """Distil mapped connector processes into propose-only skill candidates.

    CONCEPT:KG-2.90 — connector-agnostic over the ontology; propose-only. Reuses
    the same engine traversal surface as :class:`ProcessPlanCompiler` (the warm L1
    compute mirror, then the backend Cypher fallback) so it works on a seeded
    in-memory engine and a live daemon alike.
    """

    def __init__(
        self,
        engine: Any,
        *,
        staging_root: str | Path | None = None,
        embed_fn: Any = None,
        max_candidates: int = 50,
    ) -> None:
        self.engine = engine
        # Drafts go to a STAGING dir, never into any repo. Default under the
        # workspace scratch area so review artifacts never pollute a source tree.
        self.staging_root = Path(
            staging_root or (Path.home() / "workspace" / "scratch" / "skill-proposals")
        )
        self._embed_fn = embed_fn
        self.max_candidates = max_candidates

    # ── engine traversal (matches ProcessPlanCompiler's robust pattern) ────── #
    @staticmethod
    def _edge_rel(edge_data: dict[str, Any]) -> str:
        return str(edge_data.get("type") or edge_data.get("rel_type") or "").upper()

    def _node_props(self, node_id: str) -> dict[str, Any]:
        graph = getattr(self.engine, "graph", None)
        if graph is not None:
            try:
                data = graph.nodes[node_id]
                if data:
                    return dict(data)
            except Exception:  # noqa: BLE001 — fall through to the backend
                pass
        backend = getattr(self.engine, "backend", None)
        if backend is not None:
            try:
                rows = backend.execute(
                    "MATCH (p) WHERE p.id = $pid RETURN p", {"pid": node_id}
                )
                if rows and isinstance(rows[0].get("p"), dict):
                    return dict(rows[0]["p"])
            except Exception:  # noqa: BLE001 — absent node handled by caller
                pass
        return {}

    def _iter_nodes(self) -> list[tuple[str, dict[str, Any]]]:
        """Every (id, props) node in the graph (compute mirror, then backend)."""
        graph = getattr(self.engine, "graph", None)
        if graph is not None:
            try:
                return [(nid, dict(d or {})) for nid, d in graph.nodes(data=True)]
            except (TypeError, AttributeError):
                pass
        backend = getattr(self.engine, "backend", None)
        if backend is not None:
            try:
                rows = backend.execute("MATCH (n) RETURN n", {})
                out: list[tuple[str, dict[str, Any]]] = []
                for row in rows or []:
                    node = row.get("n")
                    if isinstance(node, dict) and node.get("id"):
                        out.append((str(node["id"]), dict(node)))
                return out
            except Exception:  # noqa: BLE001 — empty graph
                pass
        return []

    def _tasks_of(self, process_id: str) -> dict[str, dict[str, Any]]:
        """BusinessTask nodes PART_OF ``process_id`` (compute mirror then backend)."""
        tasks: dict[str, dict[str, Any]] = {}
        graph = getattr(self.engine, "graph", None)
        if graph is not None:
            try:
                for src, _tgt, edata in graph.in_edges(process_id, data=True):
                    if self._edge_rel(edata or {}) != "PART_OF":
                        continue
                    props = self._node_props(src)
                    if str(props.get("type", "")).lower() == _TASK_TYPE:
                        tasks[src] = props
            except Exception:  # noqa: BLE001 — compute mirror unavailable
                tasks = {}
        backend = getattr(self.engine, "backend", None)
        if not tasks and backend is not None:
            try:
                rows = backend.execute(
                    "MATCH (t)-[:PART_OF]->(p) WHERE p.id = $pid RETURN t",
                    {"pid": process_id},
                )
                for row in rows or []:
                    node = row.get("t")
                    if isinstance(node, dict) and node.get("id"):
                        if str(node.get("type", "")).lower() == _TASK_TYPE:
                            tasks[str(node["id"])] = dict(node)
            except Exception:  # noqa: BLE001
                pass
        return tasks

    def _flows(self, task_ids: set[str]) -> list[tuple[str, str]]:
        """FLOWS_TO edges within ``task_ids`` (compute mirror then backend)."""
        flows: list[tuple[str, str]] = []
        graph = getattr(self.engine, "graph", None)
        if graph is not None:
            try:
                for tid in task_ids:
                    for _src, tgt, edata in graph.out_edges(tid, data=True):
                        if (
                            self._edge_rel(edata or {}) == "FLOWS_TO"
                            and tgt in task_ids
                        ):
                            flows.append((tid, tgt))
            except Exception:  # noqa: BLE001
                flows = []
        backend = getattr(self.engine, "backend", None)
        if not flows and backend is not None:
            try:
                rows = backend.execute(
                    "MATCH (a)-[f:FLOWS_TO]->(b) RETURN a.id AS src, b.id AS tgt", {}
                )
                for row in rows or []:
                    s, t = row.get("src"), row.get("tgt")
                    if s in task_ids and t in task_ids:
                        flows.append((str(s), str(t)))
            except Exception:  # noqa: BLE001
                pass
        return flows

    @staticmethod
    def _label(props: dict[str, Any], fallback: str) -> str:
        return str(
            props.get("name")
            or props.get("element_id")
            or props.get("label")
            or fallback
        )

    @staticmethod
    def _source_system(props: dict[str, Any], node_id: str) -> str:
        """Best-effort provenance system from the node's id prefix / props.

        Connector-agnostic: every connector uses a recognizable id prefix
        (``bpmn_*``/``aris_*``/``egeria_*``/leanix factsheet ids) — we read it
        rather than hard-coding per-connector logic.
        """
        explicit = props.get("source_system") or props.get("connector")
        if explicit:
            return str(explicit)
        nid = str(node_id).lower()
        for prefix, system in (
            ("bpmn", "camunda"),
            ("aris", "aris"),
            ("egeria", "egeria"),
            ("leanix", "leanix"),
        ):
            if nid.startswith(prefix):
                return system
        # leanix factsheets carry their type but no prefix; flag generic.
        return str(props.get("origin") or "connector")

    # ── stage 1: discover ─────────────────────────────────────────────────── #
    def discover(self) -> dict[str, list[dict[str, Any]]]:
        """Find automation candidates over the ontology, connector-agnostic.

        Returns a bundle: ``processes`` (with their flowsTo-ordered tasks),
        standalone ``tasks`` and ``capabilities``, plus ``unresolved`` manual
        tasks (the ProcessPlanCompiler automation gaps) and ``patterns`` (recurring
        cross-process inferences harvested by OntologyReasoningDriver).
        """
        processes: list[dict[str, Any]] = []
        capabilities: list[dict[str, Any]] = []
        for nid, props in self._iter_nodes():
            ntype = str(props.get("type", "")).lower()
            if ntype == _PROCESS_TYPE:
                tasks = self._tasks_of(nid)
                flows = self._flows(set(tasks))
                processes.append(
                    {
                        "id": nid,
                        "props": props,
                        "tasks": tasks,
                        "flows": flows,
                        "system": self._source_system(props, nid),
                    }
                )
            elif ntype in _CAPABILITY_TYPES:
                capabilities.append(
                    {
                        "id": nid,
                        "props": props,
                        "system": self._source_system(props, nid),
                    }
                )
        return {
            "processes": processes,
            "capabilities": capabilities,
            "unresolved": self._discover_unresolved(processes),
            "patterns": self._discover_patterns(),
        }

    def _discover_unresolved(
        self, processes: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Unresolved ``manual:`` tasks (no agent/tool) — the automation gaps.

        Reuses :class:`ProcessPlanCompiler`; ``plan.metadata["unresolved_tasks"]``
        is the list of task labels with no agent/tool match. Best-effort: a
        compiler/event-loop hiccup never aborts discovery.
        """
        out: list[dict[str, Any]] = []
        try:
            from agent_utilities.knowledge_graph.process_plan_compiler import (
                ProcessPlanCompiler,
            )

            compiler = ProcessPlanCompiler(self.engine)
        except Exception as exc:  # noqa: BLE001 — compiler unavailable
            logger.debug("[KG-2.90] ProcessPlanCompiler unavailable: %s", exc)
            return out

        from ..research.loop_controller import _run_coro

        for proc in processes:
            try:
                plan = _run_coro(compiler.compile(proc["id"]))
                for label in plan.metadata.get("unresolved_tasks", []) or []:
                    out.append(
                        {
                            "label": label,
                            "process_id": proc["id"],
                            "system": proc["system"],
                        }
                    )
            except Exception as exc:  # noqa: BLE001 — one process failing is fine
                logger.debug("[KG-2.90] compile %s failed: %s", proc["id"], exc)
        return out

    def _discover_patterns(self) -> list[dict[str, Any]]:
        """Recurring cross-process inferences from OWL/RDF reasoning.

        Reuses :class:`OntologyReasoningDriver`; the harvested new topics are
        candidate cross-process automations the per-process scan would miss.
        Best-effort and lightweight.
        """
        try:
            from ..research.ara.reasoning_driver import OntologyReasoningDriver

            harvest = OntologyReasoningDriver(self.engine).extrapolate()
        except Exception as exc:  # noqa: BLE001 — reasoning best-effort
            logger.debug("[KG-2.90] reasoning extrapolate failed: %s", exc)
            return []
        return [dict(t) for t in (harvest.new_topics or [])][: self.max_candidates]

    # ── stage 2: classify ─────────────────────────────────────────────────── #
    def classify(
        self, discovered: dict[str, list[dict[str, Any]]]
    ) -> list[SkillCandidate]:
        """Map discovered structure to atomic-skill / skill-workflow candidates.

        - A single coherent action/capability (a lone BusinessTask, a Capability,
          or an unresolved ``manual:`` task) → an atomic-skill candidate.
        - A flowsTo-chain of ≥2 tasks → a skill-workflow candidate whose steps map
          (by semantic name) to an existing atomic skill or to an atomic candidate
          created alongside it.
        """
        candidates: list[SkillCandidate] = []
        seen: set[str] = set()

        def _add(c: SkillCandidate) -> None:
            if c.name in seen:
                return
            seen.add(c.name)
            candidates.append(c)

        for proc in discovered.get("processes", []):
            tasks = proc["tasks"]
            flows = proc["flows"]
            system = proc["system"]
            proc_label = self._label(proc["props"], proc["id"])
            # order the tasks by the flowsTo chain (topological-ish; falls back to
            # insertion order for disconnected/parallel tasks).
            ordered = self._order_tasks(tasks, flows)
            # action tasks only (skip gateways) for the workflow body.
            action_ids = [tid for tid in ordered if not tasks[tid].get("is_gateway")]
            if len(action_ids) >= _WORKFLOW_MIN_STEPS:
                steps = []
                for tid in action_ids:
                    step_name = _slug(self._label(tasks[tid], tid))
                    atomic = SkillCandidate(
                        candidate_id=f"cand:atomic:{tid}",
                        name=step_name,
                        description=f"Atomic step '{step_name}' of the {proc_label} process.",
                        kind="atomic",
                        source_id=tid,
                        source_system=system,
                        automates=proc["id"],
                        trigger_patterns=[self._label(tasks[tid], tid)],
                        rationale=f"flowsTo step in process {proc['id']}",
                    )
                    _add(atomic)
                    steps.append({"name": step_name, "atomic_id": atomic.candidate_id})
                # link each step's depends_on by the flow predecessors.
                self._wire_step_deps(steps, action_ids, flows)
                wf = SkillCandidate(
                    candidate_id=f"cand:workflow:{proc['id']}",
                    name=_slug(proc_label),
                    description=(
                        f"Automate the {proc_label} business process end-to-end "
                        f"({len(steps)} steps) — distilled from {system}."
                    ),
                    kind="workflow",
                    source_id=proc["id"],
                    source_system=system,
                    automates=proc["id"],
                    trigger_patterns=[proc_label, f"run {proc_label}"],
                    steps=steps,
                    rationale=f"flowsTo-chain of {len(steps)} tasks",
                )
                _add(wf)
            else:
                # a single coherent action → atomic-skill candidate
                for tid in action_ids:
                    name = _slug(self._label(tasks[tid], tid))
                    _add(
                        SkillCandidate(
                            candidate_id=f"cand:atomic:{tid}",
                            name=name,
                            description=f"Automate the '{self._label(tasks[tid], tid)}' task ({system}).",
                            kind="atomic",
                            source_id=tid,
                            source_system=system,
                            automates=proc["id"],
                            trigger_patterns=[self._label(tasks[tid], tid)],
                            rationale="single coherent action",
                        )
                    )

        for cap in discovered.get("capabilities", []):
            label = self._label(cap["props"], cap["id"])
            _add(
                SkillCandidate(
                    candidate_id=f"cand:atomic:{cap['id']}",
                    name=_slug(label),
                    description=f"Provide the '{label}' capability as an atomic skill ({cap['system']}).",
                    kind="atomic",
                    source_id=cap["id"],
                    source_system=cap["system"],
                    automates=cap["id"],
                    trigger_patterns=[label],
                    rationale="capability → atomic skill",
                )
            )

        for gap in discovered.get("unresolved", []):
            label = str(gap["label"])
            _add(
                SkillCandidate(
                    candidate_id=f"cand:atomic:manual:{_slug(label)}",
                    name=_slug(label),
                    description=f"Automate the currently-manual '{label}' task ({gap['system']}).",
                    kind="atomic",
                    source_id=gap["process_id"],
                    source_system=gap["system"],
                    automates=gap["process_id"],
                    trigger_patterns=[label],
                    rationale="unresolved manual task (automation gap)",
                )
            )
        return candidates

    @staticmethod
    def _order_tasks(
        tasks: dict[str, dict[str, Any]], flows: list[tuple[str, str]]
    ) -> list[str]:
        """Topologically order tasks by FLOWS_TO; stable fallback for cycles."""
        succ: dict[str, list[str]] = {tid: [] for tid in tasks}
        indeg: dict[str, int] = {tid: 0 for tid in tasks}
        for s, t in flows:
            if s in tasks and t in tasks:
                succ[s].append(t)
                indeg[t] += 1
        ready = [tid for tid in tasks if indeg[tid] == 0]
        order: list[str] = []
        seen: set[str] = set()
        while ready:
            tid = ready.pop(0)
            if tid in seen:
                continue
            seen.add(tid)
            order.append(tid)
            for nxt in succ[tid]:
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    ready.append(nxt)
        # any leftover (cycle / disconnected) appended in stable order
        for tid in tasks:
            if tid not in seen:
                order.append(tid)
        return order

    @staticmethod
    def _wire_step_deps(
        steps: list[dict[str, Any]], action_ids: list[str], flows: list[tuple[str, str]]
    ) -> None:
        """Set each step's ``depends_on`` (1-based step indices) from FLOWS_TO."""
        idx_of = {tid: i for i, tid in enumerate(action_ids)}
        for s, t in flows:
            if s in idx_of and t in idx_of:
                steps[idx_of[t]].setdefault("depends_on", []).append(idx_of[s] + 1)

    # ── stage 3: dedup ────────────────────────────────────────────────────── #
    def dedup(self, candidates: list[SkillCandidate]) -> list[SkillCandidate]:
        """Drop candidates already covered by an existing skill; keep novel ones.

        Reuses the :class:`ConceptMatcher` machinery (id + embedding-recall +
        LLM-judge fusion) against the existing ``skill`` registry nodes. When no
        embedder/LLM is reachable it degrades to a deterministic name match (an
        existing skill with the same kebab name ⇒ covered) so the propose-only
        path never blocks on an unavailable model.
        """
        existing = self._existing_skills()
        if not existing:
            for c in candidates:
                c.novelty = "novel"
            return candidates

        existing_names = {n.lower() for n in existing.values()}
        kept: list[SkillCandidate] = []
        # deterministic name pass (always runs; offline-safe)
        for c in candidates:
            if c.name.lower() in existing_names:
                c.novelty = "covered"
                c.rationale = f"{c.rationale}; covered by existing skill name match"
            else:
                c.novelty = "novel"

        # semantic pass — only if an embedder is reachable AND there are still
        # name-novel candidates. Best-effort; an embedding outage leaves the
        # name-pass verdicts intact.
        novel = [c for c in candidates if c.novelty == "novel"]
        if novel:
            try:
                self._semantic_dedup(novel, existing)
            except Exception as exc:  # noqa: BLE001 — embedder/LLM optional
                logger.debug("[KG-2.90] semantic dedup skipped: %s", exc)

        kept = [c for c in candidates if c.novelty != "covered"]
        return kept

    def _existing_skills(self) -> dict[str, str]:
        """Map existing ``skill`` node id → name (compute mirror then backend)."""
        out: dict[str, str] = {}
        for nid, props in self._iter_nodes():
            if str(props.get("type", "")).lower() == RegistryNodeType.SKILL.value:
                out[nid] = str(props.get("name") or props.get("title") or nid)
        return out

    def _semantic_dedup(
        self, candidates: list[SkillCandidate], existing: dict[str, str]
    ) -> None:
        """Embed candidates + existing skills; mark covered when cosine is high."""
        from ..assimilation.concept_matcher import COVERED_COSINE, _top_k_cosine

        if self._embed_fn is None:
            from ..enrichment.semantic import make_embed_fn

            self._embed_fn = make_embed_fn()
        existing_ids = list(existing)
        ex_vecs = self._embed_fn([existing[i] for i in existing_ids])
        concept_vecs = list(zip(existing_ids, ex_vecs, strict=False))
        cand_vecs = self._embed_fn([f"{c.name} — {c.description}" for c in candidates])
        for c, vec in zip(candidates, cand_vecs, strict=False):
            top = _top_k_cosine(list(vec), concept_vecs, 1, COVERED_COSINE)
            if top:
                c.novelty = "covered"
                c.rationale = (
                    f"{c.rationale}; covered by '{existing[top[0][0]]}' "
                    f"(cos={top[0][1]:.2f})"
                )

    # ── stage 4: propose (PROPOSE-ONLY) ───────────────────────────────────── #
    def propose(self, candidates: list[SkillCandidate]) -> list[str]:
        """Persist SkillProposal / SkillWorkflowProposal nodes — propose-only.

        Writes the proposal node plus its provenance edges and nothing else:
        ``AUTOMATES`` → the source process/capability, ``DERIVED_FROM`` → the
        source system node. A workflow proposal also writes ``COMPOSES`` edges to
        its atomic step candidates. No code runs, no repo is touched. Idempotent
        per ``candidate_id`` (re-runs upsert the same node id).
        """
        ids: list[str] = []
        for c in candidates:
            node_type = (
                RegistryNodeType.SKILL_WORKFLOW_PROPOSAL.value
                if c.kind == "workflow"
                else RegistryNodeType.SKILL_PROPOSAL.value
            )
            pid = f"{node_type}:{c.name}"
            try:
                self.engine.add_node(pid, node_type, properties=c.to_props())
            except Exception as exc:  # noqa: BLE001 — persistence best-effort
                logger.debug("[KG-2.90] proposal persist failed for %s: %s", pid, exc)
                continue
            link = getattr(self.engine, "link_nodes", None)
            if callable(link):
                try:
                    if c.automates:
                        link(pid, c.automates, RegistryEdgeType.AUTOMATES.value)
                    link(pid, c.source_id, RegistryEdgeType.DERIVED_FROM.value)
                    if c.kind == "workflow":
                        for step in c.steps:
                            aid = step.get("atomic_id")
                            if aid:
                                # COMPOSES → the atomic step proposal node id.
                                step_pid = (
                                    f"{RegistryNodeType.SKILL_PROPOSAL.value}:"
                                    f"{step['name']}"
                                )
                                link(pid, step_pid, RegistryEdgeType.COMPOSES.value)
                except Exception as exc:  # noqa: BLE001 — edge writes best-effort
                    logger.debug("[KG-2.90] proposal edges failed for %s: %s", pid, exc)
            ids.append(pid)
        return ids

    # ── stage 5: draft_artifact (STAGING dir — never a repo) ───────────────── #
    def draft_artifact(self, candidate: SkillCandidate) -> str:
        """Render a reviewable SKILL.md into the STAGING dir; return its path.

        Atomic skills get a standard frontmatter SKILL.md (reused by the
        PhysicalDistillationEngine on approval). Workflows are emitted in the
        DUAL-MODE format so they run under Claude AND graph-os (see
        :func:`render_workflow_skill_md`).
        """
        out_dir = self.staging_root / candidate.name
        out_dir.mkdir(parents=True, exist_ok=True)
        skill_md = out_dir / "SKILL.md"
        if candidate.kind == "workflow":
            content = render_workflow_skill_md(candidate)
        else:
            content = render_atomic_skill_md(candidate)
        skill_md.write_text(content, encoding="utf-8")
        return str(skill_md)

    # ── materialization (human-approved → physical SKILL.md) ───────────────── #
    def materialize(
        self, proposal_id: str, *, skill_code_path: str | None = None
    ) -> dict[str, Any]:
        """Materialize an APPROVED proposal to a physical SKILL.md, then stamp it.

        The review→approve→materialize close-out: reads the SkillProposal /
        SkillWorkflowProposal node, renders its SKILL.md into the staging dir, and
        writes the frontmatter via :class:`PhysicalDistillationEngine.distill_skill`
        (the same physical-distillation path used for evolved skills). Marks the
        node ``proposal_status="approved"``. Still never touches a source repo —
        the artifact lands in the staging dir for a human to place. Returns a
        JSON-able report.
        """
        props = self._node_props(proposal_id)
        if not props:
            return {"proposal_id": proposal_id, "status": "not_found"}
        kind = "workflow" if "workflow" in str(props.get("type", "")) else "atomic"
        candidate = SkillCandidate(
            candidate_id=proposal_id,
            name=str(props.get("name") or proposal_id),
            description=str(props.get("description") or ""),
            kind=kind,
            source_id=proposal_id,
            source_system=str(props.get("provenance") or "connector").split(":", 1)[0],
            trigger_patterns=list(props.get("trigger_patterns") or []),
        )
        skill_md = self.draft_artifact(candidate)
        code_path = skill_code_path or skill_md
        from .physical_distiller import PhysicalDistillationEngine

        ok = False
        try:
            ok = PhysicalDistillationEngine().distill_skill(
                skill_id=proposal_id,
                new_name=candidate.name,
                new_description=candidate.description,
                skill_code_path=code_path,
                tags=["skill", kind, candidate.source_system],
            )
        except Exception as exc:  # noqa: BLE001 — materialization best-effort
            logger.debug(
                "[KG-2.90] physical distill failed for %s: %s", proposal_id, exc
            )
        try:
            self.engine.add_node(
                proposal_id,
                str(props.get("type")),
                properties={
                    **props,
                    "proposal_status": "approved",
                    "status": "approved",
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("[KG-2.90] approval stamp failed for %s: %s", proposal_id, exc)
        return {
            "proposal_id": proposal_id,
            "status": "approved" if ok else "drafted",
            "skill_md": skill_md,
            "materialized": ok,
        }

    # ── orchestration ─────────────────────────────────────────────────────── #
    def run(self, *, draft: bool = False) -> DistillReport:
        """One propose-only pass: discover → classify → dedup → propose [→ draft].

        Returns a :class:`DistillReport`. ``draft=True`` also renders SKILL.md
        artifacts into the staging dir (off by default in the loop to keep the
        cycle cheap — drafting happens on review/approval).
        """
        report = DistillReport()
        discovered = self.discover()
        report.discovered = (
            len(discovered["processes"])
            + len(discovered["capabilities"])
            + len(discovered["unresolved"])
        )
        candidates = self.classify(discovered)
        report.atomic_candidates = sum(1 for c in candidates if c.kind == "atomic")
        report.workflow_candidates = sum(1 for c in candidates if c.kind == "workflow")
        before = len(candidates)
        candidates = self.dedup(candidates)
        report.covered_skipped = before - len(candidates)
        report.proposal_ids = self.propose(candidates)
        report.proposed = len(report.proposal_ids)
        if draft:
            for c in candidates:
                try:
                    report.artifacts.append(self.draft_artifact(c))
                except Exception as exc:  # noqa: BLE001 — drafting best-effort
                    report.errors.append(f"draft {c.name}: {exc}")
        return report


# --------------------------------------------------------------------------- #
# Artifact rendering (reuses the universal-skills SKILL.md conventions)
# --------------------------------------------------------------------------- #
def render_atomic_skill_md(candidate: SkillCandidate) -> str:
    """Render a standard atomic-skill SKILL.md (frontmatter + body)."""
    triggers = ", ".join(candidate.trigger_patterns) or candidate.name
    suffix = f" Triggers: {triggers}."
    desc = candidate.description
    if len(desc) + len(suffix) > 1024:
        desc = desc[: 1024 - len(suffix) - 3] + "..."
    # JSON literal = YAML-safe double-quoted scalar (the Triggers suffix adds a colon).
    return (
        "---\n"
        f"name: {candidate.name}\n"
        f"description: {json.dumps(desc + suffix)}\n"
        "domain: process-automation\n"
        f"tags: [skill, atomic, {candidate.source_system}]\n"
        "concept: KG-2.90\n"
        "---\n\n"
        f"# {candidate.name}\n\n"
        f"{candidate.description}\n\n"
        "## Provenance\n\n"
        f"Distilled (propose-only) from `{candidate.source_system}` node "
        f"`{candidate.source_id}`. This is a PROPOSAL for human review — it has "
        "not landed in any repository.\n"
    )


def render_workflow_skill_md(candidate: SkillCandidate) -> str:
    """Render the DUAL-MODE skill-workflow SKILL.md (Claude AND graph-os).

    The body carries a machine-readable step DAG (``### Step N: <atomic-skill>
    [depends_on: ...]``) followed by a Claude-executable ``## Execution`` section
    and the standard graph-os delegation footer.
    """
    triggers = ", ".join(candidate.trigger_patterns) or candidate.name
    specialist_ids = [s["name"] for s in candidate.steps]
    # The SKILL.md step DAG references steps by their 1-based number ("Step N") so the
    # universal-skills validator resolves them. ``depends_on`` may carry step numbers
    # (the distiller's own shape), "Step N", or step names — normalize all three.
    name_to_idx = {s["name"]: i for i, s in enumerate(candidate.steps, start=1)}

    def _dep_to_num(dep: Any) -> int | None:
        sd = str(dep).strip()
        m = re.fullmatch(r"(?:step\s*)?(\d+)", sd, re.IGNORECASE)
        if m:
            return int(m.group(1))
        return name_to_idx.get(sd)

    # Keep the description + appended "Triggers:" suffix within the 1024-char Claude
    # frontmatter limit that the universal-skills gate enforces.
    suffix = f" Triggers: {triggers}."
    desc = candidate.description
    if len(desc) + len(suffix) > 1024:
        desc = desc[: 1024 - len(suffix) - 3] + "..."

    lines: list[str] = []
    lines.append("---")
    lines.append(f"name: {candidate.name}")
    # Quote the description: it contains ": " (the Triggers suffix) which would make
    # an unquoted YAML scalar parse as a mapping. A JSON string literal is a valid
    # double-quoted YAML scalar (handles colons, quotes, unicode).
    lines.append(f"description: {json.dumps(desc + suffix)}")
    lines.append("domain: process-automation")
    lines.append(f"tags: [skill-workflow, dual-mode, {candidate.source_system}]")
    lines.append("team_config:")
    lines.append("  specialist_ids: [" + ", ".join(specialist_ids) + "]")
    lines.append("  tool_assignments:")
    for s in candidate.steps:
        lines.append(f"    {s['name']}: [{s['name']}]")
    lines.append("concept: KG-2.90")
    lines.append("---")
    lines.append("")
    lines.append(f"# {candidate.name}")
    lines.append("")
    lines.append(desc)
    lines.append("")
    lines.append("## Steps")
    lines.append("")
    for i, s in enumerate(candidate.steps, start=1):
        dep_nums = sorted(
            {
                n
                for d in set(s.get("depends_on", []))
                if (n := _dep_to_num(d)) is not None
            }
        )
        if dep_nums:
            dep_str = ", ".join(f"Step {n}" for n in dep_nums)
            lines.append(f"### Step {i}: {s['name']} [depends_on: {dep_str}]")
            lines.append(f"Run after {dep_str} completes.")
        else:
            lines.append(f"### Step {i}: {s['name']}")
            lines.append(
                "No dependencies — safe to run in parallel with other "
                "independent steps."
            )
        lines.append("")
    lines.append("## Execution")
    lines.append("")
    lines.append(
        "Execute the steps above as a dependency DAG. Run every step with NO "
        "`depends_on` in parallel; run each dependent step only after all the "
        "steps it lists in `depends_on` have completed. Each step names an atomic "
        "skill — invoke that skill for the step."
    )
    lines.append("")
    lines.append(
        "If graph-os is reachable, offload the whole DAG via `graph_orchestrate "
        "action=execute_workflow` (or the kg-delegation-router skill); otherwise "
        "execute steps natively in dependency order."
    )
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append(
        f"Distilled (propose-only) from `{candidate.source_system}` process "
        f"`{candidate.source_id}`. This is a PROPOSAL for human review — it has "
        "not landed in any repository."
    )
    lines.append("")
    return "\n".join(lines)
