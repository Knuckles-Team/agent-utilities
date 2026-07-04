"""Universal-skills workflow → KG WorkflowDefinition ingestion.

CONCEPT:AU-KG.ingest.skill-workflow-ingestion — Skill-Workflow Ingestion ("Claude drives, graph-os runs").

The ~300 *skill-workflows* under ``universal_skills/workflows/<domain>/<name>/
SKILL.md`` are dual-mode artefacts: a YAML frontmatter (``name``/``description``/
``domain``/``tags``/``team_config``) plus a body ``## Steps`` section whose
``### Step N: <component> [depends_on: ...]`` headings encode the machine DAG.
Each step's ``<component>`` names an *atomic* skill the workflow composes.

Until now those workflows lived only on disk — a live query shows ~2
``WorkflowDefinition`` nodes vs ~300 workflows on disk, so the graph-os
orchestrator had nothing to dispatch. This module closes that gap: it parses
each ``SKILL.md`` and lands a ``WorkflowDefinition`` (+ ``WorkflowStep`` DAG +
``Skill`` links) in the **exact** shape
:class:`~agent_utilities.knowledge_graph.workflow_store.WorkflowStore` writes,
so ``graph_orchestrate action=execute_workflow`` / the ``kg-delegate``
skill can discover them by ``name`` and dispatch them.

Node / edge shape (mirrors ORCH-1.22 ``WorkflowStore`` + ORCH-1.41 compiler)::

    (:WorkflowDefinition {id: "skill_workflow:<name>", name, description,
                          domain, source: "universal-skills", tags_json,
                          nl_spec, step_count, content_hash, ...})
      -[:HAS_STEP {step_order}]-> (:WorkflowStep {node_id, step_order,
                          component, depends_on_json, ...})
    (:WorkflowStep) -[:TRANSITION_TO]-> (:WorkflowStep)   # depends_on edges
    (:WorkflowStep) -[:USES_SKILL]->     (:Skill {id: "skill:<component>", name})

``ingest_skill_workflows`` is **idempotent**: deterministic ids mean a re-run
upserts in place, and a ``content_hash`` on the definition lets an unchanged
workflow be a no-op (counted under ``skipped``).

NOTE on the *execution* seam: this module only INGESTS — it puts the workflow
in the store/shape ``execute_workflow`` reads. It deliberately does NOT touch
the step-by-step execution dispatch (a known-flaky area); a workflow is made
*dispatchable*, not executed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)

# ``### Step 1: network-topology-sweep`` or
# ``### Step 2: Decompose By Strategy [depends_on: fetch_trades, Step 1]``
_STEP_RE = re.compile(
    r"^###\s+Step\s+(\d+):\s*([^\[\n]+?)(?:\s*\[depends_on:\s*([^\]]+)\])?\s*$",
    re.MULTILINE,
)
# Body fields the corpus renders under a step heading.
_AGENT_RE = re.compile(r"\*\*Agent\*\*:\s*`?([^`\n]+)`?")
_TOOLS_RE = re.compile(r"\*\*Tools\*\*:\s*`?([^`\n]+)`?")


def _slug(text: str) -> str:
    """Stable id-safe slug (matches the workflow corpus' depends_on dialect)."""
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_") or "step"


def _looks_like_skill_name(component: str) -> bool:
    """A kebab/snake token (one word-group, no spaces) is an atomic-skill ref."""
    return bool(re.fullmatch(r"[a-z0-9][a-z0-9_-]*", component.strip()))


def parse_workflow_skill(skill_md: Path) -> dict[str, Any] | None:
    """Parse a workflow ``SKILL.md`` into ``{name, description, domain, tags,
    specialist_ids, tool_assignments, concept, steps}``.

    Reimplemented (not imported) from the skill-workflow-builder reference so
    agent-utilities owns its parser. Accepts all three ``depends_on`` dialects
    the corpus uses: numeric (``Step 2`` / ``2``) and name-based (the slugified
    component/title).
    """
    try:
        content = skill_md.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("[KG-2.97] cannot read %s: %s", skill_md, exc)
        return None

    frontmatter: dict[str, Any] = {}
    body = content
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            body = parts[2]
            try:
                import yaml

                frontmatter = yaml.safe_load(parts[1]) or {}
            except Exception as exc:  # noqa: BLE001 — degrade to dir-name defaults
                logger.warning("[KG-2.97] YAML parse failed for %s: %s", skill_md, exc)
                frontmatter = {}

    team_config = frontmatter.get("team_config") or {}
    if not isinstance(team_config, dict):
        team_config = {}

    matches = list(_STEP_RE.finditer(body))
    steps: list[dict[str, Any]] = []
    for i, m in enumerate(matches):
        step_num = int(m.group(1))
        component = m.group(2).strip()
        depends_raw = m.group(3)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        step_body = body[start:end]

        depends_on = (
            [d.strip() for d in depends_raw.split(",") if d.strip()]
            if depends_raw
            else []
        )
        agent_m = _AGENT_RE.search(step_body)
        tools_m = _TOOLS_RE.search(step_body)
        tools = (
            [t.strip() for t in tools_m.group(1).split(",") if t.strip()]
            if tools_m
            else []
        )
        # The atomic skill: the kebab heading component, else the **Agent** body.
        if _looks_like_skill_name(component):
            skill_name = component
        elif agent_m:
            skill_name = agent_m.group(1).strip()
        else:
            skill_name = _slug(component)

        steps.append(
            {
                "step": step_num,
                "component": component,
                "skill_name": skill_name,
                "depends_on": depends_on,
                "tools": tools,
                "description": step_body.strip().split("\n", 1)[0].strip(),
            }
        )

    name = str(frontmatter.get("name") or skill_md.parent.name)
    return {
        "path": str(skill_md),
        "name": name,
        "description": str(frontmatter.get("description") or "").strip(),
        "domain": str(frontmatter.get("domain") or skill_md.parent.parent.name),
        "tags": frontmatter.get("tags") or [],
        "specialist_ids": team_config.get("specialist_ids") or [],
        "tool_assignments": team_config.get("tool_assignments") or {},
        "concept": frontmatter.get("concept"),
        "steps": steps,
        "body": body,
    }


def _resolve_dep(dep: str, comp_to_num: dict[str, int]) -> int | None:
    """Resolve one ``depends_on`` token to a step number (numeric or name)."""
    m = re.fullmatch(r"(?:step\s*)?(\d+)", dep.strip(), re.IGNORECASE)
    if m:
        return int(m.group(1))
    return comp_to_num.get(_slug(dep))


def discover_workflow_skill_files(root: str | None = None) -> list[Path]:
    """Locate every workflow ``SKILL.md`` under the universal-skills package.

    Uses the installed ``universal_skills`` package path (filtered to
    ``/workflows/``) and also accepts an explicit ``root`` (a directory that
    is, or contains, ``workflows/``) for tests and out-of-tree corpora.
    """
    roots: list[Path] = []
    if root:
        rp = Path(root)
        roots.append(rp / "workflows" if (rp / "workflows").is_dir() else rp)
    else:
        try:
            from universal_skills import skill_utilities

            for p in skill_utilities.get_universal_skills_path():
                if "/workflows/" in str(p):
                    roots.append(Path(p))
        except Exception as exc:  # noqa: BLE001 — package may be absent
            logger.warning("[KG-2.97] universal_skills not importable: %s", exc)
        # Fallback: resolve the package root directly when the enable-flag
        # discovery above returns nothing (editable installs can yield []).
        if not roots:
            try:
                import universal_skills

                pkg = Path(next(iter(universal_skills.__path__))) / "workflows"
                if pkg.is_dir():
                    roots.append(pkg)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[KG-2.97] package-path fallback failed: %s", exc)

    seen: set[Path] = set()
    files: list[Path] = []
    for r in roots:
        if not r.exists():
            continue
        for f in sorted(r.rglob("SKILL.md")):
            rf = f.resolve()
            if rf not in seen:
                seen.add(rf)
                files.append(f)
    return files


def _content_hash(parsed: dict[str, Any]) -> str:
    """Stable hash over the parsed semantics → idempotent re-ingest no-op."""
    payload = {
        "name": parsed["name"],
        "description": parsed["description"],
        "domain": parsed["domain"],
        "tags": parsed["tags"],
        "steps": [
            {
                "step": s["step"],
                "component": s["component"],
                "skill_name": s["skill_name"],
                "depends_on": s["depends_on"],
                "tools": s["tools"],
            }
            for s in parsed["steps"]
        ],
    }
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _chunk_workflow_body(
    engine: IntelligenceGraphEngine, wf_id: str, body: str, title: str
) -> None:
    """Chunk + embed a workflow body into Chunk objects linked to ``wf_id`` (KG-2.48).

    The shared semantic-search substrate every skill-type object shares (atomic Skill /
    SkillGraph docs / WorkflowDefinition). Best-effort — never raises (enrichment must
    not block workflow registration); embeds are bounded so a flaky GPU can't hang it.
    """
    if not body.strip():
        return
    try:
        from ..ontology.document_processing import ChunkingConfig, DocumentProcessor

        # Chunk under a distinct body-doc id so the WorkflowDefinition node is NOT
        # clobbered (process() writes a Document at its document_id); link it back.
        body_id = f"{wf_id}::body"
        backend = getattr(engine, "backend", None) or engine
        DocumentProcessor(backend, chunking=ChunkingConfig()).process(
            body, document_id=body_id, title=title, doc_type="skill_workflow"
        )
        try:
            engine.link_nodes(wf_id, body_id, "HAS_BODY", properties={})
        except Exception:  # noqa: BLE001 — linking is best-effort
            pass
    except Exception as exc:  # noqa: BLE001 — enrichment must not block ingest
        logger.debug("[KG-2.97] workflow body chunking skipped for %s: %s", wf_id, exc)


def ingest_one(engine: IntelligenceGraphEngine, parsed: dict[str, Any]) -> str:
    """Upsert a single parsed workflow into the KG as a WorkflowDefinition DAG.

    Returns ``"ingested"`` or ``"skipped"`` (unchanged content_hash).
    """
    name = parsed["name"]
    wf_id = f"skill_workflow:{_slug(name)}"
    chash = _content_hash(parsed)

    # Idempotent no-op: identical content already present.
    existing = engine.query_cypher(
        "MATCH (w:WorkflowDefinition) WHERE w.id = $wid RETURN w.content_hash AS h",
        {"wid": wf_id},
    )
    if existing and existing[0].get("h") == chash:
        return "skipped"

    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    steps = parsed["steps"]
    comp_to_num = {_slug(s["component"]): s["step"] for s in steps}
    num_to_stepid = {s["step"]: f"{wf_id}:step:{s['step']}" for s in steps}

    # nl_spec: a compact, dispatchable rendering of the step DAG.
    nl_lines = [
        f"Step {s['step']}: {s['component']}"
        + (f" [depends_on: {', '.join(s['depends_on'])}]" if s["depends_on"] else "")
        for s in steps
    ]
    nl_spec = parsed["description"] + "\n\nSteps:\n" + "\n".join(nl_lines)

    props: dict[str, Any] = {
        "name": name,
        "description": parsed["description"],
        "domain": parsed["domain"],
        "source": "universal-skills",
        "tags_json": json.dumps(parsed["tags"], default=str),
        "specialist_ids_json": json.dumps(parsed["specialist_ids"], default=str),
        "nl_spec": nl_spec,
        "step_count": len(steps),
        "content_hash": chash,
        "source_path": parsed["path"],
        "last_used": ts,
        "use_count": 0,
        "version": 1,
        # Skill-type-unique marker so the skill family (atomic|graph|workflow) is
        # queryable as one set while each keeps its own structure (here: the step DAG).
        "skill_type": "workflow",
    }
    if parsed.get("concept"):
        props["concept"] = str(parsed["concept"])
    engine.add_node(wf_id, "WorkflowDefinition", properties=props)

    # Same enrichment substrate as skills/skill-graphs/documents: chunk + embed the
    # workflow's prose body into Chunk objects linked to the WorkflowDefinition, so the
    # workflow corpus is semantically searchable (not just dispatchable). Best-effort.
    _chunk_workflow_body(engine, wf_id, str(parsed.get("body") or ""), name)

    for s in steps:
        step_id = num_to_stepid[s["step"]]
        resolved_deps = sorted(
            {
                num_to_stepid[n]
                for d in s["depends_on"]
                if (n := _resolve_dep(d, comp_to_num)) is not None
                and n in num_to_stepid
            }
        )
        step_props: dict[str, Any] = {
            "node_id": step_id,
            "step_order": s["step"],
            "component": s["component"],
            "skill_name": s["skill_name"],
            "is_parallel": not resolved_deps,
            "timeout": 120.0,
            "status": "pending",
            "depends_on_json": json.dumps(resolved_deps),
        }
        if s.get("tools"):
            step_props["tools_json"] = json.dumps(s["tools"], default=str)
        if s.get("description"):
            step_props["refined_subtask"] = s["description"]
        engine.add_node(step_id, "WorkflowStep", properties=step_props)
        engine.link_nodes(
            wf_id, step_id, "HAS_STEP", properties={"step_order": s["step"]}
        )

        # depends_on → TRANSITION_TO edges (predecessor → this step).
        for dep_id in resolved_deps:
            engine.link_nodes(
                dep_id, step_id, "TRANSITION_TO", properties={"condition": "on_success"}
            )

        # Link the step to its atomic Skill node (create-if-absent).
        skill_id = f"skill:{_slug(s['skill_name'])}"
        engine.add_node(
            skill_id,
            "Skill",
            properties={"name": s["skill_name"], "source": "universal-skills"},
        )
        engine.link_nodes(step_id, skill_id, "USES_SKILL")

    return "ingested"


def ingest_skill_workflows(
    engine: IntelligenceGraphEngine, root: str | None = None
) -> dict[str, Any]:
    """Ingest every universal-skills workflow into the KG (CONCEPT:AU-KG.ingest.skill-workflow-ingestion).

    For each ``workflows/<domain>/<name>/SKILL.md`` this parses the frontmatter
    + step DAG and upserts a ``WorkflowDefinition`` (+ ``WorkflowStep`` DAG +
    ``USES_SKILL`` links) in the ``WorkflowStore`` shape ``execute_workflow``
    reads — making the corpus discoverable & dispatchable by graph-os.

    Args:
        engine: the live ``IntelligenceGraphEngine``.
        root: optional explicit corpus root (a dir that is/contains
            ``workflows/``). Defaults to the installed ``universal_skills``
            package.

    Returns:
        Report dict: ``{workflows, steps, skill_links, skipped, errors,
        error_detail}``.
    """
    files = discover_workflow_skill_files(root)
    report: dict[str, Any] = {
        "workflows": 0,
        "steps": 0,
        "skill_links": 0,
        "skipped": 0,
        "errors": 0,
        "error_detail": [],
        "scanned": len(files),
    }
    for f in files:
        try:
            parsed = parse_workflow_skill(f)
            if parsed is None:
                report["errors"] += 1
                report["error_detail"].append(f"{f}: parse returned None")
                continue
            outcome = ingest_one(engine, parsed)
            if outcome == "skipped":
                report["skipped"] += 1
            else:
                report["workflows"] += 1
                report["steps"] += len(parsed["steps"])
                report["skill_links"] += len(parsed["steps"])
        except Exception as exc:  # noqa: BLE001 — one bad file must not abort the run
            report["errors"] += 1
            report["error_detail"].append(f"{f}: {exc}")
            logger.exception("[KG-2.97] ingest failed for %s", f)

    logger.info(
        "[KG-2.97] skill-workflow ingest: %d workflows, %d steps, %d skill links, "
        "%d skipped, %d errors (of %d scanned)",
        report["workflows"],
        report["steps"],
        report["skill_links"],
        report["skipped"],
        report["errors"],
        report["scanned"],
    )
    return report
