"""Orchestration substrate: agent/team/prompt/workflow specs → graph (KG-2.10 S4b).

Shared base for KG‑driven orchestration synthesis. Defines the artifact specs
(``AgentSpec``/``TeamSpec``/``PromptSpec``/``WorkflowSpec``) the synthesis engine
produces, and converts them to ``ExtractionBatch`` (typed nodes + edges) so they
persist through the one ``GraphBackend`` via ``registry.write_batch`` — same as
every other source. The OWL layer (``ontology_orchestration.ttl`` + owl_bridge)
then reasons over capability reachability, team coverage, and gaps.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .models import EnrichmentEdge, ExtractionBatch, GraphNode


def _slug(text: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")[:80] or "x"


class PromptSpec(BaseModel):
    """A prompt the KG created or evolved to meet an identified need."""

    name: str
    content: str
    role: str = "system"
    evolved_from: str | None = None  # prior prompt id (lineage)
    rationale: str = ""

    @property
    def id(self) -> str:
        return f"prompt:{_slug(self.name)}"


class AgentSpec(BaseModel):
    """A synthesized agent: prompt + tools + skills + chosen model for a goal."""

    name: str
    goal: str = ""
    system_prompt: str = ""
    prompt_id: str | None = None
    tools: list[str] = Field(default_factory=list)  # tool names/ids
    skills: list[str] = Field(default_factory=list)  # skill names/ids
    model: str = ""  # chosen model id (e.g. "qwen-lite" / "qwen/qwen3.5-9b")
    description: str = ""

    @property
    def id(self) -> str:
        return f"agent:{_slug(self.name)}"


def select_model(models: list[dict] | None, complexity: str = "normal") -> str:
    """Pick the right model id from the KG-known models for a task complexity.

    ``complexity``: "light" (routing/simple) → a routing-capable/light model;
    "normal"/"super" → a KG/heavier model. ``models`` are dicts shaped like the
    config.json chat-model entries (id, intelligence_level, can_route, can_kg).
    Returns "" when no models are known (caller falls back to defaults).
    """
    if not models:
        return ""

    def pick(pred):
        for m in models:
            if pred(m):
                return m.get("id", "")
        return ""

    if complexity == "light":
        return (
            pick(lambda m: m.get("can_route"))
            or pick(lambda m: m.get("intelligence_level") == "light")
            or models[0].get("id", "")
        )
    # heavier work prefers a KG-capable / non-light model
    return (
        pick(lambda m: m.get("can_kg"))
        or pick(lambda m: m.get("intelligence_level") in ("normal", "super"))
        or models[0].get("id", "")
    )


class TeamSpec(BaseModel):
    """A hierarchical team of specialized agents for a goal."""

    name: str
    goal: str = ""
    lead: str = ""  # lead agent name
    members: list[str] = Field(default_factory=list)  # agent names
    # explicit reporting edges (child reports_to parent); defaults to all→lead
    reports_to: list[tuple[str, str]] = Field(default_factory=list)
    description: str = ""

    @property
    def id(self) -> str:
        return f"team:{_slug(self.name)}"


class WorkflowSpec(BaseModel):
    """An orchestration flow over agents/skills (a skill-workflow)."""

    name: str
    steps: list[str] = Field(default_factory=list)  # ordered step ids/names
    orchestrates: list[str] = Field(default_factory=list)  # agent/skill ids

    @property
    def id(self) -> str:
        return f"workflow:{_slug(self.name)}"


# ── spec → graph batch (uniform persistence) ──────────────────────────────────
def prompt_to_batch(p: PromptSpec) -> ExtractionBatch:
    nodes = [
        GraphNode(
            id=p.id,
            type="Prompt",
            props={
                "name": p.name,
                "role": p.role,
                "content": p.content[:8000],
                "rationale": p.rationale,
            },
        )
    ]
    edges = []
    if p.evolved_from:
        edges.append(
            EnrichmentEdge(source=p.id, target=p.evolved_from, rel_type="EVOLVED_FROM")
        )
    return ExtractionBatch(category="orchestration", nodes=nodes, edges=edges)


def agent_to_batch(a: AgentSpec) -> ExtractionBatch:
    nodes = [
        GraphNode(
            id=a.id,
            type="Agent",
            props={
                "name": a.name,
                "goal": a.goal,
                "description": a.description,
                "system_prompt": a.system_prompt[:8000],
            },
        )
    ]
    edges: list[EnrichmentEdge] = []
    if a.goal:
        gid = f"goal:{_slug(a.goal)}"
        nodes.append(GraphNode(id=gid, type="Goal", props={"name": a.goal}))
        edges.append(EnrichmentEdge(source=a.id, target=gid, rel_type="SOLVES"))
    if a.prompt_id:
        edges.append(
            EnrichmentEdge(source=a.id, target=a.prompt_id, rel_type="HAS_PROMPT")
        )
    for t in a.tools:
        edges.append(
            EnrichmentEdge(source=a.id, target=f"tool:{_slug(t)}", rel_type="USES_TOOL")
        )
    for s in a.skills:
        edges.append(
            EnrichmentEdge(
                source=a.id, target=f"skill:{_slug(s)}", rel_type="HAS_SKILL"
            )
        )
    if a.model:
        edges.append(
            EnrichmentEdge(
                source=a.id, target=f"model:{_slug(a.model)}", rel_type="USES_MODEL"
            )
        )
    return ExtractionBatch(category="orchestration", nodes=nodes, edges=edges)


def team_to_batch(t: TeamSpec) -> ExtractionBatch:
    nodes = [
        GraphNode(
            id=t.id,
            type="Team",
            props={
                "name": t.name,
                "goal": t.goal,
                "description": t.description,
                "lead": t.lead,
            },
        )
    ]
    edges: list[EnrichmentEdge] = []
    if t.goal:
        gid = f"goal:{_slug(t.goal)}"
        nodes.append(GraphNode(id=gid, type="Goal", props={"name": t.goal}))
        edges.append(EnrichmentEdge(source=t.id, target=gid, rel_type="SOLVES"))
    for m in t.members:
        aid = f"agent:{_slug(m)}"
        edges.append(EnrichmentEdge(source=aid, target=t.id, rel_type="MEMBER_OF_TEAM"))
    # Hierarchy: explicit reports_to, else every non-lead member reports to lead.
    pairs = t.reports_to or [(m, t.lead) for m in t.members if t.lead and m != t.lead]
    for child, parent in pairs:
        edges.append(
            EnrichmentEdge(
                source=f"agent:{_slug(child)}",
                target=f"agent:{_slug(parent)}",
                rel_type="REPORTS_TO",
            )
        )
    return ExtractionBatch(category="orchestration", nodes=nodes, edges=edges)


def workflow_to_batch(w: WorkflowSpec) -> ExtractionBatch:
    nodes = [
        GraphNode(
            id=w.id, type="Workflow", props={"name": w.name, "steps": ",".join(w.steps)}
        )
    ]
    edges = [
        EnrichmentEdge(source=w.id, target=o, rel_type="ORCHESTRATES")
        for o in w.orchestrates
    ]
    return ExtractionBatch(category="orchestration", nodes=nodes, edges=edges)
