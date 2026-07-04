#!/usr/bin/python
"""DSTDD (Design-Spec-Test Driven Development) Utility Module.

CONCEPT:AU-AHE.sdd.spec-driven-development — Spec-Driven Development (Extended to DSTDD)

This module provides high-level utilities for managing structured DSTDD artifacts
(Designs, Specifications, Plans, Constitutions) and their relationship to tasks.
It handles disk persistence in the agent's '.specify' directory and provides
logic for dependency analysis.

DSTDD Lifecycle:
    1. **Design Phase**: KG analysis → extension strategy → C4 diagram
    2. **Spec Phase**: User stories → acceptance criteria → NFRs
    3. **Test Phase**: TDD artifacts → validation against KG integrity
"""

import contextlib
import re
from pathlib import Path
from typing import Any, TypeVar

from ..models import (
    DesignDocument,
    ExtensionStrategy,
    ImplementationPlan,
    KGAnalysis,
    ProjectConstitution,
    Spec,
    Task,
    Tasks,
    TaskStatus,
)

T = TypeVar("T", ProjectConstitution, Spec, ImplementationPlan, Tasks, DesignDocument)


def _heuristic_complexity(task: Task) -> dict[str, Any]:
    """Zero-infra structural complexity estimate (CONCEPT:AU-ORCH.planning.sdd-task-ergonomics).

    Proxies complexity from description length, dependency fan-in, touched files,
    and existing subtask count — a deployable default when no LLM scorer is
    injected. Returns a 0-10 score plus a recommended subtask count and an
    expansion prompt.
    """
    desc = str(task.description or "")
    words = len(desc.split())
    score = 0.0
    score += min(4.0, words / 40.0)  # longer specs -> more complex
    score += min(3.0, len(task.depends_on) * 0.75)  # coordination cost
    score += min(2.0, len(task.file_paths) * 0.5)  # surface area
    score += min(1.0, len(task.subtasks) * 0.25)
    score = round(min(10.0, score), 2)
    recommended = 0 if score < 4 else min(8, 2 + int((score - 4) / 1.5))
    prompt = (
        f"Break '{task.title or task.id}' into {recommended} focused subtasks, "
        "each independently testable."
    )
    return {
        "complexity_score": score,
        "recommended_subtasks": recommended,
        "expansion_prompt": prompt if recommended else "",
    }


def _structural_prd_to_tasks(prd_text: str, feature_id: str) -> Tasks:
    """Heuristic PRD → sequential Tasks (CONCEPT:AU-ORCH.planning.sdd-task-ergonomics).

    Splits on markdown headings, numbered items, and bullets. Each becomes a task
    that depends on the prior one (a safe linear default; refine with branch/scope).
    """
    items: list[str] = []
    for raw in prd_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"^(?:#{1,6}\s+|\d+[.)]\s+|[-*+]\s+)(.*)$", line)
        if m and m.group(1).strip():
            items.append(m.group(1).strip())
    if not items:
        # Fall back to non-empty paragraphs.
        items = [p.strip() for p in prd_text.split("\n\n") if p.strip()]
    tasks: list[Task] = []
    prev_id: str | None = None
    for idx, text in enumerate(items, start=1):
        tid = str(idx)
        tasks.append(
            Task(
                id=tid,
                title=text[:80],
                description=text,
                depends_on=[prev_id] if prev_id else [],
                status="pending",
            )
        )
        prev_id = tid
    return Tasks(feature_id=feature_id, tasks=tasks)


class SDDManager:
    """Manages structured DSTDD data within an agent's workspace.

    Supports the full Design → Spec → Test lifecycle with KG-gated
    design validation and Extend-Before-Invent governance.
    """

    def __init__(self, workspace_path: str | Path | None = None):
        self.workspace_root = Path(workspace_path or ".")
        self.specify_dir = self.workspace_root / ".specify"

    def initialize(self, project_name: str):
        """Initialize the SDD environment."""
        self.specify_dir.mkdir(parents=True, exist_ok=True)
        # Create initial constitution
        from ..models import ProjectConstitution

        c = ProjectConstitution(metadata={"project_name": project_name})
        self.save(c)

    def _get_path(self, model_type: type[T], feature_id: str | None = None) -> Path:
        """Resolve the standard path for a DSTDD model (Markdown-first)."""
        if model_type == ProjectConstitution:
            return self.specify_dir / "constitution.md"

        if model_type == DesignDocument:
            if feature_id is None:
                raise ValueError("feature_id is required for DesignDocument")
            return self.specify_dir / "design" / feature_id / "design.md"

        if feature_id is None:
            raise ValueError(f"feature_id is required for {model_type.__name__}")

        if model_type == Spec:
            return self.specify_dir / "specs" / feature_id / "spec.md"

        if model_type == ImplementationPlan:
            return self.specify_dir / "specs" / feature_id / "plan.md"

        if model_type == Tasks:
            return self.specify_dir / "specs" / feature_id / "tasks.md"

        raise ValueError(f"Unsupported DSTDD model type: {model_type}")

    def save(self, model: T, feature_id: str | None = None) -> Path:
        """Persist a DSTDD model to the .specify directory as Markdown."""
        path = self._get_path(type(model), feature_id)
        path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(model, ProjectConstitution):
            content = self._render_constitution_md(model)
        elif isinstance(model, DesignDocument):
            content = self._render_design_md(model)
            # Also write JSON sidecar for round-trip loading
            import json

            json_path = path.parent / "design.json"
            json_path.write_text(
                json.dumps(model.model_dump(), indent=2, default=str),
                encoding="utf-8",
            )
        elif isinstance(model, Spec):
            content = self._render_spec_md(model)
        elif isinstance(model, ImplementationPlan):
            content = self._render_plan_md(model)
        elif isinstance(model, Tasks):
            content = self._render_tasks_md(model)
            # Full-fidelity JSON sidecar (markdown is lossy for fields like
            # priority/complexity_score/subtasks). CONCEPT:AU-ORCH.planning.sdd-task-ergonomics.
            import json

            (path.parent / "tasks.json").write_text(
                json.dumps(model.model_dump(), indent=2, default=str),
                encoding="utf-8",
            )
        else:
            raise ValueError(f"Unsupported model for saving: {type(model)}")

        path.write_text(content, encoding="utf-8")

        # Sync to Knowledge Graph
        self.record_sdd_outcome(model, feature_id)

        return path

    def load(self, model_type: type[T], feature_id: str | None = None) -> T | None:
        """Load a DSTDD model from the .specify directory by parsing Markdown."""
        path = self._get_path(model_type, feature_id)
        if not path.exists():
            return None

        content = path.read_text(encoding="utf-8")

        from typing import cast

        if model_type == ProjectConstitution:
            return cast(T, self._parse_constitution_md(content))
        if model_type == DesignDocument:
            return cast(T, self._load_design(feature_id or ""))
        if model_type == Spec:
            return cast(T, self._parse_spec_md(content, feature_id))
        if model_type == ImplementationPlan:
            return cast(T, self._parse_plan_md(content, feature_id))
        if model_type == Tasks:
            # Prefer the full-fidelity JSON sidecar; fall back to markdown.
            json_path = path.parent / "tasks.json"
            if json_path.exists():
                import json

                data = json.loads(json_path.read_text(encoding="utf-8"))
                return cast(T, Tasks.model_validate(data))
            return cast(T, self.import_from_markdown(path, feature_id or "default"))
        return None

    def _load_design(self, feature_id: str) -> DesignDocument | None:
        """Load a DesignDocument from its JSON sidecar file.

        Design documents are persisted as both markdown (human-readable)
        and JSON (machine-readable) to support round-trip loading.
        """
        json_path = self.specify_dir / "design" / feature_id / "design.json"
        if json_path.exists():
            import json

            data = json.loads(json_path.read_text(encoding="utf-8"))
            return DesignDocument.model_validate(data)
        return None

    def list_specs(self) -> list[dict[str, Any]]:
        """List all specifications in the workspace."""
        specs = []
        specs_dir = self.specify_dir / "specs"
        if specs_dir.exists():
            for d in specs_dir.iterdir():
                if d.is_dir():
                    spec = self.load(Spec, d.name)
                    if spec:
                        specs.append({"id": d.name, "title": spec.title})
        return specs

    def create_spec(self, data: dict[str, Any]) -> Spec:
        """Create and persist a new specification."""
        spec = Spec(**data)
        self.save(spec, spec.feature_id)
        return spec

    def list_plans(self) -> list[dict[str, Any]]:
        """List all implementation plans in the workspace."""
        plans = []
        specs_dir = self.specify_dir / "specs"
        if specs_dir.exists():
            for d in specs_dir.iterdir():
                if d.is_dir():
                    plan = self.load(ImplementationPlan, d.name)
                    if plan:
                        plans.append({"id": d.name, "title": plan.title})
        return plans

    def get_tasks(self, feature_id: str) -> Tasks | None:
        """Retrieve tasks for a specific feature."""
        return self.load(Tasks, feature_id)

    def get_all_tasks(self) -> list[Tasks]:
        """Retrieve all tasks from all features."""
        all_tasks = []
        specs_dir = self.specify_dir / "specs"
        if specs_dir.exists():
            for d in specs_dir.iterdir():
                if d.is_dir():
                    t = self.load(Tasks, d.name)
                    if t:
                        all_tasks.append(t)
        return all_tasks

    def get_constitution(self) -> dict[str, Any] | None:
        """Retrieve the project constitution."""
        c = self.load(ProjectConstitution)
        return c.model_dump() if c else None

    def save_constitution(self, data: dict[str, Any]):
        """Save the project constitution."""
        c = ProjectConstitution(**data)
        self.save(c)

    def sync_to_memory(self, engine: Any, **kwargs):
        """Sync SDD artifacts to Knowledge Graph memory."""
        # record_sdd_outcome is already called on save()
        return None

    def record_sdd_outcome(self, model: T, feature_id: str | None = None):
        """Record the creation or update of an SDD artifact in the Knowledge Graph."""
        from ..knowledge_graph.core.engine import IntelligenceGraphEngine

        engine = IntelligenceGraphEngine.get_active()
        if not engine or not engine.backend:
            return

        artifact_type = type(model).__name__
        name = feature_id if feature_id else "Global"

        query = (
            "MERGE (a:SDDArtifact {id: $id}) "
            "SET a.type = $type, a.name = $name, a.last_updated = timestamp() "
            "RETURN a.id"
        )
        props = {
            "id": f"sdd:{artifact_type}:{name}",
            "type": artifact_type,
            "name": name,
        }
        with contextlib.suppress(Exception):
            engine.backend.execute(query, props)
            # Link to workspace/project node if it exists
            engine.backend.execute(
                "MATCH (p:Project) WHERE p.name = 'current' "
                "MATCH (a:SDDArtifact {id: $id}) "
                "MERGE (p)-[:HAS_ARTIFACT]->(a)",
                {"id": props["id"]},
            )

    def get_parallel_opportunities(self, task_list: Tasks) -> list[list[str]]:
        """Identify ordered waves of tasks that can be safely run in parallel.

        Performs a true dependency-aware topological batching of the pending
        tasks. Tasks are grouped into successive *waves* (batches) such that:

        1. Every task in a wave has all of its dependencies already satisfied --
           either already ``COMPLETED`` or scheduled in a strictly earlier wave.
           Dependent tasks therefore always land in a later batch than the
           tasks they depend on (never the same or an earlier one).
        2. No two tasks in the same wave touch the same ``file_paths`` (file
           collision detection), so tasks within a wave are safe to execute
           concurrently.

        Unlike a single linear pass, this scheduler resolves chains across
        multiple waves: if ``C`` depends on a still-pending ``A``, ``A`` is
        scheduled first and ``C`` is held back until a subsequent wave rather
        than being dropped. Dependencies on unknown/completed task ids are
        treated as already satisfied so the schedule always drains.

        Returns a list of waves, each a list of task ids.
        """
        all_tasks: dict[str, Any] = {}
        completed: set[str] = set()
        for task in task_list.tasks:
            all_tasks[task.id] = task
            if task.status == TaskStatus.COMPLETED:
                completed.add(task.id)

        # Preserve declaration order for deterministic, stable batches.
        pending = [t for t in task_list.tasks if t.status != TaskStatus.COMPLETED]
        pending_ids = {t.id for t in pending}

        # A dependency is "satisfied" once it has been scheduled in an earlier
        # wave or was already completed. Dependencies that are not part of the
        # pending set (unknown ids or already-completed tasks) count as met.
        scheduled: set[str] = set(completed)

        groups: list[list[str]] = []
        remaining = list(pending)

        while remaining:
            current_batch: list[str] = []
            occupied_files: set[str] = set()
            deferred: list[Any] = []

            for task in remaining:
                deps_met = all(
                    dep not in pending_ids or dep in scheduled
                    for dep in task.depends_on
                )
                if not deps_met:
                    # Dependency still unscheduled -> hold for a later wave.
                    deferred.append(task)
                    continue

                # File collision -> cannot share this wave with an earlier task.
                has_collision = any(f in occupied_files for f in task.file_paths)
                if has_collision:
                    deferred.append(task)
                    continue

                current_batch.append(task.id)
                occupied_files.update(task.file_paths)

            if not current_batch:
                # No task became schedulable this pass: a dependency cycle (or
                # mutual file contention with nothing else movable). Break the
                # deadlock by forcing the first remaining task into its own wave
                # so the scheduler always terminates.
                stuck = remaining[0]
                current_batch.append(stuck.id)
                deferred = [t for t in remaining if t.id != stuck.id]

            groups.append(current_batch)
            scheduled.update(current_batch)
            remaining = deferred

        return groups

    # ------------------------------------------------------------------ #
    # Task-management ergonomics (CONCEPT:AU-ORCH.planning.sdd-task-ergonomics)
    # ------------------------------------------------------------------ #
    def parse_prd(
        self,
        prd_text: str,
        feature_id: str,
        generator: Any = None,
    ) -> Tasks:
        """Decompose a PRD into a persisted, dependency-aware task list.

        CONCEPT:AU-ORCH.planning.sdd-task-ergonomics. ``generator(prd_text, feature_id) -> Tasks`` is an
        optional LLM-backed decomposer; when omitted, a zero-infra structural
        parser turns headings / numbered items / bullet lines into sequential
        tasks (each depending on the previous), so PRD intake works with nothing
        deployed. The result is saved under ``feature_id`` and returned.
        """
        if generator is not None:
            tasks = generator(prd_text, feature_id)
        else:
            tasks = _structural_prd_to_tasks(prd_text, feature_id)
        self.save(tasks, feature_id)
        return tasks

    def next_task(self, feature_id: str) -> Task | None:
        """Return the next actionable task for a feature, deps-validated.

        Raises ValueError if the dependency graph is unschedulable (cycle /
        dangling dependency), surfacing the concrete problems.
        """
        tasks = self.get_tasks(feature_id)
        if tasks is None:
            return None
        errors = tasks.validate_dependencies()
        if any(e.startswith("dependency cycle") for e in errors):
            raise ValueError("; ".join(errors))
        return tasks.next_task()

    def set_task_status(
        self, feature_id: str, task_id: str, status: str
    ) -> Tasks | None:
        """Update a task or subtask status and persist (CONCEPT:AU-ORCH.planning.sdd-task-ergonomics)."""
        tasks = self.get_tasks(feature_id)
        if tasks is None:
            return None
        for task in tasks.tasks:
            if task.id == task_id:
                task.status = status
                break
            matched = False
            for sub in task.subtasks:
                if sub.id == task_id:
                    sub.status = status
                    matched = True
                    break
            if matched:
                break
        else:
            raise ValueError(f"task {task_id} not found in feature {feature_id}")
        self.save(tasks, feature_id)
        return tasks

    def analyze_complexity(
        self,
        feature_id: str,
        scorer: Any = None,
    ) -> dict[str, Any]:
        """Score each task's complexity and recommend subtask counts.

        CONCEPT:AU-ORCH.planning.sdd-task-ergonomics. ``scorer`` is an optional callable ``(Task) -> dict``
        returning ``{complexity_score, recommended_subtasks, expansion_prompt}``
        (e.g. LLM-backed). When omitted a zero-infra structural heuristic is used,
        so this works with nothing deployed. Persists a report under
        ``.specify/reports/`` and writes the scores back onto the tasks.
        """
        tasks = self.get_tasks(feature_id)
        if tasks is None:
            raise ValueError(f"no tasks for feature {feature_id}")
        score_fn = scorer or _heuristic_complexity
        analysis: list[dict[str, Any]] = []
        for task in tasks.tasks:
            scored = score_fn(task)
            task.complexity_score = float(scored.get("complexity_score", 0.0))
            task.recommended_subtasks = int(scored.get("recommended_subtasks", 0))
            if scored.get("expansion_prompt"):
                task.expansion_prompt = str(scored["expansion_prompt"])
            analysis.append(
                {
                    "task_id": task.id,
                    "title": task.title,
                    "complexity_score": task.complexity_score,
                    "recommended_subtasks": task.recommended_subtasks,
                    "expansion_prompt": task.expansion_prompt,
                }
            )
        self.save(tasks, feature_id)
        report = {"feature_id": feature_id, "complexity_analysis": analysis}
        reports_dir = self.specify_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        import json

        (reports_dir / f"task-complexity-{feature_id}.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        return report

    def scope_task(
        self,
        feature_id: str,
        task_id: str,
        direction: str,
        strength: str = "regular",
        transformer: Any = None,
    ) -> Task | None:
        """Adjust a task's scope up or down (CONCEPT:AU-ORCH.planning.sdd-task-ergonomics).

        Structural by default: ``scope_down`` collapses pending subtasks and lowers
        the recommended count; ``scope_up`` raises it. Done/in-progress subtasks are
        preserved. An optional ``transformer(task, direction, strength) -> Task``
        (e.g. LLM-backed) can rewrite the task body.
        """
        if direction not in {"up", "down"}:
            raise ValueError("direction must be 'up' or 'down'")
        tasks = self.get_tasks(feature_id)
        if tasks is None:
            return None
        target = next((t for t in tasks.tasks if t.id == task_id), None)
        if target is None:
            raise ValueError(f"task {task_id} not found in feature {feature_id}")

        preserved = {
            "in_progress",
            "completed",
            "done",
            "review",
            "cancelled",
            "deferred",
            "blocked",
        }
        step = {"light": 1, "regular": 2, "heavy": 4}.get(strength, 2)
        if direction == "down":
            target.subtasks = [s for s in target.subtasks if str(s.status) in preserved]
            target.recommended_subtasks = max(0, target.recommended_subtasks - step)
            target.complexity_score = max(0.0, target.complexity_score - step)
        else:
            target.recommended_subtasks += step
            target.complexity_score = min(10.0, target.complexity_score + step)
        if transformer is not None:
            replacement = transformer(target, direction, strength)
            if replacement is not None:
                tasks.tasks = [
                    replacement if t.id == task_id else t for t in tasks.tasks
                ]
                target = replacement
        self.save(tasks, feature_id)
        return target

    def list_task_contexts(self) -> list[str]:
        """List feature ids that have a task list (CONCEPT:AU-ORCH.planning.sdd-task-ergonomics tagged contexts).

        feature_id is the context key: each is an independent, parallel task
        stream (the equivalent of task-master's tags).
        """
        return [t.feature_id for t in self.get_all_tasks()]

    def branch_tasks(self, source_feature_id: str, new_feature_id: str) -> Tasks | None:
        """Fork a task context into a new feature id (CONCEPT:AU-ORCH.planning.sdd-task-ergonomics).

        Copies the source task list under a new context so experiments / feature
        branches can diverge without touching the original.
        """
        src = self.get_tasks(source_feature_id)
        if src is None:
            return None
        clone = src.model_copy(deep=True)
        clone.feature_id = new_feature_id
        self.save(clone, new_feature_id)
        return clone

    def export_to_markdown(self, model: Spec | Tasks, feature_id: str) -> Path:
        """Export an SDD model to a human-readable Markdown file.

        This provides spec-kit parity by maintaining mirrored .md files in the workspace.
        """
        if isinstance(model, Spec):
            path = self.workspace_root / f"spec-{feature_id}.md"
            content = self._render_spec_md(model)
        elif isinstance(model, Tasks):
            path = self.workspace_root / f"tasks-{feature_id}.md"
            content = self._render_tasks_md(model)
        else:
            raise ValueError(f"Unsupported model for markdown export: {type(model)}")

        path.write_text(content, encoding="utf-8")
        return path

    def _render_spec_md(self, spec: Spec) -> str:
        md = [f"# Spec: {spec.title}\n"]
        md.append(f"**Feature ID**: {spec.feature_id}\n")
        md.append("## User Stories")
        for us in spec.user_stories:
            md.append(f"- **{us.title}**: {us.description}")
            for ac in us.acceptance_criteria:
                md.append(f"  - [ ] {ac}")

        if spec.non_functional_requirements:
            md.append("\n## Non-Functional Requirements")
            for req in spec.non_functional_requirements:
                md.append(f"- {req}")

        return "\n".join(md)

    def _render_constitution_md(self, constitution: ProjectConstitution) -> str:
        md = ["# Project Constitution\n"]
        md.append(f"**Vision**: {constitution.vision}")
        md.append(f"**Mission**: {constitution.mission}\n")
        md.append("## Core Principles")
        for principle in constitution.core_principles:
            md.append(f"- {principle}")
        md.append("\n## Tech Stack")
        for k, v in constitution.tech_stack.items():
            md.append(f"- **{k}**: {v}")
        md.append("\n## Metadata")
        for k, v in constitution.metadata.items():
            md.append(f"- **{k}**: {v}")
        return "\n".join(md)

    def _render_plan_md(self, plan: ImplementationPlan) -> str:
        md = ["# Implementation Plan\n"]
        md.append(f"## Approach\n{plan.approach}\n")
        md.append("## Risks")
        for risk in plan.risks:
            md.append(f"- {risk}")
        return "\n".join(md)

    def _render_tasks_md(self, tasks: Tasks) -> str:
        md = [f"# Tasks: {tasks.feature_id}\n"]
        for task in tasks.tasks:
            status_marker = "[x]" if task.status == TaskStatus.COMPLETED else "[ ]"
            parallel_marker = " [P]" if task.parallel else ""
            md.append(f"### {status_marker} {task.id}: {task.title}{parallel_marker}")
            md.append(f"{task.description}")
            if task.depends_on:
                md.append(f"\n**Depends on**: {', '.join(task.depends_on)}")
            if task.file_paths:
                md.append(f"\n**Files**: {', '.join(task.file_paths)}")
            md.append("")

        return "\n".join(md)

    def _parse_constitution_md(self, content: str) -> ProjectConstitution:
        vision = ""
        mission = ""
        principles = []
        tech_stack = {}

        vision_match = re.search(r"\*\*Vision\*\*:\s*(.*)", content)
        if vision_match:
            vision = vision_match.group(1).strip()

        mission_match = re.search(r"\*\*Mission\*\*:\s*(.*)", content)
        if mission_match:
            mission = mission_match.group(1).strip()

        metadata = {}

        # Simple bullet list parsing
        lines = content.splitlines()
        current_section = None
        for line in lines:
            if "## Core Principles" in line:
                current_section = "principles"
            elif "## Tech Stack" in line:
                current_section = "tech"
            elif "## Metadata" in line:
                current_section = "metadata"
            elif line.startswith("- ") and current_section == "principles":
                principles.append(line[2:].strip())
            elif line.startswith("- **") and (
                current_section == "tech" or current_section == "metadata"
            ):
                match = re.match(r"- \*\*(.*?)\*\*:\s*(.*)", line)
                if match:
                    val = match.group(2).strip()
                    if current_section == "tech":
                        tech_stack[match.group(1).strip()] = val
                    else:
                        metadata[match.group(1).strip()] = val

        return ProjectConstitution(
            vision=vision,
            mission=mission,
            core_principles=principles,
            tech_stack=tech_stack,
            metadata=metadata,
        )

    def _parse_spec_md(self, content: str, feature_id: str | None = None) -> Spec:
        title = ""
        title_match = re.search(r"# Spec:\s*(.*)", content)
        if title_match:
            title = title_match.group(1).strip()

        # Simplified parsing
        from ..models import UserStory

        stories = []
        current_story = None
        for line in content.splitlines():
            if line.startswith("- **"):
                match = re.match(r"- \*\*(.*?)\*\*:\s*(.*)", line)
                if match:
                    if current_story:
                        stories.append(current_story)
                    current_story = UserStory(
                        id=match.group(1).strip(),
                        title=match.group(1).strip(),
                        description=match.group(2).strip(),
                        acceptance_criteria=[],
                    )
            elif line.startswith("  - [ ]") and current_story:
                current_story.acceptance_criteria.append(line[7:].strip())
        if current_story:
            stories.append(current_story)

        return Spec(
            feature_id=feature_id or "unknown", title=title, user_stories=stories
        )

    def _parse_plan_md(
        self, content: str, feature_id: str | None = None
    ) -> ImplementationPlan:
        title = "Implementation Plan"
        title_match = re.search(r"^# (.*)", content)
        if title_match:
            title = title_match.group(1).strip()

        approach = ""
        risks = []
        # Simplified parsing
        approach_match = re.search(r"## Approach\n(.*?)\n##", content, re.DOTALL)
        if approach_match:
            approach = approach_match.group(1).strip()

        for line in content.splitlines():
            if line.startswith("- ") and "Risks" in content:
                # This is a bit fragile, but works for basic cases
                risks.append(line[2:].strip())

        return ImplementationPlan(
            feature_id=feature_id or "unknown",
            title=title,
            approach=approach,
            risks=risks,
        )

    def import_from_markdown(self, markdown_path: str | Path, feature_id: str) -> Tasks:
        """Parse a tasks.md file back into a structured Tasks model.

        Supports spec-kit [P] markers for parallel execution detection.
        """
        path = Path(markdown_path)
        content = path.read_text(encoding="utf-8")

        tasks: list[Task] = []
        # Pattern for "### [ ] T1: Title [P]"
        task_pattern = re.compile(
            r"### \[(?P<status>[ xX])\] (?P<id>[A-Za-z0-9_-]+): (?P<title>.*?)(?P<parallel> \[P\])?$"
        )

        current_task: dict[str, Any] | None = None
        for line in content.splitlines():
            match = task_pattern.match(line)
            if match:
                if current_task:
                    tasks.append(Task(**current_task))

                status_char = match.group("status").lower()
                status = (
                    TaskStatus.COMPLETED if status_char == "x" else TaskStatus.PENDING
                )

                current_task = {
                    "id": match.group("id"),
                    "title": match.group("title").strip(),
                    "description": "",
                    "status": status,
                    "parallel": bool(match.group("parallel")),
                    "depends_on": [],
                    "file_paths": [],
                }
            elif current_task:
                if line.startswith("**Depends on**:"):
                    deps = line.replace("**Depends on**:", "").strip()
                    current_task["depends_on"] = [
                        d.strip() for d in deps.split(",") if d.strip()
                    ]
                elif line.startswith("**Files**:"):
                    files = line.replace("**Files**:", "").strip()
                    current_task["file_paths"] = [
                        f.strip() for f in files.split(",") if f.strip()
                    ]
                elif (
                    line.strip()
                    and not line.startswith("#")
                    and not line.startswith("##")
                ):
                    current_task["description"] += line.strip() + "\n"

        if current_task:
            tasks.append(Task(**current_task))

        return Tasks(feature_id=feature_id, tasks=tasks)

    # --- DSTDD Design Phase Methods ---

    def create_design(
        self, feature_id: str, kg_analysis: KGAnalysis, **kwargs: Any
    ) -> DesignDocument:
        """Create and persist a new design document for the DSTDD pipeline.

        This is the entry point for the Design phase. The KG analysis
        must be completed before calling this method.

        Args:
            feature_id: Unique feature identifier.
            kg_analysis: Results of the KG search and extension analysis.
            **kwargs: Additional DesignDocument fields (title, c4_diagram, etc.).

        Returns:
            The persisted DesignDocument.
        """
        title = kwargs.pop("title", feature_id)
        doc = DesignDocument(
            feature_id=feature_id,
            title=title,
            kg_analysis=kg_analysis,
            **kwargs,
        )
        self.save(doc, feature_id)
        return doc

    def validate_design(self, feature_id: str) -> list[str]:
        """Validate a design document against Extend-Before-Invent rules.

        Returns a list of violation messages. An empty list means the design
        passes validation and a Spec can be created.

        Validation rules:
            1. KG analysis must contain at least one nearest concept.
            2. If any nearest concept has similarity >= 0.7, extension_strategy
               must NOT be 'new'.
            3. If extension_strategy is 'new', a NewConceptProposal is required.
            4. New concept proposals must specify a target pillar.
        """
        design = self.load(DesignDocument, feature_id)
        if design is None:
            return [f"Design document not found for feature '{feature_id}'"]

        violations: list[str] = []
        kg = design.kg_analysis

        # Rule 1: Must have analyzed the KG
        if not kg.nearest_concepts:
            violations.append(
                "KG analysis is empty. Run kg_search against the Knowledge Graph "
                "to find nearest existing concepts before creating a design."
            )

        # Rule 2: High-similarity match requires extension, not new concept
        high_sim = [c for c in kg.nearest_concepts if c.similarity >= 0.7]
        if high_sim and kg.extension_strategy == ExtensionStrategy.NEW:
            names = ", ".join(f"{c.concept_id} ({c.similarity:.0%})" for c in high_sim)
            violations.append(
                f"High-similarity concepts found ({names}) but extension_strategy is 'new'. "
                f"You MUST extend an existing concept when similarity >= 70%."
            )

        # Rule 3: New concept requires proposal
        if (
            kg.extension_strategy == ExtensionStrategy.NEW
            and not kg.new_concept_proposal
        ):
            violations.append(
                "Extension strategy is 'new' but no NewConceptProposal provided. "
                "A justification is required for introducing new concepts."
            )

        # Rule 4: Proposal must have pillar assignment
        if kg.new_concept_proposal:
            valid_pillars = {"ORCH", "KG", "AHE", "ECO", "OS"}
            if kg.new_concept_proposal.target_pillar not in valid_pillars:
                violations.append(
                    f"New concept pillar '{kg.new_concept_proposal.target_pillar}' "
                    f"is not valid. Must be one of: {valid_pillars}"
                )

        return violations

    def design_to_spec(self, feature_id: str) -> Spec:
        """Auto-generate a Spec skeleton from a validated design document.

        The design must pass validation (no violations) before a Spec
        can be generated. The generated Spec includes a reference back
        to the design document.

        Args:
            feature_id: The feature ID of the design to convert.

        Returns:
            A new Spec model with user story placeholders.

        Raises:
            ValueError: If the design has validation violations.
        """
        violations = self.validate_design(feature_id)
        if violations:
            raise ValueError(
                f"Design has {len(violations)} violation(s): {'; '.join(violations)}"
            )

        design = self.load(DesignDocument, feature_id)
        if design is None:
            raise ValueError(f"Design document not found for '{feature_id}'")

        from ..models import UserStory

        # Build a default user story from the design
        strategy_desc = design.kg_analysis.extension_strategy.value
        extension_note = (
            f" (extends {design.kg_analysis.extension_point})"
            if design.kg_analysis.extension_point
            else ""
        )

        stories = [
            UserStory(
                id=f"{feature_id}-US1",
                title=design.title,
                description=(
                    f"As a developer, I want {design.title.lower()} "
                    f"so that the system capabilities are enhanced. "
                    f"Strategy: {strategy_desc}{extension_note}."
                ),
                acceptance_criteria=[
                    "Feature integrates with existing pillar architecture",
                    "All existing tests continue to pass",
                    "Design document validation passes",
                ],
            )
        ]

        spec = Spec(
            feature_id=feature_id,
            title=design.title,
            user_stories=stories,
            metadata={
                "design_ref": f".specify/design/{feature_id}/design.md",
                "extension_strategy": strategy_desc,
            },
        )
        self.save(spec, feature_id)
        return spec

    def _render_design_md(self, design: DesignDocument) -> str:
        """Render a DesignDocument as Markdown."""
        md = [f"# Design Document: {design.title}\n"]
        md.append(f"**Feature ID**: {design.feature_id}\n")

        # KG Analysis section
        md.append("## KG Analysis\n")
        if design.kg_analysis.nearest_concepts:
            md.append("### Nearest Existing Concepts\n")
            md.append("| Concept ID | Name | Similarity | Pillar |")
            md.append("|---|---|---|---|")
            for nc in design.kg_analysis.nearest_concepts:
                md.append(
                    f"| {nc.concept_id} | {nc.name} | {nc.similarity:.0%} | {nc.pillar} |"
                )
            md.append("")

        md.append("### Extension Analysis\n")
        md.append(
            f"- **Extension Strategy**: {design.kg_analysis.extension_strategy.value}"
        )
        if design.kg_analysis.extension_point:
            md.append(f"- **Extension Point**: {design.kg_analysis.extension_point}")

        if design.kg_analysis.new_concept_proposal:
            ncp = design.kg_analysis.new_concept_proposal
            md.append("\n### New Concept Proposal\n")
            md.append(f"- **Proposed ID**: {ncp.proposed_id}")
            md.append(f"- **Target Pillar**: {ncp.target_pillar}")
            if ncp.pipeline_phase:
                md.append(f"- **Pipeline Phase**: {ncp.pipeline_phase}")
            md.append(f"- **Justification**: {ncp.justification}")

        # C4 Diagram
        if design.c4_diagram:
            md.append("\n## C4 Context Diagram\n")
            md.append("```mermaid")
            md.append(design.c4_diagram)
            md.append("```")

        # Data Flow
        if design.data_flow:
            md.append(f"\n## Data Flow\n\n{design.data_flow}")

        # Risk Assessment
        md.append("\n## Risk Assessment\n")
        ra = design.risk_assessment
        md.append(
            f"- **Backward Compatible**: {'Yes' if ra.backward_compatible else 'No'}"
        )
        if ra.blast_radius:
            md.append(f"- **Blast Radius**: {', '.join(ra.blast_radius)}")
        if ra.breaking_changes:
            md.append("- **Breaking Changes**:")
            for bc in ra.breaking_changes:
                md.append(f"  - {bc}")

        return "\n".join(md)
