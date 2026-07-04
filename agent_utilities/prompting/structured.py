from __future__ import annotations

"""Structured Prompt models for JSON-as-Code prompting.

CONCEPT:AU-ORCH.optimization.structured-prompting — Structured Prompting
    Provides Pydantic models for defining agent system prompts as structured
    JSON documents with decomposed ``metadata``, ``identity``, and
    ``instructions`` sections for machine-parseable prompt engineering.
"""


import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Nested models
# ---------------------------------------------------------------------------


class PromptMetadata(BaseModel):
    """Descriptive metadata about the prompt (topic, tone, style, audience)."""

    description: str | None = Field(
        None, description="Human-readable summary of the prompt's purpose"
    )
    topic: str | None = Field(default=None, description="Primary subject domain")
    tone: str | None = Field(
        None, description="Communication tone (e.g., 'technical', 'conversational')"
    )
    style: str | None = Field(
        None, description="Response style (e.g., 'professional assistant')"
    )
    audience: str | None = Field(
        default=None, description="Target audience for the output"
    )

    model_config = ConfigDict(extra="allow")


class PromptIdentity(BaseModel):
    """Agent identity and persona definition."""

    role: str | None = Field(
        None, description="Primary role title (e.g., 'Python Systems Wizard')"
    )
    goal: str | None = Field(default=None, description="What the agent aims to achieve")
    personality: list[str] | None = Field(
        default=None, description="Personality traits for the agent persona"
    )

    model_config = ConfigDict(extra="allow")


class PromptInstructions(BaseModel):
    """Structured behavioral instructions for the agent.

    Decomposes the monolithic prompt body into typed sections that can be
    independently validated, merged, and rendered.
    """

    core_directive: str | None = Field(
        default=None, description="Primary behavioral instruction / full prompt body"
    )
    responsibilities: list[str] | None = Field(
        default=None, description="Key responsibility areas"
    )
    capabilities: dict[str, list[str]] | None = Field(
        default=None,
        description="Categorized capability lists (e.g., {'testing': ['pytest', 'fixtures']})",
    )
    workflow: list[str] | None = Field(
        default=None, description="Ordered workflow steps"
    )
    quality_checklist: list[str] | None = Field(
        default=None, description="Quality verification items"
    )
    methodology: str | None = Field(
        default=None, description="Detailed methodology description"
    )
    output_format: str | None = Field(
        default=None, description="Expected output structure"
    )

    model_config = ConfigDict(extra="allow")

    def render_section(self) -> str:
        """Render instructions into a formatted string."""
        parts: list[str] = []

        if self.core_directive:
            parts.append(self.core_directive)

        if self.responsibilities and not self.core_directive:
            resp_items = "\n".join(f"- {r}" for r in self.responsibilities)
            parts.append(f"### KEY RESPONSIBILITIES\n{resp_items}")

        if self.capabilities and not self.core_directive:
            cap_parts: list[str] = []
            for category, cap_items in self.capabilities.items():
                title = category.replace("_", " ").title()
                bullet_list = "\n".join(f"- {item}" for item in cap_items)
                cap_parts.append(f"#### {title}\n{bullet_list}")
            parts.append("### CAPABILITIES\n" + "\n\n".join(cap_parts))

        if self.workflow and not self.core_directive:
            flow_items = "\n".join(
                f"{i + 1}. {step}" for i, step in enumerate(self.workflow)
            )
            parts.append(f"### WORKFLOW\n{flow_items}")

        if self.quality_checklist and not self.core_directive:
            chk_items = "\n".join(f"- [ ] {item}" for item in self.quality_checklist)
            parts.append(f"### QUALITY CHECKLIST\n{chk_items}")

        if self.methodology and not self.core_directive:
            parts.append(f"### METHODOLOGY\n{self.methodology}")

        if self.output_format and not self.core_directive:
            parts.append(f"### OUTPUT FORMAT\n{self.output_format}")

        # Render any extra fields
        if not self.core_directive:
            extra = self.model_extra or {}
            for key, value in extra.items():
                title = key.replace("_", " ").title()
                if isinstance(value, list):
                    ext_items = "\n".join(f"- {v}" for v in value)
                    parts.append(f"### {title}\n{ext_items}")
                elif isinstance(value, str):
                    parts.append(f"### {title}\n{value}")

        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Legacy nested structure (preserved for backward compatibility with
# non-agent prompt use cases like content generation)
# ---------------------------------------------------------------------------


class NestedStructure(BaseModel):
    """Reusable nested blueprint for structured content (hook/body/cta, etc.)."""

    hook: str | None = Field(
        default=None, description="e.g., curiosity-driven, under 10 words"
    )
    body: str | None = Field(default=None, description="e.g., 3 insights with examples")
    cta: str | None = Field(
        default=None, description="e.g., question that sparks replies"
    )

    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# Engineering Rules Engine (CONCEPT:AU-KG.ingest.engineering-rules)
# ---------------------------------------------------------------------------


class TriggerRule(BaseModel):
    """A conditional rule that activates based on task context."""

    when: str = Field(
        description="Trigger condition (e.g., 'adding external dependency')"
    )
    apply: str = Field(
        description="Rule set or rule ID to apply (e.g., 'release-it.mini#trigger-rules')"
    )

    model_config = ConfigDict(extra="allow")


class EngineeringRulesSection(BaseModel):
    """Task-scoped engineering rules from agent-rules-books.

    CONCEPT:AU-KG.ingest.engineering-rules — Engineering Rules Engine

    Defines which engineering rules are injected into the agent's prompt
    based on task context. Supports three loading patterns:
      - ``always_on``: Rule IDs always included (nano tier recommended)
      - ``on_demand``: task_type → rule IDs mapping
      - ``trigger_rules``: Conditional rules activated by context
    """

    always_on: list[str] = Field(
        default_factory=list,
        description="Rule set IDs always included (e.g., 'clean-code.nano')",
    )
    on_demand: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Task type → rule set IDs (e.g., {'refactoring': ['refactoring.mini']})",
    )
    trigger_rules: list[TriggerRule] = Field(
        default_factory=list,
        description="Conditional rules activated by task context",
    )
    context_budget: str = Field(
        default="mini",
        description="Maximum tier to use: nano, mini, or full",
    )

    model_config = ConfigDict(extra="allow")

    def render_section(self) -> str:
        """Render the engineering rules section as markdown."""
        parts: list[str] = []

        if self.always_on:
            items = "\n".join(f"- {r}" for r in self.always_on)
            parts.append(f"### ALWAYS-ON RULES\n{items}")

        if self.on_demand:
            od_parts: list[str] = []
            for task_type, rule_ids in self.on_demand.items():
                title = task_type.replace("_", " ").title()
                bullet_list = "\n".join(f"  - {r}" for r in rule_ids)
                od_parts.append(f"- **{title}**:\n{bullet_list}")
            parts.append("### ON-DEMAND RULES\n" + "\n".join(od_parts))

        if self.trigger_rules:
            tr_items = "\n".join(
                f"- ⚡ When {tr.when} → apply `{tr.apply}`" for tr in self.trigger_rules
            )
            parts.append(f"### TRIGGER RULES\n{tr_items}")

        return "\n\n".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class StructuredPrompt(BaseModel):
    """Core standardized model for JSON prompting.

    Defines agent system prompts as structured JSON with decomposed sections::

        {
            "task": "python_programmer",
            "type": "prompt",
            "metadata": {"topic": "...", "tone": "...", "style": "..."},
            "identity": {"role": "...", "goal": "..."},
            "instructions": {"core_directive": "...", "capabilities": {...}},
            "tools": [...]
        }
    """

    # Required
    task: str = Field(
        ...,
        description="The primary action — e.g., 'python_programmer', 'write a tweet'",
    )

    # Structured sections
    metadata: PromptMetadata | None = Field(
        default=None, description="Descriptive metadata"
    )
    identity: PromptIdentity | None = Field(
        default=None, description="Agent persona definition"
    )
    instructions: PromptInstructions | None = Field(
        default=None, description="Behavioral instructions"
    )

    # Common fields
    type: str | None = Field(default=None, description="Prompt type (e.g., 'prompt')")
    topic: str | None = Field(default=None, description="Main subject of the task")
    platform: str | None = Field(default=None, description="Target platform")
    tone: str | None = Field(default=None, description="Tone of the output")
    style: str | None = Field(default=None, description="Specific style requirements")
    length: str | None = Field(default=None, description="Length constraints")
    audience: str | None = Field(default=None, description="Target audience")
    goal: str | None = Field(default=None, description="What the task aims to achieve")
    description: str | None = Field(
        default=None, description="Human-readable description"
    )
    structure: NestedStructure | dict[str, Any] | None = Field(
        default=None, description="Nested structure for content generation"
    )
    input: str | None = Field(
        default=None, description="Legacy: monolithic system prompt text"
    )
    constraints: list[str] | None = Field(
        default=None, description="Specific constraints"
    )
    deliverables: list[str] | None = Field(
        default=None, description="Expected deliverables"
    )
    output_format: str | None = Field(default=None, description="Desired output format")
    visual_style: str | None = Field(
        default=None, description="Visual style (e.g., for video)"
    )
    language: str | None = Field(
        default=None, description="Programming language (for code)"
    )
    tools: list[str] | None = Field(
        default=None, description="List of tool/skill names"
    )
    # Engineering Rules Engine (CONCEPT:AU-KG.ingest.engineering-rules)
    engineering_rules: EngineeringRulesSection | None = Field(
        default=None,
        description="Task-scoped engineering rules from agent-rules-books",
    )
    rules: list[str] | dict[str, list[str]] | None = Field(
        default=None,
        description="KG-ingestible engineering rules. Supports simple list or categorized dict.",
    )

    # Canonical contract fields (CONCEPT:AU-ORCH.routing.resolve-body-single-canonical) — provenance, versioning,
    # skill/tool wiring, and base-prompt composition. ``extra="allow"`` already
    # accepted these informally; promoting them to typed fields makes them
    # validated, documented, and emitted into the generated JSON Schema.
    schema_version: str = Field(
        default="1.0",
        description="Canonical prompt-schema version this blueprint conforms to.",
    )
    prompt_version: str | None = Field(
        default=None,
        description="Semver of THIS prompt's content (a bumpversion sync point).",
    )
    source: str | None = Field(
        default=None,
        description=(
            "Provenance / KG namespace for this prompt, e.g. 'gitlab-api' or "
            "'agent-utilities:base'. Used to namespace the PromptNode id so "
            "fleet-contributed prompts never collide."
        ),
    )
    skills: list[str] | None = Field(
        default=None,
        description=(
            "Skill slugs this prompt expects installed (resolved by "
            "check_prompt_refs). Companion to ``tools``."
        ),
    )
    extends: str | None = Field(
        default=None,
        description=(
            "Base prompt to compose onto, e.g. 'agent-utilities:base' or a "
            "'@base_agent.json' workspace reference. Render-time composition."
        ),
    )
    compose: str = Field(
        default="append",
        description="How to merge this body onto ``extends``: append | prepend | replace.",
    )

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        json_schema_extra={
            "examples": [
                {
                    "task": "python_programmer",
                    "type": "prompt",
                    "metadata": {
                        "description": "Python specialist agent",
                        "topic": "Python Development",
                        "tone": "technical and precise",
                        "style": "professional assistant",
                    },
                    "identity": {
                        "role": "Python Systems Wizard",
                        "goal": "Craft production-ready Python solutions",
                    },
                    "instructions": {
                        "core_directive": "Write idiomatic, high-performance Python code.",
                        "capabilities": {
                            "modern_python": ["decorators", "dataclasses", "protocols"],
                            "testing": ["pytest", "fixtures", "coverage"],
                        },
                    },
                    "tools": ["agent-builder", "tdd-methodology"],
                }
            ]
        },
    )

    def render(self) -> str:
        """Render the structured prompt into the system prompt text the LLM sees.

        Priority:
        1. If ``instructions`` section exists, render from structured data
        2. If ``input`` field exists (legacy), return it directly
        3. Fall back to JSON serialization
        """
        if self.instructions:
            parts: list[str] = [self.instructions.render_section()]
            if self.engineering_rules:
                er_text = self.engineering_rules.render_section()
                if er_text:
                    parts.append(er_text)
            if self.rules:
                parts.append(self._render_rules())
            return "\n\n".join(p for p in parts if p)
        if self.input:
            return self.input
        return self.model_dump_json(indent=2, exclude_unset=True, exclude_none=True)

    def version_hash(self) -> str:
        """Content hash of the rendered prompt (CONCEPT:AU-AHE.evaluation.generationnode-records) — addresses a version."""
        import hashlib

        return hashlib.sha256(self.render().encode("utf-8")).hexdigest()[:16]

    def version(
        self,
        prompt_id: str | None = None,
        *,
        backend: Any = None,
        parent_hash: str | None = None,
    ) -> Any:
        """Build (and optionally persist) a ``PromptVersionNode`` for this prompt.

        A ``GenerationNode`` records the returned ``version_hash`` as its
        ``prompt_version_id``, making 'which prompt version regressed which dimension'
        a graph query — the prompt→experiment→regression half of the closed loop.
        """
        from agent_utilities.models.knowledge_graph import PromptVersionNode

        pid = prompt_id or self.task
        vhash = self.version_hash()
        node = PromptVersionNode(
            id=f"prompt_version:{pid}:{vhash}",
            name=f"{pid}@{vhash}",
            prompt_id=pid,
            version_hash=vhash,
            content=self.render()[:8000],
            parent_hash=parent_hash,
        )
        if backend is not None and hasattr(backend, "add_node"):
            try:
                props = node.model_dump()
                props.pop("id", None)
                props["type"] = str(props.get("type", ""))
                backend.add_node(node.id, **props)
            except Exception:  # pragma: no cover - best-effort
                pass
        return node

    def _render_rules(self) -> str:
        """Render rules in both simple list and categorized dict formats."""
        if isinstance(self.rules, list):
            rule_items = "\n".join(f"- {r}" for r in self.rules)
            return f"### ENGINEERING RULES\n{rule_items}"
        elif isinstance(self.rules, dict):
            sections: list[str] = ["### ENGINEERING RULES\n"]
            for category, items in self.rules.items():
                heading = category.replace("_", " ").title()
                sections.append(f"**{heading}**")
                for item in items:
                    sections.append(f"- {item}")
                sections.append("")
            return "\n".join(sections)
        return ""

    @classmethod
    def from_kg(cls, kg_data: dict[str, Any]) -> StructuredPrompt:
        """Factory: hydrate from Knowledge Graph query result.

        This method maps KG node properties (semantic + causal + entity views)
        to StructuredPrompt fields.
        """
        # If the KG data is already a JSON string of a prompt, parse it
        if "json_blueprint" in kg_data and kg_data["json_blueprint"]:
            try:
                blueprint = json.loads(kg_data["json_blueprint"])
                return cls.model_validate(blueprint)
            except Exception as e:
                logger.warning(f"Failed to parse json_blueprint from KG: {e}")

        # Otherwise, try to map generic properties
        mapped_data = {
            "task": kg_data.get("task") or kg_data.get("name", "generic task"),
            "topic": kg_data.get("topic"),
            "tone": kg_data.get("tone"),
            "goal": kg_data.get("goal"),
            "audience": kg_data.get("audience"),
            "input": kg_data.get("input") or kg_data.get("description"),
        }

        # Handle extra fields
        for k, v in kg_data.items():
            if k not in mapped_data and k not in ["id", "type"]:
                mapped_data[k] = v

        return cls.model_validate(mapped_data)

    @classmethod
    def load(cls, file_path: Path | str) -> StructuredPrompt:
        """Load a StructuredPrompt from a local JSON blueprint file."""
        from pathlib import Path

        path = Path(file_path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)

    def save(self, file_path: Path | str) -> None:
        """Save this StructuredPrompt cleanly to a local JSON blueprint file.

        Preserves custom properties via extra="allow" and formats with double spacing indentation.
        """
        from pathlib import Path

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump(exclude_none=True, exclude_unset=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        logger.info("Saved structured prompt blueprint to %s", path)


# ---------------------------------------------------------------------------
# Canonical body resolution + validation (CONCEPT:AU-ORCH.routing.resolve-body-single-canonical)
# ---------------------------------------------------------------------------

# Legacy flat body keys, retained transitionally for the one-time migration of
# the existing fleet prompts. The canonical body location is
# ``instructions.core_directive``; these are read but never written by new code.
_LEGACY_BODY_KEYS = ("content", "input")

# Canonical schema version emitted by new prompts. Mirrors
# ``StructuredPrompt.schema_version`` default.
CANONICAL_SCHEMA_VERSION = "1.0"


def resolve_body(data: dict[str, Any]) -> str:
    """Single source of truth for a prompt blueprint's body text.

    CONCEPT:AU-ORCH.routing.resolve-body-single-canonical. Replaces three divergent ad-hoc readers
    (``builder._extract_prompt_content``, ``builder.extract_agent_metadata``,
    ``registry_builder._resolve_fields``) that each read a different subset and
    silently missed ``instructions.core_directive`` — the bug that left
    StructuredPrompt-shaped files (incl. the packaged ``main_agent.json``) with
    an empty body on the workspace path.

    Precedence (canonical first, legacy transitional fallbacks after):
      1. ``instructions.core_directive``           (CANONICAL)
      2. ``content``                               (legacy flat key)
      3. ``input``                                 (legacy flat key)
      4. rendered structured sections via the model (decomposed-only prompts)
      5. ``""`` (no body)
    """
    if not isinstance(data, dict):
        return ""
    instructions = data.get("instructions")
    if isinstance(instructions, dict):
        core = instructions.get("core_directive")
        if isinstance(core, str) and core.strip():
            return core
    for key in _LEGACY_BODY_KEYS:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value
    # Decomposed-only blueprint (structured instruction sections, no flat body):
    # render through the model so responsibilities/capabilities/workflow surface.
    if isinstance(instructions, dict) and instructions:
        try:
            return StructuredPrompt.model_validate(data).render()
        except Exception:  # pragma: no cover - defensive
            return ""
    return ""


def validate_canonical(data: dict[str, Any], *, strict: bool = False) -> list[str]:
    """Validate a raw prompt blueprint against the canonical contract.

    Returns a list of human-readable violation strings (empty == conformant).
    The ONE validator shared by ``prompt-builder/validate_prompt.py``,
    ``scripts/check_prompt_schema.py``, and per-package ``test_prompt_parity``,
    so the canonical rules can never drift between authoring and CI.

    Beyond Pydantic model validation it enforces the rules the bare model does
    not: ``type == "prompt"``, a non-empty renderable body, and — in ``strict``
    mode — no lingering legacy ``content``/``input`` keys.
    """
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["prompt blueprint is not a JSON object"]

    try:
        StructuredPrompt.model_validate(data)
    except Exception as exc:
        errors.append(f"schema: {exc}")
        return errors  # downstream checks assume a structurally valid model

    if data.get("type") != "prompt":
        errors.append("'type' must be 'prompt'")

    if not str(data.get("schema_version") or "").strip():
        errors.append("'schema_version' is required (e.g. '1.0')")

    if not resolve_body(data).strip():
        errors.append(
            "empty body: set 'instructions.core_directive' or a structured "
            "instruction section"
        )

    if strict:
        legacy = [k for k in _LEGACY_BODY_KEYS if k in data]
        if legacy:
            errors.append(
                "legacy body key(s) present "
                f"({', '.join(legacy)}); move body to 'instructions.core_directive'"
            )

    return errors
