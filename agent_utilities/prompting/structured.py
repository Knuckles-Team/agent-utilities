"""Structured Prompt models for JSON-as-Code prompting.

CONCEPT:ORCH-1.0 — Structured Prompting
    Provides Pydantic models for defining agent system prompts as structured
    JSON documents with decomposed ``metadata``, ``identity``, and
    ``instructions`` sections for machine-parseable prompt engineering.
"""

from __future__ import annotations

import json
import logging
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
# Engineering Rules Engine (CONCEPT:KG-2.2)
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

    CONCEPT:KG-2.2 — Engineering Rules Engine

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
    # Engineering Rules Engine (CONCEPT:KG-2.2)
    engineering_rules: EngineeringRulesSection | None = Field(
        default=None,
        description="Task-scoped engineering rules from agent-rules-books",
    )
    rules: list[str] | dict[str, list[str]] | None = Field(
        default=None,
        description="KG-ingestible engineering rules. Supports simple list or categorized dict.",
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
