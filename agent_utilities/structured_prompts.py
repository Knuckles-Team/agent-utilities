from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class NestedStructure(BaseModel):
    """Reusable nested blueprint for structured content (hook/body/cta, deliverables, etc.)."""

    hook: str | None = Field(None, description="e.g., curiosity-driven, under 10 words")
    body: str | None = Field(None, description="e.g., 3 insights with examples")
    cta: str | None = Field(None, description="e.g., question that sparks replies")

    model_config = ConfigDict(extra="allow")


class StructuredPrompt(BaseModel):
    """Core standardized model for JSON prompting.
    Covers tweets, threads, cold emails, video gen, code, brand strategy, and more.
    """

    task: str = Field(
        ...,
        description="The primary action — e.g., 'write a tweet', 'generate video', 'create consulting doc'",
    )
    topic: str | None = Field(None, description="The main subject of the task")
    platform: str | None = Field(
        None, description="Target platform (twitter, linkedin, email, etc.)"
    )
    tone: str | None = Field(None, description="Tone of the output")
    style: str | None = Field(None, description="Specific style requirements")
    length: str | None = Field(None, description="Length constraints")
    audience: str | None = Field(None, description="Target audience")
    goal: str | None = Field(None, description="What the task aims to achieve")
    structure: NestedStructure | dict[str, Any] | None = Field(
        None, description="Nested structure for the content"
    )
    input: str | None = Field(None, description="Input text for rewrite/improve tasks")
    constraints: list[str] | None = Field(
        None, description="List of specific constraints"
    )
    deliverables: list[str] | None = Field(
        None, description="List of expected deliverables"
    )
    output_format: str | None = Field(None, description="Desired output format")
    visual_style: str | None = Field(None, description="Visual style (e.g., for video)")
    language: str | None = Field(None, description="Programming language (for code)")
    tools: list[str] | None = Field(None, description="List of tool/skill names")

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "examples": [
                {
                    "task": "write a tweet",
                    "topic": "dopamine detox",
                    "style": "viral",
                    "length": "under 280 characters",
                    "tone": "punchy and contrarian",
                }
            ]
        },
    )

    def render(self) -> str:
        """Turn the model into the exact JSON prompt the LLM sees."""
        return self.model_dump_json(indent=2, exclude_unset=True, exclude_none=True)

    @classmethod
    def from_kg(cls, kg_data: dict[str, Any]) -> StructuredPrompt:
        """Factory: hydrate from Knowledge Graph query result.

        This method map KG node properties (semantic + causal + entity views)
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
