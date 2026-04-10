#!/usr/bin/python

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field

from ..models import GraphPlan


# Domain classification models used by the router step
class DomainChoice(BaseModel):
    """Structured output from the router LLM."""

    domain: str = Field(description="The domain tag to route to")
    confidence: float = Field(ge=0, le=1, description="Routing confidence 0-1")
    reasoning: str = Field(description="Brief reasoning for the classification")


class MultiDomainChoice(BaseModel):
    """Structured output for dynamic graph planning."""

    plan: GraphPlan = Field(description="The sequential/parallel execution plan")
    reasoning: str = Field(description="Brief reasoning for the plan architecture")
    is_resumed: bool = Field(False, description="Whether this is a resumed operation")


class ValidationResult(BaseModel):
    """Structured output for result validation."""

    is_valid: bool = Field(
        description="True if the result is high quality and accurate"
    )
    feedback: Optional[str] = Field(
        None, description="Detailed feedback if invalid, explaining what to improve"
    )
    score: float = Field(ge=0, le=1, description="Quality score from 0 to 1")
