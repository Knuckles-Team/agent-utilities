#!/usr/bin/python
"""Graph Models Module.

This module defines the structured Pydantic models used for input and output
across various graph nodes, including routing decisions, planning
metadata, and quality validation results.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..models import GraphPlan


# Domain classification models used by the router step
class DomainChoice(BaseModel):
    """Structured output from the router step for domain classification.

    Attributes:
        domain: The specialized domain tag to route the request to.
        confidence: Normalized confidence level of the classification.
        reasoning: Rationale behind the selected routing path.

    """

    domain: str = Field(description="The domain tag to route to")
    confidence: float = Field(ge=0, le=1, description="Routing confidence 0-1")
    reasoning: str = Field(description="Brief reasoning for the classification")


class MultiDomainChoice(BaseModel):
    """Structured output for advanced dynamic graph planning.

    Attributes:
        plan: The generated sequential or parallel execution strategy.
        reasoning: Rationale behind the complex plan architecture.
        is_resumed: Flag indicating if this plan resumes a previous state.

    """

    plan: GraphPlan = Field(description="The sequential/parallel execution plan")
    reasoning: str = Field(description="Brief reasoning for the plan architecture")
    is_resumed: bool = Field(False, description="Whether this is a resumed operation")


class ValidationResult(BaseModel):
    """Structured output for the verifier step to assess result quality.

    Attributes:
        is_valid: Boolean flag indicating if the results pass the quality gate.
        feedback: Constructive feedback if validation fails, detailing gaps.
        score: Numerical quality score (0.0 to 1.0) for internal tracking.

    """

    is_valid: bool = Field(
        description="True if the result is high quality and accurate"
    )
    feedback: str | None = Field(
        None, description="Detailed feedback if invalid, explaining what to improve"
    )
    score: float = Field(ge=0, le=1, description="Quality score from 0 to 1")
