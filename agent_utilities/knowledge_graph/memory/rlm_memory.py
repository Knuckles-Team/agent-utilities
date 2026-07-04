"""RLM-based Semantic Memory Extraction.

CONCEPT:AU-KG.memory.rlm-memory-extraction — RLM Memory Extraction Signature

Defines the Pydantic-based RLM signatures for evaluating
and extracting high-confidence temporal facts from unstructured context.
"""

from pydantic import BaseModel, Field

from agent_utilities.rlm.predict_rlm import InputField, OutputField


class ExtractedFact(BaseModel):
    """A discrete extracted fact with temporal validity."""

    fact: str = Field(description="The standalone extracted fact.")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0.")
    valid_until: int | None = Field(
        default=None, description="Unix timestamp for expiration. None if permanent."
    )


class MemoryExtractionSignature(BaseModel):
    """Analyze a user interaction and extract new, evolving facts about the user.

    Extract high-confidence facts that should be stored in the temporal knowledge graph.
    Only extract facts that are likely to persist, not transient session state.
    """

    session_context: str = InputField(
        description="The raw conversational context or text to extract facts from."
    )
    user_id: str = InputField(description="The target user ID to extract facts for.")

    reasoning: str = OutputField(
        description="Step-by-step reasoning evaluating the facts and their temporal properties."
    )
    extracted_facts: list[ExtractedFact] = OutputField(
        description="List of extracted facts with confidence and validity metadata."
    )
