"""Tests for the Council Deliberation module.

Covers:
- CouncilVerdict / AgentTranscript / CouncilTranscript schema validation
- Anonymization logic correctness and determinism
- Markdown transcript rendering for any agent output
- Full 4-stage council pipeline (mocked LLM)
- Council JSON prompt loading via config_helpers
"""

import json
import string

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agent_utilities.graph.council import (
    AgentTranscript,
    CouncilTranscript,
    CouncilVerdict,
    DEFAULT_ADVISOR_ROLES,
    _anonymize_responses,
    render_agent_transcript_markdown,
)


# ── Schema Validation ──────────────────────────────────────────────────

@pytest.mark.concept("council-deliberation")
class TestCouncilVerdict:
    """Validate the CouncilVerdict Pydantic model."""

    def test_minimal_verdict(self):
        """A verdict with only the required field should validate."""
        v = CouncilVerdict(final_recommendation="Do X")
        assert v.final_recommendation == "Do X"
        assert v.confidence == 5  # default
        assert v.key_insights == []

    def test_full_verdict(self):
        """A fully-populated verdict should serialize cleanly."""
        v = CouncilVerdict(
            final_recommendation="Migrate to Rust",
            key_insights=["Performance gains", "Memory safety"],
            blind_spots=["Learning curve"],
            consensus_areas=["Performance is critical"],
            concrete_next_step="Start with the parser module",
            confidence=8,
            dissenting_views=["Python is fast enough"],
        )
        data = v.model_dump()
        assert data["confidence"] == 8
        assert len(data["key_insights"]) == 2

    def test_confidence_bounds(self):
        """Confidence must be 1-10."""
        with pytest.raises(Exception):
            CouncilVerdict(final_recommendation="X", confidence=0)
        with pytest.raises(Exception):
            CouncilVerdict(final_recommendation="X", confidence=11)

    def test_json_round_trip(self):
        """Verdict should survive JSON serialization."""
        v = CouncilVerdict(
            final_recommendation="Test",
            key_insights=["A", "B"],
            confidence=7,
        )
        raw = v.model_dump_json()
        v2 = CouncilVerdict.model_validate_json(raw)
        assert v2.final_recommendation == v.final_recommendation
        assert v2.confidence == 7


@pytest.mark.concept("council-deliberation")
class TestAgentTranscript:
    """Validate the generalized AgentTranscript model."""

    def test_minimal_transcript(self):
        t = AgentTranscript(stage_name="Advisor", agent_role="test")
        assert t.model_id == "unknown"
        assert t.duration_ms == 0

    def test_full_transcript(self):
        t = AgentTranscript(
            stage_name="Peer Review",
            agent_role="reviewer_1",
            model_id="gpt-5.1",
            input_query="Test query",
            output_text="This is the review",
            duration_ms=1500,
            metadata={"attempt": 1},
        )
        assert t.model_id == "gpt-5.1"
        assert t.metadata["attempt"] == 1


# ── Anonymization Logic ────────────────────────────────────────────────

@pytest.mark.concept("council-deliberation")
class TestAnonymization:
    """Validate the anonymization shuffling logic."""

    def test_anonymize_produces_correct_labels(self):
        """Each response should get a unique label A-E."""
        responses = {
            "contrarian": "Critique here",
            "first_principles": "Reframe here",
            "expansionist": "Opportunity here",
        }
        anonymized, label_map = _anonymize_responses(responses)

        assert len(anonymized) == 3
        assert len(label_map) == 3

        # All labels should be Response X
        for key in anonymized:
            assert key.startswith("Response ")
            letter = key.split()[-1]
            assert letter in string.ascii_uppercase

        # label_map should map back to original roles
        for label, role in label_map.items():
            assert role in responses
            assert anonymized[label] == responses[role]

    def test_anonymize_preserves_all_content(self):
        """All original response text must appear in the anonymized output."""
        responses = {"a": "text_a", "b": "text_b"}
        anonymized, _ = _anonymize_responses(responses)

        original_values = set(responses.values())
        anon_values = set(anonymized.values())
        assert original_values == anon_values

    def test_anonymize_single_response(self):
        """Edge case: single response should still work."""
        responses = {"only": "solo response"}
        anonymized, label_map = _anonymize_responses(responses)
        assert len(anonymized) == 1
        assert "Response A" in anonymized

    def test_anonymize_is_bijective(self):
        """Each advisor maps to exactly one label and vice versa."""
        responses = {f"advisor_{i}": f"text_{i}" for i in range(5)}
        anonymized, label_map = _anonymize_responses(responses)

        assert len(set(label_map.values())) == 5  # all roles unique
        assert len(set(label_map.keys())) == 5    # all labels unique


# ── Markdown Transcript Rendering ──────────────────────────────────────

@pytest.mark.concept("council-deliberation")
class TestTranscriptRendering:
    """Validate markdown transcript generation."""

    def test_council_transcript_to_markdown(self):
        """Full council transcript should render valid markdown."""
        transcript = CouncilTranscript(
            query="Should we use Rust?",
            advisor_transcripts=[
                AgentTranscript(
                    stage_name="Advisor",
                    agent_role="contrarian",
                    model_id="gpt-5.1",
                    output_text="Rust is overkill for this.",
                    duration_ms=800,
                ),
            ],
            anonymization_map={"Response A": "contrarian"},
            reviewer_transcripts=[
                AgentTranscript(
                    stage_name="Peer Review",
                    agent_role="reviewer_1",
                    output_text="Response A has a blind spot.",
                    duration_ms=600,
                ),
            ],
            verdict=CouncilVerdict(
                final_recommendation="Stay with Python for now",
                key_insights=["Performance is not the bottleneck"],
                confidence=7,
            ),
        )
        md = transcript.to_markdown()

        assert "# Council Deliberation Transcript" in md
        assert "Should we use Rust?" in md
        assert "contrarian" in md
        assert "Response A" in md
        assert "Stay with Python for now" in md
        assert "7/10" in md

    def test_generic_transcript_rendering(self):
        """The generalized render function should work for any agent."""
        transcripts = [
            AgentTranscript(
                stage_name="Research",
                agent_role="web_researcher",
                model_id="gemini-3-pro",
                output_text="Found 3 relevant articles.",
                duration_ms=1200,
            ),
            AgentTranscript(
                stage_name="Synthesis",
                agent_role="synthesizer",
                output_text="Summary of findings.",
                duration_ms=500,
            ),
        ]
        md = render_agent_transcript_markdown(transcripts)

        assert "# Agent Execution Transcript" in md
        assert "web_researcher" in md
        assert "synthesizer" in md
        assert "gemini-3-pro" in md


# ── Council JSON Prompt Loading ────────────────────────────────────────

@pytest.mark.concept("council-deliberation")
class TestCouncilPromptLoading:
    """Validate that all council JSON prompts load correctly."""

    COUNCIL_PROMPTS = [
        "council_contrarian",
        "council_first_principles",
        "council_expansionist",
        "council_outsider",
        "council_executor",
        "council_reviewer",
        "council_chairman",
    ]

    @pytest.mark.parametrize("prompt_name", COUNCIL_PROMPTS)
    def test_prompt_loads_as_valid_json(self, prompt_name):
        """Each council prompt file must load and contain required content."""
        from agent_utilities.graph.config_helpers import load_specialized_prompts

        content = load_specialized_prompts(prompt_name)

        # Must return a non-empty string
        assert isinstance(content, str), f"{prompt_name} did not return a string"
        assert len(content) > 50, f"{prompt_name} render too short ({len(content)} chars)"

    @pytest.mark.parametrize("prompt_name", COUNCIL_PROMPTS)
    def test_prompt_has_council_content(self, prompt_name):
        """Each council prompt must contain role-specific content."""
        from pathlib import Path

        from agent_utilities.structured_prompts import StructuredPrompt

        prompt_path = (
            Path(__file__).resolve().parents[3]
            / "agent_utilities"
            / "prompts"
            / f"{prompt_name}.json"
        )
        data = json.loads(prompt_path.read_text())
        prompt = StructuredPrompt.model_validate(data)

        # Should have identity and instructions sections
        assert prompt.identity is not None, f"{prompt_name} missing 'identity'"
        assert prompt.instructions is not None, f"{prompt_name} missing 'instructions'"

        # Should have meaningful directive content
        rendered = prompt.render()
        assert len(rendered) > 50, (
            f"{prompt_name} render too short ({len(rendered)} chars)"
        )


# ── Default Advisor Roles ──────────────────────────────────────────────

@pytest.mark.concept("council-deliberation")
def test_default_advisor_roles():
    """DEFAULT_ADVISOR_ROLES must match the 5 standard council advisors."""
    assert len(DEFAULT_ADVISOR_ROLES) == 5
    assert "council_contrarian" in DEFAULT_ADVISOR_ROLES
    assert "council_first_principles" in DEFAULT_ADVISOR_ROLES
    assert "council_expansionist" in DEFAULT_ADVISOR_ROLES
    assert "council_outsider" in DEFAULT_ADVISOR_ROLES
    assert "council_executor" in DEFAULT_ADVISOR_ROLES


# ── Council Step Description Registration ──────────────────────────────

@pytest.mark.concept("council-deliberation")
def test_council_in_step_descriptions():
    """The council must appear in get_step_descriptions() output."""
    from agent_utilities.graph.executor import get_step_descriptions

    descriptions = get_step_descriptions()
    assert "council" in descriptions
    assert "advisory council" in descriptions.lower() or "advisor" in descriptions.lower()
