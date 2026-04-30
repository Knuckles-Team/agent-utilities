"""Test that prompt JSON files load correctly via load_specialized_prompts.

CONCEPT:AU-006 — Structured Prompting
"""

import json
from pathlib import Path

from agent_utilities.graph.config_helpers import load_specialized_prompts
from agent_utilities.prompting.structured import StructuredPrompt

PROMPTS_DIR = Path(__file__).resolve().parents[3] / "agent_utilities" / "prompts"


def test_load_planner_prompt():
    """Test that the planner prompt loads and renders correctly."""
    content = load_specialized_prompts("planner")
    assert isinstance(content, str)
    assert len(content) > 100
    assert "planner" in content.lower() or "plan" in content.lower()


def test_load_base_agent():
    """Test that the large base_agent prompt loads correctly."""
    content = load_specialized_prompts("base_agent")
    assert isinstance(content, str)
    assert len(content) > 1000, f"base_agent render too short: {len(content)} chars"


def test_all_prompts_validate():
    """All prompt JSON files should parse via StructuredPrompt."""
    ok = 0
    for f in sorted(PROMPTS_DIR.glob("*.json")):
        with open(f) as fp:
            data = json.load(fp)
        p = StructuredPrompt.model_validate(data)
        rendered = p.render()
        assert len(rendered) > 10, f"{f.name}: render too short"
        ok += 1
    assert ok >= 40, f"Expected at least 40 prompt files, got {ok}"


def test_prompts_have_structured_sections():
    """Prompts should have metadata and/or identity sections."""
    structured_count = 0
    for f in sorted(PROMPTS_DIR.glob("*.json")):
        with open(f) as fp:
            data = json.load(fp)
        if data.get("metadata") or data.get("identity") or data.get("instructions"):
            structured_count += 1
    assert structured_count >= 40, f"Expected at least 40 structured prompts, got {structured_count}"


def test_render_preserves_content():
    """Render should preserve the full prompt content."""
    with open(PROMPTS_DIR / "python_programmer.json") as fp:
        data = json.load(fp)
    p = StructuredPrompt.model_validate(data)
    rendered = p.render()
    assert len(rendered) > 3000, f"Python programmer render too short: {len(rendered)} chars"
    assert "Python" in rendered or "python" in rendered
