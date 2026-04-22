import pytest
from unittest.mock import AsyncMock, MagicMock
from agent_utilities.agent_chat.parser import parse_codemap_mentions
from agent_utilities.models.codemap import CodemapArtifact

@pytest.mark.asyncio
async def test_parse_codemap_mentions():
    # Mock KG engine
    kg = MagicMock()
    kg.get_codemap_by_id = AsyncMock()
    kg.get_codemap_by_slug = AsyncMock()

    # Mock artifact
    artifact = CodemapArtifact(
        id="auth-flow-uuid",
        prompt="Trace the authentication flow",
        mode="smart",
        hierarchy=[]
    )

    # 1. Test valid mention by ID
    kg.get_codemap_by_id.return_value = artifact
    prompt = "Look at this: @codemap{auth-flow-uuid}"
    clean_prompt, mentions = await parse_codemap_mentions(prompt, kg)

    assert "auth-flow-uuid" in mentions
    assert "[Codemap Reference: Trace the authentication flow]" in clean_prompt
    kg.get_codemap_by_id.assert_called_with("auth-flow-uuid")

    # 2. Test mention not found
    kg.get_codemap_by_id.return_value = None
    kg.get_codemap_by_slug.return_value = None
    prompt = "Look at this: @codemap{missing}"
    clean_prompt, mentions = await parse_codemap_mentions(prompt, kg)

    assert len(mentions) == 0
    assert "[Codemap Reference Not Found: missing]" in clean_prompt

    # 3. Test multiple mentions
    kg.get_codemap_by_id.return_value = artifact
    prompt = "@codemap{id1} and @codemap{id2}"
    clean_prompt, mentions = await parse_codemap_mentions(prompt, kg)
    assert len(mentions) == 2
    assert "id1" in mentions
    assert "id2" in mentions
