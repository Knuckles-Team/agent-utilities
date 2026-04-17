import pytest
from agent_utilities.workspace import (
    parse_identity,
    serialize_identity,
    parse_user_info,
    serialize_user_info,
    parse_a2a_registry,
    serialize_a2a_registry,
    md_table_escape,
    smart_truncate
)
from agent_utilities.models import IdentityModel, UserModel, A2ARegistryModel, A2APeerModel

def test_parse_identity():
    content = """# IDENTITY.md
 * **Name:** TestBot
 * **Role:** Tester
 * **Emoji:** bot
 * **Vibe:** Chill

 ### System Prompt
 This is a test prompt.
"""
    model = parse_identity(content)
    assert model.name == "TestBot"
    assert model.role == "Tester"
    assert model.emoji == "bot"
    assert model.vibe == "Chill"
    assert "This is a test prompt." in model.system_prompt

def test_serialize_identity():
    model = IdentityModel(name="TestBot", role="Tester", emoji="bot", vibe="Chill", system_prompt="Do test.")
    res = serialize_identity(model)
    assert "**Name:** TestBot" in res
    assert "Do test." in res

def test_parse_user_info():
    content = """# USER.md
* **Name:** Human
* **Emoji:** user
"""
    model = parse_user_info(content)
    assert model.name == "Human"
    assert model.emoji == "user"

def test_serialize_user_info():
    model = UserModel(name="Human", emoji="user")
    res = serialize_user_info(model)
    assert "**Name:** Human" in res

def test_md_table_escape():
    assert md_table_escape("a|b\nc") == "a\\|b<br/>c"
    assert md_table_escape("a|b\\nc") == "a\\|b<br/>c"

def test_parse_a2a_registry():
    content = """
| Name | Endpoint URL | Description | Capabilities | Auth | Notes / Last Connected |
|------|--------------|-------------|--------------|------|------------------------|
| Peer1 | http://p1 | Desc1 | Cap1 | None | Notes1 |
"""
    model = parse_a2a_registry(content)
    assert model.peers[0].name == "Peer1"
    assert model.peers[0].url == "http://p1"

def test_serialize_a2a_registry():
    peer = A2APeerModel(name="Peer1", url="http://p1", description="desc", capabilities="cap", notes="Notes1")
    model = A2ARegistryModel(peers=[peer])
    res = serialize_a2a_registry(model)
    assert "Peer1" in res
    assert "http://p1" in res


def test_smart_truncate():
    # No truncation needed
    assert smart_truncate("Hello", 10) == "Hello"
    # Basic truncation at space
    assert smart_truncate("Hello World", 8) == "Hello..."
    # Truncation at word boundary (already at space)
    assert smart_truncate("Hello World", 5) == "Hello..."
    # Graceful handling of no spaces (cuts at limit)
    assert smart_truncate("HelloWorld", 5) == "Hello..."
    # Empty/None handling
    assert smart_truncate("", 10) == "-"
    assert smart_truncate(None, 10) == "-"
