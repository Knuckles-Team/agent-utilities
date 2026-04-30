from agent_utilities.core.workspace import md_table_escape, smart_truncate


def test_md_table_escape():
    assert md_table_escape("a|b\nc") == "a\\|b<br/>c"
    assert md_table_escape("a|b\\nc") == "a\\|b<br/>c"


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
