"""FalkorDB rejects control chars in query-parameter string values (KG-2.74)."""

from agent_utilities.knowledge_graph.backends.contrib.falkordb_backend import (
    _clean_param_value,
)


def test_strips_control_chars_keeps_tab_newline():
    s = "1 \x01 2\tC\nD\rE \x1f end"
    assert _clean_param_value(s) == "1  2\tC\nD\rE  end"


def test_recurses_into_list_and_dict():
    assert _clean_param_value(["a\x01b", "c"]) == ["ab", "c"]
    assert _clean_param_value({"k": "x\x07y"}) == {"k": "xy"}


def test_non_strings_pass_through():
    assert _clean_param_value(42) == 42
    assert _clean_param_value(3.14) == 3.14
    assert _clean_param_value(True) is True
    assert _clean_param_value(None) is None


def test_clean_string_unchanged():
    assert _clean_param_value("normal text 123") == "normal text 123"
