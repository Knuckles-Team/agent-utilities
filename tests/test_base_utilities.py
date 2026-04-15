import pytest
import os
from unittest.mock import patch
from agent_utilities.base_utilities import (
    to_float, to_boolean, to_integer, to_list, to_dict,
    expand_env_vars, is_loopback_url
)

def test_type_conversions():
    """Test standard type conversion functions."""
    # float
    assert to_float("1.5") == 1.5
    assert to_float("") == 0.0
    assert to_float(None) == 0.0
    assert to_float("invalid") == 0.0
    assert to_float(2.0) == 2.0

    # boolean
    assert to_boolean("true") is True
    assert to_boolean("yes") is True
    assert to_boolean("1") is True
    assert to_boolean("false") is False
    assert to_boolean("") is False
    assert to_boolean(True) is True

    # integer
    assert to_integer("123") == 123
    assert to_integer("abc") == 0
    assert to_integer(None) == 0
    assert to_integer(42) == 42

    # list
    assert to_list("[1, 2, 3]") == [1, 2, 3]
    assert to_list("a,b,c") == ["a", "b", "c"]
    assert to_list(None) == []
    assert to_list([1, 2]) == [1, 2]

    # dict
    assert to_dict('{"a": 1}') == {"a": 1}
    assert to_dict(None) == {}
    with pytest.raises(ValueError):
        to_dict("invalid_json")

def test_expand_env_vars():
    """Test environment variable expansion logic."""
    with patch.dict(os.environ, {"TEST_VAR": "value", "EMPTY_VAR": ""}):
        assert expand_env_vars("prefix_${TEST_VAR}_suffix") == "prefix_value_suffix"
        assert expand_env_vars("${MISSING_VAR:-default}") == "default"
        assert expand_env_vars("${EMPTY_VAR:-default}") == ""
        assert expand_env_vars("${MISSING_VAR}") == ""

def test_expand_env_vars_validation_mode():
    """Test VALIDATION_MODE dummy value generation."""
    with patch.dict(os.environ, {"VALIDATION_MODE": "True"}):
        # Secret candidate
        assert expand_env_vars("${MY_API_KEY}") == "dummy_my_api_key"
        # Non-secret candidate
        assert expand_env_vars("${SOME_SETTING}") == "validation_some_setting"

def test_is_loopback_url():
    """Test loopback URL detection."""
    assert is_loopback_url("http://localhost:8000", current_port=8000) is True
    assert is_loopback_url("http://127.0.0.1:8000", current_port=8000) is True
    assert is_loopback_url("http://localhost:8001", current_port=8000) is False
    assert is_loopback_url("http://google.com:80", current_port=80) is False
    assert is_loopback_url("http://custom-host:8000", current_host="custom-host", current_port=8000) is True
