import pytest
import os
import json
from agent_utilities import base_utilities

def test_to_float():
    assert base_utilities.to_float("1.5") == 1.5
    assert base_utilities.to_float(2.0) == 2.0
    assert base_utilities.to_float("invalid") == 0.0
    assert base_utilities.to_float(None) == 0.0

def test_to_boolean():
    assert base_utilities.to_boolean("true") is True
    assert base_utilities.to_boolean("YES") is True
    assert base_utilities.to_boolean("1") is True
    assert base_utilities.to_boolean(True) is True
    assert base_utilities.to_boolean("false") is False
    assert base_utilities.to_boolean(None) is False

def test_to_integer():
    assert base_utilities.to_integer("10") == 10
    assert base_utilities.to_integer(5) == 5
    assert base_utilities.to_integer("abc") == 0
    assert base_utilities.to_integer(None) == 0

def test_to_list():
    assert base_utilities.to_list("[1, 2, 3]") == [1, 2, 3]
    assert base_utilities.to_list("a,b,c") == ["a", "b", "c"]
    assert base_utilities.to_list([1, 2]) == [1, 2]
    assert base_utilities.to_list(None) == []

def test_to_dict():
    assert base_utilities.to_dict('{"a": 1}') == {"a": 1}
    assert base_utilities.to_dict({"b": 2}) == {"b": 2}
    assert base_utilities.to_dict(None) == {}
    with pytest.raises(ValueError):
        base_utilities.to_dict("not a dict")

def test_expand_env_vars(monkeypatch):
    monkeypatch.setenv("TEST_VAR", "hello")
    assert base_utilities.expand_env_vars("${TEST_VAR}") == "hello"
    assert base_utilities.expand_env_vars("${MISSING:-default}") == "default"
    assert base_utilities.expand_env_vars("${MISSING}") == ""
    
    # Validation mode
    monkeypatch.setenv("VALIDATION_MODE", "true")
    assert base_utilities.expand_env_vars("${API_KEY}") == "dummy_api_key"
    assert base_utilities.expand_env_vars("${NORMAL_VAR}") == "validation_normal_var"

def test_is_loopback_url():
    assert base_utilities.is_loopback_url("http://localhost:8000", current_port=8000) is True
    assert base_utilities.is_loopback_url("http://127.0.0.1:8000", current_port=8000) is True
    assert base_utilities.is_loopback_url("http://google.com", current_port=8000) is False
    assert base_utilities.is_loopback_url("http://localhost:9000", current_port=8000) is False

def test_retrieve_package_name():
    # In a test environment, it might return 'agent_utilities' or the test runner package
    pkg = base_utilities.retrieve_package_name()
    assert isinstance(pkg, str)
    assert pkg != ""

def test_save_load_model(tmp_path):
    data = {"key": "value"}
    path = base_utilities.save_model(data, "test_model", str(tmp_path))
    assert os.path.exists(path)
    
    loaded = base_utilities.load_model(path)
    assert loaded == data

def test_result_class():
    res = base_utilities.Result()
    with pytest.raises(ValueError):
        _ = res.is_successful
    
    res._failed = False
    assert res.is_successful is True
    
    res._failed = True
    assert res.is_successful is False

def test_optional_import_block():
    with base_utilities.optional_import_block() as result:
        import os # Should succeed
    assert result.is_successful is True
    
    with base_utilities.optional_import_block() as result:
        import non_existent_module # Should fail
    assert result.is_successful is False

def test_module_info_from_str():
    mi = base_utilities.ModuleInfo.from_str("requests>=2.0.0")
    assert mi.name == "requests"
    assert mi.min_version == "2.0.0"
    assert mi.min_inclusive is True
    
    mi2 = base_utilities.ModuleInfo.from_str("pytest<8.0")
    assert mi2.name == "pytest"
    assert mi2.max_version == "8.0"
    assert mi2.max_inclusive is False

def test_get_missing_imports():
    # Assuming requests is installed
    missing = base_utilities.get_missing_imports(["requests", "non_existent_pkg"])
    assert "non_existent_pkg" in missing
    assert "requests" not in missing
