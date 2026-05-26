"""CONCEPT:ORCH-1.29 WASM Micro-Agent execution tests."""

import pytest
from agent_utilities.core.wasm_runner import WasmAgentRunner, safe_eval_math

def test_safe_eval_math():
    """Test that safe_eval_math safely parses mathematical expressions."""
    # Test simple additions
    assert safe_eval_math("1 + 2", {}) == 3.0
    assert safe_eval_math("10 - 4", {}) == 6.0
    assert safe_eval_math("3 * 5", {}) == 15.0
    assert safe_eval_math("20 / 4", {}) == 5.0

    # Test nested variables
    vars_dict = {"base": 100.0, "tax": 15.0, "discount": 5.0}
    assert safe_eval_math("base + tax - discount", vars_dict) == 110.0
    assert safe_eval_math("(base + tax) * 2", vars_dict) == 230.0

    # Test unary operators
    assert safe_eval_math("-base", vars_dict) == -100.0
    assert safe_eval_math("+base", vars_dict) == 100.0

    # Test malicious expressions / unsupported actions are safely blocked
    with pytest.raises(ValueError, match="Malicious or unsupported expression"):
        safe_eval_math("base.__class__", vars_dict)
    with pytest.raises(ValueError, match="Malicious or unsupported expression"):
        safe_eval_math("__import__('os').system('ls')", {})
    with pytest.raises(ValueError, match="Malicious or unsupported expression"):
        safe_eval_math("eval('1+1')", {})

def test_wasm_runner_fallback_emulation():
    """Test that the WASM runner executes correctly in emulation/fallback mode."""
    runner = WasmAgentRunner()

    # Verify the runner is initialized
    assert runner is not None

    # Run executing input payload
    payload = {"agent_id": "test_agent_1", "task": "run_inference", "params": [1, 2, 3]}
    result = runner.execute(payload)

    # Assert correct structure and success status
    assert result["status"] == "success"
    assert result["emulated"] is True
    assert result["input_received"] == payload
    assert "Processed:" in result["output"]
    assert "test_agent_1" in result["output"]

def test_wasm_runner_calculate_fees():
    """Test the Wasm runner fee calculation emulator fallback."""
    runner = WasmAgentRunner()

    payload = {
        "action": "calculate_fees",
        "base_fee": 150.0,
        "state_fee": 50.0,
        "expedited": True,
        "formula": "base_fee + state_fee + expedited_fee + 10"
    }

    result = runner.execute(payload)
    assert result["status"] == "success"
    assert result["emulated"] is True
    assert result["action"] == "calculate_fees"
    assert result["base_fee"] == 150.0
    assert result["state_fee"] == 50.0
    assert result["expedited_fee"] == 100.0
    assert result["total_fee"] == 310.0

def test_wasm_runner_draft_calculations():
    """Test the Wasm runner draft calculations emulator fallback."""
    runner = WasmAgentRunner()

    payload = {
        "action": "draft_calculations",
        "authorized_shares": 10000,
        "par_value": 0.01
    }

    result = runner.execute(payload)
    assert result["status"] == "success"
    assert result["emulated"] is True
    assert result["action"] == "draft_calculations"
    assert result["authorized_shares"] == 10000
    assert result["par_value"] == 0.01
    assert result["total_capital"] == 100.0

def test_wasm_runner_expand_template():
    """Test the Wasm runner template expander emulator fallback."""
    runner = WasmAgentRunner()

    payload = {
        "action": "expand_template",
        "template": "Hello {{ name }}, welcome to {{ state }}!",
        "variables": {
            "name": "Jane",
            "state": "Texas"
        }
    }

    result = runner.execute(payload)
    assert result["status"] == "success"
    assert result["emulated"] is True
    assert result["action"] == "expand_template"
    assert result["expanded"] == "Hello Jane, welcome to Texas!"

def test_wasm_runner_empty_payload():
    """Test the runner handling an empty payload in emulation mode."""
    runner = WasmAgentRunner()
    result = runner.execute({})

    assert result["status"] == "success"
    assert result["emulated"] is True
    assert result["input_received"] == {}
    assert "Processed:" in result["output"]
