"""
Tests for JupyterKernelAdapter and SandboxExecutor (CONCEPT:ECO-4.3).
"""

import pytest
from agent_utilities.tools.jupyter_adapter import JupyterKernelAdapter
from agent_utilities.tools.sandbox_executor import SandboxExecutor


def test_jupyter_adapter_execute():
    adapter = JupyterKernelAdapter()
    result = adapter.execute("print('Hello Trader')")
    assert result["status"] == "ok"
    assert (
        "Hello Trader" in result["output"]
        or "Successfully executed" in result["output"]
    )
    adapter.restart()


def test_sandbox_executor_safe():
    sandbox = SandboxExecutor()
    code = "x = [i for i in range(10)]"
    result = sandbox.run_safe(code)
    assert result["status"] == "ok"


def test_sandbox_executor_unsafe():
    sandbox = SandboxExecutor()
    # Invariant violation: trying to break out
    code = "import os; os.system('rm -rf /')"
    result = sandbox.run_safe(code)
    assert result["status"] == "error"
    assert "InvariantViolation" in result["error"]
