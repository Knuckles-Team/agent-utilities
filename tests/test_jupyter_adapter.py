"""
Tests for JupyterKernelAdapter and SandboxExecutor (CONCEPT:AU-ECO.messaging.native-backend-abstraction).
"""

from agent_utilities.tools.jupyter_adapter import JupyterKernelAdapter
from agent_utilities.tools.sandbox_executor import SandboxExecutor


def test_jupyter_adapter_execute():
    adapter = JupyterKernelAdapter()
    result = adapter.execute("print('Hello Trader')")
    assert result["status"] == "error"
    assert result["code"] == "execution_backend_unavailable"
    adapter.restart()


def test_sandbox_executor_safe():
    sandbox = SandboxExecutor()
    code = "x = [i for i in range(10)]"
    result = sandbox.run_safe(code)
    assert result["status"] == "error"
    assert result["code"] == "execution_backend_unavailable"


def test_sandbox_executor_unsafe():
    sandbox = SandboxExecutor()
    # Invariant violation: trying to break out
    code = "import os; os.system('rm -rf /')"
    result = sandbox.run_safe(code)
    assert result["status"] == "error"
    assert "InvariantViolation" in result["error"]
