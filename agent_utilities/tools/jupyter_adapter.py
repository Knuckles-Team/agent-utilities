"""
Jupyter Kernel Adapter (CONCEPT:AU-ECO.messaging.native-backend-abstraction)

Defines the governed Jupyter execution boundary. No local execution backend is
bundled: callers fail closed until a separately isolated runtime is configured.
"""

from typing import Any


class JupyterKernelAdapter:
    """Fail-closed adapter for an externally governed execution runtime.

    The historical implementation returned a fabricated success without
    starting a kernel. That could make orchestration acknowledge work that was
    never executed and encouraged callers to treat an in-process kernel as a
    sandbox. A future backend must provide OS/container isolation, resource
    quotas, an egress policy, and approval before replacing this boundary.
    """

    def __init__(self, kernel_name: str = "python3"):
        self.kernel_name = kernel_name
        self._is_ready = True

    def execute(self, code: str, timeout: int = 30) -> dict[str, Any]:
        """Reject execution when no governed runtime has been provisioned."""
        if not isinstance(code, str) or len(code.encode("utf-8")) > 65_536:
            return {
                "status": "error",
                "code": "execution_input_invalid",
                "error": "Execution input is empty or exceeds the supported limit.",
            }
        if not code.strip() or not isinstance(timeout, int) or not 1 <= timeout <= 600:
            return {
                "status": "error",
                "code": "execution_input_invalid",
                "error": "Execution input or timeout is invalid.",
            }
        return {
            "status": "error",
            "code": "execution_backend_unavailable",
            "error": "No governed isolated execution backend is configured.",
        }

    def restart(self):
        """Restarts the underlying Jupyter kernel."""
        self._is_ready = True
