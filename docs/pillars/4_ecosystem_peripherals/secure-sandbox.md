# Secure Jupyter Sandbox (CONCEPT:ECO-4.12)

The **Secure Sandbox Engine** provides a constrained, isolated environment for dynamic code generation and execution, specifically tailored for backtesting and quantitative trading logic.

## Overview
Running dynamically generated quantitative strategies natively poses immense risk. `agent-utilities` mitigates this by integrating a `JupyterKernelAdapter` wrapped inside a `SandboxExecutor`. Before any generated code is executed, it undergoes rigorous mathematical and structural validation.

### Key Benefits
- **Programmatic Safety**: By validating the Abstract Syntax Tree (AST), the sandbox blocks dangerous OS-level calls (e.g., `os.system`).
- **Mathematical Invariants**: We enforce **State Machine Invariants (MCS Ch 6)** to guarantee that quantitative loops terminate and do not exceed pre-defined execution budgets.
- **Full Reproducibility**: The sandbox can be instantiated identically across test, staging, and production environments.

## Implementation Details
Located in `agent_utilities.tools.sandbox_executor`, the engine uses standard Jupyter kernel protocols (`jupyter-client`) behind the scenes while front-loading the security checks.

### Workflow
1. The orchestrator receives a generated strategy from a specialist agent.
2. The orchestrator forwards the strategy to the `SandboxExecutor`.
3. The executor runs `_validate_invariants(code)` against the AST.
4. If the code passes validation, it is forwarded to the isolated `JupyterKernelAdapter`.
5. The result (or a runtime error) is returned structured as JSON.

### Example
```python
from agent_utilities import SandboxExecutor

sandbox = SandboxExecutor()

# Safe execution
result = sandbox.run_safe("import pandas as pd; df = pd.DataFrame()")
print(result["status"]) # ok

# Unsafe execution (Blocked by Invariants)
unsafe_result = sandbox.run_safe("import os; os.system('rm -rf /')")
print(unsafe_result["error"]) # InvariantViolation...
```
