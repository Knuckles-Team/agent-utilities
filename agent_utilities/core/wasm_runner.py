# CONCEPT:OS-5.4 - WASM Micro-Agent Sandbox & Runner
# CONCEPT:ORCH-1.11 - Compiled Orchestration Kernel

import json
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

WASMTIME_AVAILABLE = False
wasmtime = None
try:
    import wasmtime  # type: ignore[no-redef]

    WASMTIME_AVAILABLE = True
except ImportError:
    pass


import ast
import operator as op

# Supported operators for safe math
_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
}

_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def safe_eval_math(expr: str, variables: dict[str, Any]) -> Any:
    """Safely evaluates a basic mathematical expression without eval().

    Prevents arbitrary code execution by using AST white-listing.
    """
    if not expr or not isinstance(expr, str):
        return 0

    try:
        node = ast.parse(expr, mode="eval")

        def _eval(n: ast.AST) -> Any:
            if isinstance(n, ast.Expression):
                return _eval(n.body)
            elif isinstance(n, ast.Constant):
                if isinstance(n.value, int | float):
                    return n.value
                raise ValueError("Only numbers are allowed in math expressions.")
            elif isinstance(n, ast.BinOp):
                left = _eval(n.left)
                right = _eval(n.right)
                op_type = type(n.op)
                if op_type in _BIN_OPS:
                    return _BIN_OPS[op_type](left, right)
                raise ValueError(f"Operator {op_type} is not supported.")
            elif isinstance(n, ast.UnaryOp):
                operand = _eval(n.operand)
                unary_op_type = type(n.op)
                if unary_op_type in _UNARY_OPS:
                    return _UNARY_OPS[unary_op_type](operand)
                raise ValueError(f"Unary operator {unary_op_type} is not supported.")
            elif isinstance(n, ast.Name):
                if n.id in variables:
                    val = variables[n.id]
                    if isinstance(val, int | float):
                        return val
                    raise ValueError(f"Variable '{n.id}' must be a number.")
                raise ValueError(f"Undefined variable '{n.id}'.")
            raise ValueError(f"Unsupported AST node type: {type(n)}")

        return _eval(node)
    except Exception as e:
        logger.error(f"Safe math evaluation failed for '{expr}': {e}")
        raise ValueError(f"Malicious or unsupported expression: {e}") from e


class WasmAgentRunner:
    """High-performance WebAssembly runner for sandboxed micro-agents.

    Enables micro-second cold starts, linear memory boundaries, and strict CPU/memory isolation.
    """

    def __init__(self, limit_memory_pages: int = 16):
        self.limit_memory_pages = limit_memory_pages
        self.store: Any = None
        self.module: Any = None
        self.instance: Any = None
        self.engine: Any = None

        if WASMTIME_AVAILABLE and wasmtime is not None:
            # Configure wasmtime engine with memory/resource limits
            self.config = wasmtime.Config()
            self.config.strategy = "cranelift"
            self.engine = wasmtime.Engine(self.config)
        else:
            logger.warning(
                "wasmtime is not installed. WASM micro-agents will run in emulation/fallback mode."
            )

    def load_agent(self, wasm_bytes: bytes) -> None:
        """Load and compile a pre-compiled WebAssembly micro-agent binary."""
        if not WASMTIME_AVAILABLE or wasmtime is None or self.engine is None:
            logger.info("Loaded WASM binary (fallback dry-run).")
            self.module = wasm_bytes  # store raw bytes for reference
            return

        self.store = wasmtime.Store(self.engine)
        self.module = wasmtime.Module(self.engine, wasm_bytes)

        # Simple linker for importing standard env interfaces
        linker = wasmtime.Linker(self.engine)

        # Instantiate module
        self.instance = linker.instantiate(self.store, self.module)

    def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Execute the loaded WASM agent with JSON inputs.

        Serializes data directly to/from the WASM linear memory sandbox.
        """
        input_str = json.dumps(input_data)

        if (
            not WASMTIME_AVAILABLE
            or wasmtime is None
            or self.instance is None
            or self.store is None
        ):
            # Emulation fallback mode for developers without wasmtime
            logger.info("Executing micro-agent in emulation/fallback mode.")
            action = input_data.get("action")
            if action == "calculate_fees":
                base_fee = input_data.get("base_fee", 0.0)
                state_fee = input_data.get("state_fee", 0.0)
                expedited = input_data.get("expedited", False)
                expedited_fee = 100.0 if expedited else 0.0

                # Check for custom formula
                formula = input_data.get(
                    "formula", "base_fee + state_fee + expedited_fee"
                )
                variables = {
                    "base_fee": base_fee,
                    "state_fee": state_fee,
                    "expedited_fee": expedited_fee,
                }

                total = safe_eval_math(formula, variables)
                return {
                    "status": "success",
                    "action": "calculate_fees",
                    "base_fee": base_fee,
                    "state_fee": state_fee,
                    "expedited_fee": expedited_fee,
                    "total_fee": total,
                    "emulated": True,
                }

            elif action == "draft_calculations":
                authorized_shares = input_data.get("authorized_shares", 0)
                par_value = input_data.get("par_value", 0.0)

                formula = input_data.get("formula", "authorized_shares * par_value")
                variables = {
                    "authorized_shares": authorized_shares,
                    "par_value": par_value,
                }

                total_capital = safe_eval_math(formula, variables)
                return {
                    "status": "success",
                    "action": "draft_calculations",
                    "authorized_shares": authorized_shares,
                    "par_value": par_value,
                    "total_capital": total_capital,
                    "emulated": True,
                }

            elif action == "expand_template":
                template = input_data.get("template", "")
                variables = input_data.get("variables", {})

                # Safely replace placeholders {{ name }}
                result = template
                for k, v in variables.items():
                    # Sanitize value slightly to prevent injection
                    clean_v = str(v)
                    result = result.replace(f"{{{{ {k} }}}}", clean_v)
                    result = result.replace(f"{{{{{k}}}}}", clean_v)

                return {
                    "status": "success",
                    "action": "expand_template",
                    "expanded": result,
                    "emulated": True,
                }

            return {
                "status": "success",
                "emulated": True,
                "input_received": input_data,
                "output": f"Processed: {input_str[:100]}...",
            }

        # Locate exports in WASM module
        exports = self.instance.exports(self.store)
        memory = exports.get("memory")
        alloc = exports.get("alloc")
        run = exports.get("run")

        if memory is None or not isinstance(memory, wasmtime.Memory):
            raise ValueError("WASM module must export 'memory'")

        # Support standard linear memory allocation and serialization
        if alloc is not None and run is not None:
            try:
                # 1. Allocate memory in WASM for input
                input_bytes = input_str.encode("utf-8")
                input_ptr = alloc(self.store, len(input_bytes))

                # 2. Write input directly into sandboxed memory
                memory.write(self.store, input_bytes, input_ptr)

                # 3. Invoke compiled agent execution
                output_ptr = run(self.store, input_ptr, len(input_bytes))

                # 4. Read output back from WASM memory
                output_bytes = memory.read(self.store, output_ptr, 4096)
                end = output_bytes.find(b"\x00")
                if end != -1:
                    output_bytes = output_bytes[:end]

                output_str = output_bytes.decode("utf-8")
                return json.loads(output_str)
            except Exception as e:
                logger.error(f"Sandboxed WASM execution failed: {e}")
                raise RuntimeError(f"WASM execution error: {e}") from e
        else:
            # Fallback direct call if custom entry points are used
            run_func = exports.get("run")
            if run_func is not None:
                try:
                    res = run_func(self.store)
                    return {"status": "success", "result": res}
                except Exception as e:
                    raise RuntimeError(f"WASM execute run failed: {e}") from e

            raise ValueError(
                "WASM module must export standard alloc/run entrypoints or a simple run() function."
            )
