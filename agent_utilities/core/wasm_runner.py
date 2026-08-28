# CONCEPT:AU-OS.governance.wasm-micro-agent-sandbox - WASM Micro-Agent Sandbox & Runner
# CONCEPT:AU-ORCH.sandbox.compiled-orchestration-kernel - Compiled Orchestration Kernel

import json
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

WASMTIME_AVAILABLE = False
wasmtime: Any = None
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


def _eval_math_constant(node: ast.Constant) -> Any:
    """Evaluate a numeric literal node for safe_eval_math."""
    if isinstance(node.value, int | float):
        return node.value
    raise ValueError("Only numbers are allowed in math expressions.")


def _eval_math_binop(node: ast.BinOp, eval_fn: Callable[[ast.AST], Any]) -> Any:
    """Evaluate a whitelisted binary operator node for safe_eval_math."""
    left = eval_fn(node.left)
    right = eval_fn(node.right)
    op_type = type(node.op)
    if op_type in _BIN_OPS:
        return _BIN_OPS[op_type](left, right)
    raise ValueError(f"Operator {op_type} is not supported.")


def _eval_math_unaryop(node: ast.UnaryOp, eval_fn: Callable[[ast.AST], Any]) -> Any:
    """Evaluate a whitelisted unary operator node for safe_eval_math."""
    operand = eval_fn(node.operand)
    unary_op_type = type(node.op)
    if unary_op_type in _UNARY_OPS:
        return _UNARY_OPS[unary_op_type](operand)
    raise ValueError(f"Unary operator {unary_op_type} is not supported.")


def _eval_math_name(node: ast.Name, variables: dict[str, Any]) -> Any:
    """Resolve a variable reference node for safe_eval_math."""
    if node.id in variables:
        val = variables[node.id]
        if isinstance(val, int | float):
            return val
        raise ValueError(f"Variable '{node.id}' must be a number.")
    raise ValueError(f"Undefined variable '{node.id}'.")


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
            if isinstance(n, ast.Constant):
                return _eval_math_constant(n)
            if isinstance(n, ast.BinOp):
                return _eval_math_binop(n, _eval)
            if isinstance(n, ast.UnaryOp):
                return _eval_math_unaryop(n, _eval)
            if isinstance(n, ast.Name):
                return _eval_math_name(n, variables)
            raise ValueError(f"Unsupported AST node type: {type(n)}")

        return _eval(node)
    except Exception as e:
        logger.error(f"Safe math evaluation failed for '{expr}': {e}")
        raise ValueError(f"Malicious or unsupported expression: {e}") from e


def _emulate_calculate_fees(input_data: dict[str, Any]) -> dict[str, Any]:
    """Emulation fallback for the 'calculate_fees' micro-agent action."""
    base_fee = input_data.get("base_fee", 0.0)
    state_fee = input_data.get("state_fee", 0.0)
    expedited = input_data.get("expedited", False)
    expedited_fee = 100.0 if expedited else 0.0

    # Check for custom formula
    formula = input_data.get("formula", "base_fee + state_fee + expedited_fee")
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


def _emulate_draft_calculations(input_data: dict[str, Any]) -> dict[str, Any]:
    """Emulation fallback for the 'draft_calculations' micro-agent action."""
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


def _emulate_expand_template(input_data: dict[str, Any]) -> dict[str, Any]:
    """Emulation fallback for the 'expand_template' micro-agent action."""
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


_EMULATED_ACTIONS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "calculate_fees": _emulate_calculate_fees,
    "draft_calculations": _emulate_draft_calculations,
    "expand_template": _emulate_expand_template,
}


def _run_emulated(input_data: dict[str, Any], input_str: str) -> dict[str, Any]:
    """Dispatch to the emulation/fallback handler for developers without wasmtime."""
    logger.info("Executing micro-agent in emulation/fallback mode.")
    action = input_data.get("action")
    handler = _EMULATED_ACTIONS.get(action) if isinstance(action, str) else None
    if handler is not None:
        return handler(input_data)

    return {
        "status": "success",
        "emulated": True,
        "input_received": input_data,
        "output": f"Processed: {input_str[:100]}...",
    }


def _validate_int_range(value: Any, low: int, high: int, message: str) -> None:
    """Validate a plain (non-bool) int within [low, high], raising ValueError otherwise."""
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(message)


def _resolve_wasm_limits(
    admission: Any | None,
    limit_memory_pages: int,
    limit_cpu_fuel: int,
    max_payload_bytes: int,
) -> tuple[int, int, int]:
    """Resolve effective (memory_pages, cpu_fuel, payload_bytes), clamped to admission policy."""
    if admission is None:
        return limit_memory_pages, limit_cpu_fuel, max_payload_bytes

    admission.require_capabilities(("rlm.execute",))
    limits = getattr(admission, "resource_limits", None)
    if limits is None:
        raise ValueError("governed WASM execution requires resource limits")
    if float(limits.cpu_cores) != 1.0:
        raise ValueError(
            "Wasmtime cannot enforce a fractional or multi-core CPU share"
        )

    memory_pages = min(limit_memory_pages, int(limits.max_wasm_pages))
    payload_bytes = min(max_payload_bytes, int(admission.max_payload_bytes))
    cpu_fuel = min(
        limit_cpu_fuel,
        max(1, int(float(limits.deadline_s) * 1_000_000)),
    )
    return memory_pages, cpu_fuel, payload_bytes


class WasmAgentRunner:
    """High-performance WebAssembly runner for sandboxed micro-agents.

    Enables micro-second cold starts, linear memory boundaries, and strict CPU/memory isolation.
    """

    def __init__(
        self,
        limit_memory_pages: int = 16,
        *,
        limit_cpu_fuel: int = 50_000_000,
        max_payload_bytes: int = 4 * 1024 * 1024,
        admission: Any | None = None,
    ):
        _validate_int_range(
            limit_memory_pages, 1, 1_048_576, "WASM memory page limit is out of range"
        )
        _validate_int_range(
            limit_cpu_fuel, 1, 10_000_000_000, "WASM CPU fuel limit is out of range"
        )
        _validate_int_range(
            max_payload_bytes, 1, 64 * 1024 * 1024, "WASM payload limit is out of range"
        )

        self.admission = admission
        (
            self.limit_memory_pages,
            self.limit_cpu_fuel,
            self.max_payload_bytes,
        ) = _resolve_wasm_limits(
            admission, limit_memory_pages, limit_cpu_fuel, max_payload_bytes
        )

        self.store: Any = None
        self.module: Any = None
        self.instance: Any = None
        self.engine: Any = None
        self._init_wasmtime_engine()

    def _init_wasmtime_engine(self) -> None:
        """Configure the wasmtime engine with memory/resource limits, if available."""
        if not (WASMTIME_AVAILABLE and wasmtime is not None):
            logger.warning(
                "wasmtime is not installed. WASM micro-agents will run in emulation/fallback mode."
            )
            return

        # Configure wasmtime engine with memory/resource limits
        self.config = wasmtime.Config()
        self.config.strategy = "cranelift"
        try:
            self.config.consume_fuel = True
        except Exception:  # noqa: BLE001 - epoch deadline remains the runtime guard
            logger.debug("Wasmtime fuel accounting unavailable")
        self.engine = wasmtime.Engine(self.config)

    def load_agent(self, wasm_bytes: bytes) -> None:
        """Load and compile a pre-compiled WebAssembly micro-agent binary."""
        if self.admission is not None:
            self.admission.remaining_seconds()
            self.admission.require_payload(wasm_bytes, label="WASM module payload")
        if not WASMTIME_AVAILABLE or wasmtime is None or self.engine is None:
            if self.admission is not None:
                raise RuntimeError("governed WASM execution requires Wasmtime")
            logger.info("Loaded WASM binary (fallback dry-run).")
            self.module = wasm_bytes  # store raw bytes for reference
            return

        self.store = wasmtime.Store(self.engine)
        set_limits = getattr(self.store, "set_limits", None)
        if not callable(set_limits):
            raise RuntimeError("Wasmtime store limiter is unavailable")
        try:
            set_limits(memory_size=self.limit_memory_pages * 65_536)
        except Exception as exc:  # noqa: BLE001 - a stored limit is not enforcement
            raise RuntimeError("Wasmtime memory limiter could not be applied") from exc
        set_fuel = getattr(self.store, "set_fuel", None)
        if not callable(set_fuel):
            raise RuntimeError("Wasmtime CPU fuel limiter is unavailable")
        try:
            set_fuel(self.limit_cpu_fuel)
        except Exception as exc:  # noqa: BLE001 - fail closed when fuel was advertised but unusable
            raise RuntimeError("Wasmtime CPU fuel limit could not be applied") from exc
        self.module = wasmtime.Module(self.engine, wasm_bytes)

        # Simple linker for importing standard env interfaces
        linker = wasmtime.Linker(self.engine)

        # Instantiate module
        self.instance = linker.instantiate(self.store, self.module)

    def _wasm_engine_ready(self) -> bool:
        """Whether a compiled wasmtime module is loaded and ready to execute."""
        return (
            WASMTIME_AVAILABLE
            and wasmtime is not None
            and self.instance is not None
            and self.store is not None
        )

    def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Execute the loaded WASM agent with JSON inputs.

        Serializes data directly to/from the WASM linear memory sandbox.
        """
        input_str = json.dumps(input_data, separators=(",", ":"))
        if self.admission is not None:
            self.admission.remaining_seconds()
            self.admission.require_payload(input_str, label="WASM input payload")
        elif len(input_str.encode("utf-8")) > self.max_payload_bytes:
            raise ValueError("WASM input payload exceeds the admission limit")

        if not self._wasm_engine_ready():
            if self.admission is not None:
                raise RuntimeError(
                    "governed WASM execution requires a loaded Wasmtime module"
                )
            # Emulation fallback mode for developers without wasmtime
            return _run_emulated(input_data, input_str)

        return self._execute_wasm(input_str)

    def _execute_wasm(self, input_str: str) -> dict[str, Any]:
        """Run the compiled WASM module against sandboxed linear memory."""
        # Locate exports in WASM module
        exports = self.instance.exports(self.store)
        memory = exports.get("memory")
        alloc = exports.get("alloc")
        run = exports.get("run")

        if memory is None or not isinstance(memory, wasmtime.Memory):
            raise ValueError("WASM module must export 'memory'")

        # Support standard linear memory allocation and serialization
        if alloc is not None and run is not None:
            return self._run_via_linear_memory(alloc, run, memory, input_str)
        return self._run_via_direct_call(exports)

    def _run_via_linear_memory(
        self, alloc: Any, run: Any, memory: Any, input_str: str
    ) -> dict[str, Any]:
        """Serialize input into WASM linear memory, invoke run(), and decode the result."""
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

    def _run_via_direct_call(self, exports: Any) -> dict[str, Any]:
        """Fallback direct call for modules that only export a simple run() function."""
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
