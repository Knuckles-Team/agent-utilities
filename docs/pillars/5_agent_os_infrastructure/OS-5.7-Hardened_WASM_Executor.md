# Hardened WASM Sandbox Executor (CONCEPT:OS-5.4)

## Overview
Rather than executing raw local subprocesses or shell commands which pose severe prompt and command injection risks, the kernel executes micro-agents and untrusted utility tools inside an isolated **WebAssembly (WASM) Sandbox**.

By utilizing the standard `wasmtime` engine, the sandbox guarantees memory safety, execution budget enforcement, and microsecond-level isolation.

## Architecture

The WASM executor operates in two modes:

### Native Mode (wasmtime installed)
```
                ┌─────────────────────────────────────────────┐
                │              WasmAgentRunner                │
                │                                             │
  JSON input ──►│  1. Configure wasmtime Engine (Cranelift)   │
                │  2. Store + Module compilation              │
                │  3. Linker → Instantiate module             │
                │  4. Allocate linear memory for input        │
                │  5. Write JSON bytes to WASM memory         │
                │  6. Call run(ptr, len) → output_ptr         │
                │  7. Read output from WASM memory            │
                │  8. Parse JSON output                       │──► JSON output
                └─────────────────────────────────────────────┘
```

### Emulation/Fallback Mode (no wasmtime)
```
  JSON input ──► Action Router ──► safe_eval_math() ──► JSON output
                     │
                     ├── calculate_fees (base + state + expedited)
                     ├── draft_calculations (shares × par value)
                     └── expand_template ({{variable}} substitution)
```

## Gas Model

WASM execution is constrained by the Cranelift compiler's native instruction metering:

| Resource | Default Limit | Rationale |
|---|---|---|
| **Memory Pages** | 16 pages (1 MB) | Prevents OOM crashes on the host |
| **Linear Memory** | Bounded by `limit_memory_pages × 64KB` | WASM spec: 1 page = 64KB |
| **Compilation Strategy** | Cranelift (AOT) | Sub-millisecond cold starts |
| **Output Buffer** | 4096 bytes | Prevents unbounded output extraction |

Future enhancement: Explicit `fuel` metering (wasmtime's `consume_fuel` API) to enforce per-invocation CPU budgets.

## Memory Model

WASM provides **linear memory isolation** — the sandbox cannot access host memory, filesystem, or network:

```
Host Process Memory
├── Python heap (unreachable from WASM)
├── wasmtime Engine state
└── WASM Store
    └── Linear Memory (sandboxed)
        ├── [0..input_ptr]: Reserved
        ├── [input_ptr..input_ptr+len]: Input JSON bytes
        ├── [output_ptr..output_ptr+4096]: Output JSON bytes
        └── [4096+..]: Available for module-internal use
```

### Memory Safety Guarantees
1. **No pointer escapes** — WASM pointers are offsets into linear memory, not host addresses
2. **Bounds checking** — Every load/store is validated against memory size
3. **Stack isolation** — WASM operand stack is separate from host call stack
4. **No system calls** — Unless explicitly linked via the wasmtime `Linker`

## Syscall Policy

The default `WasmAgentRunner` linker imports **zero host functions**, creating a fully isolated sandbox:

| Capability | Default | Notes |
|---|---|---|
| Filesystem access | ❌ Denied | No WASI filesystem imports |
| Network access | ❌ Denied | No WASI socket imports |
| Environment variables | ❌ Denied | No `environ_get` import |
| Clock/time | ❌ Denied | No `clock_time_get` import |
| Random | ❌ Denied | No `random_get` import |
| stdout/stderr | ❌ Denied | No `fd_write` import |

To enable selective capabilities (e.g., for WASI-compatible modules), extend the linker:

```python
runner = WasmAgentRunner(limit_memory_pages=32)
# Future: runner.enable_wasi(fs=False, net=False, clock=True)
```

## Safe Math Evaluation

The fallback mode uses `safe_eval_math()` instead of Python's `eval()`:

1. **AST Whitelisting** — Parses expression into AST, only allows `Constant`, `BinOp`, `UnaryOp`, `Name`
2. **Operator Restriction** — Only `+`, `-`, `*`, `/`, unary `-`, unary `+`
3. **Variable Binding** — Only pre-declared numeric variables from the input dict
4. **No Imports** — No access to `__builtins__`, `os`, `sys`, or any module
5. **Type Enforcement** — Only `int` and `float` values permitted in expressions

```python
# Safe:
safe_eval_math("base_fee + state_fee * 1.5", {"base_fee": 100, "state_fee": 50})
# → 175.0

# Blocked:
safe_eval_math("__import__('os').system('rm -rf /')", {})
# → ValueError: Unsupported AST node type
```

## API Surface

```python
from agent_utilities.core.wasm_runner import WasmAgentRunner

runner = WasmAgentRunner(limit_memory_pages=16)

# Load pre-compiled WASM binary
with open("micro_agent.wasm", "rb") as f:
    runner.load_agent(f.read())

# Execute with JSON I/O
result = runner.execute({
    "action": "calculate_fees",
    "base_fee": 150.0,
    "state_fee": 50.0,
    "expedited": True,
})
# → {"status": "success", "total_fee": 300.0, ...}
```

## Integration Points

- **Security Kernel (OS-5.1)**: Tool guard checks run before WASM dispatch
- **Ontological Guardrails (OS-5.10)**: Arguments validated against OWL policies before sandbox entry
- **Cognitive Scheduler (OS-5.2)**: WASM executions are tracked as agent processes with token quotas
- **SandboxedExecutor (OS-5.6)**: Process-level isolation complement — WASM for untrusted code, subprocess for trusted scripts

## Implementation Details
- **Source Code**: [`wasm_runner.py`](file:///home/apps/workspace/agent-packages/agent-utilities/agent_utilities/core/wasm_runner.py) (257 lines)
- **Classes**: `WasmAgentRunner`, `safe_eval_math`
- **Tests**: [`test_wasm_runner.py`](file:///home/apps/workspace/agent-packages/agent-utilities/tests/unit/core/test_wasm_runner.py)
- **Pillar**: OS
- **Package Export**: `agent_utilities.core.WasmAgentRunner`
- **Optional Dependency**: `wasmtime` (graceful fallback to emulation mode)
