# Process Lifecycle Management

> Server-side process cleanup for MCP sidecars, TUI, and background threads.

## Overview

When `agent-utilities` starts a server via `create_agent_server()` or `create_graph_agent_server()`, it may spawn several child processes:

- **MCP Servers**: stdio-based subprocesses for each configured MCP server
- **Terminal UI**: The `agent-terminal-ui` process launched via `subprocess.Popen`
- **Background Threads**: Pipeline runners, maintenance tasks, cron schedulers

If the main server process exits (via `SIGTERM`, `SIGINT`, or normal shutdown), these child processes must be cleaned up to avoid orphaned processes consuming system resources.

## Problem

Without explicit cleanup:

1. MCP server processes become orphans and keep running in the background
2. The TUI process may block terminal restoration
3. Background threads may hold file locks or database connections
4. On the next server start, port conflicts or stale state may cause failures

## Solution

The `server/__init__.py` module registers cleanup handlers at three levels:

```python
import atexit
import os
import signal

_main_pid = os.getpid()
_cleanup_done = False

def _cleanup_child_processes(signum=None, frame=None):
    """Kill all child processes spawned by this server on exit."""
    nonlocal _cleanup_done
    if _cleanup_done:
        return
    _cleanup_done = True
    result = subprocess.run(
        ["pgrep", "-P", str(_main_pid)],
        capture_output=True, text=True, timeout=5,
    )
    for pid_str in result.stdout.strip().split("\n"):
        os.kill(int(pid_str.strip()), signal.SIGTERM)

# Register for all exit paths
atexit.register(_cleanup_child_processes)
signal.signal(signal.SIGTERM, _cleanup_child_processes)
signal.signal(signal.SIGINT, _cleanup_child_processes)
```

### Why `pgrep -P` Instead of `os.killpg()`?

The initial implementation used `os.killpg(os.getpgid(pid), signal.SIGTERM)` to kill the entire process group. However, this had a critical side effect: **it kills the main process itself**, including any parent process (like a test runner). The `pgrep -P <pid>` approach only targets direct child processes, leaving the main process and any parent processes intact.

### Signal Handler Chaining

If a previous signal handler is already registered (e.g., by uvicorn or pytest), the cleanup handler chains with it rather than replacing it:

```python
prev = signal.getsignal(signal.SIGTERM)
if prev in (signal.SIG_DFL, signal.SIG_IGN, None):
    signal.signal(signal.SIGTERM, _cleanup_child_processes)
else:
    def _chained(signum, frame, _prev=prev):
        _cleanup_child_processes(signum, frame)
        if callable(_prev):
            _prev(signum, frame)
    signal.signal(signal.SIGTERM, _chained)
```

### Idempotency Guard

The `_cleanup_done` flag ensures the cleanup function runs exactly once, even if triggered by multiple signals in quick succession:

```python
_cleanup_done = False

def _cleanup_child_processes(signum=None, frame=None):
    nonlocal _cleanup_done
    if _cleanup_done:
        return
    _cleanup_done = True
    # ... actual cleanup
```

## Exit Paths Covered

| Exit Path | Handler | Behavior |
|-----------|---------|----------|
| `Ctrl+C` (SIGINT) | `signal.signal(SIGINT, ...)` | Cleanup then exit |
| `kill <pid>` (SIGTERM) | `signal.signal(SIGTERM, ...)` | Cleanup then exit |
| Normal exit | `atexit.register(...)` | Cleanup during interpreter shutdown |
| `kill -9 <pid>` (SIGKILL) | ❌ Not catchable | Children become orphans (OS-level limitation) |

## Diagnostic Logging

Child process cleanup is logged at DEBUG level:

```
DEBUG - Cleaned up child process 12345
DEBUG - Cleaned up child process 12346
```

## Related Documentation

- [Configuration](../5_agent_os_infrastructure/configuration.md) — Server startup flags and environment variables
- [Architecture](../1_graph_orchestration/architecture.md) — Server endpoint architecture
- [First Principles Architecture](../1_graph_orchestration/first-principles.md) — Full CONCEPT:ORCH-1.2 through CONCEPT:ECO-4.0 overview
