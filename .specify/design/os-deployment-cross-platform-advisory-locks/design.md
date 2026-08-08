# Design Document: One shared helper bridges POSIX `fcntl.flock` and a stdlib-only Windows `msvcrt.locking` fallback for every advisory lock the KG guards need, instead of a third-party cross-platform locking dependency or POSIX-only support

CONCEPT:AU-OS.deployment.cross-platform-locks-plus

> `agent_utilities/knowledge_graph/core/file_lock.py:1-40` (module docstring,
> `lock_exclusive_nb`, `lock_shared_nb`, `unlock`).

## Decision — `file_lock.py` is the ONE routing point the KG's single-instance guards (`host_lock`, `engine_lock`) and the liveness probe go through for advisory locking, implemented with `fcntl.flock` on POSIX and a stdlib `msvcrt.locking`-based emulation on Windows (no new third-party dependency), matching semantics as closely as each OS allows

`fcntl.flock` is POSIX-only, but the guards that depend on advisory locking's
crash-safety property — a lock auto-releases when its holder process dies, so there
is never a stale-PID lock file to clean up by hand — need to work on every
supported platform. `file_lock.py` is that ONE helper: `lock_exclusive_nb` (the
spawn/host guard), `lock_shared_nb` (the liveness probe), and `unlock`. On POSIX it
is a direct `fcntl.flock` wrapper. On Windows, which has no shared-lock mode at the
OS level, a "shared" acquire is deliberately emulated as a non-blocking exclusive
lock that is immediately released — sufficient for the liveness probe's only real
question ("is anyone holding it?"): acquire, see it free, unlock; if a real holder
has it, the acquire fails (`file_lock.py:19-23`). Windows byte-range locks are
released when the owning handle/process closes, preserving the same
crash-safety property the POSIX side relies on. A losing non-blocking acquire
raises one typed `LockUnavailable(OSError)` on both platforms, so callers keep a
single `except (BlockingIOError, OSError)` arm unchanged regardless of OS.

## Rejected alternative — a third-party cross-platform locking library, or POSIX-only support with Windows left unlocked/unsupported

Two alternatives are named directly in the source. First: pull in a third-party
cross-platform file-locking package — rejected in favour of "a Windows fallback
built on stdlib `msvcrt.locking` (**no new third-party dep**)" (`file_lock.py:5-6`),
consistent with the platform's general dependency-discipline stance (keep the base
install lean; every new dependency is a supply-chain and install-size cost paid by
every deployment, not just Windows ones). Second: simply not supporting Windows for
these locks — rejected because the single-instance guards' crash-safety property
(no manual stale-lock cleanup) is exactly the property a deployment needs
regardless of host OS; leaving Windows unlocked would mean either skipping the
guard there (reintroducing the multi-instance/stale-PID problems the guard exists
to prevent) or blocking Windows from running these components at all. Emulating
"shared" via an immediately-released exclusive lock accepts a narrower semantic
match (no true concurrent-shared-readers mode on Windows) in exchange for covering
the one thing the liveness probe actually needs, without either alternative's cost.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/core/file_lock.py`; every
  caller of `host_lock`/`engine_lock`/the liveness probe.
- **Backward Compatible**: Yes — same call surface (`lock_exclusive_nb`/
  `lock_shared_nb`/`unlock`), same `LockUnavailable` contract, on both platforms.
- **Known weak point**: the Windows "shared" emulation is not a true shared lock —
  two Windows processes both wanting a genuinely concurrent shared read (not just
  "probe once and release") would serialize through the exclusive-lock emulation
  instead of holding the lock concurrently; today's only shared-lock caller (the
  liveness probe) does not need true concurrency, but a future caller that does
  would not get it on Windows.
