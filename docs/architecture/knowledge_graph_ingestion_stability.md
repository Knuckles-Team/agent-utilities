# Knowledge Graph Ingestion Stability & Locking Architecture

This document details the robust locking and process lifecycle architecture implemented to resolve persistent C++ segmentation faults and database connection/lock failures during bulk ingestion of open-source repositories into the `agent-utilities` Knowledge Graph.

> **Scope note**: The locking/lifecycle hardening below is specific to the
> **LadybugDB** backend, which is now an **opt-in contrib** driver
> (`backends/contrib/ladybug_backend.py`, instantiated explicitly as a projection).
> The operational backend is the Rust-native EpistemicGraph — the one authority, compute + cache +
> semantic + durable persistence in a single store (`memory`/`file` are
> snapshot modes of the same engine). Optional mirrors (PostgreSQL + pg-age,
> declared through `GRAPH_MIRROR_TARGETS`) are write-only fan-out targets. None of these use
> the SQLite file-lock mechanics described here. This document therefore applies
> only when LadybugDB is explicitly enabled.

---

## 1. POSIX Advisory Locking & Watcher Synchronization

### The Locking Hazard (Before)
Previously, the backend attempted to actively delete the advisory lock file (`*.lock`) in `LadybugBackend._recover_connection()` or upon release. Since `filelock` uses OS-level advisory locking (`flock`/`fcntl`), unlinking the lock file from disk while another process is holding the active lock descriptor breaks mutual exclusion on POSIX systems.

Unlinking the file allows a second process to create a new inode with the exact same name and acquire a lock on the new descriptor, leading to concurrent writes, catalog exceptions, or fatal segmentation faults.

### Robust POSIX Advisory Locking (After)
To prevent this concurrency hazard, the lock file is **never unlinked** from disk once created. Mutual exclusion is managed naturally via the filesystem inode's file-lock metadata.

The background synchronization watcher (`agent_utilities/sdd/watcher.py`) no longer checks for file existence using `os.path.exists(lock_path)`. Instead, it attempts a **non-blocking lock acquisition** (`timeout=0`). If the lock is held by active ingestion, it gracefully skips the iteration; if acquired, it releases it immediately and runs the scan safely.

### Process Synchronization Flow
<div class="admonition architecture" markdown>
<p class="admonition-title">A non-blocking flock lets the watcher skip instead of block</p>

**Scenario 1 — ingestion active (non-blocking skip).** The bulk ingester
acquires the POSIX flock (`.lock`) exclusively. The background watcher
tries a non-blocking acquisition (`timeout=0`), gets a timeout exception
because the lock is already held, and gracefully skips the iteration —
it never blocks.

**Scenario 2 — ingestion complete (scan execution).** The bulk ingester
releases the flock (fd closed, file remains). The watcher's non-blocking
acquisition now succeeds; it releases the lock immediately and proceeds
to scan and update repository metadata in LadybugDB.
</div>

---

## 2. Native C++ Handle Destruction & Connection Cleanup

### The Destruction Sequence Hazard (Before)
LadybugDB uses native C++ bindings for high-performance SQLite operations and HNSW vector computations. The C++ `ladybug.Connection` instances rely on active references to `ladybug.Database` handles.

If the python interpreter performs garbage collection out-of-order, or if Python attempts to destroy the parent `Database` handle while the children `Connection` handles are still active, it results in native C++ null-pointer dereferences or double-free segmentation faults.

### Explicit Cleanup & Reference Ordering (After)
We enforce a strict connection cleanup sequence in `LadybugBackend.close()` to ensure child handles are entirely freed and garbage-collected before unreferencing the parent database handles.

<div class="admonition architecture" markdown>
<p class="admonition-title">Flaw vs. fix: destruction order determines segfault or safe termination</p>

**Flaw — out-of-order destruction (segfault).** Unreferencing the DB and
Connections together, then calling `gc.collect()` at a random point, can
destroy the DB handle first while an active Connection handle is still
unreferenced afterward — crashing with a C++ segfault or double free.

**Correct — strict cleanup sequence (stable).** `close()` runs five steps
in order: (1) close and unreference the Connection
(`self.conn = None`); (2) force `gc.collect()` to clear Connection
handles from C++ memory; (3) unreference the Database
(`self.db = None`); (4) force `gc.collect()` again to clear the Database
handle from C++ memory; (5) safe termination without dangling pointers.
</div>

### Destruction & Cleanup Sequence Details
<div class="admonition architecture" markdown>
<p class="admonition-title">Destruction and cleanup sequence, in detail</p>

`LadybugBackend` closes the `ladybug.Connection` (C++) explicitly, sets
`self.conn = None`, and invokes `gc.collect()` — fully destroying the
Connection handle in the C++ layer. Only then does it set
`self.db = None` and invoke `gc.collect()` again — fully destroying the
`ladybug.Database` (C++) handle in the C++ layer.
</div>

---

## 3. Nested Metadata Scanning & Domain Routing

Bulk ingestion handles 65 open-source repositories categorized by domain slugs. The scanner uses direct domain matching to assign repositories to the correct Concept Schemes and folder hierarchies:

| Domain Category | Category Path Pattern | Target Repositories (Examples) |
|---|---|---|
| `agent-frameworks` | `agent-frameworks/` | `pydantic-ai`, `crewai`, `langgraph` |
| `enterprise-ai-infra` | `enterprise-ai-infra/` | `caddy`, `keycloak`, `twenty`, `mattermost` |
| `memory-rag-kg` | `memory-rag-kg/` | `ladybugdb`, `chromadb`, `milvus` |
| `quant-trading` | `quant-trading/` | `crypto-trader`, `qlib`, `freqtrade` |

<div class="admonition architecture" markdown>
<p class="admonition-title">Nested domain match, or fallback by name — both assign to one hierarchy</p>

A repository path input is checked for whether it is nested under a
domain. If yes, the scanner extracts the slug, repo, and nested
subfolders; if no, it falls back to classification by repository name.
Either path assigns the repository to the active Concept Scheme and SKOS
concept hierarchy.
</div>
