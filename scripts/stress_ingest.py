#!/usr/bin/env python3
"""KG ingestion stress-test harness (CONCEPT:KG-2.7 / KG-2.8).

Drives the **unified IngestionEngine** (content-typed adaptors) across every
artifact category — config, prompts, codebases (+ ``.specify`` specs), documents,
chats, mcp servers (with live tool discovery), skills — and records per-category
+ total performance metrics (wall time, peak RSS, avg/peak CPU, pggraph durable
disk growth, node-count delta). Within-category work runs concurrently, and an
incremental ``progress.json`` is written so a watcher can report live %.

Runs against the tiered backend (epistemic-graph L1 + pggraph L3). Example::

    GRAPH_BACKEND=tiered \\
    GRAPH_DB_URI=postgresql://postgres:postgres@pggraph.arpa:5432/pggraph \\
    KG_BULK_INGEST=1 \\
    .venv/bin/python scripts/stress_ingest.py --limit 3        # quick validation
    .venv/bin/python scripts/stress_ingest.py                  # full run

Kept operational tool, not scratch.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

WORKSPACE = Path("/home/apps/workspace")
AGENT_PACKAGES = WORKSPACE / "agent-packages"
OSS = WORKSPACE / "open-source-libraries"
SKILLS_ROOT = AGENT_PACKAGES / "skills" / "universal-skills" / "universal_skills"
XDG_CONFIG = Path.home() / ".config" / "agent-utilities" / "config.json"
MCP_CONFIG = WORKSPACE / "mcp_config.json"

# All documents are ingested uniformly (single light document path). Add roots
# here — scholarx papers, workspace prompt/plan docs, any document corpus.
DOCUMENT_ROOTS = [
    Path.home() / ".local" / "share" / "scholarx" / "papers",
    Path.home() / ".scholarx" / "papers",
    WORKSPACE / "prompts",
]
DOCUMENT_EXTS = {".md", ".txt", ".rst", ".pdf"}

# Per-category concurrency (tuned for the beefy host but mindful of the shared
# epistemic-graph daemon for codebases and vLLM for LLM-bound documents).
CONCURRENCY = {
    "config": 1,
    "prompts": 8,
    "skills": 12,
    "documents": 4,
    "codebases": 3,
    "mcp_servers": 1,  # one call; internal discovery_concurrency fans out
    "chats": 1,  # one call; auto-discovers all logs
}

_progress_path: Path | None = None
_progress_lock = threading.Lock()
_progress: dict[str, Any] = {}


def _write_progress(**kw: Any) -> None:
    if _progress_path is None:
        return
    with _progress_lock:
        _progress.update(kw)
        _progress["updated"] = datetime.now(UTC).isoformat()
        try:
            _progress_path.write_text(json.dumps(_progress, indent=2))
        except OSError:
            pass


# ── Resource sampling ──────────────────────────────────────────────────────
class ResourceSampler:
    def __init__(self, interval: float = 0.5) -> None:
        self._interval = interval
        self._proc = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.cpu_samples: list[float] = []
        self.rss_samples: list[int] = []

    def _run(self) -> None:
        self._proc.cpu_percent(None)
        while not self._stop.wait(self._interval):
            try:
                self.cpu_samples.append(self._proc.cpu_percent(None))
                rss = self._proc.memory_info().rss
                for c in self._proc.children(recursive=True):
                    try:
                        rss += c.memory_info().rss
                    except psutil.Error:
                        pass
                self.rss_samples.append(rss)
            except psutil.Error:
                pass

    def __enter__(self) -> ResourceSampler:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def summary(self) -> dict[str, float]:
        return {
            "cpu_avg_pct": round(sum(self.cpu_samples) / len(self.cpu_samples), 1)
            if self.cpu_samples
            else 0.0,
            "cpu_peak_pct": round(max(self.cpu_samples), 1)
            if self.cpu_samples
            else 0.0,
            "rss_peak_mb": round(max(self.rss_samples) / 1e6, 1)
            if self.rss_samples
            else 0.0,
        }


def _pg_db_size() -> int:
    try:
        import psycopg

        dsn = os.environ.get("GRAPH_DB_URI", "")
        if not dsn:
            return 0
        with psycopg.connect(dsn, connect_timeout=5, autocommit=True) as c:
            with c.cursor() as cur:
                cur.execute("select pg_database_size(current_database())")
                return int(cur.fetchone()[0])
    except Exception:
        return 0


def _node_count(engine: Any) -> int:
    try:
        res = engine.query_cypher("MATCH (n) RETURN count(n) as c")
        return int(res[0]["c"]) if res else 0
    except Exception:
        try:
            return len(engine.graph._get_all_nodes())
        except Exception:
            return 0


# ── Category → manifest builders ────────────────────────────────────────────
def _manifests(cat: str, limit: int | None) -> list[dict[str, Any]]:
    """Return a list of {content_type, source_uri, metadata} dicts per category."""

    def cap(xs):
        return xs[:limit] if limit else xs

    if cat == "config":
        return (
            [{"content_type": "config", "source_uri": str(XDG_CONFIG)}]
            if XDG_CONFIG.exists()
            else []
        )
    if cat == "prompts":
        d = AGENT_PACKAGES / "agent-utilities" / "agent_utilities" / "prompts"
        return [
            {"content_type": "prompt", "source_uri": str(p)}
            for p in cap(sorted(d.glob("*.json")))
        ]
    if cat == "skills":
        return [
            {"content_type": "skill", "source_uri": str(p.parent)}
            for p in cap(sorted(SKILLS_ROOT.rglob("SKILL.md")))
        ]
    if cat == "documents":
        # Unified: every document corpus (scholarx papers, workspace prompt/plan
        # docs, …) discovered the same way and ingested via the light document path.
        seen: set[str] = set()
        files: list[Path] = []
        for root in DOCUMENT_ROOTS:
            if not root.exists():
                continue
            for p in sorted(root.rglob("*")):
                if (
                    p.is_file()
                    and p.suffix.lower() in DOCUMENT_EXTS
                    and str(p) not in seen
                ):
                    seen.add(str(p))
                    files.append(p)
        return [{"content_type": "document", "source_uri": str(p)} for p in cap(files)]
    if cat == "codebases":
        repos: list[Path] = []
        for base in (AGENT_PACKAGES, OSS):
            if base.exists():
                repos += [
                    p
                    for p in sorted(base.iterdir())
                    if p.is_dir() and not p.name.startswith(".")
                ]
        return [
            {
                "content_type": "codebase",
                "source_uri": str(p),
                "metadata": {"features": False},
            }
            for p in cap(repos)
        ]
    if cat == "mcp_servers":
        return (
            [
                {
                    "content_type": "mcp_server",
                    "source_uri": str(MCP_CONFIG),
                    "metadata": {"discovery_concurrency": 16, "discovery_timeout": 20},
                }
            ]
            if MCP_CONFIG.exists()
            else []
        )
    if cat == "chats":
        return [{"content_type": "conversation", "source_uri": "chats"}]
    return []


# ── Concurrent per-category ingest ──────────────────────────────────────────
async def _ingest_category(
    ing: Any, cat: str, manifests: list[dict[str, Any]], concurrency: int
) -> dict[str, Any]:
    from agent_utilities.knowledge_graph.ingestion.engine import IngestionManifest

    total = len(manifests)
    done = 0
    statuses: dict[str, int] = {}
    sem = asyncio.Semaphore(max(1, concurrency))
    t0 = time.time()

    async def _one(m: dict[str, Any]) -> None:
        nonlocal done
        async with sem:
            try:
                r = await ing.ingest(IngestionManifest(**m))
                st = r.status
            except Exception as e:  # noqa: BLE001
                st = f"error:{type(e).__name__}"
            statuses[st] = statuses.get(st, 0) + 1
            done += 1
            _write_progress(
                category=cat,
                done=done,
                total=total,
                pct=round(100 * done / total, 1) if total else 100.0,
                elapsed=round(time.time() - t0, 1),
                statuses=dict(statuses),
            )

    await asyncio.gather(*[_one(m) for m in manifests])
    return {"submitted": total, "statuses": statuses}


async def _amain(cats: list[str], limit: int | None, rdir: Path) -> int:
    from agent_utilities.knowledge_graph.ingestion.engine import IngestionEngine
    from agent_utilities.mcp.kg_server import _get_engine

    engine = _get_engine()
    if engine is None:
        print("FATAL: engine not available")
        return 1
    ing = IngestionEngine(kg_engine=engine)
    backend = type(getattr(engine, "backend", None)).__name__
    print(f"engine ready (backend={backend}); categories={cats} limit={limit}")
    _write_progress(backend=backend, categories=cats, limit=limit, phase="running")

    results: dict[str, Any] = {}
    grand0 = time.time()
    for cat in cats:
        manifests = _manifests(cat, limit)
        if not manifests:
            print(f"  [{cat}] no sources, skipping")
            continue
        n0, sz0, t0 = _node_count(engine), _pg_db_size(), time.time()
        _write_progress(category=cat, done=0, total=len(manifests), pct=0.0)
        with ResourceSampler() as samp:
            try:
                detail = await _ingest_category(
                    ing, cat, manifests, CONCURRENCY.get(cat, 4)
                )
            except Exception as e:  # noqa: BLE001
                detail = {"error": str(e)[:200]}
        dt = time.time() - t0
        n1, sz1 = _node_count(engine), _pg_db_size()
        results[cat] = {
            **detail,
            "seconds": round(dt, 1),
            "nodes_delta": n1 - n0,
            "pggraph_growth_mb": round((sz1 - sz0) / 1e6, 2),
            **samp.summary(),
        }
        print(f"  [{cat}] {results[cat]}")
        (rdir / "metrics.json").write_text(
            json.dumps({"partial": True, "categories": results}, indent=2)
        )

    # Durability backstop: reflect L1 → pggraph.
    try:
        if hasattr(engine.backend, "reconcile_to_durable"):
            _write_progress(phase="reconcile")
            results["_reconcile"] = engine.backend.reconcile_to_durable()
    except Exception as e:  # noqa: BLE001
        results["_reconcile_error"] = str(e)[:200]

    total = {
        "total_seconds": round(time.time() - grand0, 1),
        "total_nodes": _node_count(engine),
        "pggraph_size_mb": round(_pg_db_size() / 1e6, 2),
        "backend": backend,
        "limit": limit,
    }
    report = {
        "generated": datetime.now(UTC).isoformat(),
        "total": total,
        "categories": results,
    }
    (rdir / "metrics.json").write_text(json.dumps(report, indent=2))
    _write_progress(phase="done", total_summary=total)
    print(f"\n=== TOTAL: {total} ===\nreport: {rdir}/metrics.json")
    return 0


def main() -> int:
    global _progress_path
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", default="all")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--report-dir", default=None)
    args = ap.parse_args()

    all_cats = [
        "config",
        "prompts",
        "mcp_servers",
        "skills",
        "documents",
        "chats",
        "codebases",
    ]
    cats = all_cats if args.categories == "all" else args.categories.split(",")

    rdir = (
        Path(args.report_dir)
        if args.report_dir
        else WORKSPACE
        / "reports"
        / f"ingest-stress-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
    )
    rdir.mkdir(parents=True, exist_ok=True)
    _progress_path = rdir / "progress.json"
    print(f"progress: {_progress_path}")
    return asyncio.run(_amain(cats, args.limit, rdir))


if __name__ == "__main__":
    raise SystemExit(main())
