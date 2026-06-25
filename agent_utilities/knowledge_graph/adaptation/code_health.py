"""CONCEPT:CE-038 — periodic code-health (liveness) sweep across workspace repos.

A bounded, **LLM-free** maintenance pass: locates the code-enhancer liveness
analyzer (the deterministic dead-pathway detector) and runs it over each repo
sequentially, recording a ``CodeHealthReport`` node per repo so dead-pathway trends
are queryable. Detection only — fix proposals stay on-demand (the golden loop's
propose-only discipline). Driven by the ``code_health`` maintenance tick (opt-in via
``KG_CODE_HEALTH``); this is exactly the "run code-enhancer across all repos every
few hours" the operator asked for, with detection on the daemon and enhancement
left to a reviewed agent run.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_ROOT = Path("/home/apps/workspace/agent-packages")
_PER_REPO_TIMEOUT_S = 180
# Per-repo baseline snapshots so each sweep can report *new vs. resolved* dead
# pathways instead of a bare score — a regression is what matters, not legacy debt.
# Stored as ``:CodeHealthBaseline`` nodes on the durable engine authority when one
# is reachable (CONCEPT:KG-2.209); the local file cache is the zero-infra ``tiny``
# fallback only.
_BASELINE_DIR = Path.home() / ".cache" / "agent_utilities" / "code_health_baselines"
_BASELINE_LABEL = "CodeHealthBaseline"


def _baseline_backend(engine: Any) -> Any:
    """The durable graph backend to persist baselines on, or ``None`` for the file
    fallback. Reuses the shared dual-mode predicate (CONCEPT:KG-2.209)."""
    from ..backends.base import is_durable_backend

    backend = getattr(engine, "backend", None) or engine
    return backend if is_durable_backend(backend) else None


def _load_baseline_snapshot(backend: Any, repo: str) -> dict[str, Any] | None:
    """Read the prior baseline snapshot for ``repo`` (engine node or file)."""
    if backend is not None:
        try:
            rows = backend.execute(
                f"MATCH (b:{_BASELINE_LABEL} {{repo: $repo}}) RETURN b",
                {"repo": repo},
            )
            for row in rows if isinstance(rows, list) else []:
                node = row.get("b") if isinstance(row, dict) else None
                if isinstance(node, dict) and node.get("snapshot_json"):
                    return json.loads(node["snapshot_json"])
        except Exception as e:  # noqa: BLE001 - baseline read best-effort
            logger.debug("code_health: baseline read failed for %s: %s", repo, e)
        return None
    cache = _BASELINE_DIR / f"{repo}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    return None


def _save_baseline_snapshot(
    backend: Any, repo: str, snapshot: dict[str, Any]
) -> None:
    """Persist the refreshed baseline snapshot for ``repo`` (engine node or file)."""
    if backend is not None:
        backend.execute(
            f"MERGE (b:{_BASELINE_LABEL} {{repo: $repo}}) SET "
            "b.snapshot_json = $snapshot, b.updated_at = $ts",
            {
                "repo": repo,
                "snapshot": json.dumps(snapshot),
                "ts": time.time(),
            },
        )
        return
    _BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    (_BASELINE_DIR / f"{repo}.json").write_text(json.dumps(snapshot, indent=2))


def _find_analyzer() -> Path | None:
    spec = importlib.util.find_spec("universal_skills")
    for loc in getattr(spec, "submodule_search_locations", []) or []:
        cand = Path(loc) / "core" / "code-enhancer" / "scripts" / "analyze_liveness.py"
        if cand.exists():
            return cand
    return None


def _load_baseline_module(analyzer: Path) -> Any | None:
    """Import the code-enhancer ``analyze_baseline`` module that ships next to the
    liveness analyzer (CE-039). Returns None if unavailable so deltas degrade gracefully."""
    mod_path = analyzer.parent / "analyze_baseline.py"
    if not mod_path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location("ce_analyze_baseline", mod_path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:  # noqa: BLE001
        return None


def _baseline_delta(
    baseline_mod: Any, repo: str, report: dict[str, Any], backend: Any = None
) -> dict[str, Any]:
    """Diff this run's findings against the persisted per-repo baseline, then refresh
    it. Baselines live on the durable engine (``backend``) when available, else the
    local file cache. Returns ``{new, fixed, new_debt_score}`` (empty if unavailable)."""
    if baseline_mod is None:
        return {}
    delta: dict[str, Any] = {}
    try:
        prior = _load_baseline_snapshot(backend, repo)
        if prior is not None:
            d = baseline_mod.diff(report, prior)
            delta = {
                "new": d["counts"]["new"],
                "fixed": d["counts"]["fixed"],
                "new_debt_score": d["new_debt_score"],
            }
        _save_baseline_snapshot(backend, repo, baseline_mod.snapshot(report))
    except Exception as e:  # noqa: BLE001
        logger.debug("code_health: baseline delta failed for %s: %s", repo, e)
    return delta


def run_code_health_sweep(
    engine: Any,
    repos_root: Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run the liveness analyzer over every Python repo under ``repos_root``,
    sequentially, recording a ``CodeHealthReport`` node per repo. Returns a summary
    with the lowest-scoring repos. Never raises — a per-repo failure is logged and
    skipped so one bad repo cannot stall the daemon sweep."""
    analyzer = _find_analyzer()
    if analyzer is None:
        logger.info(
            "code_health: code-enhancer skill (universal_skills) not installed; skipping"
        )
        return {"status": "skipped", "reason": "analyzer_unavailable"}

    baseline_mod = _load_baseline_module(analyzer)
    baseline_backend = _baseline_backend(engine)
    root = repos_root or _DEFAULT_ROOT
    if not root.is_dir():
        return {"status": "skipped", "reason": "repos_root_missing"}
    repos = sorted(
        p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    if limit:
        repos = repos[:limit]

    swept: list[tuple[str, int]] = []
    regressions: list[tuple[str, int]] = []
    for repo in repos:
        if not next(repo.rglob("__init__.py"), None):  # python repos only
            continue
        try:
            res = subprocess.run(
                [sys.executable, str(analyzer), str(repo)],
                capture_output=True,
                text=True,
                timeout=_PER_REPO_TIMEOUT_S,
            )
            if res.returncode not in (0, 1):
                logger.warning(
                    "code_health: %s analyzer rc=%s", repo.name, res.returncode
                )
                continue
            report = json.loads(res.stdout)
        except Exception as e:  # noqa: BLE001
            logger.warning("code_health: %s sweep failed: %s", repo.name, e)
            continue

        counts = report.get("counts", {})
        delta = _baseline_delta(baseline_mod, repo.name, report, baseline_backend)
        if delta.get("new"):
            regressions.append((repo.name, delta["new"]))
        try:
            engine.add_node(  # type: ignore[attr-defined]
                f"code_health:{repo.name}",
                type="CodeHealthReport",
                repo=repo.name,
                score=report.get("score"),
                grade=report.get("grade"),
                counts=json.dumps(counts),
                new_findings=delta.get("new"),
                fixed_findings=delta.get("fixed"),
                new_debt_score=delta.get("new_debt_score"),
                ts=time.time(),
            )
        except Exception:  # noqa: BLE001
            pass  # recording is best-effort; the sweep value is the detection itself
        swept.append((repo.name, int(report.get("score", 0))))

    swept.sort(key=lambda x: x[1])
    regressions.sort(key=lambda x: x[1], reverse=True)
    logger.info(
        "code_health sweep: %d repo(s); lowest=%s; regressions=%s",
        len(swept),
        swept[:3],
        regressions[:3],
    )
    return {
        "status": "ok",
        "swept": len(swept),
        "lowest": swept[:5],
        "regressions": regressions[:5],
    }
