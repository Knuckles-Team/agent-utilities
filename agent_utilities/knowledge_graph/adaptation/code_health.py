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


def _find_analyzer() -> Path | None:
    spec = importlib.util.find_spec("universal_skills")
    for loc in getattr(spec, "submodule_search_locations", []) or []:
        cand = Path(loc) / "core" / "code-enhancer" / "scripts" / "analyze_liveness.py"
        if cand.exists():
            return cand
    return None


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

    root = repos_root or _DEFAULT_ROOT
    if not root.is_dir():
        return {"status": "skipped", "reason": "repos_root_missing"}
    repos = sorted(
        p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    if limit:
        repos = repos[:limit]

    swept: list[tuple[str, int]] = []
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
        try:
            engine.add_node(  # type: ignore[attr-defined]
                f"code_health:{repo.name}",
                type="CodeHealthReport",
                repo=repo.name,
                score=report.get("score"),
                grade=report.get("grade"),
                counts=json.dumps(counts),
                ts=time.time(),
            )
        except Exception:  # noqa: BLE001
            pass  # recording is best-effort; the sweep value is the detection itself
        swept.append((repo.name, int(report.get("score", 0))))

    swept.sort(key=lambda x: x[1])
    logger.info("code_health sweep: %d repo(s); lowest=%s", len(swept), swept[:3])
    return {"status": "ok", "swept": len(swept), "lowest": swept[:5]}
