#!/usr/bin/python
from __future__ import annotations

"""Ingestion coverage + freshness SLA assessment (CONCEPT:AU-OS.deployment.flagging-repos).

Backs the ``agent-utilities doctor`` ingestion-coverage check: compare the repos
the platform is *expected* to know (the ``agent-packages`` subtree of
``workspace.yml``) against what is actually in the KG (repos with >=1 ``:Code``
symbol) and how fresh each is (the :class:`DeltaManifest` last-sync watermark).
Missing or stale repos then surface as a doctor warning instead of silently
degrading every KG code query to grep — the freshness guarantee GAP 1 of the
codebase-context-via-KG plan requires.

Pure functions (no engine) so the assessment is unit-tested directly; the doctor
check wires them to the live backend + manifest.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: Default freshness SLA — a repo not delta-synced within this window is "stale".
DEFAULT_SLA_DAYS = 7


def find_workspace_manifest(start: Path | None = None) -> Path | None:
    """Walk up from ``start`` (default: this package) to find ``workspace.yml``."""
    here = start or Path(__file__).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / "workspace.yml"
        if candidate.is_file():
            return candidate
    return None


def _repo_name(url: str) -> str:
    name = str(url).rstrip("/").rsplit("/", 1)[-1]
    return name[:-4] if name.endswith(".git") else name


def _flatten_repos(node: Any, out: set[str]) -> None:
    """Recursively collect every ``repositories[*].url`` basename under a node."""
    if not isinstance(node, dict):
        return
    for repo in node.get("repositories", []) or []:
        url = repo.get("url") if isinstance(repo, dict) else repo
        if url:
            out.add(_repo_name(url))
    for child in (node.get("subdirectories", {}) or {}).values():
        _flatten_repos(child, out)


def enumerate_agent_packages_repos(manifest_path: Path) -> list[str]:
    """Repo names under ``subdirectories.agent-packages`` of ``workspace.yml``.

    Returns the sorted set of repo basenames (``agent-utilities``,
    ``servicenow-api`` …) the platform's code KG is expected to cover.
    """
    import yaml

    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    ap = (data.get("subdirectories", {}) or {}).get("agent-packages", {})
    out: set[str] = set()
    _flatten_repos(ap, out)
    return sorted(out)


def _count(backend: Any, cypher: str, params: dict[str, Any]) -> int:
    try:
        rows = backend.execute(cypher, params)
    except Exception:
        return 0
    for row in rows or []:
        if isinstance(row, dict):
            for v in row.values():
                try:
                    return int(v)
                except (TypeError, ValueError):
                    continue
    return 0


def repo_symbol_counts(backend: Any, repos: list[str]) -> dict[str, int]:
    """Per-repo ``:Code`` symbol count (0 = not ingested), via the live backend."""
    counts: dict[str, int] = {}
    for repo in repos:
        needle = f"/agent-packages/{repo}/"
        counts[repo] = _count(
            backend,
            "MATCH (c:Code) WHERE c.file_path CONTAINS $needle RETURN count(c) AS n",
            {"needle": needle},
        )
    return counts


def _age_days(updated_at: str, now: datetime) -> float | None:
    if not updated_at:
        return None
    try:
        ts = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        return (now - ts).total_seconds() / 86400.0
    except (ValueError, TypeError):
        return None


def assess_coverage(
    repos: list[str],
    counts: dict[str, int],
    freshness: dict[str, str] | None = None,
    *,
    sla_days: int = DEFAULT_SLA_DAYS,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Compare expected repos vs ingested symbol counts + freshness watermarks.

    ``freshness`` maps a source_uri → ISO last-sync timestamp (DeltaManifest);
    a repo whose newest matching watermark is older than ``sla_days`` is stale.
    """
    now = now or datetime.now(UTC)
    freshness = freshness or {}
    covered = [r for r in repos if counts.get(r, 0) > 0]
    missing = [r for r in repos if counts.get(r, 0) <= 0]

    # Newest watermark per repo (match the repo name inside the source_uri key).
    stale: list[dict[str, Any]] = []
    for repo in covered:
        ages = [
            age
            for uri, ts in freshness.items()
            if repo in uri
            for age in [_age_days(ts, now)]
            if age is not None
        ]
        if ages:
            youngest = min(ages)
            if youngest > sla_days:
                stale.append({"repo": repo, "age_days": round(youngest, 1)})

    total = len(repos)
    total_symbols = sum(counts.values())
    return {
        "total": total,
        "covered": len(covered),
        "missing": missing,
        "stale": stale,
        "coverage_pct": round(100.0 * len(covered) / total, 1) if total else 0.0,
        "total_symbols": total_symbols,
        "sla_days": sla_days,
    }
