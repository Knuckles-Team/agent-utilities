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
    """Return the aggregate count, or 0 when the backend returns no rows.

    A genuine backend/query failure is NOT swallowed here (D-28): it
    propagates to the caller, which is the only place that can tell "this
    repo really has zero symbols" apart from "the query failed and we don't
    know". Collapsing both to 0 previously made a transient read error
    indistinguishable from an honest zero.
    """
    rows = backend.execute(cypher, params)
    for row in rows or []:
        if isinstance(row, dict):
            for v in row.values():
                try:
                    return int(v)
                except (TypeError, ValueError):
                    continue
    return 0


def repo_symbol_counts(
    backend: Any, repos: list[str]
) -> tuple[dict[str, int], dict[str, str]]:
    """Per-repo ``:Code`` symbol count (0 = not ingested), via the live backend.

    Returns ``(counts, errors)``: ``counts`` holds a real integer (including a
    genuine ``0``) only for repos whose query succeeded; a repo whose query
    raised is OMITTED from ``counts`` and instead keyed in ``errors`` with the
    exception's ``type: message`` text (D-28 — a query failure must never be
    reported as the same "0" a genuinely un-ingested repo gets, since
    downstream verdicts such as :func:`assess_coverage` and the signed
    hydration manifest's ``_fuse_verdict`` treat "0" as "never ingested").
    """
    counts: dict[str, int] = {}
    errors: dict[str, str] = {}
    for repo in repos:
        needle = f"/agent-packages/{repo}/"
        try:
            counts[repo] = _count(
                backend,
                "MATCH (c:Code) WHERE c.file_path CONTAINS $needle RETURN count(c) AS n",
                {"needle": needle},
            )
        except Exception as exc:  # noqa: BLE001 — classified per-repo, never silently 0
            errors[repo] = f"{type(exc).__name__}: {exc}"
    return counts, errors


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
    errors: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Compare expected repos vs ingested symbol counts + freshness watermarks.

    ``freshness`` maps a source_uri → ISO last-sync timestamp (DeltaManifest);
    a repo whose newest matching watermark is older than ``sla_days`` is stale.

    ``errors`` (D-28) is the per-repo failure map :func:`repo_symbol_counts`
    returns alongside its counts. A repo present in ``errors`` is reported
    ONLY in the ``errors`` list — never in ``missing`` — so a transient query
    failure is never mis-reported as "genuinely not ingested".
    """
    now = now or datetime.now(UTC)
    freshness = freshness or {}
    errors = errors or {}
    errored = [r for r in repos if r in errors]
    covered = [r for r in repos if r not in errors and counts.get(r, 0) > 0]
    missing = [r for r in repos if r not in errors and counts.get(r, 0) <= 0]

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
        "errors": errored,
        "coverage_pct": round(100.0 * len(covered) / total, 1) if total else 0.0,
        "total_symbols": total_symbols,
        "sla_days": sla_days,
    }
