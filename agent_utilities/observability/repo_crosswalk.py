#!/usr/bin/python
from __future__ import annotations

"""Repo-node ``owl:sameAs`` crosswalk resolver — gap #1's identity reconciliation
(``reports/autonomous-sdlc-loop-design.md`` §4.3).

CONCEPT:AU-KG.enrichment.ops-causal-graph. The fleet ingests the SAME git
repository under THREE different node identities, and nothing links them:

* ``portainer-agent`` writes a URL-keyed ``:Repository`` (id ``git:repo:<host>/<path>``,
  ``url`` prop) for a GitOps stack's ``RepositoryURL``;
* ``github-agent`` writes a numeric-id ``:Repository`` (id ``github:repository:<id>``,
  ``htmlUrl``/``fullName`` props);
* ``gitlab-api`` writes a numeric-id ``:Project`` (id ``gitlab:project:<id>``,
  ``web_url``/``path_with_namespace`` props).

So a deployed ``:Service`` traced back through the §4 ``:builtFrom`` edge lands on the
Portainer URL-node, NOT the numeric-id node the code ingestors (call graph, MRs, CI,
issues) populated. This resolver reconciles them: it reads every ``:Repository`` /
``:Project`` node, normalizes each one's clone URL to a canonical ``host/owner/name``
key, and emits ``owl:sameAs`` (symmetric) + ``:aliasOf`` edges between the nodes that
share a key but differ in id-namespace — so a lifecycle walk unifies the deployed
service with the same repo the code ingestors created.

Best-effort + engine-guarded (every entry point no-ops with no reachable engine) and
idempotent (``owl:sameAs``/``:aliasOf`` MERGE on ``(source, target, type)``). Run as
``python -m agent_utilities.observability.repo_crosswalk`` (a periodic reconcile pass,
same primitive as the incident-correlation CronJob).
"""

import logging
import re
from typing import Any

from agent_utilities.observability import health_ingest

logger = logging.getLogger("agent_utilities.observability.repo_crosswalk")

_SOURCE = "agent-utilities-repo-crosswalk"

# The node labels that carry a git-repository identity, and the props (in
# priority order) each producer stamps a clone/web URL onto.
_REPO_LABELS = ("Repository", "Project")
_URL_PROPS = ("url", "htmlUrl", "webUrl", "web_url", "cloneUrl", "clone_url", "sshUrl")

# id-namespace prefix -> a stable short tag, so we only cross-link nodes from
# DIFFERENT producers (never two github ids that happen to share a URL).
_NAMESPACE_RE = re.compile(r"^([a-z]+):")


def normalize_clone_url(url: str) -> str:
    """Normalize any git remote/web URL to a canonical ``host/owner/name`` key.

    Strips the scheme, ``git@`` / userinfo, a trailing ``.git``/slash, a leading
    ``www.``, and lowercases — so ``https://github.com/O/N.git``,
    ``git@github.com:O/N.git``, and ``https://www.github.com/O/N/`` all collapse to
    ``github.com/o/n``. Returns ``""`` for an unusable value.
    """
    if not url:
        return ""
    u = str(url).strip()
    # scp-style git@host:owner/name -> host/owner/name
    scp = re.match(r"^[\w.+-]+@([^:]+):(.+)$", u)
    if scp:
        u = f"{scp.group(1)}/{scp.group(2)}"
    else:
        u = re.sub(r"^[a-zA-Z][\w+.-]*://", "", u)  # drop scheme
        u = re.sub(r"^[^/@]+@", "", u)  # drop userinfo
    u = u.split("?", 1)[0].split("#", 1)[0]  # drop query/fragment
    u = re.sub(r"\.git$", "", u)
    u = u.strip("/")
    u = re.sub(r"^www\.", "", u, flags=re.IGNORECASE)
    return u.lower()


def _url_of(props: dict[str, Any]) -> str:
    for key in _URL_PROPS:
        val = props.get(key)
        if val:
            return str(val)
    return ""


def _namespace(node_id: str) -> str:
    m = _NAMESPACE_RE.match(node_id or "")
    return m.group(1) if m else ""


def _collect_repo_nodes(engine: Any) -> list[tuple[str, dict[str, Any]]]:
    nodes: list[tuple[str, dict[str, Any]]] = []
    for label in _REPO_LABELS:
        try:
            rows = engine.get_nodes_by_label(label, 0) or []
        except Exception as e:  # noqa: BLE001 — read is best-effort
            logger.debug("repo crosswalk: get_nodes_by_label(%s) failed: %s", label, e)
            continue
        for node_id, props in rows:
            if isinstance(props, dict) and node_id:
                nodes.append((str(node_id), props))
    return nodes


def resolve_crosswalk(*, engine: Any | None = None) -> list[dict[str, Any]]:
    """Compute the repo-node crosswalk without writing anything.

    Groups every ``:Repository``/``:Project`` node by its normalized clone URL and
    returns one crosswalk dict per URL that has ≥2 nodes from DIFFERENT id
    namespaces::

        {"url": "github.com/o/n", "canonical": <numeric-id node>,
         "aliases": [<url-keyed node>, ...], "members": [all ids]}

    The *canonical* node is the code-ingestor's numeric-id node (``github:``/
    ``gitlab:``) when present (that is where the call graph / MRs / CI live);
    otherwise the first member. Best-effort: ``[]`` with no reachable engine.
    """
    eng = engine or health_ingest._engine()
    if eng is None:
        return []
    by_url: dict[str, list[tuple[str, str]]] = {}
    for node_id, props in _collect_repo_nodes(eng):
        key = normalize_clone_url(_url_of(props))
        if not key:
            continue
        by_url.setdefault(key, []).append((node_id, _namespace(node_id)))

    out: list[dict[str, Any]] = []
    for url, members in by_url.items():
        namespaces = {ns for _id, ns in members}
        if len(members) < 2 or len(namespaces) < 2:
            continue  # nothing to reconcile — one node, or all same producer
        ids = [node_id for node_id, _ns in members]
        canonical = next(
            (nid for nid, ns in members if ns in ("github", "gitlab")), ids[0]
        )
        aliases = [nid for nid in ids if nid != canonical]
        out.append(
            {
                "url": url,
                "canonical": canonical,
                "aliases": aliases,
                "members": ids,
            }
        )
    out.sort(key=lambda c: c["url"])
    return out


def _write_crosswalk_edges(crosswalks: list[dict[str, Any]]) -> dict[str, int] | None:
    """MERGE the ``owl:sameAs`` (symmetric) + ``:aliasOf`` edges for each resolved
    crosswalk. Passes minimal id-only node stubs so the edge endpoints MERGE onto
    the existing repo nodes without clobbering their props."""
    from agent_utilities.knowledge_graph.memory.native_ingest import ingest_entities

    entities: dict[str, dict[str, Any]] = {}
    relationships: list[dict[str, Any]] = []
    for cw in crosswalks:
        canonical = cw["canonical"]
        entities.setdefault(canonical, {"id": canonical, "type": "Repository"})
        for alias in cw["aliases"]:
            entities.setdefault(alias, {"id": alias, "type": "Repository"})
            # alias -[:aliasOf/owl:sameAs]-> canonical (the numeric-id node), both
            # directions of sameAs so a walk from either side unifies.
            relationships.append(
                {"source": alias, "target": canonical, "type": "aliasOf"}
            )
            relationships.append(
                {"source": alias, "target": canonical, "type": "sameAs"}
            )
            relationships.append(
                {"source": canonical, "target": alias, "type": "sameAs"}
            )
    if not relationships:
        return None
    return ingest_entities(
        list(entities.values()),
        relationships,
        source=_SOURCE,
        domain="sdlc",
    )


def run_repo_crosswalk(*, write: bool = True) -> dict[str, Any]:
    """One resolve → (optionally) write pass — the ``python -m`` / CronJob entry.

    Resolves the crosswalk (:func:`resolve_crosswalk`) and, when ``write`` is set
    (default — these are idempotent derived-fact edges, not a high-stakes action),
    MERGEs the ``owl:sameAs``/``:aliasOf`` edges. Best-effort throughout: with no
    reachable engine this returns an all-zero summary rather than raising.
    """
    crosswalks = resolve_crosswalk()
    result = _write_crosswalk_edges(crosswalks) if (write and crosswalks) else None
    return {
        "reconciled": len(crosswalks),
        "aliases": sum(len(c["aliases"]) for c in crosswalks),
        "edges_written": (result or {}).get("edges", 0),
        "crosswalks": crosswalks,
    }


def main() -> None:
    """CLI (``python -m agent_utilities.observability.repo_crosswalk``): one
    resolve→write pass; prints a JSON summary."""
    import json

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    print(json.dumps(run_repo_crosswalk(), default=str, indent=2))


if __name__ == "__main__":
    main()
