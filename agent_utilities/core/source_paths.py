#!/usr/bin/python
from __future__ import annotations

"""Source-path canonicalization for the served fleet (CONCEPT:AU-KG.retrieval.route-question-its-domain).

The agent-packages tree is bind-mounted at stable aliases inside the served
containers (``/au`` = agent-utilities, ``/src`` = a generic package mount), so the
*same* file is ingested/cited under several paths. Anything that dedups or groups
code citations (``code_context``, cross-repo usage, the context plane) must fold
those mount aliases to ONE canonical workspace path first. Centralized here so the
map has a single source of truth instead of a copy per call site.
"""

#: Mount alias -> canonical workspace prefix. ``/au`` is the agent-utilities
#: source mount; extend here (never inline a second copy) when a new mount alias
#: appears in ingested ``file_path``s.
MOUNT_ALIASES: dict[str, str] = {
    "/au/": "/home/apps/workspace/agent-packages/agent-utilities/",
}

_AGENT_PACKAGES = "/agent-packages/"
_OSS_LIBS = "/open-source-libraries/"


def normalize_path(path: str | None) -> str:
    """Fold a mount alias (e.g. ``/au/…``) to its canonical workspace path."""
    if not path:
        return ""
    for alias, canonical in MOUNT_ALIASES.items():
        if path.startswith(alias):
            return canonical + path[len(alias) :]
    return path


def repo_of(path: str) -> str:
    """Best-effort repo label from a (normalized) path, for cross-repo grouping.

    ``…/agent-packages/<repo>/…`` -> ``<repo>``; ``…/open-source-libraries/<repo>/…``
    -> ``oss/<repo>``; otherwise the parent directory.
    """
    if not path:
        return "unknown"
    if _AGENT_PACKAGES in path:
        rest = path.split(_AGENT_PACKAGES, 1)[1]
        return rest.split("/", 1)[0] if "/" in rest else rest
    if _OSS_LIBS in path:
        rest = path.split(_OSS_LIBS, 1)[1]
        return "oss/" + (rest.split("/", 1)[0] if "/" in rest else rest)
    return path.rsplit("/", 1)[0] if "/" in path else path
