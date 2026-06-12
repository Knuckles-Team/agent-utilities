"""Agent-source registry — the pluggable parser catalog (CONCEPT:ECO-4.38).

Mirrors agentsview ``internal/parser/types.go``: one ``AgentSource`` per agent
declaring where its session logs live and how to parse them. The registry powers
**auto-detection** — :func:`detect_installed` probes each source's default dirs
so the user never has to list which agents they run. New agents are a one-file
drop: define the parser, decorate it with ``@register_source``.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from agent_utilities.usage.models import ParsedSessionBundle

# parse_fn: (path, source) -> iterable of ParsedSessionBundle (one file may hold
# one or many sessions). Kept as an iterator so huge histories stream.
ParseFn = Callable[[Path, "AgentSource"], Iterator[ParsedSessionBundle]]


@dataclass(frozen=True)
class AgentSource:
    agent_type: str
    display_name: str
    default_dirs: tuple[str, ...]  # ~-expandable templates
    parse_fn: ParseFn
    env_var: str = ""  # optional dir override
    id_prefix: str = ""
    file_based: bool = True
    file_glob: str = "**/*.jsonl"

    def resolved_dirs(self) -> list[Path]:
        """Default dirs, with an env override taking precedence when set."""
        if self.env_var and os.environ.get(self.env_var):
            return [Path(os.environ[self.env_var]).expanduser()]
        return [Path(d).expanduser() for d in self.default_dirs]

    def root(self) -> Path | None:
        """First existing source dir, or ``None`` when the agent isn't installed."""
        for d in self.resolved_dirs():
            if d.exists():
                return d
        return None

    def discover(self) -> list[Path]:
        """All session files under the source's root (empty if not installed)."""
        root = self.root()
        if root is None or not self.file_based:
            return []
        return sorted(p for p in root.glob(self.file_glob) if p.is_file())

    def parse(self, path: Path) -> Iterator[ParsedSessionBundle]:
        return self.parse_fn(path, self)


AGENT_REGISTRY: dict[str, AgentSource] = {}


def register_source(source: AgentSource) -> AgentSource:
    """Register (or replace) an agent source by ``agent_type``."""
    AGENT_REGISTRY[source.agent_type] = source
    return source


def get_source(agent_type: str) -> AgentSource | None:
    return AGENT_REGISTRY.get(agent_type)


def all_sources() -> list[AgentSource]:
    return list(AGENT_REGISTRY.values())


def detect_installed() -> list[AgentSource]:
    """Auto-detect installed agents by probing each source's default dirs.

    This is the zero-config heart of ingestion: the user runs nothing; we sync
    exactly the agents whose logs actually exist on this host.
    """
    return [s for s in AGENT_REGISTRY.values() if s.root() is not None]


@dataclass
class _LoadState:
    loaded: bool = False
    fields: dict = field(default_factory=dict)


_state = _LoadState()


def ensure_parsers_loaded() -> None:
    """Import every parser module so its ``@register_source`` runs (idempotent).

    Explicit eager discovery so the wiring linter never flags the self-
    registering parser modules as dead code.
    """
    if _state.loaded:
        return
    from . import parsers  # noqa: F401 — imports all parser submodules

    parsers.load_all()
    _state.loaded = True
