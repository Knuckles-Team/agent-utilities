"""CONCEPT:ORCH-1.33 — Adapter registry + non-blocking PATH detection.

Loads :class:`AdapterDefinition` objects and probes which are available on the host, mirroring
open-design's ``runtimes/registry.ts`` + ``detection.ts``. Detection is best-effort and never raises:
a missing/broken CLI yields ``DetectedAdapter(available=False, ...)`` so callers (and a future picker
UI) degrade gracefully.

See ``.specify/design/orch-1.33-multi-cli-adapter-registry/design.md``.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time

from .base import AdapterDefinition, DetectedAdapter

logger = logging.getLogger(__name__)


class AdapterRegistry:
    """Registry of agent-CLI adapters with cached host detection.

    The registry is intentionally tiny and dependency-free: defs are registered (built-ins are loaded
    lazily from :mod:`.defs`), and :meth:`detect` probes ``PATH`` with a short per-probe timeout.
    Adding a new backend means adding an :class:`AdapterDefinition` — never editing the engine.
    """

    def __init__(self, *, detect_ttl: float = 60.0, load_builtins: bool = True) -> None:
        self._defs: dict[str, AdapterDefinition] = {}
        self._detect_cache: dict[str, DetectedAdapter] = {}
        self._detect_cache_at: float = 0.0
        self._detect_ttl = detect_ttl
        if load_builtins:
            self._load_builtins()

    # -- registration -------------------------------------------------
    def register(self, definition: AdapterDefinition) -> None:
        """Register (or shadow) an adapter by id."""
        self._defs[definition.id] = definition
        self._detect_cache_at = 0.0  # invalidate detection cache

    def get(self, adapter_id: str) -> AdapterDefinition | None:
        return self._defs.get(adapter_id)

    def ids(self) -> list[str]:
        return sorted(self._defs)

    def _load_builtins(self) -> None:
        try:
            from .defs import BUILTIN_ADAPTERS

            for d in BUILTIN_ADAPTERS:
                self._defs.setdefault(d.id, d)
        except (
            Exception
        ):  # pragma: no cover - defensive: built-ins must never break import
            logger.debug("no builtin adapters loaded", exc_info=True)

    # -- detection ----------------------------------------------------
    def _probe(
        self, definition: AdapterDefinition, *, timeout: float
    ) -> DetectedAdapter:
        path = shutil.which(definition.bin)
        if not path:
            return DetectedAdapter(
                id=definition.id, available=False, error="not on PATH"
            )
        version: str | None = None
        try:
            proc = subprocess.run(  # noqa: S603 - bin resolved via shutil.which, args are static
                [path, *definition.version_args],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            _version_lines = (proc.stdout or proc.stderr or "").strip().splitlines()
            version = _version_lines[0] if _version_lines else None
        except (
            subprocess.TimeoutExpired,
            OSError,
        ) as exc:  # broken/stalled CLI → unavailable, no raise
            return DetectedAdapter(
                id=definition.id, available=False, path=path, error=str(exc)
            )
        models: tuple[str, ...] = definition.fallback_models
        if definition.list_models is not None:
            try:
                live = tuple(definition.list_models())
                if live:
                    models = live
            except (
                Exception
            ):  # live listing is best-effort; fall back to declared models
                logger.debug("list_models failed for %s", definition.id, exc_info=True)
        return DetectedAdapter(
            id=definition.id, available=True, path=path, version=version, models=models
        )

    def detect(
        self, *, force: bool = False, timeout: float = 3.0
    ) -> dict[str, DetectedAdapter]:
        """Probe all registered adapters (cached for ``detect_ttl`` seconds).

        Non-blocking in spirit: each probe is bounded by ``timeout`` and failures are captured, never
        raised, so one stalled CLI cannot wedge detection of the rest.
        """
        now = time.monotonic()
        if (
            not force
            and self._detect_cache
            and (now - self._detect_cache_at) < self._detect_ttl
        ):
            return dict(self._detect_cache)
        self._detect_cache = {
            d.id: self._probe(d, timeout=timeout) for d in self._defs.values()
        }
        self._detect_cache_at = now
        return dict(self._detect_cache)

    def available(self, *, timeout: float = 3.0) -> list[str]:
        """Ids of adapters detected as available on this host."""
        return sorted(i for i, d in self.detect(timeout=timeout).items() if d.available)


_default_registry: AdapterRegistry | None = None


def get_adapter_registry() -> AdapterRegistry:
    """Process-wide default registry (lazy singleton)."""
    global _default_registry
    if _default_registry is None:
        _default_registry = AdapterRegistry()
    return _default_registry
