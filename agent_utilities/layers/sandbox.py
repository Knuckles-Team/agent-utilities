"""The local :class:`SandboxPort`, promoting the RLM sandbox backends (§7).

Selection is capability matching over the existing backends'
:class:`~agent_utilities.rlm.sandboxes.base.SandboxCapabilities`: filter by the
job's requirements and the pin/deny policy, drop unavailable or non-isolating
backends, then take the lowest preference rank. Every exclusion is recorded on
the lease with its reason, so the choice is visible and reversible in L5.

A lease also owns a private per-run workspace directory under the port's
root; :meth:`RouterSandboxPort.release` removes it. Ranking by L5 evidence
(replacing the in-memory reward EMA) needs EG outcome queries and is not
implemented here.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from agent_utilities.layers.contracts import SandboxBoundaryError
from agent_utilities.layers.ports import SandboxLease, SandboxRequirements

#: (backend, requirements) -> refusal reason or ``None``; first hit wins.
_Check = Callable[[Any, SandboxRequirements], str | None]

_CHECKS: tuple[_Check, ...] = (
    lambda b, r: "denied by policy" if b.name in r.deny else None,
    lambda b, r: "not the pinned backend" if r.pin and b.name != r.pin else None,
    lambda b, r: None if b.capabilities.isolated else "no isolation boundary",
    lambda b, r: None if b.is_available() else "unavailable",
    lambda b, r: (
        "network egress not allowed"
        if b.capabilities.network and r.network == "none"
        else None
    ),
    lambda b, r: (
        "no network egress"
        if r.network == "egress" and not b.capabilities.network
        else None
    ),
    lambda b, r: (
        "no third-party libraries"
        if r.third_party_libs and not b.capabilities.third_party_libs
        else None
    ),
    lambda b, r: (
        "no class support" if r.classes and not b.capabilities.classes else None
    ),
    lambda b, r: (
        "no host callbacks"
        if r.host_callbacks and not b.capabilities.host_callbacks
        else None
    ),
)


def _exclusion(backend: Any, requirements: SandboxRequirements) -> str | None:
    return next(
        (reason for check in _CHECKS if (reason := check(backend, requirements))),
        None,
    )


class RouterSandboxPort:
    """:class:`SandboxPort` over the RLM sandbox backends."""

    def __init__(
        self, workspace_root: str, backends: Sequence[Any] | None = None
    ) -> None:
        if backends is None:
            from agent_utilities.rlm.sandboxes.registry import default_sandboxes

            backends = default_sandboxes()
        self._backends = tuple(backends)
        self._root = Path(workspace_root).resolve()

    def lease(self, requirements: SandboxRequirements) -> SandboxLease:
        excluded: dict[str, str] = {}
        eligible = []
        for backend in self._backends:
            reason = _exclusion(backend, requirements)
            if reason is None:
                eligible.append(backend)
            else:
                excluded[backend.name] = reason
        if not eligible:
            raise SandboxBoundaryError(
                f"no sandbox backend satisfies the requirements: {excluded}"
            )
        chosen = min(eligible, key=lambda b: (b.capabilities.preference_rank, b.name))
        lease_id = f"lease-{uuid.uuid4().hex}"
        workspace = self._root / lease_id
        workspace.mkdir(parents=True, mode=0o700)
        return SandboxLease(
            lease_id=lease_id,
            backend=chosen.name,
            workspace=str(workspace),
            reason={
                "requirements": requirements.model_dump(mode="json"),
                "excluded": dict(excluded),
                "preference_rank": chosen.capabilities.preference_rank,
            },
        )

    async def execute(self, lease: SandboxLease, code: str) -> dict[str, object]:
        from agent_utilities.rlm.sandboxes.base import SandboxEnv, SandboxRejected

        backend = self._backend(lease.backend)
        try:
            result = await backend.execute(code, SandboxEnv(vars={}))
        except SandboxRejected as exc:
            raise SandboxBoundaryError(
                f"{lease.backend} rejected the code: {exc.reason}"
            ) from exc
        return {
            "stdout": result.stdout,
            "error": result.error,
            "vars": sorted(result.updated_vars),
        }

    def release(self, lease: SandboxLease) -> None:
        from agent_utilities.layers.execution import release_workspace

        release_workspace(str(self._root), lease)

    def health(self) -> dict[str, bool]:
        return {
            backend.name: bool(backend.is_available()) for backend in self._backends
        }

    def _backend(self, name: str) -> Any:
        for backend in self._backends:
            if backend.name == name:
                return backend
        raise SandboxBoundaryError(f"unknown sandbox backend {name!r}")


__all__ = ["RouterSandboxPort"]
