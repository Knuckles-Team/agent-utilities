"""Runtime self-check for the D-EGK-1 dead-mount defect class.

Every ``*-mcp`` / ``agent-utilities-messaging`` Deployment in the ``apps``
namespace hostPath-mounts this package's live source tree over its own
``/usr/local/lib/pythonX.Y/site-packages/agent_utilities`` -- the same
editable-install pattern used for ``epistemic_graph`` and the per-service
``<app>_mcp`` package (see ``services/*/k8s/manifests.yaml``). The mount
target hardcodes ``pythonX.Y``. If the image's actual interpreter version
ever drifts from the version encoded in that mount path, the mount lands on
a directory nothing imports: the running interpreter reads its OWN
``pythonZ.W`` site-packages instead, where it finds only the STALE copy of
this package that was baked into the image at build time. The pod stays
Running, the mount is present in the manifest, kubelet reports success --
and every fix landed on ``main`` afterwards silently stops reaching that
pod. No crash, no error, no Kubernetes event. Confirmed live for 9 days on
``aris-mcp`` and ``freshrss-mcp`` (D-EGK-1).

This module is the runtime half of the fix (options (a)/(b) in D-EGK-1 --
a version-independent mount path, and a manifest/image parity gate -- are
handled separately; see ``scripts/check_python_mount_parity.py`` at the
workspace root and its wiring into the deploy path). A version-independent
mount or a pre-deploy gate can both be bypassed by something outside their
view (a manual ``kubectl apply``, an image rebuilt without the matching
gate run, a hand-edited manifest). This check cannot be bypassed that way:
it runs from *inside* the process that would otherwise be silently serving
stale code, every time that process starts, using only facts observable at
that moment -- so it is the one guard that still fires when the other two
were skipped.

Detection has nothing to do with comparing "pythonX.Y in the mount path"
against "the running interpreter's version" -- by construction, whichever
``pythonZ.W`` site-packages directory the interpreter actually reads from
IS this package's ``__file__`` location, so that comparison can never
disagree with itself. The real question is orthogonal: is the directory
this package was imported from an active source mount, or just the image's
own filesystem layer? An exact package-directory mount proves that directly.
A repository-root mount such as ``/au`` proves it only when the mounted
ancestor is also an explicit, resolved ``PYTHONPATH`` entry; an unrelated
mount such as ``/usr`` or ``/usr/local`` must not validate stale code.
``/proc/self/mountinfo`` answers the mount half of that contract directly,
independent of *why* the intended mount missed (version drift is the known
cause; a typo'd hostPath or a removed source directory would look identical
from here, and this check catches those too).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from agent_utilities.core._env import setting

logger = logging.getLogger("agent_utilities.live_mount_guard")

#: Set to skip the check entirely -- for environments that intentionally run
#: this package from a plain pip/wheel install inside a pod (none exist in
#: this fleet today; every apps-namespace consumer live-mounts). Escape
#: hatch kept cheap rather than omitted, per feedback-abstraction-first.
_SKIP_ENV_VAR = "AGENT_UTILITIES_SKIP_LIVE_MOUNT_CHECK"

#: Only ever meaningful inside a pod. Kubernetes injects this for every
#: container in every pod unconditionally, so its absence is a reliable
#: "not running under Kubernetes" signal (local dev, CI, a bare venv).
_IN_POD_ENV_VAR = "KUBERNETES_SERVICE_HOST"

_MOUNTINFO_ESCAPES = {
    r"\040": " ",
    r"\011": "\t",
    r"\012": "\n",
    r"\134": "\\",
}


def _decode_mountinfo_path(value: str) -> str:
    """Decode the octal escapes permitted in mountinfo path fields."""
    for encoded, decoded in _MOUNTINFO_ESCAPES.items():
        value = value.replace(encoded, decoded)
    return value


def _configured_source_roots() -> tuple[Path, ...]:
    """Return canonical live-source roots explicitly declared on ``PYTHONPATH``."""
    python_path = setting("PYTHONPATH", "")
    return tuple(
        Path(entry).resolve() for entry in python_path.split(os.pathsep) if entry
    )


def _has_active_source_mount(
    path: Path,
    *,
    source_roots: tuple[Path, ...] = (),
) -> bool | None:
    """Return whether *path* is on an explicitly identified active source mount.

    The package directory itself may be a mount point. An ancestor is accepted
    only when it is also a canonical ``source_roots`` entry; the filesystem root
    is never evidence of live source.

    Returns ``None`` when the answer cannot be determined (no
    ``/proc/self/mountinfo``, e.g. a non-Linux container runtime) rather
    than guessing -- an unknown must never be reported as a drift.
    """
    try:
        with open("/proc/self/mountinfo", encoding="utf-8", errors="replace") as handle:
            mount_points = {
                _decode_mountinfo_path(fields[4])
                for line in handle
                if len(fields := line.split()) > 4
            }
    except OSError:
        return None

    active_ancestors = [
        Path(mount_point)
        for mount_point in mount_points
        if mount_point != "/" and path.is_relative_to(mount_point)
    ]
    if not active_ancestors:
        return False

    deepest_mount = max(active_ancestors, key=lambda mount: len(mount.parts))
    return deepest_mount == path or deepest_mount in source_roots


def check_live_mount(*, package_dir: Path | None = None) -> bool | None:
    """Verify this package was imported from a live hostPath mount.

    Returns ``True`` (mounted), ``False`` (drifted -- logged at CRITICAL),
    or ``None`` (not applicable / undeterminable). Never raises: a false
    positive that crashes the whole MCP fleet would be strictly worse than
    the silent-staleness defect it is meant to catch.
    """
    if setting(_SKIP_ENV_VAR, False, cast=bool):
        return None
    if not setting(_IN_POD_ENV_VAR, ""):
        return None

    try:
        resolved = (package_dir or Path(__file__).parent.parent).resolve()
        mounted = _has_active_source_mount(
            resolved,
            source_roots=_configured_source_roots(),
        )
    except Exception:  # noqa: BLE001 - this check must never break startup
        logger.debug("live-mount check could not introspect mount state", exc_info=True)
        return None

    if mounted is None:
        return None

    if not mounted:
        logger.critical(
            "agent_utilities.live_mount_guard: this package was imported from a "
            "directory that is NOT under an active source mount inside this pod. "
            "If this "
            "Deployment declares an agent_utilities hostPath live-mount volume, "
            "its mountPath pythonX.Y does not match this image's actual "
            "interpreter version -- the mount landed on a path nothing reads, "
            "and this pod is running the STALE agent_utilities baked into the "
            "image at build time. Every fix merged to agent-utilities since "
            "this image was built is NOT present in this process. "
            "See D-EGK-1 (python3 scripts/deferred_registry.py show D-EGK-1) "
            "and run scripts/check_python_mount_parity.py against this "
            "Deployment's manifest."
        )
        return False

    return True


# Run at import time -- this module is imported exactly once, from
# agent_utilities/__init__.py, on every process that depends on this
# package, which in this fleet is every one of the 67 apps-namespace
# Deployments enumerated by D-EGK-1.
check_live_mount()
