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
this package was imported from an ACTIVE bind mount, or just the image's
own filesystem layer? ``/proc/self/mountinfo`` answers that directly and
authoritatively, independent of *why* the intended mount missed (version
drift is the known cause; a typo'd hostPath or a removed source directory
would look identical from here, and this check catches those too).
"""

from __future__ import annotations

import logging
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


def _is_bind_mounted(path: Path) -> bool | None:
    """Return True/False if *path* is a mount point per this mount namespace.

    Returns ``None`` when the answer cannot be determined (no
    ``/proc/self/mountinfo``, e.g. a non-Linux container runtime) rather
    than guessing -- an unknown must never be reported as a drift.
    """
    try:
        with open("/proc/self/mountinfo", encoding="utf-8", errors="replace") as handle:
            mount_points = {line.split()[4] for line in handle if len(line.split()) > 4}
    except OSError:
        return None
    return str(path) in mount_points


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

    resolved = package_dir or Path(__file__).resolve().parent.parent
    try:
        mounted = _is_bind_mounted(resolved)
    except Exception:  # noqa: BLE001 - this check must never break startup
        logger.debug("live-mount check could not introspect mount state", exc_info=True)
        return None

    if mounted is None:
        return None

    if not mounted:
        logger.critical(
            "agent_utilities.live_mount_guard: this package was imported from a "
            "directory that is NOT an active bind mount inside this pod. If this "
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
