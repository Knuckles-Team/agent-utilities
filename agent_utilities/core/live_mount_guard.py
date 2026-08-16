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

U-26/BUG-172 -- a THIRD deployment shape broke the guard's founding premise.
The docstring above says "every consumer live-mounts" -- that was true of the
apps-namespace ``*-mcp`` fleet this module was written for, but the
self-contained ``docker/graphos-unified.Dockerfile`` image installs
``agent_utilities`` EDITABLE from local source *at build time*
(``pip install -e /opt/agent-utilities``) and ships with no hostPath/source
volume at all by design -- "editable" here only means a deployer MAY choose to
bind-mount fresher source over it later, not that one is expected. Naively
applying the old rule ("no active mount ⇒ CRITICAL stale-code drift") to that
pod is a false positive: the exact code that was reviewed, built, and shipped
IS what is running: there is no live mount to have drifted FROM. The fix is
provenance, not mount topology -- ``docker/graphos-unified.Dockerfile`` writes
a ``SOURCE_REVISION`` build ARG into a non-secret
``agent_utilities/.source-revision`` marker file (see
:func:`_installed_source_revision`) beside this package at build time. Its
PRESENCE is what distinguishes "an image that was deliberately built without
an intended live mount" from "a mount that was intended but missed" -- the
latter still has no marker (older images predate this fix) and keeps the
original fail-loud CRITICAL behavior unchanged. See :class:`LiveMountStatus`
and :func:`check_live_mount_status` for the typed, four-state result this
now feeds into ``observability.runtime_health``'s capability-health model
(CONCEPT:AU-OS.observability.co-service-visibility) instead of a bare log line.
"""

from __future__ import annotations

import enum
import logging
import os
from pathlib import Path
from typing import TypedDict

from agent_utilities.core._env import setting

logger = logging.getLogger("agent_utilities.live_mount_guard")

#: Filename of the non-secret build-provenance marker
#: ``docker/graphos-unified.Dockerfile`` writes beside this package at image
#: build time (``agent_utilities/.source-revision``, containing only a short
#: git revision or ``"unknown"`` -- never a credential or path). Read from
#: ``package_dir`` at runtime by :func:`_installed_source_revision`.
SOURCE_REVISION_MARKER = ".source-revision"


class LiveMountStatus(enum.StrEnum):
    """Typed outcome of the source-layout guard (U-26/R-24/BUG-172).

    Four states so "no active mount" is never conflated with "unhealthy" --
    exactly the R-24 rule: a capability/deployment shape that is
    INTENTIONALLY disabled (here: no live mount, by an immutable-image
    design) must report as such, distinctly from a genuine failure.

    * ``ACTIVE_MOUNT``       -- this package IS being read from a live
      source mount right now. Always healthy, regardless of any marker.
    * ``IMMUTABLE_VERIFIED`` -- no active mount, but the build-time
      ``.source-revision`` marker proves this image was deliberately built
      without one. Healthy; not a drift.
    * ``DRIFT``              -- no active mount AND no provenance marker.
      Ambiguous by construction (indistinguishable from real D-EGK-1 drift
      from here), so this is the ONLY state that fails loud (CRITICAL) --
      the historical, still-correct behavior for the apps-namespace fleet
      this guard was originally written for.
    * ``NOT_APPLICABLE``     -- the check does not apply here (skip env var
      set, not running under Kubernetes, or ``/proc/self/mountinfo``
      unreadable/undeterminable). Never treated as evidence either way.
    """

    ACTIVE_MOUNT = "active_mount"
    IMMUTABLE_VERIFIED = "immutable_verified"
    DRIFT = "drift"
    NOT_APPLICABLE = "not_applicable"


class LiveMountDetail(TypedDict, total=False):
    """The typed payload accompanying a :class:`LiveMountStatus`.

    ``total=False``: which keys are present varies by status (``reason`` only
    for ``NOT_APPLICABLE``; ``active_source_mount``/optionally
    ``source_revision`` otherwise) -- pinning the key set here, rather than a
    bare ``dict[str, object]``, is what keeps ``observability.runtime_health``
    (the sole caller) from drifting out of sync with what this module actually
    emits, the exact seam class this module's own docstring calls out.
    """

    reason: str
    active_source_mount: bool
    source_revision: str


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


def _installed_source_revision(package_dir: Path) -> str | None:
    """Read the build-time provenance marker beside *package_dir*, if any.

    Returns the marker's (stripped, single-line) content -- typically a short
    git revision, or the literal ``"unknown"`` if the build could not resolve
    one -- or ``None`` if no marker file exists at all. Presence alone (not
    the specific value) is what proves "this image was deliberately built
    without an intended live mount"; the value is carried through only for
    diagnostics (:func:`check_live_mount_status`'s detail payload). Never
    raises -- an unreadable marker is treated the same as an absent one.
    """
    try:
        marker = package_dir / SOURCE_REVISION_MARKER
        content = marker.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return content or None


def check_live_mount_status(
    *, package_dir: Path | None = None
) -> tuple[LiveMountStatus, LiveMountDetail]:
    """Typed, provenance-aware evaluation of the source-layout guard.

    Returns ``(status, detail)``. ``detail`` never contains secrets or raw
    filesystem paths beyond the package's own resolved directory -- only
    booleans/short strings, matching ``observability.runtime_health``'s
    redaction convention (that module is this function's primary caller,
    reporting it as one more typed capability check rather than a bare log
    line -- CONCEPT:AU-OS.observability.co-service-visibility).

    Never raises: a false positive that crashes the whole MCP fleet would be
    strictly worse than the silent-staleness defect this guard exists to
    catch, so any internal failure degrades to ``NOT_APPLICABLE``.
    """
    if setting(_SKIP_ENV_VAR, False, cast=bool):
        return LiveMountStatus.NOT_APPLICABLE, {"reason": "check skipped by env var"}
    if not setting(_IN_POD_ENV_VAR, ""):
        return LiveMountStatus.NOT_APPLICABLE, {
            "reason": "not running under Kubernetes"
        }

    try:
        resolved = (package_dir or Path(__file__).parent.parent).resolve()
        mounted = _has_active_source_mount(
            resolved,
            source_roots=_configured_source_roots(),
        )
    except Exception:  # noqa: BLE001 - this check must never break startup
        logger.debug("live-mount check could not introspect mount state", exc_info=True)
        return LiveMountStatus.NOT_APPLICABLE, {"reason": "mount state undeterminable"}

    if mounted is None:
        return LiveMountStatus.NOT_APPLICABLE, {"reason": "mount state undeterminable"}

    if mounted:
        return LiveMountStatus.ACTIVE_MOUNT, {"active_source_mount": True}

    revision = _installed_source_revision(resolved)
    if revision is not None:
        # U-26: no live mount, but the build wrote its own provenance marker
        # -- this is a deliberately immutable image, not a missed mount.
        # Informational only; never the CRITICAL drift warning below.
        logger.info(
            "agent_utilities.live_mount_guard: no active source mount, but this "
            "image carries a build-time provenance marker (source_revision=%s) "
            "-- running the exact code baked in at build time, as intended for "
            "an immutable image. Not a drift.",
            revision,
        )
        return LiveMountStatus.IMMUTABLE_VERIFIED, {
            "active_source_mount": False,
            "source_revision": revision,
        }

    logger.critical(
        "agent_utilities.live_mount_guard: this package was imported from a "
        "directory that is NOT under an active source mount inside this pod, "
        "and no build-time provenance marker (%s) is present to prove this is "
        "an intentionally immutable image. If this "
        "Deployment declares an agent_utilities hostPath live-mount volume, "
        "its mountPath pythonX.Y does not match this image's actual "
        "interpreter version -- the mount landed on a path nothing reads, "
        "and this pod is running the STALE agent_utilities baked into the "
        "image at build time. Every fix merged to agent-utilities since "
        "this image was built is NOT present in this process. "
        "See D-EGK-1 (python3 scripts/deferred_registry.py show D-EGK-1) "
        "and run scripts/check_python_mount_parity.py against this "
        "Deployment's manifest.",
        SOURCE_REVISION_MARKER,
    )
    return LiveMountStatus.DRIFT, {"active_source_mount": False}


def check_live_mount(*, package_dir: Path | None = None) -> bool | None:
    """Verify this package was imported from a live hostPath mount.

    Backward-compatible boolean view of :func:`check_live_mount_status`.
    Returns ``True`` for ``ACTIVE_MOUNT``/``IMMUTABLE_VERIFIED`` (both are
    healthy -- U-26: an immutable image without an intended mount is not a
    drift), ``False`` only for ``DRIFT`` (logged at CRITICAL inside
    :func:`check_live_mount_status`), or ``None`` for ``NOT_APPLICABLE``.
    Never raises, for the same reason :func:`check_live_mount_status` never does.
    """
    status, _detail = check_live_mount_status(package_dir=package_dir)
    if status is LiveMountStatus.NOT_APPLICABLE:
        return None
    return status is not LiveMountStatus.DRIFT


# Run at import time -- this module is imported exactly once, from
# agent_utilities/__init__.py, on every process that depends on this
# package, which in this fleet is every one of the 67 apps-namespace
# Deployments enumerated by D-EGK-1.
check_live_mount()
