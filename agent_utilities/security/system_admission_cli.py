#!/usr/bin/python
from __future__ import annotations

"""Operator CLI for au's own system-principal admission (BUG-295 / NE-009 /
NE-020 / NE-021) — the operator-gated escape hatch alongside
:func:`agent_utilities.security.system_rbac_admission.ensure_system_principal_access`'s
auto-at-boot path. Mirrors ``tenant_admission_cli.py``/``tier2_admission_cli.py``
exactly: DEFAULT-IS-DRY-RUN, ``--apply`` required for a live, mutating call,
because this is the same class of live security-boundary mutation those two
bridges already gate.

CONCEPT:AU-OS.identity.system-principal-admission-deploy-bridge

Why this exists in addition to the auto-at-boot path (see
:mod:`agent_utilities.security.system_rbac_admission`'s module docstring,
"Auto-admission at boot" for the full reasoning): auto-admission only ever
runs from inside a graph-os daemon process at its own boot. An operator
needs a way to run the SAME provisioning (a) before a rollout, to pre-warm
a fresh store; (b) immediately after seeding the NE-021 provisioner
credential, without waiting for or forcing a pod restart; or (c) as a
dry-run preview to confirm a manifest is well-formed before trusting it to
auto-admission. Both paths call the exact same
:func:`~agent_utilities.security.system_rbac_admission.provision_system_principal_access`
composition, so they always produce identical provisioning for the same
principal — proven directly in
``tests/unit/security/test_system_rbac_admission.py``.

Resolves the provisioner's signer credentials from the configured
:class:`~agent_utilities.security.secrets_client.SecretsClient` — never
minting, hard-coding, or persisting one itself (this repo's "Secrets &
credential retrieval" doctrine, ``AGENTS.md``).

Usage::

    python3 -m agent_utilities.security.system_admission_cli \\
        --manifest-file manifest.json [--apply] [--role control:system]

where ``manifest.json`` is::

    {"role": "control:system",
     "principals": [{"agent_id": "graph-os-scheduler", "role": "Agent",
                      "teams": [], "existing_roles": []}]}

``role`` is optional (defaults to
:data:`~agent_utilities.security.system_rbac_admission.CONTROL_ROLE_NAME`).

For the full end-to-end procedure this CLI is one step of — generating a
dedicated signer, writing the engine-side ``EPISTEMIC_GRAPH_SIGNER_KEYS_JSON``
entry and the matching ``engine-admission/provisioner`` secret this module
reads, the restart this credential requires, and the exact observable that
proves admission actually worked (not merely that this command exited 0) —
see ``agent_utilities/skills/workflows/agent-os-genesis/references/
engine-identity-admission.md``. That document also states the four design
properties of this credential an operator should treat as known risks, not
incidental detail (unconstrained role-granting authority per signer, a
shared symmetric secret, bootstrap circularity against the default
``SECRETS_BACKEND=engine``, and no live-reload rotation path).
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from .system_rbac_admission import (
    CONTROL_ROLE_NAME,
    DEFAULT_PROVISIONER_SECRET_KEY,
    SystemAccessResult,
    SystemAdmissionClient,
    SystemAdmissionError,
    SystemPrincipal,
    provision_system_principal_access,
    resolve_provisioner_authority,
    resolve_system_admission_client,
)

logger = logging.getLogger(__name__)

__all__ = [
    "SystemAdmissionCliError",
    "load_manifest",
    "main",
    "run_system_admission",
]


class SystemAdmissionCliError(RuntimeError):
    """Raised when this bridge cannot complete admission: a credential-
    resolution failure (missing/malformed secret — the NE-021 condition) or
    a propagated
    :class:`~agent_utilities.security.system_rbac_admission.SystemAdmissionError`.
    Deliberately never swallowed by callers — a silent failure here would
    leave a principal looking provisioned when it is not, reproducing
    BUG-295's own "no diagnosis" failure shape; the caller must fail the
    deploy step, not report success."""


def load_manifest(raw: str) -> tuple[str, list[SystemPrincipal]]:
    """Parse ``{"role": ..., "principals": [...]}`` into
    ``(role, [SystemPrincipal, ...])``. ``role`` defaults to
    :data:`~agent_utilities.security.system_rbac_admission.CONTROL_ROLE_NAME`
    when omitted. A malformed row raises immediately
    (``SystemPrincipal.__post_init__`` fails loud) rather than being
    silently dropped from the manifest."""

    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise SystemAdmissionCliError(
            "manifest JSON must be an object with role/principals"
        )
    role = payload.get("role") or CONTROL_ROLE_NAME
    if not isinstance(role, str) or not role.strip():
        raise SystemAdmissionCliError("manifest role must be a non-empty string")
    rows = payload.get("principals")
    if not isinstance(rows, list) or not rows:
        raise SystemAdmissionCliError("manifest principals must be a non-empty list")
    principals: list[SystemPrincipal] = []
    for row in rows:
        principals.append(
            SystemPrincipal(
                agent_id=row["agent_id"],
                role=row.get("role", "Agent"),
                teams=tuple(row.get("teams", ())),
                existing_roles=tuple(row.get("existing_roles", ())),
            )
        )
    return role, principals


def run_system_admission(
    principals: list[SystemPrincipal],
    *,
    role: str = CONTROL_ROLE_NAME,
    apply: bool = False,
    client: SystemAdmissionClient | None = None,
    secrets_client: Any = None,
    secret_key: str = DEFAULT_PROVISIONER_SECRET_KEY,
) -> SystemAccessResult:
    """Run system-principal admission for ``principals`` into ``role``.

    ``apply=False`` (the default — mirrors ``run_tenant_admission``'s /
    ``run_tier2_admission``'s DEFAULT-IS-DRY-RUN convention): never resolves
    a real credential and never touches a live engine or the
    ``client``/``secrets_client`` arguments. Runs the SAME
    :func:`~agent_utilities.security.system_rbac_admission.provision_system_principal_access`
    pass against a fresh
    :class:`~agent_utilities.security.system_rbac_admission.FixtureSystemAdmissionClient`
    seeded with a dry-run-only authority, so a caller gets a REAL preview
    rather than just a printed plan.

    ``apply=True`` resolves the real provisioner authority via
    :func:`~agent_utilities.security.system_rbac_admission.resolve_provisioner_authority`
    and, when ``client`` is not given, a live engine via
    :func:`~agent_utilities.security.system_rbac_admission.resolve_system_admission_client`.
    This is the exact NE-021 condition on the target deployment: with no
    credential seeded, this call raises
    :class:`SystemAdmissionCliError` naming exactly the missing secret key
    — it never silently no-ops and never pretends success.
    """

    if not apply:
        from .system_rbac_admission import (
            FixtureSystemAdmissionClient,
            SystemAdmissionAuthority,
        )

        preview_client = FixtureSystemAdmissionClient()
        dry_run_authority = SystemAdmissionAuthority(
            agent_id="dry-run:provisioner",
            signer_id="dry-run:provisioner",
            signer_key="dry-run-synthetic-not-a-real-credential",  # nosec B106 - dry-run only; sanitizer:ignore synthetic placeholder, never a real credential
        )
        return provision_system_principal_access(
            preview_client, principals, admin_authority=dry_run_authority, role=role
        )

    try:
        authority = resolve_provisioner_authority(
            secrets_client=secrets_client, key=secret_key
        )
    except SystemAdmissionError as exc:
        raise SystemAdmissionCliError(str(exc)) from exc

    live_client = client if client is not None else resolve_system_admission_client()
    try:
        return provision_system_principal_access(
            live_client, principals, admin_authority=authority, role=role
        )
    except SystemAdmissionError as exc:
        raise SystemAdmissionCliError(f"system admission failed: {exc}") from exc


def _print_result(result: SystemAccessResult, *, applied: bool) -> None:
    verb = "APPLIED" if applied else "DRY-RUN"
    print(f"{verb}: role={result.role}")
    for outcome in result.outcomes:
        print(f"  {outcome.agent_id}: {outcome.detail}")
    print(f"all_admitted={result.all_admitted}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-file",
        default=None,
        help="path to a JSON manifest file (default: read from stdin) — "
        '{"role": ..., "principals": [{"agent_id", "role", "teams", '
        '"existing_roles"}, ...]}',
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually run admission against a LIVE engine (default: "
        "dry-run preview against an in-memory fixture, never touches a "
        "live engine)",
    )
    parser.add_argument("--secret-key", default=DEFAULT_PROVISIONER_SECRET_KEY)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO)

    raw_manifest = (
        Path(args.manifest_file).read_text(encoding="utf-8")
        if args.manifest_file
        else sys.stdin.read()
    )
    role, principals = load_manifest(raw_manifest)
    try:
        result = run_system_admission(
            principals, role=role, apply=args.apply, secret_key=args.secret_key
        )
    except SystemAdmissionCliError as exc:
        print(f"SYSTEM ADMISSION FAILED: {exc}", file=sys.stderr)
        return 1

    _print_result(result, applied=args.apply)
    return 0 if result.all_admitted else 1


if __name__ == "__main__":
    sys.exit(main())
