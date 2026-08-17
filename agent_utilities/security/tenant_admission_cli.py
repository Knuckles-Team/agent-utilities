#!/usr/bin/python
from __future__ import annotations

"""Deployment-tooling bridge that actually RUNS ``provision_tenant_access``
against a resolved principal manifest — the same Wire-First closure
``tier2_admission_cli.py`` provides for ``provision_tier2_admission`` (BUG-068/
BUG-038's "nothing calls it" gap), applied to ordinary tenant Read/Write
admission instead of the 5 Tier-2 admin actions.

CONCEPT:AU-OS.identity.tenant-rbac-admission-deploy-bridge

Resolves the provisioner's signer credentials from the configured
:class:`~agent_utilities.security.secrets_client.SecretsClient` — never
minting, hard-coding, or persisting one itself (this repo's "Secrets &
credential retrieval" doctrine, ``AGENTS.md``). Mirrors
``tier2_admission_cli.py``'s DEFAULT-IS-DRY-RUN, ``--apply``-required
convention exactly, for the same reason: this is another live, mutating,
security-sensitive engine-admin RPC.

Two callers, by design (mirrors ``tier2_admission_cli.py``):

- ``python3 -m agent_utilities.security.tenant_admission_cli`` — a principal
  manifest JSON file/stdin naming ``tenant_slug`` + a list of principals,
  dry-run unless ``--apply``.
- :func:`run_tenant_admission` — called **in-process** by deployment tooling
  (e.g. the same Tier-1 Keycloak provisioning stage
  ``tier2_admission_cli.run_tier2_admission`` is already called from) once it
  has resolved a real end-user/service principal's ``agent_id`` and knows
  which tenant it belongs to.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from .tenant_rbac_admission import (
    EngineIdentityClient,
    TenantAccessResult,
    TenantAdmissionAuthority,
    TenantAdmissionError,
    TenantPrincipal,
    provision_tenant_access,
    resolve_engine_identity_client,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_PROVISIONER_SECRET_KEY",
    "TenantAdmissionCliError",
    "load_principal_manifest",
    "main",
    "resolve_provisioner_authority",
    "run_tenant_admission",
]


#: Reuses the SAME provisioner credential ``tier2_admission_cli.py`` reads —
#: one provisioner identity is the signer for every engine-admin RPC this
#: repo's deployment tooling issues (Tier-1, Tier-2, and this tenant bridge
#: alike), never a separate credential per bridge.
DEFAULT_PROVISIONER_SECRET_KEY = "engine-admission/provisioner"


class TenantAdmissionCliError(RuntimeError):
    """Raised when this bridge cannot complete admission: a credential-
    resolution failure (missing/malformed secret) or a propagated
    :class:`~agent_utilities.security.tenant_rbac_admission.TenantAdmissionError`.
    Deliberately never swallowed — a silent failure here would leave a
    principal looking provisioned when it is not; the caller must fail the
    deploy step, not report success."""


def resolve_provisioner_authority(
    *, secrets_client: Any = None, key: str = DEFAULT_PROVISIONER_SECRET_KEY
) -> TenantAdmissionAuthority:
    """Resolve the provisioner's signer credentials from the configured
    secrets backend. Never returns a synthesized authority — a missing or
    malformed secret is a :class:`TenantAdmissionCliError`, not a
    silently-skipped admission (fail loud, per this repo's fail-closed
    doctrine)."""

    if secrets_client is None:
        from .secrets_client import create_secrets_client

        secrets_client = create_secrets_client()

    raw = secrets_client.get(key)
    if not raw:
        raise TenantAdmissionCliError(
            f"no provisioner credential at secret key {key!r} — seed it once "
            "via `python -m agent_utilities.security.cli set "
            f"{key} --value-ref <vault://...>` before running tenant "
            "admission with apply=True against a real engine"
        )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TenantAdmissionCliError(
            f"secret key {key!r} is not valid JSON — expected "
            '{"agent_id": ..., "signer_id": ..., "signer_key": ...}'
        ) from exc
    if not isinstance(payload, dict):
        raise TenantAdmissionCliError(
            f"secret key {key!r} must decode to a JSON object with "
            "agent_id/signer_id/signer_key"
        )
    try:
        return TenantAdmissionAuthority(
            agent_id=str(payload["agent_id"]),
            signer_id=str(payload["signer_id"]),
            signer_key=str(payload["signer_key"]),
        )
    except (KeyError, ValueError) as exc:
        raise TenantAdmissionCliError(
            f"secret key {key!r} is missing or has an invalid "
            "agent_id/signer_id/signer_key"
        ) from exc


def load_principal_manifest(raw: str) -> tuple[str, list[TenantPrincipal]]:
    """Parse ``{"tenant_slug": ..., "principals": [...]}`` into
    ``(tenant_slug, [TenantPrincipal, ...])``. A malformed row raises
    immediately (``TenantPrincipal.__post_init__`` fails loud) rather than
    being silently dropped from the manifest."""

    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise TenantAdmissionCliError(
            "manifest JSON must be an object with tenant_slug/principals"
        )
    tenant_slug = payload.get("tenant_slug")
    rows = payload.get("principals")
    if not isinstance(tenant_slug, str) or not tenant_slug.strip():
        raise TenantAdmissionCliError("manifest tenant_slug must be a non-empty string")
    if not isinstance(rows, list):
        raise TenantAdmissionCliError("manifest principals must be a list")
    principals: list[TenantPrincipal] = []
    for row in rows:
        principals.append(
            TenantPrincipal(
                agent_id=row["agent_id"],
                role=row.get("role", "Agent"),
                teams=tuple(row.get("teams", ())),
                existing_roles=tuple(row.get("existing_roles", ())),
            )
        )
    return tenant_slug, principals


def run_tenant_admission(
    tenant_slug: str,
    principals: list[TenantPrincipal],
    *,
    apply: bool = False,
    client: EngineIdentityClient | None = None,
    secrets_client: Any = None,
    secret_key: str = DEFAULT_PROVISIONER_SECRET_KEY,
) -> TenantAccessResult:
    """Run tenant Read/Write admission for ``principals`` into ``tenant_slug``.

    ``apply=False`` (the default — mirrors ``run_tier2_admission``'s
    DEFAULT-IS-DRY-RUN convention): never resolves a real credential and
    never touches a live engine or the ``client``/``secrets_client``
    arguments. Runs the SAME :func:`~agent_utilities.security.tenant_rbac_admission.provision_tenant_access`
    pass against a fresh
    :class:`~agent_utilities.security.tenant_rbac_admission.FixtureEngineIdentityClient`
    seeded with a dry-run-only authority, so a caller gets a REAL preview
    (proves the manifest is well-formed and reachable) rather than just a
    printed plan.

    ``apply=True`` resolves the real provisioner authority via
    :func:`resolve_provisioner_authority` and, when ``client`` is not given,
    a live engine via
    :func:`~agent_utilities.security.tenant_rbac_admission.resolve_engine_identity_client`.
    A propagated
    :class:`~agent_utilities.security.tenant_rbac_admission.TenantAdmissionError`
    is never swallowed — it is re-raised as :class:`TenantAdmissionCliError`
    so every caller sees a failed admission as a failed deploy step, never a
    silent partial success.
    """

    if not apply:
        from .tenant_rbac_admission import FixtureEngineIdentityClient

        preview_client = FixtureEngineIdentityClient()
        dry_run_authority = TenantAdmissionAuthority(
            agent_id="dry-run:provisioner",
            signer_id="dry-run:provisioner",
            signer_key="dry-run-synthetic-not-a-real-credential",  # nosec B106 - dry-run only, never a real credential
        )
        return provision_tenant_access(
            preview_client, tenant_slug, principals, admin_authority=dry_run_authority
        )

    authority = resolve_provisioner_authority(
        secrets_client=secrets_client, key=secret_key
    )
    live_client = client if client is not None else resolve_engine_identity_client()
    try:
        return provision_tenant_access(
            live_client, tenant_slug, principals, admin_authority=authority
        )
    except TenantAdmissionError as exc:
        raise TenantAdmissionCliError(f"tenant admission failed: {exc}") from exc


def _print_result(result: TenantAccessResult, *, applied: bool) -> None:
    verb = "APPLIED" if applied else "DRY-RUN"
    print(f"{verb}: tenant={result.tenant_slug} role={result.role}")
    for outcome in result.outcomes:
        print(f"  {outcome.agent_id}: {outcome.detail}")
    print(f"all_admitted={result.all_admitted}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-file",
        default=None,
        help="path to a JSON manifest file (default: read from stdin) — "
        '{"tenant_slug": ..., "principals": [{"agent_id", "role", "teams", '
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
    tenant_slug, principals = load_principal_manifest(raw_manifest)
    try:
        result = run_tenant_admission(
            tenant_slug, principals, apply=args.apply, secret_key=args.secret_key
        )
    except TenantAdmissionCliError as exc:
        print(f"TENANT ADMISSION FAILED: {exc}", file=sys.stderr)
        return 1

    _print_result(result, applied=args.apply)
    return 0 if result.all_admitted else 1


if __name__ == "__main__":
    sys.exit(main())
