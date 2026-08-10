#!/usr/bin/python
from __future__ import annotations

"""Deployment-tooling bridge that actually RUNS ``provision_tier2_admission``
(closes BUG-068/BUG-038's "nothing calls it" gap) against a resolved manifest,
resolving the provisioner's signer credentials from the configured
:class:`~agent_utilities.security.secrets_client.SecretsClient` — never
minting, hard-coding, or persisting one itself (this repo's "Secrets &
credential retrieval" doctrine, ``AGENTS.md``).

CONCEPT:AU-OS.identity.engine-rbac-admission-deploy-bridge

Mirrors :mod:`agent_utilities.knowledge_graph.maintenance.graph_ownership_apply`
/ ``scripts/apply_graph_ownership_grants.py``'s DEFAULT-IS-DRY-RUN,
``--apply``-required convention for the same reason that pair uses it: this is
another live, mutating, security-sensitive engine-admin RPC, and the
codebase's existing precedent for "same shape of danger" is exactly that
split — not a freshly invented one.

Two callers, by design:

- ``python3 -m agent_utilities.security.tier2_admission_cli`` — a manifest
  JSON file/stdin, dry-run unless ``--apply``. Useful standalone (a human
  admitting a NEW service) and as the CLI-wiring proof this module's own
  tests exercise.
- :func:`run_tier2_admission` — called **in-process** by a Tier-1 provisioner
  (``agent-webui/scripts/provision_identity.py``'s ``tier2-admission`` stage)
  immediately after it resolves a service's Keycloak identity, so the
  manifest's ``agent_id`` is always the JUST-RESOLVED value that will appear
  as that service's own ``VerifiedRequestContext.agent_id`` — never a value
  guessed ahead of time. The ``client``/``secrets_client`` parameters exist
  so a caller (or a test) can inject a
  :class:`~agent_utilities.security.engine_rbac_admission.FixtureEngineAdmissionClient`
  and a fake secrets source and exercise the REAL credential-resolution +
  admission code path without ever constructing a
  :class:`~agent_utilities.security.engine_rbac_admission.LiveEngineAdmissionClient`
  or touching a live secrets backend — which is how this module's own test
  suite proves the bridge works without invoking the admin RPC against a real
  cluster (see that module's header for what "fresh-store deploy" evidence
  this represents vs. what remains unexecuted).
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from .engine_rbac_admission import (
    AdmissionAuthority,
    AdmissionResult,
    EngineAdmissionClient,
    EngineAdmissionError,
    FixtureEngineAdmissionClient,
    ServiceAdmissionEntry,
    provision_tier2_admission,
    resolve_engine_admission_client,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_PROVISIONER_SECRET_KEY",
    "Tier2AdmissionError",
    "load_manifest",
    "main",
    "resolve_provisioner_authority",
    "run_tier2_admission",
]


#: The one secret this bridge ever reads: a JSON object
#: ``{"agent_id": ..., "signer_id": ..., "signer_key": ...}`` for the
#: provisioner identity used as BOTH the bootstrap attempt (a harmless,
#: expected no-op once bootstrap is already consumed — see
#: ``provision_tier2_admission``'s own docstring for why) and the
#: steady-state admin authority on every subsequent run. Seeded exactly once,
#: by hand, through the existing secret-manager CLI, e.g.::
#:
#:     python -m agent_utilities.security.cli set \\
#:         engine-admission/provisioner \\
#:         --value-ref vault://platform/engine-admission#provisioner
#:
#: This module never mints, prints, logs, or persists the value itself — it
#: only ever reads it through the configured ``SecretsClient``.
DEFAULT_PROVISIONER_SECRET_KEY = "engine-admission/provisioner"


class Tier2AdmissionError(RuntimeError):
    """Raised when this bridge cannot complete admission: a credential-
    resolution failure (missing/malformed secret) or a propagated
    :class:`~agent_utilities.security.engine_rbac_admission.EngineAdmissionError`.
    Deliberately never swallowed by callers — a silent failure here recreates
    BUG-038 (something that looks provisioned but is not); the caller must
    fail the deploy step, not report success."""


def resolve_provisioner_authority(
    *, secrets_client: Any = None, key: str = DEFAULT_PROVISIONER_SECRET_KEY
) -> AdmissionAuthority:
    """Resolve the provisioner's signer credentials from the configured
    secrets backend. Never returns a placeholder — a missing or malformed
    secret is a :class:`Tier2AdmissionError`, not a silently-skipped
    admission (fail loud, per this repo's fail-closed doctrine)."""

    if secrets_client is None:
        from .secrets_client import create_secrets_client

        secrets_client = create_secrets_client()

    raw = secrets_client.get(key)
    if not raw:
        raise Tier2AdmissionError(
            f"no provisioner credential at secret key {key!r} — seed it once "
            "via `python -m agent_utilities.security.cli set "
            f"{key} --value-ref <vault://...>` before running Tier-2 "
            "admission with apply=True against a real engine"
        )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise Tier2AdmissionError(
            f"secret key {key!r} is not valid JSON — expected "
            '{"agent_id": ..., "signer_id": ..., "signer_key": ...}'
        ) from exc
    if not isinstance(payload, dict):
        raise Tier2AdmissionError(
            f"secret key {key!r} must decode to a JSON object with "
            "agent_id/signer_id/signer_key"
        )
    try:
        return AdmissionAuthority(
            agent_id=str(payload["agent_id"]),
            signer_id=str(payload["signer_id"]),
            signer_key=str(payload["signer_key"]),
        )
    except (KeyError, ValueError) as exc:
        raise Tier2AdmissionError(
            f"secret key {key!r} is missing or has an invalid "
            "agent_id/signer_id/signer_key"
        ) from exc


def load_manifest(raw: str) -> list[ServiceAdmissionEntry]:
    """Parse a JSON array of admission-manifest rows into
    :class:`~agent_utilities.security.engine_rbac_admission.ServiceAdmissionEntry`
    objects. A malformed row raises immediately
    (``ServiceAdmissionEntry.__post_init__`` fails loud) rather than being
    silently dropped from the manifest."""

    rows = json.loads(raw)
    if not isinstance(rows, list):
        raise Tier2AdmissionError("manifest JSON must be a list of entries")
    entries: list[ServiceAdmissionEntry] = []
    for row in rows:
        entries.append(
            ServiceAdmissionEntry(
                agent_id=row["agent_id"],
                tier2_actions=tuple(row["tier2_actions"]),
                grant_mode=row.get("grant_mode", "system_role"),
                role=row.get("role"),
            )
        )
    return entries


def run_tier2_admission(
    manifest: list[ServiceAdmissionEntry],
    *,
    apply: bool = False,
    client: EngineAdmissionClient | None = None,
    secrets_client: Any = None,
    secret_key: str = DEFAULT_PROVISIONER_SECRET_KEY,
) -> AdmissionResult:
    """Run Tier-2 admission for ``manifest``.

    ``apply=False`` (the default — mirrors ``apply_graph_ownership_grants.py``'s
    DEFAULT-IS-DRY-RUN convention for the same class of danger): never
    resolves a real credential and never touches a live engine or the
    ``client``/``secrets_client`` arguments. Runs the SAME
    :func:`~agent_utilities.security.engine_rbac_admission.provision_tier2_admission`
    pass against a fresh
    :class:`~agent_utilities.security.engine_rbac_admission.FixtureEngineAdmissionClient`
    seeded with a placeholder authority, so a caller gets a REAL preview
    (proves the manifest is well-formed and reachable, exactly like the
    fixture-based tests in ``test_engine_rbac_admission.py``) rather than
    just a printed plan.

    ``apply=True`` resolves the real provisioner authority via
    :func:`resolve_provisioner_authority` and, when ``client`` is not given,
    a live engine via
    :func:`~agent_utilities.security.engine_rbac_admission.resolve_engine_admission_client`.
    Passing an explicit ``client`` (a
    :class:`~agent_utilities.security.engine_rbac_admission.FixtureEngineAdmissionClient`,
    in every test that exercises this branch) still resolves REAL credentials
    through ``secrets_client`` but never constructs a
    :class:`~agent_utilities.security.engine_rbac_admission.LiveEngineAdmissionClient`
    — this is how the apply path is proven correct without ever invoking the
    admin RPC against a live cluster. A propagated
    :class:`~agent_utilities.security.engine_rbac_admission.EngineAdmissionError`
    is never swallowed — it is re-raised as :class:`Tier2AdmissionError` so
    every caller sees a failed admission as a failed deploy step, never a
    silent partial success.
    """

    if not apply:
        preview_client = FixtureEngineAdmissionClient()
        placeholder = AdmissionAuthority(
            agent_id="dry-run:provisioner",
            signer_id="dry-run:provisioner",
            signer_key="dry-run-placeholder-not-a-real-credential",  # nosec B106 - dry-run only, never a real credential
        )
        return provision_tier2_admission(
            preview_client,
            manifest,
            admin_authority=placeholder,
            bootstrap_authority=placeholder,
        )

    authority = resolve_provisioner_authority(
        secrets_client=secrets_client, key=secret_key
    )
    live_client = client if client is not None else resolve_engine_admission_client()
    try:
        return provision_tier2_admission(
            live_client,
            manifest,
            admin_authority=authority,
            bootstrap_authority=authority,
        )
    except EngineAdmissionError as exc:
        raise Tier2AdmissionError(f"Tier-2 admission failed: {exc}") from exc


def _print_result(result: AdmissionResult, *, applied: bool) -> None:
    verb = "APPLIED" if applied else "DRY-RUN"
    print(
        f"{verb}: bootstrap_attempted={result.bootstrap_attempted} "
        f"bootstrap_succeeded={result.bootstrap_succeeded} "
        f"bootstrap_already_consumed={result.bootstrap_already_consumed}"
    )
    for outcome in result.outcomes:
        print(f"  {outcome.agent_id}: {outcome.detail}")
    print(f"all_admitted={result.all_admitted}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-file",
        default=None,
        help="path to a JSON manifest file (default: read from stdin) — a "
        'list of {"agent_id", "tier2_actions", "grant_mode", "role"} rows',
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
    manifest = load_manifest(raw_manifest)
    try:
        result = run_tier2_admission(
            manifest, apply=args.apply, secret_key=args.secret_key
        )
    except Tier2AdmissionError as exc:
        print(f"TIER-2 ADMISSION FAILED: {exc}", file=sys.stderr)
        return 1

    _print_result(result, applied=args.apply)
    return 0 if result.all_admitted else 1


if __name__ == "__main__":
    sys.exit(main())
