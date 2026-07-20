#!/usr/bin/env python3
"""Execute one real certification fault and prove recovery with aggregate probes.

The campaign invokes this process with a ``CertificationFaultRequest`` on stdin.
For each exact scenario identifier, the process resolves JSON argv from the typed
``AgentConfig`` maps ``CERT_FAULT_ACTION_COMMANDS`` and
``CERT_FAULT_PROBE_COMMANDS``.  The action must report ``{"applied": true}``; the
probe must report only aggregate invariant booleans and an observed RPO.  No command
uses a shell and no command output is copied into evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from typing import Any

from agent_utilities.core.config import (
    PRODUCTION_CERTIFICATION_SCENARIOS,
    AgentConfig,
)
from scripts.certification.subprocess_boundary import (
    AdapterBoundaryError,
    run_bounded,
)


class FaultHookError(RuntimeError):
    """The actual fault or its deterministic recovery proof failed."""


def _argv(value: Any, field: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise FaultHookError(f"{field} must be a non-empty JSON argv array")
    from agent_utilities.skills.runtime_validation import (
        _validate_external_command_argv,
    )

    try:
        return _validate_external_command_argv(value)
    except RuntimeError as exc:
        raise FaultHookError(f"{field} is not a safe executable argv array") from exc


def _invoke(
    command: list[str], request: dict[str, Any], timeout: float
) -> tuple[dict[str, Any], bytes]:
    try:
        result = run_bounded(
            command,
            payload=json.dumps(
                request, sort_keys=True, separators=(",", ":")
            ).encode(),
            timeout=timeout,
        )
    except AdapterBoundaryError as exc:
        raise FaultHookError("external fault operation violated its boundary") from exc
    material = result.stdout + result.stderr
    if result.returncode != 0:
        raise FaultHookError("external fault operation failed")
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise FaultHookError("external fault operation returned non-JSON") from exc
    if not isinstance(value, dict):
        raise FaultHookError("external fault operation returned a non-object")
    return value, material


def main() -> int:
    try:
        request = json.load(sys.stdin)
        if (
            request.get("apiVersion") != "graphos.io/v1"
            or request.get("kind") != "CertificationFaultRequest"
        ):
            raise FaultHookError("unsupported fault request")
        scenario = str(request.get("scenario") or "")
        if scenario not in PRODUCTION_CERTIFICATION_SCENARIOS:
            raise FaultHookError("invalid scenario id")
        config = AgentConfig()
        if set(config.cert_fault_action_commands) != set(
            PRODUCTION_CERTIFICATION_SCENARIOS
        ) or set(config.cert_fault_probe_commands) != set(
            PRODUCTION_CERTIFICATION_SCENARIOS
        ):
            raise FaultHookError("production fault command sets are not exact")
        deadline = time.monotonic() + int(request["timeoutSeconds"])
        started = time.monotonic()
        action, action_bytes = _invoke(
            _argv(
                config.cert_fault_action_commands[scenario],
                f"CERT_FAULT_ACTION_COMMANDS.{scenario}",
            ),
            request,
            max(1.0, deadline - time.monotonic()),
        )
        if action.get("applied") is not True:
            raise FaultHookError("fault command did not attest application")
        required = action.get("requiredInvariants")
        if (
            not isinstance(required, list)
            or not required
            or not all(isinstance(item, str) and item for item in required)
        ):
            raise FaultHookError("fault command omitted its invariant contract")
        recovered: dict[str, Any] | None = None
        observation_material = bytearray()
        while time.monotonic() < deadline:
            probe, probe_bytes = _invoke(
                _argv(
                    config.cert_fault_probe_commands[scenario],
                    f"CERT_FAULT_PROBE_COMMANDS.{scenario}",
                ),
                request,
                min(60.0, max(1.0, deadline - time.monotonic())),
            )
            observation_material.extend(hashlib.sha256(probe_bytes).digest())
            invariants = probe.get("invariants")
            if isinstance(invariants, dict) and all(
                invariants.get(name) is True for name in required
            ):
                recovered = probe
                break
            time.sleep(min(5.0, max(0.0, deadline - time.monotonic())))
        if recovered is None:
            raise FaultHookError("recovery invariants did not converge")
        observed_rpo_value = recovered.get("observedRpoSeconds", -1)
        if isinstance(observed_rpo_value, bool):
            raise FaultHookError("recovery probe returned invalid observed RPO")
        observed_rpo = float(observed_rpo_value)
        if not math.isfinite(observed_rpo) or observed_rpo < 0:
            raise FaultHookError("recovery probe omitted observed RPO")
        response = {
            "ok": True,
            "faultApplied": True,
            "actionDigest": "sha256:" + hashlib.sha256(action_bytes).hexdigest(),
            "observationDigest": "sha256:"
            + hashlib.sha256(observation_material).hexdigest(),
            "recoverySeconds": round(time.monotonic() - started, 6),
            "observedRpoSeconds": observed_rpo,
            "invariants": {name: True for name in sorted(required)},
        }
    except Exception as exc:  # noqa: BLE001 - no external output or environment values
        print(json.dumps({"ok": False, "error": type(exc).__name__}, sort_keys=True))
        return 1
    print(json.dumps(response, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
