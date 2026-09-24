"""Devin adapter over the hosted External API v3 (provider-managed remote).

Endpoints (docs.devin.ai API reference, checked 2026-09-22):
``POST /v3/organizations/{org_id}/sessions`` creates a session,
``GET  /v3/organizations/{org_id}/sessions/{devin_id}`` reads its status,
``GET  /v3/organizations/{org_id}/sessions/{devin_id}/messages`` pages its
messages (``after``/``first``, ``end_cursor``/``has_next_page``) and
``DELETE /v3/organizations/{org_id}/sessions/{devin_id}`` terminates it.

Devin executes on the provider's infrastructure, so the only environment mode
is ``provider-managed-remote``: the adapter records the provider session and
never claims local sandbox containment. Only the message stream is visible
(``final-output`` fidelity); MCP/EG tool access and skills-as-playbooks are not
proven, so negotiation refuses any RunSpec that needs them. The create call
has no idempotency key, so any failure after it is ``outcome_uncertain``
unless the RunSpec declared no side effects.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, Protocol

from agent_utilities.layers.contracts import (
    HarnessDescriptor,
    HarnessNotConfigured,
    HarnessOutcomeUncertain,
    HarnessRunFailed,
    RunSpec,
    UsageRecord,
    VendorTerms,
)
from agent_utilities.layers.credentials import (
    CredentialResolver,
    SecretsCredentialResolver,
    api_key_for,
)
from agent_utilities.layers.session import DriveOutcome, HarnessRuntime, RunContext

HARNESS_NAME = "devin"
DEVIN_API_BASE = "https://api.devin.ai"
DEFAULT_POLL_INTERVAL_S = 10.0

DESCRIPTOR = HarnessDescriptor(
    name=HARNESS_NAME,
    version="devin-api/v3",
    fidelity="final-output",
    capabilities=frozenset(
        {"code_edit", "shell", "browse", "cancellation", "structured_output"}
    ),
    usage_quality="unavailable",
    enforceable_budgets=frozenset({"wall_time"}),
    account_modes=frozenset({"api_key", "subscription"}),
    environment_modes=frozenset({"provider-managed-remote"}),
    tool_proof="none",
    skill_proof="none",
    max_skills=0,
    reconciliation="provider_session",
    vendor_terms=VendorTerms(
        subscription_automation_allowed=True,
        note="API access uses a service-user or personal access token; "
        "ACU consumption is billed to the organization",
    ),
)

#: ``(status, status_detail)`` -> terminal verdict; absent keys keep polling.
_TERMINAL: dict[tuple[str, str], str] = {
    ("running", "finished"): "succeeded",
    ("exit", ""): "succeeded",
    ("error", ""): "failed",
    ("running", "waiting_for_user"): "failed",
    ("running", "waiting_for_approval"): "failed",
}


class DevinApi(Protocol):
    """The four v3 session operations the adapter uses."""

    async def create_session(self, body: dict[str, Any]) -> dict[str, Any]: ...

    async def get_session(self, session_id: str) -> dict[str, Any]: ...

    async def list_messages(
        self, session_id: str, after: str | None
    ) -> dict[str, Any]: ...

    async def terminate(self, session_id: str) -> None: ...

    async def aclose(self) -> None: ...


class HttpDevinApi:
    """:class:`DevinApi` over AU's governed async HTTP client."""

    def __init__(self, token: str, org_id: str, base_url: str = DEVIN_API_BASE) -> None:
        from agent_utilities.httpsupport import AsyncBaseApiClient, TokenAuth

        self._org = org_id
        self._client = AsyncBaseApiClient(
            base_url, auth=TokenAuth(token), allow_destructive=True
        )

    def _path(self, suffix: str = "") -> str:
        return f"/v3/organizations/{self._org}/sessions{suffix}"

    async def create_session(self, body: dict[str, Any]) -> dict[str, Any]:
        return (await self._client.post(self._path(), json=body))["data"]

    async def get_session(self, session_id: str) -> dict[str, Any]:
        return (await self._client.get(self._path(f"/{session_id}")))["data"]

    async def list_messages(self, session_id: str, after: str | None) -> dict[str, Any]:
        params = {"first": 200, **({"after": after} if after else {})}
        path = self._path(f"/{session_id}/messages")
        return (await self._client.get(path, params=params))["data"]

    async def terminate(self, session_id: str) -> None:
        await self._client.delete(self._path(f"/{session_id}"))

    async def aclose(self) -> None:
        await self._client.aclose()


DevinApiFactory = Callable[[str, str], DevinApi]


def _http_api(token: str, org_id: str) -> DevinApi:
    return HttpDevinApi(token, org_id)


def _verdict(session: dict[str, Any]) -> str | None:
    status = str(session.get("status") or "")
    detail = str(session.get("status_detail") or "")
    if status == "suspended":
        return "failed"
    return _TERMINAL.get((status, detail)) or _TERMINAL.get((status, ""))


class DevinHarness(HarnessRuntime):
    """:class:`HarnessPort` over Devin's hosted sessions."""

    def __init__(
        self,
        *,
        org_id: str | None,
        credentials: CredentialResolver | None = None,
        api_factory: DevinApiFactory | None = None,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    ) -> None:
        super().__init__()
        self._org_id = org_id
        self._credentials = credentials or SecretsCredentialResolver()
        self._api_factory = api_factory or _http_api
        self._poll_interval_s = poll_interval_s

    def describe(self) -> HarnessDescriptor:
        return DESCRIPTOR

    def preflight(self, spec: RunSpec) -> None:
        if not self._org_id:
            raise HarnessNotConfigured("devin needs an organization id")
        if not spec.account_ref:
            raise HarnessNotConfigured("devin needs an account_ref for its API token")

    async def drive(self, run: RunContext) -> DriveOutcome:
        token = api_key_for(run.spec, self._credentials, HARNESS_NAME)
        api = self._api_factory(token, str(self._org_id))
        try:
            return await self._run_session(run, api)
        except asyncio.CancelledError:
            if run.provider_session:
                await asyncio.shield(api.terminate(run.provider_session))
            raise
        finally:
            await api.aclose()

    async def _run_session(self, run: RunContext, api: DevinApi) -> DriveOutcome:
        created = await api.create_session(
            {"prompt": run.spec.task, "title": run.spec.run_id, "tags": ["au-run"]}
        )
        session_id = str(created.get("session_id") or "")
        if not session_id:
            raise HarnessRunFailed("devin create_session returned no session_id")
        run.provider_session = session_id
        run.emit(
            "step",
            "observation",
            name="session.created",
            data={"url": str(created.get("url") or "")},
        )
        return await self._poll(run, api, session_id)

    async def _poll(
        self, run: RunContext, api: DevinApi, session_id: str
    ) -> DriveOutcome:
        cursor: str | None = None
        last_message = ""
        while True:
            cursor, last_message = await _drain_messages(
                run, api, session_id, cursor=cursor, last_message=last_message
            )
            session = await api.get_session(session_id)
            verdict = _verdict(session)
            if verdict is not None:
                return _outcome(
                    run, session, verdict=verdict, last_message=last_message
                )
            await asyncio.sleep(self._poll_interval_s)


async def _drain_messages(
    run: RunContext,
    api: DevinApi,
    session_id: str,
    *,
    cursor: str | None,
    last_message: str,
) -> tuple[str | None, str]:
    while True:
        page = await api.list_messages(session_id, cursor)
        for message in page.get("items") or ():
            if message.get("source") == "devin":
                last_message = str(message.get("message") or "")
                run.emit("message", "claim", detail=last_message)
        cursor = page.get("end_cursor") or cursor
        if not page.get("has_next_page"):
            return cursor, last_message


def _outcome(
    run: RunContext, session: dict[str, Any], *, verdict: str, last_message: str
) -> DriveOutcome:
    acus = session.get("acus_consumed")
    run.emit(
        "usage",
        "observation",
        name="acus_consumed",
        data={"acus": float(acus) if isinstance(acus, int | float) else None},
    )
    usage = UsageRecord(quality="unavailable", source="devin:acu-only")
    structured = session.get("structured_output")
    output = str(structured) if structured else last_message
    if verdict == "succeeded":
        return DriveOutcome(
            status="succeeded",
            output=output,
            usage=usage,
            provider_session=run.provider_session,
        )
    detail = f"{session.get('status')}/{session.get('status_detail')}"
    message = f"devin session ended {detail}: {last_message[:2_000]}"
    if run.spec.side_effects == "none":
        raise HarnessRunFailed(message)
    raise HarnessOutcomeUncertain(message)


__all__ = [
    "DESCRIPTOR",
    "HARNESS_NAME",
    "DevinApi",
    "DevinHarness",
    "HttpDevinApi",
]
