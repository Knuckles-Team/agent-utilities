"""CONCEPT:AU-ORCH.sandbox.forkd-backed-microvm-strongest — firecracker sandbox: forkd-backed microVM warm-fork (top isolation rung).

The strongest-isolation rung of the warm-fork ladder: each child is its own Firecracker microVM
(KVM hardware isolation — escape needs a hypervisor/kernel bug, not a runc regression). It is the
peer-backend wrapper around **forkd** (open-source-libraries/forkd, the project that motivated
this whole ladder; see reports/forkd-comparative-analysis-2026-06-22.md): a warm parent snapshot
is booted once and children fork from its copy-on-write guest RAM. We drive forkd's controller
through the shared bounded HTTP boundary, so controller calls inherit exact-host egress, DNS
pinning, redirect denial, response limits, and the operator's TLS profile.

Hard constraints (why this rung is detection-gated and ranked last):
* **x86_64 + KVM, single-host.** The CoW mmap can't cross the wire, and microVMs need ``/dev/kvm``
  — so this rung only ``is_available`` where a reachable ``forkd-controller`` exists. On hosts
  without it (e.g. ARM, or no KVM) it simply never registers and the router uses a cheaper rung.
* **host_callbacks=False (v1).** The RLM host helpers are served over a host-filesystem UDS bridge
  (``_bridge``); a microVM guest is network-isolated and cannot reach that socket without a
  vsock/TCP bridge (future work). So this rung runs self-contained compute only — the router
  never sends ``rlm_query``-using snippets here (same posture as the ``wasm`` rung v1).

Unique capability: :meth:`branch` — snapshot a *running* child microVM into a new parent (fork
mid-execution), the one warm-fork verb ``os.fork`` can't provide. It lives only on this rung.
"""

from __future__ import annotations

import logging
from urllib.parse import urlsplit

from agent_utilities.core.config import config, setting
from agent_utilities.protocols.source_connectors.http_safety import (
    SourceEgressError,
    require_safe_source_url,
    safe_delete_json,
    safe_get_bytes,
    safe_get_json,
    safe_post_json,
)

from ..telemetry import SandboxFatalError
from .base import (
    ForkableSandbox,
    ParentHandle,
    SandboxCapabilities,
    SandboxEnv,
    SandboxResult,
    WarmSpec,
)

logger = logging.getLogger(__name__)


def _forkd_identifier(value: object) -> str:
    rendered = str(value or "")
    if (
        not 1 <= len(rendered) <= 128
        or not rendered[0].isalnum()
        or any(not (character.isalnum() or character in "._:-") for character in rendered)
    ):
        raise SandboxFatalError("forkd returned an invalid sandbox identifier")
    return rendered


class _ForkdClient:
    """Bounded HTTP client for the forkd controller REST API (bearer auth)."""

    def __init__(self, base_url: str, token: str, timeout: float = 120.0) -> None:
        candidate = str(base_url).rstrip("/")
        parsed = urlsplit(candidate)
        host = parsed.hostname or ""
        if (
            parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise SourceEgressError("Forkd controller URL must be an HTTP origin")
        allowed_hosts = set(config.source_http_allowed_private_hosts)
        if host.casefold().rstrip(".") in {"localhost", "127.0.0.1", "::1"}:
            allowed_hosts.add(host)
        require_safe_source_url(
            candidate,
            allowed_private_hosts=allowed_hosts,
            resolve_dns=False,
        )
        self.base_url = candidate
        self._token = token
        self.timeout = timeout
        self._allowed_private_hosts = tuple(allowed_hosts)

    @staticmethod
    def _validated_path(path: str) -> str:
        if (
            not isinstance(path, str)
            or not path.startswith("/")
            or path.startswith("//")
            or len(path) > 2_048
            or "\\" in path
            or any(ord(character) < 32 or ord(character) == 127 for character in path)
        ):
            raise ValueError("Forkd request path is invalid")
        parsed = urlsplit(path)
        if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError("Forkd request path is invalid")
        if any(segment in {"", ".", ".."} for segment in path.split("/")[1:]):
            raise ValueError("Forkd request path is invalid")
        return path

    def _headers(self) -> dict[str, str]:
        if not self._token:
            return {}
        return {"Authorization": f"Bearer {self._token}"}

    def request(
        self, method: str, path: str, body: dict | None = None
    ) -> dict | list:
        selected_method = str(method).upper()
        url = f"{self.base_url}{self._validated_path(path)}"
        kwargs = {
            "headers": self._headers(),
            "timeout": self.timeout,
            "max_bytes": 4 * 1024 * 1024,
            "allowed_private_hosts": self._allowed_private_hosts,
            "tls_service": "forkd",
        }
        if selected_method == "GET":
            if body is not None:
                raise ValueError("Forkd GET requests cannot contain a body")
            response = safe_get_json(url, max_redirects=0, **kwargs)
        elif selected_method == "POST":
            response = safe_post_json(
                url,
                body or {},
                max_request_bytes=4 * 1024 * 1024,
                **kwargs,
            )
        elif selected_method == "DELETE":
            if body is not None:
                raise ValueError("Forkd DELETE requests cannot contain a body")
            response = safe_delete_json(url, **kwargs)
        else:
            raise ValueError("Unsupported forkd HTTP method")
        if not isinstance(response, (dict, list)):
            raise SourceEgressError("Forkd controller returned an invalid JSON envelope")
        return response

    def healthy(self) -> bool:
        try:
            safe_get_bytes(
                f"{self.base_url}/healthz",
                headers=self._headers(),
                timeout=5,
                max_bytes=64 * 1024,
                max_redirects=0,
                allowed_private_hosts=self._allowed_private_hosts,
                tls_service="forkd",
            )
            return True
        except Exception:  # noqa: BLE001 - any failure => not reachable
            return False


class FirecrackerSandbox(ForkableSandbox):
    """Run a snippet in a Firecracker microVM child forked from a warm forkd snapshot."""

    name = "firecracker"
    capabilities = SandboxCapabilities(
        host_callbacks=False,  # v1: microVM guest can't reach the host UDS bridge (needs vsock)
        third_party_libs=True,  # whatever the warm snapshot image baked in
        classes=True,
        full_stdlib=True,
        network=True,
        isolated=True,  # KVM hardware isolation — the strongest rung
        preference_rank=25,  # last/heaviest; tried only when cheaper rungs can't or are unhealthy
        warm_fork=True,
    )

    def __init__(
        self,
        *,
        base_url: str | None = None,
        token: str | None = None,
        snapshot_tag: str | None = None,
        timeout_secs: float = 120.0,
    ) -> None:
        # Deployment-varying (URL / secret / which snapshot) → justified config knobs, read
        # through config.setting (never bare os.environ), per Configuration discipline.
        self.base_url = base_url or setting("FORKD_URL", "http://127.0.0.1:8889")
        self._token = token if token is not None else setting("FORKD_TOKEN", "")
        self.snapshot_tag = snapshot_tag or setting("FORKD_SNAPSHOT_TAG", "pyagent")
        self.timeout_secs = timeout_secs
        self._client = _ForkdClient(self.base_url, self._token, timeout_secs)
        self._available: bool | None = None

    def is_available(self) -> bool:
        """Available only where a reachable forkd controller exists (implies x86_64+KVM+forkd)."""
        if self._available is None:
            self._available = self._client.healthy()
        return self._available

    def warm_spec(self) -> WarmSpec:
        # The warm parent is forkd's snapshot tag (booted+warmed by forkd out-of-band).
        return WarmSpec(backend=self.name, extra=(("snapshot", self.snapshot_tag),))

    async def warm(self, spec: WarmSpec) -> ParentHandle:
        """Verify the controller is reachable and the snapshot exists (forkd warmed it already).

        forkd builds/pulls the parent snapshot out-of-band (``forkd from-image`` / ``forkd pull``);
        warming here is confirming that warm parent is present to fork from — not booting a VM
        per run.
        """
        tag = dict(spec.extra).get("snapshot", self.snapshot_tag)
        try:
            snaps = self._client.request("GET", "/v1/snapshots")
            tags = {
                s.get("tag")
                for s in (snaps.get("snapshots") or snaps.get("items") or [])
            }
            if tags and tag not in tags:
                raise SandboxFatalError(
                    "configured forkd snapshot was not found"
                )
        except SandboxFatalError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise SandboxFatalError(
                f"forkd controller unreachable ({type(exc).__name__})"
            ) from exc
        return ParentHandle(backend=self.name, spec=spec, ref={"snapshot": tag})

    async def run_forked(
        self, parent: ParentHandle, code: str, env: SandboxEnv
    ) -> SandboxResult:
        """Fork one microVM child from the warm snapshot, eval the snippet, tear the child down."""
        import asyncio

        tag = parent.ref["snapshot"]
        return await asyncio.get_running_loop().run_in_executor(
            None, self._run_blocking, tag, code
        )

    def _run_blocking(self, tag: str, code: str) -> SandboxResult:
        child_id: str | None = None
        try:
            spawned = self._client.request(
                "POST", "/v1/sandboxes", {"snapshot_tag": tag, "n": 1}
            )
            children = (
                spawned if isinstance(spawned, list) else spawned.get("sandboxes", [])
            )
            if not children:
                raise SandboxFatalError("forkd returned no child sandbox")
            if not isinstance(children[0], dict):
                raise SandboxFatalError("forkd returned an invalid child sandbox")
            child_id = _forkd_identifier(children[0].get("id"))
            res = self._client.request(
                "POST", f"/v1/sandboxes/{child_id}/eval", {"code": code}
            )
            return SandboxResult(
                updated_vars={},
                stdout=str(res.get("stdout", res.get("result", ""))),
                error=res.get("error"),
            )
        except SandboxFatalError:
            raise
        except Exception as exc:  # noqa: BLE001 - committed to the microVM path => infra failure
            raise SandboxFatalError(
                f"firecracker sandbox failed ({type(exc).__name__})"
            ) from exc
        finally:
            if child_id:
                try:
                    self._client.request("DELETE", f"/v1/sandboxes/{child_id}")
                except Exception:  # noqa: BLE001 - best-effort teardown
                    logger.debug("forkd child %s teardown failed", child_id)

    async def branch(
        self, child_id: str, *, tag: str, mode: str = "diff"
    ) -> ParentHandle:
        """Snapshot a *running* child microVM into a new parent (fork mid-execution).

        The microVM-only warm-fork verb (forkd BRANCH) — ``os.fork``/container rungs cannot snapshot
        live in-flight memory. ``mode`` in full|diff|live (forkd v0.3/v0.4). Returns the new parent.
        """
        import asyncio

        child_id = _forkd_identifier(child_id)

        def _branch() -> ParentHandle:
            try:
                self._client.request(
                    "POST",
                    f"/v1/sandboxes/{child_id}/branch",
                    {"tag": tag, "mode": mode},
                )
            except Exception as exc:  # noqa: BLE001
                raise SandboxFatalError(
                    f"forkd branch failed ({type(exc).__name__})"
                ) from exc
            spec = WarmSpec(backend=self.name, extra=(("snapshot", tag),))
            return ParentHandle(backend=self.name, spec=spec, ref={"snapshot": tag})

        return await asyncio.get_running_loop().run_in_executor(None, _branch)
