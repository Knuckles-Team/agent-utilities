"""CONCEPT:ORCH-1.90 — firecracker sandbox: forkd-backed microVM warm-fork (top isolation rung).

The strongest-isolation rung of the warm-fork ladder: each child is its own Firecracker microVM
(KVM hardware isolation — escape needs a hypervisor/kernel bug, not a runc regression). It is the
peer-backend wrapper around **forkd** (open-source-libraries/forkd, the project that motivated
this whole ladder; see reports/forkd-comparative-analysis-2026-06-22.md): a warm parent snapshot
is booted once and children fork from its copy-on-write guest RAM. We drive forkd's controller
over its REST API with stdlib ``urllib`` only (no new dependency — forkd's own Python SDK is
likewise pure-stdlib), so nothing heavy lands in core.

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

import json
import logging
import urllib.error
import urllib.request

from agent_utilities.core.config import setting

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


class _ForkdClient:
    """Thin stdlib HTTP client for the forkd controller REST API (bearer auth)."""

    def __init__(self, base_url: str, token: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._token = token
        self.timeout = timeout

    def request(self, method: str, path: str, body: dict | None = None) -> dict:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # nosec B310 - fixed scheme
            raw = resp.read().decode()
        return json.loads(raw) if raw else {}

    def healthy(self) -> bool:
        try:
            urllib.request.urlopen(  # nosec B310 - fixed loopback controller URL
                f"{self.base_url}/healthz", timeout=5
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
                    f"forkd snapshot {tag!r} not found (build it: `forkd from-image` / `forkd pull`)"
                )
        except SandboxFatalError:
            raise
        except Exception as e:  # noqa: BLE001
            raise SandboxFatalError(f"forkd controller unreachable: {e}") from e
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
            child_id = children[0].get("id")
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
        except Exception as e:  # noqa: BLE001 - committed to the microVM path => infra failure
            raise SandboxFatalError(f"firecracker sandbox failed: {e}") from e
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

        def _branch() -> ParentHandle:
            try:
                self._client.request(
                    "POST",
                    f"/v1/sandboxes/{child_id}/branch",
                    {"tag": tag, "mode": mode},
                )
            except Exception as e:  # noqa: BLE001
                raise SandboxFatalError(f"forkd branch failed: {e}") from e
            spec = WarmSpec(backend=self.name, extra=(("snapshot", tag),))
            return ParentHandle(backend=self.name, spec=spec, ref={"snapshot": tag})

        return await asyncio.get_running_loop().run_in_executor(None, _branch)
