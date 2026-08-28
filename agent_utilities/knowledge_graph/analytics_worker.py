"""Authenticated standalone worker for the Epistemic Graph analytics job plane.

The worker has no direct database access.  It claims opaque work, renews a fenced
lease, stages one governed KnowledgeBatch-shaped result and asks the authoritative
engine to publish it.  Runtime endpoints and identity arrive only through the
environment; logs and durable results contain no source labels, local paths or
principal values.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import secrets
import signal
import struct
import threading
import time
import unicodedata
from dataclasses import dataclass
from typing import Any

import msgpack

from agent_utilities.core.config import setting


class AnalyticsWorkerError(RuntimeError):
    """A worker protocol, payload or governed-result invariant failed."""


class KernelCancelled(AnalyticsWorkerError):
    """The active kernel observed cooperative cancellation."""


_SCHEMA = [
    {"name": "id", "logical_type": "string", "nullable": False},
    {"name": "kind", "logical_type": "string", "nullable": False},
    {"name": "confidence", "logical_type": "float64", "nullable": False},
    {"name": "evidence_refs", "logical_type": "list<string>", "nullable": False},
    {"name": "source_refs", "logical_type": "list<string>", "nullable": False},
    {"name": "proof_ids", "logical_type": "list<string>", "nullable": False},
    {"name": "contradiction_ids", "logical_type": "list<string>", "nullable": False},
    {"name": "antecedent", "logical_type": "list<string>", "nullable": False},
    {"name": "consequent", "logical_type": "list<string>", "nullable": False},
    {"name": "support", "logical_type": "float64", "nullable": False},
    {"name": "lift", "logical_type": "float64", "nullable": False},
]
_ALGORITHMS = {"apriori", "fpgrowth", "fp-growth", "fp_growth", "eclat"}


def _required(name: str) -> str:
    value = str(setting(name, "") or "").strip()
    if not value:
        raise AnalyticsWorkerError(f"required runtime setting {name} is absent")
    return value


def _endpoints() -> list[str]:
    from agent_utilities.core.config import config
    from agent_utilities.knowledge_graph.core.shard_topology import resolve_endpoints

    endpoints = resolve_endpoints(config)
    if not endpoints or any(
        not endpoint.startswith(("tcp://", "tls://")) for endpoint in endpoints
    ):
        raise AnalyticsWorkerError(
            "analytics workers require native TCP graph service contacts"
        )
    return endpoints


def _verified_context() -> dict[str, Any]:
    principal = _required("GRAPH_OS_ANALYTICS_PRINCIPAL")
    return {
        "principal": principal,
        "tenant": _required("GRAPH_OS_ANALYTICS_TENANT"),
        "audience": _required("AUTH_JWT_AUDIENCE"),
        "agent_id": principal,
        "roles": ["analytics-worker"],
        "scopes": ["kg:write", "analytics:worker"],
        "delegation": [],
        "policy_version": _required("KG_POLICY_VERSION"),
    }


def _capabilities() -> list[str]:
    raw = str(
        setting(
            "EG_ANALYTICS_WORKER_CAPABILITIES",
            "mining.association,pool:default",
        )
    )
    values = list(
        dict.fromkeys(value.strip() for value in raw.split(",") if value.strip())
    )
    if (
        "mining.association" not in values
        or len(values) > 64
        or any(
            len(value) > 128
            or any(unicodedata.category(char) == "Cc" for char in value)
            for value in values
        )
    ):
        raise AnalyticsWorkerError(
            "worker capabilities are outside the governed schema"
        )
    return values


def _check_cancel(cancel: threading.Event) -> None:
    if cancel.is_set():
        raise KernelCancelled("kernel cancelled")


def _association_rule_id(
    antecedent: list[str],
    consequent: list[str],
    support: float,
    confidence: float,
    lift: float,
) -> str:
    """Cross-language rule identity shared with graph-os's Rust publisher."""

    digest = hashlib.sha256(b"eg-jobs.association-rule.v1\0")
    for values in (antecedent, consequent):
        digest.update(struct.pack("<Q", len(values)))
        for value in values:
            encoded = value.encode("utf-8")
            digest.update(struct.pack("<Q", len(encoded)))
            digest.update(encoded)
    for metric in (support, confidence, lift):
        digest.update(struct.pack("<d", metric))
    return f"eg:rule:{digest.hexdigest()}"


@dataclass(frozen=True)
class _Rule:
    antecedent: tuple[int, ...]
    consequent: tuple[int, ...]
    support: float
    confidence: float
    lift: float


def _normalize_transactions(
    transactions: list[list[str]], cancel: threading.Event
) -> tuple[list[str], list[set[int]]]:
    """Validate + intern each transaction's items; return ``(labels, normalized rows)``."""
    labels: list[str] = []
    indexes: dict[str, int] = {}
    normalized: list[set[int]] = []
    for raw_transaction in transactions:
        _check_cancel(cancel)
        transaction: set[int] = set()
        for item in raw_transaction:
            if not isinstance(item, str) or not item.startswith("eg:"):
                raise AnalyticsWorkerError("analytics items must be opaque references")
            if item not in indexes:
                indexes[item] = len(labels)
                labels.append(item)
            transaction.add(indexes[item])
        normalized.append(transaction)
    return labels, normalized


def _vertical_index(
    labels: list[str], normalized: list[set[int]], minimum_count: int
) -> dict[int, frozenset[int]]:
    """Item -> the frozenset of transaction indexes containing it, filtered by support."""
    vertical: dict[int, frozenset[int]] = {}
    for item_index in range(len(labels)):
        tids = frozenset(
            index for index, row in enumerate(normalized) if item_index in row
        )
        if len(tids) >= minimum_count:
            vertical[item_index] = tids
    return vertical


def _mine_itemsets(
    prefix: tuple[int, ...],
    atoms: list[tuple[int, frozenset[int]]],
    *,
    minimum_count: int,
    cancel: threading.Event,
    itemsets: dict[tuple[int, ...], int],
) -> None:
    """Depth-first vertical (Eclat-style) frequent-itemset extension; mutates ``itemsets``."""
    for index, (item, tids) in enumerate(atoms):
        _check_cancel(cancel)
        candidate = (*prefix, item)
        itemsets[candidate] = len(tids)
        children: list[tuple[int, frozenset[int]]] = []
        for next_item, next_tids in atoms[index + 1 :]:
            intersection = tids & next_tids
            if len(intersection) >= minimum_count:
                children.append((next_item, intersection))
        if children:
            _mine_itemsets(
                candidate,
                children,
                minimum_count=minimum_count,
                cancel=cancel,
                itemsets=itemsets,
            )


def _mask_split(
    itemset: tuple[int, ...], mask: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split ``itemset`` into ``(antecedent, consequent)`` by ``mask``'s bits."""
    antecedent = tuple(item for bit, item in enumerate(itemset) if mask & (1 << bit))
    consequent = tuple(
        item for bit, item in enumerate(itemset) if not mask & (1 << bit)
    )
    return antecedent, consequent


def _rule_for_split(
    itemset: tuple[int, ...],
    mask: int,
    itemsets: dict[tuple[int, ...], int],
    *,
    full_count: int,
    min_confidence: float,
    denominator: int,
) -> _Rule | None:
    """The rule for one antecedent/consequent split, or ``None`` if it doesn't qualify."""
    antecedent, consequent = _mask_split(itemset, mask)
    antecedent_count = itemsets.get(antecedent)
    consequent_count = itemsets.get(consequent)
    if antecedent_count is None or consequent_count is None:
        return None
    confidence = full_count / antecedent_count
    if confidence + 1e-12 < min_confidence:
        return None
    consequent_support = consequent_count / denominator
    return _Rule(
        antecedent=antecedent,
        consequent=consequent,
        support=full_count / denominator,
        confidence=confidence,
        lift=confidence / consequent_support if consequent_support else 0.0,
    )


def _itemset_rules(
    itemset: tuple[int, ...],
    itemsets: dict[tuple[int, ...], int],
    *,
    min_confidence: float,
    denominator: int,
) -> list[_Rule]:
    """Every valid antecedent/consequent split of one frequent itemset."""
    full_count = itemsets[itemset]
    rules: list[_Rule] = []
    for mask in range(1, (1 << len(itemset)) - 1):
        rule = _rule_for_split(
            itemset,
            mask,
            itemsets,
            full_count=full_count,
            min_confidence=min_confidence,
            denominator=denominator,
        )
        if rule is not None:
            rules.append(rule)
    return rules


def _rules_from_itemsets(
    itemsets: dict[tuple[int, ...], int],
    *,
    count: int,
    min_confidence: float,
    cancel: threading.Event,
) -> list[_Rule]:
    """Every rule meeting ``min_confidence``, sorted by the engine's stable rule order."""
    denominator = max(count, 1)
    rules: list[_Rule] = []
    for itemset in sorted(itemsets, key=lambda value: (len(value), value)):
        _check_cancel(cancel)
        if len(itemset) < 2:
            continue
        rules.extend(
            _itemset_rules(
                itemset,
                itemsets,
                min_confidence=min_confidence,
                denominator=denominator,
            )
        )
    rules.sort(
        key=lambda rule: (
            -rule.confidence,
            -rule.lift,
            -rule.support,
            rule.antecedent,
            rule.consequent,
        )
    )
    return rules


def _rules_to_result(rules: list[_Rule], labels: list[str]) -> list[dict[str, Any]]:
    """Render :class:`_Rule` objects to the governed association-rule row shape."""
    result: list[dict[str, Any]] = []
    for rule in rules:
        antecedent_labels = [labels[item] for item in rule.antecedent]
        consequent_labels = [labels[item] for item in rule.consequent]
        result.append(
            {
                "antecedent": antecedent_labels,
                "confidence": rule.confidence,
                "consequent": consequent_labels,
                "contradiction_ids": [],
                "evidence_refs": [],
                "id": _association_rule_id(
                    antecedent_labels,
                    consequent_labels,
                    rule.support,
                    rule.confidence,
                    rule.lift,
                ),
                "kind": "association_rule",
                "lift": rule.lift,
                "proof_ids": [],
                "source_refs": [],
                "support": rule.support,
            }
        )
    return result


def _association_rules(
    transactions: list[list[str]],
    min_support: float,
    min_confidence: float,
    cancel: threading.Event,
) -> list[dict[str, Any]]:
    """Exact vertical-set mining with the engine's stable rule semantics."""
    labels, normalized = _normalize_transactions(transactions, cancel)
    count = len(normalized)
    minimum_count = max(1, math.ceil(min_support * max(count, 1)))
    vertical = _vertical_index(labels, normalized, minimum_count)
    itemsets: dict[tuple[int, ...], int] = {}
    _mine_itemsets(
        (),
        sorted(vertical.items()),
        minimum_count=minimum_count,
        cancel=cancel,
        itemsets=itemsets,
    )
    rules = _rules_from_itemsets(
        itemsets, count=count, min_confidence=min_confidence, cancel=cancel
    )
    return _rules_to_result(rules, labels)


def _decode_payload_bytes(payload: Any) -> bytes:
    """The claimed job's raw input payload, as bytes."""
    if isinstance(payload, list) and all(
        isinstance(value, int) and 0 <= value <= 255 for value in payload
    ):
        return bytes(payload)
    if isinstance(payload, bytes):
        return payload
    raise AnalyticsWorkerError("claimed job has no governed input payload")


def _unpack_mine_associate(encoded: bytes) -> tuple[list[list[str]], float, float, str]:
    """Unpack the msgpack ``MineAssociate`` envelope. Raises on any malformed shape."""
    try:
        value = msgpack.unpackb(encoded, raw=False, strict_map_key=True)
        parameters = value["MineAssociate"]
        transactions = parameters["transactions"]
        min_support = float(parameters.get("min_support", 0.1))
        min_confidence = float(parameters.get("min_confidence", 0.5))
        algorithm = str(parameters.get("algorithm", "fpgrowth")).casefold()
    except (KeyError, TypeError, ValueError, msgpack.UnpackException) as exc:
        raise AnalyticsWorkerError(
            "claimed job payload is not MineAssociate v1"
        ) from exc
    return transactions, min_support, min_confidence, algorithm


def _transactions_malformed(transactions: Any) -> bool:
    """``True`` unless ``transactions`` is a list of lists of ``str``."""
    return not isinstance(transactions, list) or any(
        not isinstance(transaction, list)
        or any(not isinstance(item, str) for item in transaction)
        for transaction in transactions
    )


def _support_confidence_out_of_bounds(
    min_support: float, min_confidence: float
) -> bool:
    """``True`` iff either bound is non-finite or outside ``[0.0, 1.0]``."""
    return (
        not math.isfinite(min_support)
        or not 0.0 <= min_support <= 1.0
        or not math.isfinite(min_confidence)
        or not 0.0 <= min_confidence <= 1.0
    )


def _validate_mine_associate(
    transactions: Any, min_support: float, min_confidence: float, algorithm: str
) -> None:
    """Enforce the governed ``MineAssociate`` schema bounds. Raises if violated.

    Short-circuits in the SAME order as the original inline chain: the item-count
    check on the last line is only reachable once ``_transactions_malformed`` has
    already proven ``transactions`` is a well-formed list of lists of ``str``, so
    it is always safe to iterate.
    """
    if (
        _transactions_malformed(transactions)
        or _support_confidence_out_of_bounds(min_support, min_confidence)
        or algorithm not in _ALGORITHMS
        or len({item for transaction in transactions for item in transaction}) > 31
    ):
        raise AnalyticsWorkerError(
            "claimed association payload is outside its governed schema"
        )


def _decode_job(job: dict[str, Any]) -> tuple[list[list[str]], float, float, str]:
    payload = job.get("input_payload")
    encoded = _decode_payload_bytes(payload)
    transactions, min_support, min_confidence, algorithm = _unpack_mine_associate(
        encoded
    )
    _validate_mine_associate(transactions, min_support, min_confidence, algorithm)
    return transactions, min_support, min_confidence, algorithm


def _content_digest(schema: list[dict[str, Any]], rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    digest.update(b"eg-jobs.knowledge-batch.v1\0")
    digest.update(msgpack.packb(schema, use_bin_type=True))
    digest.update(b"\0")
    digest.update(msgpack.packb(rows, use_bin_type=True))
    return digest.hexdigest()


def _typed_result(job: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    snapshot = job["input_snapshot"]
    algorithm = job["algo"]
    policy = job["policy"]
    dataset_ref = str(snapshot["dataset_ref"])
    for row in rows:
        row["evidence_refs"] = [dataset_ref]
        row["source_refs"] = [dataset_ref]
    digest = _content_digest(_SCHEMA, rows)
    scores = [float(row["support"]) * float(row["confidence"]) for row in rows]
    return {
        "schema_version": 1,
        "dataset_ref": f"eg:knowledge_batch:{digest}",
        "content_digest": digest,
        "schema": _SCHEMA,
        "rows": rows,
        "evidence_refs": [dataset_ref],
        "counterexample_refs": [],
        "uncertainty": (
            sum(1.0 - float(row["confidence"]) for row in rows) / len(rows)
            if rows
            else None
        ),
        "calibration": [min(scores), max(scores)] if scores else None,
        "reproducibility": {
            "input_dataset_ref": dataset_ref,
            "input_content_digest": str(snapshot["content_digest"]),
            "input_snapshot_version": int(snapshot["version"]),
            "algorithm_ref": f"{algorithm['family']}:{algorithm['algorithm']}",
            "params_digest": str(algorithm["params_digest"]),
            "implementation_version": str(algorithm["code_version"]),
            "environment_version": str(algorithm["env_version"]),
            "policy_fingerprint": str(policy["policy_fingerprint"]),
        },
    }


def _is_publishing(job: dict[str, Any]) -> bool:
    state = job.get("state")
    return isinstance(state, dict) and "Publishing" in state


@dataclass(slots=True, frozen=True)
class _ClaimHandle:
    """A claimed job's identity + lease, bundled for the helpers below."""

    client: Any
    job_id: str
    worker_instance: str
    epoch: int
    lease_ms: int


async def _short_circuit_claim(
    client: Any, job: dict[str, Any], job_id: str, worker_instance: str, epoch: int
) -> bool:
    """Handle a claim that needs no mining (already cancelled/publishing).

    ``True`` iff handled -- the caller must return without proceeding further.
    """
    if job.get("cancel_requested") is True:
        await client.jobs.worker_cancel(job_id, worker_instance, epoch)
        return True
    if _is_publishing(job):
        await client.jobs.worker_publish(job_id, worker_instance, epoch)
        return True
    return False


async def _decode_claim_job(
    client: Any, job: dict[str, Any], job_id: str, worker_instance: str, epoch: int
) -> tuple[list[list[str]], float, float, str] | None:
    """Decode the job payload; on failure, mark it failed and return ``None``."""
    try:
        return _decode_job(job)
    except AnalyticsWorkerError:
        await client.jobs.worker_fail(job_id, worker_instance, epoch, "invalid_payload")
        return None


def _watchdog_reason(
    current: dict[str, Any], *, now_ms: int, began: float
) -> str | None:
    """Why the running kernel should be cancelled, or ``None`` if it should keep going."""
    deadline = (current.get("policy") or {}).get("deadline_unix_ms")
    budget = (current.get("policy") or {}).get("resources", {}).get("cpu_ms")
    budget = budget or (current.get("policy") or {}).get("quota_cpu_ms")
    if current.get("cancel_requested") is True:
        return "kernel_cancelled"
    if deadline is not None and now_ms >= int(deadline):
        return "deadline_exceeded"
    if budget is not None and (time.monotonic() - began) * 1000 >= int(budget):
        return "cpu_budget_exceeded"
    return None


async def _watch_kernel(
    handle: _ClaimHandle,
    kernel: asyncio.Task[Any],
    *,
    began: float,
    stop: asyncio.Event,
    cancel: threading.Event,
) -> str | None:
    """Renew the lease / observe cancellation-deadline-budget until the kernel finishes.

    Extracted from :func:`_execute_claim`. Returns the cancellation ``reason``, if
    any -- ``None`` means the kernel finished on its own with nothing pending.
    """
    # Lease renewal can remain at one-third of the lease, but cancellation and
    # deadline observation must not become a 20-100 second blind spot for long
    # leases. Five seconds bounds control-plane responsiveness and RPC load.
    interval = min(5.0, max(0.25, handle.lease_ms / 3000.0))
    reason: str | None = None
    try:
        while not kernel.done():
            if stop.is_set():
                raise asyncio.CancelledError
            try:
                await asyncio.wait_for(asyncio.shield(kernel), timeout=interval)
                break
            except TimeoutError:
                current = await handle.client.jobs.status(handle.job_id)
                now_ms = int(time.time() * 1000)
                reason = _watchdog_reason(current, now_ms=now_ms, began=began)
                if reason is not None:
                    cancel.set()
                else:
                    await handle.client.jobs.worker_renew(
                        handle.job_id,
                        handle.worker_instance,
                        handle.epoch,
                        lease_ms=handle.lease_ms,
                    )
    except BaseException:
        # A lost coordinator connection or process cancellation must not leave a
        # detached CPU-heavy thread running after its lease can no longer renew.
        cancel.set()
        try:
            await asyncio.shield(kernel)
        except BaseException:
            pass
        raise
    return reason


async def _finalize_reason(handle: _ClaimHandle, reason: str) -> None:
    """Dispatch a claim's terminal non-success outcome: cancel, or fail with ``reason``."""
    if reason == "kernel_cancelled":
        await handle.client.jobs.worker_cancel(
            handle.job_id, handle.worker_instance, handle.epoch
        )
    else:
        await handle.client.jobs.worker_fail(
            handle.job_id, handle.worker_instance, handle.epoch, reason
        )


async def _execute_claim(
    client: Any,
    claim: dict[str, Any],
    worker_instance: str,
    lease_ms: int,
    stop: asyncio.Event,
) -> None:
    job = claim["job"]
    lease = claim["lease"]
    job_id = str(job["job_id"])
    epoch = int(lease["epoch"])
    if await _short_circuit_claim(client, job, job_id, worker_instance, epoch):
        return
    cancel = threading.Event()
    began = time.monotonic()
    decoded = await _decode_claim_job(client, job, job_id, worker_instance, epoch)
    if decoded is None:
        return
    transactions, min_support, min_confidence, _algorithm = decoded
    await client.jobs.worker_checkpoint(
        job_id,
        worker_instance,
        epoch,
        progress=0.1,
        stage="mining",
    )
    kernel = asyncio.create_task(
        asyncio.to_thread(
            _association_rules,
            transactions,
            min_support,
            min_confidence,
            cancel,
        )
    )
    handle = _ClaimHandle(
        client=client,
        job_id=job_id,
        worker_instance=worker_instance,
        epoch=epoch,
        lease_ms=lease_ms,
    )
    reason = await _watch_kernel(handle, kernel, began=began, stop=stop, cancel=cancel)
    try:
        rows = await kernel
    except KernelCancelled:
        await _finalize_reason(handle, reason or "kernel_cancelled")
        return
    except Exception:  # noqa: BLE001 - bounded failure code; no payload logging
        await client.jobs.worker_fail(job_id, worker_instance, epoch, "kernel_failure")
        return
    if reason is not None:
        await _finalize_reason(handle, reason)
        return
    await client.jobs.worker_checkpoint(
        job_id,
        worker_instance,
        epoch,
        progress=0.9,
        stage="computed",
    )
    await client.jobs.worker_stage(
        job_id,
        worker_instance,
        epoch,
        _typed_result(job, rows),
    )
    await client.jobs.worker_publish(job_id, worker_instance, epoch)


async def _connect_slot_client(endpoint: str, transport_config: Any) -> Any:
    """Connect one slot's engine client. Extracted from :func:`_slot`."""
    from epistemic_graph import EpistemicGraphClient

    from agent_utilities.knowledge_graph.core.engine_transport import (
        engine_client_transport_kwargs,
        native_endpoint_address,
    )

    connect_kwargs = engine_client_transport_kwargs(endpoint, config=transport_config)
    return await EpistemicGraphClient.connect(
        tcp_addr=native_endpoint_address(endpoint)[0],
        auth_secret=_required("GRAPH_SERVICE_AUTH_SECRET"),
        verified_context=_verified_context(),
        **connect_kwargs,
    )


async def _drain_claims(
    client: Any,
    *,
    instance: str,
    capabilities: list[str],
    lease_ms: int,
    poll_seconds: float,
    stop: asyncio.Event,
) -> None:
    """Claim and execute jobs on ``client`` until ``stop`` fires. Extracted from :func:`_slot`."""
    while not stop.is_set():
        claim = await client.jobs.worker_claim(
            instance,
            capabilities,
            lease_ms=lease_ms,
        )
        if claim is None:
            try:
                await asyncio.wait_for(stop.wait(), timeout=poll_seconds)
            except TimeoutError:
                pass
            continue
        await _execute_claim(client, claim, instance, lease_ms, stop)


async def _slot(
    slot: int,
    *,
    lease_ms: int,
    poll_seconds: float,
    stop: asyncio.Event,
) -> None:
    from agent_utilities.core.config import AgentConfig

    instance = f"slot-{slot}-{secrets.token_hex(16)}"
    endpoints = _endpoints()
    transport_config = AgentConfig()
    capabilities = _capabilities()
    cursor = slot % len(endpoints)
    while not stop.is_set():
        client = None
        try:
            endpoint = endpoints[cursor % len(endpoints)]
            client = await _connect_slot_client(endpoint, transport_config)
            await _drain_claims(
                client,
                instance=instance,
                capabilities=capabilities,
                lease_ms=lease_ms,
                poll_seconds=poll_seconds,
                stop=stop,
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - reconnect without endpoint/payload logging
            cursor += 1
            try:
                await asyncio.wait_for(stop.wait(), timeout=min(5.0, poll_seconds * 4))
            except TimeoutError:
                pass
        finally:
            if client is not None:
                await client.close()


async def run(*, slots: int, lease_ms: int, poll_seconds: float) -> None:
    if not 1 <= slots <= 64:
        raise AnalyticsWorkerError("slots must be in 1..64")
    if not 5_000 <= lease_ms <= 300_000:
        raise AnalyticsWorkerError("lease-ms must be in 5000..300000")
    if not 0.05 <= poll_seconds <= 30.0:
        raise AnalyticsWorkerError("poll-seconds must be in 0.05..30")
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for event in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(event, stop.set)
        except NotImplementedError:  # pragma: no cover - Windows event-loop contract
            signal.signal(event, lambda *_: loop.call_soon_threadsafe(stop.set))
    tasks = [
        asyncio.create_task(
            _slot(
                slot,
                lease_ms=lease_ms,
                poll_seconds=poll_seconds,
                stop=stop,
            )
        )
        for slot in range(slots)
    ]
    try:
        await asyncio.gather(*tasks)
    finally:
        stop.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="graph-os-analytics-worker")
    parser.add_argument(
        "--slots",
        type=int,
        default=int(setting("EG_ANALYTICS_WORKER_SLOTS", 1)),
    )
    parser.add_argument(
        "--lease-ms",
        type=int,
        default=int(setting("EG_ANALYTICS_WORKER_LEASE_MS", 60_000)),
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=float(setting("EG_ANALYTICS_WORKER_POLL_SECONDS", 0.25)),
    )
    args = parser.parse_args(argv)
    try:
        asyncio.run(
            run(
                slots=args.slots,
                lease_ms=args.lease_ms,
                poll_seconds=args.poll_seconds,
            )
        )
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001 - never emit endpoints, payloads or identity
        print(json.dumps({"ok": False, "error": type(exc).__name__}, sort_keys=True))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
