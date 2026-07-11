"""CONCEPT:AU-KG.memory.episodic-procedural-memory-distillation — Episodic→procedural memory distillation (AU-side).

The parametric-memory module of the agent-native-memory paper. Where the lifecycle
loop (CONCEPT:AU-KG.memory.drive-one-agent-native, :mod:`agent_utilities.knowledge_graph.memory.lifecycle`)
drives the engine's summarize/consolidate/maintain primitives over EPISODIC memory,
this is the *distillation* phase: it mines RECURRING clusters of related
episodic/semantic memories — a routine that keeps happening — and distills them into
a reusable PROCEDURAL artifact (a named skill/pattern with steps + preconditions),
then folds that artifact into the AU evolution flywheel so it is reward-reinforced
when it later helps.

Pipeline (one ``run_distillation`` cycle):

  * **select** recurring clusters (:meth:`select_recurring_clusters`) — groups of
    ACTIVE episodic/semantic memories on the same ``cluster_key`` whose recurrence
    (member count) clears ``min_recurrence``. Localized: it reuses the KG-2.307
    lifecycle's BOUNDED working-set reader, never a global scan.
  * **distill** each cluster (:meth:`distill`) — the LLM turns the cluster into a
    :class:`ProceduralArtifact` (name / intent / steps / preconditions / tags). A
    clean, resilient JSON parse with a prose fallback; no LLM ⇒ a safe skip.
  * **record** the artifact as PROCEDURAL memory via the existing
    ``engine.store_memory`` write path (``memory_type="procedural"``), tagging its
    provenance (source memory ids + concept).
  * **register** it with the evolution flywheel (:meth:`_register_flywheel`) —
    seeds a reward-EMA entry on the capability index keyed ``procedural:<slug>`` so
    ``FeedbackService.record_action_outcome`` later reinforces the artifact when it
    is used and helps (reward-weighted self-optimization, DSPy-style).

Design notes (mirrors KG-2.307):

  * **Additive + off by default.** ``MemoryDistillerConfig.enabled`` defaults to
    ``False`` (env ``AGENT_UTILITIES_MEMORY_DISTILL``). ``run_distillation`` is a
    no-op until enabled, so it stays inert even if scheduled.
  * **Reuses existing seams.** Working-set read + clustering compose the KG-2.307
    :class:`MemoryLifecycle` (never edited); LLM calls default to the shared
    ``memento_compressor._memento_llm`` one-shot helper; writes go through
    ``engine.store_memory``; the flywheel is the same ``CapabilityIndex`` the router
    and :class:`FeedbackService` already use.
  * **Injectable LLM + flywheel seams; clean fallback with no LLM.** Tests inject a
    stub LLM/flywheel; with no model the distiller degrades to a safe skip and
    never raises into the scheduler. A repeated cluster (stable signature) is
    deduped so a re-run does not re-distill it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .lifecycle import MemoryLifecycle, MemoryLifecycleConfig

logger = logging.getLogger(__name__)

# Memory states that are retired / superseded — never distilled from.
_INACTIVE_STATUSES = frozenset(
    {"ARCHIVED", "MERGED", "RETIRED", "DELETED", "SUPERSEDED"}
)

# Memory tiers a recurring routine is distilled FROM (the observed history).
_SOURCE_TYPES = frozenset({"episodic", "semantic"})


def _truthy(val: str | None) -> bool:
    return str(val or "").strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    return _truthy(val) if val is not None else default


# The LLM is asked for a strict JSON procedural artifact. Terse, no invented steps
# beyond what the episodes support; the artifact must be reusable across contexts.
_DISTILL_SYSTEM_PROMPT = (
    "You are the procedural-memory distiller for an autonomous agent. You are given "
    "a cluster of related EPISODIC/SEMANTIC memories describing a routine the agent "
    "keeps performing. Distill them into ONE reusable PROCEDURAL artifact: a named "
    "skill/pattern with ordered steps and the preconditions under which it applies. "
    "Generalize away incidental detail; invent nothing beyond what the memories "
    "support. Respond with ONLY a JSON object with keys: "
    '"name" (short imperative title), "intent" (one sentence), '
    '"preconditions" (array of strings), "steps" (array of ordered step strings), '
    '"tags" (array of short keyword strings). No prose outside the JSON.'
)


@dataclass
class ProceduralArtifact:
    """A distilled reusable procedure (CONCEPT:AU-KG.memory.episodic-procedural-memory-distillation).

    The parametric-memory output: a named routine with ordered steps and the
    preconditions under which it applies, plus the provenance (source memory ids)
    of the episodes it was distilled from.
    """

    name: str
    intent: str = ""
    steps: list[str] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)

    @property
    def slug(self) -> str:
        """Stable lowercase identifier slug derived from the artifact name."""
        s = re.sub(r"[^a-z0-9]+", "-", self.name.strip().lower()).strip("-")
        return s or "procedure"

    def render(self, max_chars: int = 4000) -> str:
        """Render the artifact as a readable procedural document for storage."""
        lines = [f"PROCEDURE: {self.name}"]
        if self.intent:
            lines.append(f"INTENT: {self.intent}")
        if self.preconditions:
            lines.append("PRECONDITIONS:")
            lines.extend(f"- {p}" for p in self.preconditions)
        if self.steps:
            lines.append("STEPS:")
            lines.extend(f"{i}. {s}" for i, s in enumerate(self.steps, start=1))
        return "\n".join(lines)[:max_chars]


@dataclass
class MemoryDistillerConfig:
    """Policy knobs for the distillation phase (CONCEPT:AU-KG.memory.episodic-procedural-memory-distillation).

    Additive + conservative; ``enabled`` gates the whole phase off by default so the
    component is inert until an operator opts in.
    """

    enabled: bool = False
    # Bound the localized working set read each cycle (never a global scan).
    max_working_set: int = 500
    # A cluster is a distillation candidate once it RECURS this many times — i.e. at
    # least this many related source memories on the same ``cluster_key``.
    min_recurrence: int = 3
    # Cap episodes fed into one distillation (token budget + latency).
    max_cluster_size: int = 32
    # Cap the number of clusters distilled per cycle (bounded work).
    max_clusters_per_cycle: int = 4
    # Node property clustered on (memories about the same routine distill together);
    # falls back to ``category`` then ``"general"``.
    cluster_key: str = "target_entity"
    # Rendered procedural text is truncated to this many chars before it is stored.
    artifact_max_chars: int = 4000
    # Register the distilled artifact with the reward-EMA flywheel.
    register_flywheel: bool = True
    # Neutral-ish seed reward for a freshly distilled (as-yet unproven) artifact.
    seed_reward: float = 0.5
    system_prompt: str = _DISTILL_SYSTEM_PROMPT
    # Extra metadata stamped onto procedural memory nodes.
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> MemoryDistillerConfig:
        """Build a config from ``AGENT_UTILITIES_MEMORY_DISTILL*`` env vars."""
        return cls(
            enabled=_env_bool("AGENT_UTILITIES_MEMORY_DISTILL"),
            max_working_set=_env_int("AGENT_UTILITIES_MEMORY_DISTILL_WORKING_SET", 500),
            min_recurrence=_env_int("AGENT_UTILITIES_MEMORY_DISTILL_MIN_RECURRENCE", 3),
            max_cluster_size=_env_int("AGENT_UTILITIES_MEMORY_DISTILL_MAX_CLUSTER", 32),
            max_clusters_per_cycle=_env_int(
                "AGENT_UTILITIES_MEMORY_DISTILL_MAX_CLUSTERS", 4
            ),
            register_flywheel=_env_bool(
                "AGENT_UTILITIES_MEMORY_DISTILL_FLYWHEEL", True
            ),
            seed_reward=_env_float("AGENT_UTILITIES_MEMORY_DISTILL_SEED_REWARD", 0.5),
        )


class MemoryDistiller:
    """Distill recurring episodic/semantic clusters into procedural memory (KG-2.309).

    Instantiate with the same ``engine`` the scheduler hands maintenance handlers
    (it exposes ``.backend``, ``.store_memory`` and — for the flywheel — a capability
    index). ``llm`` is an injectable one-shot completion callable
    ``(system_prompt, user_content) -> str | None`` (defaults to the shared
    ``memento_compressor._memento_llm``). ``flywheel`` is an injectable object
    exposing ``record_outcome(id, reward=...)`` (defaults to the engine's capability
    index, resolved via :class:`FeedbackService`); tests inject stubs for both.
    """

    def __init__(
        self,
        engine: Any,
        config: MemoryDistillerConfig | None = None,
        llm: Callable[[str, str], str | None] | None = None,
        flywheel: Any = None,
    ) -> None:
        self.engine = engine
        self.config = config or MemoryDistillerConfig.from_env()
        self._llm = llm
        self._flywheel = flywheel
        # Reuse the KG-2.307 lifecycle purely for its BOUNDED working-set reader —
        # never edited, never drives its own primitives here.
        self._lifecycle = MemoryLifecycle(
            engine,
            config=MemoryLifecycleConfig(
                enabled=False, max_working_set=self.config.max_working_set
            ),
        )
        # Stable signatures of clusters already distilled by this instance — makes a
        # repeated cycle idempotent even before the memories churn out of the set.
        self._processed: set[str] = set()

    # ── LLM distillation call ─────────────────────────────────────────────────
    def _run_llm(self, system_prompt: str, user_content: str) -> str | None:
        if self._llm is not None:
            return self._llm(system_prompt, user_content)
        from .memento_compressor import _memento_llm

        return _memento_llm(system_prompt, user_content)

    # ── Working-set read (localized, bounded — reuses KG-2.307) ───────────────
    def _read_working_set(self, now: datetime) -> list[dict[str, Any]]:
        return self._lifecycle._read_working_set(now)

    # ── Candidate selection ───────────────────────────────────────────────────
    def select_recurring_clusters(
        self,
        nodes: list[dict[str, Any]] | None = None,
        now: datetime | None = None,
    ) -> list[list[dict[str, Any]]]:
        """Return recurring clusters ripe for distillation, largest first.

        A *recurring* cluster (CONCEPT:AU-KG.memory.episodic-procedural-memory-distillation) is a group of ACTIVE
        episodic/semantic memories on the same ``cluster_key`` (→ ``category`` →
        ``general``) whose member count reaches ``min_recurrence`` — evidence of a
        routine the agent keeps performing. Members within each cluster are ordered
        deterministically (by id) so the cluster signature is stable, and the
        clusters are returned largest-recurrence-first, capped at
        ``max_clusters_per_cycle``.
        """
        now = now or datetime.now(UTC)
        if nodes is None:
            nodes = self._read_working_set(now)

        groups: dict[str, list[dict[str, Any]]] = {}
        for n in nodes:
            if str(n.get("memory_type", "")).lower() not in _SOURCE_TYPES:
                continue
            if str(n.get("status", "ACTIVE")).upper() in _INACTIVE_STATUSES:
                continue
            if not (n.get("content") or n.get("name")):
                continue
            key = str(n.get(self.config.cluster_key) or n.get("category") or "general")
            groups.setdefault(key, []).append(n)

        ripe: list[list[dict[str, Any]]] = []
        for group in groups.values():
            if len(group) < self.config.min_recurrence:
                continue
            group.sort(key=lambda g: str(g.get("id", "")))
            ripe.append(group[: max(1, int(self.config.max_cluster_size))])

        ripe.sort(key=lambda g: (-len(g), self._signature(g)))
        return ripe[: max(1, int(self.config.max_clusters_per_cycle))]

    @staticmethod
    def _cluster_ids(cluster: list[dict[str, Any]]) -> list[str]:
        return [str(n.get("id", "")) for n in cluster if n.get("id")]

    def _signature(self, cluster: list[dict[str, Any]]) -> str:
        joined = "|".join(sorted(self._cluster_ids(cluster)))
        return hashlib.sha1(joined.encode(), usedforsecurity=False).hexdigest()

    def _cluster_text(self, cluster: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for n in cluster:
            text = str(n.get("content") or n.get("name") or "").strip()
            if not text:
                continue
            tags = n.get("tags") or []
            tag_s = f" [tags: {', '.join(map(str, tags))}]" if tags else ""
            parts.append(f"- ({n.get('id', '?')}){tag_s} {text}")
        return "\n".join(parts)

    # ── Artifact parsing (resilient JSON → prose fallback) ────────────────────
    @staticmethod
    def _coerce_str_list(val: Any) -> list[str]:
        if isinstance(val, list):
            return [str(x).strip() for x in val if str(x).strip()]
        if isinstance(val, str) and val.strip():
            return [val.strip()]
        return []

    def _parse_artifact(
        self, raw: str, source_ids: list[str], fallback_name: str
    ) -> ProceduralArtifact | None:
        """Parse the LLM response into a :class:`ProceduralArtifact`, or ``None``.

        Tolerates a fenced ```json block or surrounding prose. If JSON parsing
        fails but there IS content, degrades to a single-step artifact built from
        the raw prose so a usable procedure is still captured. Returns ``None`` only
        when the response is genuinely empty or has no usable step content.
        """
        text = (raw or "").strip()
        if not text:
            return None

        obj: Any = None
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(0))
            except (ValueError, TypeError):
                obj = None

        if isinstance(obj, dict):
            steps = self._coerce_str_list(obj.get("steps"))
            name = str(obj.get("name") or "").strip()
            if steps or name:
                return ProceduralArtifact(
                    name=name or fallback_name,
                    intent=str(obj.get("intent") or "").strip(),
                    steps=steps,
                    preconditions=self._coerce_str_list(obj.get("preconditions")),
                    tags=self._coerce_str_list(obj.get("tags")),
                    source_ids=source_ids,
                )

        # Prose fallback — capture non-empty lines as steps so nothing is lost.
        lines = [ln.strip("-* \t") for ln in text.splitlines() if ln.strip()]
        if not lines:
            return None
        return ProceduralArtifact(
            name=fallback_name,
            intent=lines[0][:200],
            steps=lines,
            source_ids=source_ids,
        )

    # ── Flywheel registration (reward-EMA seed) ───────────────────────────────
    def _resolve_flywheel(self) -> Any:
        if self._flywheel is not None:
            return self._flywheel
        try:
            from ..adaptation.feedback import FeedbackService

            return FeedbackService.from_engine(self.engine).capability_index
        except Exception as e:  # noqa: BLE001 — flywheel is optional, degrade quietly
            logger.debug("[KG-2.309] flywheel resolution failed: %s", e)
            return None

    def _register_flywheel(self, capability_id: str) -> dict[str, Any]:
        """Seed a reward-EMA entry so the artifact is reward-reinforced later.

        Registers ``capability_id`` (``procedural:<slug>``) on the capability index
        with a neutral-ish seed reward. Once the artifact is USED and helps,
        ``FeedbackService.record_action_outcome`` moves the same EMA up (DSPy-style
        reward-weighted self-optimization); if it hurts, the EMA falls.
        """
        if not self.config.register_flywheel:
            return {"status": "skipped", "reason": "disabled"}
        flywheel = self._resolve_flywheel()
        record = getattr(flywheel, "record_outcome", None)
        if not callable(record):
            return {"status": "skipped", "reason": "no_flywheel"}
        try:
            reward = record(capability_id, reward=self.config.seed_reward)
            return {"status": "ok", "capability_id": capability_id, "reward": reward}
        except Exception as e:  # noqa: BLE001 — never abort a distillation on this
            logger.warning("[KG-2.309] flywheel register failed: %s", e)
            return {"status": "skipped", "reason": f"error:{e}"}

    # ── Recording (procedural memory write) ───────────────────────────────────
    def _record_artifact(self, artifact: ProceduralArtifact) -> dict[str, Any]:
        """Store the artifact as PROCEDURAL memory via ``engine.store_memory``."""
        store = getattr(self.engine, "store_memory", None)
        if not callable(store):
            return {"status": "skipped", "reason": "no_store_memory"}
        capability_id = f"procedural:{artifact.slug}"
        content = artifact.render(self.config.artifact_max_chars)
        extra_props: dict[str, Any] = {
            "artifact_name": artifact.name,
            "capability_id": capability_id,
            "concept": "AU-KG.memory.episodic-procedural-memory-distillation",
            "source": "memory_distiller",
            "step_count": len(artifact.steps),
            "source_count": len(artifact.source_ids),
            "distilled_from": ",".join(artifact.source_ids),
            **dict(self.config.extra_metadata),
        }
        try:
            memory_id = store(
                content=content,
                memory_type="procedural",
                name=artifact.name,
                tags=list(artifact.tags),
                extra_props=extra_props,
            )
        except Exception as e:  # noqa: BLE001 — degrade, never abort the cycle
            logger.warning("[KG-2.309] store_memory failed: %s", e)
            return {"status": "skipped", "reason": f"store_error:{e}"}
        return {
            "status": "ok",
            "memory_id": memory_id,
            "capability_id": capability_id,
        }

    # ── Distill one cluster ───────────────────────────────────────────────────
    def distill(self, cluster: list[dict[str, Any]]) -> dict[str, Any]:
        """Distill one recurring cluster into a procedural artifact (CONCEPT:AU-KG.memory.episodic-procedural-memory-distillation).

        LLM-generates the artifact, records it as procedural memory, and (if
        enabled) registers it with the reward-EMA flywheel. Never raises: a missing
        LLM / empty response / unavailable write path all degrade to a safe
        ``{"status": "skipped", ...}``.
        """
        source_ids = self._cluster_ids(cluster)
        if not source_ids:
            return {"status": "skipped", "reason": "empty_cluster"}

        fallback_name = str(
            cluster[0].get(self.config.cluster_key)
            or cluster[0].get("category")
            or "recurring-routine"
        )
        raw = self._run_llm(
            self.config.system_prompt,
            "Distill this recurring routine into a reusable procedure:\n"
            + self._cluster_text(cluster),
        )
        if not raw or not str(raw).strip():
            return {"status": "skipped", "reason": "no_llm", "source_ids": source_ids}

        artifact = self._parse_artifact(str(raw), source_ids, fallback_name)
        if artifact is None:
            return {
                "status": "skipped",
                "reason": "no_artifact",
                "source_ids": source_ids,
            }

        recorded = self._record_artifact(artifact)
        if recorded.get("status") != "ok":
            return {
                "status": "skipped",
                "reason": recorded.get("reason", "record_failed"),
                "artifact_name": artifact.name,
                "source_ids": source_ids,
            }

        flywheel = self._register_flywheel(recorded["capability_id"])
        return {
            "status": "ok",
            "artifact_name": artifact.name,
            "memory_id": recorded.get("memory_id"),
            "capability_id": recorded["capability_id"],
            "step_count": len(artifact.steps),
            "source_ids": source_ids,
            "flywheel": flywheel,
        }

    # ── The scheduled entry point ─────────────────────────────────────────────
    def run_distillation(self, now: datetime | None = None) -> dict[str, Any]:
        """One distillation cycle — the entry the lifecycle loop can call (KG-2.309).

        Safe + idempotent: a no-op when disabled, never raises into the scheduler,
        and skips re-distilling a cluster already processed by this instance.
        """
        if not self.config.enabled:
            return {"status": "disabled"}
        now = now or datetime.now(UTC)
        result: dict[str, Any] = {"status": "ok", "distilled": 0, "artifacts": []}
        try:
            working_set = self._read_working_set(now)
            result["working_set"] = len(working_set)

            clusters = self.select_recurring_clusters(working_set, now)
            result["candidates"] = len(clusters)
            for cluster in clusters:
                sig = self._signature(cluster)
                if sig in self._processed:
                    continue
                outcome = self.distill(cluster)
                if outcome.get("status") == "ok":
                    result["distilled"] = int(result["distilled"]) + 1
                    result["artifacts"].append(outcome)
                    # Mark processed once we have distilled+stored the artifact, so a
                    # repeat cycle over the same cluster is a no-op.
                    self._processed.add(sig)
            return result
        except Exception as e:  # noqa: BLE001 — must never raise into the scheduler
            logger.error("[KG-2.309] memory distillation cycle failed: %s", e)
            return {"status": "error", "error": str(e)}


def run_memory_distillation(
    engine: Any,
    *,
    now: datetime | None = None,
    config: MemoryDistillerConfig | None = None,
) -> dict[str, Any]:
    """CLI/daemon/lifecycle entry for one distillation cycle (CONCEPT:AU-KG.memory.episodic-procedural-memory-distillation).

    Gated off unless ``AGENT_UTILITIES_MEMORY_DISTILL`` is set, so it is inert even
    if a (default-disabled) schedule is turned on. Callable from the KG-2.307
    lifecycle loop as a post-consolidation hook.
    """
    return MemoryDistiller(engine, config=config).run_distillation(now=now)
