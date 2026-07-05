#!/usr/bin/python
from __future__ import annotations

"""Operational action policy — the single autonomy decision point.

CONCEPT:AU-OS.governance.action-policy-decision-point — Operational ActionPolicy decision point with per-action autonomy tiers, durable rate limits, maintenance windows, and blast-radius caps consulted before any autonomous mutating fleet action

Until now every autonomy gate in the platform was a binary env flag
(``KG_GOLDEN_AUTO_MERGE``, ``FLEET_RECONCILER`` …) — an action was either
fully autonomous or fully off. This module replaces that cliff with per-action
*tiers* for operational actions (restart/scale/deploy/playbook), the
operational sibling of the AHE-3.20 promotion-governance validator — which
has since adopted this decision point too: ``GovernedAutoMerger`` consults
the reserved ``merge_promotion`` kind before any proposal→active lifecycle
flip (see ``docs/architecture/fleet_autonomy.md``).

One call decides everything::

    decision = get_action_policy(engine).decide(
        ActionRequest(kind="restart_service", target="caddy-mcp", source="reconciler")
    )

Decision pipeline (most-specific rule wins; KG rules beat file rules):

1. **rule match** — policies load from YAML (``AgentConfig.action_policy_path``,
   default = the shipped conservative ``deploy/action-policy.default.yml``)
   plus KG-stored ``governance_rule`` overrides (``scope='action_policy'``).
2. **tier** — ``auto`` | ``auto_notify`` | ``approval_required`` | ``forbidden``.
3. **maintenance window** — an auto tier outside the rule's UTC window is
   downgraded to queue-approval (a human can still push it through).
4. **rate limit** — more than ``max`` allowed executions for the same
   action+target inside ``window_s`` ⇒ deny (a looping remediation is broken;
   denying is safer than flooding the approval queue).
5. **blast radius** — more than ``max_targets`` distinct targets for the same
   action kind inside ``window_s`` ⇒ queue-approval (a wide change wave needs
   a human, not a refusal).

Every decision is audit-logged as an ``ActionDecision`` KG node; rate/blast
accounting reads those same nodes back, so the ledger is durable and shared
across processes. Queue-approval reuses the existing fleet approvals flow:
it files an ``ActionApproval`` node that ``GET /api/fleet/approvals`` lists
and ``POST /api/fleet/approvals/grant`` resolves; the fleet reconciler tick
executes granted entries (CONCEPT:AU-OS.config.desired-state-fleet-reconciler).
"""

import fnmatch
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Tiers (what a rule allows) and decisions (what the gate answers).
TIER_AUTO = "auto"
TIER_AUTO_NOTIFY = "auto_notify"
TIER_APPROVAL = "approval_required"
TIER_FORBIDDEN = "forbidden"
_TIERS = {TIER_AUTO, TIER_AUTO_NOTIFY, TIER_APPROVAL, TIER_FORBIDDEN}

DECISION_ALLOW = "allow"
DECISION_ALLOW_NOTIFY = "allow_notify"
DECISION_QUEUE = "queue_approval"
DECISION_DENY = "deny"

_ALLOWING = {DECISION_ALLOW, DECISION_ALLOW_NOTIFY}

# How many recent audit rows the durable rate/blast accounting scans.
_LEDGER_SCAN_LIMIT = 500

# The conservative shipped policy (kept byte-for-byte in sync with
# ``deploy/action-policy.default.yml`` — tests assert the parity) so an
# installed wheel without the repo's ``deploy/`` tree gets identical behavior.
DEFAULT_POLICY: dict[str, Any] = {
    "version": 1,
    "defaults": {
        "tier": TIER_APPROVAL,
        "rate_limit": {"max": 3, "window_s": 3600},
        "blast_radius": {"max_targets": 3, "window_s": 3600},
    },
    "rules": [
        # No-op / diagnostic kinds run automatically; everything mutating
        # falls through to the approval_required default.
        {"kind": "diagnose", "target": "*", "tier": TIER_AUTO},
        {"kind": "observe", "target": "*", "tier": TIER_AUTO},
        {"kind": "notify", "target": "*", "tier": TIER_AUTO},
        # Outbound user messaging (CONCEPT:AU-ECO.messaging.messaging-reach-service-governed): governed but default-on —
        # auto+notify so reaching the user is auditable; operators may tighten a
        # ``message.send`` rule to approval_required for a stricter posture.
        {"kind": "message.send", "target": "*", "tier": TIER_AUTO_NOTIFY},
        # Reactions/emotes (CONCEPT:AU-ECO.reactions.one-emote-registry-governance) are cosmetic, low-risk output any
        # entrypoint may render — auto by default. Tighten to forbidden to disable
        # reactions for a principal/context, or scope by target (the channel/surface).
        {"kind": "reaction", "target": "*", "tier": TIER_AUTO},
        # Agent-to-agent bus traffic (CONCEPT:AU-ECO.bus.agentbus-federated-agent-agent): a message between sessions is
        # auto+notify so it's auditable but frictionless; a dispatch (CONCEPT:AU-ORCH.routing.resolve-body-single-canonical)
        # hands an objective to the fleet — also auto+notify, since the Loop's own mutating
        # steps re-enter this gate individually (same posture as run_playbook). Tighten a
        # ``bus.send``/``bus.dispatch`` rule to approval_required for a stricter posture.
        {"kind": "bus.send", "target": "*", "tier": TIER_AUTO_NOTIFY},
        {"kind": "bus.dispatch", "target": "*", "tier": TIER_AUTO_NOTIFY},
        {"kind": "record_dry_run", "target": "*", "tier": TIER_AUTO},
        # Sandboxed developer-workspace actions (CONCEPT:AU-OS.scaling.bridge-developer-workspace-mutating) default to
        # auto: the workspace container/process IS the containment boundary.
        # Operators override any of these to approval_required to require human
        # sign-off on in-sandbox shell/file mutations.
        {"kind": "workspace.cmd", "target": "*", "tier": TIER_AUTO},
        {"kind": "workspace.write", "target": "*", "tier": TIER_AUTO},
        {"kind": "workspace.edit", "target": "*", "tier": TIER_AUTO},
        {"kind": "workspace.browse", "target": "*", "tier": TIER_AUTO},
        # GUI computer-use (CONCEPT:AU-OS.governance.action-policy-rule): driving a desktop in a gui-sandbox
        # container. The sandbox is the containment boundary (like the other
        # workspace.* actions), but a click/type is more notable than a file write
        # and per-click approval would make the agent loop unusable — so auto+notify:
        # the loop runs, every desktop mutation is audited. (Capture is read-only and
        # bypasses the gate.) Tighten to approval_required for human sign-off on input.
        {"kind": "workspace.computer_use", "target": "*", "tier": TIER_AUTO_NOTIFY},
        # Playbook *dispatch* is auto+notify: a playbook's mutating steps
        # re-enter this gate individually with their concrete kinds.
        {"kind": "run_playbook", "target": "*", "tier": TIER_AUTO_NOTIFY},
        {"kind": "restart_service", "target": "*", "tier": TIER_APPROVAL},
        {"kind": "rollback_service", "target": "*", "tier": TIER_APPROVAL},
        {"kind": "scale_service", "target": "*", "tier": TIER_APPROVAL},
        {"kind": "deploy_service", "target": "*", "tier": TIER_APPROVAL},
        {"kind": "redeploy_stack", "target": "*", "tier": TIER_APPROVAL},
        {"kind": "merge_promotion", "target": "*", "tier": TIER_APPROVAL},
        # Spec-level review/veto (CONCEPT:AU-OS.config.autonomous-spec-develop-off): a distilled :SpecProposal must
        # clear this gate before it becomes a develop Loop — the EARLY checkpoint
        # before any code is synthesized (merge_promotion is the LATE, publish-time
        # gate). Default approval_required so the 24/7 loop holds specs for Claude/
        # human review; relax this tier to auto-develop selected specs.
        {"kind": "spec_promotion", "target": "*", "tier": TIER_APPROVAL},
        # Secret-store mutations (CONCEPT:AU-OS.identity.encrypted-secret-store): writing or deleting a secret in
        # the engine-encrypted ``__secrets__`` store is sensitive — approval_required.
        # (Reads — secret.get / secret.list — are not gated, mirroring read posture.)
        {"kind": "secret.set", "target": "*", "tier": TIER_APPROVAL},
        {"kind": "secret.delete", "target": "*", "tier": TIER_APPROVAL},
        # Retained run-output review gate (CONCEPT:AU-ORCH.runvcs.retained-output-gate):
        # a completed run's world delta is HELD as a proposal and materialized only on
        # accept. ``run.select`` (materialize the held fs/KG delta into the real world) is
        # the review itself — approval_required. ``run.discard`` (drop the proposal, world
        # untouched) is always safe — auto.
        {"kind": "run.select", "target": "*", "tier": TIER_APPROVAL},
        {"kind": "run.discard", "target": "*", "tier": TIER_AUTO},
    ],
}


def _now() -> float:
    return time.time()


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class ActionRequest:
    """One proposed autonomous operational action, pre-decision."""

    kind: str
    target: str
    params: dict[str, Any] = field(default_factory=dict)
    source: str = "manual"  # reconciler | playbook | deploy_watch | manual | ...
    reason: str = ""
    actor_id: str = ""  # CONCEPT:AU-OS.governance.autonomy-change-proposer — who proposes it (for the autonomy ramp)

    def summary(self) -> str:
        return f"{self.kind}({self.target})" + (
            f" — {self.reason}" if self.reason else ""
        )


@dataclass
class ActionRule:
    """One policy rule: who may do what, how fast, how wide, and when."""

    kind: str = "*"
    target: str = "*"
    tier: str = TIER_APPROVAL
    rate_limit: dict[str, Any] | None = None  # {max, window_s}
    blast_radius: dict[str, Any] | None = None  # {max_targets, window_s}
    maintenance_window: str | None = None  # "HH:MM-HH:MM" UTC
    origin: str = "file"  # file | kg
    priority: int = 0

    def matches(self, request: ActionRequest) -> bool:
        return fnmatch.fnmatchcase(request.kind, self.kind) and fnmatch.fnmatchcase(
            request.target, self.target
        )


@dataclass
class ActionDecision:
    """The gate's answer, mirrored into the ``ActionDecision`` audit node."""

    decision: str
    tier: str
    request: ActionRequest
    reason: str = ""
    rule_origin: str = "default"
    approval_id: str | None = None
    audit_id: str | None = None

    @property
    def allowed(self) -> bool:
        return self.decision in _ALLOWING


def _parse_window(window: str) -> tuple[int, int] | None:
    """Parse ``"HH:MM-HH:MM"`` into start/end minutes-of-day (UTC)."""
    try:
        start_s, end_s = str(window).split("-", 1)
        sh, sm = (list(map(int, start_s.strip().split(":"))) + [0])[:2]
        eh, em = (list(map(int, end_s.strip().split(":"))) + [0])[:2]
        return sh * 60 + sm, eh * 60 + em
    except (ValueError, AttributeError):
        logger.warning("action_policy: unparseable maintenance_window %r", window)
        return None


def in_maintenance_window(window: str | None, now: float | None = None) -> bool:
    """True when ``now`` (UTC) falls inside the ``HH:MM-HH:MM`` window.

    No window configured ⇒ always inside (no time restriction). Windows that
    wrap midnight (``22:00-04:00``) are supported.
    """
    if not window:
        return True
    parsed = _parse_window(window)
    if parsed is None:
        return True  # fail open on config typo — the tier still gates
    start, end = parsed
    t = time.gmtime(now if now is not None else _now())
    minute = t.tm_hour * 60 + t.tm_min
    if start <= end:
        return start <= minute < end
    return minute >= start or minute < end


def _coerce_rule(raw: dict[str, Any], origin: str) -> ActionRule | None:
    if not isinstance(raw, dict):
        return None
    tier = str(raw.get("tier") or TIER_APPROVAL)
    if tier not in _TIERS:
        logger.warning("action_policy: unknown tier %r — treating as approval", tier)
        tier = TIER_APPROVAL
    return ActionRule(
        kind=str(raw.get("kind") or "*"),
        target=str(raw.get("target") or "*"),
        tier=tier,
        rate_limit=raw.get("rate_limit")
        if isinstance(raw.get("rate_limit"), dict)
        else None,
        blast_radius=raw.get("blast_radius")
        if isinstance(raw.get("blast_radius"), dict)
        else None,
        maintenance_window=raw.get("maintenance_window"),
        origin=origin,
        priority=int(raw.get("priority") or 0),
    )


def resolve_policy_path(explicit: str | None = None) -> Path | None:
    """Resolve the policy YAML: explicit flag → repo's shipped default → None."""
    if explicit:
        return Path(explicit)
    shipped = (
        Path(__file__).resolve().parents[2] / "deploy" / "action-policy.default.yml"
    )
    return shipped if shipped.is_file() else None


class ActionPolicy:
    """Single policy decision point for autonomous operational actions.

    Stateless apart from a mtime-cached parse of the policy file; all
    accounting (rate limits, blast radius) reads the durable ``ActionDecision``
    audit ledger in the KG, so N processes share one budget.
    """

    def __init__(self, engine: Any = None, policy_path: str | Path | None = None):
        self.engine = engine
        if policy_path is None:
            try:
                from agent_utilities.core.config import config as _cfg

                policy_path = getattr(_cfg, "action_policy_path", "") or None
            except Exception:  # noqa: BLE001 — config import must never block a decision
                policy_path = None
        self.policy_path = resolve_policy_path(
            str(policy_path) if policy_path else None
        )
        self._file_cache: tuple[float, dict[str, Any]] | None = None
        # CONCEPT:AU-OS.safety.irreversibility-aversion — irreversibility aversion (opt-in; default off so the
        # shipped behavior is unchanged). When on, an irreversible action that the
        # tier would auto-execute is downgraded to a human approval.
        from agent_utilities.core.config import setting

        self._irreversibility_aversion = bool(
            setting("ACTION_IRREVERSIBILITY_AVERSION", False, cast=bool)
        )

    # ── policy loading ──────────────────────────────────────────────

    def _load_file_policy(self) -> dict[str, Any]:
        """Parse the policy YAML, mtime-cached; fall back to DEFAULT_POLICY."""
        path = self.policy_path
        if path is None:
            return DEFAULT_POLICY
        try:
            mtime = path.stat().st_mtime
            if self._file_cache and self._file_cache[0] == mtime:
                return self._file_cache[1]
            import yaml

            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if not isinstance(data, dict) or not isinstance(data.get("rules"), list):
                logger.warning(
                    "action_policy: %s has no rules list — using shipped default", path
                )
                data = DEFAULT_POLICY
            self._file_cache = (mtime, data)
            return data
        except Exception as e:  # noqa: BLE001 — a broken file must not open the gate
            logger.warning(
                "action_policy: failed loading %s (%s) — using default", path, e
            )
            return DEFAULT_POLICY

    def option(self, name: str, default: Any = None) -> Any:
        """Read one entry from the policy file's optional ``options:`` mapping.

        Deployment-tunable behavior knobs that are policy (not config) —
        e.g. ``watch_scale_down: true`` makes the autoscaler schedule an
        OS-5.27 health watch after scale-DOWN actions too (CONCEPT:AU-OS.scaling.reactive-replica-autoscaling).
        Missing file/section/key ⇒ ``default``.
        """
        options = self._load_file_policy().get("options")
        if not isinstance(options, dict):
            return default
        return options.get(name, default)

    def _kg_rules(self) -> list[ActionRule]:
        """KG-stored overrides: ``governance_rule`` nodes with scope='action_policy'.

        Stored either as flat properties (kind/target/tier/...) or as a
        ``rule_json`` payload. Inactive rules (``active=false``) are skipped.
        """
        if self.engine is None:
            return []
        try:
            rows = self.engine.query_cypher(
                "MATCH (r:governance_rule {scope: 'action_policy'}) RETURN r LIMIT 100"
            )
        except Exception as e:  # noqa: BLE001 — overrides are best-effort
            logger.debug("action_policy: KG rule load failed: %s", e)
            return []
        rules: list[ActionRule] = []
        for row in rows or []:
            props = row.get("r") if isinstance(row, dict) else None
            if not isinstance(props, dict):
                continue
            if str(props.get("active", "true")).lower() in ("false", "0", "no"):
                continue
            raw: dict[str, Any] = props
            if props.get("rule_json"):
                try:
                    raw = {**props, **json.loads(props["rule_json"])}
                except (TypeError, ValueError):
                    pass
            rule = _coerce_rule(raw, origin="kg")
            if rule is not None:
                rules.append(rule)
        rules.sort(key=lambda r: -r.priority)
        return rules

    def _match(self, request: ActionRequest) -> tuple[ActionRule, dict[str, Any]]:
        """First matching rule — KG overrides first, then file order — plus defaults."""
        data = self._load_file_policy()
        raw_defaults = data.get("defaults")
        defaults: dict[str, Any] = (
            raw_defaults if isinstance(raw_defaults, dict) else {}
        )
        for rule in self._kg_rules():
            if rule.matches(request):
                return rule, defaults
        for raw in data.get("rules") or []:
            file_rule = _coerce_rule(raw, origin="file")
            if file_rule is not None and file_rule.matches(request):
                return file_rule, defaults
        tier = str(defaults.get("tier") or TIER_APPROVAL)
        return ActionRule(
            tier=tier if tier in _TIERS else TIER_APPROVAL, origin="default"
        ), defaults

    # ── durable accounting (reads the ActionDecision ledger) ────────

    def _recent_decisions(self, kind: str) -> list[dict[str, Any]]:
        if self.engine is None:
            return []
        try:
            rows = self.engine.query_cypher(
                "MATCH (d:ActionDecision {kind: $kind}) "
                "RETURN d.target AS target, d.decision AS decision, "
                f"d.decided_unix AS ts LIMIT {_LEDGER_SCAN_LIMIT}",
                {"kind": kind},
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("action_policy: ledger scan failed: %s", e)
            return []
        return [r for r in rows or [] if isinstance(r, dict)]

    def _rate_exceeded(self, request: ActionRequest, limit: dict[str, Any]) -> bool:
        """True when (kind, target) already used its allowed budget in the window."""
        try:
            max_n = int(limit.get("max", 0))
            window_s = float(limit.get("window_s", 3600))
        except (TypeError, ValueError):
            return False
        if max_n <= 0:
            return False
        since = _now() - window_s
        n = sum(
            1
            for d in self._recent_decisions(request.kind)
            if d.get("target") == request.target
            and d.get("decision") in _ALLOWING
            and float(d.get("ts") or 0) >= since
        )
        return n >= max_n

    def _blast_exceeded(self, request: ActionRequest, cap: dict[str, Any]) -> bool:
        """True when allowing this target widens the kind's window beyond the cap."""
        try:
            max_targets = int(cap.get("max_targets", 0))
            window_s = float(cap.get("window_s", 3600))
        except (TypeError, ValueError):
            return False
        if max_targets <= 0:
            return False
        since = _now() - window_s
        targets = {
            d.get("target")
            for d in self._recent_decisions(request.kind)
            if d.get("decision") in _ALLOWING and float(d.get("ts") or 0) >= since
        }
        targets.add(request.target)
        return len(targets) > max_targets

    # ── the decision ────────────────────────────────────────────────

    def _graduates(self, request: ActionRequest) -> bool:
        """Whether the autonomy ramp lifts this ask→allow (CONCEPT:AU-OS.governance.autonomy-change-proposer).

        Requires (1) an actor, (2) the action kind on the policy's ``ramp_eligible``
        allowlist (empty by default → never graduates), and (3) an earned trust
        record. All best-effort; any failure leaves the tier unchanged."""
        if not request.actor_id:
            return False
        try:
            import fnmatch

            policy = self._load_file_policy()
            eligible = policy.get("ramp_eligible") or []
            if not any(fnmatch.fnmatch(request.kind, pat) for pat in eligible):
                return False
            from agent_utilities.orchestration.autonomy_ramp import clears_ramp

            backend = getattr(self.engine, "backend", None)
            return clears_ramp(backend, request.actor_id, request.kind)
        except Exception:  # noqa: BLE001 — ramp never widens scope on error
            return False

    def classify(self, request: ActionRequest) -> str:
        """Return the policy *tier* for ``request`` with NO side effects.

        Unlike :meth:`decide`, this writes no ``ActionDecision`` audit node and
        files no ``ActionApproval`` — it only resolves the matching rule (file +
        KG ``governance_rule`` overrides) and returns its tier
        (``auto`` | ``auto_notify`` | ``approval_required`` | ``forbidden``).

        For surfaces that need the governed *verdict* per call without the
        durable fleet-accounting side effects — e.g. the Claude Code PreToolUse
        gate (CONCEPT:AU-OS.deployment.dynamic-two-fail-closed), which would otherwise spam the KG with an audit
        node for every IDE tool call. Never raises — fails CLOSED to
        ``forbidden`` on any internal error.
        """
        try:
            rule, _defaults = self._match(request)
            tier = rule.tier
            # CONCEPT:AU-OS.governance.autonomy-change-proposer — earned-autonomy ramp: an allowlisted action kind the
            # actor has performed verifiably correctly enough times graduates one rung
            # (ask → auto_notify). Safe by default: the allowlist is empty unless an
            # operator opts a kind in, and forbidden is never graduated.
            if tier == TIER_APPROVAL and self._graduates(request):
                return TIER_AUTO_NOTIFY
            return tier
        except Exception as e:  # noqa: BLE001 — classification fails CLOSED
            logger.warning(
                "action_policy: classify error for %s: %s", request.summary(), e
            )
            return TIER_FORBIDDEN

    def decide(self, request: ActionRequest) -> ActionDecision:
        """Decide one action: allow / allow+notify / queue-approval / deny.

        Never raises — an internal error denies (fail closed) with the error
        recorded as the reason.
        """
        try:
            decision = self._decide_inner(request)
        except Exception as e:  # noqa: BLE001 — the gate fails CLOSED
            logger.warning(
                "action_policy: decision error for %s: %s", request.summary(), e
            )
            decision = ActionDecision(
                decision=DECISION_DENY,
                tier=TIER_FORBIDDEN,
                request=request,
                reason=f"policy error (fail closed): {e}",
            )
        decision.audit_id = self._audit(decision)
        if decision.decision == DECISION_ALLOW_NOTIFY:
            self._notify(
                f"[fleet-autonomy] executing {request.summary()} "
                f"(source={request.source}, tier={decision.tier})"
            )
        return decision

    def _decide_inner(self, request: ActionRequest) -> ActionDecision:
        rule, defaults = self._match(request)
        base = ActionDecision(
            decision=DECISION_DENY,
            tier=rule.tier,
            request=request,
            rule_origin=rule.origin,
        )

        if rule.tier == TIER_FORBIDDEN:
            base.reason = "forbidden by policy"
            return base

        if rule.tier == TIER_APPROVAL:
            base.decision = DECISION_QUEUE
            base.reason = "tier requires human approval"
            base.approval_id = self.queue_approval(request, reason=base.reason)
            return base

        # auto / auto_notify — run the safety pre-checks.
        rate = rule.rate_limit or defaults.get("rate_limit") or {}
        if isinstance(rate, dict) and self._rate_exceeded(request, rate):
            base.decision = DECISION_DENY
            base.reason = (
                f"rate limit exceeded ({rate.get('max')}/{rate.get('window_s')}s "
                "for this action+target)"
            )
            return base

        blast = rule.blast_radius or defaults.get("blast_radius") or {}
        if isinstance(blast, dict) and self._blast_exceeded(request, blast):
            base.decision = DECISION_QUEUE
            base.reason = (
                f"blast-radius cap ({blast.get('max_targets')} targets/"
                f"{blast.get('window_s')}s) — queued for approval"
            )
            base.approval_id = self.queue_approval(request, reason=base.reason)
            return base

        # CONCEPT:AU-OS.safety.irreversibility-aversion — irreversibility aversion (opt-in). An irreversible
        # action that would otherwise auto-execute is routed to a human, since
        # policy-gating alone is brittle as autonomy rises.
        if self._irreversibility_aversion:
            from agent_utilities.core.corrigibility import is_irreversible

            if is_irreversible(request.kind):
                base.decision = DECISION_QUEUE
                base.reason = "irreversible action — queued for approval (SAFE-1.5)"
                base.approval_id = self.queue_approval(request, reason=base.reason)
                return base

        if not in_maintenance_window(rule.maintenance_window):
            base.decision = DECISION_QUEUE
            base.reason = (
                f"outside maintenance window {rule.maintenance_window} — queued"
            )
            base.approval_id = self.queue_approval(request, reason=base.reason)
            return base

        base.decision = (
            DECISION_ALLOW_NOTIFY if rule.tier == TIER_AUTO_NOTIFY else DECISION_ALLOW
        )
        base.reason = f"tier {rule.tier}"
        return base

    # ── side effects: approval queue, audit ledger, notification ────

    def queue_approval(self, request: ActionRequest, reason: str = "") -> str | None:
        """File an ``ActionApproval`` node for the existing fleet approvals flow.

        Listed by ``GET /api/fleet/approvals``; resolved by
        ``POST /api/fleet/approvals/grant`` (job_id = this node id); executed
        by the fleet reconciler's approved-action drain (CONCEPT:AU-OS.config.desired-state-fleet-reconciler).
        Deliberately NOT a ``Task`` node: pending Tasks are claimed by the
        engine's task workers, which would execute the action unapproved.
        """
        if self.engine is None:
            return None
        # Dedup: a still-pending approval for the same kind+target is reused so
        # a recurring divergence doesn't flood the queue one entry per tick.
        try:
            rows = self.engine.query_cypher(
                "MATCH (a:ActionApproval {kind: $kind, target: $target, "
                "status: 'pending'}) RETURN a.id AS id LIMIT 1",
                {"kind": request.kind, "target": request.target},
            )
            if rows and rows[0].get("id"):
                return str(rows[0]["id"])
        except Exception as e:  # noqa: BLE001 — dedup is best-effort
            logger.debug("action_policy: approval dedup probe failed: %s", e)
        approval_id = f"action_approval:{uuid.uuid4().hex[:12]}"
        try:
            self.engine.add_node(
                approval_id,
                "ActionApproval",
                properties={
                    "kind": request.kind,
                    "target": request.target,
                    "params_json": json.dumps(request.params, default=str)[:2000],
                    "source": request.source,
                    "reason": reason or request.reason,
                    "status": "pending",
                    "created_at": _now_iso(),
                    "created_unix": _now(),
                },
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("action_policy: approval queue write failed: %s", e)
            return None
        self._notify(
            f"[fleet-autonomy] approval required: {request.summary()} "
            f"(id={approval_id}, source={request.source})"
        )
        return approval_id

    def _audit(self, decision: ActionDecision) -> str | None:
        """Write the ``ActionDecision`` audit node (also the rate/blast ledger)."""
        if self.engine is None:
            return None
        audit_id = f"action_decision:{uuid.uuid4().hex[:12]}"
        req = decision.request
        try:
            self.engine.add_node(
                audit_id,
                "ActionDecision",
                properties={
                    "kind": req.kind,
                    "target": req.target,
                    "params_json": json.dumps(req.params, default=str)[:2000],
                    "source": req.source,
                    "tier": decision.tier,
                    "decision": decision.decision,
                    "reason": decision.reason[:500],
                    "rule_origin": decision.rule_origin,
                    "approval_id": decision.approval_id or "",
                    "decided_at": _now_iso(),
                    "decided_unix": _now(),
                },
            )
        except Exception as e:  # noqa: BLE001 — audit is best-effort, decision stands
            logger.debug("action_policy: audit write failed: %s", e)
            return None
        return audit_id

    def _notify(self, message: str) -> None:
        """Route through the KG-2.42 notification seam (journaled by default)."""
        try:
            from agent_utilities.knowledge_graph.actions.dispatch import (
                send_notification,
            )
            from agent_utilities.knowledge_graph.actions.models import NotificationSpec

            send_notification(
                NotificationSpec(channel="fleet", recipient="operators"), message
            )
        except Exception as e:  # noqa: BLE001
            logger.info(
                "action_policy notification (fallback log): %s (%s)", message, e
            )


def get_action_policy(engine: Any = None) -> ActionPolicy:
    """Construct the policy gate for ``engine`` (cheap; file parse is cached)."""
    return ActionPolicy(engine=engine)
