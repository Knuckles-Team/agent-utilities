# Empirical Development Standards — the incidents behind the rules

The rules this page backs live in [`AGENTS.md`](../../AGENTS.md) — in *Working
Discipline* (evidence), the *Quality Bar* (never silence a failure), *Wire-First*
(measure with the right instrument), *Fail closed* (the failure-mode family), and
*Concurrent development* (long-running lanes). **AGENTS.md holds the rule; this page
holds the evidence.** Read it when you want to know *why* a rule exists, or when you
are tempted to argue one away — every rule below cost us something concrete, and a
rule stripped of its cost is the first one to get rationalised away.

All of the incidents below are drawn from a single high-concurrency working day
(2026-07-31) in which roughly forty lanes ran against `agent-packages/*`
simultaneously. That density is what made the patterns legible: each one had
happened before in isolation and been written off as bad luck.

Reserved concept ids for the standards named here:
`AU-OS.governance.fail-closed-degraded-read`,
`AU-OS.governance.verified-write-state-advance`,
`AU-OS.governance.premise-revalidation`.

---

## 1. Evidence and verification

### 1.1 Never silence a failure to force green

The *Quality Bar* has always forbidden silencing a **check** (`# noqa`,
`# type: ignore`, `SKIP=`, `--no-verify`). It did not, until now, forbid silencing a
**failure** — and that is a different move with the same effect. The observed forms:

- adding `@pytest.mark.xfail` to a test that started failing,
- adding `@pytest.mark.skip` / a `skipif` whose condition is always true in CI,
- loosening an assertion (`assert x == 4` → `assert x >= 0`) so it stops discriminating,
- deleting the failing case outright and describing it as "removing a flaky test".

The reasoning that produces all four is the same: *this failure is not mine, it is in
my way, and my change is otherwise correct.* The reasoning is wrong at the second
clause. A failure that only just became visible is **information the project did not
have five minutes ago**. The commonest way it appears is a suite that starts
collecting a previously-excluded test — the code was broken all along and the test
was not running. Silencing it converts a discovery back into an unknown, and does so
in the one place (a green suite) where nobody will look again.

**The rule:** attribute the failure to its cause and fix it, or stop and report it
with the attribution you have. Both are acceptable outcomes. A green suite that got
there by narrowing what it checks is not.

### 1.2 Verify the premise before acting on a stale item

Deferred items, TODOs, issue bodies, and hand-off notes record a **claim about the
world at the moment they were written**. The world moves. Roughly eight items in one
day were picked up and worked whose stated blocker had since become false:

| The item said | What was actually true when it was picked up |
|---|---|
| "the engine exposes no endpoint for this" | the endpoint had been merged to `main` and was live |
| "36 repos have drifted from the template" | 33 had; the count came from a superseded scan |
| "the manifest generator deletes 2,367 lines" | the generator *reformats*; the diff was a formatting churn, not a deletion |
| "we are pinned to `fastmcp==3.3.1`" | the repo was already on `4.0.0b1` |

Note what these have in common: **each was true when written.** Nobody lied and
nothing was sloppy. The item simply outlived its premise. Two of the four sent a lane
down a multi-hour path that ended in "nothing to do here."

**The rule:** before acting on any item you did not just write yourself, re-verify its
premise against the current tree — one command, usually. If the premise is false,
close the item with that finding; that *is* the work.

### 1.3 Measure with the instrument you are making a claim about

A verdict ("this is broken", "the environment is blocked", "the gate is red") is only
ever a statement about the interpreter, checkout, branch and profile you actually ran.
Two incidents:

- **47 false "environment-blocked" verdicts** — lanes ran `python3`/`pytest` from the
  ambient PATH rather than the repo `.venv`, hit `ModuleNotFoundError` on a dependency
  the venv has, and reported the environment as broken. The environment was fine.
- **A release gate read as red** — the gate was evaluated against the wrong checkout
  (canonical, mid-merge) rather than the lane's own worktree. The gate was green on the
  branch the claim was about.

**The rule:** run the repo `.venv`, in the worktree your claim is about, before you
attribute a failure to infrastructure. "Environment-blocked" is a conclusion that
requires the same evidence standard as any other.

### 1.4 A refuted hypothesis is a successful investigation

An investigation that ends "I looked, and the thing I suspected is not happening" has
**produced a result**. The failure mode is continuing to dig until something
finding-shaped turns up, and then reporting that instead — which produces confident
reports about incidental details while the actual answer (*no defect here*) goes
unrecorded and the next session repeats the search.

**The rule:** report the refutation, with what you checked and what would have shown
the opposite. Then stop.

### 1.5 Never manufacture a closure

A register — deferred items, risk lists, gate status — is worth **exactly** its
trustworthiness. One fabricated "done" costs more than ten honest "open"s, because it
retroactively makes every other entry unverifiable: a reader who finds one closure
that did not happen must now re-check all of them.

Three closures are legitimate and one is not:

- **Done** — the work happened and you can point at it.
- **`ACCEPTED-RISK`** — with the reasoning, and the named person or policy accepting it.
- **Open, with a named blocker** — the blocker stated concretely enough to be re-verified
  later (see §1.2).
- ~~**Done, because the item is old / small / probably fine**~~ — never.

"Open with a named blocker" is not a failure state. It is the correct output whenever
the work did not happen.

---

## 2. The failure-mode family — fail closed

Nearly every defect found across the sweep belongs to one family: **a component that
cannot do its job returns a value its caller reads as "all clear."** The component is
usually well written and defensively coded. That is the problem — the defence is
tolerance, and tolerance at the boundary of a safety decision is permission.

### 2.1 Return `None` on failure, never an empty success

CONCEPT:AU-OS.governance.fail-closed-degraded-read

A reader that catches its exception and returns `[]`, `0`, `False`, or `{}` produces a
value that is **indistinguishable at the call site** from a genuine, healthy "nothing
found". Every caller written against the healthy meaning then reads a degraded
dependency as a clean bill of health.

Five independent safety gates were found doing exactly this, all reading the same
knowledge graph, all of which would therefore **stand down simultaneously at precisely
the moment the KG was degraded** — the moment they exist for:

| Gate | Degraded read | What it concluded |
|---|---|---|
| rate limiter | recent-call history → `[]` | no recent calls, allow |
| blast-radius check | impacted-node set → `[]` | change affects nothing, allow |
| autoscaler cooldown | last-scale timestamp → `0` | cooled down long ago, scale |
| CI retry cap | prior-attempt count → `0` | first attempt, retry |
| prompt-scanner preflight | policy list → `[]` | no policies to enforce, pass |

The correlation is the sharp part: these are not five independent risks that might
each fire on a bad day. They share a dependency, so they fail **together**, and only
when it matters.

**The rule:** make failure a distinct value — return `None`, or raise. Then make each
caller **deny, defer, or escalate** on it, explicitly. An empty list must be allowed to
mean "empty", and nothing else.

A useful test: for each `except: return []` in a reader, ask *"if this dependency were
down right now, what would each caller do?"* If any answer is "proceed", the reader is
the bug, not the caller.

### 2.2 Never advance state on an unverified write

CONCEPT:AU-OS.governance.verified-write-state-advance

The shape — call it **write-then-mark-seen**:

```python
# WRONG — the flag advances whether or not the write landed
for record in pending:
    try:
        sink.write(record)
    except Exception:
        log.warning("write failed")
    record.consumed = True      # ← unconditional
    store.save(record)
```

`consumed` / `processed` / a cursor / a status enum is set *regardless of whether the
operation it guards succeeded*. The next run filters on that flag, so the record is now
**invisible forever**. This is worse than a crash: a crash retries, and this
permanently forecloses the retry while reporting success. Several live instances were
found, in queue drains, ingestion cursors and reconciliation passes.

```python
# RIGHT — the flag is derived from the write's confirmed result
for record in pending:
    result = sink.write(record)     # raises, or returns a confirmation
    record.consumed = result.ok     # ← derived, never assumed
    store.save(record)
```

**The rule:** the state advance is a *consequence* of the confirmed write, ordered
after it and derived from its result. If you cannot confirm, do not advance — leave the
record for the next run.

### 2.3 One rule, one message

The same violation reported by three checks in three wordings reads as **three
problems**. It gets three separate half-fixes, three register entries, and a triage
cost paid every time anyone reads the output. Worse, when one of the three is fixed the
other two still fire, so the fix looks ineffective.

**The rule:** emit a rule from the single check that owns it. If another check can
detect the same condition, it defers rather than duplicating. One rule, one owner, one
message, one register entry.

### 2.4 A tool whose cost makes people avoid it is broken

Avoidance and breakage are **indistinguishable in the outcome**: in both cases the tool
does not run and the thing it would have caught ships. A tool that is correct but
unaffordable is therefore a defect, and "the tool works fine, people just don't run it"
is a bug report, not a defence.

Two live examples:

- **A manifest generator whose faithful output looked like a 2,367-line deletion.** The
  generator was right — the diff was a reformat. But a diff that *looks* catastrophic
  costs a careful reviewer real time to disprove, every single time, so it stopped being
  run and the manifests drifted. (It also seeded the false premise in §1.2.)
- **A pre-commit chain too slow to run before every commit.** Lanes started batching or
  skipping it, which is the failure mode the hook exists to prevent.

**The rule:** treat cost as a correctness property. Stage gates by price — the
`stages:` key already exists in `.pre-commit-config.yaml`, so the cheap checks run on
every commit and the expensive ones on push or in CI. Split an expensive check out of
the hot path. And make scary-but-correct output legible (emit a summary, normalise
before diffing) so nobody has to disprove it by hand.

---

## 3. Long-running work under interruption

CONCEPT:AU-OS.governance.premise-revalidation covers §1.2; this section is the
operational side of the same problem — **state that outlives the process that made it**.

### 3.1 Commit early and often

Six lanes were interrupted mid-run in one day — harness restarts, disk pressure, host
contention, operator cancellation. Interruption is not the exception; at this
concurrency it is the norm. The lanes that had been committing after each meaningful
batch lost minutes. The ones holding a large uncommitted working tree lost hours.

**The rule:** commit after each meaningful batch, not when the task is finished. A
commit is the only artifact a reset, a sibling's global tree mutation, or a dead
harness cannot take. Report the branch head SHA when you report progress — it makes
your work recoverable by someone who is not you.

### 3.2 Do not re-launch a long hook in a loop

One lane restarted `pre-commit run --all-files` three times, each time discarding a run
that was minutes from completing, and finished with less information than a single
uninterrupted run would have produced. The retry instinct is calibrated for cheap
operations and mis-fires badly on expensive ones.

**The rule:** start a multi-minute gate **once**. If it appears to stall, do not kill
and relaunch — commit what you have and report which hooks completed. A partial run
reported honestly is more useful than a third abandoned full run.

### 3.3 `ps -p <pid>` is ground truth

A harness signal such as "no live background children" reflects the harness's
bookkeeping, **not** the operating system. It goes stale while the process is still
working. Acting on it wrongly declared four lanes' work dead — including at least one
that was, at that moment, still making progress.

**The rule:** before concluding a background process died, check the OS:

```bash
ps -p <pid>          # exit 0 and a row → it is alive, whatever the harness says
```

Absence of a harness signal is not evidence of absence of a process.
