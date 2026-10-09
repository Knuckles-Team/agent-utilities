# Tasks

**States:** TODO, IN PROGRESS, IMPLEMENTED, VERIFIED, ACCEPTED. Current state: TODO pending exact merged-head audit.

| Task | IDs | State | Exit |
|---|---|---|---|
| Public ecosystem bootstrap skill | AU-DEV-R001 | TODO | fresh-clone and composed fixture proof |
| Move AU genesis/deployment skill authority | AU-DEV-R002 | TODO | one provider and no stale AU copy |
| Deterministic skill inventory and refresh | AU-DEV-R003 | TODO | stale/moved/renamed tests and idempotent install |
| Contributor guide and CI | AU-DEV-R001, AU-DEV-R003 | TODO | public links and fork CI without private state |

**Split recorded (rapid-delivery, 2026-10-09):** both rows name more than two code roots (skill content audit + AGENTS.md trims for AU-DEV-R001; Codex and XDG refresh paths for AU-DEV-R003), so each ships a `.1` slice first.
- `AU-DEV-R001.1`: typed `DevelopmentSkillTopicInventory` model (`agent_utilities/skills/dev_skill_topics.py`) that refuses a skill missing any required topic, plus its refusal test (`tests/unit/skills/test_dev_skill_topics.py`). Auditing every real per-repo skill and trimming AGENTS.md files remains `AU-DEV-R001.2+`.
- `AU-DEV-R003.1`: typed `SkillInventoryRefreshEntry` model (`agent_utilities/skills/inventory_refresh_model.py`) that refuses a refresh which mutates before printing its plan, or claims completion with a mismatched hash, plus its refusal test (`tests/unit/skills/test_inventory_refresh_model.py`). Wiring this into the real move/rename refresh command for both the Codex and runtime XDG paths remains `AU-DEV-R003.2+`.
