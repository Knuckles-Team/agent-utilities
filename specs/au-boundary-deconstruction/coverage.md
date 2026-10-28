# Current package-directory ownership

The first two directory levels of the current Meristem package are retained
by Meristem. A directory move changes this inventory and its owner decision
in the same change. The owner guard independently checks Python modules.

| Directory | Disposition | Destination |
|---|---|---|
| `agent_utilities/core` | keep | meristem |
| `agent_utilities/graph` | keep | meristem |
| `agent_utilities/harness` | keep | meristem |
| `agent_utilities/rlm` | keep | meristem |
| `agent_utilities/runtime` | keep | meristem |
| `agent_utilities/skills` | keep | meristem |
| `agent_utilities/skills/agent-utilities-development` | keep | meristem |
