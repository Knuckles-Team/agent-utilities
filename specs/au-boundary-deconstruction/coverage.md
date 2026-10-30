# Current package-directory ownership

The first two directory levels of the current Meristem package are retained
by Meristem. A directory move changes this inventory and its owner decision
in the same change. The owner guard independently checks Python modules.

| Directory | Disposition | Destination |
|---|---|---|
| `meristem/core` | keep | meristem |
| `meristem/graph` | keep | meristem |
| `meristem/harness` | keep | meristem |
| `meristem/rlm` | keep | meristem |
| `meristem/runtime` | keep | meristem |
| `meristem/skills` | keep | meristem |
| `meristem/skills/meristem-development` | keep | meristem |
