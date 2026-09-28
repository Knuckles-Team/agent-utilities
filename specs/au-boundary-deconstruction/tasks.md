# Tasks and status

**Legend:** TODO = no accepted implementation proof; IN PROGRESS = linked change under review; IMPLEMENTED = code at merged head with source proof; VERIFIED = acceptance tests at that head; ACCEPTED = owner sign-off and release proof. A task is not completed by a draft branch.

| Task | IDs | State | Done when |
|---|---|---|---|
| Served shell and caller cut | EH-476–479, EH-488–492, EH-515 | TODO | graph-os parity plus AU deletion and script/import census |
| Connector and source cut | EH-480–486, EH-499–501 | TODO | SDK transport/pack certification plus all AU caller deletions |
| Graph, schema, retrieval, memory cut | EH-487, EH-493–510, EH-516 | TODO | generated EG method parity, trusted migration and AU adapter-only shape |
| Usage and governance cut | EH-511–512 | TODO | EG durable usage and repository-manager governance parity |
| Permanent owner guard | EH-514 | TODO | generated manifest, CI gate and exact-head check |

Each row expands into a PR checklist with changed paths, owner operation, positive and negative test IDs, scanner delta, review, merged commit and acceptance artifact. Do not move a row to VERIFIED solely because another row in its range passed.
