import re
import os

with open(
    "/home/apps/workspace/agent-packages/agent-utilities/docs/overview.md", "r"
) as f:
    content = f.read()

# Find the table rows
rows = re.findall(r"\|(.*?)\|(.*?)\|(.*?)\|(.*?)\|", content)

missing_files = []
existing_files = []

for row in rows:
    if "Concept ID" in row[0] or "---" in row[0]:
        continue

    concept_id = row[0].strip().replace("**", "")
    concept_name = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", row[1]).strip().replace("**", "")
    paths = row[3].strip().replace("`", "")

    # Can be multiple paths separated by comma
    path_list = [p.strip() for p in paths.split(",") if p.strip()]

    for path in path_list:
        full_path = os.path.join(
            "/home/apps/workspace/agent-packages/agent-utilities", path
        )
        if not os.path.exists(full_path):
            missing_files.append((concept_id, concept_name, path))
        else:
            existing_files.append((concept_id, concept_name, path))

print(f"Missing Files: {len(missing_files)}")
for c_id, c_name, path in missing_files:
    print(f"{c_id} - {c_name}: {path}")
