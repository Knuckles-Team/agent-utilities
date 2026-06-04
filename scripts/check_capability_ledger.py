import yaml
import sys
import os

def check_ledger():
    ledger_path = "gap_plans/_feature_ledger.yaml"
    if not os.path.exists(ledger_path):
        print(f"Error: {ledger_path} does not exist.")
        sys.exit(1)

    with open(ledger_path, "r", encoding="utf-8") as f:
        ledger = yaml.safe_load(f)

    if not ledger:
        print("Error: Ledger is empty.")
        sys.exit(1)

    errors = []
    live_or_migrated_count = 0

    for idx, entry in enumerate(ledger):
        status = entry.get("status")
        target = entry.get("target")
        test = entry.get("characterization_test")

        if status not in {"live", "migrated"}:
            errors.append(f"Entry {entry.get('id', idx)} has unresolved status: {status}")
        else:
            live_or_migrated_count += 1

        if target == "TODO":
            errors.append(f"Entry {entry.get('id', idx)} is missing a target symbol.")
            
        if test == "TODO":
            errors.append(f"Entry {entry.get('id', idx)} is missing a characterization test.")

    print(f"Capabilities Live/Migrated: {live_or_migrated_count} / {len(ledger)}")

    if errors:
        print("No-Capability-Lost Gate FAILED:")
        for e in errors[:20]:
            print(f"  - {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors.")
        sys.exit(1)
        
    print("No-Capability-Lost Gate PASSED.")

if __name__ == "__main__":
    check_ledger()
