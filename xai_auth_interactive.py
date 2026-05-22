from agent_utilities.security.xai_auth import XaiAuthManager

print("Starting auth flow...", flush=True)
manager = XaiAuthManager()
try:
    tokens = manager.login()
    print("Success!", flush=True)
except Exception as e:
    print(f"Error: {e}", flush=True)
