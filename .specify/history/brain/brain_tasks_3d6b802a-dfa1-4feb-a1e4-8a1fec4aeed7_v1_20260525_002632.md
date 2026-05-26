# Tasks: Universal Browser-based OAuth PKCE Standard

- [x] Create generic `BaseBrowserAuthManager` in `agent_utilities/security/browser_auth.py` [agent-utilities]
- [x] Refactor `agent_utilities/security/xai_auth.py` to inherit from `BaseBrowserAuthManager` [agent-utilities]
- [x] Refactor `leanix-agent` to inherit from `BaseBrowserAuthManager` [leanix-agent]
- [x] Update and verify unit tests for `agent-utilities` (`test_xai_auth.py` and new `test_browser_auth.py`)
- [x] Update and verify unit tests for `leanix-agent` (`test_browser_auth.py`)
