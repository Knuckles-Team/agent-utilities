import asyncio
import json
from typing import Any

import pytest

from agent_utilities.core.sessions import (
    cancel_goal,
    cancel_session_run,
    create_goal,
    delete_session,
    get_all_sessions,
    get_goal_iterations,
    get_session_details,
    list_goals,
    submit_session_reply,
)


@pytest.fixture
def client(tiny_engine):
    # ``tiny_engine`` is depended on (its value unused) purely so this real-
    # engine need is visible in ``item.fixturenames`` --
    # ``tests/conftest.py``'s ``pytest_collection_modifyitems`` decides once
    # per xdist worker whether to pay for a real session engine by scanning
    # for exactly that fixture name (or ``engine_graph``). ``create_goal``
    # below reaches the engine through ``sessions.py``'s enqueue path without
    # going through either fixture directly, so without this explicit
    # dependency a worker grouped only with other non-engine tests never
    # started one, and ``POST /goals`` returned a plain 503 (the ConnectionError
    # is caught and turned into an HTTP response server-side, so it never
    # reaches pytest's engine-unreachable-to-skip conversion) instead of a
    # graceful skip or the real 200 the full suite's engine-requesting
    # grouping happens to give it.
    return _Client()


class _Request:
    def __init__(self, path_params: dict[str, Any], body: Any = None) -> None:
        self.path_params = path_params
        self._body = body

    async def json(self) -> Any:
        if self._body is None:
            raise ValueError("no body")
        return self._body


class _Result:
    def __init__(self, response: Any) -> None:
        self.status_code = response.status_code
        self._body = response.body

    def json(self) -> Any:
        return json.loads(self._body)


class _Client:
    """Drive the transport-free operations the way graph-os mounts them."""

    _ROUTES = {
        ("GET", "/sessions"): get_all_sessions,
        ("GET", "/sessions/{session_id}"): get_session_details,
        ("DELETE", "/sessions/{session_id}"): delete_session,
        ("POST", "/sessions/{session_id}/reply"): submit_session_reply,
        ("POST", "/sessions/{session_id}/cancel"): cancel_session_run,
        ("POST", "/goals"): create_goal,
        ("GET", "/goals"): list_goals,
        ("GET", "/goals/{goal_id}/iterations"): get_goal_iterations,
        ("POST", "/goals/{goal_id}/cancel"): cancel_goal,
    }

    def _call(self, method: str, path: str, body: Any = None) -> _Result:
        parts = path.strip("/").split("/")
        for (route_method, pattern), handler in self._ROUTES.items():
            pattern_parts = pattern.strip("/").split("/")
            if route_method != method or len(pattern_parts) != len(parts):
                continue
            params: dict[str, Any] = {}
            for want, got in zip(pattern_parts, parts, strict=True):
                if want.startswith("{"):
                    params[want.strip("{}")] = got
                elif want != got:
                    break
            else:
                return _Result(asyncio.run(handler(_Request(params, body))))
        raise AssertionError(f"no operation for {method} {path}")

    def get(self, path: str) -> _Result:
        return self._call("GET", path)

    def post(self, path: str, json: Any = None) -> _Result:
        return self._call("POST", path, json)

    def delete(self, path: str) -> _Result:
        return self._call("DELETE", path)


def test_sessions_and_goals_flow(client):
    # 1. Retrieve sessions list
    resp = client.get("/sessions")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)

    # 2. Launch a mock autonomous goal
    resp = client.post(
        "/goals", json={"objective": "Test autonomous pipeline execution"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "goal_id" in data
    assert "session_id" in data

    goal_id = data["goal_id"]
    session_id = data["session_id"]

    # 3. Verify goal iteration logs
    resp = client.get(f"/goals/{goal_id}/iterations")
    assert resp.status_code == 200
    iter_data = resp.json()
    assert iter_data["goal_id"] == goal_id

    # 4. Retrieve active goals list
    resp = client.get("/goals")
    assert resp.status_code == 200
    goals_list = resp.json()
    assert any(g["goal_id"] == goal_id for g in goals_list)

    # 5. Verify session retrieval
    resp = client.get(f"/sessions/{session_id}")
    assert resp.status_code == 200
    sess_data = resp.json()
    assert sess_data["id"] == session_id

    # 6. Submit reply to session
    resp = client.post(
        f"/sessions/{session_id}/reply", json={"content": "Continue execution"}
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"

    # 7. Cancel active goal loop
    resp = client.post(f"/goals/{goal_id}/cancel")
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"

    # 8. Delete session
    resp = client.delete(f"/sessions/{session_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"
