import pytest
from starlette.testclient import TestClient
from starlette.applications import Starlette
from agent_utilities.core.sessions import (
    get_all_sessions,
    get_session_details,
    delete_session,
    submit_session_reply,
    cancel_session_run,
    create_goal,
    list_goals,
    get_goal_iterations,
    cancel_goal,
)

@pytest.fixture
def client():
    app = Starlette()
    app.add_route("/sessions", get_all_sessions, methods=["GET"])
    app.add_route("/sessions/{session_id}", get_session_details, methods=["GET"])
    app.add_route("/sessions/{session_id}", delete_session, methods=["DELETE"])
    app.add_route("/sessions/{session_id}/reply", submit_session_reply, methods=["POST"])
    app.add_route("/sessions/{session_id}/cancel", cancel_session_run, methods=["POST"])
    app.add_route("/goals", create_goal, methods=["POST"])
    app.add_route("/goals", list_goals, methods=["GET"])
    app.add_route("/goals/{goal_id}/iterations", get_goal_iterations, methods=["GET"])
    app.add_route("/goals/{goal_id}/cancel", cancel_goal, methods=["POST"])
    return TestClient(app)

def test_sessions_and_goals_flow(client):
    # 1. Retrieve sessions list
    resp = client.get("/sessions")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)

    # 2. Launch a mock autonomous goal
    resp = client.post("/goals", json={"objective": "Test autonomous pipeline execution"})
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
    resp = client.post(f"/sessions/{session_id}/reply", json={"content": "Continue execution"})
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
