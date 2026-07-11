"""End-to-end tests: a real WebSocketServer on a real socket, a real
Flask REST API (via its test client) sharing the same SQLite file and
Broadcaster, and a hand-rolled real socket client (ws_test_client.py).
This is the test that actually proves the REST write path pushes a live
update out over a real WebSocket connection -- test_routes.py's
broadcast test only proves the broadcaster *method* gets called."""
import os
import socket
import tempfile
import threading

import pytest

from app import create_app, db as db_module
from app.ws.server import WebSocketServer
from .ws_test_client import WSClient

JWT_SECRET = "integration-test-secret"


@pytest.fixture
def env():
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(db_path)  # let sqlite create it fresh

    flask_app = create_app(db_path=db_path, jwt_secret=JWT_SECRET)
    flask_app.testing = True
    rest_client = flask_app.test_client()
    broadcaster = flask_app.config["BROADCASTER"]

    ws_server = WebSocketServer(
        host="127.0.0.1", port=0,
        conn_factory=lambda: db_module.connect(db_path),
        jwt_secret=JWT_SECRET,
        broadcaster=broadcaster,
    )
    thread = threading.Thread(target=ws_server.serve_forever, daemon=True)
    thread.start()
    ws_server.wait_ready()

    yield rest_client, ws_server

    ws_server.stop()
    thread.join(timeout=2)
    if os.path.exists(db_path):
        os.remove(db_path)
    wal = db_path + "-wal"
    shm = db_path + "-shm"
    for extra in (wal, shm):
        if os.path.exists(extra):
            os.remove(extra)


def auth_headers(token):
    return {"Authorization": f"Bearer {token}"}


def signup(rest_client, email="owner@example.com", workspace="Acme"):
    resp = rest_client.post("/api/auth/signup", json={
        "email": email, "password": "password123", "workspace_name": workspace,
    })
    assert resp.status_code == 201
    return resp.get_json()


def test_subscribe_before_auth_is_rejected(env):
    rest_client, ws_server = env
    ws = WSClient("127.0.0.1", ws_server.port)
    try:
        ws.send_json({"type": "subscribe", "project_id": 1})
        reply = ws.recv_json()
        assert reply["type"] == "error"
        assert "not authenticated" in reply["message"]
    finally:
        ws.close()


def test_auth_with_bad_token_is_rejected(env):
    rest_client, ws_server = env
    ws = WSClient("127.0.0.1", ws_server.port)
    try:
        ws.send_json({"type": "auth", "token": "not-a-real-token"})
        reply = ws.recv_json()
        assert reply["type"] == "error"
    finally:
        ws.close()


def test_auth_then_subscribe_success(env):
    rest_client, ws_server = env
    account = signup(rest_client)
    project = rest_client.post("/api/projects", headers=auth_headers(account["token"]),
                                json={"name": "P"}).get_json()

    ws = WSClient("127.0.0.1", ws_server.port)
    try:
        ws.send_json({"type": "auth", "token": account["token"]})
        auth_reply = ws.recv_json()
        assert auth_reply == {"type": "auth_ok", "workspace_id": account["workspace_id"], "role": "owner"}

        ws.send_json({"type": "subscribe", "project_id": project["id"]})
        sub_reply = ws.recv_json()
        assert sub_reply == {"type": "subscribed", "project_id": project["id"]}
    finally:
        ws.close()


def test_cannot_subscribe_to_another_workspaces_project(env):
    rest_client, ws_server = env
    account_a = signup(rest_client, email="a@example.com", workspace="A Co")
    account_b = signup(rest_client, email="b@example.com", workspace="B Co")
    project_a = rest_client.post("/api/projects", headers=auth_headers(account_a["token"]),
                                  json={"name": "Secret"}).get_json()

    ws = WSClient("127.0.0.1", ws_server.port)
    try:
        ws.send_json({"type": "auth", "token": account_b["token"]})
        ws.recv_json()  # auth_ok
        ws.send_json({"type": "subscribe", "project_id": project_a["id"]})
        reply = ws.recv_json()
        assert reply["type"] == "error"
        assert "not found" in reply["message"]
    finally:
        ws.close()


def test_task_created_broadcasts_to_live_subscriber(env):
    rest_client, ws_server = env
    account = signup(rest_client)
    project = rest_client.post("/api/projects", headers=auth_headers(account["token"]),
                                json={"name": "P"}).get_json()

    ws = WSClient("127.0.0.1", ws_server.port)
    try:
        ws.send_json({"type": "auth", "token": account["token"]})
        ws.recv_json()
        ws.send_json({"type": "subscribe", "project_id": project["id"]})
        ws.recv_json()

        resp = rest_client.post(f"/api/projects/{project['id']}/tasks",
                                 headers=auth_headers(account["token"]),
                                 json={"title": "Ship the WS server"})
        created = resp.get_json()

        ws.settimeout(3)
        event = ws.recv_json()
        assert event["type"] == "task_created"
        assert event["project_id"] == project["id"]
        assert event["task"]["title"] == "Ship the WS server"
        assert event["task"]["id"] == created["id"]
    finally:
        ws.close()


def test_task_updated_and_deleted_broadcast(env):
    rest_client, ws_server = env
    account = signup(rest_client)
    project = rest_client.post("/api/projects", headers=auth_headers(account["token"]),
                                json={"name": "P"}).get_json()
    task = rest_client.post(f"/api/projects/{project['id']}/tasks",
                             headers=auth_headers(account["token"]),
                             json={"title": "X"}).get_json()

    ws = WSClient("127.0.0.1", ws_server.port)
    try:
        ws.send_json({"type": "auth", "token": account["token"]})
        ws.recv_json()
        ws.send_json({"type": "subscribe", "project_id": project["id"]})
        ws.recv_json()
        ws.settimeout(3)

        rest_client.patch(f"/api/tasks/{task['id']}", headers=auth_headers(account["token"]),
                           json={"status": "done"})
        update_event = ws.recv_json()
        assert update_event["type"] == "task_updated"
        assert update_event["task"]["status"] == "done"

        rest_client.delete(f"/api/tasks/{task['id']}", headers=auth_headers(account["token"]))
        delete_event = ws.recv_json()
        assert delete_event == {"type": "task_deleted", "project_id": project["id"], "task_id": task["id"]}
    finally:
        ws.close()


def test_multiple_subscribers_all_receive_broadcast(env):
    rest_client, ws_server = env
    account = signup(rest_client)
    project = rest_client.post("/api/projects", headers=auth_headers(account["token"]),
                                json={"name": "P"}).get_json()

    clients = [WSClient("127.0.0.1", ws_server.port) for _ in range(3)]
    try:
        for ws in clients:
            ws.send_json({"type": "auth", "token": account["token"]})
            ws.recv_json()
            ws.send_json({"type": "subscribe", "project_id": project["id"]})
            ws.recv_json()
            ws.settimeout(3)

        rest_client.post(f"/api/projects/{project['id']}/tasks",
                          headers=auth_headers(account["token"]), json={"title": "Broadcast me"})

        for ws in clients:
            event = ws.recv_json()
            assert event["type"] == "task_created"
            assert event["task"]["title"] == "Broadcast me"
    finally:
        for ws in clients:
            ws.close()


def test_unsubscribe_stops_further_broadcasts(env):
    rest_client, ws_server = env
    account = signup(rest_client)
    project = rest_client.post("/api/projects", headers=auth_headers(account["token"]),
                                json={"name": "P"}).get_json()

    ws = WSClient("127.0.0.1", ws_server.port)
    try:
        ws.send_json({"type": "auth", "token": account["token"]})
        ws.recv_json()
        ws.send_json({"type": "subscribe", "project_id": project["id"]})
        ws.recv_json()

        ws.send_json({"type": "unsubscribe", "project_id": project["id"]})
        unsub_reply = ws.recv_json()
        assert unsub_reply == {"type": "unsubscribed", "project_id": project["id"]}

        rest_client.post(f"/api/projects/{project['id']}/tasks",
                          headers=auth_headers(account["token"]), json={"title": "Should not arrive"})

        ws.settimeout(1)
        with pytest.raises(socket.timeout):
            ws.recv_json()
    finally:
        ws.close()


def test_disconnect_cleans_up_subscription(env):
    """A dead socket shouldn't stop later broadcasts to the survivors --
    the broadcaster must drop it rather than erroring the whole broadcast."""
    rest_client, ws_server = env
    account = signup(rest_client)
    project = rest_client.post("/api/projects", headers=auth_headers(account["token"]),
                                json={"name": "P"}).get_json()

    doomed = WSClient("127.0.0.1", ws_server.port)
    doomed.send_json({"type": "auth", "token": account["token"]})
    doomed.recv_json()
    doomed.send_json({"type": "subscribe", "project_id": project["id"]})
    doomed.recv_json()
    doomed.sock.close()  # hard close, no close frame

    survivor = WSClient("127.0.0.1", ws_server.port)
    try:
        survivor.send_json({"type": "auth", "token": account["token"]})
        survivor.recv_json()
        survivor.send_json({"type": "subscribe", "project_id": project["id"]})
        survivor.recv_json()
        survivor.settimeout(3)

        rest_client.post(f"/api/projects/{project['id']}/tasks",
                          headers=auth_headers(account["token"]), json={"title": "Still works"})
        event = survivor.recv_json()
        assert event["task"]["title"] == "Still works"
    finally:
        survivor.close()


def test_subscribe_rejects_boolean_project_id(env):
    """Same bool-is-an-int trap as the REST position field, on the WS
    side's project_id validation."""
    rest_client, ws_server = env
    account = signup(rest_client)

    ws = WSClient("127.0.0.1", ws_server.port)
    try:
        ws.send_json({"type": "auth", "token": account["token"]})
        ws.recv_json()
        ws.send_json({"type": "subscribe", "project_id": True})
        reply = ws.recv_json()
        assert reply == {"type": "error", "message": "project_id must be an int"}
    finally:
        ws.close()


def test_comment_created_and_deleted_broadcast_to_project_subscribers(env):
    rest_client, ws_server = env
    account = signup(rest_client)
    project = rest_client.post("/api/projects", headers=auth_headers(account["token"]),
                                json={"name": "P"}).get_json()
    task = rest_client.post(f"/api/projects/{project['id']}/tasks",
                             headers=auth_headers(account["token"]), json={"title": "X"}).get_json()

    ws = WSClient("127.0.0.1", ws_server.port)
    try:
        ws.send_json({"type": "auth", "token": account["token"]})
        ws.recv_json()
        ws.send_json({"type": "subscribe", "project_id": project["id"]})
        ws.recv_json()
        ws.settimeout(3)

        resp = rest_client.post(f"/api/tasks/{task['id']}/comments",
                                 headers=auth_headers(account["token"]), json={"body": "hello"})
        comment_id = resp.get_json()["id"]

        created_event = ws.recv_json()
        assert created_event["type"] == "comment_created"
        assert created_event["task_id"] == task["id"]
        assert created_event["comment"]["body"] == "hello"

        rest_client.delete(f"/api/comments/{comment_id}", headers=auth_headers(account["token"]))
        deleted_event = ws.recv_json()
        assert deleted_event == {
            "type": "comment_deleted", "project_id": project["id"],
            "task_id": task["id"], "comment_id": comment_id,
        }
    finally:
        ws.close()


def test_mention_notification_pushed_live_to_subscribed_user(env):
    rest_client, ws_server = env
    owner = signup(rest_client, email="owner@example.com", workspace="Acme")
    rest_client.post("/api/auth/signup", json={
        "email": "member@example.com", "password": "password123", "workspace_name": "Member Co",
    })
    rest_client.post("/api/workspace/invite", headers=auth_headers(owner["token"]),
                      json={"email": "member@example.com", "role": "member"})
    member = rest_client.post("/api/auth/login", json={
        "email": "member@example.com", "password": "password123", "workspace_slug": "acme",
    }).get_json()

    project = rest_client.post("/api/projects", headers=auth_headers(owner["token"]),
                                json={"name": "P"}).get_json()
    task = rest_client.post(f"/api/projects/{project['id']}/tasks",
                             headers=auth_headers(owner["token"]), json={"title": "X"}).get_json()

    member_ws = WSClient("127.0.0.1", ws_server.port)
    try:
        member_ws.send_json({"type": "auth", "token": member["token"]})
        member_ws.recv_json()
        member_ws.send_json({"type": "subscribe_notifications"})
        sub_reply = member_ws.recv_json()
        assert sub_reply == {"type": "subscribed_notifications"}
        member_ws.settimeout(3)

        rest_client.post(f"/api/tasks/{task['id']}/comments", headers=auth_headers(owner["token"]),
                          json={"body": "cc @member@example.com"})

        event = member_ws.recv_json()
        assert event["type"] == "notification"
        assert event["notification"]["type"] == "mention"
        assert "owner@example.com" in event["notification"]["message"]
    finally:
        member_ws.close()


def test_notification_channel_is_per_user_not_broadcast_to_everyone(env):
    """A notification for user B must not leak to user A's notification
    subscription, even though both are connected at the same time."""
    rest_client, ws_server = env
    owner = signup(rest_client, email="owner@example.com", workspace="Acme")
    rest_client.post("/api/auth/signup", json={
        "email": "bystander@example.com", "password": "password123", "workspace_name": "Bystander Co",
    })
    rest_client.post("/api/workspace/invite", headers=auth_headers(owner["token"]),
                      json={"email": "bystander@example.com", "role": "member"})
    bystander = rest_client.post("/api/auth/login", json={
        "email": "bystander@example.com", "password": "password123", "workspace_slug": "acme",
    }).get_json()
    rest_client.post("/api/auth/signup", json={
        "email": "member@example.com", "password": "password123", "workspace_name": "Member Co",
    })
    rest_client.post("/api/workspace/invite", headers=auth_headers(owner["token"]),
                      json={"email": "member@example.com", "role": "member"})

    project = rest_client.post("/api/projects", headers=auth_headers(owner["token"]),
                                json={"name": "P"}).get_json()
    task = rest_client.post(f"/api/projects/{project['id']}/tasks",
                             headers=auth_headers(owner["token"]), json={"title": "X"}).get_json()

    bystander_ws = WSClient("127.0.0.1", ws_server.port)
    try:
        bystander_ws.send_json({"type": "auth", "token": bystander["token"]})
        bystander_ws.recv_json()
        bystander_ws.send_json({"type": "subscribe_notifications"})
        bystander_ws.recv_json()

        rest_client.post(f"/api/tasks/{task['id']}/comments", headers=auth_headers(owner["token"]),
                          json={"body": "cc @member@example.com"})  # mentions member, NOT bystander

        bystander_ws.settimeout(1)
        with pytest.raises(socket.timeout):
            bystander_ws.recv_json()
    finally:
        bystander_ws.close()
