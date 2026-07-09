import pytest

from app import create_app


@pytest.fixture
def client():
    app = create_app(db_path=":memory:", jwt_secret="test-secret")
    app.testing = True
    return app.test_client()


def signup(client, email="owner@example.com", password="password123", workspace="Acme"):
    return client.post("/api/auth/signup", json={
        "email": email, "password": password, "workspace_name": workspace,
    })


def auth_headers(token):
    return {"Authorization": f"Bearer {token}"}


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}


def test_signup_creates_workspace_and_owner(client):
    resp = signup(client)
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["role"] == "owner"
    assert body["workspace_slug"] == "acme"
    assert "token" in body


def test_signup_duplicate_email_rejected(client):
    signup(client)
    resp = signup(client, workspace="Other Co")
    assert resp.status_code == 409


def test_signup_duplicate_workspace_name_gets_unique_slug(client):
    signup(client, email="a@example.com", workspace="Acme")
    resp = signup(client, email="b@example.com", workspace="Acme")
    assert resp.status_code == 201
    assert resp.get_json()["workspace_slug"] == "acme-2"


def test_signup_rejects_short_password(client):
    resp = client.post("/api/auth/signup", json={
        "email": "a@example.com", "password": "short", "workspace_name": "Acme",
    })
    assert resp.status_code == 400


def test_login_success(client):
    signup(client)
    resp = client.post("/api/auth/login", json={"email": "owner@example.com", "password": "password123"})
    assert resp.status_code == 200
    assert "token" in resp.get_json()


def test_login_wrong_password(client):
    signup(client)
    resp = client.post("/api/auth/login", json={"email": "owner@example.com", "password": "nope12345"})
    assert resp.status_code == 401


def test_login_unknown_email(client):
    resp = client.post("/api/auth/login", json={"email": "ghost@example.com", "password": "password123"})
    assert resp.status_code == 401


def test_login_multiple_workspaces_requires_slug(client):
    signup(client, email="a@example.com", workspace="First Co")
    token1 = signup(client, email="b@example.com", workspace="Second Co").get_json()["token"]

    # invite a@example.com into Second Co so it has 2 memberships
    client.post("/api/workspace/invite", headers=auth_headers(token1), json={"email": "a@example.com"})

    resp = client.post("/api/auth/login", json={"email": "a@example.com", "password": "password123"})
    assert resp.status_code == 409
    assert len(resp.get_json()["workspaces"]) == 2


def test_protected_route_requires_token(client):
    resp = client.get("/api/workspace")
    assert resp.status_code == 401


def test_protected_route_rejects_garbage_token(client):
    resp = client.get("/api/workspace", headers=auth_headers("garbage"))
    assert resp.status_code == 401


def test_get_current_workspace(client):
    token = signup(client).get_json()["token"]
    resp = client.get("/api/workspace", headers=auth_headers(token))
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["slug"] == "acme"
    assert len(body["members"]) == 1
    assert body["members"][0]["role"] == "owner"


def test_invite_requires_existing_account(client):
    token = signup(client).get_json()["token"]
    resp = client.post("/api/workspace/invite", headers=auth_headers(token),
                        json={"email": "nobody@example.com"})
    assert resp.status_code == 404


def test_invite_member_role_gating(client):
    owner_token = signup(client, email="owner@example.com").get_json()["token"]
    signup(client, email="member@example.com", workspace="Member Co")

    resp = client.post("/api/workspace/invite", headers=auth_headers(owner_token),
                        json={"email": "member@example.com", "role": "admin"})
    assert resp.status_code == 201
    assert resp.get_json()["role"] == "admin"


def test_create_and_list_projects(client):
    token = signup(client).get_json()["token"]
    resp = client.post("/api/projects", headers=auth_headers(token), json={"name": "Website Revamp"})
    assert resp.status_code == 201
    project_id = resp.get_json()["id"]

    resp = client.get("/api/projects", headers=auth_headers(token))
    assert resp.status_code == 200
    projects = resp.get_json()["projects"]
    assert len(projects) == 1
    assert projects[0]["id"] == project_id


def test_project_requires_name(client):
    token = signup(client).get_json()["token"]
    resp = client.post("/api/projects", headers=auth_headers(token), json={"name": "  "})
    assert resp.status_code == 400


def test_cannot_read_another_workspaces_project(client):
    token_a = signup(client, email="a@example.com", workspace="A Co").get_json()["token"]
    token_b = signup(client, email="b@example.com", workspace="B Co").get_json()["token"]

    project = client.post("/api/projects", headers=auth_headers(token_a), json={"name": "Secret"}).get_json()

    resp = client.get(f"/api/projects/{project['id']}", headers=auth_headers(token_b))
    assert resp.status_code == 404


def test_task_board_crud_and_status_columns(client):
    token = signup(client).get_json()["token"]
    project = client.post("/api/projects", headers=auth_headers(token), json={"name": "P"}).get_json()
    pid = project["id"]

    t1 = client.post(f"/api/projects/{pid}/tasks", headers=auth_headers(token),
                      json={"title": "Design", "status": "todo"}).get_json()
    t2 = client.post(f"/api/projects/{pid}/tasks", headers=auth_headers(token),
                      json={"title": "Build", "status": "todo"}).get_json()

    board = client.get(f"/api/projects/{pid}/tasks", headers=auth_headers(token)).get_json()["board"]
    assert [t["id"] for t in board["todo"]] == [t1["id"], t2["id"]]
    assert board["done"] == []

    # move t1 to done
    resp = client.patch(f"/api/tasks/{t1['id']}", headers=auth_headers(token), json={"status": "done"})
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "done"

    board = client.get(f"/api/projects/{pid}/tasks", headers=auth_headers(token)).get_json()["board"]
    assert [t["id"] for t in board["todo"]] == [t2["id"]]
    assert [t["id"] for t in board["done"]] == [t1["id"]]


def test_task_requires_title(client):
    token = signup(client).get_json()["token"]
    pid = client.post("/api/projects", headers=auth_headers(token), json={"name": "P"}).get_json()["id"]
    resp = client.post(f"/api/projects/{pid}/tasks", headers=auth_headers(token), json={"title": "  "})
    assert resp.status_code == 400


def test_task_rejects_invalid_status(client):
    token = signup(client).get_json()["token"]
    pid = client.post("/api/projects", headers=auth_headers(token), json={"name": "P"}).get_json()["id"]
    resp = client.post(f"/api/projects/{pid}/tasks", headers=auth_headers(token),
                        json={"title": "X", "status": "archived"})
    assert resp.status_code == 400


def test_update_task_empty_title_rejected(client):
    token = signup(client).get_json()["token"]
    pid = client.post("/api/projects", headers=auth_headers(token), json={"name": "P"}).get_json()["id"]
    tid = client.post(f"/api/projects/{pid}/tasks", headers=auth_headers(token),
                       json={"title": "X"}).get_json()["id"]
    resp = client.patch(f"/api/tasks/{tid}", headers=auth_headers(token), json={"title": "   "})
    assert resp.status_code == 400


def test_delete_task(client):
    token = signup(client).get_json()["token"]
    pid = client.post("/api/projects", headers=auth_headers(token), json={"name": "P"}).get_json()["id"]
    tid = client.post(f"/api/projects/{pid}/tasks", headers=auth_headers(token),
                       json={"title": "X"}).get_json()["id"]
    resp = client.delete(f"/api/tasks/{tid}", headers=auth_headers(token))
    assert resp.status_code == 200
    assert resp.get_json()["deleted"] is True

    resp = client.patch(f"/api/tasks/{tid}", headers=auth_headers(token), json={"title": "Y"})
    assert resp.status_code == 404


def test_cannot_touch_task_in_another_workspace(client):
    token_a = signup(client, email="a@example.com", workspace="A Co").get_json()["token"]
    token_b = signup(client, email="b@example.com", workspace="B Co").get_json()["token"]
    pid = client.post("/api/projects", headers=auth_headers(token_a), json={"name": "P"}).get_json()["id"]
    tid = client.post(f"/api/projects/{pid}/tasks", headers=auth_headers(token_a),
                       json={"title": "X"}).get_json()["id"]

    resp = client.patch(f"/api/tasks/{tid}", headers=auth_headers(token_b), json={"title": "hijacked"})
    assert resp.status_code == 404
    resp = client.delete(f"/api/tasks/{tid}", headers=auth_headers(token_b))
    assert resp.status_code == 404


def test_task_create_broadcasts_to_subscribers(client):
    """Verifies the REST write path calls the shared Broadcaster, without
    needing a real socket -- see test_ws_integration.py for the full
    real-socket, real-server end-to-end version of this."""
    token = signup(client).get_json()["token"]
    pid = client.post("/api/projects", headers=auth_headers(token), json={"name": "P"}).get_json()["id"]

    broadcaster = client.application.config["BROADCASTER"]
    received = []
    broadcaster.broadcast = lambda project_id, message, exclude=None: received.append((project_id, message))

    client.post(f"/api/projects/{pid}/tasks", headers=auth_headers(token), json={"title": "X"})
    assert len(received) == 1
    assert received[0][0] == pid
    assert received[0][1]["type"] == "task_created"


def test_update_task_rejects_boolean_position(client):
    """bool is a subclass of int in Python -- isinstance(True, int) is
    True -- so a naive `isinstance(x, int)` check would silently accept
    {"position": true} and store position=1. Regression test for that."""
    token = signup(client).get_json()["token"]
    pid = client.post("/api/projects", headers=auth_headers(token), json={"name": "P"}).get_json()["id"]
    tid = client.post(f"/api/projects/{pid}/tasks", headers=auth_headers(token),
                       json={"title": "X"}).get_json()["id"]
    resp = client.patch(f"/api/tasks/{tid}", headers=auth_headers(token), json={"position": True})
    assert resp.status_code == 400
