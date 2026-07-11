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


def _setup_mention(client):
    """owner mentions member in a task comment -> member gets one notification."""
    owner_token = signup(client, email="owner@example.com").get_json()["token"]
    signup(client, email="member@example.com", workspace="Member Co")
    client.post("/api/workspace/invite", headers=auth_headers(owner_token),
                json={"email": "member@example.com", "role": "member"})
    member_token = client.post("/api/auth/login", json={
        "email": "member@example.com", "password": "password123", "workspace_slug": "acme",
    }).get_json()["token"]

    pid = client.post("/api/projects", headers=auth_headers(owner_token), json={"name": "P"}).get_json()["id"]
    tid = client.post(f"/api/projects/{pid}/tasks", headers=auth_headers(owner_token),
                       json={"title": "X"}).get_json()["id"]
    client.post(f"/api/tasks/{tid}/comments", headers=auth_headers(owner_token),
                json={"body": "cc @member@example.com"})
    return owner_token, member_token


def test_notifications_requires_auth(client):
    resp = client.get("/api/notifications")
    assert resp.status_code == 401


def test_list_notifications_empty_by_default(client):
    token = signup(client).get_json()["token"]
    resp = client.get("/api/notifications", headers=auth_headers(token))
    assert resp.get_json() == {"notifications": [], "unread_count": 0}


def test_mark_single_notification_read(client):
    _owner_token, member_token = _setup_mention(client)
    notif_id = client.get("/api/notifications", headers=auth_headers(member_token)) \
        .get_json()["notifications"][0]["id"]

    resp = client.post(f"/api/notifications/{notif_id}/read", headers=auth_headers(member_token))
    assert resp.status_code == 200
    assert resp.get_json()["read_at"] is not None

    resp = client.get("/api/notifications", headers=auth_headers(member_token))
    assert resp.get_json()["unread_count"] == 0


def test_mark_all_notifications_read(client):
    owner_token, member_token = _setup_mention(client)
    pid = client.post("/api/projects", headers=auth_headers(owner_token), json={"name": "P2"}).get_json()["id"]
    tid = client.post(f"/api/projects/{pid}/tasks", headers=auth_headers(owner_token),
                       json={"title": "Y"}).get_json()["id"]
    client.post(f"/api/tasks/{tid}/comments", headers=auth_headers(owner_token),
                json={"body": "again cc @member@example.com"})

    assert client.get("/api/notifications", headers=auth_headers(member_token)) \
        .get_json()["unread_count"] == 2

    resp = client.post("/api/notifications/read-all", headers=auth_headers(member_token))
    assert resp.get_json()["marked"] == 2

    assert client.get("/api/notifications", headers=auth_headers(member_token)) \
        .get_json()["unread_count"] == 0


def test_unread_only_filter(client):
    _owner_token, member_token = _setup_mention(client)
    notif_id = client.get("/api/notifications", headers=auth_headers(member_token)) \
        .get_json()["notifications"][0]["id"]
    client.post(f"/api/notifications/{notif_id}/read", headers=auth_headers(member_token))

    resp = client.get("/api/notifications?unread=1", headers=auth_headers(member_token))
    assert resp.get_json()["notifications"] == []


def test_cannot_mark_someone_elses_notification_read(client):
    owner_token, member_token = _setup_mention(client)
    notif_id = client.get("/api/notifications", headers=auth_headers(member_token)) \
        .get_json()["notifications"][0]["id"]

    resp = client.post(f"/api/notifications/{notif_id}/read", headers=auth_headers(owner_token))
    assert resp.status_code == 404


def test_notifications_scoped_to_workspace(client):
    """A user who is a member of two workspaces only sees the current
    (token-scoped) workspace's notifications, not the other one's."""
    owner_token, member_token = _setup_mention(client)

    # member also joins a second workspace where they get mentioned again.
    signup(client, email="owner2@example.com", workspace="Second Co")
    owner2_token = client.post("/api/auth/login", json={"email": "owner2@example.com", "password": "password123"}) \
        .get_json()["token"]
    client.post("/api/workspace/invite", headers=auth_headers(owner2_token),
                json={"email": "member@example.com", "role": "member"})
    member_token_ws2 = client.post("/api/auth/login", json={
        "email": "member@example.com", "password": "password123", "workspace_slug": "second-co",
    }).get_json()["token"]
    pid2 = client.post("/api/projects", headers=auth_headers(owner2_token), json={"name": "P"}).get_json()["id"]
    tid2 = client.post(f"/api/projects/{pid2}/tasks", headers=auth_headers(owner2_token),
                        json={"title": "T"}).get_json()["id"]
    client.post(f"/api/tasks/{tid2}/comments", headers=auth_headers(owner2_token),
                json={"body": "cc @member@example.com"})

    # First workspace's token should still only show 1 notification (from acme), not 2.
    resp = client.get("/api/notifications", headers=auth_headers(member_token))
    assert len(resp.get_json()["notifications"]) == 1
    resp = client.get("/api/notifications", headers=auth_headers(member_token_ws2))
    assert len(resp.get_json()["notifications"]) == 1
