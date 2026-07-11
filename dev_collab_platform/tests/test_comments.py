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


def make_task(client, token, title="X"):
    pid = client.post("/api/projects", headers=auth_headers(token), json={"name": "P"}).get_json()["id"]
    tid = client.post(f"/api/projects/{pid}/tasks", headers=auth_headers(token),
                       json={"title": title}).get_json()["id"]
    return pid, tid


def test_create_and_list_comments(client):
    token = signup(client).get_json()["token"]
    _pid, tid = make_task(client, token)

    resp = client.post(f"/api/tasks/{tid}/comments", headers=auth_headers(token), json={"body": "first"})
    assert resp.status_code == 201
    comment = resp.get_json()
    assert comment["body"] == "first"
    assert comment["task_id"] == tid

    client.post(f"/api/tasks/{tid}/comments", headers=auth_headers(token), json={"body": "second"})

    resp = client.get(f"/api/tasks/{tid}/comments", headers=auth_headers(token))
    bodies = [c["body"] for c in resp.get_json()["comments"]]
    assert bodies == ["first", "second"]


def test_comment_requires_body(client):
    token = signup(client).get_json()["token"]
    _pid, tid = make_task(client, token)
    resp = client.post(f"/api/tasks/{tid}/comments", headers=auth_headers(token), json={"body": "   "})
    assert resp.status_code == 400


def test_comment_on_task_in_another_workspace_is_404(client):
    token_a = signup(client, email="a@example.com", workspace="A Co").get_json()["token"]
    token_b = signup(client, email="b@example.com", workspace="B Co").get_json()["token"]
    _pid, tid = make_task(client, token_a)

    resp = client.post(f"/api/tasks/{tid}/comments", headers=auth_headers(token_b), json={"body": "hijack"})
    assert resp.status_code == 404
    resp = client.get(f"/api/tasks/{tid}/comments", headers=auth_headers(token_b))
    assert resp.status_code == 404


def test_comment_on_nonexistent_task_is_404(client):
    token = signup(client).get_json()["token"]
    resp = client.post("/api/tasks/999/comments", headers=auth_headers(token), json={"body": "x"})
    assert resp.status_code == 404


def test_author_can_delete_own_comment(client):
    token = signup(client).get_json()["token"]
    _pid, tid = make_task(client, token)
    cid = client.post(f"/api/tasks/{tid}/comments", headers=auth_headers(token),
                       json={"body": "mine"}).get_json()["id"]

    resp = client.delete(f"/api/comments/{cid}", headers=auth_headers(token))
    assert resp.status_code == 200
    assert resp.get_json()["deleted"] is True

    resp = client.get(f"/api/tasks/{tid}/comments", headers=auth_headers(token))
    assert resp.get_json()["comments"] == []


def test_member_cannot_delete_someone_elses_comment(client):
    owner_token = signup(client, email="owner@example.com").get_json()["token"]
    signup(client, email="member@example.com", workspace="Member Co")
    client.post("/api/workspace/invite", headers=auth_headers(owner_token),
                json={"email": "member@example.com", "role": "member"})
    member_token = client.post("/api/auth/login", json={
        "email": "member@example.com", "password": "password123", "workspace_slug": "acme",
    }).get_json()["token"]

    _pid, tid = make_task(client, owner_token)
    cid = client.post(f"/api/tasks/{tid}/comments", headers=auth_headers(owner_token),
                       json={"body": "owner's comment"}).get_json()["id"]

    resp = client.delete(f"/api/comments/{cid}", headers=auth_headers(member_token))
    assert resp.status_code == 403


def test_admin_can_delete_anyones_comment(client):
    owner_token = signup(client, email="owner@example.com").get_json()["token"]
    signup(client, email="member@example.com", workspace="Member Co")
    client.post("/api/workspace/invite", headers=auth_headers(owner_token),
                json={"email": "member@example.com", "role": "admin"})
    admin_token = client.post("/api/auth/login", json={
        "email": "member@example.com", "password": "password123", "workspace_slug": "acme",
    }).get_json()["token"]

    _pid, tid = make_task(client, owner_token)
    cid = client.post(f"/api/tasks/{tid}/comments", headers=auth_headers(owner_token),
                       json={"body": "owner's comment"}).get_json()["id"]

    resp = client.delete(f"/api/comments/{cid}", headers=auth_headers(admin_token))
    assert resp.status_code == 200


def test_deleting_task_cascades_to_comments(client):
    token = signup(client).get_json()["token"]
    _pid, tid = make_task(client, token)
    client.post(f"/api/tasks/{tid}/comments", headers=auth_headers(token), json={"body": "will vanish"})

    client.delete(f"/api/tasks/{tid}", headers=auth_headers(token))
    # The task is gone, so even asking for its comments 404s -- but the
    # real point of this test is that the DELETE above didn't raise a
    # foreign-key-constraint error, proving the CASCADE actually fired.
    resp = client.get(f"/api/tasks/{tid}/comments", headers=auth_headers(token))
    assert resp.status_code == 404


def test_mention_of_workspace_member_creates_notification(client):
    owner_token = signup(client, email="owner@example.com").get_json()["token"]
    signup(client, email="member@example.com", workspace="Member Co")
    client.post("/api/workspace/invite", headers=auth_headers(owner_token),
                json={"email": "member@example.com", "role": "member"})
    member_token = client.post("/api/auth/login", json={
        "email": "member@example.com", "password": "password123", "workspace_slug": "acme",
    }).get_json()["token"]

    _pid, tid = make_task(client, owner_token)
    client.post(f"/api/tasks/{tid}/comments", headers=auth_headers(owner_token),
                json={"body": "cc @member@example.com please take a look"})

    resp = client.get("/api/notifications", headers=auth_headers(member_token))
    body = resp.get_json()
    assert body["unread_count"] == 1
    assert len(body["notifications"]) == 1
    notif = body["notifications"][0]
    assert notif["type"] == "mention"
    assert "owner@example.com" in notif["message"]
    assert notif["task_id"] == tid


def test_mention_of_non_member_email_is_silently_ignored(client):
    token = signup(client).get_json()["token"]
    _pid, tid = make_task(client, token)
    # Should not raise or notify anyone -- @nobody isn't a workspace member.
    resp = client.post(f"/api/tasks/{tid}/comments", headers=auth_headers(token),
                        json={"body": "cc @nobody@example.com"})
    assert resp.status_code == 201


def test_mention_matching_is_case_insensitive(client):
    """Regression test: mentions are lower-cased before matching against
    workspace member emails (which are stored lower-cased at signup), so
    @Member@Example.COM should still resolve to member@example.com."""
    owner_token = signup(client, email="owner@example.com").get_json()["token"]
    signup(client, email="member@example.com", workspace="Member Co")
    client.post("/api/workspace/invite", headers=auth_headers(owner_token),
                json={"email": "member@example.com", "role": "member"})
    member_token = client.post("/api/auth/login", json={
        "email": "member@example.com", "password": "password123", "workspace_slug": "acme",
    }).get_json()["token"]

    _pid, tid = make_task(client, owner_token)
    client.post(f"/api/tasks/{tid}/comments", headers=auth_headers(owner_token),
                json={"body": "cc @Member@Example.COM"})

    resp = client.get("/api/notifications", headers=auth_headers(member_token))
    assert resp.get_json()["unread_count"] == 1


def test_self_mention_does_not_notify_yourself(client):
    token = signup(client, email="owner@example.com").get_json()["token"]
    _pid, tid = make_task(client, token)
    client.post(f"/api/tasks/{tid}/comments", headers=auth_headers(token),
                json={"body": "note to self @owner@example.com"})

    resp = client.get("/api/notifications", headers=auth_headers(token))
    assert resp.get_json()["notifications"] == []
