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


def invite_and_login(client, owner_token, email, role, workspace_slug="acme"):
    client.post("/api/workspace/invite", headers=auth_headers(owner_token), json={"email": email, "role": role})
    return client.post("/api/auth/login", json={
        "email": email, "password": "password123", "workspace_slug": workspace_slug,
    }).get_json()["token"]


# ----------------------------------------------------------- role changes
def test_owner_can_promote_member_to_admin(client):
    owner_token = signup(client).get_json()["token"]
    signup(client, email="member@example.com", workspace="Member Co")
    member_token = invite_and_login(client, owner_token, "member@example.com", "member")
    member_id = client.get("/api/workspace", headers=auth_headers(member_token)).get_json()["members"][1]["id"]

    resp = client.patch(f"/api/workspace/members/{member_id}", headers=auth_headers(owner_token),
                         json={"role": "admin"})
    assert resp.status_code == 200
    assert resp.get_json()["role"] == "admin"


def test_admin_can_promote_member_but_not_grant_owner(client):
    owner_token = signup(client).get_json()["token"]
    signup(client, email="admin@example.com", workspace="Admin Co")
    admin_token = invite_and_login(client, owner_token, "admin@example.com", "admin")
    signup(client, email="member@example.com", workspace="Member Co")
    member_token = invite_and_login(client, owner_token, "member@example.com", "member")
    member_id = client.get("/api/workspace", headers=auth_headers(member_token)).get_json()["members"][2]["id"]

    resp = client.patch(f"/api/workspace/members/{member_id}", headers=auth_headers(admin_token),
                         json={"role": "admin"})
    assert resp.status_code == 200

    resp = client.patch(f"/api/workspace/members/{member_id}", headers=auth_headers(admin_token),
                         json={"role": "owner"})
    assert resp.status_code == 403


def test_admin_cannot_demote_an_owner(client):
    owner_token = signup(client).get_json()["token"]
    signup(client, email="admin@example.com", workspace="Admin Co")
    admin_token = invite_and_login(client, owner_token, "admin@example.com", "admin")
    owner_id = client.get("/api/workspace", headers=auth_headers(admin_token)).get_json()["members"][0]["id"]

    resp = client.patch(f"/api/workspace/members/{owner_id}", headers=auth_headers(admin_token),
                         json={"role": "member"})
    assert resp.status_code == 403


def test_cannot_demote_the_last_owner(client):
    owner_token = signup(client).get_json()["token"]
    owner_id = client.get("/api/workspace", headers=auth_headers(owner_token)).get_json()["members"][0]["id"]

    resp = client.patch(f"/api/workspace/members/{owner_id}", headers=auth_headers(owner_token),
                         json={"role": "admin"})
    assert resp.status_code == 409


def test_can_demote_an_owner_if_another_owner_remains(client):
    owner_token = signup(client).get_json()["token"]
    signup(client, email="owner2@example.com", workspace="Owner2 Co")
    owner_id = client.get("/api/workspace", headers=auth_headers(owner_token)).get_json()["members"][0]["id"]
    invite_and_login(client, owner_token, "owner2@example.com", "owner")

    resp = client.patch(f"/api/workspace/members/{owner_id}", headers=auth_headers(owner_token),
                         json={"role": "admin"})
    assert resp.status_code == 200


def test_member_cannot_change_anyones_role(client):
    owner_token = signup(client).get_json()["token"]
    signup(client, email="member@example.com", workspace="Member Co")
    member_token = invite_and_login(client, owner_token, "member@example.com", "member")
    owner_id = client.get("/api/workspace", headers=auth_headers(member_token)).get_json()["members"][0]["id"]

    resp = client.patch(f"/api/workspace/members/{owner_id}", headers=auth_headers(member_token),
                         json={"role": "member"})
    assert resp.status_code == 403


def test_update_role_rejects_invalid_role_value(client):
    owner_token = signup(client).get_json()["token"]
    owner_id = client.get("/api/workspace", headers=auth_headers(owner_token)).get_json()["members"][0]["id"]
    resp = client.patch(f"/api/workspace/members/{owner_id}", headers=auth_headers(owner_token),
                         json={"role": "superadmin"})
    assert resp.status_code == 400


def test_role_change_creates_a_notification_for_the_target(client):
    owner_token = signup(client).get_json()["token"]
    signup(client, email="member@example.com", workspace="Member Co")
    member_token = invite_and_login(client, owner_token, "member@example.com", "member")
    member_id = client.get("/api/workspace", headers=auth_headers(member_token)).get_json()["members"][1]["id"]

    client.patch(f"/api/workspace/members/{member_id}", headers=auth_headers(owner_token), json={"role": "admin"})

    resp = client.get("/api/notifications", headers=auth_headers(member_token))
    body = resp.get_json()
    assert body["unread_count"] == 1
    assert body["notifications"][0]["type"] == "role_changed"


# --------------------------------------------------------------- removal
def test_member_can_remove_themselves(client):
    owner_token = signup(client).get_json()["token"]
    signup(client, email="member@example.com", workspace="Member Co")
    member_token = invite_and_login(client, owner_token, "member@example.com", "member")
    member_id = client.get("/api/workspace", headers=auth_headers(member_token)).get_json()["members"][1]["id"]

    resp = client.delete(f"/api/workspace/members/{member_id}", headers=auth_headers(member_token))
    assert resp.status_code == 200


def test_member_cannot_remove_another_member(client):
    owner_token = signup(client).get_json()["token"]
    signup(client, email="m1@example.com", workspace="M1 Co")
    m1_token = invite_and_login(client, owner_token, "m1@example.com", "member")
    signup(client, email="m2@example.com", workspace="M2 Co")
    invite_and_login(client, owner_token, "m2@example.com", "member")
    m2_id = client.get("/api/workspace", headers=auth_headers(m1_token)).get_json()["members"][2]["id"]

    resp = client.delete(f"/api/workspace/members/{m2_id}", headers=auth_headers(m1_token))
    assert resp.status_code == 403


def test_admin_cannot_remove_an_owner(client):
    owner_token = signup(client).get_json()["token"]
    signup(client, email="admin@example.com", workspace="Admin Co")
    admin_token = invite_and_login(client, owner_token, "admin@example.com", "admin")
    owner_id = client.get("/api/workspace", headers=auth_headers(admin_token)).get_json()["members"][0]["id"]

    resp = client.delete(f"/api/workspace/members/{owner_id}", headers=auth_headers(admin_token))
    assert resp.status_code == 403


def test_cannot_remove_the_last_owner_even_by_self(client):
    owner_token = signup(client).get_json()["token"]
    owner_id = client.get("/api/workspace", headers=auth_headers(owner_token)).get_json()["members"][0]["id"]

    resp = client.delete(f"/api/workspace/members/{owner_id}", headers=auth_headers(owner_token))
    assert resp.status_code == 409


def test_remove_nonmember_is_404(client):
    owner_token = signup(client).get_json()["token"]
    resp = client.delete("/api/workspace/members/999", headers=auth_headers(owner_token))
    assert resp.status_code == 404


# --------------------------------------------------------------- projects
def test_owner_can_delete_project(client):
    token = signup(client).get_json()["token"]
    pid = client.post("/api/projects", headers=auth_headers(token), json={"name": "P"}).get_json()["id"]

    resp = client.delete(f"/api/projects/{pid}", headers=auth_headers(token))
    assert resp.status_code == 200
    assert client.get(f"/api/projects/{pid}", headers=auth_headers(token)).status_code == 404


def test_member_cannot_delete_project(client):
    owner_token = signup(client).get_json()["token"]
    pid = client.post("/api/projects", headers=auth_headers(owner_token), json={"name": "P"}).get_json()["id"]
    signup(client, email="member@example.com", workspace="Member Co")
    member_token = invite_and_login(client, owner_token, "member@example.com", "member")

    resp = client.delete(f"/api/projects/{pid}", headers=auth_headers(member_token))
    assert resp.status_code == 403


def test_deleting_project_cascades_to_tasks_and_comments(client):
    token = signup(client).get_json()["token"]
    pid = client.post("/api/projects", headers=auth_headers(token), json={"name": "P"}).get_json()["id"]
    tid = client.post(f"/api/projects/{pid}/tasks", headers=auth_headers(token),
                       json={"title": "T"}).get_json()["id"]
    client.post(f"/api/tasks/{tid}/comments", headers=auth_headers(token), json={"body": "c"})

    resp = client.delete(f"/api/projects/{pid}", headers=auth_headers(token))
    assert resp.status_code == 200  # would 500 on an FK violation if cascade didn't fire

    assert client.get(f"/api/projects/{pid}/tasks", headers=auth_headers(token)).status_code == 404
