def signup(client, org_name="Acme", email="owner@acme.com", password="hunter2pass"):
    return client.post("/api/auth/signup", json={
        "org_name": org_name, "email": email, "password": password,
    })


def auth_header(token):
    return {"Authorization": f"Bearer {token}"}


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_signup_creates_tenant_and_returns_token(client):
    resp = signup(client)
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["role"] == "owner"
    assert body["tenant"]["slug"] == "acme"
    assert body["token"]


def test_signup_duplicate_email_rejected(client):
    signup(client, email="dup@acme.com")
    resp = signup(client, org_name="Other Org", email="dup@acme.com")
    assert resp.status_code == 409


def test_signup_weak_password_rejected(client):
    resp = client.post("/api/auth/signup", json={
        "org_name": "Acme", "email": "a@acme.com", "password": "short",
    })
    assert resp.status_code == 400


def test_signup_bad_email_rejected(client):
    resp = client.post("/api/auth/signup", json={
        "org_name": "Acme", "email": "not-an-email", "password": "longenoughpw",
    })
    assert resp.status_code == 400


def test_login_success(client):
    signup(client, email="owner@acme.com", password="hunter2pass")
    resp = client.post("/api/auth/login", json={
        "email": "owner@acme.com", "password": "hunter2pass",
    })
    assert resp.status_code == 200
    assert resp.get_json()["token"]


def test_login_wrong_password(client):
    signup(client, email="owner@acme.com", password="hunter2pass")
    resp = client.post("/api/auth/login", json={
        "email": "owner@acme.com", "password": "wrong-password",
    })
    assert resp.status_code == 401


def test_login_unknown_email(client):
    resp = client.post("/api/auth/login", json={
        "email": "nobody@nowhere.com", "password": "whatever123",
    })
    assert resp.status_code == 401


def test_dashboard_requires_token(client):
    resp = client.get("/api/dashboard")
    assert resp.status_code == 401


def test_dashboard_rejects_garbage_token(client):
    resp = client.get("/api/dashboard", headers=auth_header("not-a-real-token"))
    assert resp.status_code == 401


def test_dashboard_returns_tenant_scoped_data(client):
    signup_resp = signup(client, email="owner@acme.com")
    token = signup_resp.get_json()["token"]
    resp = client.get("/api/dashboard", headers=auth_header(token))
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["tenant"]["slug"] == "acme"
    assert body["member_count"] == 1
    assert body["your_role"] == "owner"


def test_me_endpoint(client):
    signup_resp = signup(client, email="owner@acme.com")
    token = signup_resp.get_json()["token"]
    resp = client.get("/api/auth/me", headers=auth_header(token))
    assert resp.status_code == 200
    assert resp.get_json()["user"]["email"] == "owner@acme.com"


def test_invite_adds_member_to_tenant(client):
    signup_resp = signup(client, email="owner@acme.com")
    token = signup_resp.get_json()["token"]

    resp = client.post("/api/auth/invite", json={
        "email": "teammate@acme.com", "role": "member",
    }, headers=auth_header(token))
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["temp_password"]

    members_resp = client.get("/api/tenants/me/members", headers=auth_header(token))
    emails = [m["email"] for m in members_resp.get_json()["members"]]
    assert "teammate@acme.com" in emails


def test_invite_forbidden_for_member_role(client):
    owner_token = signup(client, email="owner@acme.com").get_json()["token"]
    invite_resp = client.post("/api/auth/invite", json={
        "email": "teammate@acme.com", "role": "member",
    }, headers=auth_header(owner_token))
    temp_password = invite_resp.get_json()["temp_password"]

    login_resp = client.post("/api/auth/login", json={
        "email": "teammate@acme.com", "password": temp_password,
    })
    member_token = login_resp.get_json()["token"]

    resp = client.post("/api/auth/invite", json={
        "email": "another@acme.com", "role": "member",
    }, headers=auth_header(member_token))
    assert resp.status_code == 403


def test_tenant_isolation_across_two_orgs(client):
    acme_token = signup(client, org_name="Acme", email="owner@acme.com").get_json()["token"]
    globex_token = signup(client, org_name="Globex", email="owner@globex.com").get_json()["token"]

    client.post("/api/auth/invite", json={"email": "acme-teammate@acme.com", "role": "member"},
                 headers=auth_header(acme_token))

    acme_members = client.get("/api/tenants/me/members", headers=auth_header(acme_token)).get_json()["members"]
    globex_members = client.get("/api/tenants/me/members", headers=auth_header(globex_token)).get_json()["members"]

    acme_emails = {m["email"] for m in acme_members}
    globex_emails = {m["email"] for m in globex_members}

    assert "acme-teammate@acme.com" in acme_emails
    assert "acme-teammate@acme.com" not in globex_emails
    assert acme_emails.isdisjoint(globex_emails)


def test_login_disambiguates_multi_tenant_user(client):
    acme_token = signup(client, org_name="Acme", email="multi@example.com", password="samepassword1").get_json()["token"]
    signup(client, org_name="Globex", email="owner-globex@example.com", password="anotherpassword1")

    # invite multi@example.com into Globex too, so that user now belongs to 2 tenants
    globex_owner_token = client.post("/api/auth/login", json={
        "email": "owner-globex@example.com", "password": "anotherpassword1",
    }).get_json()["token"]
    client.post("/api/auth/invite", json={"email": "multi@example.com", "role": "member"},
                headers=auth_header(globex_owner_token))

    # login without tenant_slug should now be ambiguous
    ambiguous = client.post("/api/auth/login", json={
        "email": "multi@example.com", "password": "samepassword1",
    })
    assert ambiguous.status_code == 409
    assert len(ambiguous.get_json()["tenants"]) == 2

    # login with tenant_slug should resolve to the right tenant
    resolved = client.post("/api/auth/login", json={
        "email": "multi@example.com", "password": "samepassword1", "tenant_slug": "acme",
    })
    assert resolved.status_code == 200
    assert resolved.get_json()["tenant"]["slug"] == "acme"
