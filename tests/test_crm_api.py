"""Tests for the crm_api Flask app, using Flask's test client."""
import pytest

from crm_api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _contact(client, name="Sara Ahmed", email="sara@acme.com", **extra):
    payload = {"name": name, "email": email}
    payload.update(extra)
    return client.post("/contacts", json=payload)


def _deal(client, contact_id, title="Website", value=1000, **extra):
    payload = {"title": title, "value": value}
    payload.update(extra)
    return client.post(f"/contacts/{contact_id}/deals", json=payload)


@pytest.fixture
def contact(client):
    return _contact(client).get_json()["id"]


def test_list_contacts_starts_empty(client):
    assert client.get("/contacts").get_json() == []


def test_create_contact_returns_201(client):
    resp = _contact(client, company="Acme Corp")
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["id"] == 1
    assert body["company"] == "Acme Corp"
    assert body["deals"] == []


def test_create_contact_validation_400(client):
    assert _contact(client, name="").status_code == 400
    assert _contact(client, email="not-an-email").status_code == 400
    assert _contact(client, email="missing-domain@").status_code == 400
    assert _contact(client, email="no-at-sign.com").status_code == 400


def test_create_contact_duplicate_email_409(client, contact):
    assert _contact(client, name="Someone Else", email="SARA@acme.com").status_code == 409


def test_get_contact_404(client):
    assert client.get("/contacts/99").status_code == 404


def test_update_contact(client, contact):
    resp = client.put(f"/contacts/{contact}", json={"phone": "0300-1234567"})
    assert resp.status_code == 200
    assert resp.get_json()["phone"] == "0300-1234567"
    assert client.put(f"/contacts/{contact}", json={}).status_code == 400
    assert client.put(
        f"/contacts/{contact}", json={"email": "bad"}
    ).status_code == 400


def test_update_contact_duplicate_email_409(client, contact):
    _contact(client, name="Omar", email="omar@globex.com")
    assert client.put(
        f"/contacts/{contact}", json={"email": "omar@globex.com"}
    ).status_code == 409


def test_delete_contact_with_deals_409(client, contact):
    _deal(client, contact)
    assert client.delete(f"/contacts/{contact}").status_code == 409


def test_delete_contact_empty_ok(client, contact):
    assert client.delete(f"/contacts/{contact}").status_code == 200
    assert client.get(f"/contacts/{contact}").status_code == 404


def test_create_deal_returns_201(client, contact):
    resp = _deal(client, contact, value=5000, stage="qualified")
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["stage"] == "qualified"
    assert body["contact_id"] == contact


def test_create_deal_defaults_to_lead(client, contact):
    assert _deal(client, contact).get_json()["stage"] == "lead"


def test_create_deal_validation_400(client, contact):
    assert _deal(client, contact, title="").status_code == 400
    assert _deal(client, contact, value=-5).status_code == 400
    assert _deal(client, contact, value="lots").status_code == 400
    assert _deal(client, contact, stage="maybe").status_code == 400
    assert _deal(client, contact).status_code == 201  # value present
    assert client.post(
        f"/contacts/{contact}/deals", json={"title": "No value"}
    ).status_code == 400


def test_create_deal_unknown_contact_404(client):
    assert _deal(client, 99).status_code == 404


def test_list_deals_filters(client, contact):
    _deal(client, contact, title="A", value=100, stage="lead")
    _deal(client, contact, title="B", value=200, stage="won")
    assert len(client.get("/deals").get_json()) == 2
    won = client.get("/deals?stage=won").get_json()
    assert [d["title"] for d in won] == ["B"]
    assert client.get("/deals?stage=nope").status_code == 400
    by_contact = client.get(f"/deals?contact_id={contact}").get_json()
    assert len(by_contact) == 2
    assert client.get("/deals?contact_id=99").get_json() == []


def test_get_deal_404(client):
    assert client.get("/deals/99").status_code == 404


def test_update_deal(client, contact):
    deal = _deal(client, contact).get_json()
    resp = client.put(f"/deals/{deal['id']}", json={"value": 2500})
    assert resp.status_code == 200
    assert resp.get_json()["value"] == 2500
    assert client.put(f"/deals/{deal['id']}", json={}).status_code == 400


def test_stage_move_and_terminal_409(client, contact):
    deal = _deal(client, contact).get_json()
    url = f"/deals/{deal['id']}/stage"
    assert client.post(url, json={"stage": "won"}).status_code == 200
    assert client.post(url, json={"stage": "proposal"}).status_code == 409
    assert client.post(url, json={"stage": "won"}).status_code == 409
    assert client.post(url, json={"stage": "nope"}).status_code == 400


def test_delete_deal(client, contact):
    deal = _deal(client, contact).get_json()
    assert client.delete(f"/deals/{deal['id']}").status_code == 200
    assert client.get(f"/deals/{deal['id']}").status_code == 404


def test_pipeline_stats(client, contact):
    _deal(client, contact, title="A", value=5000, stage="lead")
    _deal(client, contact, title="B", value=12000, stage="qualified")
    _deal(client, contact, title="C", value=3000, stage="won")
    _deal(client, contact, title="D", value=4000, stage="lost")
    pipeline = client.get("/pipeline").get_json()
    assert pipeline["total_deals"] == 4
    assert pipeline["total_value"] == 24000
    assert pipeline["by_stage"]["qualified"] == {
        "count": 1, "total_value": 12000,
    }
    assert pipeline["won_value"] == 3000
    assert pipeline["lost_value"] == 4000
    assert pipeline["conversion_rate"] == 50.0


def test_pipeline_conversion_null_when_nothing_closed(client, contact):
    _deal(client, contact, title="A", value=1000, stage="lead")
    pipeline = client.get("/pipeline").get_json()
    assert pipeline["conversion_rate"] is None
    assert pipeline["total_value"] == 1000


def test_contact_includes_deals(client, contact):
    _deal(client, contact, title="A", value=100)
    body = client.get(f"/contacts/{contact}").get_json()
    assert [d["title"] for d in body["deals"]] == ["A"]
