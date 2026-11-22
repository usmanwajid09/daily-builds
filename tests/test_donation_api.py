"""Tests for the donation_api Flask app, using Flask's test client."""
import pytest

from donation_api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture
def campaign(client):
    return client.post(
        "/campaigns",
        json={
            "title": "School Books",
            "goal_amount": 100000,
            "organizer": "NovaStack",
        },
    ).get_json()


def _donate(client, campaign_id, donor="Usman", amount=25000):
    return client.post(
        f"/campaigns/{campaign_id}/donations",
        json={"donor_name": donor, "amount": amount},
    )


def test_create_campaign_returns_201(client):
    resp = client.post(
        "/campaigns", json={"title": "Water Wells", "goal_amount": 50000}
    )
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["title"] == "Water Wells"
    assert body["goal_amount"] == 50000.0
    assert body["id"] == 1


def test_create_campaign_rejects_missing_title(client):
    assert client.post("/campaigns", json={"goal_amount": 100}).status_code == 400


def test_create_campaign_rejects_zero_goal(client):
    assert client.post(
        "/campaigns", json={"title": "X", "goal_amount": 0}
    ).status_code == 400


def test_create_campaign_rejects_negative_goal(client):
    assert client.post(
        "/campaigns", json={"title": "X", "goal_amount": -500}
    ).status_code == 400


def test_create_campaign_rejects_non_numeric_goal(client):
    assert client.post(
        "/campaigns", json={"title": "X", "goal_amount": "lots"}
    ).status_code == 400


def test_get_campaign_and_404(client, campaign):
    assert client.get(f"/campaigns/{campaign['id']}").get_json()["title"] == "School Books"
    assert client.get("/campaigns/999").status_code == 404


def test_update_campaign_title_and_goal(client, campaign):
    resp = client.put(
        f"/campaigns/{campaign['id']}",
        json={"title": "School Books 2026", "goal_amount": 150000},
    )
    assert resp.status_code == 200
    assert resp.get_json()["title"] == "School Books 2026"
    assert resp.get_json()["goal_amount"] == 150000.0


def test_update_campaign_cannot_lower_goal_below_raised(client, campaign):
    _donate(client, campaign["id"], amount=60000)
    resp = client.put(
        f"/campaigns/{campaign['id']}", json={"goal_amount": 50000}
    )
    assert resp.status_code == 409
    assert "already raised" in resp.get_json()["error"]


def test_delete_empty_campaign(client, campaign):
    assert client.delete(f"/campaigns/{campaign['id']}").status_code == 200
    assert client.get(f"/campaigns/{campaign['id']}").status_code == 404


def test_delete_campaign_with_donations_returns_409(client, campaign):
    _donate(client, campaign["id"])
    assert client.delete(f"/campaigns/{campaign['id']}").status_code == 409


def test_donate_returns_201(client, campaign):
    resp = _donate(client, campaign["id"])
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["donor_name"] == "Usman"
    assert body["amount"] == 25000.0
    assert body["campaign_id"] == campaign["id"]


def test_donate_rejects_zero_and_negative_amounts(client, campaign):
    assert _donate(client, campaign["id"], amount=0).status_code == 400
    assert _donate(client, campaign["id"], amount=-100).status_code == 400


def test_donate_rejects_missing_donor(client, campaign):
    resp = client.post(
        f"/campaigns/{campaign['id']}/donations", json={"amount": 100}
    )
    assert resp.status_code == 400


def test_donate_to_missing_campaign_404(client):
    assert _donate(client, 999).status_code == 404


def test_donate_rounds_to_two_decimals(client, campaign):
    resp = _donate(client, campaign["id"], amount=100.555)
    assert resp.get_json()["amount"] == 100.56


def test_summary_tracks_totals_and_progress(client, campaign):
    _donate(client, campaign["id"], "Usman", 25000)
    _donate(client, campaign["id"], "Hamza", 40000)
    _donate(client, campaign["id"], "Usman", 10000)
    summary = client.get(f"/campaigns/{campaign['id']}/summary").get_json()
    assert summary["total_raised"] == 75000.0
    assert summary["donation_count"] == 3
    assert summary["donor_count"] == 2
    assert summary["progress_percent"] == 75.0
    assert summary["amount_remaining"] == 25000.0
    assert summary["goal_reached"] is False


def test_summary_goal_reached_at_exact_goal(client, campaign):
    _donate(client, campaign["id"], amount=100000)
    summary = client.get(f"/campaigns/{campaign['id']}/summary").get_json()
    assert summary["goal_reached"] is True
    assert summary["progress_percent"] == 100.0
    assert summary["amount_remaining"] == 0.0


def test_summary_caps_progress_over_goal(client, campaign):
    _donate(client, campaign["id"], amount=150000)
    summary = client.get(f"/campaigns/{campaign['id']}/summary").get_json()
    assert summary["progress_percent"] == 100.0
    assert summary["goal_reached"] is True


def test_summary_empty_campaign(client, campaign):
    summary = client.get(f"/campaigns/{campaign['id']}/summary").get_json()
    assert summary["total_raised"] == 0.0
    assert summary["donation_count"] == 0
    assert summary["progress_percent"] == 0.0
    assert client.get("/campaigns/999/summary").status_code == 404


def test_list_campaigns_goal_reached_filter(client, campaign):
    client.post("/campaigns", json={"title": "Done", "goal_amount": 100})
    _donate(client, campaign["id"], amount=100000)
    reached = client.get("/campaigns", query_string={"goal_reached": "true"}).get_json()
    assert [c["title"] for c in reached] == ["School Books"]
    pending = client.get("/campaigns", query_string={"goal_reached": "false"}).get_json()
    assert [c["title"] for c in pending] == ["Done"]
    assert client.get("/campaigns", query_string={"goal_reached": "maybe"}).status_code == 400


def test_list_donations_and_get_one(client, campaign):
    donation = _donate(client, campaign["id"]).get_json()
    all_donations = client.get("/donations").get_json()
    assert len(all_donations) == 1
    assert client.get(f"/donations/{donation['id']}").get_json()["amount"] == 25000.0
    assert client.get("/donations/999").status_code == 404


def test_list_campaign_donations(client, campaign):
    _donate(client, campaign["id"], "Usman", 1000)
    _donate(client, campaign["id"], "Hamza", 2000)
    donations = client.get(f"/campaigns/{campaign['id']}/donations").get_json()
    assert [d["amount"] for d in donations] == [1000.0, 2000.0]
    assert client.get("/campaigns/999/donations").status_code == 404
