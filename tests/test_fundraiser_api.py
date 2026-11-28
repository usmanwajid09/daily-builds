"""Tests for the fundraiser_api Flask app, using Flask's test client."""
import pytest

from fundraiser_api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _campaign(client, title="School Library Fund", goal_amount=1000,
              **extra):
    payload = {"title": title, "goal_amount": goal_amount}
    payload.update(extra)
    return client.post("/campaigns", json=payload)


def _donate(client, campaign_id, donor_name="Ali", amount=100, **extra):
    payload = {"donor_name": donor_name, "amount": amount}
    payload.update(extra)
    return client.post(f"/campaigns/{campaign_id}/donate", json=payload)


def test_list_starts_empty(client):
    assert client.get("/campaigns").get_json() == []


def test_create_campaign_returns_201(client):
    response = _campaign(client)
    assert response.status_code == 201
    body = response.get_json()
    assert body["title"] == "School Library Fund"
    assert body["goal_amount"] == 1000
    assert body["raised"] == 0
    assert body["progress_percent"] == 0
    assert body["donor_count"] == 0
    assert body["id"] == 1


def test_create_campaign_missing_title_returns_400(client):
    response = client.post("/campaigns", json={"goal_amount": 100})
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_create_campaign_invalid_goal_returns_400(client):
    for bad_goal in [0, -50, "1000", None]:
        response = client.post(
            "/campaigns", json={"title": "Bad", "goal_amount": bad_goal})
        assert response.status_code == 400
        assert "error" in response.get_json()


def test_get_unknown_campaign_returns_404(client):
    response = client.get("/campaigns/999")
    assert response.status_code == 404
    assert "error" in response.get_json()


def test_get_campaign_includes_derived_fields(client):
    _campaign(client)
    _donate(client, 1, "Ali", 250)
    body = client.get("/campaigns/1").get_json()
    assert body["raised"] == 250
    assert body["progress_percent"] == 25.0
    assert body["donor_count"] == 1


def test_donate_returns_201(client):
    _campaign(client)
    response = _donate(client, 1, "Sara", 400, message="Good luck!")
    assert response.status_code == 201
    body = response.get_json()
    assert body["donor_name"] == "Sara"
    assert body["amount"] == 400
    assert body["message"] == "Good luck!"
    assert body["campaign_id"] == 1
    assert body["id"] == 1


def test_donate_invalid_amount_returns_400(client):
    _campaign(client)
    for bad_amount in [0, -10, "50", None]:
        response = _donate(client, 1, "Ali", bad_amount)
        assert response.status_code == 400
        assert "error" in response.get_json()


def test_donate_missing_donor_name_returns_400(client):
    _campaign(client)
    response = client.post("/campaigns/1/donate", json={"amount": 50})
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_donate_to_unknown_campaign_returns_404(client):
    response = _donate(client, 999, "Ali", 50)
    assert response.status_code == 404
    assert "error" in response.get_json()


def test_progress_math(client):
    _campaign(client, goal_amount=1000)
    _donate(client, 1, "Ali", 250)
    _donate(client, 1, "Sara", 400)
    body = client.get("/campaigns/1/progress").get_json()
    assert body["campaign_id"] == 1
    assert body["title"] == "School Library Fund"
    assert body["goal_amount"] == 1000
    assert body["raised"] == 650
    assert body["remaining"] == 350
    assert body["progress_percent"] == 65.0
    assert body["donor_count"] == 2
    assert body["goal_reached"] is False


def test_goal_reached_true_when_raised_meets_goal(client):
    _campaign(client, goal_amount=500)
    _donate(client, 1, "Ali", 300)
    _donate(client, 1, "Sara", 300)
    body = client.get("/campaigns/1/progress").get_json()
    assert body["raised"] == 600
    assert body["remaining"] == 0
    assert body["progress_percent"] == 120.0
    assert body["goal_reached"] is True


def test_donations_list_newest_first(client):
    _campaign(client)
    _donate(client, 1, "Ali", 100)
    _donate(client, 1, "Sara", 200)
    _donate(client, 1, "Bilal", 50)
    body = client.get("/campaigns/1/donations").get_json()
    assert [d["donor_name"] for d in body] == ["Bilal", "Sara", "Ali"]
    assert [d["amount"] for d in body] == [50, 200, 100]


def test_donations_unknown_campaign_returns_404(client):
    response = client.get("/campaigns/999/donations")
    assert response.status_code == 404
    assert "error" in response.get_json()


def test_top_donors_aggregated_and_sorted_desc(client):
    _campaign(client)
    _donate(client, 1, "Ali", 250)
    _donate(client, 1, "Sara", 400)
    _donate(client, 1, "Ali", 150)
    _donate(client, 1, "Bilal", 100)
    body = client.get("/campaigns/1/top-donors").get_json()
    assert body == [
        {"donor_name": "Ali", "total_donated": 400, "donation_count": 2},
        {"donor_name": "Sara", "total_donated": 400, "donation_count": 1},
        {"donor_name": "Bilal", "total_donated": 100, "donation_count": 1},
    ]


def test_top_donors_unknown_campaign_returns_404(client):
    response = client.get("/campaigns/999/top-donors")
    assert response.status_code == 404
    assert "error" in response.get_json()


def test_delete_campaign(client):
    _campaign(client)
    response = client.delete("/campaigns/1")
    assert response.status_code == 200
    assert client.get("/campaigns/1").status_code == 404
    assert client.get("/campaigns").get_json() == []


def test_delete_unknown_campaign_returns_404(client):
    response = client.delete("/campaigns/999")
    assert response.status_code == 404
    assert "error" in response.get_json()
