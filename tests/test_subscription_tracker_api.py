"""Tests for the subscription_tracker_api Flask app, using Flask's test client."""
import pytest

from subscription_tracker_api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _sub(client, name="Netflix", cost=9.99, cycle="monthly",
         renewal="2026-12-01", **extra):
    payload = {"name": name, "cost": cost, "billing_cycle": cycle,
               "renewal_date": renewal}
    payload.update(extra)
    return client.post("/subscriptions", json=payload)


def test_list_starts_empty(client):
    assert client.get("/subscriptions").get_json() == []


def test_create_monthly_returns_201(client):
    resp = _sub(client, category="streaming")
    assert resp.status_code == 201
    assert resp.get_json() == {
        "id": 1,
        "name": "Netflix",
        "cost": 9.99,
        "billing_cycle": "monthly",
        "renewal_date": "2026-12-01",
        "category": "streaming",
        "monthly_cost": 9.99,
        "yearly_cost": 119.88,
    }


def test_create_yearly_converts_costs(client):
    body = _sub(client, name="GitHub Pro", cost=48, cycle="yearly",
                renewal="2027-03-10").get_json()
    assert body["monthly_cost"] == 4.0
    assert body["yearly_cost"] == 48.0


def test_category_defaults_to_none(client):
    assert _sub(client).get_json()["category"] is None


def test_create_missing_name_400(client):
    resp = client.post("/subscriptions", json={"cost": 5,
                                               "billing_cycle": "monthly",
                                               "renewal_date": "2026-12-01"})
    assert resp.status_code == 400


def test_create_bad_cost_400(client):
    assert _sub(client, cost=0).status_code == 400
    assert _sub(client, cost=-5).status_code == 400
    assert _sub(client, cost="free").status_code == 400


def test_create_bad_cycle_400(client):
    assert _sub(client, cycle="weekly").status_code == 400
    assert _sub(client, cycle="Monthly").status_code == 400


def test_create_bad_renewal_date_400(client):
    assert _sub(client, renewal="01-12-2026").status_code == 400
    assert _sub(client, renewal="2026-13-01").status_code == 400


def test_duplicate_name_409_case_insensitive(client):
    assert _sub(client).status_code == 201
    assert _sub(client, name="netflix").status_code == 409


def test_get_subscription(client):
    _sub(client)
    resp = client.get("/subscriptions/1")
    assert resp.status_code == 200
    assert resp.get_json()["name"] == "Netflix"


def test_get_missing_404(client):
    assert client.get("/subscriptions/99").status_code == 404


def test_filter_by_category(client):
    _sub(client, name="Netflix", category="streaming")
    _sub(client, name="Spotify", cost=5.99, renewal="2026-11-25",
         category="music")
    rows = client.get("/subscriptions?category=streaming").get_json()
    assert [r["name"] for r in rows] == ["Netflix"]


def test_filter_by_cycle(client):
    _sub(client, name="Netflix")
    _sub(client, name="GitHub Pro", cost=48, cycle="yearly",
         renewal="2027-03-10")
    rows = client.get("/subscriptions?cycle=yearly").get_json()
    assert [r["name"] for r in rows] == ["GitHub Pro"]


def test_filter_bad_cycle_400(client):
    assert client.get("/subscriptions?cycle=weekly").status_code == 400


def test_update_cost_and_cycle(client):
    _sub(client)
    resp = client.put("/subscriptions/1",
                      json={"cost": 120, "billing_cycle": "yearly"})
    assert resp.status_code == 200
    assert resp.get_json()["monthly_cost"] == 10.0
    assert resp.get_json()["yearly_cost"] == 120.0


def test_update_to_duplicate_name_409(client):
    _sub(client, name="Netflix")
    _sub(client, name="Spotify", cost=5.99)
    assert client.put("/subscriptions/2",
                      json={"name": "NETFLIX"}).status_code == 409


def test_update_nothing_400(client):
    _sub(client)
    assert client.put("/subscriptions/1", json={}).status_code == 400


def test_update_missing_404(client):
    assert client.put("/subscriptions/99", json={"cost": 5}).status_code == 404


def test_delete_subscription(client):
    _sub(client)
    assert client.delete("/subscriptions/1").get_json() == {"deleted": 1}
    assert client.get("/subscriptions/1").status_code == 404


def test_delete_missing_404(client):
    assert client.delete("/subscriptions/99").status_code == 404


def test_totals_empty(client):
    totals = client.get("/totals").get_json()
    assert totals == {"count": 0, "monthly_total": 0, "yearly_total": 0,
                      "by_category_monthly": {}}


def test_totals_mixed_cycles(client):
    _sub(client, name="Netflix", cost=9.99, category="streaming")
    _sub(client, name="GitHub Pro", cost=48, cycle="yearly",
         renewal="2027-03-10", category="dev tools")
    totals = client.get("/totals").get_json()
    assert totals["count"] == 2
    assert totals["monthly_total"] == 13.99
    assert totals["yearly_total"] == 167.88
    assert totals["by_category_monthly"] == {"streaming": 9.99,
                                             "dev tools": 4.0}


def test_upcoming_within_window(client):
    _sub(client, name="Spotify", cost=5.99, renewal="2026-11-25")
    _sub(client, name="Netflix", cost=9.99, renewal="2026-12-01")
    rows = client.get("/upcoming?as_of=2026-11-21&days=10").get_json()
    assert len(rows) == 2
    assert rows[0]["name"] == "Spotify"
    assert rows[0]["next_renewal"] == "2026-11-25"
    assert rows[0]["days_until_renewal"] == 4
    assert rows[1]["name"] == "Netflix"
    assert rows[1]["days_until_renewal"] == 10


def test_upcoming_rolls_past_renewal_forward(client):
    # Monthly plan whose stored date is in the past rolls to next month.
    _sub(client, name="Gym", cost=25, renewal="2026-10-15")
    rows = client.get("/upcoming?as_of=2026-11-21&days=30").get_json()
    assert len(rows) == 1
    assert rows[0]["next_renewal"] == "2026-12-15"
    assert rows[0]["days_until_renewal"] == 24


def test_upcoming_excludes_far_renewals(client):
    _sub(client, name="GitHub Pro", cost=48, cycle="yearly",
         renewal="2027-03-10")
    assert client.get("/upcoming?as_of=2026-11-21&days=30").get_json() == []


def test_upcoming_bad_params_400(client):
    assert client.get("/upcoming?as_of=21-11-2026").status_code == 400
    assert client.get("/upcoming?days=soon").status_code == 400
    assert client.get("/upcoming?days=-1").status_code == 400


def test_unknown_route_404(client):
    assert client.get("/nope").status_code == 404
