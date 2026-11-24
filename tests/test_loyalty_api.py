"""Tests for the loyalty_api Flask app, using Flask's test client."""
import pytest

from loyalty_api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _member(client, name="Alice", email="alice@example.com"):
    return client.post("/members", json={"name": name, "email": email})


def test_create_member_returns_201_with_zero_points_and_bronze(client):
    resp = _member(client)
    assert resp.status_code == 201
    data = resp.get_json()
    assert data["id"] == 1
    assert data["points"] == 0
    assert data["tier"] == "bronze"


def test_create_member_missing_name_returns_400(client):
    resp = client.post("/members", json={"email": "a@example.com"})
    assert resp.status_code == 400
    assert "name" in resp.get_json()["error"]


def test_create_member_invalid_email_returns_400(client):
    resp = client.post("/members", json={"name": "Alice", "email": "not-an-email"})
    assert resp.status_code == 400
    assert "email" in resp.get_json()["error"]


def test_create_member_duplicate_email_returns_409(client):
    _member(client)
    resp = _member(client, name="Alicia")
    assert resp.status_code == 409
    assert "already registered" in resp.get_json()["error"]


def test_get_member_and_missing_member_404(client):
    member = _member(client).get_json()
    assert client.get(f"/members/{member['id']}").get_json()["name"] == "Alice"
    assert client.get("/members/99").status_code == 404


def test_list_members(client):
    _member(client)
    _member(client, name="Bob", email="bob@example.com")
    members = client.get("/members").get_json()
    assert [m["name"] for m in members] == ["Alice", "Bob"]


def test_earn_points_upgrades_tier_to_silver(client):
    member = _member(client).get_json()
    resp = client.post(f"/members/{member['id']}/earn", json={"points": 1200})
    assert resp.status_code == 200
    assert resp.get_json()["points"] == 1200
    assert resp.get_json()["tier"] == "silver"


def test_earn_points_upgrades_tier_to_gold(client):
    member = _member(client).get_json()
    resp = client.post(f"/members/{member['id']}/earn", json={"points": 5000})
    assert resp.get_json()["tier"] == "gold"


def test_earn_points_invalid_amount_returns_400(client):
    member = _member(client).get_json()
    assert client.post(f"/members/{member['id']}/earn", json={"points": 0}).status_code == 400
    assert client.post(f"/members/{member['id']}/earn", json={"points": -5}).status_code == 400
    assert client.post(f"/members/{member['id']}/earn", json={"points": "ten"}).status_code == 400


def test_earn_points_missing_member_returns_404(client):
    assert client.post("/members/99/earn", json={"points": 10}).status_code == 404


def test_redeem_points_reduces_balance_and_tier(client):
    member = _member(client).get_json()
    client.post(f"/members/{member['id']}/earn", json={"points": 1500})
    resp = client.post(f"/members/{member['id']}/redeem", json={"points": 600})
    assert resp.status_code == 200
    assert resp.get_json()["points"] == 900
    assert resp.get_json()["tier"] == "bronze"


def test_redeem_more_than_balance_returns_409(client):
    member = _member(client).get_json()
    client.post(f"/members/{member['id']}/earn", json={"points": 100})
    resp = client.post(f"/members/{member['id']}/redeem", json={"points": 101})
    assert resp.status_code == 409
    assert client.get(f"/members/{member['id']}").get_json()["points"] == 100


def test_create_and_list_rewards(client):
    resp = client.post("/rewards", json={"name": "Free mug", "cost": 500})
    assert resp.status_code == 201
    assert resp.get_json()["id"] == 1
    assert client.post("/rewards", json={"name": "Bad", "cost": 0}).status_code == 400
    assert len(client.get("/rewards").get_json()) == 1


def test_redeem_reward_deducts_cost(client):
    member = _member(client).get_json()
    client.post(f"/members/{member['id']}/earn", json={"points": 1200})
    reward = client.post("/rewards", json={"name": "Free mug", "cost": 500}).get_json()
    resp = client.post(
        f"/members/{member['id']}/redeem-reward", json={"reward_id": reward["id"]}
    )
    assert resp.status_code == 200
    assert resp.get_json()["member"]["points"] == 700
    assert resp.get_json()["reward"]["name"] == "Free mug"


def test_redeem_reward_insufficient_points_returns_409(client):
    member = _member(client).get_json()
    client.post(f"/members/{member['id']}/earn", json={"points": 100})
    reward = client.post("/rewards", json={"name": "Free mug", "cost": 500}).get_json()
    resp = client.post(
        f"/members/{member['id']}/redeem-reward", json={"reward_id": reward["id"]}
    )
    assert resp.status_code == 409


def test_redeem_reward_missing_reward_returns_404(client):
    member = _member(client).get_json()
    assert client.post(
        f"/members/{member['id']}/redeem-reward", json={"reward_id": 99}
    ).status_code == 404


def test_transactions_record_earn_and_redeem(client):
    member = _member(client).get_json()
    client.post(f"/members/{member['id']}/earn", json={"points": 200})
    client.post(f"/members/{member['id']}/redeem", json={"points": 50})
    history = client.get(f"/members/{member['id']}/transactions").get_json()
    assert len(history) == 2
    assert history[0]["kind"] == "earn" and history[0]["points"] == 200
    assert history[1]["kind"] == "redeem" and history[1]["points"] == -50


def test_leaderboard_orders_by_points_descending(client):
    alice = _member(client).get_json()
    bob = _member(client, name="Bob", email="bob@example.com").get_json()
    client.post(f"/members/{bob['id']}/earn", json={"points": 300})
    client.post(f"/members/{alice['id']}/earn", json={"points": 100})
    board = client.get("/leaderboard").get_json()
    assert [m["name"] for m in board] == ["Bob", "Alice"]


def test_tiers_endpoint_lists_thresholds(client):
    tiers = client.get("/tiers").get_json()
    assert tiers == [
        {"tier": "bronze", "min_points": 0},
        {"tier": "silver", "min_points": 1000},
        {"tier": "gold", "min_points": 5000},
    ]


def test_delete_member(client):
    member = _member(client).get_json()
    assert client.delete(f"/members/{member['id']}").status_code == 200
    assert client.get(f"/members/{member['id']}").status_code == 404
