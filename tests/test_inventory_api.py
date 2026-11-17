"""Tests for inventory_api."""
import pytest

from inventory_api.app import create_app


@pytest.fixture
def client():
    return create_app().test_client()


def _add(client, name="Widget", stock=10, threshold=5, price=None):
    payload = {"name": name, "stock": stock, "threshold": threshold}
    if price is not None:
        payload["price"] = price
    return client.post("/items", json=payload)


def test_add_item(client):
    resp = _add(client, price=2.5)
    assert resp.status_code == 201
    data = resp.get_json()
    assert data["id"] == 1
    assert data["stock"] == 10
    assert data["low_stock"] is False


def test_add_item_low_stock_flag(client):
    data = _add(client, stock=3).get_json()
    assert data["low_stock"] is True


def test_list_items(client):
    _add(client, "A")
    _add(client, "B")
    listed = client.get("/items").get_json()
    assert len(listed) == 2
    assert [i["name"] for i in listed] == ["A", "B"]


def test_get_item(client):
    _add(client)
    assert client.get("/items/1").get_json()["name"] == "Widget"


def test_add_stock(client):
    _add(client)
    data = client.patch("/items/1/stock", json={"delta": 5}).get_json()
    assert data["stock"] == 15


def test_remove_stock(client):
    _add(client)
    data = client.patch("/items/1/stock", json={"delta": -4}).get_json()
    assert data["stock"] == 6


def test_remove_to_zero_ok(client):
    _add(client, stock=2)
    data = client.patch("/items/1/stock", json={"delta": -2}).get_json()
    assert data["stock"] == 0
    assert data["low_stock"] is True


def test_remove_more_than_stock_400(client):
    _add(client, stock=2)
    resp = client.patch("/items/1/stock", json={"delta": -3})
    assert resp.status_code == 400
    assert "only 2 in stock" in resp.get_json()["error"]
    assert client.get("/items/1").get_json()["stock"] == 2


def test_negative_stock_400(client):
    assert _add(client, stock=-1).status_code == 400


def test_missing_name_400(client):
    resp = client.post("/items", json={"stock": 5})
    assert resp.status_code == 400


def test_bad_delta_400(client):
    _add(client)
    assert client.patch("/items/1/stock", json={"delta": "five"}).status_code == 400
    assert client.patch("/items/1/stock", json={}).status_code == 400


def test_low_stock_listing(client):
    _add(client, "Plenty", stock=20, threshold=5)
    _add(client, "Low", stock=4, threshold=5)
    _add(client, "At edge", stock=5, threshold=5)
    low = client.get("/items/low-stock").get_json()
    assert [i["name"] for i in low] == ["Low", "At edge"]


def test_low_stock_threshold_override(client):
    _add(client, "A", stock=8, threshold=5)
    _add(client, "B", stock=12, threshold=5)
    low = client.get("/items/low-stock?threshold=10").get_json()
    assert [i["name"] for i in low] == ["A"]


def test_low_stock_bad_threshold_400(client):
    assert client.get("/items/low-stock?threshold=abc").status_code == 400


def test_unknown_item_404(client):
    assert client.get("/items/99").status_code == 404
    assert client.patch("/items/99/stock", json={"delta": 1}).status_code == 404


def test_delete_item(client):
    _add(client)
    assert client.delete("/items/1").status_code == 200
    assert client.get("/items/1").status_code == 404
