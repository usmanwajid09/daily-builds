"""Tests for the food_delivery_api Flask app, using Flask's test client."""
import pytest

from food_delivery_api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _restaurant(client, name="Biryani House", **extra):
    payload = {"name": name}
    payload.update(extra)
    return client.post("/restaurants", json=payload)


def _menu_item(client, rid, name="Biryani", price=350, **extra):
    payload = {"name": name, "price": price}
    payload.update(extra)
    return client.post(f"/restaurants/{rid}/menu", json=payload)


def _setup(client, fee=150):
    rid = _restaurant(client, cuisine="Pakistani",
                      delivery_fee=fee).get_json()["id"]
    biryani = _menu_item(client, rid, "Chicken Biryani", 350,
                         category="Main").get_json()
    raita = _menu_item(client, rid, "Raita", 50, category="Side").get_json()
    return rid, biryani, raita


def _order(client, rid, items, customer="Usman"):
    return client.post("/orders", json={
        "restaurant_id": rid, "customer_name": customer, "items": items})


def test_list_restaurants_starts_empty(client):
    assert client.get("/restaurants").get_json() == []


def test_create_restaurant_returns_201(client):
    resp = _restaurant(client, "Spice Hub", cuisine="Desi", delivery_fee=100)
    assert resp.status_code == 201
    assert resp.get_json() == {
        "id": 1, "name": "Spice Hub", "cuisine": "Desi", "delivery_fee": 100.0}


def test_create_restaurant_defaults(client):
    restaurant = _restaurant(client, "Plain").get_json()
    assert restaurant["cuisine"] == ""
    assert restaurant["delivery_fee"] == 0.0


def test_create_restaurant_validation_errors(client):
    assert _restaurant(client, "").status_code == 400
    assert _restaurant(client, "   ").status_code == 400
    assert _restaurant(client, 42).status_code == 400
    assert _restaurant(client, "X", cuisine=5).status_code == 400
    assert _restaurant(client, "X", delivery_fee=-1).status_code == 400
    assert _restaurant(client, "X", delivery_fee="free").status_code == 400
    assert client.post("/restaurants", json=[1, 2]).status_code == 400


def test_get_restaurant_and_404(client):
    rid = _restaurant(client, "Found").get_json()["id"]
    assert client.get(f"/restaurants/{rid}").get_json()["name"] == "Found"
    assert client.get("/restaurants/999").status_code == 404


def test_list_restaurants_cuisine_filter(client):
    _restaurant(client, "A", cuisine="Pakistani")
    _restaurant(client, "B", cuisine="Chinese")
    names = [r["name"] for r in client.get(
        "/restaurants", query_string={"cuisine": "pakistani"}).get_json()]
    assert names == ["A"]


def test_update_restaurant(client):
    rid = _restaurant(client, "Old", delivery_fee=50).get_json()["id"]
    updated = client.put(f"/restaurants/{rid}",
                         json={"name": "New", "delivery_fee": 0}).get_json()
    assert updated["name"] == "New"
    assert updated["delivery_fee"] == 0.0
    assert client.put(f"/restaurants/{rid}",
                      json={"delivery_fee": -5}).status_code == 400
    assert client.put("/restaurants/999", json={"name": "X"}).status_code == 404


def test_delete_restaurant_conflict_with_menu(client):
    rid, _, _ = _setup(client)
    assert client.delete(f"/restaurants/{rid}").status_code == 409
    assert client.get(f"/restaurants/{rid}").status_code == 200


def test_add_menu_item_and_list(client):
    rid = _restaurant(client, "Menu Test").get_json()["id"]
    resp = _menu_item(client, rid, "Naan", 40, category="Bread")
    assert resp.status_code == 201
    item = resp.get_json()
    assert item == {"id": 1, "restaurant_id": rid, "name": "Naan",
                    "price": 40.0, "category": "Bread"}
    menu = client.get(f"/restaurants/{rid}/menu").get_json()
    assert [i["name"] for i in menu] == ["Naan"]
    assert client.get(f"/restaurants/{rid}/menu",
                      query_string={"category": "Bread"}).get_json() == menu
    assert client.get(f"/restaurants/{rid}/menu",
                      query_string={"category": "Drink"}).get_json() == []


def test_menu_item_validation_errors(client):
    rid = _restaurant(client, "V").get_json()["id"]
    assert _menu_item(client, rid, "", 10).status_code == 400
    assert _menu_item(client, rid, "X", 0).status_code == 400
    assert _menu_item(client, rid, "X", -5).status_code == 400
    assert _menu_item(client, rid, "X", "ten").status_code == 400
    assert _menu_item(client, rid, "X", 10, category=3).status_code == 400
    assert _menu_item(client, 999, "X", 10).status_code == 404


def test_get_update_delete_menu_item(client):
    rid, biryani, _ = _setup(client)
    item_id = biryani["id"]
    got = client.get(f"/restaurants/{rid}/menu/{item_id}").get_json()
    assert got["name"] == "Chicken Biryani"
    updated = client.put(f"/restaurants/{rid}/menu/{item_id}",
                         json={"price": 400}).get_json()
    assert updated["price"] == 400.0
    assert client.put(f"/restaurants/{rid}/menu/{item_id}",
                      json={"price": 0}).status_code == 400
    assert client.get(f"/restaurants/{rid}/menu/999").status_code == 404
    assert client.delete(f"/restaurants/{rid}/menu/{item_id}").status_code == 200
    assert client.get(f"/restaurants/{rid}/menu/{item_id}").status_code == 404


def test_menu_item_wrong_restaurant_404(client):
    rid1, _, _ = _setup(client)
    rid2 = _restaurant(client, "Other").get_json()["id"]
    item_id = _menu_item(client, rid1, "Solo", 10).get_json()["id"]
    assert client.get(f"/restaurants/{rid2}/menu/{item_id}").status_code == 404


def test_create_order_totals(client):
    rid, biryani, raita = _setup(client, fee=150)
    resp = _order(client, rid, [
        {"menu_item_id": biryani["id"], "quantity": 2},
        {"menu_item_id": raita["id"], "quantity": 1},
    ])
    assert resp.status_code == 201
    order = resp.get_json()
    assert order["subtotal"] == 750.0
    assert order["delivery_fee"] == 150.0
    assert order["total"] == 900.0
    assert order["status"] == "placed"
    assert order["items"][0] == {
        "menu_item_id": biryani["id"], "name": "Chicken Biryani",
        "unit_price": 350.0, "quantity": 2, "line_total": 700.0}


def test_create_order_validation_errors(client):
    rid, biryani, raita = _setup(client)
    other = _restaurant(client, "Elsewhere").get_json()["id"]
    foreign = _menu_item(client, other, "Pizza", 900).get_json()
    assert _order(client, 999, []).status_code == 404
    assert _order(client, rid, []).status_code == 400
    assert _order(client, rid, "nope").status_code == 400
    assert _order(client, rid, [{"menu_item_id": biryani["id"]}]).status_code == 400
    assert _order(client, rid, [{"menu_item_id": biryani["id"],
                                  "quantity": 0}]).status_code == 400
    assert _order(client, rid, [{"menu_item_id": biryani["id"],
                                  "quantity": 1.5}]).status_code == 400
    assert _order(client, rid, [{"menu_item_id": biryani["id"],
                                  "quantity": True}]).status_code == 400
    assert _order(client, rid, [{"menu_item_id": 999, "quantity": 1}]).status_code == 404
    assert _order(client, rid, [{"menu_item_id": foreign["id"],
                                  "quantity": 1}]).status_code == 400
    assert _order(client, rid, [{"menu_item_id": biryani["id"], "quantity": 1},
                                {"menu_item_id": biryani["id"],
                                 "quantity": 1}]).status_code == 400
    assert _order(client, rid, [{"menu_item_id": raita["id"], "quantity": 1}],
                  customer=" ").status_code == 400
    assert _order(client, "1", [{"menu_item_id": raita["id"],
                                 "quantity": 1}]).status_code == 400


def test_order_status_flow(client):
    rid, biryani, _ = _setup(client)
    oid = _order(client, rid, [{"menu_item_id": biryani["id"],
                                "quantity": 1}]).get_json()["id"]
    for step in ("preparing", "out_for_delivery", "delivered"):
        moved = client.post(f"/orders/{oid}/status",
                            json={"status": step}).get_json()
        assert moved["status"] == step
    assert client.post(f"/orders/{oid}/status",
                       json={"status": "cancelled"}).status_code == 400


def test_order_status_illegal_jumps(client):
    rid, biryani, _ = _setup(client)
    oid = _order(client, rid, [{"menu_item_id": biryani["id"],
                                "quantity": 1}]).get_json()["id"]
    assert client.post(f"/orders/{oid}/status",
                       json={"status": "delivered"}).status_code == 400
    assert client.post(f"/orders/{oid}/status",
                       json={"status": "out_for_delivery"}).status_code == 400
    assert client.post(f"/orders/{oid}/status",
                       json={"status": "shipped"}).status_code == 400
    assert client.post("/orders/999/status",
                       json={"status": "preparing"}).status_code == 404


def test_cancel_then_delete_order(client):
    rid, biryani, _ = _setup(client)
    oid = _order(client, rid, [{"menu_item_id": biryani["id"],
                                "quantity": 1}]).get_json()["id"]
    assert client.delete(f"/orders/{oid}").status_code == 409
    cancelled = client.post(f"/orders/{oid}/status",
                            json={"status": "cancelled"}).get_json()
    assert cancelled["status"] == "cancelled"
    assert client.post(f"/orders/{oid}/status",
                       json={"status": "preparing"}).status_code == 400
    assert client.delete(f"/orders/{oid}").status_code == 200
    assert client.get(f"/orders/{oid}").status_code == 404


def test_list_orders_filters(client):
    rid, biryani, _ = _setup(client)
    oid1 = _order(client, rid, [{"menu_item_id": biryani["id"], "quantity": 1}],
                  customer="A").get_json()["id"]
    _order(client, rid, [{"menu_item_id": biryani["id"], "quantity": 1}],
           customer="B")
    client.post(f"/orders/{oid1}/status", json={"status": "cancelled"})
    cancelled = client.get("/orders",
                           query_string={"status": "cancelled"}).get_json()
    assert [o["id"] for o in cancelled] == [oid1]
    assert client.get("/orders", query_string={"status": "nope"}).status_code == 400
    by_restaurant = client.get(
        "/orders", query_string={"restaurant_id": rid}).get_json()
    assert len(by_restaurant) == 2
    assert client.get("/orders",
                      query_string={"restaurant_id": "abc"}).status_code == 400


def test_delete_menu_item_in_live_order_conflict(client):
    rid, biryani, _ = _setup(client)
    oid = _order(client, rid, [{"menu_item_id": biryani["id"],
                                "quantity": 1}]).get_json()["id"]
    assert client.delete(
        f"/restaurants/{rid}/menu/{biryani['id']}").status_code == 409
    client.post(f"/orders/{oid}/status", json={"status": "cancelled"})
    assert client.delete(
        f"/restaurants/{rid}/menu/{biryani['id']}").status_code == 200
