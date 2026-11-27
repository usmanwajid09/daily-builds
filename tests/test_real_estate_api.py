"""Tests for the real_estate_api Flask app, using Flask's test client."""
import pytest

from real_estate_api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _property(client, title="Sunny 2-bed flat", city="Lahore", **extra):
    payload = {
        "title": title, "city": city, "price": 8500000,
        "bedrooms": 2, "type": "apartment",
    }
    payload.update(extra)
    return client.post("/properties", json=payload)


@pytest.fixture
def prop(client):
    return _property(client).get_json()["id"]


def test_list_properties_starts_empty(client):
    assert client.get("/properties").get_json() == []


def test_create_property_returns_201(client):
    resp = _property(client, bathrooms=2, area_sqft=950)
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["id"] == 1
    assert body["bathrooms"] == 2
    assert body["area_sqft"] == 950
    assert body["favourite"] is False


def test_create_property_validation_400(client):
    assert _property(client, title="").status_code == 400
    assert _property(client, city="  ").status_code == 400
    assert _property(client, price=0).status_code == 400
    assert _property(client, price=-100).status_code == 400
    assert _property(client, price="cheap").status_code == 400
    assert _property(client, bedrooms=-1).status_code == 400
    assert _property(client, bedrooms=2.5).status_code == 400
    assert _property(client, type="castle").status_code == 400
    assert _property(client, area_sqft=0).status_code == 400
    assert client.post(
        "/properties",
        json={"title": "No price", "city": "Lahore", "bedrooms": 1},
    ).status_code == 400


def test_create_property_duplicate_title_city_409(client, prop):
    assert _property(client, title="sunny 2-bed flat", city="lahore").status_code == 409
    # Same title in a different city is fine.
    assert _property(client, city="Islamabad").status_code == 201


def test_get_property_404(client):
    assert client.get("/properties/99").status_code == 404


def test_update_property(client, prop):
    resp = client.put(
        f"/properties/{prop}", json={"price": 9000000, "bedrooms": 3}
    )
    assert resp.status_code == 200
    assert resp.get_json()["price"] == 9000000
    assert client.put(f"/properties/{prop}", json={}).status_code == 400
    assert client.put(
        f"/properties/{prop}", json={"price": 0}
    ).status_code == 400


def test_update_property_duplicate_409(client, prop):
    _property(client, title="Other flat", city="Lahore")
    assert client.put(
        f"/properties/{prop}", json={"title": "other flat"}
    ).status_code == 409


def test_delete_property(client, prop):
    assert client.delete(f"/properties/{prop}").status_code == 200
    assert client.get(f"/properties/{prop}").status_code == 404


def test_search_filters(client):
    _property(client, title="A", city="Lahore", price=8500000, bedrooms=2)
    _property(client, title="B", city="Lahore", price=25000000,
              bedrooms=4, type="house")
    _property(client, title="C", city="Islamabad", price=6200000,
              bedrooms=1, type="studio")
    assert len(client.get("/properties").get_json()) == 3
    assert len(client.get("/properties?city=lahore").get_json()) == 2
    assert len(client.get("/properties?type=house").get_json()) == 1
    assert len(client.get("/properties?min_price=9000000").get_json()) == 1
    assert len(client.get("/properties?max_price=9000000").get_json()) == 2
    assert len(client.get("/properties?min_bedrooms=3").get_json()) == 1
    combined = client.get(
        "/properties?city=Lahore&max_price=10000000"
    ).get_json()
    assert [p["title"] for p in combined] == ["A"]
    assert client.get("/properties?type=nope").status_code == 400
    assert client.get(
        "/properties?min_price=100&max_price=50"
    ).status_code == 400


def test_favourite_flow(client, prop):
    url = f"/properties/{prop}/favourite"
    resp = client.post(url)
    assert resp.status_code == 200
    assert resp.get_json()["favourite"] is True
    assert client.post(url).status_code == 409
    favourites = client.get("/favourites").get_json()
    assert [p["id"] for p in favourites] == [prop]
    assert client.delete(url).status_code == 200
    assert client.get("/favourites").get_json() == []
    assert client.delete(url).status_code == 404


def test_favourite_unknown_property_404(client):
    assert client.post("/properties/99/favourite").status_code == 404
    assert client.delete("/properties/99/favourite").status_code == 404


def test_stats(client):
    _property(client, title="A", city="Lahore", price=10000000, bedrooms=2)
    _property(client, title="B", city="Lahore", price=20000000, bedrooms=3)
    _property(client, title="C", city="Karachi", price=15000000, bedrooms=2)
    third = _property(
        client, title="D", city="Karachi", price=9000000, bedrooms=1
    ).get_json()
    client.post(f"/properties/{third['id']}/favourite")
    stats = client.get("/stats").get_json()
    assert stats["total_listings"] == 4
    assert stats["average_price"] == 13500000.0
    assert stats["average_price_by_city"] == {
        "Karachi": {"count": 2, "average_price": 12000000.0},
        "Lahore": {"count": 2, "average_price": 15000000.0},
    }
    assert stats["favourite_count"] == 1


def test_stats_empty(client):
    stats = client.get("/stats").get_json()
    assert stats["total_listings"] == 0
    assert stats["average_price"] is None
    assert stats["average_price_by_city"] == {}
    assert stats["favourite_count"] == 0
