"""Tests for the recipe_api Flask app, using Flask's test client."""
import pytest

from recipe_api.app import create_app

RECIPE = {
    "title": "Garlic Butter Pasta",
    "ingredients": ["pasta", "garlic", "butter"],
    "steps": ["Boil the pasta.", "Toss with garlic butter."],
    "servings": 2,
}


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_list_starts_empty(client):
    assert client.get("/recipes").get_json() == []


def test_create_recipe_returns_201_with_id(client):
    resp = client.post("/recipes", json=RECIPE)
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["id"] == 1
    assert body["title"] == "Garlic Butter Pasta"
    assert body["servings"] == 2


def test_create_trims_strings(client):
    payload = dict(RECIPE, title="  Soup  ", ingredients=["  tomato "])
    body = client.post("/recipes", json=payload).get_json()
    assert body["title"] == "Soup"
    assert body["ingredients"] == ["tomato"]


def test_create_rejects_missing_title(client):
    payload = dict(RECIPE)
    del payload["title"]
    resp = client.post("/recipes", json=payload)
    assert resp.status_code == 400
    assert "title" in resp.get_json()["error"]


def test_create_rejects_empty_ingredients_or_steps(client):
    assert client.post("/recipes", json=dict(RECIPE, ingredients=[])).status_code == 400
    assert client.post("/recipes", json=dict(RECIPE, steps=["  "])).status_code == 400
    assert client.post("/recipes", json=dict(RECIPE, ingredients="garlic")).status_code == 400


def test_create_rejects_bad_servings(client):
    for bad in (0, -1, 2.5, "two", True):
        assert client.post("/recipes", json=dict(RECIPE, servings=bad)).status_code == 400


def test_create_rejects_non_json_body(client):
    resp = client.post("/recipes", data="not json", content_type="text/plain")
    assert resp.status_code == 400


def test_search_by_ingredient_is_case_insensitive(client):
    client.post("/recipes", json=RECIPE)
    client.post(
        "/recipes",
        json=dict(RECIPE, title="Tomato Salad", ingredients=["tomato", "onion"]),
    )
    found = client.get("/recipes?ingredient=GARLIC").get_json()
    assert [r["title"] for r in found] == ["Garlic Butter Pasta"]
    assert client.get("/recipes?ingredient=chocolate").get_json() == []


def test_get_update_and_delete_round_trip(client):
    created = client.post("/recipes", json=RECIPE).get_json()
    recipe_id = created["id"]
    assert client.get(f"/recipes/{recipe_id}").get_json()["title"] == "Garlic Butter Pasta"
    updated = client.put(f"/recipes/{recipe_id}", json={"servings": 4}).get_json()
    assert updated["servings"] == 4
    assert updated["title"] == "Garlic Butter Pasta"
    assert client.delete(f"/recipes/{recipe_id}").get_json() == {"deleted": recipe_id}
    assert client.get(f"/recipes/{recipe_id}").status_code == 404


def test_update_rejects_invalid_values(client):
    recipe_id = client.post("/recipes", json=RECIPE).get_json()["id"]
    assert client.put(f"/recipes/{recipe_id}", json={"servings": 0}).status_code == 400
    assert client.put(f"/recipes/{recipe_id}", json={}).status_code == 400


def test_unknown_recipe_is_404_json(client):
    for method in ("get", "put", "delete"):
        resp = getattr(client, method)("/recipes/999", json={})
        assert resp.status_code == 404
        assert "not found" in resp.get_json()["error"]


def test_unknown_route_is_404_json(client):
    resp = client.get("/nope")
    assert resp.status_code == 404
    assert "error" in resp.get_json()
