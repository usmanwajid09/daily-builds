"""Tests for the meal_planner_api Flask app, using Flask's test client."""
import pytest

from meal_planner_api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _add(client, name="Overnight oats", day="mon", meal_type="breakfast",
         calories=350, ingredients=None):
    payload = {"name": name, "day": day, "meal_type": meal_type,
               "calories": calories}
    if ingredients is not None:
        payload["ingredients"] = ingredients
    return client.post("/meals", json=payload)


OATS = [{"name": "Oats", "quantity": 50, "unit": "g"},
        {"name": "Milk", "quantity": 200, "unit": "ml"}]


def test_list_starts_empty(client):
    assert client.get("/meals").get_json() == []


def test_create_meal_returns_201(client):
    resp = _add(client, ingredients=OATS)
    assert resp.status_code == 201
    assert resp.get_json() == {
        "id": 1,
        "name": "Overnight oats",
        "day": "mon",
        "meal_type": "breakfast",
        "calories": 350,
        "ingredients": OATS,
    }


def test_create_defaults_to_no_ingredients(client):
    assert _add(client).get_json()["ingredients"] == []


def test_create_bad_day_400(client):
    assert _add(client, day="funday").status_code == 400


def test_create_bad_meal_type_400(client):
    assert _add(client, meal_type="brunch").status_code == 400


def test_create_negative_calories_400(client):
    assert _add(client, calories=-10).status_code == 400


def test_create_fractional_calories_400(client):
    assert _add(client, calories=350.5).status_code == 400


def test_create_missing_name_400(client):
    resp = client.post("/meals", json={"day": "mon", "meal_type": "lunch",
                                       "calories": 100})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_create_bad_ingredient_400(client):
    bad = [{"name": "Oats", "quantity": 0, "unit": "g"}]
    assert _add(client, ingredients=bad).status_code == 400
    bad = [{"name": "Oats", "quantity": 50}]
    assert _add(client, ingredients=bad).status_code == 400
    assert _add(client, ingredients="oats").status_code == 400


def test_get_meal_and_404(client):
    _add(client)
    assert client.get("/meals/1").get_json()["name"] == "Overnight oats"
    resp = client.get("/meals/99")
    assert resp.status_code == 404
    assert "error" in resp.get_json()


def test_filter_by_day_and_meal_type(client):
    _add(client)
    _add(client, name="Chicken curry", day="mon", meal_type="dinner",
         calories=650)
    _add(client, name="Salad", day="tue", meal_type="lunch", calories=420)
    assert len(client.get("/meals", query_string={"day": "mon"}).get_json()) == 2
    lunch = client.get("/meals", query_string={"meal_type": "lunch"}).get_json()
    assert [m["name"] for m in lunch] == ["Salad"]
    assert client.get("/meals", query_string={"day": "nope"}).status_code == 400


def test_update_meal(client):
    _add(client)
    resp = client.put("/meals/1", json={"calories": 400, "day": "tue"})
    assert resp.status_code == 200
    assert resp.get_json()["calories"] == 400
    assert resp.get_json()["day"] == "tue"


def test_update_bad_day_400(client):
    _add(client)
    assert client.put("/meals/1", json={"day": "x"}).status_code == 400


def test_delete_meal(client):
    _add(client)
    assert client.delete("/meals/1").get_json() == {"deleted": 1}
    assert client.get("/meals").get_json() == []
    assert client.delete("/meals/1").status_code == 404


def test_day_plan_totals_and_order(client):
    _add(client, name="Curry", day="mon", meal_type="dinner", calories=650)
    _add(client, name="Oats", day="mon", meal_type="breakfast", calories=350)
    plan = client.get("/plan/mon").get_json()
    assert plan["meal_count"] == 2
    assert plan["total_calories"] == 1000
    assert [m["meal_type"] for m in plan["meals"]] == ["breakfast", "dinner"]


def test_day_plan_empty_day(client):
    plan = client.get("/plan/wed").get_json()
    assert plan == {"day": "wed", "meals": [], "meal_count": 0,
                    "total_calories": 0}


def test_day_plan_bad_day_400(client):
    assert client.get("/plan/funday").status_code == 400


def test_week_plan(client):
    _add(client, calories=350)
    _add(client, name="Salad", day="tue", meal_type="lunch", calories=420)
    week = client.get("/plan/week").get_json()
    assert week["week_total_calories"] == 770
    assert week["days"]["mon"]["total_calories"] == 350
    assert week["days"]["tue"]["total_calories"] == 420
    assert week["days"]["wed"]["meal_count"] == 0
    assert set(week["days"]) == {"mon", "tue", "wed", "thu", "fri", "sat", "sun"}


def test_calories_endpoint(client):
    _add(client, calories=350)
    _add(client, name="Salad", day="tue", meal_type="lunch", calories=420)
    total = client.get("/calories").get_json()
    assert total == {"day": None, "meal_count": 2, "total_calories": 770}
    mon = client.get("/calories", query_string={"day": "mon"}).get_json()
    assert mon["total_calories"] == 350
    assert client.get("/calories", query_string={"day": "x"}).status_code == 400


def test_shopping_list_aggregates(client):
    _add(client, ingredients=OATS)
    _add(client, name="Porridge", day="tue", meal_type="breakfast",
         calories=300,
         ingredients=[{"name": "oats", "quantity": 30, "unit": "g"},
                      {"name": "Honey", "quantity": 10, "unit": "ml"}])
    result = client.get("/shopping-list").get_json()
    assert result["item_count"] == 3
    by_name = {i["name"].lower(): i for i in result["items"]}
    assert by_name["oats"]["quantity"] == 80
    assert by_name["oats"]["meals"] == 2
    assert by_name["milk"]["quantity"] == 200
    assert by_name["honey"]["quantity"] == 10


def test_shopping_list_empty(client):
    assert client.get("/shopping-list").get_json() == {"items": [], "item_count": 0}


def test_shopping_list_same_name_different_unit_not_merged(client):
    _add(client, ingredients=[{"name": "Sugar", "quantity": 10, "unit": "g"}])
    _add(client, name="Tea", day="tue", meal_type="breakfast", calories=50,
         ingredients=[{"name": "Sugar", "quantity": 1, "unit": "tsp"}])
    result = client.get("/shopping-list").get_json()
    assert result["item_count"] == 2
