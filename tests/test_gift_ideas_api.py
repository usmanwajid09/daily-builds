"""Tests for gift_ideas_api."""
import pytest

from gift_ideas_api.app import create_app


@pytest.fixture()
def client():
    return create_app().test_client()


def _person(client, name="Ayesha", budget=100):
    return client.post("/people", json={"name": name, "budget": budget})


def _idea(client, person_id, **overrides):
    payload = {"title": "Novel", "price": 25, "occasion": "Birthday"}
    payload.update(overrides)
    return client.post(f"/people/{person_id}/ideas", json=payload)


def test_add_person(client):
    resp = _person(client)
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["name"] == "Ayesha"
    assert body["budget"] == 100
    assert body["spent"] == 0
    assert body["remaining"] == 100


def test_add_person_validation(client):
    assert client.post("/people", json={"name": ""}).status_code == 400
    assert client.post("/people", json={"budget": 50}).status_code == 400
    assert client.post("/people", json={"name": "X", "budget": -5}).status_code == 400


def test_add_person_default_budget(client):
    resp = client.post("/people", json={"name": "Bilal"})
    assert resp.status_code == 201
    assert resp.get_json()["budget"] == 0


def test_list_and_get_person(client):
    _person(client)
    _person(client, "Bilal", 50)
    listed = client.get("/people").get_json()
    assert len(listed) == 2
    assert client.get("/people/1").get_json()["name"] == "Ayesha"
    assert client.get("/people/999").status_code == 404


def test_add_idea(client):
    person = _person(client).get_json()
    resp = _idea(client, person["id"])
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["title"] == "Novel"
    assert body["price"] == 25
    assert body["status"] == "idea"
    assert body["occasion"] == "Birthday"


def test_add_idea_validation(client):
    person = _person(client).get_json()
    assert _idea(client, person["id"], title="").status_code == 400
    assert _idea(client, person["id"], price=-1).status_code == 400
    assert _idea(client, person["id"], price="cheap").status_code == 400
    assert _idea(client, person["id"], status="ordered").status_code == 400
    assert client.post("/people/999/ideas", json={"title": "X", "price": 1}).status_code == 404


def test_list_ideas(client):
    person = _person(client).get_json()
    _idea(client, person["id"])
    _idea(client, person["id"], title="Headphones", price=60)
    ideas = client.get(f"/people/{person['id']}/ideas").get_json()
    assert len(ideas) == 2
    assert client.get("/people/999/ideas").status_code == 404


def test_budget_math_only_bought_counts(client):
    person = _person(client, budget=100).get_json()
    pid = person["id"]
    i1 = _idea(client, pid, price=25).get_json()
    _idea(client, pid, title="Headphones", price=60)
    budget = client.get(f"/people/{pid}/budget").get_json()
    assert budget["spent"] == 0
    assert budget["remaining"] == 100
    client.put(f"/ideas/{i1['id']}", json={"status": "bought"})
    budget = client.get(f"/people/{pid}/budget").get_json()
    assert budget["spent"] == 25
    assert budget["remaining"] == 75
    assert budget["bought_count"] == 1
    assert budget["idea_count"] == 2


def test_budget_can_go_negative(client):
    person = _person(client, budget=10).get_json()
    idea = _idea(client, person["id"], price=25).get_json()
    client.put(f"/ideas/{idea['id']}", json={"status": "bought"})
    budget = client.get(f"/people/{person['id']}/budget").get_json()
    assert budget["remaining"] == -15


def test_update_idea(client):
    person = _person(client).get_json()
    idea = _idea(client, person["id"]).get_json()
    resp = client.put(
        f"/ideas/{idea['id']}",
        json={"title": "Hardcover novel", "price": 30, "occasion": "Eid"},
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["title"] == "Hardcover novel"
    assert body["price"] == 30
    assert body["occasion"] == "Eid"


def test_update_idea_validation(client):
    person = _person(client).get_json()
    idea = _idea(client, person["id"]).get_json()
    assert client.put(f"/ideas/{idea['id']}", json={"status": "ordered"}).status_code == 400
    assert client.put(f"/ideas/{idea['id']}", json={"price": -2}).status_code == 400
    assert client.put("/ideas/999", json={"status": "bought"}).status_code == 404


def test_delete_idea(client):
    person = _person(client).get_json()
    idea = _idea(client, person["id"]).get_json()
    assert client.delete(f"/ideas/{idea['id']}").status_code == 200
    assert client.get(f"/people/{person['id']}/ideas").get_json() == []
    assert client.delete("/ideas/999").status_code == 404


def test_budget_endpoint_404(client):
    assert client.get("/people/999/budget").status_code == 404
