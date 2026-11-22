"""Tests for the pet_adoption_api Flask app, using Flask's test client."""
import pytest

from pet_adoption_api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture
def pet(client):
    return client.post(
        "/pets",
        json={"name": "Buddy", "species": "dog", "breed": "Labrador", "age": 3},
    ).get_json()


def _apply(client, pet_id, email="usman@example.com"):
    return client.post(
        f"/pets/{pet_id}/applications",
        json={"applicant_name": "Usman", "applicant_email": email},
    )


def test_create_pet_returns_201_with_available_status(client):
    resp = client.post(
        "/pets", json={"name": "Milo", "species": "cat", "breed": "Tabby", "age": 1}
    )
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["name"] == "Milo"
    assert body["species"] == "cat"
    assert body["status"] == "available"
    assert body["id"] == 1


def test_create_pet_rejects_missing_name(client):
    resp = client.post("/pets", json={"species": "dog"})
    assert resp.status_code == 400
    assert "name" in resp.get_json()["error"]


def test_create_pet_rejects_bad_species(client):
    resp = client.post("/pets", json={"name": "Rex", "species": "dragon"})
    assert resp.status_code == 400
    assert "species" in resp.get_json()["error"]


def test_create_pet_rejects_negative_age(client):
    resp = client.post("/pets", json={"name": "Rex", "species": "dog", "age": -1})
    assert resp.status_code == 400


def test_create_pet_rejects_non_json_body(client):
    resp = client.post("/pets", data="not json", content_type="text/plain")
    assert resp.status_code == 400


def test_get_pet_and_404(client, pet):
    assert client.get(f"/pets/{pet['id']}").get_json()["name"] == "Buddy"
    assert client.get("/pets/999").status_code == 404


def test_list_pets_filters_by_species_and_status(client, pet):
    client.post("/pets", json={"name": "Whiskers", "species": "cat"})
    dogs = client.get("/pets", query_string={"species": "dog"}).get_json()
    assert [p["name"] for p in dogs] == ["Buddy"]
    cats = client.get("/pets", query_string={"species": "cat"}).get_json()
    assert [p["name"] for p in cats] == ["Whiskers"]
    available = client.get("/pets", query_string={"status": "available"}).get_json()
    assert len(available) == 2


def test_list_pets_rejects_bad_species_filter(client):
    assert client.get("/pets", query_string={"species": "dragon"}).status_code == 400


def test_list_pets_filters_by_breed_case_insensitive(client, pet):
    result = client.get("/pets", query_string={"breed": "labrador"}).get_json()
    assert len(result) == 1
    assert client.get("/pets", query_string={"breed": "Poodle"}).get_json() == []


def test_update_pet(client, pet):
    resp = client.put(f"/pets/{pet['id']}", json={"age": 4, "name": "Buddy Jr"})
    assert resp.status_code == 200
    assert resp.get_json()["age"] == 4
    assert resp.get_json()["name"] == "Buddy Jr"


def test_update_pet_rejects_bad_species(client, pet):
    assert client.put(
        f"/pets/{pet['id']}", json={"species": "dragon"}
    ).status_code == 400


def test_delete_available_pet(client, pet):
    assert client.delete(f"/pets/{pet['id']}").status_code == 200
    assert client.get(f"/pets/{pet['id']}").status_code == 404


def test_delete_pending_pet_returns_409(client, pet):
    _apply(client, pet["id"])
    assert client.delete(f"/pets/{pet['id']}").status_code == 409


def test_apply_moves_pet_to_pending(client, pet):
    resp = _apply(client, pet["id"])
    assert resp.status_code == 201
    assert resp.get_json()["status"] == "pending"
    assert client.get(f"/pets/{pet['id']}").get_json()["status"] == "pending"


def test_apply_requires_valid_email(client, pet):
    resp = client.post(
        f"/pets/{pet['id']}/applications",
        json={"applicant_name": "Usman", "applicant_email": "not-an-email"},
    )
    assert resp.status_code == 400


def test_apply_to_missing_pet_404(client):
    assert _apply(client, 999).status_code == 404


def test_duplicate_pending_application_returns_409(client, pet):
    assert _apply(client, pet["id"]).status_code == 201
    assert _apply(client, pet["id"]).status_code == 409


def test_approve_adopts_pet_and_rejects_others(client, pet):
    first = _apply(client, pet["id"], "a@example.com").get_json()
    second = _apply(client, pet["id"], "b@example.com").get_json()
    resp = client.post(f"/applications/{first['id']}/approve")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "approved"
    assert client.get(f"/pets/{pet['id']}").get_json()["status"] == "adopted"
    assert client.get(f"/applications/{second['id']}").get_json()["status"] == "rejected"


def test_approve_twice_returns_409(client, pet):
    application = _apply(client, pet["id"]).get_json()
    assert client.post(f"/applications/{application['id']}/approve").status_code == 200
    assert client.post(f"/applications/{application['id']}/approve").status_code == 409


def test_apply_to_adopted_pet_returns_409(client, pet):
    application = _apply(client, pet["id"]).get_json()
    client.post(f"/applications/{application['id']}/approve")
    assert _apply(client, pet["id"], "other@example.com").status_code == 409


def test_reject_returns_pet_to_available(client, pet):
    application = _apply(client, pet["id"]).get_json()
    resp = client.post(f"/applications/{application['id']}/reject")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "rejected"
    assert client.get(f"/pets/{pet['id']}").get_json()["status"] == "available"


def test_reject_keeps_pet_pending_when_another_application_exists(client, pet):
    first = _apply(client, pet["id"], "a@example.com").get_json()
    _apply(client, pet["id"], "b@example.com")
    client.post(f"/applications/{first['id']}/reject")
    assert client.get(f"/pets/{pet['id']}").get_json()["status"] == "pending"


def test_list_applications_filters_by_status(client, pet):
    first = _apply(client, pet["id"], "a@example.com").get_json()
    _apply(client, pet["id"], "b@example.com")
    client.post(f"/applications/{first['id']}/reject")
    rejected = client.get("/applications", query_string={"status": "rejected"}).get_json()
    assert [a["id"] for a in rejected] == [first["id"]]
    assert client.get("/applications/999").status_code == 404
