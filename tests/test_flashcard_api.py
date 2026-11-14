"""Tests for the flashcard_api Flask app, using Flask's test client."""
import pytest

from flashcard_api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _deck(client, name="Python"):
    return client.post("/decks", json={"name": name})


def _card(client, deck_id, front="Q", back="A"):
    return client.post(f"/decks/{deck_id}/cards", json={"front": front, "back": back})


def test_list_decks_starts_empty(client):
    assert client.get("/decks").get_json() == []


def test_create_deck_returns_201(client):
    resp = _deck(client, "Spanish")
    assert resp.status_code == 201
    assert resp.get_json() == {"id": 1, "name": "Spanish", "card_count": 0}


def test_create_deck_requires_non_empty_name(client):
    assert _deck(client, "").status_code == 400
    assert _deck(client, "   ").status_code == 400
    assert client.post("/decks", json={}).status_code == 400
    assert client.post("/decks", json={"name": 3}).status_code == 400


def test_get_deck_and_404(client):
    deck_id = _deck(client).get_json()["id"]
    assert client.get(f"/decks/{deck_id}").get_json()["name"] == "Python"
    assert client.get("/decks/999").status_code == 404


def test_add_card_returns_201_with_zero_counts(client):
    deck_id = _deck(client).get_json()["id"]
    resp = _card(client, deck_id, "What is 2+2?", "4")
    assert resp.status_code == 201
    assert resp.get_json() == {
        "id": 1,
        "deck_id": deck_id,
        "front": "What is 2+2?",
        "back": "4",
        "known_count": 0,
        "unknown_count": 0,
        "last_reviewed": None,
    }
    assert client.get(f"/decks/{deck_id}").get_json()["card_count"] == 1


def test_add_card_validates_input(client):
    deck_id = _deck(client).get_json()["id"]
    assert _card(client, deck_id, "", "A").status_code == 400
    assert _card(client, deck_id, "Q", "").status_code == 400
    assert _card(client, deck_id, "Q", 5).status_code == 400
    assert client.post("/decks/999/cards", json={"front": "Q", "back": "A"}).status_code == 404


def test_list_cards(client):
    deck_id = _deck(client).get_json()["id"]
    _card(client, deck_id, "Q1", "A1")
    _card(client, deck_id, "Q2", "A2")
    cards = client.get(f"/decks/{deck_id}/cards").get_json()
    assert [c["front"] for c in cards] == ["Q1", "Q2"]
    assert client.get("/decks/999/cards").status_code == 404


def test_study_picks_unreviewed_first(client):
    deck_id = _deck(client).get_json()["id"]
    first = _card(client, deck_id, "Q1", "A1").get_json()
    _card(client, deck_id, "Q2", "A2")
    nxt = client.get(f"/decks/{deck_id}/study").get_json()
    assert nxt["id"] == first["id"]
    client.post(f"/decks/{deck_id}/cards/{first['id']}/review", json={"known": True})
    nxt = client.get(f"/decks/{deck_id}/study").get_json()
    assert nxt["front"] == "Q2"


def test_study_prefers_fewest_known_reviews(client):
    deck_id = _deck(client).get_json()["id"]
    c1 = _card(client, deck_id, "Q1", "A1").get_json()
    c2 = _card(client, deck_id, "Q2", "A2").get_json()
    client.post(f"/decks/{deck_id}/cards/{c1['id']}/review", json={"known": True})
    client.post(f"/decks/{deck_id}/cards/{c1['id']}/review", json={"known": True})
    client.post(f"/decks/{deck_id}/cards/{c2['id']}/review", json={"known": True})
    nxt = client.get(f"/decks/{deck_id}/study").get_json()
    assert nxt["id"] == c2["id"]


def test_review_updates_counts_and_timestamp(client):
    deck_id = _deck(client).get_json()["id"]
    card_id = _card(client, deck_id).get_json()["id"]
    reviewed = client.post(
        f"/decks/{deck_id}/cards/{card_id}/review", json={"known": False}
    ).get_json()
    assert reviewed["unknown_count"] == 1
    assert reviewed["known_count"] == 0
    assert reviewed["last_reviewed"] is not None
    reviewed = client.post(
        f"/decks/{deck_id}/cards/{card_id}/review", json={"known": True}
    ).get_json()
    assert reviewed["known_count"] == 1 and reviewed["unknown_count"] == 1


def test_review_validates_known_flag(client):
    deck_id = _deck(client).get_json()["id"]
    card_id = _card(client, deck_id).get_json()["id"]
    assert client.post(
        f"/decks/{deck_id}/cards/{card_id}/review", json={}
    ).status_code == 400
    assert client.post(
        f"/decks/{deck_id}/cards/{card_id}/review", json={"known": "yes"}
    ).status_code == 400
    assert client.post(
        f"/decks/{deck_id}/cards/999/review", json={"known": True}
    ).status_code == 404
    assert client.post(
        f"/decks/999/cards/{card_id}/review", json={"known": True}
    ).status_code == 404


def test_study_empty_deck_is_404(client):
    deck_id = _deck(client).get_json()["id"]
    resp = client.get(f"/decks/{deck_id}/study")
    assert resp.status_code == 404
    assert "no cards" in resp.get_json()["error"]


def test_delete_card_and_deck(client):
    deck_id = _deck(client).get_json()["id"]
    card_id = _card(client, deck_id).get_json()["id"]
    assert client.delete(f"/decks/{deck_id}/cards/{card_id}").get_json() == {
        "deleted": card_id
    }
    assert client.delete(f"/decks/{deck_id}/cards/{card_id}").status_code == 404
    assert client.delete(f"/decks/{deck_id}").get_json() == {"deleted": deck_id}
    assert client.get(f"/decks/{deck_id}").status_code == 404
    assert client.delete("/decks/999").status_code == 404


def test_unknown_route_is_404(client):
    assert client.get("/nope").status_code == 404
