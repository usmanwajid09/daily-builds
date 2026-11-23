"""Tests for book_club_api."""
import pytest

from book_club_api.app import create_app


@pytest.fixture()
def client():
    return create_app().test_client()


def _book(client, title="Dune", author="Frank Herbert"):
    return client.post("/books", json={"title": title, "author": author})


def _member(client, name="Ayesha"):
    return client.post("/members", json={"name": name})


def test_add_book(client):
    resp = _book(client)
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["title"] == "Dune"
    assert body["author"] == "Frank Herbert"
    assert body["votes"] == 0


def test_add_book_validation(client):
    assert client.post("/books", json={"title": "", "author": "X"}).status_code == 400
    assert client.post("/books", json={"title": "X"}).status_code == 400
    assert client.post("/books", json={"title": "X", "author": ""}).status_code == 400


def test_list_and_get_book(client):
    _book(client)
    _book(client, "Neuromancer", "William Gibson")
    assert len(client.get("/books").get_json()) == 2
    assert client.get("/books/1").get_json()["title"] == "Dune"
    assert client.get("/books/999").status_code == 404


def test_add_member(client):
    resp = _member(client)
    assert resp.status_code == 201
    assert resp.get_json()["name"] == "Ayesha"
    assert client.post("/members", json={"name": "  "}).status_code == 400
    assert len(client.get("/members").get_json()) == 1


def test_vote_counts(client):
    _book(client)
    m1 = _member(client, "Ayesha").get_json()
    m2 = _member(client, "Bilal").get_json()
    assert client.post("/books/1/vote", json={"member_id": m1["id"]}).status_code == 200
    client.post("/books/1/vote", json={"member_id": m2["id"]})
    assert client.get("/books/1").get_json()["votes"] == 2


def test_vote_one_per_member_changeable(client):
    _book(client)
    _book(client, "Neuromancer", "William Gibson")
    m1 = _member(client).get_json()
    client.post("/books/1/vote", json={"member_id": m1["id"]})
    resp = client.post("/books/2/vote", json={"member_id": m1["id"]})
    assert resp.get_json()["previous_book_id"] == 1
    assert client.get("/books/1").get_json()["votes"] == 0
    assert client.get("/books/2").get_json()["votes"] == 1


def test_vote_validation(client):
    _book(client)
    m1 = _member(client).get_json()
    assert client.post("/books/999/vote", json={"member_id": m1["id"]}).status_code == 404
    assert client.post("/books/1/vote", json={"member_id": 999}).status_code == 404
    assert client.post("/books/1/vote", json={}).status_code == 400
    assert client.post("/books/1/vote", json={"member_id": "one"}).status_code == 400


def test_leaderboard_order(client):
    _book(client)
    _book(client, "Neuromancer", "William Gibson")
    _book(client, "Foundation", "Isaac Asimov")
    m1 = _member(client, "Ayesha").get_json()
    m2 = _member(client, "Bilal").get_json()
    m3 = _member(client, "Sara").get_json()
    client.post("/books/3/vote", json={"member_id": m1["id"]})
    client.post("/books/3/vote", json={"member_id": m2["id"]})
    client.post("/books/1/vote", json={"member_id": m3["id"]})
    ranked = client.get("/votes").get_json()
    assert [b["id"] for b in ranked] == [3, 1, 2]
    assert ranked[0]["votes"] == 2


def test_schedule_meeting(client):
    _book(client)
    resp = client.post(
        "/meetings",
        json={"book_id": 1, "date": "2026-12-05", "location": "Library"},
    )
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["book_id"] == 1
    assert body["attending_count"] == 0


def test_schedule_meeting_validation(client):
    _book(client)
    assert client.post("/meetings", json={"book_id": 999, "date": "2026-12-05"}).status_code == 404
    assert client.post("/meetings", json={"book_id": 1, "date": "Dec 5"}).status_code == 400
    assert client.post("/meetings", json={"date": "2026-12-05"}).status_code == 400


def test_rsvp(client):
    _book(client)
    m1 = _member(client, "Ayesha").get_json()
    m2 = _member(client, "Bilal").get_json()
    client.post("/meetings", json={"book_id": 1, "date": "2026-12-05"})
    client.post("/meetings/1/rsvp", json={"member_id": m1["id"], "attending": True})
    resp = client.post("/meetings/1/rsvp", json={"member_id": m2["id"], "attending": False})
    assert resp.get_json()["attending_count"] == 1
    # Changing an RSVP updates the count.
    resp = client.post("/meetings/1/rsvp", json={"member_id": m2["id"], "attending": True})
    assert resp.get_json()["attending_count"] == 2


def test_rsvp_validation(client):
    _book(client)
    _member(client)
    client.post("/meetings", json={"book_id": 1, "date": "2026-12-05"})
    assert client.post("/meetings/999/rsvp", json={"member_id": 1, "attending": True}).status_code == 404
    assert client.post("/meetings/1/rsvp", json={"member_id": 999, "attending": True}).status_code == 404
    assert client.post("/meetings/1/rsvp", json={"member_id": 1}).status_code == 400
    assert client.post("/meetings/1/rsvp", json={"member_id": 1, "attending": "yes"}).status_code == 400
