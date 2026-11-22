"""Tests for the movie_booking_api Flask app, using Flask's test client."""
import pytest

from movie_booking_api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture
def movie(client):
    return client.post(
        "/movies",
        json={"title": "GTA VI: The Movie", "duration_minutes": 150, "genre": "Action"},
    ).get_json()


@pytest.fixture
def screening(client, movie):
    return client.post(
        "/screenings",
        json={
            "movie_id": movie["id"],
            "screen": "Screen 1",
            "start_time": "2026-11-22T19:00",
            "rows": 3,
            "seats_per_row": 5,
            "ticket_price": 1200,
        },
    ).get_json()


def _book(client, screening_id, seats, name="Usman"):
    return client.post(
        f"/screenings/{screening_id}/book",
        json={"customer_name": name, "seats": seats},
    )


def test_create_movie_returns_201(client):
    resp = client.post(
        "/movies", json={"title": "Dune", "duration_minutes": 155}
    )
    assert resp.status_code == 201
    assert resp.get_json()["title"] == "Dune"
    assert resp.get_json()["id"] == 1


def test_create_movie_rejects_bad_duration(client):
    assert client.post(
        "/movies", json={"title": "Dune", "duration_minutes": 0}
    ).status_code == 400
    assert client.post(
        "/movies", json={"title": "Dune", "duration_minutes": "long"}
    ).status_code == 400


def test_create_movie_rejects_missing_title(client):
    assert client.post("/movies", json={"duration_minutes": 90}).status_code == 400


def test_get_movie_and_404(client, movie):
    assert client.get(f"/movies/{movie['id']}").get_json()["genre"] == "Action"
    assert client.get("/movies/999").status_code == 404


def test_delete_movie_blocked_with_screenings(client, movie, screening):
    assert client.delete(f"/movies/{movie['id']}").status_code == 409


def test_delete_movie_without_screenings(client, movie):
    assert client.delete(f"/movies/{movie['id']}").status_code == 200
    assert client.get(f"/movies/{movie['id']}").status_code == 404


def test_create_screening_builds_seat_map(client, movie):
    resp = client.post(
        "/screenings",
        json={
            "movie_id": movie["id"],
            "screen": "Screen 2",
            "start_time": "2026-11-22T21:00",
            "rows": 2,
            "seats_per_row": 3,
            "ticket_price": 800,
        },
    )
    assert resp.status_code == 201
    body = resp.get_json()
    assert sorted(body["seats"]) == ["A1", "A2", "A3", "B1", "B2", "B3"]
    assert all(state == "available" for state in body["seats"].values())
    assert body["available_seats"] == 6
    assert body["booked_seats"] == 0


def test_create_screening_rejects_unknown_movie(client):
    resp = client.post(
        "/screenings",
        json={
            "movie_id": 999,
            "screen": "Screen 1",
            "start_time": "2026-11-22T19:00",
            "rows": 2,
            "seats_per_row": 3,
            "ticket_price": 800,
        },
    )
    assert resp.status_code == 404


def test_create_screening_rejects_bad_rows_and_price(client, movie):
    base = {
        "movie_id": movie["id"],
        "screen": "Screen 1",
        "start_time": "2026-11-22T19:00",
        "rows": 3,
        "seats_per_row": 5,
        "ticket_price": 1200,
    }
    assert client.post("/screenings", json=dict(base, rows=0)).status_code == 400
    assert client.post("/screenings", json=dict(base, rows=27)).status_code == 400
    assert client.post("/screenings", json=dict(base, seats_per_row=0)).status_code == 400
    assert client.post("/screenings", json=dict(base, ticket_price=0)).status_code == 400


def test_list_screenings_filters_by_movie(client, movie, screening):
    other = client.post(
        "/movies", json={"title": "Other", "duration_minutes": 90}
    ).get_json()
    client.post(
        "/screenings",
        json={
            "movie_id": other["id"],
            "screen": "Screen 2",
            "start_time": "2026-11-22T21:00",
            "rows": 1,
            "seats_per_row": 2,
            "ticket_price": 500,
        },
    )
    result = client.get(
        "/screenings", query_string={"movie_id": movie["id"]}
    ).get_json()
    assert [s["id"] for s in result] == [screening["id"]]


def test_book_seats_returns_201_with_total_price(client, screening):
    resp = _book(client, screening["id"], ["A1", "A2", "A3"])
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["seats"] == ["A1", "A2", "A3"]
    assert body["total_price"] == 3600.0
    assert body["status"] == "active"


def test_book_updates_seat_map(client, screening):
    _book(client, screening["id"], ["A1", "B2"])
    state = client.get(f"/screenings/{screening['id']}").get_json()
    assert state["seats"]["A1"] == "booked"
    assert state["seats"]["B2"] == "booked"
    assert state["seats"]["A2"] == "available"
    assert state["available_seats"] == 13
    assert state["booked_seats"] == 2


def test_book_taken_seat_returns_409_and_books_nothing(client, screening):
    _book(client, screening["id"], ["A1"])
    resp = _book(client, screening["id"], ["A1", "A2"], name="Hamza")
    assert resp.status_code == 409
    state = client.get(f"/screenings/{screening['id']}").get_json()
    assert state["seats"]["A2"] == "available"


def test_book_nonexistent_seat_returns_400(client, screening):
    assert _book(client, screening["id"], ["Z9"]).status_code == 400
    assert _book(client, screening["id"], ["A6"]).status_code == 400


def test_book_rejects_bad_seat_labels(client, screening):
    assert _book(client, screening["id"], ["a1"]).status_code == 400
    assert _book(client, screening["id"], ["A"]).status_code == 400
    assert _book(client, screening["id"], ["A1", "A1"]).status_code == 400
    assert _book(client, screening["id"], []).status_code == 400


def test_book_rejects_missing_customer(client, screening):
    resp = client.post(
        f"/screenings/{screening['id']}/book", json={"seats": ["A1"]}
    )
    assert resp.status_code == 400


def test_book_missing_screening_404(client):
    assert _book(client, 999, ["A1"]).status_code == 404


def test_cancel_booking_frees_seats(client, screening):
    booking = _book(client, screening["id"], ["A1", "A2"]).get_json()
    resp = client.post(f"/bookings/{booking['id']}/cancel")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "cancelled"
    state = client.get(f"/screenings/{screening['id']}").get_json()
    assert state["seats"]["A1"] == "available"
    assert state["booked_seats"] == 0
    # The freed seats can be booked again.
    assert _book(client, screening["id"], ["A1"], name="Hamza").status_code == 201


def test_cancel_twice_returns_409(client, screening):
    booking = _book(client, screening["id"], ["A1"]).get_json()
    assert client.post(f"/bookings/{booking['id']}/cancel").status_code == 200
    assert client.post(f"/bookings/{booking['id']}/cancel").status_code == 409


def test_cancel_missing_booking_404(client):
    assert client.post("/bookings/999/cancel").status_code == 404


def test_list_bookings_filters_by_status(client, screening):
    first = _book(client, screening["id"], ["A1"]).get_json()
    _book(client, screening["id"], ["A2"], name="Hamza")
    client.post(f"/bookings/{first['id']}/cancel")
    cancelled = client.get("/bookings", query_string={"status": "cancelled"}).get_json()
    assert [b["id"] for b in cancelled] == [first["id"]]
    active = client.get("/bookings", query_string={"status": "active"}).get_json()
    assert len(active) == 1
    assert client.get("/bookings/999").status_code == 404


def test_list_screening_bookings(client, screening):
    _book(client, screening["id"], ["A1"])
    _book(client, screening["id"], ["A2"], name="Hamza")
    bookings = client.get(f"/screenings/{screening['id']}/bookings").get_json()
    assert len(bookings) == 2
    assert client.get("/screenings/999/bookings").status_code == 404
