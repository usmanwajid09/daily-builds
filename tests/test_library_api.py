"""Tests for the library_api Flask app, using Flask's test client."""
import pytest

from library_api.app import create_app

BOOK = {
    "title": "Dune",
    "author": "Frank Herbert",
    "isbn": "9780441172719",
    "total_copies": 2,
}


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture
def book_id(client):
    return client.post("/books", json=BOOK).get_json()["id"]


def test_list_starts_empty(client):
    assert client.get("/books").get_json() == []


def test_create_book_returns_201_with_available_copies(client):
    resp = client.post("/books", json=BOOK)
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["id"] == 1
    assert body["available_copies"] == 2
    assert body["total_copies"] == 2


def test_create_rejects_missing_title_or_author(client):
    assert client.post("/books", json={"author": "x", "total_copies": 1}).status_code == 400
    assert client.post("/books", json={"title": "x", "total_copies": 1}).status_code == 400


def test_create_rejects_bad_total_copies(client):
    for bad in (0, -1, 1.5, "two", True):
        assert client.post("/books", json=dict(BOOK, total_copies=bad)).status_code == 400


def test_borrow_reduces_availability_and_sets_due_date(client, book_id):
    resp = client.post(f"/books/{book_id}/borrow", json={"borrower": "Ayesha"})
    assert resp.status_code == 201
    loan = resp.get_json()
    assert loan["loan_id"] == 1
    assert loan["borrower"] == "Ayesha"
    assert loan["returned"] is False
    assert loan["overdue"] is False
    assert loan["due_on"] > loan["borrowed_on"]
    assert client.get(f"/books/{book_id}").get_json()["available_copies"] == 1


def test_borrow_rejects_when_no_copies_left(client, book_id):
    client.post(f"/books/{book_id}/borrow", json={"borrower": "A"})
    client.post(f"/books/{book_id}/borrow", json={"borrower": "B"})
    resp = client.post(f"/books/{book_id}/borrow", json={"borrower": "C"})
    assert resp.status_code == 400
    assert "no copies" in resp.get_json()["error"]


def test_borrow_rejects_missing_borrower_and_bad_due_days(client, book_id):
    assert client.post(f"/books/{book_id}/borrow", json={}).status_code == 400
    assert client.post(
        f"/books/{book_id}/borrow", json={"borrower": "A", "due_days": 0}
    ).status_code == 400


def test_return_restores_availability(client, book_id):
    loan = client.post(f"/books/{book_id}/borrow", json={"borrower": "Ayesha"}).get_json()
    resp = client.post(f"/books/{book_id}/return", json={"loan_id": loan["loan_id"]})
    assert resp.status_code == 200
    assert resp.get_json()["returned"] is True
    assert client.get(f"/books/{book_id}").get_json()["available_copies"] == 2


def test_return_twice_is_400_and_unknown_loan_is_404(client, book_id):
    loan = client.post(f"/books/{book_id}/borrow", json={"borrower": "A"}).get_json()
    client.post(f"/books/{book_id}/return", json={"loan_id": loan["loan_id"]})
    resp = client.post(f"/books/{book_id}/return", json={"loan_id": loan["loan_id"]})
    assert resp.status_code == 400
    assert "already returned" in resp.get_json()["error"]
    assert client.post(f"/books/{book_id}/return", json={"loan_id": 999}).status_code == 404


def test_delete_blocked_while_on_loan(client, book_id):
    client.post(f"/books/{book_id}/borrow", json={"borrower": "A"})
    assert client.delete(f"/books/{book_id}").status_code == 400


def test_loans_list_and_overdue_filter(client, book_id):
    client.post(f"/books/{book_id}/borrow", json={"borrower": "A", "due_days": 14})
    loans = client.get("/loans").get_json()
    assert len(loans) == 1
    assert loans[0]["overdue"] is False
    assert client.get("/loans?overdue=true").get_json() == []
    assert len(client.get("/loans?overdue=false").get_json()) == 1
    assert client.get("/loans?overdue=maybe").status_code == 400


def test_update_cannot_drop_total_below_borrowed(client, book_id):
    client.post(f"/books/{book_id}/borrow", json={"borrower": "A"})
    resp = client.put(f"/books/{book_id}", json={"total_copies": 0})
    assert resp.status_code == 400
    updated = client.put(f"/books/{book_id}", json={"total_copies": 3}).get_json()
    assert updated["total_copies"] == 3
    assert updated["available_copies"] == 2


def test_unknown_book_is_404_json(client):
    for method in ("get", "put", "delete"):
        resp = getattr(client, method)("/books/999", json={})
        assert resp.status_code == 404
        assert "not found" in resp.get_json()["error"]
    assert client.post("/books/999/borrow", json={"borrower": "A"}).status_code == 404


def test_unknown_route_is_404_json(client):
    resp = client.get("/nope")
    assert resp.status_code == 404
    assert "error" in resp.get_json()
