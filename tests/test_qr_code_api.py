"""Tests for the qr_code_api Flask app, using Flask's test client."""
import pytest

from qr_code_api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_encode_single_char_grid(client):
    resp = client.post("/encode", json={"text": "A"})
    assert resp.status_code == 200
    data = resp.get_json()
    # "A" is 0x41 = 01000001, padded to one 16-bit row.
    assert data["rows"] == 1 and data["cols"] == 16
    assert data["grid"] == [[0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]
    assert data["ascii"] == " #     #        "


def test_encode_multi_char_rows(client):
    data = client.post("/encode", json={"text": "Hi!"}).get_json()
    assert data["rows"] == 2 and data["cols"] == 16
    assert all(len(row) == 16 for row in data["grid"])
    assert data["ascii"].count("\n") == 1


def test_encode_decode_round_trip(client):
    for text in ("Hello, world!", "PKR 278", "café ☕", "12345"):
        grid = client.post("/encode", json={"text": text}).get_json()["grid"]
        decoded = client.post("/decode", json={"grid": grid}).get_json()
        assert decoded["text"] == text


def test_decode_rejects_invalid_grid(client):
    assert client.post("/decode", json={}).status_code == 400
    assert client.post("/decode", json={"grid": []}).status_code == 400
    assert client.post("/decode", json={"grid": [[0, 1], [1]]}).status_code == 400
    resp = client.post("/decode", json={"grid": [[0, 2]]})
    assert resp.status_code == 400
    assert "0 or 1" in resp.get_json()["error"]
    assert client.post("/decode", json={"grid": [[True]]}).status_code == 400
    assert client.post("/decode", json={"grid": "nope"}).status_code == 400


def test_decode_rejects_non_utf8(client):
    # 0xFF on its own is never valid UTF-8.
    grid = [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]
    resp = client.post("/decode", json={"grid": grid})
    assert resp.status_code == 400
    assert "UTF-8" in resp.get_json()["error"]


def test_encode_rejects_missing_and_empty_text(client):
    assert client.post("/encode", json={}).status_code == 400
    assert client.post("/encode", json={"text": ""}).status_code == 400
    assert client.post("/encode", json={"text": 42}).status_code == 400
    resp = client.post("/encode", json={"text": "a\x00b"})
    assert resp.status_code == 400


def test_encode_rejects_too_long_text(client):
    resp = client.post("/encode", json={"text": "x" * 257})
    assert resp.status_code == 400
    assert "256" in resp.get_json()["error"]
    assert client.post("/encode", json={"text": "x" * 256}).status_code == 200


def test_encode_is_deterministic(client):
    first = client.post("/encode", json={"text": "same"}).get_json()
    second = client.post("/encode", json={"text": "same"}).get_json()
    assert first["grid"] == second["grid"]


def test_non_json_body_is_400(client):
    resp = client.post("/encode", data="not json", content_type="text/plain")
    assert resp.status_code == 400


def test_unknown_route_is_404(client):
    assert client.get("/nope").status_code == 404
