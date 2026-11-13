"""Tests for the password_api Flask app, using Flask's test client."""
import string

import pytest

from password_api.app import check_strength, create_app, generate_password


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_index_describes_service(client):
    assert client.get("/").get_json()["service"] == "password_api"


def test_generate_default_length_16_with_all_classes(client):
    password = client.post("/generate", json={}).get_json()["password"]
    assert len(password) == 16
    assert any(c.islower() for c in password)
    assert any(c.isupper() for c in password)
    assert any(c.isdigit() for c in password)
    assert any(c in string.punctuation for c in password)


def test_generate_custom_length_and_digits_only(client):
    resp = client.post(
        "/generate",
        json={
            "length": 8,
            "symbols": False,
            "uppercase": False,
            "lowercase": False,
            "digits": True,
        },
    )
    password = resp.get_json()["password"]
    assert len(password) == 8
    assert password.isdigit()


def test_generate_two_passwords_differ(client):
    first = client.post("/generate", json={}).get_json()["password"]
    second = client.post("/generate", json={}).get_json()["password"]
    assert first != second


def test_generate_rejects_bad_lengths(client):
    for bad in (3, 129, "16", 12.5, True):
        resp = client.post("/generate", json={"length": bad})
        assert resp.status_code == 400, bad
        assert "length" in resp.get_json()["error"]


def test_generate_rejects_no_character_class(client):
    resp = client.post(
        "/generate",
        json={
            "symbols": False,
            "digits": False,
            "uppercase": False,
            "lowercase": False,
        },
    )
    assert resp.status_code == 400
    assert "character class" in resp.get_json()["error"]


def test_generate_rejects_non_bool_option(client):
    resp = client.post("/generate", json={"symbols": "yes"})
    assert resp.status_code == 400


def test_strength_weak_password(client):
    result = client.post("/strength", json={"password": "password"}).get_json()
    assert result["score"] == 1
    assert result["label"] == "Weak"
    assert "add a digit" in result["feedback"]


def test_strength_strong_password(client):
    result = client.post(
        "/strength", json={"password": "Tr0ub4dor&3xK9!z"}
    ).get_json()
    assert result["score"] == 5
    assert result["label"] == "Strong"
    assert result["feedback"] == []


def test_strength_requires_password(client):
    assert client.post("/strength", json={}).status_code == 400
    assert client.post("/strength", json={"password": ""}).status_code == 400
    assert client.post("/strength", json={"password": 123}).status_code == 400


def test_generate_password_function_validates_directly():
    assert len(generate_password(4, lowercase=True)) == 4
    with pytest.raises(ValueError):
        generate_password(3, lowercase=True)
    with pytest.raises(ValueError):
        generate_password(10)
    assert check_strength("abc")["score"] == 0
