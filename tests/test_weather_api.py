"""Tests for weather_api."""
import pytest

from weather_api.app import create_app


@pytest.fixture
def client():
    return create_app().test_client()


def test_list_all_cities(client):
    resp = client.get("/weather")
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data) == 5
    names = {c["city"] for c in data}
    assert names == {"Lahore", "London", "Karachi", "New York", "Tokyo"}


def test_get_city_celsius_default(client):
    data = client.get("/weather/Lahore").get_json()
    assert data["city"] == "Lahore"
    assert data["temp_c"] == 24
    assert data["temp_unit"] == "C"
    assert data["condition"] == "Hazy sunshine"


def test_city_lookup_case_insensitive(client):
    assert client.get("/weather/lAhOrE").get_json()["city"] == "Lahore"


def test_get_city_fahrenheit(client):
    data = client.get("/weather/Lahore?units=f").get_json()
    assert data["temp_c"] == 75.2
    assert data["temp_unit"] == "F"


def test_fahrenheit_conversion_london(client):
    data = client.get("/weather/London?units=f").get_json()
    assert data["temp_c"] == 48.2


def test_forecast_three_days(client):
    data = client.get("/weather/Lahore/forecast").get_json()
    assert data["city"] == "Lahore"
    assert len(data["forecast"]) == 3
    assert data["forecast"][0]["day"] == "2026-11-18"
    assert data["forecast"][0]["temp_c"] == 25


def test_forecast_fahrenheit(client):
    data = client.get("/weather/Tokyo/forecast?units=f").get_json()
    assert data["forecast"][0]["temp_c"] == 62.6


def test_unknown_city_404(client):
    resp = client.get("/weather/Atlantis")
    assert resp.status_code == 404
    assert "unknown city" in resp.get_json()["error"]


def test_unknown_city_forecast_404(client):
    resp = client.get("/weather/Atlantis/forecast")
    assert resp.status_code == 404


def test_invalid_units_400(client):
    resp = client.get("/weather/Lahore?units=k")
    assert resp.status_code == 400
    assert "units" in resp.get_json()["error"]


def test_invalid_units_on_list_400(client):
    resp = client.get("/weather?units=kelvin")
    assert resp.status_code == 400
