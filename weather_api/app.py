"""Mock weather API built with Flask.

In-memory store of a few cities with current conditions and a 3-day
forecast. Nothing is written to disk. All responses are JSON. Endpoints:

- ``GET /weather`` — list all cities' current conditions
  (``?units=c|f``)
- ``GET /weather/<city>`` — current conditions for one city
  (``?units=c|f``)
- ``GET /weather/<city>/forecast`` — 3-day forecast (``?units=c|f``)

Temperatures are stored in Celsius and converted to Fahrenheit when
``units=f`` is requested. Unknown cities return 404; an invalid
``units`` value returns 400.
"""
from flask import Flask, jsonify, request

_CITIES = {
    "lahore": {
        "city": "Lahore",
        "temp_c": 24,
        "condition": "Hazy sunshine",
        "humidity": 48,
        "wind_kmh": 11,
        "forecast": [
            {"day": "2026-11-18", "temp_c": 25, "condition": "Sunny"},
            {"day": "2026-11-19", "temp_c": 23, "condition": "Hazy sunshine"},
            {"day": "2026-11-20", "temp_c": 22, "condition": "Partly cloudy"},
        ],
    },
    "london": {
        "city": "London",
        "temp_c": 9,
        "condition": "Light rain",
        "humidity": 84,
        "wind_kmh": 19,
        "forecast": [
            {"day": "2026-11-18", "temp_c": 10, "condition": "Cloudy"},
            {"day": "2026-11-19", "temp_c": 8, "condition": "Rain"},
            {"day": "2026-11-20", "temp_c": 11, "condition": "Sunny intervals"},
        ],
    },
    "karachi": {
        "city": "Karachi",
        "temp_c": 29,
        "condition": "Sunny",
        "humidity": 61,
        "wind_kmh": 16,
        "forecast": [
            {"day": "2026-11-18", "temp_c": 30, "condition": "Sunny"},
            {"day": "2026-11-19", "temp_c": 29, "condition": "Sunny"},
            {"day": "2026-11-20", "temp_c": 28, "condition": "Partly cloudy"},
        ],
    },
    "new york": {
        "city": "New York",
        "temp_c": 12,
        "condition": "Partly cloudy",
        "humidity": 55,
        "wind_kmh": 14,
        "forecast": [
            {"day": "2026-11-18", "temp_c": 13, "condition": "Sunny"},
            {"day": "2026-11-19", "temp_c": 10, "condition": "Rain"},
            {"day": "2026-11-20", "temp_c": 9, "condition": "Cloudy"},
        ],
    },
    "tokyo": {
        "city": "Tokyo",
        "temp_c": 16,
        "condition": "Clear",
        "humidity": 52,
        "wind_kmh": 9,
        "forecast": [
            {"day": "2026-11-18", "temp_c": 17, "condition": "Clear"},
            {"day": "2026-11-19", "temp_c": 15, "condition": "Cloudy"},
            {"day": "2026-11-20", "temp_c": 14, "condition": "Light rain"},
        ],
    },
}


def _to_f(temp_c):
    return round(temp_c * 9 / 5 + 32, 1)


def create_app():
    """Return a Flask app serving the mock weather data."""
    app = Flask(__name__)

    def _error(message, status):
        return jsonify({"error": message}), status

    def _parse_units():
        units = request.args.get("units", "c").lower()
        if units not in ("c", "f"):
            return None, _error(
                "units must be 'c' or 'f'", 400
            )
        return units, None

    def _convert(entry, units):
        out = dict(entry)
        if units == "f":
            out["temp_c"] = _to_f(entry["temp_c"])
            out["temp_unit"] = "F"
        else:
            out["temp_unit"] = "C"
        out.pop("forecast", None)
        return out

    def _get_city(name):
        city = _CITIES.get(name.strip().lower())
        if city is None:
            return None, _error(f"unknown city: {name}", 404)
        return city, None

    @app.get("/weather")
    def list_weather():
        units, err = _parse_units()
        if err is not None:
            return err
        return jsonify([_convert(c, units) for c in _CITIES.values()])

    @app.get("/weather/<path:city>")
    def get_weather(city):
        units, err = _parse_units()
        if err is not None:
            return err
        entry, err = _get_city(city)
        if err is not None:
            return err
        return jsonify(_convert(entry, units))

    @app.get("/weather/<path:city>/forecast")
    def get_forecast(city):
        units, err = _parse_units()
        if err is not None:
            return err
        entry, err = _get_city(city)
        if err is not None:
            return err
        days = []
        for day in entry["forecast"]:
            item = dict(day)
            if units == "f":
                item["temp_c"] = _to_f(day["temp_c"])
                item["temp_unit"] = "F"
            else:
                item["temp_unit"] = "C"
            days.append(item)
        return jsonify({"city": entry["city"], "forecast": days})

    return app
