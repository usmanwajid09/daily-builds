"""Plant care API built with Flask.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Endpoints:

- ``POST /plants`` — create a plant
- ``GET /plants`` — list all plants with ``next_watering`` and
  ``days_until_watering`` (negative when overdue)
- ``GET /plants/due`` — plants due today or overdue, earliest first
- ``GET /plants/<id>`` — fetch one plant
- ``PUT /plants/<id>`` — update a plant
- ``DELETE /plants/<id>`` — delete a plant
- ``POST /plants/<id>/water`` — log a watering
- ``GET /plants/<id>/water-log`` — watering events, newest first

A plant is ``{"id": int, "name": str, "species": str | None,
"watering_interval_days": int, "last_watered": "YYYY-MM-DD",
"next_watering": "YYYY-MM-DD", "days_until_watering": int}``.
``last_watered`` defaults to today when omitted.
"""
from datetime import date, timedelta

from flask import Flask, jsonify, request


def create_app():
    """Return a Flask app with a fresh, empty in-memory plant store."""
    app = Flask(__name__)
    app.config["PLANTS"] = {}
    app.config["NEXT_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _parse_date(value):
        """Parse a YYYY-MM-DD string, returning (date, None) or (None, error)."""
        if not isinstance(value, str):
            return None, _error("date must be a YYYY-MM-DD string", 400)
        try:
            return date.fromisoformat(value), None
        except ValueError:
            return None, _error("date must be a valid YYYY-MM-DD date", 400)

    def _validate_interval(value):
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
        ):
            return _error(
                "watering_interval_days is required and must be an integer above 0",
                400,
            )
        return None

    def _public(plant):
        """Return a plant dict with computed watering fields."""
        last = date.fromisoformat(plant["last_watered"])
        next_watering = last + timedelta(days=plant["watering_interval_days"])
        return {
            "id": plant["id"],
            "name": plant["name"],
            "species": plant["species"],
            "watering_interval_days": plant["watering_interval_days"],
            "last_watered": plant["last_watered"],
            "next_watering": next_watering.isoformat(),
            "days_until_watering": (next_watering - date.today()).days,
        }

    def _get_plant(plant_id):
        plant = app.config["PLANTS"].get(plant_id)
        if plant is None:
            return None, _error(f"plant {plant_id} not found", 404)
        return plant, None

    @app.post("/plants")
    def create_plant():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            return _error("name is required and must be a non-empty string", 400)
        species = data.get("species")
        if species is not None and not isinstance(species, str):
            return _error("species must be a string", 400)
        err = _validate_interval(data.get("watering_interval_days"))
        if err is not None:
            return err
        if "last_watered" in data and data["last_watered"] is not None:
            parsed, err = _parse_date(data["last_watered"])
            if err is not None:
                return err
            last_watered = parsed
        else:
            last_watered = date.today()
        plant_id = app.config["NEXT_ID"]
        app.config["NEXT_ID"] += 1
        plant = {
            "id": plant_id,
            "name": name.strip(),
            "species": species.strip() if isinstance(species, str) else None,
            "watering_interval_days": data["watering_interval_days"],
            "last_watered": last_watered.isoformat(),
            "water_log": [
                {"date": last_watered.isoformat()}
            ],
        }
        app.config["PLANTS"][plant_id] = plant
        return jsonify(_public(plant)), 201

    @app.get("/plants")
    def list_plants():
        plants = sorted(app.config["PLANTS"].values(), key=lambda p: p["id"])
        return jsonify([_public(p) for p in plants])

    @app.get("/plants/due")
    def due_plants():
        today = date.today()
        due = [
            p
            for p in app.config["PLANTS"].values()
            if date.fromisoformat(p["last_watered"])
            + timedelta(days=p["watering_interval_days"])
            <= today
        ]
        due.sort(
            key=lambda p: date.fromisoformat(p["last_watered"])
            + timedelta(days=p["watering_interval_days"])
        )
        return jsonify([_public(p) for p in due])

    @app.get("/plants/<int:plant_id>")
    def get_plant(plant_id):
        plant, err = _get_plant(plant_id)
        if err is not None:
            return err
        return jsonify(_public(plant))

    @app.put("/plants/<int:plant_id>")
    def update_plant(plant_id):
        plant, err = _get_plant(plant_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        if "name" in data:
            if not isinstance(data["name"], str) or not data["name"].strip():
                return _error("name must be a non-empty string", 400)
            plant["name"] = data["name"].strip()
        if "species" in data:
            if data["species"] is not None and not isinstance(data["species"], str):
                return _error("species must be a string", 400)
            plant["species"] = (
                data["species"].strip()
                if isinstance(data["species"], str)
                else None
            )
        if "watering_interval_days" in data:
            err = _validate_interval(data["watering_interval_days"])
            if err is not None:
                return err
            plant["watering_interval_days"] = data["watering_interval_days"]
        if "last_watered" in data and data["last_watered"] is not None:
            parsed, err = _parse_date(data["last_watered"])
            if err is not None:
                return err
            plant["last_watered"] = parsed.isoformat()
        return jsonify(_public(plant))

    @app.delete("/plants/<int:plant_id>")
    def delete_plant(plant_id):
        if plant_id not in app.config["PLANTS"]:
            return _error(f"plant {plant_id} not found", 404)
        del app.config["PLANTS"][plant_id]
        return jsonify({"deleted": plant_id})

    @app.post("/plants/<int:plant_id>/water")
    def water_plant(plant_id):
        plant, err = _get_plant(plant_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if data is None:
            data = {}
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        if "date" in data and data["date"] is not None:
            parsed, err = _parse_date(data["date"])
            if err is not None:
                return err
            watered_on = parsed
        else:
            watered_on = date.today()
        if watered_on > date.today():
            return _error("watering date cannot be in the future", 400)
        plant["last_watered"] = watered_on.isoformat()
        plant["water_log"].append({"date": watered_on.isoformat()})
        return jsonify(_public(plant))

    @app.get("/plants/<int:plant_id>/water-log")
    def water_log(plant_id):
        plant, err = _get_plant(plant_id)
        if err is not None:
            return err
        return jsonify(list(reversed(plant["water_log"])))

    @app.errorhandler(404)
    def handle_404(exc):
        return _error("route not found", 404)

    return app
