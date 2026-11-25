"""Flask travel planner API: trips with itinerary days and activities, plus per-trip cost totals."""
from datetime import datetime

from flask import Flask, jsonify, request


def _parse_date(value):
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


def _bad(message, code=400):
    return jsonify({"error": message}), code


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def create_app():
    """Return a Flask app with a fresh, empty in-memory travel store."""
    app = Flask(__name__)
    trips = {}
    days = {}
    activities = {}
    next_trip = [1]
    next_day = [1]
    next_activity = [1]

    def _trip_cost(trip_id):
        day_ids = {d["id"] for d in days.values() if d["trip_id"] == trip_id}
        return round(sum(a["cost"] for a in activities.values() if a["day_id"] in day_ids), 2)

    def _trip_view(trip):
        trip_days = [d for d in days.values() if d["trip_id"] == trip["id"]]
        activity_count = sum(
            1 for a in activities.values()
            if a["day_id"] in {d["id"] for d in trip_days}
        )
        return {
            **trip,
            "day_count": len(trip_days),
            "activity_count": activity_count,
            "total_cost": _trip_cost(trip["id"]),
        }

    @app.route("/trips", methods=["POST"])
    def create_trip():
        data = request.get_json(silent=True) or {}
        name = data.get("name")
        destination = data.get("destination")
        start = _parse_date(data.get("start_date"))
        end = _parse_date(data.get("end_date"))
        if not isinstance(name, str) or not name.strip():
            return _bad("name is required and must be a non-empty string")
        if not isinstance(destination, str) or not destination.strip():
            return _bad("destination is required and must be a non-empty string")
        if start is None or end is None:
            return _bad("start_date and end_date are required as YYYY-MM-DD")
        if end < start:
            return _bad("end_date must not be before start_date")
        trip_id = next_trip[0]
        next_trip[0] += 1
        trips[trip_id] = {
            "id": trip_id,
            "name": name.strip(),
            "destination": destination.strip(),
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
        }
        return jsonify(_trip_view(trips[trip_id])), 201

    @app.route("/trips", methods=["GET"])
    def list_trips():
        return jsonify([_trip_view(t) for t in trips.values()])

    @app.route("/trips/<int:trip_id>", methods=["GET"])
    def get_trip(trip_id):
        trip = trips.get(trip_id)
        if trip is None:
            return _bad("trip not found", 404)
        return jsonify(_trip_view(trip))

    @app.route("/trips/<int:trip_id>", methods=["DELETE"])
    def delete_trip(trip_id):
        if trip_id not in trips:
            return _bad("trip not found", 404)
        del trips[trip_id]
        day_ids = [k for k, v in days.items() if v["trip_id"] == trip_id]
        for day_id in day_ids:
            del days[day_id]
        for act_id in [k for k, v in activities.items() if v["day_id"] in day_ids]:
            del activities[act_id]
        return jsonify({"deleted": trip_id})

    @app.route("/trips/<int:trip_id>/days", methods=["POST"])
    def create_day(trip_id):
        trip = trips.get(trip_id)
        if trip is None:
            return _bad("trip not found", 404)
        data = request.get_json(silent=True) or {}
        day_date = _parse_date(data.get("date"))
        title = data.get("title")
        if day_date is None:
            return _bad("date is required as YYYY-MM-DD")
        if not isinstance(title, str) or not title.strip():
            return _bad("title is required and must be a non-empty string")
        if day_date < _parse_date(trip["start_date"]) or day_date > _parse_date(trip["end_date"]):
            return _bad("date must fall within the trip's start and end dates")
        if any(d["trip_id"] == trip_id and d["date"] == day_date.isoformat() for d in days.values()):
            return _bad("an itinerary day already exists for that date", 409)
        day_id = next_day[0]
        next_day[0] += 1
        days[day_id] = {
            "id": day_id,
            "trip_id": trip_id,
            "date": day_date.isoformat(),
            "title": title.strip(),
        }
        return jsonify(days[day_id]), 201

    @app.route("/trips/<int:trip_id>/days", methods=["GET"])
    def list_days(trip_id):
        if trip_id not in trips:
            return _bad("trip not found", 404)
        return jsonify(sorted(
            (d for d in days.values() if d["trip_id"] == trip_id),
            key=lambda d: d["date"],
        ))

    @app.route("/trips/<int:trip_id>/days/<int:day_id>", methods=["DELETE"])
    def delete_day(trip_id, day_id):
        day = days.get(day_id)
        if day is None or day["trip_id"] != trip_id:
            return _bad("itinerary day not found", 404)
        del days[day_id]
        for act_id in [k for k, v in activities.items() if v["day_id"] == day_id]:
            del activities[act_id]
        return jsonify({"deleted": day_id})

    @app.route("/trips/<int:trip_id>/days/<int:day_id>/activities", methods=["POST"])
    def create_activity(trip_id, day_id):
        day = days.get(day_id)
        if day is None or day["trip_id"] != trip_id:
            return _bad("itinerary day not found", 404)
        data = request.get_json(silent=True) or {}
        name = data.get("name")
        cost = data.get("cost")
        if not isinstance(name, str) or not name.strip():
            return _bad("name is required and must be a non-empty string")
        if not _is_number(cost) or cost < 0:
            return _bad("cost is required and must be a non-negative number")
        description = data.get("description", "")
        if description is not None and not isinstance(description, str):
            return _bad("description must be a string")
        act_id = next_activity[0]
        next_activity[0] += 1
        activities[act_id] = {
            "id": act_id,
            "day_id": day_id,
            "name": name.strip(),
            "cost": cost,
            "description": description or "",
        }
        return jsonify(activities[act_id]), 201

    @app.route("/trips/<int:trip_id>/days/<int:day_id>/activities", methods=["GET"])
    def list_activities(trip_id, day_id):
        day = days.get(day_id)
        if day is None or day["trip_id"] != trip_id:
            return _bad("itinerary day not found", 404)
        return jsonify([a for a in activities.values() if a["day_id"] == day_id])

    @app.route("/trips/<int:trip_id>/cost", methods=["GET"])
    def trip_cost(trip_id):
        trip = trips.get(trip_id)
        if trip is None:
            return _bad("trip not found", 404)
        by_day = []
        for day in sorted(
            (d for d in days.values() if d["trip_id"] == trip_id),
            key=lambda d: d["date"],
        ):
            mine = [a for a in activities.values() if a["day_id"] == day["id"]]
            by_day.append({
                "day_id": day["id"],
                "date": day["date"],
                "title": day["title"],
                "activity_count": len(mine),
                "cost": round(sum(a["cost"] for a in mine), 2),
            })
        return jsonify({
            "trip_id": trip_id,
            "name": trip["name"],
            "total_cost": _trip_cost(trip_id),
            "by_day": by_day,
        })

    @app.errorhandler(404)
    def not_found(_error):
        return jsonify({"error": "route not found"}), 404

    return app
