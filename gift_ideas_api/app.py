"""REST gift-ideas API built with Flask.

In-memory store, fresh for each app instance. Track people, the
occasions you buy for, and gift ideas per person with a price and a
status (``idea`` or ``bought``). Each person has a budget; bought gifts
count against it.

Endpoints:
- ``GET /people`` / ``POST /people`` — list / add a person
  (``{"name", "budget": n?}``)
- ``GET /people/<id>`` — one person, with budget summary
- ``GET /people/<id>/budget`` — budget, spent, remaining
- ``GET /people/<id>/ideas`` / ``POST /people/<id>/ideas`` — list / add
  a gift idea (``{"title", "price", "occasion"?}``)
- ``PUT /ideas/<id>`` — update title / price / occasion / status
- ``DELETE /ideas/<id>`` — delete a gift idea
"""
from flask import Flask, jsonify, request

STATUSES = ("idea", "bought")


def create_app():
    """Return a Flask app with a fresh, empty in-memory gift store."""
    app = Flask(__name__)
    app.config["PEOPLE"] = {}
    app.config["IDEAS"] = {}
    app.config["NEXT_PERSON_ID"] = 1
    app.config["NEXT_IDEA_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _valid_price(value):
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value >= 0
        )

    def _budget_summary(person):
        ideas = [i for i in app.config["IDEAS"].values() if i["person_id"] == person["id"]]
        spent = sum(i["price"] for i in ideas if i["status"] == "bought")
        return {
            "budget": person["budget"],
            "spent": spent,
            "remaining": round(person["budget"] - spent, 2),
            "idea_count": len(ideas),
            "bought_count": sum(1 for i in ideas if i["status"] == "bought"),
        }

    def _public_person(person):
        return {**person, **_budget_summary(person)}

    def _get_person(person_id):
        person = app.config["PEOPLE"].get(person_id)
        if person is None:
            return None, _error(f"person {person_id} not found", 404)
        return person, None

    @app.get("/people")
    def list_people():
        people = sorted(app.config["PEOPLE"].values(), key=lambda p: p["id"])
        return jsonify([_public_person(p) for p in people])

    @app.post("/people")
    def add_person():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            return _error("name is required and must be a non-empty string", 400)
        budget = data.get("budget", 0)
        if not _valid_price(budget):
            return _error("budget must be a non-negative number", 400)
        person_id = app.config["NEXT_PERSON_ID"]
        app.config["NEXT_PERSON_ID"] += 1
        person = {"id": person_id, "name": name.strip(), "budget": budget}
        app.config["PEOPLE"][person_id] = person
        return jsonify(_public_person(person)), 201

    @app.get("/people/<int:person_id>")
    def get_person(person_id):
        person, err = _get_person(person_id)
        if err is not None:
            return err
        return jsonify(_public_person(person))

    @app.get("/people/<int:person_id>/budget")
    def get_budget(person_id):
        person, err = _get_person(person_id)
        if err is not None:
            return err
        summary = _budget_summary(person)
        return jsonify({"person_id": person_id, "name": person["name"], **summary})

    @app.get("/people/<int:person_id>/ideas")
    def list_ideas(person_id):
        person, err = _get_person(person_id)
        if err is not None:
            return err
        ideas = sorted(
            (i for i in app.config["IDEAS"].values() if i["person_id"] == person_id),
            key=lambda i: i["id"],
        )
        return jsonify(ideas)

    @app.post("/people/<int:person_id>/ideas")
    def add_idea(person_id):
        person, err = _get_person(person_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        title = data.get("title")
        if not isinstance(title, str) or not title.strip():
            return _error("title is required and must be a non-empty string", 400)
        price = data.get("price")
        if not _valid_price(price):
            return _error("price is required and must be a non-negative number", 400)
        occasion = data.get("occasion", "")
        if not isinstance(occasion, str):
            return _error("occasion must be a string", 400)
        status = data.get("status", "idea")
        if status not in STATUSES:
            return _error("status must be 'idea' or 'bought'", 400)
        idea_id = app.config["NEXT_IDEA_ID"]
        app.config["NEXT_IDEA_ID"] += 1
        idea = {
            "id": idea_id,
            "person_id": person_id,
            "title": title.strip(),
            "price": price,
            "occasion": occasion,
            "status": status,
        }
        app.config["IDEAS"][idea_id] = idea
        return jsonify(idea), 201

    @app.put("/ideas/<int:idea_id>")
    def update_idea(idea_id):
        idea = app.config["IDEAS"].get(idea_id)
        if idea is None:
            return _error(f"gift idea {idea_id} not found", 404)
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        if "title" in data:
            if not isinstance(data["title"], str) or not data["title"].strip():
                return _error("title must be a non-empty string", 400)
            idea["title"] = data["title"].strip()
        if "price" in data:
            if not _valid_price(data["price"]):
                return _error("price must be a non-negative number", 400)
            idea["price"] = data["price"]
        if "occasion" in data:
            if not isinstance(data["occasion"], str):
                return _error("occasion must be a string", 400)
            idea["occasion"] = data["occasion"]
        if "status" in data:
            if data["status"] not in STATUSES:
                return _error("status must be 'idea' or 'bought'", 400)
            idea["status"] = data["status"]
        return jsonify(idea)

    @app.delete("/ideas/<int:idea_id>")
    def delete_idea(idea_id):
        idea = app.config["IDEAS"].get(idea_id)
        if idea is None:
            return _error(f"gift idea {idea_id} not found", 404)
        del app.config["IDEAS"][idea_id]
        return jsonify({"deleted": idea_id})

    return app
