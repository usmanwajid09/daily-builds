"""Flask meal planner API.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Endpoints:

- ``POST /meals`` — add a meal (``{\"name\": str, \"day\": \"mon\" | ... |
  \"sun\", \"meal_type\": \"breakfast\" | \"lunch\" | \"dinner\" | \"snack\",
  \"calories\": int, \"ingredients\": [{\"name\": str, \"quantity\": number,
  \"unit\": str}]?}``)
- ``GET /meals`` — list meals; ``?day=`` and ``?meal_type=`` filters
- ``GET /meals/<id>`` — fetch one meal
- ``PUT /meals/<id>`` — update name, day, meal_type, calories
  and/or ingredients
- ``DELETE /meals/<id>`` — delete a meal
- ``GET /plan/<day>`` — the day's meals plus its total calories
- ``GET /plan/week`` — every day's meals and totals, plus the week total
- ``GET /calories`` — total calories; ``?day=`` for a single day
- ``GET /shopping-list`` — ingredients aggregated across all planned
  meals (same name + unit summed), plus how many meals use each item

A meal is ``{\"id\": int, \"name\": str, \"day\": str,
\"meal_type\": str, \"calories\": int, \"ingredients\": [...]}``.
An ingredient is ``{\"name\": str, \"quantity\": number, \"unit\": str}``.
"""
from flask import Flask, jsonify, request

DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
MEAL_TYPES = ("breakfast", "lunch", "dinner", "snack")


def _is_number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool))


def create_app():
    """Return a Flask app with a fresh, empty in-memory meal plan."""
    app = Flask(__name__)
    app.config["MEALS"] = {}
    app.config["NEXT_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _get_meal(meal_id):
        meal = app.config["MEALS"].get(meal_id)
        if meal is None:
            return None, _error(f"meal {meal_id} not found", 404)
        return meal, None

    def _validate_ingredients(ingredients):
        if not isinstance(ingredients, list):
            return "ingredients must be a list"
        for item in ingredients:
            if not isinstance(item, dict):
                return "each ingredient must be an object"
            if not isinstance(item.get("name"), str) or not item["name"].strip():
                return "each ingredient needs a non-empty name"
            if not _is_number(item.get("quantity")) or item["quantity"] <= 0:
                return "each ingredient quantity must be a positive number"
            if not isinstance(item.get("unit"), str) or not item["unit"].strip():
                return "each ingredient needs a non-empty unit"
        return None

    def _validate_meal(data, partial=False):
        if not isinstance(data, dict):
            return "request body must be a JSON object"
        if "name" in data or not partial:
            if not isinstance(data.get("name"), str) or not data["name"].strip():
                return "name is required and must be a non-empty string"
        if "day" in data or not partial:
            if data.get("day") not in DAYS:
                return f"day must be one of {', '.join(DAYS)}"
        if "meal_type" in data or not partial:
            if data.get("meal_type") not in MEAL_TYPES:
                return f"meal_type must be one of {', '.join(MEAL_TYPES)}"
        if "calories" in data or not partial:
            if not _is_number(data.get("calories")) or data["calories"] < 0:
                return "calories is required and must be a non-negative number"
            if isinstance(data["calories"], float) and not data["calories"].is_integer():
                return "calories must be a whole number"
        if "ingredients" in data:
            err = _validate_ingredients(data["ingredients"])
            if err:
                return err
        return None

    @app.post("/meals")
    def create_meal():
        data = request.get_json(silent=True)
        err = _validate_meal(data)
        if err:
            return _error(err, 400)
        meal_id = app.config["NEXT_ID"]
        app.config["NEXT_ID"] += 1
        meal = {
            "id": meal_id,
            "name": data["name"].strip(),
            "day": data["day"],
            "meal_type": data["meal_type"],
            "calories": int(data["calories"]),
            "ingredients": [
                {
                    "name": i["name"].strip(),
                    "quantity": i["quantity"],
                    "unit": i["unit"].strip(),
                }
                for i in data.get("ingredients", [])
            ],
        }
        app.config["MEALS"][meal_id] = meal
        return jsonify(meal), 201

    @app.get("/meals")
    def list_meals():
        meals = sorted(app.config["MEALS"].values(), key=lambda m: m["id"])
        day = request.args.get("day")
        if day is not None:
            if day not in DAYS:
                return _error(f"day must be one of {', '.join(DAYS)}", 400)
            meals = [m for m in meals if m["day"] == day]
        meal_type = request.args.get("meal_type")
        if meal_type is not None:
            if meal_type not in MEAL_TYPES:
                return _error(f"meal_type must be one of {', '.join(MEAL_TYPES)}", 400)
            meals = [m for m in meals if m["meal_type"] == meal_type]
        return jsonify(meals)

    @app.get("/meals/<int:meal_id>")
    def get_meal(meal_id):
        meal, err = _get_meal(meal_id)
        if err is not None:
            return err
        return jsonify(meal)

    @app.put("/meals/<int:meal_id>")
    def update_meal(meal_id):
        meal, err = _get_meal(meal_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        fields = ("name", "day", "meal_type", "calories", "ingredients")
        if not any(k in data for k in fields):
            return _error(
                "nothing to update: provide name, day, meal_type, "
                "calories and/or ingredients",
                400,
            )
        err = _validate_meal(data, partial=True)
        if err:
            return _error(err, 400)
        if "name" in data:
            meal["name"] = data["name"].strip()
        if "day" in data:
            meal["day"] = data["day"]
        if "meal_type" in data:
            meal["meal_type"] = data["meal_type"]
        if "calories" in data:
            meal["calories"] = int(data["calories"])
        if "ingredients" in data:
            meal["ingredients"] = [
                {
                    "name": i["name"].strip(),
                    "quantity": i["quantity"],
                    "unit": i["unit"].strip(),
                }
                for i in data["ingredients"]
            ]
        return jsonify(meal)

    @app.delete("/meals/<int:meal_id>")
    def delete_meal(meal_id):
        _meal, err = _get_meal(meal_id)
        if err is not None:
            return err
        del app.config["MEALS"][meal_id]
        return jsonify({"deleted": meal_id})

    @app.get("/plan/<day>")
    def day_plan(day):
        if day not in DAYS:
            return _error(f"day must be one of {', '.join(DAYS)}", 400)
        meals = [m for m in app.config["MEALS"].values() if m["day"] == day]
        meals.sort(key=lambda m: (MEAL_TYPES.index(m["meal_type"]), m["id"]))
        return jsonify(
            {
                "day": day,
                "meals": meals,
                "meal_count": len(meals),
                "total_calories": sum(m["calories"] for m in meals),
            }
        )

    @app.get("/plan/week")
    def week_plan():
        days = {}
        week_total = 0
        for day in DAYS:
            meals = [m for m in app.config["MEALS"].values() if m["day"] == day]
            meals.sort(key=lambda m: (MEAL_TYPES.index(m["meal_type"]), m["id"]))
            total = sum(m["calories"] for m in meals)
            week_total += total
            days[day] = {
                "meals": meals,
                "meal_count": len(meals),
                "total_calories": total,
            }
        return jsonify({"days": days, "week_total_calories": week_total})

    @app.get("/calories")
    def calories():
        day = request.args.get("day")
        meals = list(app.config["MEALS"].values())
        if day is not None:
            if day not in DAYS:
                return _error(f"day must be one of {', '.join(DAYS)}", 400)
            meals = [m for m in meals if m["day"] == day]
        return jsonify(
            {
                "day": day,
                "meal_count": len(meals),
                "total_calories": sum(m["calories"] for m in meals),
            }
        )

    @app.get("/shopping-list")
    def shopping_list():
        aggregated = {}
        for meal in app.config["MEALS"].values():
            for item in meal["ingredients"]:
                key = (item["name"].lower(), item["unit"].lower())
                if key not in aggregated:
                    aggregated[key] = {
                        "name": item["name"],
                        "unit": item["unit"],
                        "quantity": 0,
                        "meals": 0,
                    }
                aggregated[key]["quantity"] += item["quantity"]
                aggregated[key]["meals"] += 1
        items = sorted(aggregated.values(), key=lambda i: i["name"].lower())
        return jsonify({"items": items, "item_count": len(items)})

    @app.errorhandler(404)
    def handle_404(exc):
        return _error("route not found", 404)

    return app
