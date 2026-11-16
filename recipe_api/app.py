"""REST recipe API built with Flask.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Endpoints:

- ``GET /recipes`` — list all recipes (``?ingredient=garlic`` filters to
  recipes containing that ingredient)
- ``POST /recipes`` — create a recipe
- ``GET /recipes/<id>`` — fetch one recipe
- ``PUT /recipes/<id>`` — update a recipe
- ``DELETE /recipes/<id>`` — delete a recipe

A recipe is ``{"id": int, "title": str, "ingredients": [str, ...],
"steps": [str, ...], "servings": int}``.
"""
from flask import Flask, jsonify, request


def create_app():
    """Return a Flask app with a fresh, empty in-memory recipe store."""
    app = Flask(__name__)
    app.config["RECIPES"] = {}
    app.config["NEXT_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _validate_recipe(data):
        """Validate a recipe payload. Returns (cleaned, None) or (None, error)."""
        if not isinstance(data, dict):
            return None, _error("request body must be a JSON object", 400)
        title = data.get("title")
        if not isinstance(title, str) or not title.strip():
            return None, _error("title is required and must be a non-empty string", 400)
        ingredients = data.get("ingredients")
        if (
            not isinstance(ingredients, list)
            or not ingredients
            or any(not isinstance(i, str) or not i.strip() for i in ingredients)
        ):
            return None, _error(
                "ingredients is required and must be a non-empty list of non-empty strings",
                400,
            )
        steps = data.get("steps")
        if (
            not isinstance(steps, list)
            or not steps
            or any(not isinstance(s, str) or not s.strip() for s in steps)
        ):
            return None, _error(
                "steps is required and must be a non-empty list of non-empty strings",
                400,
            )
        servings = data.get("servings")
        if not isinstance(servings, int) or isinstance(servings, bool) or servings < 1:
            return None, _error("servings is required and must be a positive integer", 400)
        cleaned = {
            "title": title.strip(),
            "ingredients": [i.strip() for i in ingredients],
            "steps": [s.strip() for s in steps],
            "servings": servings,
        }
        return cleaned, None

    def _get_recipe(recipe_id):
        recipe = app.config["RECIPES"].get(recipe_id)
        if recipe is None:
            return None, _error(f"recipe {recipe_id} not found", 404)
        return recipe, None

    @app.get("/recipes")
    def list_recipes():
        recipes = sorted(app.config["RECIPES"].values(), key=lambda r: r["id"])
        ingredient = request.args.get("ingredient")
        if ingredient:
            needle = ingredient.strip().lower()
            recipes = [
                r
                for r in recipes
                if any(needle in ing.lower() for ing in r["ingredients"])
            ]
        return jsonify(recipes)

    @app.post("/recipes")
    def create_recipe():
        cleaned, err = _validate_recipe(request.get_json(silent=True))
        if err is not None:
            return err
        recipe_id = app.config["NEXT_ID"]
        app.config["NEXT_ID"] += 1
        recipe = {"id": recipe_id, **cleaned}
        app.config["RECIPES"][recipe_id] = recipe
        return jsonify(recipe), 201

    @app.get("/recipes/<int:recipe_id>")
    def get_recipe(recipe_id):
        recipe, err = _get_recipe(recipe_id)
        if err is not None:
            return err
        return jsonify(recipe)

    @app.put("/recipes/<int:recipe_id>")
    def update_recipe(recipe_id):
        recipe, err = _get_recipe(recipe_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or not data:
            return _error("request body must be a non-empty JSON object", 400)
        merged = {
            "title": data.get("title", recipe["title"]),
            "ingredients": data.get("ingredients", recipe["ingredients"]),
            "steps": data.get("steps", recipe["steps"]),
            "servings": data.get("servings", recipe["servings"]),
        }
        cleaned, err = _validate_recipe(merged)
        if err is not None:
            return err
        recipe.update(cleaned)
        return jsonify(recipe)

    @app.delete("/recipes/<int:recipe_id>")
    def delete_recipe(recipe_id):
        _recipe, err = _get_recipe(recipe_id)
        if err is not None:
            return err
        del app.config["RECIPES"][recipe_id]
        return jsonify({"deleted": recipe_id})

    @app.errorhandler(404)
    def handle_404(exc):
        return _error("route not found", 404)

    return app
