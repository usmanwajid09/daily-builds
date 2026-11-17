"""Inventory API built with Flask.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Endpoints:

- ``POST /items`` — add an item
  (``{"name": str, "stock": int >= 0, "threshold": int >= 0 (default 5),
  "price": number >= 0 (optional)}``)
- ``GET /items`` — list all items
- ``GET /items/<id>`` — fetch one item
- ``PATCH /items/<id>/stock`` — adjust stock
  (``{"delta": int}``; negative deltas remove stock but stock can never
  go below 0)
- ``GET /items/low-stock`` — items at or below their threshold
  (``?threshold=N`` overrides each item's own threshold)
- ``DELETE /items/<id>`` — delete an item

Stock is never negative: creating an item with negative stock or a
removal larger than the current stock both return 400.
"""
from flask import Flask, jsonify, request


def create_app():
    """Return a Flask app with a fresh, empty in-memory inventory."""
    app = Flask(__name__)
    app.config["ITEMS"] = {}
    app.config["NEXT_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _is_int(value):
        return isinstance(value, int) and not isinstance(value, bool)

    def _get_item(item_id):
        item = app.config["ITEMS"].get(item_id)
        if item is None:
            return None, _error(f"item {item_id} not found", 404)
        return item, None

    @app.post("/items")
    def create_item():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            return _error("name is required and must be a non-empty string", 400)
        stock = data.get("stock")
        if not _is_int(stock) or stock < 0:
            return _error("stock is required and must be an integer >= 0", 400)
        threshold = data.get("threshold", 5)
        if not _is_int(threshold) or threshold < 0:
            return _error("threshold must be an integer >= 0", 400)
        price = data.get("price")
        if price is not None:
            if not isinstance(price, (int, float)) or isinstance(price, bool) or price < 0:
                return _error("price must be a number >= 0", 400)
        item_id = app.config["NEXT_ID"]
        app.config["NEXT_ID"] += 1
        item = {
            "id": item_id,
            "name": name.strip(),
            "stock": stock,
            "threshold": threshold,
            "price": price,
            "low_stock": stock <= threshold,
        }
        app.config["ITEMS"][item_id] = item
        return jsonify(item), 201

    @app.get("/items")
    def list_items():
        items = sorted(app.config["ITEMS"].values(), key=lambda i: i["id"])
        return jsonify(items)

    @app.get("/items/low-stock")
    def low_stock():
        threshold = request.args.get("threshold")
        if threshold is not None:
            try:
                limit = int(threshold)
            except ValueError:
                return _error("threshold must be an integer", 400)
            if limit < 0:
                return _error("threshold must be >= 0", 400)
            items = [i for i in app.config["ITEMS"].values() if i["stock"] <= limit]
        else:
            items = [i for i in app.config["ITEMS"].values() if i["low_stock"]]
        return jsonify(sorted(items, key=lambda i: i["id"]))

    @app.get("/items/<int:item_id>")
    def get_item(item_id):
        item, err = _get_item(item_id)
        if err is not None:
            return err
        return jsonify(item)

    @app.patch("/items/<int:item_id>/stock")
    def adjust_stock(item_id):
        item, err = _get_item(item_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        delta = data.get("delta")
        if not _is_int(delta):
            return _error("delta is required and must be an integer", 400)
        new_stock = item["stock"] + delta
        if new_stock < 0:
            return _error(
                f"cannot remove {abs(delta)} units: only {item['stock']} in stock",
                400,
            )
        item["stock"] = new_stock
        item["low_stock"] = new_stock <= item["threshold"]
        return jsonify(item)

    @app.delete("/items/<int:item_id>")
    def delete_item(item_id):
        item, err = _get_item(item_id)
        if err is not None:
            return err
        del app.config["ITEMS"][item_id]
        return jsonify({"deleted": item_id})

    return app
