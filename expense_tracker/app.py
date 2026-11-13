"""Expense tracker API built with Flask.

In-memory store, fresh for each app instance. Endpoints:

- ``POST /expenses`` — add one: ``{"amount", "category", "date",
  "description"?}``
- ``GET /expenses`` — list all (``?category=food`` filters)
- ``GET /expenses/<id>`` — fetch one expense
- ``DELETE /expenses/<id>`` — delete one expense
- ``GET /totals`` — grand total plus totals per category
- ``GET /totals/<YYYY-MM>`` — total spent in one month
"""
from datetime import datetime

from flask import Flask, jsonify, request


def _valid_date(value):
    """Return a datetime for a ``YYYY-MM-DD`` string, else None."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return None
    if parsed.strftime("%Y-%m-%d") != value:
        return None
    return parsed


def create_app():
    """Return a Flask app with a fresh, empty in-memory expense store."""
    app = Flask(__name__)
    app.config["EXPENSES"] = {}
    app.config["NEXT_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    @app.post("/expenses")
    def add_expense():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        amount = data.get("amount")
        if isinstance(amount, bool) or not isinstance(amount, (int, float)):
            return _error("amount is required and must be a number", 400)
        if amount <= 0:
            return _error("amount must be greater than 0", 400)
        category = data.get("category")
        if not isinstance(category, str) or not category.strip():
            return _error("category is required and must be a non-empty string", 400)
        date = _valid_date(data.get("date"))
        if date is None:
            return _error("date is required and must use YYYY-MM-DD format", 400)
        description = data.get("description", "")
        if not isinstance(description, str):
            return _error("description must be a string", 400)
        expense_id = app.config["NEXT_ID"]
        app.config["NEXT_ID"] += 1
        expense = {
            "id": expense_id,
            "amount": round(float(amount), 2),
            "category": category.strip(),
            "date": data["date"],
            "description": description.strip(),
        }
        app.config["EXPENSES"][expense_id] = expense
        return jsonify(expense), 201

    @app.get("/expenses")
    def list_expenses():
        expenses = sorted(
            app.config["EXPENSES"].values(), key=lambda e: (e["date"], e["id"])
        )
        category = request.args.get("category")
        if category:
            expenses = [e for e in expenses if e["category"] == category]
        return jsonify(expenses)

    @app.get("/expenses/<int:expense_id>")
    def get_expense(expense_id):
        expense = app.config["EXPENSES"].get(expense_id)
        if expense is None:
            return _error(f"expense {expense_id} not found", 404)
        return jsonify(expense)

    @app.delete("/expenses/<int:expense_id>")
    def delete_expense(expense_id):
        if expense_id not in app.config["EXPENSES"]:
            return _error(f"expense {expense_id} not found", 404)
        del app.config["EXPENSES"][expense_id]
        return jsonify({"deleted": expense_id})

    @app.get("/totals")
    def totals():
        by_category = {}
        for expense in app.config["EXPENSES"].values():
            by_category[expense["category"]] = round(
                by_category.get(expense["category"], 0.0) + expense["amount"], 2
            )
        total = round(sum(by_category.values()), 2)
        return jsonify(
            {
                "total": total,
                "by_category": by_category,
                "count": len(app.config["EXPENSES"]),
            }
        )

    @app.get("/totals/<month>")
    def monthly_total(month):
        try:
            datetime.strptime(month, "%Y-%m")
        except ValueError:
            return _error("month must use YYYY-MM format", 400)
        spent = [
            e for e in app.config["EXPENSES"].values() if e["date"].startswith(month)
        ]
        return jsonify(
            {
                "month": month,
                "total": round(sum(e["amount"] for e in spent), 2),
                "count": len(spent),
            }
        )

    @app.errorhandler(404)
    def handle_404(exc):
        return _error("route not found", 404)

    return app
