"""REST monthly-budget tracker API built with Flask.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Endpoints:

- ``GET /budgets`` — list every category with spent, remaining and
  an over-budget flag
- ``POST /budgets`` — set a category budget (``{"category": str, "amount": number}``)
- ``GET /budgets/<category>`` — one category's budget summary
- ``DELETE /budgets/<category>`` — remove a budget and its expenses
- ``POST /expenses`` — add an expense (``{"category": str, "amount": number, "note": str?}``)
- ``GET /expenses`` — list all expenses
- ``GET /summary`` — totals across all categories
"""
from flask import Flask, jsonify, request


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def create_app():
    """Return a Flask app with a fresh, empty in-memory budget store."""
    app = Flask(__name__)
    app.config["BUDGETS"] = {}   # category -> {"category", "amount"}
    app.config["EXPENSES"] = []  # list of {"id", "category", "amount", "note"}
    app.config["NEXT_EXPENSE_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _summary(category):
        budget = app.config["BUDGETS"][category]
        spent = sum(
            e["amount"] for e in app.config["EXPENSES"] if e["category"] == category
        )
        remaining = round(budget["amount"] - spent, 2)
        return {
            "category": category,
            "budget": budget["amount"],
            "spent": round(spent, 2),
            "remaining": remaining,
            "over_budget": remaining < 0,
        }

    @app.get("/budgets")
    def list_budgets():
        return jsonify([_summary(c) for c in sorted(app.config["BUDGETS"])])

    @app.post("/budgets")
    def set_budget():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        category = data.get("category")
        if not isinstance(category, str) or not category.strip():
            return _error("category is required and must be a non-empty string", 400)
        amount = data.get("amount")
        if not _is_number(amount) or amount <= 0:
            return _error("amount is required and must be a positive number", 400)
        category = category.strip()
        app.config["BUDGETS"][category] = {"category": category, "amount": amount}
        return jsonify(_summary(category)), 201

    @app.get("/budgets/<category>")
    def get_budget(category):
        if category not in app.config["BUDGETS"]:
            return _error(f"no budget set for category {category!r}", 404)
        return jsonify(_summary(category))

    @app.delete("/budgets/<category>")
    def delete_budget(category):
        if category not in app.config["BUDGETS"]:
            return _error(f"no budget set for category {category!r}", 404)
        del app.config["BUDGETS"][category]
        app.config["EXPENSES"] = [
            e for e in app.config["EXPENSES"] if e["category"] != category
        ]
        return jsonify({"deleted": category})

    @app.post("/expenses")
    def add_expense():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        category = data.get("category")
        if not isinstance(category, str) or not category.strip():
            return _error("category is required and must be a non-empty string", 400)
        category = category.strip()
        if category not in app.config["BUDGETS"]:
            return _error(f"no budget set for category {category!r}", 400)
        amount = data.get("amount")
        if not _is_number(amount) or amount <= 0:
            return _error("amount is required and must be a positive number", 400)
        note = data.get("note", "")
        if not isinstance(note, str):
            return _error("note must be a string", 400)
        expense_id = app.config["NEXT_EXPENSE_ID"]
        app.config["NEXT_EXPENSE_ID"] += 1
        expense = {
            "id": expense_id,
            "category": category,
            "amount": amount,
            "note": note,
        }
        app.config["EXPENSES"].append(expense)
        return jsonify(expense), 201

    @app.get("/expenses")
    def list_expenses():
        category = request.args.get("category")
        expenses = app.config["EXPENSES"]
        if category:
            expenses = [e for e in expenses if e["category"] == category]
        return jsonify(expenses)

    @app.get("/summary")
    def summary():
        summaries = [_summary(c) for c in app.config["BUDGETS"]]
        total_budget = round(sum(s["budget"] for s in summaries), 2)
        total_spent = round(sum(s["spent"] for s in summaries), 2)
        return jsonify(
            {
                "total_budget": total_budget,
                "total_spent": total_spent,
                "total_remaining": round(total_budget - total_spent, 2),
                "over_budget": [s["category"] for s in summaries if s["over_budget"]],
                "categories": summaries,
            }
        )

    return app
