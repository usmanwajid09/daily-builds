"""Flask personal finance API.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Endpoints:

- ``POST /transactions`` — record a transaction (``{\"description\": str,
  \"amount\": number, \"type\": \"income\" | \"expense\",
  \"category\": str, \"date\": \"YYYY-MM-DD\"}``)
- ``GET /transactions`` — list transactions; ``?type=``, ``?category=``
  and ``?month=YYYY-MM`` filters
- ``GET /transactions/<id>`` — fetch one transaction
- ``PUT /transactions/<id>`` — update description, amount, type,
  category and/or date
- ``DELETE /transactions/<id>`` — delete a transaction
- ``GET /balance`` — total income, total expense and balance
- ``GET /categories`` — per-category income/expense totals
- ``GET /summary?month=YYYY-MM`` — income, expense, balance and
  per-category totals (whole history when ``month`` is omitted)
- ``POST /budgets`` — set a budget (``{\"category\": str, \"limit\": number}``)
- ``GET /budgets`` — list budgets
- ``PUT /budgets/<category>`` — change a budget's limit
- ``DELETE /budgets/<category>`` — remove a budget
- ``GET /budgets/alerts`` — per-budget spend vs limit: ``ok`` (under
  80%), ``warning`` (80–100%), ``over`` (at or above the limit)

A transaction is ``{\"id\": int, \"description\": str, \"amount\": float,
\"type\": \"income\" | \"expense\", \"category\": str, \"date\": str}``.
A budget is ``{\"category\": str, \"limit\": float}``. Amounts are
stored rounded to two decimals; ``amount`` must be positive.
"""
import re
from datetime import datetime

from flask import Flask, jsonify, request

TYPES = ("income", "expense")
_MONTH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


def _valid_date(value):
    if not isinstance(value, str):
        return False
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return False
    return True


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def create_app():
    """Return a Flask app with a fresh, empty in-memory finance store."""
    app = Flask(__name__)
    app.config["TRANSACTIONS"] = {}
    app.config["BUDGETS"] = {}
    app.config["NEXT_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _get_txn(txn_id):
        txn = app.config["TRANSACTIONS"].get(txn_id)
        if txn is None:
            return None, _error(f"transaction {txn_id} not found", 404)
        return txn, None

    def _validate_txn(data, partial=False):
        if not isinstance(data, dict):
            return "request body must be a JSON object"
        if "description" in data or not partial:
            if not isinstance(data.get("description"), str) \
                    or not data["description"].strip():
                return "description is required and must be a non-empty string"
        if "amount" in data or not partial:
            amount = data.get("amount")
            if not _is_number(amount) or amount <= 0:
                return "amount is required and must be a positive number"
        if "type" in data or not partial:
            if data.get("type") not in TYPES:
                return f"type must be one of {', '.join(TYPES)}"
        if "category" in data or not partial:
            if not isinstance(data.get("category"), str) \
                    or not data["category"].strip():
                return "category is required and must be a non-empty string"
        if "date" in data or not partial:
            if not _valid_date(data.get("date")):
                return "date is required and must be YYYY-MM-DD"
        return None

    def _totals(txns):
        income = round(sum(t["amount"] for t in txns if t["type"] == "income"), 2)
        expense = round(sum(t["amount"] for t in txns if t["type"] == "expense"), 2)
        return income, expense, round(income - expense, 2)

    def _by_category(txns):
        result = {}
        for txn in txns:
            entry = result.setdefault(
                txn["category"], {"income": 0.0, "expense": 0.0, "count": 0}
            )
            entry[txn["type"]] = round(entry[txn["type"]] + txn["amount"], 2)
            entry["count"] += 1
        return result

    @app.post("/transactions")
    def create_txn():
        data = request.get_json(silent=True)
        err = _validate_txn(data)
        if err:
            return _error(err, 400)
        txn_id = app.config["NEXT_ID"]
        app.config["NEXT_ID"] += 1
        txn = {
            "id": txn_id,
            "description": data["description"].strip(),
            "amount": round(float(data["amount"]), 2),
            "type": data["type"],
            "category": data["category"].strip(),
            "date": data["date"],
        }
        app.config["TRANSACTIONS"][txn_id] = txn
        return jsonify(txn), 201

    @app.get("/transactions")
    def list_txns():
        txns = sorted(app.config["TRANSACTIONS"].values(), key=lambda t: t["id"])
        txn_type = request.args.get("type")
        if txn_type is not None:
            if txn_type not in TYPES:
                return _error(f"type must be one of {', '.join(TYPES)}", 400)
            txns = [t for t in txns if t["type"] == txn_type]
        category = request.args.get("category")
        if category:
            txns = [t for t in txns if t["category"].lower() == category.lower()]
        month = request.args.get("month")
        if month is not None:
            if not _MONTH_RE.match(month):
                return _error("month must be YYYY-MM", 400)
            txns = [t for t in txns if t["date"].startswith(month)]
        return jsonify(txns)

    @app.get("/transactions/<int:txn_id>")
    def get_txn(txn_id):
        txn, err = _get_txn(txn_id)
        if err is not None:
            return err
        return jsonify(txn)

    @app.put("/transactions/<int:txn_id>")
    def update_txn(txn_id):
        txn, err = _get_txn(txn_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        fields = ("description", "amount", "type", "category", "date")
        if not any(k in data for k in fields):
            return _error(
                "nothing to update: provide description, amount, type, "
                "category and/or date",
                400,
            )
        err = _validate_txn(data, partial=True)
        if err:
            return _error(err, 400)
        if "description" in data:
            txn["description"] = data["description"].strip()
        if "amount" in data:
            txn["amount"] = round(float(data["amount"]), 2)
        if "type" in data:
            txn["type"] = data["type"]
        if "category" in data:
            txn["category"] = data["category"].strip()
        if "date" in data:
            txn["date"] = data["date"]
        return jsonify(txn)

    @app.delete("/transactions/<int:txn_id>")
    def delete_txn(txn_id):
        _txn, err = _get_txn(txn_id)
        if err is not None:
            return err
        del app.config["TRANSACTIONS"][txn_id]
        return jsonify({"deleted": txn_id})

    @app.get("/balance")
    def balance():
        income, expense, net = _totals(app.config["TRANSACTIONS"].values())
        return jsonify(
            {"total_income": income, "total_expense": expense, "balance": net}
        )

    @app.get("/categories")
    def categories():
        return jsonify(_by_category(app.config["TRANSACTIONS"].values()))

    @app.get("/summary")
    def summary():
        month = request.args.get("month")
        txns = list(app.config["TRANSACTIONS"].values())
        if month is not None:
            if not _MONTH_RE.match(month):
                return _error("month must be YYYY-MM", 400)
            txns = [t for t in txns if t["date"].startswith(month)]
        income, expense, net = _totals(txns)
        return jsonify(
            {
                "month": month,
                "transaction_count": len(txns),
                "total_income": income,
                "total_expense": expense,
                "balance": net,
                "by_category": _by_category(txns),
            }
        )

    @app.post("/budgets")
    def create_budget():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        category = data.get("category")
        if not isinstance(category, str) or not category.strip():
            return _error("category is required and must be a non-empty string", 400)
        limit = data.get("limit")
        if not _is_number(limit) or limit <= 0:
            return _error("limit is required and must be a positive number", 400)
        category = category.strip()
        if category in app.config["BUDGETS"]:
            return _error(
                f"budget for category '{category}' already exists; "
                "use PUT to change it",
                400,
            )
        budget = {"category": category, "limit": round(float(limit), 2)}
        app.config["BUDGETS"][category] = budget
        return jsonify(budget), 201

    @app.get("/budgets")
    def list_budgets():
        budgets = sorted(app.config["BUDGETS"].values(),
                         key=lambda b: b["category"].lower())
        return jsonify(budgets)

    @app.put("/budgets/<category>")
    def update_budget(category):
        budget = app.config["BUDGETS"].get(category)
        if budget is None:
            return _error(f"no budget for category '{category}'", 404)
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        limit = data.get("limit")
        if not _is_number(limit) or limit <= 0:
            return _error("limit is required and must be a positive number", 400)
        budget["limit"] = round(float(limit), 2)
        return jsonify(budget)

    @app.delete("/budgets/<category>")
    def delete_budget(category):
        if category not in app.config["BUDGETS"]:
            return _error(f"no budget for category '{category}'", 404)
        del app.config["BUDGETS"][category]
        return jsonify({"deleted": category})

    @app.get("/budgets/alerts")
    def budget_alerts():
        txns = app.config["TRANSACTIONS"].values()
        spent = {}
        for txn in txns:
            if txn["type"] == "expense":
                spent[txn["category"]] = spent.get(txn["category"], 0.0) + txn["amount"]
        alerts = []
        for budget in sorted(app.config["BUDGETS"].values(),
                             key=lambda b: b["category"].lower()):
            used = round(spent.get(budget["category"], 0.0), 2)
            pct = round(used / budget["limit"] * 100, 1) if budget["limit"] else 0.0
            if used >= budget["limit"]:
                status = "over"
            elif used >= budget["limit"] * 0.8:
                status = "warning"
            else:
                status = "ok"
            alerts.append(
                {
                    "category": budget["category"],
                    "limit": budget["limit"],
                    "spent": used,
                    "remaining": round(budget["limit"] - used, 2),
                    "pct_used": pct,
                    "status": status,
                }
            )
        return jsonify(alerts)

    @app.errorhandler(404)
    def handle_404(exc):
        return _error("route not found", 404)

    return app
