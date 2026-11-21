"""Flask subscription tracker API.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Endpoints:

- ``POST /subscriptions`` — add a subscription (``{"name": str,
  "cost": number > 0, "billing_cycle": "monthly" | "yearly",
  "renewal_date": "YYYY-MM-DD", "category": str?}``). A second
  subscription with the same name (case-insensitive) is refused
  with 409.
- ``GET /subscriptions`` — list subscriptions; ``?category=`` and
  ``?cycle=monthly|yearly`` filters
- ``GET /subscriptions/<id>`` — fetch one subscription
- ``PUT /subscriptions/<id>`` — update name, cost, billing_cycle,
  renewal_date and/or category
- ``DELETE /subscriptions/<id>`` — delete a subscription
- ``GET /totals`` — monthly and yearly totals (yearly plans count as
  cost/12 per month, monthly plans as cost*12 per year), count, and
  a monthly breakdown by category
- ``GET /upcoming`` — renewals due within ``?days=`` (default 30)
  from ``?as_of=YYYY-MM-DD`` (default today); each row carries its
  next renewal date and days until it

A subscription is ``{"id": int, "name": str, "cost": float,
"billing_cycle": str, "renewal_date": str, "category": str | None,
"monthly_cost": float, "yearly_cost": float}``.
"""
import calendar
from datetime import date, datetime, timedelta

from flask import Flask, jsonify, request

CYCLES = ("monthly", "yearly")


def _valid_date(value):
    if not isinstance(value, str):
        return False
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return False
    return True


def _parse_date(value):
    return datetime.strptime(value, "%Y-%m-%d").date()


def _add_months(day, months):
    month = day.month - 1 + months
    year = day.year + month // 12
    month = month % 12 + 1
    last = calendar.monthrange(year, month)[1]
    return date(year, month, min(day.day, last))


def _next_renewal(sub, as_of):
    """First renewal date on or after ``as_of``."""
    renewal = _parse_date(sub["renewal_date"])
    step = 1 if sub["billing_cycle"] == "monthly" else 12
    while renewal < as_of:
        renewal = _add_months(renewal, step)
    return renewal


def create_app():
    """Return a Flask app with a fresh, empty in-memory store."""
    app = Flask(__name__)
    app.config["SUBSCRIPTIONS"] = {}
    app.config["NEXT_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _public(sub):
        cost = sub["cost"]
        if sub["billing_cycle"] == "monthly":
            monthly, yearly = cost, round(cost * 12, 2)
        else:
            monthly, yearly = round(cost / 12, 2), cost
        return {
            "id": sub["id"],
            "name": sub["name"],
            "cost": cost,
            "billing_cycle": sub["billing_cycle"],
            "renewal_date": sub["renewal_date"],
            "category": sub["category"],
            "monthly_cost": monthly,
            "yearly_cost": yearly,
        }

    def _get_sub(sub_id):
        sub = app.config["SUBSCRIPTIONS"].get(sub_id)
        if sub is None:
            return None, _error(f"subscription {sub_id} not found", 404)
        return sub, None

    def _validate(data, partial=False):
        cleaned = {}
        if not partial or "name" in data:
            name = data.get("name")
            if not isinstance(name, str) or not name.strip():
                return None, _error(
                    "name is required and must be a non-empty string", 400
                )
            cleaned["name"] = name.strip()
        if not partial or "cost" in data:
            cost = data.get("cost")
            if isinstance(cost, bool) or not isinstance(cost, (int, float)) \
                    or cost <= 0:
                return None, _error("cost is required and must be a positive number", 400)
            cleaned["cost"] = round(float(cost), 2)
        if not partial or "billing_cycle" in data:
            cycle = data.get("billing_cycle")
            if cycle not in CYCLES:
                return None, _error(
                    "billing_cycle is required and must be 'monthly' or 'yearly'",
                    400,
                )
            cleaned["billing_cycle"] = cycle
        if not partial or "renewal_date" in data:
            renewal_date = data.get("renewal_date")
            if not _valid_date(renewal_date):
                return None, _error(
                    "renewal_date is required and must be YYYY-MM-DD", 400
                )
            cleaned["renewal_date"] = renewal_date
        if "category" in data:
            category = data["category"]
            if category is not None and (not isinstance(category, str)
                                         or not category.strip()):
                return None, _error("category must be a non-empty string", 400)
            cleaned["category"] = category.strip() if category else None
        return cleaned, None

    def _duplicate_name(name, exclude_id=None):
        return any(
            s["name"].lower() == name.lower() and s["id"] != exclude_id
            for s in app.config["SUBSCRIPTIONS"].values()
        )

    @app.post("/subscriptions")
    def create_sub():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        cleaned, err = _validate(data)
        if err is not None:
            return err
        if _duplicate_name(cleaned["name"]):
            return _error(
                f"a subscription named {cleaned['name']!r} already exists", 409
            )
        sub_id = app.config["NEXT_ID"]
        app.config["NEXT_ID"] += 1
        sub = {
            "id": sub_id,
            "name": cleaned["name"],
            "cost": cleaned["cost"],
            "billing_cycle": cleaned["billing_cycle"],
            "renewal_date": cleaned["renewal_date"],
            "category": cleaned.get("category"),
        }
        app.config["SUBSCRIPTIONS"][sub_id] = sub
        return jsonify(_public(sub)), 201

    @app.get("/subscriptions")
    def list_subs():
        subs = sorted(
            app.config["SUBSCRIPTIONS"].values(), key=lambda s: s["id"]
        )
        category = request.args.get("category")
        if category is not None:
            subs = [s for s in subs
                    if (s["category"] or "").lower() == category.lower()]
        cycle = request.args.get("cycle")
        if cycle is not None:
            if cycle not in CYCLES:
                return _error("cycle must be 'monthly' or 'yearly'", 400)
            subs = [s for s in subs if s["billing_cycle"] == cycle]
        return jsonify([_public(s) for s in subs])

    @app.get("/subscriptions/<int:sub_id>")
    def get_sub(sub_id):
        sub, err = _get_sub(sub_id)
        if err is not None:
            return err
        return jsonify(_public(sub))

    @app.put("/subscriptions/<int:sub_id>")
    def update_sub(sub_id):
        sub, err = _get_sub(sub_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        fields = ("name", "cost", "billing_cycle", "renewal_date", "category")
        if not any(k in data for k in fields):
            return _error(
                "nothing to update: provide name, cost, billing_cycle, "
                "renewal_date and/or category",
                400,
            )
        cleaned, err = _validate(data, partial=True)
        if err is not None:
            return err
        if "name" in cleaned and _duplicate_name(cleaned["name"],
                                                  exclude_id=sub_id):
            return _error(
                f"a subscription named {cleaned['name']!r} already exists", 409
            )
        sub.update(cleaned)
        return jsonify(_public(sub))

    @app.delete("/subscriptions/<int:sub_id>")
    def delete_sub(sub_id):
        _sub, err = _get_sub(sub_id)
        if err is not None:
            return err
        del app.config["SUBSCRIPTIONS"][sub_id]
        return jsonify({"deleted": sub_id})

    @app.get("/totals")
    def totals():
        subs = list(app.config["SUBSCRIPTIONS"].values())
        monthly = sum(_public(s)["monthly_cost"] for s in subs)
        yearly = sum(_public(s)["yearly_cost"] for s in subs)
        by_category = {}
        for sub in subs:
            key = sub["category"] or "uncategorized"
            by_category[key] = round(
                by_category.get(key, 0) + _public(sub)["monthly_cost"], 2
            )
        return jsonify(
            {
                "count": len(subs),
                "monthly_total": round(monthly, 2),
                "yearly_total": round(yearly, 2),
                "by_category_monthly": by_category,
            }
        )

    @app.get("/upcoming")
    def upcoming():
        as_of_raw = request.args.get("as_of")
        if as_of_raw is None:
            as_of = date.today()
        elif _valid_date(as_of_raw):
            as_of = _parse_date(as_of_raw)
        else:
            return _error("as_of must be YYYY-MM-DD", 400)
        days_raw = request.args.get("days", "30")
        try:
            days = int(days_raw)
        except ValueError:
            return _error("days must be a non-negative integer", 400)
        if days < 0:
            return _error("days must be a non-negative integer", 400)
        rows = []
        for sub in app.config["SUBSCRIPTIONS"].values():
            nxt = _next_renewal(sub, as_of)
            delta = (nxt - as_of).days
            if delta <= days:
                row = _public(sub)
                row["next_renewal"] = nxt.isoformat()
                row["days_until_renewal"] = delta
                rows.append(row)
        rows.sort(key=lambda r: r["next_renewal"])
        return jsonify(rows)

    @app.errorhandler(404)
    def handle_404(exc):
        return _error("route not found", 404)

    return app
