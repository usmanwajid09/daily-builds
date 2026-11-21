"""Flask bill splitter API.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Model: the ``payer`` paid the whole
bill up front; every other participant owes their share to the payer,
and payments recorded here are reimbursements to the payer. The
payer's own share is their own cost, so they never owe anything.

Endpoints:

- ``POST /bills`` — create a bill (``{"title": str, "total": number
  > 0, "payer": str, "participants": [str, ...] (at least 2, must
  include the payer), "shares": {name: amount}?}``). Without
  ``shares`` the total is split equally (the last participant takes
  the rounding remainder so the shares sum exactly to the total).
  With ``shares`` the keys must match the participants exactly and
  the amounts must sum to the total.
- ``GET /bills`` — list bills
- ``GET /bills/<id>`` — fetch one bill, with per-person shares, paid
  amounts and balances
- ``PUT /bills/<id>`` — update title, total, payer, participants
  and/or shares (a participant who has already paid cannot be
  removed)
- ``DELETE /bills/<id>`` — delete a bill
- ``POST /bills/<id>/payments`` — record a reimbursement
  (``{"person": str, "amount": number > 0}``); paying more than the
  person's remaining share is refused with 400
- ``GET /bills/<id>/payments`` — list payments on a bill
- ``GET /bills/<id>/balances`` — per-person share, paid and amount
  still owed
- ``GET /bills/<id>/settlements`` — the minimal settlement plan:
  who pays how much to the payer
- ``GET /summary`` — bill count, total amount, total outstanding,
  and how many bills are fully settled

A bill is ``{"id": int, "title": str, "total": float, "payer": str,
"participants": [str], "shares": {name: float}, "payments": [...],
"balances": [...], "outstanding": float, "settled": bool}``.
"""
from flask import Flask, jsonify, request


def create_app():
    """Return a Flask app with a fresh, empty in-memory bill store."""
    app = Flask(__name__)
    app.config["BILLS"] = {}
    app.config["NEXT_BILL_ID"] = 1
    app.config["NEXT_PAYMENT_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _clean_names(value):
        """Validate a participant list: returns cleaned list or None."""
        if not isinstance(value, list) or len(value) < 2:
            return None
        cleaned = []
        for name in value:
            if not isinstance(name, str) or not name.strip():
                return None
            cleaned.append(name.strip())
        if len(set(cleaned)) != len(cleaned):
            return None
        return cleaned

    def _equal_shares(total, participants):
        per = round(total / len(participants), 2)
        shares = {name: per for name in participants}
        shares[participants[-1]] = round(
            total - per * (len(participants) - 1), 2
        )
        return shares

    def _check_shares(shares, participants, total):
        """Validate custom shares; returns (cleaned, error_response)."""
        if not isinstance(shares, dict):
            return None, _error("shares must be an object mapping name to amount", 400)
        if set(shares) != set(participants):
            return None, _error(
                "shares must cover exactly the participants", 400
            )
        cleaned = {}
        for name, amount in shares.items():
            if isinstance(amount, bool) or not isinstance(amount, (int, float)) \
                    or amount < 0:
                return None, _error(
                    f"share for {name!r} must be a number >= 0", 400
                )
            cleaned[name] = round(float(amount), 2)
        if abs(sum(cleaned.values()) - total) > 0.01:
            return None, _error(
                "shares must sum to the bill total", 400
            )
        return cleaned, None

    def _paid_by(bill):
        paid = {name: 0.0 for name in bill["participants"]}
        for payment in bill["payments"]:
            paid[payment["person"]] = round(
                paid[payment["person"]] + payment["amount"], 2
            )
        return paid

    def _balances(bill):
        paid = _paid_by(bill)
        rows = []
        for name in bill["participants"]:
            share = bill["shares"][name]
            owed = 0.0 if name == bill["payer"] else round(share - paid[name], 2)
            rows.append(
                {
                    "person": name,
                    "share": share,
                    "paid": paid[name],
                    "balance": owed,
                    "is_payer": name == bill["payer"],
                }
            )
        return rows

    def _public(bill):
        balances = _balances(bill)
        outstanding = round(
            sum(r["balance"] for r in balances), 2
        )
        return {
            "id": bill["id"],
            "title": bill["title"],
            "total": bill["total"],
            "payer": bill["payer"],
            "participants": list(bill["participants"]),
            "shares": dict(bill["shares"]),
            "payments": list(bill["payments"]),
            "balances": balances,
            "outstanding": outstanding,
            "settled": outstanding == 0,
        }

    def _get_bill(bill_id):
        bill = app.config["BILLS"].get(bill_id)
        if bill is None:
            return None, _error(f"bill {bill_id} not found", 404)
        return bill, None

    def _validate(data, partial=False):
        cleaned = {}
        if not partial or "title" in data:
            title = data.get("title")
            if not isinstance(title, str) or not title.strip():
                return None, _error(
                    "title is required and must be a non-empty string", 400
                )
            cleaned["title"] = title.strip()
        if not partial or "total" in data:
            total = data.get("total")
            if isinstance(total, bool) or not isinstance(total, (int, float)) \
                    or total <= 0:
                return None, _error("total is required and must be a positive number", 400)
            cleaned["total"] = round(float(total), 2)
        if not partial or "payer" in data:
            payer = data.get("payer")
            if not isinstance(payer, str) or not payer.strip():
                return None, _error(
                    "payer is required and must be a non-empty string", 400
                )
            cleaned["payer"] = payer.strip()
        if not partial or "participants" in data:
            participants = _clean_names(data.get("participants"))
            if participants is None:
                return None, _error(
                    "participants is required: at least 2 unique non-empty names",
                    400,
                )
            cleaned["participants"] = participants
        if "shares" in data:
            cleaned["shares"] = data["shares"]
        return cleaned, None

    @app.post("/bills")
    def create_bill():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        cleaned, err = _validate(data)
        if err is not None:
            return err
        if cleaned["payer"] not in cleaned["participants"]:
            return _error("payer must be one of the participants", 400)
        if "shares" in cleaned and cleaned["shares"] is not None:
            shares, err = _check_shares(
                cleaned["shares"], cleaned["participants"], cleaned["total"]
            )
            if err is not None:
                return err
        else:
            shares = _equal_shares(cleaned["total"], cleaned["participants"])
        bill_id = app.config["NEXT_BILL_ID"]
        app.config["NEXT_BILL_ID"] += 1
        bill = {
            "id": bill_id,
            "title": cleaned["title"],
            "total": cleaned["total"],
            "payer": cleaned["payer"],
            "participants": cleaned["participants"],
            "shares": shares,
            "payments": [],
        }
        app.config["BILLS"][bill_id] = bill
        return jsonify(_public(bill)), 201

    @app.get("/bills")
    def list_bills():
        bills = sorted(app.config["BILLS"].values(), key=lambda b: b["id"])
        return jsonify([_public(b) for b in bills])

    @app.get("/bills/<int:bill_id>")
    def get_bill(bill_id):
        bill, err = _get_bill(bill_id)
        if err is not None:
            return err
        return jsonify(_public(bill))

    @app.put("/bills/<int:bill_id>")
    def update_bill(bill_id):
        bill, err = _get_bill(bill_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        fields = ("title", "total", "payer", "participants", "shares")
        if not any(k in data for k in fields):
            return _error(
                "nothing to update: provide title, total, payer, "
                "participants and/or shares",
                400,
            )
        cleaned, err = _validate(data, partial=True)
        if err is not None:
            return err
        new_total = cleaned.get("total", bill["total"])
        new_payer = cleaned.get("payer", bill["payer"])
        new_participants = cleaned.get("participants", bill["participants"])
        if new_payer not in new_participants:
            return _error("payer must be one of the participants", 400)
        paid = _paid_by(bill)
        removed = set(bill["participants"]) - set(new_participants)
        if any(paid.get(name, 0) > 0 for name in removed):
            return _error(
                "cannot remove a participant who has already paid", 400
            )
        if "shares" in cleaned and cleaned["shares"] is not None:
            shares, err = _check_shares(
                cleaned["shares"], new_participants, new_total
            )
            if err is not None:
                return err
        elif "total" in cleaned or "participants" in cleaned:
            shares = _equal_shares(new_total, new_participants)
        else:
            shares = bill["shares"]
        if any(paid.get(name, 0) > shares.get(name, 0) + 0.01
               for name in new_participants):
            return _error(
                "new shares are below amounts already paid", 400
            )
        bill["title"] = cleaned.get("title", bill["title"])
        bill["total"] = new_total
        bill["payer"] = new_payer
        bill["participants"] = new_participants
        bill["shares"] = shares
        bill["payments"] = [p for p in bill["payments"]
                            if p["person"] in new_participants]
        return jsonify(_public(bill))

    @app.delete("/bills/<int:bill_id>")
    def delete_bill(bill_id):
        _bill, err = _get_bill(bill_id)
        if err is not None:
            return err
        del app.config["BILLS"][bill_id]
        return jsonify({"deleted": bill_id})

    @app.post("/bills/<int:bill_id>/payments")
    def add_payment(bill_id):
        bill, err = _get_bill(bill_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        person = data.get("person")
        if not isinstance(person, str) or not person.strip():
            return _error("person is required and must be a non-empty string", 400)
        person = person.strip()
        if person not in bill["participants"]:
            return _error(f"{person!r} is not a participant in this bill", 400)
        if person == bill["payer"]:
            return _error("the payer cannot pay themselves", 400)
        amount = data.get("amount")
        if isinstance(amount, bool) or not isinstance(amount, (int, float)) \
                or amount <= 0:
            return _error("amount is required and must be a positive number", 400)
        amount = round(float(amount), 2)
        remaining = round(
            bill["shares"][person] - _paid_by(bill)[person], 2
        )
        if amount > remaining + 0.01:
            return _error(
                f"amount exceeds {person}'s remaining share of {remaining}",
                400,
            )
        payment_id = app.config["NEXT_PAYMENT_ID"]
        app.config["NEXT_PAYMENT_ID"] += 1
        payment = {"id": payment_id, "person": person, "amount": amount}
        bill["payments"].append(payment)
        return jsonify(_public(bill)), 201

    @app.get("/bills/<int:bill_id>/payments")
    def list_payments(bill_id):
        bill, err = _get_bill(bill_id)
        if err is not None:
            return err
        return jsonify(list(bill["payments"]))

    @app.get("/bills/<int:bill_id>/balances")
    def get_balances(bill_id):
        bill, err = _get_bill(bill_id)
        if err is not None:
            return err
        return jsonify(_balances(bill))

    @app.get("/bills/<int:bill_id>/settlements")
    def get_settlements(bill_id):
        bill, err = _get_bill(bill_id)
        if err is not None:
            return err
        plan = [
            {"from": row["person"], "to": bill["payer"],
             "amount": row["balance"]}
            for row in _balances(bill)
            if row["balance"] > 0
        ]
        return jsonify(plan)

    @app.get("/summary")
    def summary():
        bills = list(app.config["BILLS"].values())
        total_amount = round(sum(b["total"] for b in bills), 2)
        outstanding = round(
            sum(_public(b)["outstanding"] for b in bills), 2
        )
        return jsonify(
            {
                "total_bills": len(bills),
                "total_amount": total_amount,
                "total_outstanding": outstanding,
                "settled_bills": sum(
                    1 for b in bills if _public(b)["settled"]
                ),
            }
        )

    @app.errorhandler(404)
    def handle_404(exc):
        return _error("route not found", 404)

    return app
