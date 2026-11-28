"""Flask fundraiser API.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON.

Endpoints:

- ``POST /campaigns`` — create a campaign (``{"title": str,
  "goal_amount": number > 0, "description": str optional}``). 201.
- ``GET /campaigns`` — list all campaigns, each with ``raised``,
  ``progress_percent`` and ``donor_count``.
- ``GET /campaigns/<id>`` — fetch one campaign (same fields);
  unknown id -> 404.
- ``DELETE /campaigns/<id>`` — delete a campaign; unknown id -> 404.
- ``POST /campaigns/<id>/donate`` — record a donation
  (``{"donor_name": str, "amount": number > 0, "message": str
  optional}``). 201; unknown campaign -> 404.
- ``GET /campaigns/<id>/donations`` — list donations for the campaign,
  newest first; unknown campaign -> 404.
- ``GET /campaigns/<id>/progress`` — ``{"campaign_id", "title",
  "goal_amount", "raised", "remaining", "progress_percent",
  "donor_count", "goal_reached"}``; unknown campaign -> 404.
- ``GET /campaigns/<id>/top-donors`` — donors aggregated by name,
  sorted by total donated desc: ``[{"donor_name", "total_donated",
  "donation_count"}]``; unknown campaign -> 404.

All errors are JSON like ``{"error": "message"}`` with a 400 or 404
status code.
"""
from flask import Flask, jsonify, request


def create_app():
    """Return a Flask app with a fresh, empty in-memory campaign store."""
    app = Flask(__name__)
    app.config["CAMPAIGNS"] = {}
    app.config["NEXT_CAMPAIGN_ID"] = 1
    app.config["NEXT_DONATION_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _is_number(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def _campaign_view(campaign):
        raised = sum(d["amount"] for d in campaign["donations"])
        donors = {d["donor_name"] for d in campaign["donations"]}
        return {
            "id": campaign["id"],
            "title": campaign["title"],
            "goal_amount": campaign["goal_amount"],
            "description": campaign["description"],
            "raised": raised,
            "progress_percent": round(raised / campaign["goal_amount"] * 100, 2),
            "donor_count": len(donors),
        }

    def _get_campaign(campaign_id):
        campaign = app.config["CAMPAIGNS"].get(campaign_id)
        if campaign is None:
            return None, _error("campaign not found", 404)
        return campaign, None

    @app.route("/campaigns", methods=["POST"])
    def create_campaign():
        data = request.get_json(silent=True) or {}
        title = data.get("title")
        if not isinstance(title, str) or not title.strip():
            return _error("title is required", 400)
        goal = data.get("goal_amount")
        if not _is_number(goal) or goal <= 0:
            return _error("goal_amount must be a number greater than 0", 400)
        campaign_id = app.config["NEXT_CAMPAIGN_ID"]
        app.config["NEXT_CAMPAIGN_ID"] += 1
        campaign = {
            "id": campaign_id,
            "title": title.strip(),
            "goal_amount": goal,
            "description": data.get("description", ""),
            "donations": [],
        }
        app.config["CAMPAIGNS"][campaign_id] = campaign
        return jsonify(_campaign_view(campaign)), 201

    @app.route("/campaigns", methods=["GET"])
    def list_campaigns():
        return jsonify(
            [_campaign_view(c) for c in app.config["CAMPAIGNS"].values()]
        )

    @app.route("/campaigns/<int:campaign_id>", methods=["GET"])
    def get_campaign(campaign_id):
        campaign, error = _get_campaign(campaign_id)
        if error is not None:
            return error
        return jsonify(_campaign_view(campaign))

    @app.route("/campaigns/<int:campaign_id>", methods=["DELETE"])
    def delete_campaign(campaign_id):
        campaign, error = _get_campaign(campaign_id)
        if error is not None:
            return error
        del app.config["CAMPAIGNS"][campaign_id]
        return jsonify({"deleted": campaign_id})

    @app.route("/campaigns/<int:campaign_id>/donate", methods=["POST"])
    def donate(campaign_id):
        campaign, error = _get_campaign(campaign_id)
        if error is not None:
            return error
        data = request.get_json(silent=True) or {}
        donor_name = data.get("donor_name")
        if not isinstance(donor_name, str) or not donor_name.strip():
            return _error("donor_name is required", 400)
        amount = data.get("amount")
        if not _is_number(amount) or amount <= 0:
            return _error("amount must be a number greater than 0", 400)
        donation_id = app.config["NEXT_DONATION_ID"]
        app.config["NEXT_DONATION_ID"] += 1
        donation = {
            "id": donation_id,
            "donor_name": donor_name.strip(),
            "amount": amount,
            "message": data.get("message", ""),
            "campaign_id": campaign_id,
        }
        campaign["donations"].append(donation)
        return jsonify(donation), 201

    @app.route("/campaigns/<int:campaign_id>/donations", methods=["GET"])
    def list_donations(campaign_id):
        campaign, error = _get_campaign(campaign_id)
        if error is not None:
            return error
        return jsonify(list(reversed(campaign["donations"])))

    @app.route("/campaigns/<int:campaign_id>/progress", methods=["GET"])
    def campaign_progress(campaign_id):
        campaign, error = _get_campaign(campaign_id)
        if error is not None:
            return error
        raised = sum(d["amount"] for d in campaign["donations"])
        donors = {d["donor_name"] for d in campaign["donations"]}
        return jsonify({
            "campaign_id": campaign_id,
            "title": campaign["title"],
            "goal_amount": campaign["goal_amount"],
            "raised": raised,
            "remaining": max(campaign["goal_amount"] - raised, 0),
            "progress_percent": round(
                raised / campaign["goal_amount"] * 100, 2
            ),
            "donor_count": len(donors),
            "goal_reached": raised >= campaign["goal_amount"],
        })

    @app.route("/campaigns/<int:campaign_id>/top-donors", methods=["GET"])
    def top_donors(campaign_id):
        campaign, error = _get_campaign(campaign_id)
        if error is not None:
            return error
        totals = {}
        for donation in campaign["donations"]:
            entry = totals.setdefault(
                donation["donor_name"],
                {"donor_name": donation["donor_name"],
                 "total_donated": 0, "donation_count": 0},
            )
            entry["total_donated"] += donation["amount"]
            entry["donation_count"] += 1
        ranked = sorted(
            totals.values(),
            key=lambda e: (-e["total_donated"], e["donor_name"]),
        )
        return jsonify(ranked)

    return app
