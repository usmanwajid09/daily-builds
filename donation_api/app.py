"""Flask charity donation API.

In-memory store, fresh for each app instance. All responses are JSON.
Money is handled as numbers (PKR by convention); amounts must be
positive and are rounded to two decimals.

Campaigns — ``POST /campaigns`` (``{"title": str, "goal_amount": number,
"description": str?, "organizer": str?}``); ``GET /campaigns``;
``GET /campaigns/<id>``; ``PUT /campaigns/<id>``;
``DELETE /campaigns/<id>`` (only while no donations exist).

Donations — ``POST /campaigns/<id>/donations``
(``{"donor_name": str, "amount": number, "message": str?}``);
``GET /campaigns/<id>/donations``; ``GET /donations`` (all donations);
``GET /donations/<id>``.

Summary — ``GET /campaigns/<id>/summary`` returns the campaign plus
``total_raised``, ``donation_count``, ``donor_count``,
``progress_percent``, ``amount_remaining`` and ``goal_reached``.
"""
from flask import Flask, jsonify, request


def create_app():
    """Return a Flask app with a fresh, empty in-memory donation store."""
    app = Flask(__name__)
    app.config["CAMPAIGNS"] = {}
    app.config["DONATIONS"] = {}
    app.config["NEXT_CAMPAIGN_ID"] = 1
    app.config["NEXT_DONATION_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _get_campaign(campaign_id):
        campaign = app.config["CAMPAIGNS"].get(campaign_id)
        if campaign is None:
            return None, _error(f"campaign {campaign_id} not found", 404)
        return campaign, None

    def _validate_amount(value, field):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value <= 0
        ):
            return _error(f"{field} must be a positive number", 400)
        return None

    def _campaign_donations(campaign_id):
        return [
            d for d in app.config["DONATIONS"].values() if d["campaign_id"] == campaign_id
        ]

    def _summary(campaign):
        donations = _campaign_donations(campaign["id"])
        total = round(sum(d["amount"] for d in donations), 2)
        goal = campaign["goal_amount"]
        return {
            **campaign,
            "total_raised": total,
            "donation_count": len(donations),
            "donor_count": len({d["donor_name"].lower() for d in donations}),
            "progress_percent": round(min(total / goal * 100, 100.0), 2)
            if goal
            else 0.0,
            "amount_remaining": round(max(goal - total, 0.0), 2),
            "goal_reached": total >= goal,
        }

    @app.post("/campaigns")
    def create_campaign():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        title = data.get("title")
        if not isinstance(title, str) or not title.strip():
            return _error("title is required and must be a non-empty string", 400)
        err = _validate_amount(data.get("goal_amount"), "goal_amount")
        if err is not None:
            return err
        description = data.get("description", "")
        if not isinstance(description, str):
            return _error("description must be a string", 400)
        organizer = data.get("organizer")
        if organizer is not None and (
            not isinstance(organizer, str) or not organizer.strip()
        ):
            return _error("organizer must be a non-empty string", 400)
        campaign_id = app.config["NEXT_CAMPAIGN_ID"]
        app.config["NEXT_CAMPAIGN_ID"] += 1
        campaign = {
            "id": campaign_id,
            "title": title.strip(),
            "goal_amount": round(float(data["goal_amount"]), 2),
            "description": description,
            "organizer": organizer.strip() if organizer else None,
        }
        app.config["CAMPAIGNS"][campaign_id] = campaign
        return jsonify(campaign), 201

    @app.get("/campaigns")
    def list_campaigns():
        campaigns = list(app.config["CAMPAIGNS"].values())
        goal_reached = request.args.get("goal_reached")
        if goal_reached is not None:
            if goal_reached not in ("true", "false"):
                return _error("goal_reached filter must be true or false", 400)
            wanted = goal_reached == "true"
            campaigns = [
                c for c in campaigns if _summary(c)["goal_reached"] == wanted
            ]
        return jsonify(campaigns)

    @app.get("/campaigns/<int:campaign_id>")
    def get_campaign(campaign_id):
        campaign, err = _get_campaign(campaign_id)
        if err is not None:
            return err
        return jsonify(campaign)

    @app.put("/campaigns/<int:campaign_id>")
    def update_campaign(campaign_id):
        campaign, err = _get_campaign(campaign_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        if "title" in data:
            if not isinstance(data["title"], str) or not data["title"].strip():
                return _error("title must be a non-empty string", 400)
            campaign["title"] = data["title"].strip()
        if "goal_amount" in data:
            err = _validate_amount(data["goal_amount"], "goal_amount")
            if err is not None:
                return err
            new_goal = round(float(data["goal_amount"]), 2)
            raised = sum(d["amount"] for d in _campaign_donations(campaign_id))
            if new_goal < raised:
                return _error(
                    f"goal_amount cannot be lowered below the "
                    f"{raised} already raised",
                    409,
                )
            campaign["goal_amount"] = new_goal
        if "description" in data:
            if not isinstance(data["description"], str):
                return _error("description must be a string", 400)
            campaign["description"] = data["description"]
        if "organizer" in data:
            organizer = data["organizer"]
            if organizer is not None and (
                not isinstance(organizer, str) or not organizer.strip()
            ):
                return _error("organizer must be a non-empty string", 400)
            campaign["organizer"] = organizer.strip() if organizer else None
        return jsonify(campaign)

    @app.delete("/campaigns/<int:campaign_id>")
    def delete_campaign(campaign_id):
        campaign, err = _get_campaign(campaign_id)
        if err is not None:
            return err
        if _campaign_donations(campaign_id):
            return _error(
                "cannot delete a campaign that has donations", 409
            )
        del app.config["CAMPAIGNS"][campaign_id]
        return jsonify({"deleted": campaign_id})

    @app.post("/campaigns/<int:campaign_id>/donations")
    def donate(campaign_id):
        campaign, err = _get_campaign(campaign_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        donor_name = data.get("donor_name")
        if not isinstance(donor_name, str) or not donor_name.strip():
            return _error(
                "donor_name is required and must be a non-empty string", 400
            )
        err = _validate_amount(data.get("amount"), "amount")
        if err is not None:
            return err
        message = data.get("message", "")
        if not isinstance(message, str):
            return _error("message must be a string", 400)
        donation_id = app.config["NEXT_DONATION_ID"]
        app.config["NEXT_DONATION_ID"] += 1
        donation = {
            "id": donation_id,
            "campaign_id": campaign_id,
            "donor_name": donor_name.strip(),
            "amount": round(float(data["amount"]), 2),
            "message": message,
        }
        app.config["DONATIONS"][donation_id] = donation
        return jsonify(donation), 201

    @app.get("/campaigns/<int:campaign_id>/donations")
    def list_campaign_donations(campaign_id):
        campaign, err = _get_campaign(campaign_id)
        if err is not None:
            return err
        return jsonify(_campaign_donations(campaign_id))

    @app.get("/campaigns/<int:campaign_id>/summary")
    def campaign_summary(campaign_id):
        campaign, err = _get_campaign(campaign_id)
        if err is not None:
            return err
        return jsonify(_summary(campaign))

    @app.get("/donations")
    def list_donations():
        return jsonify(list(app.config["DONATIONS"].values()))

    @app.get("/donations/<int:donation_id>")
    def get_donation(donation_id):
        donation = app.config["DONATIONS"].get(donation_id)
        if donation is None:
            return _error(f"donation {donation_id} not found", 404)
        return jsonify(donation)

    return app
