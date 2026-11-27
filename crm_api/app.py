"""Flask mini-CRM API.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Endpoints:

- ``POST /contacts`` — create a contact (``{\"name\": str, \"email\":
  str, \"company\": str?, \"phone\": str?}``; email must look like
  an email address and is unique, 409 on a duplicate)
- ``GET /contacts`` — list contacts
- ``GET /contacts/<id>`` — fetch one contact (with its deals)
- ``PUT /contacts/<id>`` — update a contact (409 on a duplicate
  email)
- ``DELETE /contacts/<id>`` — delete a contact (409 while it still
  has deals)
- ``POST /contacts/<id>/deals`` — add a deal (``{\"title\": str,
  \"value\": number, \"stage\": \"lead\" | \"qualified\" | \"proposal\"
  | \"won\" | \"lost\"?}``; defaults to ``lead``)
- ``GET /contacts/<id>/deals`` — list a contact's deals
- ``GET /deals`` — list all deals (``?stage=`` and ``?contact_id=``
  filters)
- ``GET /deals/<id>`` — fetch one deal
- ``PUT /deals/<id>`` — update a deal's title, value and/or stage
- ``DELETE /deals/<id>`` — delete a deal
- ``POST /deals/<id>/stage`` — move a deal to another stage
  (``{\"stage\": ...}``; ``won`` and ``lost`` are terminal — moving
  out of them is a 409)
- ``GET /pipeline`` — per-stage deal counts and value totals, total
  pipeline value, won/lost value, and the conversion rate
  (won / (won + lost) * 100, null when no deal is closed yet)

A contact is ``{\"id\": int, \"name\": str, \"email\": str,
\"company\": str | None, \"phone\": str | None, \"deals\": [...]}``;
a deal is ``{\"id\": int, \"contact_id\": int, \"title\": str,
\"value\": number, \"stage\": str}``.
"""
from flask import Flask, jsonify, request

STAGES = ("lead", "qualified", "proposal", "won", "lost")
TERMINAL_STAGES = ("won", "lost")


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _valid_email(value):
    if not isinstance(value, str):
        return False
    local, sep, domain = value.strip().partition("@")
    return bool(sep) and bool(local) and "." in domain and " " not in value


def create_app():
    """Return a Flask app with a fresh, empty in-memory CRM store."""
    app = Flask(__name__)
    app.config["CONTACTS"] = {}
    app.config["DEALS"] = {}
    app.config["NEXT_CONTACT_ID"] = 1
    app.config["NEXT_DEAL_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _get_contact(contact_id):
        contact = app.config["CONTACTS"].get(contact_id)
        if contact is None:
            return None, _error(f"contact {contact_id} not found", 404)
        return contact, None

    def _get_deal(deal_id):
        deal = app.config["DEALS"].get(deal_id)
        if deal is None:
            return None, _error(f"deal {deal_id} not found", 404)
        return deal, None

    def _contact_deals(contact_id):
        return sorted(
            (d for d in app.config["DEALS"].values()
             if d["contact_id"] == contact_id),
            key=lambda d: d["id"],
        )

    def _public_deal(deal):
        return dict(deal)

    def _public_contact(contact, with_deals=True):
        data = {
            "id": contact["id"],
            "name": contact["name"],
            "email": contact["email"],
            "company": contact["company"],
            "phone": contact["phone"],
        }
        if with_deals:
            data["deals"] = [
                _public_deal(d) for d in _contact_deals(contact["id"])
            ]
        return data

    def _validate_deal_body(data, partial=False):
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        if not partial or "title" in data:
            title = data.get("title")
            if not isinstance(title, str) or not title.strip():
                return _error(
                    "title is required and must be a non-empty string", 400
                )
        if "value" in data or not partial:
            value = data.get("value")
            if not _is_number(value) or value < 0:
                return _error("value is required and must be a number >= 0", 400)
        if "stage" in data and data["stage"] not in STAGES:
            return _error(f"stage must be one of {', '.join(STAGES)}", 400)
        return None

    @app.post("/contacts")
    def create_contact():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            return _error("name is required and must be a non-empty string", 400)
        email = data.get("email")
        if not _valid_email(email):
            return _error("email is required and must be a valid email address", 400)
        if any(c["email"].lower() == email.strip().lower()
               for c in app.config["CONTACTS"].values()):
            return _error(
                f"a contact with email {email.strip()!r} already exists", 409
            )
        company = data.get("company")
        if company is not None and (
            not isinstance(company, str) or not company.strip()
        ):
            return _error("company must be a non-empty string or null", 400)
        phone = data.get("phone")
        if phone is not None and (
            not isinstance(phone, str) or not phone.strip()
        ):
            return _error("phone must be a non-empty string or null", 400)
        contact_id = app.config["NEXT_CONTACT_ID"]
        app.config["NEXT_CONTACT_ID"] += 1
        contact = {
            "id": contact_id,
            "name": name.strip(),
            "email": email.strip(),
            "company": company.strip() if company else None,
            "phone": phone.strip() if phone else None,
        }
        app.config["CONTACTS"][contact_id] = contact
        return jsonify(_public_contact(contact)), 201

    @app.get("/contacts")
    def list_contacts():
        contacts = sorted(
            app.config["CONTACTS"].values(), key=lambda c: c["id"]
        )
        return jsonify([_public_contact(c, with_deals=False) for c in contacts])

    @app.get("/contacts/<int:contact_id>")
    def get_contact(contact_id):
        contact, err = _get_contact(contact_id)
        if err is not None:
            return err
        return jsonify(_public_contact(contact))

    @app.put("/contacts/<int:contact_id>")
    def update_contact(contact_id):
        contact, err = _get_contact(contact_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        if not any(k in data for k in ("name", "email", "company", "phone")):
            return _error(
                "nothing to update: provide name, email, company "
                "and/or phone",
                400,
            )
        if "name" in data:
            if not isinstance(data["name"], str) or not data["name"].strip():
                return _error("name must be a non-empty string", 400)
            contact["name"] = data["name"].strip()
        if "email" in data:
            if not _valid_email(data["email"]):
                return _error("email must be a valid email address", 400)
            if any(c["id"] != contact_id
                   and c["email"].lower() == data["email"].strip().lower()
                   for c in app.config["CONTACTS"].values()):
                return _error(
                    f"a contact with email {data['email'].strip()!r} "
                    "already exists",
                    409,
                )
            contact["email"] = data["email"].strip()
        for field in ("company", "phone"):
            if field in data:
                value = data[field]
                if value is not None and (
                    not isinstance(value, str) or not value.strip()
                ):
                    return _error(f"{field} must be a non-empty string or null", 400)
                contact[field] = value.strip() if value else None
        return jsonify(_public_contact(contact))

    @app.delete("/contacts/<int:contact_id>")
    def delete_contact(contact_id):
        _contact, err = _get_contact(contact_id)
        if err is not None:
            return err
        if _contact_deals(contact_id):
            return _error(
                f"contact {contact_id} still has deals; delete them first",
                409,
            )
        del app.config["CONTACTS"][contact_id]
        return jsonify({"deleted": contact_id})

    @app.post("/contacts/<int:contact_id>/deals")
    def create_deal(contact_id):
        _contact, err = _get_contact(contact_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        err = _validate_deal_body(data)
        if err is not None:
            return err
        deal_id = app.config["NEXT_DEAL_ID"]
        app.config["NEXT_DEAL_ID"] += 1
        deal = {
            "id": deal_id,
            "contact_id": contact_id,
            "title": data["title"].strip(),
            "value": data["value"],
            "stage": data.get("stage", "lead"),
        }
        app.config["DEALS"][deal_id] = deal
        return jsonify(_public_deal(deal)), 201

    @app.get("/contacts/<int:contact_id>/deals")
    def list_contact_deals(contact_id):
        _contact, err = _get_contact(contact_id)
        if err is not None:
            return err
        return jsonify([_public_deal(d) for d in _contact_deals(contact_id)])

    @app.get("/deals")
    def list_deals():
        stage = request.args.get("stage")
        if stage is not None and stage not in STAGES:
            return _error(f"stage filter must be one of {', '.join(STAGES)}", 400)
        contact_id = request.args.get("contact_id", type=int)
        deals = sorted(app.config["DEALS"].values(), key=lambda d: d["id"])
        if stage is not None:
            deals = [d for d in deals if d["stage"] == stage]
        if contact_id is not None:
            deals = [d for d in deals if d["contact_id"] == contact_id]
        return jsonify([_public_deal(d) for d in deals])

    @app.get("/deals/<int:deal_id>")
    def get_deal(deal_id):
        deal, err = _get_deal(deal_id)
        if err is not None:
            return err
        return jsonify(_public_deal(deal))

    @app.put("/deals/<int:deal_id>")
    def update_deal(deal_id):
        deal, err = _get_deal(deal_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        err = _validate_deal_body(data, partial=True)
        if err is not None:
            return err
        if not any(k in data for k in ("title", "value", "stage")):
            return _error(
                "nothing to update: provide title, value and/or stage", 400
            )
        if "title" in data:
            deal["title"] = data["title"].strip()
        if "value" in data:
            deal["value"] = data["value"]
        if "stage" in data:
            if deal["stage"] in TERMINAL_STAGES and data["stage"] not in TERMINAL_STAGES:
                return _error(
                    f"deal is already {deal['stage']}; closed deals "
                    "cannot be reopened",
                    409,
                )
            deal["stage"] = data["stage"]
        return jsonify(_public_deal(deal))

    @app.delete("/deals/<int:deal_id>")
    def delete_deal(deal_id):
        _deal, err = _get_deal(deal_id)
        if err is not None:
            return err
        del app.config["DEALS"][deal_id]
        return jsonify({"deleted": deal_id})

    @app.post("/deals/<int:deal_id>/stage")
    def change_deal_stage(deal_id):
        deal, err = _get_deal(deal_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        stage = data.get("stage")
        if stage not in STAGES:
            return _error(f"stage must be one of {', '.join(STAGES)}", 400)
        if stage == deal["stage"]:
            return _error(f"deal is already {stage}", 409)
        if deal["stage"] in TERMINAL_STAGES:
            return _error(
                f"deal is already {deal['stage']}; closed deals cannot "
                "be reopened",
                409,
            )
        deal["stage"] = stage
        return jsonify(_public_deal(deal))

    @app.get("/pipeline")
    def pipeline_stats():
        deals = list(app.config["DEALS"].values())
        by_stage = {}
        for stage in STAGES:
            stage_deals = [d for d in deals if d["stage"] == stage]
            by_stage[stage] = {
                "count": len(stage_deals),
                "total_value": sum(d["value"] for d in stage_deals),
            }
        won = by_stage["won"]["count"]
        lost = by_stage["lost"]["count"]
        closed = won + lost
        return jsonify(
            {
                "total_deals": len(deals),
                "total_value": sum(d["value"] for d in deals),
                "by_stage": by_stage,
                "won_value": by_stage["won"]["total_value"],
                "lost_value": by_stage["lost"]["total_value"],
                "conversion_rate": (
                    round(won / closed * 100, 1) if closed else None
                ),
            }
        )

    @app.errorhandler(404)
    def handle_404(exc):
        return _error("route not found", 404)

    return app
