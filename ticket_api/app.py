"""Flask support-ticket API.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Endpoints:

- ``POST /tickets`` — create a ticket (``{"title": str,
  "description": str?, "priority": "low" | "medium" | "high" |
  "urgent"?, "assignee": str?}``); new tickets start ``open``
- ``GET /tickets`` — list tickets; ``?status=``, ``?priority=`` and
  ``?assignee=`` filters
- ``GET /tickets/<id>`` — fetch one ticket
- ``PUT /tickets/<id>`` — update title, description, priority and/or
  assignee (status changes go through ``/status``)
- ``POST /tickets/<id>/status`` — move a ticket (``{"status": str}``)
- ``DELETE /tickets/<id>`` — delete a ticket

Status workflow — only these transitions are allowed::

    open -> in_progress | closed
    in_progress -> resolved | open
    resolved -> closed | in_progress
    closed -> (terminal, no outgoing transitions)

A ticket is ``{"id": int, "title": str, "description": str,
"priority": str, "assignee": str | None, "status": str}``.
"""
from flask import Flask, jsonify, request

PRIORITIES = ("low", "medium", "high", "urgent")
STATUSES = ("open", "in_progress", "resolved", "closed")
ALLOWED_TRANSITIONS = {
    "open": ("in_progress", "closed"),
    "in_progress": ("resolved", "open"),
    "resolved": ("closed", "in_progress"),
    "closed": (),
}


def create_app():
    """Return a Flask app with a fresh, empty in-memory ticket store."""
    app = Flask(__name__)
    app.config["TICKETS"] = {}
    app.config["NEXT_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _get_ticket(ticket_id):
        ticket = app.config["TICKETS"].get(ticket_id)
        if ticket is None:
            return None, _error(f"ticket {ticket_id} not found", 404)
        return ticket, None

    @app.post("/tickets")
    def create_ticket():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        title = data.get("title")
        if not isinstance(title, str) or not title.strip():
            return _error("title is required and must be a non-empty string", 400)
        description = data.get("description", "")
        if not isinstance(description, str):
            return _error("description must be a string", 400)
        priority = data.get("priority", "medium")
        if priority not in PRIORITIES:
            return _error(f"priority must be one of {', '.join(PRIORITIES)}", 400)
        assignee = data.get("assignee")
        if assignee is not None and (
            not isinstance(assignee, str) or not assignee.strip()
        ):
            return _error("assignee must be a non-empty string or omitted", 400)
        ticket_id = app.config["NEXT_ID"]
        app.config["NEXT_ID"] += 1
        ticket = {
            "id": ticket_id,
            "title": title.strip(),
            "description": description,
            "priority": priority,
            "assignee": assignee.strip() if assignee else None,
            "status": "open",
        }
        app.config["TICKETS"][ticket_id] = ticket
        return jsonify(ticket), 201

    @app.get("/tickets")
    def list_tickets():
        tickets = sorted(app.config["TICKETS"].values(), key=lambda t: t["id"])
        status = request.args.get("status")
        if status is not None:
            if status not in STATUSES:
                return _error(f"status must be one of {', '.join(STATUSES)}", 400)
            tickets = [t for t in tickets if t["status"] == status]
        priority = request.args.get("priority")
        if priority is not None:
            if priority not in PRIORITIES:
                return _error(
                    f"priority must be one of {', '.join(PRIORITIES)}", 400
                )
            tickets = [t for t in tickets if t["priority"] == priority]
        assignee = request.args.get("assignee")
        if assignee:
            tickets = [t for t in tickets if t["assignee"] == assignee]
        return jsonify(tickets)

    @app.get("/tickets/<int:ticket_id>")
    def get_ticket(ticket_id):
        ticket, err = _get_ticket(ticket_id)
        if err is not None:
            return err
        return jsonify(ticket)

    @app.put("/tickets/<int:ticket_id>")
    def update_ticket(ticket_id):
        ticket, err = _get_ticket(ticket_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        if "status" in data:
            return _error("use POST /tickets/<id>/status to change status", 400)
        if not any(k in data for k in ("title", "description", "priority", "assignee")):
            return _error(
                "nothing to update: provide title, description, priority "
                "and/or assignee",
                400,
            )
        if "title" in data:
            if not isinstance(data["title"], str) or not data["title"].strip():
                return _error("title must be a non-empty string", 400)
            ticket["title"] = data["title"].strip()
        if "description" in data:
            if not isinstance(data["description"], str):
                return _error("description must be a string", 400)
            ticket["description"] = data["description"]
        if "priority" in data:
            if data["priority"] not in PRIORITIES:
                return _error(
                    f"priority must be one of {', '.join(PRIORITIES)}", 400
                )
            ticket["priority"] = data["priority"]
        if "assignee" in data:
            assignee = data["assignee"]
            if assignee is not None and (
                not isinstance(assignee, str) or not assignee.strip()
            ):
                return _error("assignee must be a non-empty string or null", 400)
            ticket["assignee"] = assignee.strip() if assignee else None
        return jsonify(ticket)

    @app.post("/tickets/<int:ticket_id>/status")
    def change_status(ticket_id):
        ticket, err = _get_ticket(ticket_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        new_status = data.get("status")
        if new_status not in STATUSES:
            return _error(f"status must be one of {', '.join(STATUSES)}", 400)
        allowed = ALLOWED_TRANSITIONS[ticket["status"]]
        if new_status not in allowed:
            if new_status == ticket["status"]:
                return _error(
                    f"ticket is already {new_status}", 400
                )
            return _error(
                f"cannot move ticket from {ticket['status']} to {new_status}; "
                f"allowed: {', '.join(allowed) or 'none (ticket is closed)'}",
                400,
            )
        ticket["status"] = new_status
        return jsonify(ticket)

    @app.delete("/tickets/<int:ticket_id>")
    def delete_ticket(ticket_id):
        _ticket, err = _get_ticket(ticket_id)
        if err is not None:
            return err
        del app.config["TICKETS"][ticket_id]
        return jsonify({"deleted": ticket_id})

    @app.errorhandler(404)
    def handle_404(exc):
        return _error("route not found", 404)

    return app
