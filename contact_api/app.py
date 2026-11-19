"""REST contact-form API built with Flask.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Endpoints:

- ``GET /messages`` — list all submitted messages
- ``POST /messages`` — submit one (``{"name": str, "email": str, "message": str}``)
- ``GET /messages/<id>`` — fetch one message
- ``DELETE /messages/<id>`` — delete a message

Names, emails and messages are validated; bad input returns 400 and
unknown ids return 404, both as ``{"error": "..."}``.
"""
import re

from flask import Flask, jsonify, request

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def create_app():
    """Return a Flask app with a fresh, empty in-memory message store."""
    app = Flask(__name__)
    app.config["MESSAGES"] = {}
    app.config["NEXT_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    @app.get("/messages")
    def list_messages():
        messages = sorted(app.config["MESSAGES"].values(), key=lambda m: m["id"])
        return jsonify(messages)

    @app.post("/messages")
    def create_message():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            return _error("name is required and must be a non-empty string", 400)
        email = data.get("email")
        if not isinstance(email, str) or not _EMAIL_RE.match(email.strip()):
            return _error("email is required and must be a valid email address", 400)
        message = data.get("message")
        if not isinstance(message, str) or not message.strip():
            return _error("message is required and must be a non-empty string", 400)
        if len(message) > 5000:
            return _error("message must be at most 5000 characters", 400)
        message_id = app.config["NEXT_ID"]
        app.config["NEXT_ID"] += 1
        record = {
            "id": message_id,
            "name": name.strip(),
            "email": email.strip(),
            "message": message.strip(),
        }
        app.config["MESSAGES"][message_id] = record
        return jsonify(record), 201

    @app.get("/messages/<int:message_id>")
    def get_message(message_id):
        record = app.config["MESSAGES"].get(message_id)
        if record is None:
            return _error(f"message {message_id} not found", 404)
        return jsonify(record)

    @app.delete("/messages/<int:message_id>")
    def delete_message(message_id):
        if message_id not in app.config["MESSAGES"]:
            return _error(f"message {message_id} not found", 404)
        del app.config["MESSAGES"][message_id]
        return jsonify({"deleted": message_id})

    return app
