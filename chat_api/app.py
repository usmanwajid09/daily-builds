"""Flask chat API.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Endpoints:

- ``POST /rooms`` — create a room (``{"name": str}``)
- ``GET /rooms`` — list rooms
- ``GET /rooms/<id>`` — fetch one room (includes ``message_count``)
- ``DELETE /rooms/<id>`` — delete a room and all its messages
- ``POST /rooms/<id>/messages`` — post a message
  (``{"sender": str, "text": str}``)
- ``GET /rooms/<id>/messages`` — per-room history, oldest first;
  ``?sender=<name>`` filters by sender, ``?limit=<n>`` caps the count
- ``DELETE /rooms/<id>/messages/<message_id>`` — delete one message

A message is ``{"id": int, "room_id": int, "sender": str, "text": str}``.
"""
from flask import Flask, jsonify, request


def create_app():
    """Return a Flask app with a fresh, empty in-memory chat store."""
    app = Flask(__name__)
    app.config["ROOMS"] = {}
    app.config["MESSAGES"] = {}
    app.config["NEXT_ROOM_ID"] = 1
    app.config["NEXT_MESSAGE_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _get_room(room_id):
        room = app.config["ROOMS"].get(room_id)
        if room is None:
            return None, _error(f"room {room_id} not found", 404)
        return room, None

    def _room_view(room):
        count = sum(
            1 for m in app.config["MESSAGES"].values() if m["room_id"] == room["id"]
        )
        return {"id": room["id"], "name": room["name"], "message_count": count}

    @app.post("/rooms")
    def create_room():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            return _error("name is required and must be a non-empty string", 400)
        room_id = app.config["NEXT_ROOM_ID"]
        app.config["NEXT_ROOM_ID"] += 1
        room = {"id": room_id, "name": name.strip()}
        app.config["ROOMS"][room_id] = room
        return jsonify(_room_view(room)), 201

    @app.get("/rooms")
    def list_rooms():
        rooms = sorted(app.config["ROOMS"].values(), key=lambda r: r["id"])
        return jsonify([_room_view(r) for r in rooms])

    @app.get("/rooms/<int:room_id>")
    def get_room(room_id):
        room, err = _get_room(room_id)
        if err is not None:
            return err
        return jsonify(_room_view(room))

    @app.delete("/rooms/<int:room_id>")
    def delete_room(room_id):
        _room, err = _get_room(room_id)
        if err is not None:
            return err
        del app.config["ROOMS"][room_id]
        doomed = [
            mid
            for mid, m in app.config["MESSAGES"].items()
            if m["room_id"] == room_id
        ]
        for mid in doomed:
            del app.config["MESSAGES"][mid]
        return jsonify({"deleted": room_id, "messages_deleted": len(doomed)})

    @app.post("/rooms/<int:room_id>/messages")
    def post_message(room_id):
        _room, err = _get_room(room_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        sender = data.get("sender")
        if not isinstance(sender, str) or not sender.strip():
            return _error("sender is required and must be a non-empty string", 400)
        text = data.get("text")
        if not isinstance(text, str) or not text.strip():
            return _error("text is required and must be a non-empty string", 400)
        message_id = app.config["NEXT_MESSAGE_ID"]
        app.config["NEXT_MESSAGE_ID"] += 1
        message = {
            "id": message_id,
            "room_id": room_id,
            "sender": sender.strip(),
            "text": text.strip(),
        }
        app.config["MESSAGES"][message_id] = message
        return jsonify(message), 201

    @app.get("/rooms/<int:room_id>/messages")
    def list_messages(room_id):
        _room, err = _get_room(room_id)
        if err is not None:
            return err
        messages = [
            m
            for m in sorted(app.config["MESSAGES"].values(), key=lambda m: m["id"])
            if m["room_id"] == room_id
        ]
        sender = request.args.get("sender")
        if sender:
            messages = [m for m in messages if m["sender"] == sender]
        limit = request.args.get("limit")
        if limit is not None:
            try:
                n = int(limit)
            except ValueError:
                return _error("limit must be an integer", 400)
            if n < 0:
                return _error("limit must not be negative", 400)
            messages = messages[:n]
        return jsonify(messages)

    @app.delete("/rooms/<int:room_id>/messages/<int:message_id>")
    def delete_message(room_id, message_id):
        _room, err = _get_room(room_id)
        if err is not None:
            return err
        message = app.config["MESSAGES"].get(message_id)
        if message is None or message["room_id"] != room_id:
            return _error(f"message {message_id} not found in room {room_id}", 404)
        del app.config["MESSAGES"][message_id]
        return jsonify({"deleted": message_id})

    @app.errorhandler(404)
    def handle_404(exc):
        return _error("route not found", 404)

    return app
