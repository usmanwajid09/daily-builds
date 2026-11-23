"""REST book-club API built with Flask.

In-memory store, fresh for each app instance. Manage books, members and
meetings; members vote for the next book (one vote per member, changeable)
and RSVP to meetings.

Endpoints:
- ``GET /books`` / ``POST /books`` — list / add a book (``{"title", "author"}``)
- ``GET /books/<id>`` — one book, with its vote count
- ``POST /books/<id>/vote`` — vote for a book (``{"member_id": n}``)
- ``GET /votes`` — vote leaderboard, most-voted first
- ``GET /members`` / ``POST /members`` — list / add a member (``{"name"}``)
- ``GET /meetings`` / ``POST /meetings`` — list / schedule a meeting
  (``{"book_id", "date", "location"?}``)
- ``POST /meetings/<id>/rsvp`` — RSVP (``{"member_id": n, "attending": bool}``)
"""
import re
from datetime import date

from flask import Flask, jsonify, request

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def create_app():
    """Return a Flask app with a fresh, empty in-memory book-club store."""
    app = Flask(__name__)
    app.config["BOOKS"] = {}
    app.config["MEMBERS"] = {}
    app.config["MEETINGS"] = {}
    app.config["VOTES"] = {}  # member_id -> book_id
    app.config["NEXT_BOOK_ID"] = 1
    app.config["NEXT_MEMBER_ID"] = 1
    app.config["NEXT_MEETING_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _valid_date(value):
        if not isinstance(value, str) or not _DATE_RE.match(value):
            return False
        try:
            date.fromisoformat(value)
        except ValueError:
            return False
        return True

    def _votes_for(book_id):
        return sum(1 for b in app.config["VOTES"].values() if b == book_id)

    def _public_book(book):
        return {**book, "votes": _votes_for(book["id"])}

    def _public_meeting(meeting):
        attending = [m for m, a in meeting["rsvps"].items() if a]
        return {
            "id": meeting["id"],
            "book_id": meeting["book_id"],
            "date": meeting["date"],
            "location": meeting["location"],
            "rsvps": meeting["rsvps"],
            "attending_count": len(attending),
        }

    @app.get("/books")
    def list_books():
        books = sorted(app.config["BOOKS"].values(), key=lambda b: b["id"])
        return jsonify([_public_book(b) for b in books])

    @app.post("/books")
    def add_book():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        title = data.get("title")
        author = data.get("author")
        if not isinstance(title, str) or not title.strip():
            return _error("title is required and must be a non-empty string", 400)
        if not isinstance(author, str) or not author.strip():
            return _error("author is required and must be a non-empty string", 400)
        book_id = app.config["NEXT_BOOK_ID"]
        app.config["NEXT_BOOK_ID"] += 1
        book = {"id": book_id, "title": title.strip(), "author": author.strip()}
        app.config["BOOKS"][book_id] = book
        return jsonify(_public_book(book)), 201

    @app.get("/books/<int:book_id>")
    def get_book(book_id):
        book = app.config["BOOKS"].get(book_id)
        if book is None:
            return _error(f"book {book_id} not found", 404)
        return jsonify(_public_book(book))

    @app.post("/books/<int:book_id>/vote")
    def vote(book_id):
        book = app.config["BOOKS"].get(book_id)
        if book is None:
            return _error(f"book {book_id} not found", 404)
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        member_id = data.get("member_id")
        if not isinstance(member_id, int) or isinstance(member_id, bool):
            return _error("member_id is required and must be an integer", 400)
        if member_id not in app.config["MEMBERS"]:
            return _error(f"member {member_id} not found", 404)
        previous = app.config["VOTES"].get(member_id)
        app.config["VOTES"][member_id] = book_id
        return jsonify(
            {
                "member_id": member_id,
                "book_id": book_id,
                "previous_book_id": previous,
                "votes": _votes_for(book_id),
            }
        )

    @app.get("/votes")
    def leaderboard():
        books = sorted(app.config["BOOKS"].values(), key=lambda b: b["id"])
        ranked = sorted(
            [_public_book(b) for b in books],
            key=lambda b: (-b["votes"], b["id"]),
        )
        return jsonify(ranked)

    @app.get("/members")
    def list_members():
        members = sorted(app.config["MEMBERS"].values(), key=lambda m: m["id"])
        return jsonify(members)

    @app.post("/members")
    def add_member():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            return _error("name is required and must be a non-empty string", 400)
        member_id = app.config["NEXT_MEMBER_ID"]
        app.config["NEXT_MEMBER_ID"] += 1
        member = {"id": member_id, "name": name.strip()}
        app.config["MEMBERS"][member_id] = member
        return jsonify(member), 201

    @app.get("/meetings")
    def list_meetings():
        meetings = sorted(app.config["MEETINGS"].values(), key=lambda m: m["id"])
        return jsonify([_public_meeting(m) for m in meetings])

    @app.post("/meetings")
    def add_meeting():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        book_id = data.get("book_id")
        if not isinstance(book_id, int) or isinstance(book_id, bool):
            return _error("book_id is required and must be an integer", 400)
        if book_id not in app.config["BOOKS"]:
            return _error(f"book {book_id} not found", 404)
        meeting_date = data.get("date")
        if not _valid_date(meeting_date):
            return _error("date is required and must be a valid YYYY-MM-DD date", 400)
        location = data.get("location", "")
        if not isinstance(location, str):
            return _error("location must be a string", 400)
        meeting_id = app.config["NEXT_MEETING_ID"]
        app.config["NEXT_MEETING_ID"] += 1
        meeting = {
            "id": meeting_id,
            "book_id": book_id,
            "date": meeting_date,
            "location": location,
            "rsvps": {},
        }
        app.config["MEETINGS"][meeting_id] = meeting
        return jsonify(_public_meeting(meeting)), 201

    @app.post("/meetings/<int:meeting_id>/rsvp")
    def rsvp(meeting_id):
        meeting = app.config["MEETINGS"].get(meeting_id)
        if meeting is None:
            return _error(f"meeting {meeting_id} not found", 404)
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        member_id = data.get("member_id")
        if not isinstance(member_id, int) or isinstance(member_id, bool):
            return _error("member_id is required and must be an integer", 400)
        if member_id not in app.config["MEMBERS"]:
            return _error(f"member {member_id} not found", 404)
        attending = data.get("attending")
        if not isinstance(attending, bool):
            return _error("attending is required and must be a boolean", 400)
        meeting["rsvps"][member_id] = attending
        return jsonify(_public_meeting(meeting))

    return app
