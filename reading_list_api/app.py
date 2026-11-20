"""Flask reading list API.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Endpoints:

- ``POST /books`` — create a book (``{\"title\": str, \"author\": str,
  \"total_pages\": int, \"status\": \"to-read\" | \"reading\" | \"finished\"?,
  \"pages_read\": int?}``)
- ``GET /books`` — list books; ``?status=`` and ``?author=``
  (case-insensitive substring) filters
- ``GET /books/<id>`` — fetch one book
- ``PUT /books/<id>`` — update title, author, status, total_pages
  and/or pages_read
- ``DELETE /books/<id>`` — delete a book
- ``POST /books/<id>/read`` — log pages read (``{\"pages\": int}``);
  pages_read is capped at total_pages and the book is marked
  finished automatically when the last page is reached
- ``POST /books/<id>/finish`` — mark a book finished
- ``GET /stats`` — totals, counts by status and overall progress %

A book is ``{\"id\": int, \"title\": str, \"author\": str,
\"status\": \"to-read\" | \"reading\" | \"finished\", \"total_pages\": int,
\"pages_read\": int, \"progress_pct\": float}``. ``progress_pct`` is
``pages_read / total_pages * 100`` rounded to one decimal place.
"""
from flask import Flask, jsonify, request

STATUSES = ("to-read", "reading", "finished")


def _progress(book):
    if book["total_pages"] <= 0:
        return 0.0
    return round(book["pages_read"] / book["total_pages"] * 100, 1)


def _public(book):
    return dict(book, progress_pct=_progress(book))


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def create_app():
    """Return a Flask app with a fresh, empty in-memory reading list."""
    app = Flask(__name__)
    app.config["BOOKS"] = {}
    app.config["NEXT_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _get_book(book_id):
        book = app.config["BOOKS"].get(book_id)
        if book is None:
            return None, _error(f"book {book_id} not found", 404)
        return book, None

    def _validate_pages(total_pages, pages_read, status):
        if not _is_int(total_pages) or total_pages <= 0:
            return "total_pages is required and must be a positive integer"
        if not _is_int(pages_read) or pages_read < 0:
            return "pages_read must be a non-negative integer"
        if pages_read > total_pages:
            return "pages_read cannot exceed total_pages"
        if status not in STATUSES:
            return f"status must be one of {', '.join(STATUSES)}"
        if status == "finished" and pages_read != total_pages:
            return "a finished book must have pages_read equal to total_pages"
        return None

    @app.post("/books")
    def create_book():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        title = data.get("title")
        if not isinstance(title, str) or not title.strip():
            return _error("title is required and must be a non-empty string", 400)
        author = data.get("author")
        if not isinstance(author, str) or not author.strip():
            return _error("author is required and must be a non-empty string", 400)
        total_pages = data.get("total_pages")
        pages_read = data.get("pages_read", 0)
        status = data.get("status", "to-read")
        err = _validate_pages(total_pages, pages_read, status)
        if err:
            return _error(err, 400)
        book_id = app.config["NEXT_ID"]
        app.config["NEXT_ID"] += 1
        book = {
            "id": book_id,
            "title": title.strip(),
            "author": author.strip(),
            "status": status,
            "total_pages": total_pages,
            "pages_read": pages_read,
        }
        app.config["BOOKS"][book_id] = book
        return jsonify(_public(book)), 201

    @app.get("/books")
    def list_books():
        books = sorted(app.config["BOOKS"].values(), key=lambda b: b["id"])
        status = request.args.get("status")
        if status is not None:
            if status not in STATUSES:
                return _error(f"status must be one of {', '.join(STATUSES)}", 400)
            books = [b for b in books if b["status"] == status]
        author = request.args.get("author")
        if author:
            needle = author.lower()
            books = [b for b in books if needle in b["author"].lower()]
        return jsonify([_public(b) for b in books])

    @app.get("/books/<int:book_id>")
    def get_book(book_id):
        book, err = _get_book(book_id)
        if err is not None:
            return err
        return jsonify(_public(book))

    @app.put("/books/<int:book_id>")
    def update_book(book_id):
        book, err = _get_book(book_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        fields = ("title", "author", "status", "total_pages", "pages_read")
        if not any(k in data for k in fields):
            return _error(
                "nothing to update: provide title, author, status, "
                "total_pages and/or pages_read",
                400,
            )
        new = dict(book)
        if "title" in data:
            if not isinstance(data["title"], str) or not data["title"].strip():
                return _error("title must be a non-empty string", 400)
            new["title"] = data["title"].strip()
        if "author" in data:
            if not isinstance(data["author"], str) or not data["author"].strip():
                return _error("author must be a non-empty string", 400)
            new["author"] = data["author"].strip()
        if "status" in data:
            new["status"] = data["status"]
        if "total_pages" in data:
            new["total_pages"] = data["total_pages"]
        if "pages_read" in data:
            new["pages_read"] = data["pages_read"]
        err = _validate_pages(new["total_pages"], new["pages_read"], new["status"])
        if err:
            return _error(err, 400)
        book.update(new)
        return jsonify(_public(book))

    @app.delete("/books/<int:book_id>")
    def delete_book(book_id):
        _book, err = _get_book(book_id)
        if err is not None:
            return err
        del app.config["BOOKS"][book_id]
        return jsonify({"deleted": book_id})

    @app.post("/books/<int:book_id>/read")
    def log_reading(book_id):
        book, err = _get_book(book_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        pages = data.get("pages")
        if not _is_int(pages) or pages <= 0:
            return _error("pages is required and must be a positive integer", 400)
        book["pages_read"] = min(book["total_pages"], book["pages_read"] + pages)
        if book["pages_read"] == book["total_pages"]:
            book["status"] = "finished"
        elif book["status"] == "to-read":
            book["status"] = "reading"
        return jsonify(_public(book))

    @app.post("/books/<int:book_id>/finish")
    def finish_book(book_id):
        book, err = _get_book(book_id)
        if err is not None:
            return err
        book["pages_read"] = book["total_pages"]
        book["status"] = "finished"
        return jsonify(_public(book))

    @app.get("/stats")
    def stats():
        books = list(app.config["BOOKS"].values())
        total_pages = sum(b["total_pages"] for b in books)
        pages_read = sum(b["pages_read"] for b in books)
        overall = round(pages_read / total_pages * 100, 1) if total_pages else 0.0
        return jsonify(
            {
                "total_books": len(books),
                "by_status": {
                    status: sum(1 for b in books if b["status"] == status)
                    for status in STATUSES
                },
                "total_pages": total_pages,
                "pages_read": pages_read,
                "overall_progress_pct": overall,
            }
        )

    @app.errorhandler(404)
    def handle_404(exc):
        return _error("route not found", 404)

    return app
