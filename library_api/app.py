"""REST library API built with Flask.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Endpoints:

- ``GET /books`` — list all books
- ``POST /books`` — add a book
- ``GET /books/<id>`` — fetch one book
- ``PUT /books/<id>`` — update a book
- ``DELETE /books/<id>`` — delete a book
- ``POST /books/<id>/borrow`` — borrow a copy (creates a loan)
- ``POST /books/<id>/return`` — return a borrowed copy
- ``GET /loans`` — list all loans (``?overdue=true`` shows overdue only)

A book is ``{"id": int, "title": str, "author": str, "isbn": str?,
"total_copies": int, "available_copies": int}``. A loan is
``{"loan_id": int, "book_id": int, "borrower": str, "borrowed_on": date,
"due_on": date, "returned": bool, "overdue": bool}`` — ``overdue`` is
true when the due date has passed and the book has not been returned.
"""
from datetime import date, timedelta

from flask import Flask, jsonify, request

DEFAULT_LOAN_DAYS = 14


def create_app():
    """Return a Flask app with a fresh, empty in-memory library store."""
    app = Flask(__name__)
    app.config["BOOKS"] = {}
    app.config["LOANS"] = {}
    app.config["NEXT_BOOK_ID"] = 1
    app.config["NEXT_LOAN_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _with_overdue(loan):
        """Return a copy of a loan with its overdue flag computed for today."""
        out = dict(loan)
        out["overdue"] = (not loan["returned"]) and date.fromisoformat(
            loan["due_on"]
        ) < date.today()
        return out

    def _validate_book(data):
        if not isinstance(data, dict):
            return None, _error("request body must be a JSON object", 400)
        title = data.get("title")
        if not isinstance(title, str) or not title.strip():
            return None, _error("title is required and must be a non-empty string", 400)
        author = data.get("author")
        if not isinstance(author, str) or not author.strip():
            return None, _error("author is required and must be a non-empty string", 400)
        isbn = data.get("isbn")
        if isbn is not None and (not isinstance(isbn, str) or not isbn.strip()):
            return None, _error("isbn must be a non-empty string when provided", 400)
        total = data.get("total_copies")
        if not isinstance(total, int) or isinstance(total, bool) or total < 1:
            return None, _error("total_copies is required and must be a positive integer", 400)
        cleaned = {
            "title": title.strip(),
            "author": author.strip(),
            "total_copies": total,
        }
        if isbn is not None:
            cleaned["isbn"] = isbn.strip()
        return cleaned, None

    def _get_book(book_id):
        book = app.config["BOOKS"].get(book_id)
        if book is None:
            return None, _error(f"book {book_id} not found", 404)
        return book, None

    @app.get("/books")
    def list_books():
        return jsonify(sorted(app.config["BOOKS"].values(), key=lambda b: b["id"]))

    @app.post("/books")
    def create_book():
        cleaned, err = _validate_book(request.get_json(silent=True))
        if err is not None:
            return err
        book_id = app.config["NEXT_BOOK_ID"]
        app.config["NEXT_BOOK_ID"] += 1
        book = {"id": book_id, **cleaned, "available_copies": cleaned["total_copies"]}
        app.config["BOOKS"][book_id] = book
        return jsonify(book), 201

    @app.get("/books/<int:book_id>")
    def get_book(book_id):
        book, err = _get_book(book_id)
        if err is not None:
            return err
        return jsonify(book)

    @app.put("/books/<int:book_id>")
    def update_book(book_id):
        book, err = _get_book(book_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or not data:
            return _error("request body must be a non-empty JSON object", 400)
        merged = {
            "title": data.get("title", book["title"]),
            "author": data.get("author", book["author"]),
            "isbn": data.get("isbn", book.get("isbn")),
            "total_copies": data.get("total_copies", book["total_copies"]),
        }
        cleaned, err = _validate_book(merged)
        if err is not None:
            return err
        borrowed = book["total_copies"] - book["available_copies"]
        if cleaned["total_copies"] < borrowed:
            return _error(
                f"total_copies cannot be below {borrowed}: that many copies are on loan",
                400,
            )
        book.update(cleaned)
        book["available_copies"] = cleaned["total_copies"] - borrowed
        return jsonify(book)

    @app.delete("/books/<int:book_id>")
    def delete_book(book_id):
        book, err = _get_book(book_id)
        if err is not None:
            return err
        if book["available_copies"] < book["total_copies"]:
            return _error("cannot delete a book with copies on loan", 400)
        del app.config["BOOKS"][book_id]
        return jsonify({"deleted": book_id})

    @app.post("/books/<int:book_id>/borrow")
    def borrow_book(book_id):
        book, err = _get_book(book_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        borrower = data.get("borrower")
        if not isinstance(borrower, str) or not borrower.strip():
            return _error("borrower is required and must be a non-empty string", 400)
        due_days = data.get("due_days", DEFAULT_LOAN_DAYS)
        if not isinstance(due_days, int) or isinstance(due_days, bool) or due_days < 1:
            return _error("due_days must be a positive integer", 400)
        if book["available_copies"] < 1:
            return _error(f"no copies of book {book_id} are available to borrow", 400)
        loan_id = app.config["NEXT_LOAN_ID"]
        app.config["NEXT_LOAN_ID"] += 1
        today = date.today()
        loan = {
            "loan_id": loan_id,
            "book_id": book_id,
            "borrower": borrower.strip(),
            "borrowed_on": today.isoformat(),
            "due_on": (today + timedelta(days=due_days)).isoformat(),
            "returned": False,
        }
        app.config["LOANS"][loan_id] = loan
        book["available_copies"] -= 1
        return jsonify(_with_overdue(loan)), 201

    @app.post("/books/<int:book_id>/return")
    def return_book(book_id):
        book, err = _get_book(book_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        loan_id = data.get("loan_id")
        if not isinstance(loan_id, int) or isinstance(loan_id, bool):
            return _error("loan_id is required and must be an integer", 400)
        loan = app.config["LOANS"].get(loan_id)
        if loan is None or loan["book_id"] != book_id:
            return _error(f"loan {loan_id} not found for book {book_id}", 404)
        if loan["returned"]:
            return _error(f"loan {loan_id} was already returned", 400)
        loan["returned"] = True
        book["available_copies"] += 1
        return jsonify(_with_overdue(loan))

    @app.get("/loans")
    def list_loans():
        loans = [_with_overdue(l) for l in app.config["LOANS"].values()]
        loans.sort(key=lambda l: l["loan_id"])
        overdue = request.args.get("overdue")
        if overdue is not None:
            if overdue.lower() not in ("true", "false"):
                return _error("overdue must be true or false", 400)
            want = overdue.lower() == "true"
            loans = [l for l in loans if l["overdue"] == want]
        return jsonify(loans)

    @app.errorhandler(404)
    def handle_404(exc):
        return _error("route not found", 404)

    return app
