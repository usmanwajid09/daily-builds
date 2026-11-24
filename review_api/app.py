"""REST product-review API built with Flask.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Endpoints:

- ``POST /reviews`` — submit a review (``{"product_id": str,
  "author": str, "rating": int 1-5, "title": str?, "comment": str?}``;
  one review per author per product)
- ``GET /reviews`` — list reviews (``?product_id=...`` and
  ``?status=visible|flagged|hidden`` filter)
- ``GET /reviews/<id>`` — fetch one review
- ``DELETE /reviews/<id>`` — delete a review
- ``POST /reviews/<id>/helpful`` — add one helpful vote
- ``POST /reviews/<id>/flag`` — flag a review for moderation
- ``POST /reviews/<id>/moderate`` — moderate a flagged review
  (``{"action": "approve" | "hide"}``)
- ``GET /products/<product_id>/summary`` — average rating, review
  count and the 1-5 star distribution for a product

Ratings are integers from 1 to 5. The summary counts every review
except hidden ones, so flagged reviews still count until a moderator
hides them.
"""
from flask import Flask, jsonify, request

STATUSES = ("visible", "flagged", "hidden")


def create_app():
    """Return a Flask app with a fresh, empty in-memory review store."""
    app = Flask(__name__)
    app.config["REVIEWS"] = {}  # id -> review dict
    app.config["NEXT_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _is_int(value):
        return isinstance(value, int) and not isinstance(value, bool)

    def _get_review(review_id):
        review = app.config["REVIEWS"].get(review_id)
        if review is None:
            return None, _error(f"review {review_id} not found", 404)
        return review, None

    @app.post("/reviews")
    def create_review():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        product_id = data.get("product_id")
        if not isinstance(product_id, str) or not product_id.strip():
            return _error("product_id is required and must be a non-empty string", 400)
        author = data.get("author")
        if not isinstance(author, str) or not author.strip():
            return _error("author is required and must be a non-empty string", 400)
        rating = data.get("rating")
        if not _is_int(rating) or rating < 1 or rating > 5:
            return _error("rating is required and must be an integer from 1 to 5", 400)
        title = data.get("title", "")
        if not isinstance(title, str):
            return _error("title must be a string", 400)
        comment = data.get("comment", "")
        if not isinstance(comment, str):
            return _error("comment must be a string", 400)
        product_id = product_id.strip()
        author = author.strip()
        for existing in app.config["REVIEWS"].values():
            if existing["product_id"] == product_id and existing["author"] == author:
                return _error(
                    f"author {author!r} already reviewed product {product_id!r}",
                    409,
                )
        review_id = app.config["NEXT_ID"]
        app.config["NEXT_ID"] += 1
        review = {
            "id": review_id,
            "product_id": product_id,
            "author": author,
            "rating": rating,
            "title": title,
            "comment": comment,
            "helpful_votes": 0,
            "flagged": False,
            "status": "visible",
        }
        app.config["REVIEWS"][review_id] = review
        return jsonify(review), 201

    @app.get("/reviews")
    def list_reviews():
        reviews = sorted(app.config["REVIEWS"].values(), key=lambda r: r["id"])
        product_id = request.args.get("product_id")
        if product_id:
            reviews = [r for r in reviews if r["product_id"] == product_id]
        status = request.args.get("status")
        if status is not None:
            if status not in STATUSES:
                return _error("status must be visible, flagged or hidden", 400)
            reviews = [r for r in reviews if r["status"] == status]
        return jsonify(reviews)

    @app.get("/reviews/<int:review_id>")
    def get_review(review_id):
        review, err = _get_review(review_id)
        if err is not None:
            return err
        return jsonify(review)

    @app.delete("/reviews/<int:review_id>")
    def delete_review(review_id):
        review, err = _get_review(review_id)
        if err is not None:
            return err
        del app.config["REVIEWS"][review_id]
        return jsonify({"deleted": review_id})

    @app.post("/reviews/<int:review_id>/helpful")
    def helpful_vote(review_id):
        review, err = _get_review(review_id)
        if err is not None:
            return err
        review["helpful_votes"] += 1
        return jsonify(review)

    @app.post("/reviews/<int:review_id>/flag")
    def flag_review(review_id):
        review, err = _get_review(review_id)
        if err is not None:
            return err
        if review["status"] == "hidden":
            return _error(f"review {review_id} is hidden and cannot be flagged", 400)
        review["flagged"] = True
        review["status"] = "flagged"
        return jsonify(review)

    @app.post("/reviews/<int:review_id>/moderate")
    def moderate_review(review_id):
        review, err = _get_review(review_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        action = data.get("action")
        if action not in ("approve", "hide"):
            return _error("action must be 'approve' or 'hide'", 400)
        if review["status"] != "flagged":
            return _error(
                f"review {review_id} is not flagged, nothing to moderate", 400
            )
        if action == "approve":
            review["flagged"] = False
            review["status"] = "visible"
        else:
            review["status"] = "hidden"
        return jsonify(review)

    @app.get("/products/<product_id>/summary")
    def product_summary(product_id):
        reviews = [
            r
            for r in app.config["REVIEWS"].values()
            if r["product_id"] == product_id and r["status"] != "hidden"
        ]
        if not reviews:
            return _error(f"no reviews found for product {product_id!r}", 404)
        distribution = {str(star): 0 for star in range(1, 6)}
        for review in reviews:
            distribution[str(review["rating"])] += 1
        average = round(sum(r["rating"] for r in reviews) / len(reviews), 2)
        return jsonify(
            {
                "product_id": product_id,
                "count": len(reviews),
                "average_rating": average,
                "distribution": distribution,
            }
        )

    return app
