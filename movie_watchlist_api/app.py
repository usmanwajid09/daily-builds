"""Flask movie watchlist API.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Endpoints:

- ``POST /movies`` — add a film (``{"title": str, "genre": str,
  "year": int, "status": "to_watch" | "watching" | "watched"
  (default "to_watch"), "rating": int 1-5?}``). A rating is only
  allowed on a watched film. A second film with the same title
  (case-insensitive) is refused with 409.
- ``GET /movies`` — list films; ``?genre=``, ``?status=`` and
  ``?min_rating=`` filters
- ``GET /movies/<id>`` — fetch one film
- ``PUT /movies/<id>`` — update title, genre, year, status and/or
  rating (dropping a film back to "to_watch" or "watching" clears
  its rating)
- ``DELETE /movies/<id>`` — delete a film
- ``POST /movies/<id>/watch`` — mark a film watched, with an
  optional ``{"rating": int 1-5}``
- ``GET /stats`` — totals, counts by status, average rating of
  watched films, and counts by genre

A film is ``{"id": int, "title": str, "genre": str, "year": int,
"status": str, "rating": int | None}``.
"""
from flask import Flask, jsonify, request

STATUSES = ("to_watch", "watching", "watched")


def create_app():
    """Return a Flask app with a fresh, empty in-memory film store."""
    app = Flask(__name__)
    app.config["MOVIES"] = {}
    app.config["NEXT_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _public(movie):
        return {
            "id": movie["id"],
            "title": movie["title"],
            "genre": movie["genre"],
            "year": movie["year"],
            "status": movie["status"],
            "rating": movie["rating"],
        }

    def _get_movie(movie_id):
        movie = app.config["MOVIES"].get(movie_id)
        if movie is None:
            return None, _error(f"movie {movie_id} not found", 404)
        return movie, None

    def _check_rating(rating, status):
        if rating is None:
            return None, None
        if not isinstance(rating, int) or isinstance(rating, bool) \
                or rating < 1 or rating > 5:
            return None, _error("rating must be an integer 1-5", 400)
        if status != "watched":
            return None, _error(
                "rating is only allowed on a watched film", 400
            )
        return rating, None

    def _validate(data, partial=False):
        cleaned = {}
        if not partial or "title" in data:
            title = data.get("title")
            if not isinstance(title, str) or not title.strip():
                return None, _error(
                    "title is required and must be a non-empty string", 400
                )
            cleaned["title"] = title.strip()
        if not partial or "genre" in data:
            genre = data.get("genre")
            if not isinstance(genre, str) or not genre.strip():
                return None, _error(
                    "genre is required and must be a non-empty string", 400
                )
            cleaned["genre"] = genre.strip()
        if not partial or "year" in data:
            year = data.get("year")
            if not isinstance(year, int) or isinstance(year, bool) \
                    or year < 1888 or year > 2030:
                return None, _error(
                    "year is required and must be an integer 1888-2030", 400
                )
            cleaned["year"] = year
        if not partial or "status" in data:
            status = data.get("status", "to_watch") if not partial \
                else data.get("status")
            if status is None and partial:
                pass
            elif status not in STATUSES:
                return None, _error(
                    f"status must be one of {', '.join(STATUSES)}", 400
                )
            else:
                cleaned["status"] = status
        if "rating" in data:
            cleaned["rating"] = data["rating"]
        return cleaned, None

    def _duplicate_title(title, exclude_id=None):
        return any(
            m["title"].lower() == title.lower() and m["id"] != exclude_id
            for m in app.config["MOVIES"].values()
        )

    @app.post("/movies")
    def create_movie():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        cleaned, err = _validate(data)
        if err is not None:
            return err
        status = cleaned.get("status", "to_watch")
        rating, err = _check_rating(cleaned.get("rating"), status)
        if err is not None:
            return err
        if _duplicate_title(cleaned["title"]):
            return _error(
                f"a film titled {cleaned['title']!r} is already on the list",
                409,
            )
        movie_id = app.config["NEXT_ID"]
        app.config["NEXT_ID"] += 1
        movie = {
            "id": movie_id,
            "title": cleaned["title"],
            "genre": cleaned["genre"],
            "year": cleaned["year"],
            "status": status,
            "rating": rating,
        }
        app.config["MOVIES"][movie_id] = movie
        return jsonify(_public(movie)), 201

    @app.get("/movies")
    def list_movies():
        movies = sorted(
            app.config["MOVIES"].values(), key=lambda m: m["id"]
        )
        genre = request.args.get("genre")
        if genre is not None:
            movies = [m for m in movies
                      if m["genre"].lower() == genre.lower()]
        status = request.args.get("status")
        if status is not None:
            if status not in STATUSES:
                return _error(
                    f"status must be one of {', '.join(STATUSES)}", 400
                )
            movies = [m for m in movies if m["status"] == status]
        min_rating = request.args.get("min_rating")
        if min_rating is not None:
            try:
                min_rating = int(min_rating)
            except ValueError:
                return _error("min_rating must be an integer 1-5", 400)
            if min_rating < 1 or min_rating > 5:
                return _error("min_rating must be an integer 1-5", 400)
            movies = [m for m in movies
                      if m["rating"] is not None and m["rating"] >= min_rating]
        return jsonify([_public(m) for m in movies])

    @app.get("/movies/<int:movie_id>")
    def get_movie(movie_id):
        movie, err = _get_movie(movie_id)
        if err is not None:
            return err
        return jsonify(_public(movie))

    @app.put("/movies/<int:movie_id>")
    def update_movie(movie_id):
        movie, err = _get_movie(movie_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        fields = ("title", "genre", "year", "status", "rating")
        if not any(k in data for k in fields):
            return _error(
                "nothing to update: provide title, genre, year, status "
                "and/or rating",
                400,
            )
        cleaned, err = _validate(data, partial=True)
        if err is not None:
            return err
        new_status = cleaned.get("status", movie["status"])
        if "rating" in cleaned:
            rating, err = _check_rating(cleaned["rating"], new_status)
            if err is not None:
                return err
            cleaned["rating"] = rating
        elif "status" in cleaned and new_status != "watched":
            cleaned["rating"] = None
        if "title" in cleaned and _duplicate_title(cleaned["title"],
                                                    exclude_id=movie_id):
            return _error(
                f"a film titled {cleaned['title']!r} is already on the list",
                409,
            )
        movie.update(cleaned)
        return jsonify(_public(movie))

    @app.delete("/movies/<int:movie_id>")
    def delete_movie(movie_id):
        _movie, err = _get_movie(movie_id)
        if err is not None:
            return err
        del app.config["MOVIES"][movie_id]
        return jsonify({"deleted": movie_id})

    @app.post("/movies/<int:movie_id>/watch")
    def watch_movie(movie_id):
        movie, err = _get_movie(movie_id)
        if err is not None:
            return err
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        if movie["status"] == "watched" and "rating" not in data:
            return _error("film is already marked watched", 409)
        rating = data.get("rating")
        rating, err = _check_rating(rating, "watched")
        if err is not None:
            return err
        movie["status"] = "watched"
        if rating is not None:
            movie["rating"] = rating
        return jsonify(_public(movie))

    @app.get("/stats")
    def stats():
        movies = list(app.config["MOVIES"].values())
        by_status = {s: 0 for s in STATUSES}
        by_genre = {}
        ratings = []
        for movie in movies:
            by_status[movie["status"]] += 1
            by_genre[movie["genre"]] = by_genre.get(movie["genre"], 0) + 1
            if movie["rating"] is not None:
                ratings.append(movie["rating"])
        return jsonify(
            {
                "total": len(movies),
                "by_status": by_status,
                "watched": by_status["watched"],
                "average_rating": round(sum(ratings) / len(ratings), 2)
                if ratings else None,
                "rated_films": len(ratings),
                "by_genre": by_genre,
            }
        )

    @app.errorhandler(404)
    def handle_404(exc):
        return _error("route not found", 404)

    return app
