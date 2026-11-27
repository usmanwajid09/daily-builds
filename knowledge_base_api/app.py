"""Flask knowledge base API.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Endpoints:

- ``POST /articles`` — create an article (``{\"title\": str,
  \"body\": str, \"category\": str, \"tags\": [str, ...]?}``;
  article titles are unique, 409 on a duplicate)
- ``GET /articles`` — list articles (``?category=`` and ``?tag=``
  filters)
- ``GET /articles/<id>`` — fetch one article
- ``PUT /articles/<id>`` — update title, body, category and/or
  tags (409 on a duplicate title)
- ``DELETE /articles/<id>`` — delete an article
- ``GET /search?q=...`` — full-text search over title and body
  (case-insensitive); results are ranked with title matches first,
  then by the number of matching words; each result carries a
  ``score`` and a ``snippet`` of the body around the first match
- ``GET /articles/<id>/related`` — articles sharing at least one
  tag with the given article, ranked by the number of shared tags
  (each result carries ``shared_tags``)
- ``GET /categories`` — categories with their article counts

An article is ``{\"id\": int, \"title\": str, \"body\": str,
\"category\": str, \"tags\": [str, ...]}``. Tags are stored
lower-cased and de-duplicated, in first-seen order.
"""
import re

from flask import Flask, jsonify, request

_WORD_RE = re.compile(r"[a-z0-9]+")


def _words(text):
    return _WORD_RE.findall(text.lower())


def _normalise_tags(tags):
    seen = []
    for tag in tags:
        tag = tag.strip().lower()
        if tag and tag not in seen:
            seen.append(tag)
    return seen


def create_app():
    """Return a Flask app with a fresh, empty in-memory article store."""
    app = Flask(__name__)
    app.config["ARTICLES"] = {}
    app.config["NEXT_ARTICLE_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _get_article(article_id):
        article = app.config["ARTICLES"].get(article_id)
        if article is None:
            return None, _error(f"article {article_id} not found", 404)
        return article, None

    def _public_article(article):
        return {
            "id": article["id"],
            "title": article["title"],
            "body": article["body"],
            "category": article["category"],
            "tags": list(article["tags"]),
        }

    def _validate_article_body(data, partial=False):
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        for field in ("title", "body", "category"):
            if not partial or field in data:
                value = data.get(field)
                if not isinstance(value, str) or not value.strip():
                    return _error(
                        f"{field} is required and must be a "
                        "non-empty string",
                        400,
                    )
        if "tags" in data:
            tags = data["tags"]
            if not isinstance(tags, list) or any(
                not isinstance(t, str) or not t.strip() for t in tags
            ):
                return _error(
                    "tags must be a list of non-empty strings", 400
                )
        return None

    def _snippet(body, query_words):
        lowered = body.lower()
        first = len(body)
        for word in query_words:
            index = lowered.find(word)
            if index != -1 and index < first:
                first = index
        if first == len(body):
            start, end = 0, 120
        else:
            start = max(0, first - 40)
            end = min(len(body), first + 80)
        text = body[start:end]
        if start > 0:
            text = "..." + text
        if end < len(body):
            text = text + "..."
        return text

    @app.post("/articles")
    def create_article():
        data = request.get_json(silent=True)
        err = _validate_article_body(data)
        if err is not None:
            return err
        title = data["title"].strip()
        if any(a["title"].lower() == title.lower()
               for a in app.config["ARTICLES"].values()):
            return _error(f"an article titled {title!r} already exists", 409)
        article_id = app.config["NEXT_ARTICLE_ID"]
        app.config["NEXT_ARTICLE_ID"] += 1
        article = {
            "id": article_id,
            "title": title,
            "body": data["body"].strip(),
            "category": data["category"].strip(),
            "tags": _normalise_tags(data.get("tags", [])),
        }
        app.config["ARTICLES"][article_id] = article
        return jsonify(_public_article(article)), 201

    @app.get("/articles")
    def list_articles():
        category = request.args.get("category")
        tag = request.args.get("tag")
        articles = sorted(
            app.config["ARTICLES"].values(), key=lambda a: a["id"]
        )
        if category is not None:
            articles = [
                a for a in articles
                if a["category"].lower() == category.lower()
            ]
        if tag is not None:
            articles = [a for a in articles if tag.lower() in a["tags"]]
        return jsonify([_public_article(a) for a in articles])

    @app.get("/articles/<int:article_id>")
    def get_article(article_id):
        article, err = _get_article(article_id)
        if err is not None:
            return err
        return jsonify(_public_article(article))

    @app.put("/articles/<int:article_id>")
    def update_article(article_id):
        article, err = _get_article(article_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        err = _validate_article_body(data, partial=True)
        if err is not None:
            return err
        if not any(k in data for k in ("title", "body", "category", "tags")):
            return _error(
                "nothing to update: provide title, body, category "
                "and/or tags",
                400,
            )
        if "title" in data:
            title = data["title"].strip()
            if any(a["id"] != article_id
                   and a["title"].lower() == title.lower()
                   for a in app.config["ARTICLES"].values()):
                return _error(
                    f"an article titled {title!r} already exists", 409
                )
            article["title"] = title
        if "body" in data:
            article["body"] = data["body"].strip()
        if "category" in data:
            article["category"] = data["category"].strip()
        if "tags" in data:
            article["tags"] = _normalise_tags(data["tags"])
        return jsonify(_public_article(article))

    @app.delete("/articles/<int:article_id>")
    def delete_article(article_id):
        _article, err = _get_article(article_id)
        if err is not None:
            return err
        del app.config["ARTICLES"][article_id]
        return jsonify({"deleted": article_id})

    @app.get("/search")
    def search_articles():
        query = request.args.get("q", "")
        if not isinstance(query, str) or not query.strip():
            return _error("query parameter q is required", 400)
        query_words = _words(query)
        if not query_words:
            return _error("query parameter q must contain a word", 400)
        results = []
        for article in app.config["ARTICLES"].values():
            title_words = _words(article["title"])
            body_words = _words(article["body"])
            title_hits = sum(1 for w in query_words if w in title_words)
            body_hits = sum(1 for w in query_words if w in body_words)
            if title_hits == 0 and body_hits == 0:
                continue
            results.append(
                {
                    "article": article,
                    "score": title_hits * 2 + body_hits,
                    "title_hits": title_hits,
                }
            )
        results.sort(
            key=lambda r: (-r["title_hits"], -r["score"], r["article"]["id"])
        )
        return jsonify(
            [
                {
                    **_public_article(r["article"]),
                    "score": r["score"],
                    "snippet": _snippet(r["article"]["body"], query_words),
                }
                for r in results
            ]
        )

    @app.get("/articles/<int:article_id>/related")
    def related_articles(article_id):
        article, err = _get_article(article_id)
        if err is not None:
            return err
        tags = set(article["tags"])
        related = []
        for other in app.config["ARTICLES"].values():
            if other["id"] == article_id:
                continue
            shared = sorted(tags & set(other["tags"]))
            if shared:
                related.append((len(shared), other, shared))
        related.sort(key=lambda item: (-item[0], item[1]["id"]))
        return jsonify(
            [
                {**_public_article(other), "shared_tags": shared}
                for _count, other, shared in related
            ]
        )

    @app.get("/categories")
    def list_categories():
        counts = {}
        for article in app.config["ARTICLES"].values():
            counts[article["category"]] = counts.get(article["category"], 0) + 1
        return jsonify(
            [
                {"category": name, "article_count": count}
                for name, count in sorted(counts.items())
            ]
        )

    @app.errorhandler(404)
    def handle_404(exc):
        return _error("route not found", 404)

    return app
