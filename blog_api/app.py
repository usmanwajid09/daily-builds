"""Flask blog API.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Endpoints:

- ``POST /posts`` — create a post (``{"title": str, "content": str,
  "tags": [str]?, "status": "draft" | "published"?}``). The slug is
  derived from the title and made unique (``my-post``, ``my-post-2``,
  ...). The slug never changes after creation.
- ``GET /posts`` — list posts; ``?status=draft|published``,
  ``?tag=<tag>`` and ``?search=<text>`` (title or content,
  case-insensitive) filters
- ``GET /posts/<slug>`` — fetch one post
- ``PUT /posts/<slug>`` — update title, content, tags and/or status
- ``POST /posts/<slug>/publish`` — set status to ``published``
- ``POST /posts/<slug>/unpublish`` — set status back to ``draft``
- ``DELETE /posts/<slug>`` — delete a post

A post is ``{"slug": str, "title": str, "content": str, "tags": [str],
"status": "draft" | "published"}``.
"""
import re

from flask import Flask, jsonify, request

STATUSES = ("draft", "published")


def slugify(title):
    """Turn a title into a URL slug: lowercase, words joined by hyphens."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return re.sub(r"-+", "-", slug)


def create_app():
    """Return a Flask app with a fresh, empty in-memory blog store."""
    app = Flask(__name__)
    app.config["POSTS"] = {}

    def _error(message, status):
        return jsonify({"error": message}), status

    def _get_post(slug):
        post = app.config["POSTS"].get(slug)
        if post is None:
            return None, _error(f"post '{slug}' not found", 404)
        return post, None

    def _validate_tags(tags):
        if not isinstance(tags, list):
            return "tags must be a list of strings"
        for tag in tags:
            if not isinstance(tag, str) or not tag.strip():
                return "each tag must be a non-empty string"
        return None

    def _clean_tags(tags):
        seen = set()
        cleaned = []
        for tag in tags:
            tag = tag.strip().lower()
            if tag not in seen:
                seen.add(tag)
                cleaned.append(tag)
        return cleaned

    def _unique_slug(base):
        slug = base
        n = 2
        while slug in app.config["POSTS"]:
            slug = f"{base}-{n}"
            n += 1
        return slug

    @app.post("/posts")
    def create_post():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        title = data.get("title")
        if not isinstance(title, str) or not title.strip():
            return _error("title is required and must be a non-empty string", 400)
        content = data.get("content")
        if not isinstance(content, str) or not content.strip():
            return _error("content is required and must be a non-empty string", 400)
        tags = data.get("tags", [])
        tag_error = _validate_tags(tags)
        if tag_error:
            return _error(tag_error, 400)
        status = data.get("status", "draft")
        if status not in STATUSES:
            return _error(f"status must be one of {', '.join(STATUSES)}", 400)
        slug = slugify(title)
        if not slug:
            return _error("title must contain at least one letter or digit", 400)
        post = {
            "slug": _unique_slug(slug),
            "title": title.strip(),
            "content": content.strip(),
            "tags": _clean_tags(tags),
            "status": status,
        }
        app.config["POSTS"][post["slug"]] = post
        return jsonify(post), 201

    @app.get("/posts")
    def list_posts():
        posts = sorted(app.config["POSTS"].values(), key=lambda p: p["slug"])
        status = request.args.get("status")
        if status is not None:
            if status not in STATUSES:
                return _error(f"status must be one of {', '.join(STATUSES)}", 400)
            posts = [p for p in posts if p["status"] == status]
        tag = request.args.get("tag")
        if tag:
            posts = [p for p in posts if tag.lower() in p["tags"]]
        search = request.args.get("search")
        if search:
            needle = search.lower()
            posts = [
                p
                for p in posts
                if needle in p["title"].lower() or needle in p["content"].lower()
            ]
        return jsonify(posts)

    @app.get("/posts/<slug>")
    def get_post(slug):
        post, err = _get_post(slug)
        if err is not None:
            return err
        return jsonify(post)

    @app.put("/posts/<slug>")
    def update_post(slug):
        post, err = _get_post(slug)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        if not any(k in data for k in ("title", "content", "tags", "status")):
            return _error(
                "nothing to update: provide title, content, tags and/or status", 400
            )
        if "title" in data:
            if not isinstance(data["title"], str) or not data["title"].strip():
                return _error("title must be a non-empty string", 400)
            if not slugify(data["title"]):
                return _error("title must contain at least one letter or digit", 400)
            post["title"] = data["title"].strip()
        if "content" in data:
            if not isinstance(data["content"], str) or not data["content"].strip():
                return _error("content must be a non-empty string", 400)
            post["content"] = data["content"].strip()
        if "tags" in data:
            tag_error = _validate_tags(data["tags"])
            if tag_error:
                return _error(tag_error, 400)
            post["tags"] = _clean_tags(data["tags"])
        if "status" in data:
            if data["status"] not in STATUSES:
                return _error(f"status must be one of {', '.join(STATUSES)}", 400)
            post["status"] = data["status"]
        return jsonify(post)

    @app.post("/posts/<slug>/publish")
    def publish_post(slug):
        post, err = _get_post(slug)
        if err is not None:
            return err
        post["status"] = "published"
        return jsonify(post)

    @app.post("/posts/<slug>/unpublish")
    def unpublish_post(slug):
        post, err = _get_post(slug)
        if err is not None:
            return err
        post["status"] = "draft"
        return jsonify(post)

    @app.delete("/posts/<slug>")
    def delete_post(slug):
        _post, err = _get_post(slug)
        if err is not None:
            return err
        del app.config["POSTS"][slug]
        return jsonify({"deleted": slug})

    @app.errorhandler(404)
    def handle_404(exc):
        return _error("route not found", 404)

    return app
