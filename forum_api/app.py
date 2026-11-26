"""Flask mini forum API.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Endpoints:

- ``POST /threads`` — create a thread (``{"title": str, "author": str}``)
- ``GET /threads`` — list threads, optionally filtered by ``?q=keyword``
  (case-insensitive match on the title)
- ``GET /threads/<id>`` — fetch one thread with its post count
- ``POST /threads/<id>/close`` — close a thread (closed threads reject
  new posts with ``409``)
- ``POST /threads/<id>/posts`` — add a post (``{"author": str,
  "content": str}``)
- ``GET /threads/<id>/posts`` — list a thread's posts, most upvoted first
- ``GET /posts/<id>`` — fetch one post
- ``POST /posts/<id>/upvote`` — upvote a post (increments its score)

Unknown threads and posts return ``404``; closing an already-closed
thread and posting to a closed thread return ``409``.
"""
from flask import Flask, jsonify, request


def create_app():
    """Return a Flask app with a fresh, empty in-memory forum store."""
    app = Flask(__name__)
    app.config["THREADS"] = {}
    app.config["POSTS"] = {}
    app.config["NEXT_THREAD_ID"] = 1
    app.config["NEXT_POST_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _get_thread(thread_id):
        thread = app.config["THREADS"].get(thread_id)
        if thread is None:
            return None, _error(f"thread {thread_id} not found", 404)
        return thread, None

    def _thread_posts(thread_id):
        posts = [p for p in app.config["POSTS"].values() if p["thread_id"] == thread_id]
        posts.sort(key=lambda p: (-p["upvotes"], p["id"]))
        return posts

    def _public_thread(thread):
        return {
            "id": thread["id"],
            "title": thread["title"],
            "author": thread["author"],
            "status": thread["status"],
            "post_count": len(_thread_posts(thread["id"])),
        }

    def _public_post(post):
        return {
            "id": post["id"],
            "thread_id": post["thread_id"],
            "author": post["author"],
            "content": post["content"],
            "upvotes": post["upvotes"],
        }

    @app.post("/threads")
    def create_thread():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        title = data.get("title")
        if not isinstance(title, str) or not title.strip():
            return _error("title is required and must be a non-empty string", 400)
        author = data.get("author")
        if not isinstance(author, str) or not author.strip():
            return _error("author is required and must be a non-empty string", 400)
        thread_id = app.config["NEXT_THREAD_ID"]
        app.config["NEXT_THREAD_ID"] += 1
        thread = {
            "id": thread_id,
            "title": title.strip(),
            "author": author.strip(),
            "status": "open",
        }
        app.config["THREADS"][thread_id] = thread
        return jsonify(_public_thread(thread)), 201

    @app.get("/threads")
    def list_threads():
        query = request.args.get("q", "").strip().lower()
        threads = sorted(app.config["THREADS"].values(), key=lambda t: t["id"])
        if query:
            threads = [t for t in threads if query in t["title"].lower()]
        return jsonify([_public_thread(t) for t in threads])

    @app.get("/threads/<int:thread_id>")
    def get_thread(thread_id):
        thread, err = _get_thread(thread_id)
        if err is not None:
            return err
        return jsonify(_public_thread(thread))

    @app.post("/threads/<int:thread_id>/close")
    def close_thread(thread_id):
        thread, err = _get_thread(thread_id)
        if err is not None:
            return err
        if thread["status"] == "closed":
            return _error(f"thread {thread_id} is already closed", 409)
        thread["status"] = "closed"
        return jsonify(_public_thread(thread))

    @app.post("/threads/<int:thread_id>/posts")
    def create_post(thread_id):
        thread, err = _get_thread(thread_id)
        if err is not None:
            return err
        if thread["status"] == "closed":
            return _error(f"thread {thread_id} is closed", 409)
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        author = data.get("author")
        if not isinstance(author, str) or not author.strip():
            return _error("author is required and must be a non-empty string", 400)
        content = data.get("content")
        if not isinstance(content, str) or not content.strip():
            return _error("content is required and must be a non-empty string", 400)
        post_id = app.config["NEXT_POST_ID"]
        app.config["NEXT_POST_ID"] += 1
        post = {
            "id": post_id,
            "thread_id": thread_id,
            "author": author.strip(),
            "content": content.strip(),
            "upvotes": 0,
        }
        app.config["POSTS"][post_id] = post
        return jsonify(_public_post(post)), 201

    @app.get("/threads/<int:thread_id>/posts")
    def list_posts(thread_id):
        thread, err = _get_thread(thread_id)
        if err is not None:
            return err
        return jsonify([_public_post(p) for p in _thread_posts(thread_id)])

    @app.get("/posts/<int:post_id>")
    def get_post(post_id):
        post = app.config["POSTS"].get(post_id)
        if post is None:
            return _error(f"post {post_id} not found", 404)
        return jsonify(_public_post(post))

    @app.post("/posts/<int:post_id>/upvote")
    def upvote_post(post_id):
        post = app.config["POSTS"].get(post_id)
        if post is None:
            return _error(f"post {post_id} not found", 404)
        post["upvotes"] += 1
        return jsonify(_public_post(post))

    @app.errorhandler(404)
    def handle_404(exc):
        return _error("route not found", 404)

    return app
