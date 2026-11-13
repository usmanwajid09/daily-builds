"""REST todo-list API built with Flask.

In-memory store, fresh for each app instance (nothing is written to
disk). All responses are JSON. Endpoints:

- ``GET /todos`` — list all todos
- ``POST /todos`` — create a todo (``{"title": str, "done": bool?}``)
- ``GET /todos/<id>`` — fetch one todo
- ``PUT /todos/<id>`` — update title and/or done
- ``DELETE /todos/<id>`` — delete a todo
"""
from flask import Flask, jsonify, request


def create_app():
    """Return a Flask app with a fresh, empty in-memory todo store."""
    app = Flask(__name__)
    app.config["TODOS"] = {}
    app.config["NEXT_ID"] = 1

    def _error(message, status):
        return jsonify({"error": message}), status

    def _get_todo(todo_id):
        todo = app.config["TODOS"].get(todo_id)
        if todo is None:
            return None, _error(f"todo {todo_id} not found", 404)
        return todo, None

    @app.get("/todos")
    def list_todos():
        todos = sorted(app.config["TODOS"].values(), key=lambda t: t["id"])
        return jsonify(todos)

    @app.post("/todos")
    def create_todo():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        title = data.get("title")
        if not isinstance(title, str) or not title.strip():
            return _error("title is required and must be a non-empty string", 400)
        done = data.get("done", False)
        if not isinstance(done, bool):
            return _error("done must be a boolean", 400)
        todo_id = app.config["NEXT_ID"]
        app.config["NEXT_ID"] += 1
        todo = {"id": todo_id, "title": title.strip(), "done": done}
        app.config["TODOS"][todo_id] = todo
        return jsonify(todo), 201

    @app.get("/todos/<int:todo_id>")
    def get_todo(todo_id):
        todo, err = _get_todo(todo_id)
        if err is not None:
            return err
        return jsonify(todo)

    @app.put("/todos/<int:todo_id>")
    def update_todo(todo_id):
        todo, err = _get_todo(todo_id)
        if err is not None:
            return err
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return _error("request body must be a JSON object", 400)
        if "title" not in data and "done" not in data:
            return _error("nothing to update: provide title and/or done", 400)
        if "title" in data:
            if not isinstance(data["title"], str) or not data["title"].strip():
                return _error("title must be a non-empty string", 400)
            todo["title"] = data["title"].strip()
        if "done" in data:
            if not isinstance(data["done"], bool):
                return _error("done must be a boolean", 400)
            todo["done"] = data["done"]
        return jsonify(todo)

    @app.delete("/todos/<int:todo_id>")
    def delete_todo(todo_id):
        _todo, err = _get_todo(todo_id)
        if err is not None:
            return err
        del app.config["TODOS"][todo_id]
        return jsonify({"deleted": todo_id})

    @app.errorhandler(404)
    def handle_404(exc):
        return _error("route not found", 404)

    return app
