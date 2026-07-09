"""Flask application factory for the dev-collab-platform REST API.

The REST API and the WebSocket server (app/ws/server.py) share a single
SQLite connection and a single in-process Broadcaster instance, both
stashed on app.config so route handlers and the WS server can both
persist to the DB and push real-time updates from the same process --
see task_routes.py for where writes trigger broadcasts.
"""
from flask import Flask, jsonify

from . import db as db_module
from .ws.server import Broadcaster


def create_app(db_path: str = ":memory:", jwt_secret: str = "dev-secret-change-me",
               broadcaster: Broadcaster = None) -> Flask:
    app = Flask(__name__)
    conn = db_module.connect(db_path)
    db_module.init_db(conn)

    app.config["DB_CONN"] = conn
    app.config["DB_PATH"] = db_path
    app.config["JWT_SECRET"] = jwt_secret
    app.config["BROADCASTER"] = broadcaster if broadcaster is not None else Broadcaster()

    from .routes import auth_routes, workspace_routes, project_routes, task_routes
    app.register_blueprint(auth_routes.bp)
    app.register_blueprint(workspace_routes.bp)
    app.register_blueprint(project_routes.bp)
    app.register_blueprint(task_routes.bp)

    @app.get("/api/health")
    def health():
        return jsonify(status="ok")

    return app
