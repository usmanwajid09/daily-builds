"""Flask application factory for the LMS platform's REST API.

Milestone 1 scope: course/lesson data model, full CRUD on courses and
lessons (instructor-owned), a file-upload endpoint for lesson content,
student enrollment, and per-course/per-student progress tracking.
Consistent with the rest of this repo's arcs: SQLite via stdlib
sqlite3, bcrypt + HS256 JWT auth, Flask for routing (used the same way
dev_collab_platform and saas_starter do).
"""
import os

from flask import Flask, jsonify

from . import db as db_module

DEFAULT_UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "data", "uploads")


def create_app(db_path: str = ":memory:", jwt_secret: str = "dev-secret-change-me",
               upload_dir: str | None = None) -> Flask:
    app = Flask(__name__)
    conn = db_module.connect(db_path)
    db_module.init_db(conn)

    app.config["DB_CONN"] = conn
    app.config["DB_PATH"] = db_path
    app.config["JWT_SECRET"] = jwt_secret
    app.config["UPLOAD_DIR"] = upload_dir or DEFAULT_UPLOAD_DIR
    app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024  # 12 MB request-level cap

    from .routes import auth_routes, course_routes, lesson_routes, enrollment_routes
    app.register_blueprint(auth_routes.bp)
    app.register_blueprint(course_routes.bp)
    app.register_blueprint(lesson_routes.bp)
    app.register_blueprint(enrollment_routes.bp)

    @app.get("/api/health")
    def health():
        return jsonify(status="ok")

    @app.errorhandler(413)
    def too_large(_exc):
        return jsonify(error="request body too large"), 413

    return app
