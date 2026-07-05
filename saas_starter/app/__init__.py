"""Flask application factory for the multi-tenant SaaS starter."""
import os

from flask import Flask, jsonify

from . import db as db_module


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.update(
        JWT_SECRET=os.environ.get("SAAS_JWT_SECRET", "dev-secret-change-me"),
        DB_PATH=os.environ.get("SAAS_DB_PATH", "saas_starter.db"),
    )
    if test_config:
        app.config.update(test_config)

    db_module.init_db(app.config["DB_PATH"])

    from .routes.auth_routes import bp as auth_bp
    from .routes.dashboard_routes import bp as dashboard_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)

    @app.get("/api/health")
    def health():
        return jsonify(status="ok")

    @app.errorhandler(404)
    def not_found(_exc):
        return jsonify(error="not found"), 404

    return app
