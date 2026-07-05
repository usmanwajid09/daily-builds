"""Flask application factory for the multi-tenant SaaS starter."""
import os

from flask import Flask, jsonify

from . import billing as billing_module
from . import db as db_module


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.update(
        JWT_SECRET=os.environ.get("SAAS_JWT_SECRET", "dev-secret-change-me"),
        DB_PATH=os.environ.get("SAAS_DB_PATH", "saas_starter.db"),
        # Billing (Milestone 2): unset by default, which means
        # get_billing_provider() below picks the offline FakeStripeProvider.
        # See app/billing.py's module docstring for the safety rule this
        # arc follows: a non-test ("sk_live_...") key is refused outright.
        STRIPE_SECRET_KEY=os.environ.get("STRIPE_SECRET_KEY"),
        STRIPE_WEBHOOK_SECRET=os.environ.get("STRIPE_WEBHOOK_SECRET"),
        FAKE_WEBHOOK_SECRET=os.environ.get("SAAS_FAKE_WEBHOOK_SECRET", "test-only-fake-webhook-secret"),
    )
    if test_config:
        app.config.update(test_config)

    db_module.init_db(app.config["DB_PATH"])

    # Fails fast at startup (not on first request) if STRIPE_SECRET_KEY is
    # set to something other than a test key -- see billing.get_billing_provider.
    app.config["BILLING_PROVIDER"] = billing_module.get_billing_provider(app.config)

    from .routes.auth_routes import bp as auth_bp
    from .routes.billing_routes import bp as billing_bp
    from .routes.dashboard_routes import bp as dashboard_bp
    from .routes.landing_routes import bp as landing_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(billing_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(landing_bp)

    @app.get("/api/health")
    def health():
        return jsonify(status="ok")

    @app.errorhandler(404)
    def not_found(_exc):
        return jsonify(error="not found"), 404

    return app
