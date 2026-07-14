"""Dev server entrypoint. Not for production use (see README)."""
import os

from fintech_app import create_app

if __name__ == "__main__":
    db_path = os.environ.get("FINTECH_DB_PATH", "fintech_app/data/fintech.db")
    jwt_secret = os.environ.get("FINTECH_JWT_SECRET", "dev-secret-change-me")
    app = create_app(db_path=db_path, jwt_secret=jwt_secret)
    app.run(host="127.0.0.1", port=5060, debug=False, threaded=True)
