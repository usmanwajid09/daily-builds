"""Dev server entrypoint. Not for production use (see README)."""
import os

from lms_platform import create_app

if __name__ == "__main__":
    db_path = os.environ.get("LMS_DB_PATH", "lms_platform/data/lms.db")
    jwt_secret = os.environ.get("LMS_JWT_SECRET", "dev-secret-change-me")
    app = create_app(db_path=db_path, jwt_secret=jwt_secret)
    app.run(host="127.0.0.1", port=5050, debug=False, threaded=True)
