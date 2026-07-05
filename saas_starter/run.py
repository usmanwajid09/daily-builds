"""Dev server entry point.

    SAAS_JWT_SECRET=some-long-random-value SAAS_DB_PATH=saas_starter.db python run.py

Never run with the default dev JWT secret in production.
"""
from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True, port=5000)
