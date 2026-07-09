"""Starts the REST API (Flask/Werkzeug) and the real-time WebSocket
server in the same process, sharing one SQLite file and one Broadcaster
instance -- see app/ws/server.py's module docstring for the message
protocol and app/__init__.py for how the two are wired together.

Usage:
    python run.py                       # sqlite file ./dev_collab.db, ports 5000 (HTTP) / 8765 (WS)
    DB_PATH=/tmp/x.db HTTP_PORT=5001 WS_PORT=8766 python run.py
"""
import os
import threading

from app import create_app
from app.ws.server import WebSocketServer
from app import db as db_module

DB_PATH = os.environ.get("DB_PATH", "dev_collab.db")
JWT_SECRET = os.environ.get("JWT_SECRET", "dev-secret-change-me")
HTTP_HOST = os.environ.get("HTTP_HOST", "127.0.0.1")
HTTP_PORT = int(os.environ.get("HTTP_PORT", "5000"))
WS_HOST = os.environ.get("WS_HOST", "127.0.0.1")
WS_PORT = int(os.environ.get("WS_PORT", "8765"))


def main():
    flask_app = create_app(db_path=DB_PATH, jwt_secret=JWT_SECRET)
    broadcaster = flask_app.config["BROADCASTER"]

    ws_server = WebSocketServer(
        host=WS_HOST,
        port=WS_PORT,
        conn_factory=lambda: db_module.connect(DB_PATH),
        jwt_secret=JWT_SECRET,
        broadcaster=broadcaster,
    )
    ws_thread = threading.Thread(target=ws_server.serve_forever, daemon=True)
    ws_thread.start()
    print(f"WebSocket server listening on ws://{WS_HOST}:{WS_PORT}")
    print(f"REST API listening on http://{HTTP_HOST}:{HTTP_PORT}")

    try:
        flask_app.run(host=HTTP_HOST, port=HTTP_PORT, threaded=False)
    finally:
        ws_server.stop()


if __name__ == "__main__":
    main()
