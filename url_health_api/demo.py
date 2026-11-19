"""Demo: check a local server (200) and an unreachable host (502)."""
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from url_health_api.app import create_app


class _OKHandler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _OKHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]

    client = create_app().test_client()
    good = client.post("/check", json={"url": f"http://127.0.0.1:{port}/"}).get_json()
    print(f"Local server check: {good}")
    bad = client.post("/check", json={"url": "http://127.0.0.1:1/"})
    print(f"Unreachable host check: {bad.status_code} {bad.get_json()}")
    history = client.get("/checks").get_json()
    print(f"{len(history)} successful check(s) in history")
    server.shutdown()


if __name__ == "__main__":
    main()
