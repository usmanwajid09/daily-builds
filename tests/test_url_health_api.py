"""Tests for the url_health_api Flask app, using Flask's test client.

All checks run against a local HTTP server or unreachable local
addresses — nothing here depends on the public internet.
"""
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from url_health_api.app import create_app


class _OKHandler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args):
        pass


class _ErrorHandler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        self.send_response(404)
        self.end_headers()

    def do_GET(self):
        self.send_response(500)
        self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture
def local_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _OKHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/"
    server.shutdown()
    server.server_close()


@pytest.fixture
def error_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ErrorHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/"
    server.shutdown()
    server.server_close()


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_list_checks_starts_empty(client):
    assert client.get("/checks").get_json() == []


def test_check_local_server_returns_200(client, local_server):
    resp = client.post("/check", json={"url": local_server})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["id"] == 1
    assert body["url"] == local_server
    assert body["status"] == 200
    assert body["ok"] is True
    assert body["response_time_ms"] >= 0


def test_check_records_history(client, local_server):
    client.post("/check", json={"url": local_server})
    client.post("/check", json={"url": local_server})
    history = client.get("/checks").get_json()
    assert len(history) == 2
    assert history[0]["id"] == 1 and history[1]["id"] == 2


def test_get_check_by_id(client, local_server):
    created = client.post("/check", json={"url": local_server}).get_json()
    fetched = client.get(f"/checks/{created['id']}").get_json()
    assert fetched == created


def test_get_unknown_check_404(client):
    resp = client.get("/checks/99")
    assert resp.status_code == 404
    assert "error" in resp.get_json()


def test_http_error_status_still_recorded(client, error_server):
    resp = client.post("/check", json={"url": error_server})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] in (404, 500)
    assert body["ok"] is False


def test_unreachable_host_returns_502(client):
    resp = client.post("/check", json={"url": "http://127.0.0.1:1/"})
    assert resp.status_code == 502
    assert "error" in resp.get_json()


def test_invalid_host_returns_502(client):
    resp = client.post("/check", json={"url": "http://nonexistent.invalid/"})
    assert resp.status_code == 502


def test_rejects_missing_url(client):
    assert client.post("/check", json={}).status_code == 400


def test_rejects_empty_url(client):
    assert client.post("/check", json={"url": "   "}).status_code == 400


def test_rejects_non_http_scheme(client):
    assert client.post("/check", json={"url": "ftp://example.com/"}).status_code == 400


def test_rejects_url_without_host(client):
    assert client.post("/check", json={"url": "http:///path"}).status_code == 400


def test_rejects_non_json_body(client):
    assert client.post("/check", data="not json").status_code == 400


def test_failed_check_not_added_to_history(client):
    client.post("/check", json={"url": "http://127.0.0.1:1/"})
    assert client.get("/checks").get_json() == []
